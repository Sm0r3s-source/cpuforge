//! `cpuforge demo [--steps N] [--lora-steps N]`
//!
//! A self-contained tour that needs no downloads, run from the repository root:
//!   1. pretrain a tiny byte-level Llama on this project's own Rust sources,
//!   2. save it as a Hugging Face directory and load it back,
//!   3. LoRA-finetune it on a different kind of text (docs, TOML, Python),
//!   4. compare what the base and the adapted model write.

use std::fs;
use std::path::{Path, PathBuf};

use cf_io::{load_adapter, load_model, save_adapter, save_model};
use cf_model::{LlamaConfig, MemoryMode, Model, TrainSpec};
use cf_train::{evaluate, ByteTokenizer, Dataset, Event, TrainConfig, Tokenizer, Trainer};

use crate::args::Args;
use crate::cmd_generate::generate;
use crate::cmd_train::fmt_count;

fn collect(dir: &Path, ext: &str, out: &mut Vec<PathBuf>) {
    let Ok(rd) = fs::read_dir(dir) else { return };
    let mut entries: Vec<_> = rd.flatten().map(|e| e.path()).collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            if p.file_name().is_some_and(|n| n == "target" || n == ".git" || n.to_string_lossy().starts_with('.')) {
                continue;
            }
            collect(&p, ext, out);
        } else if p.extension().is_some_and(|e| e == ext) {
            out.push(p);
        }
    }
}

fn read_all(files: &[PathBuf]) -> String {
    files.iter().filter_map(|p| fs::read_to_string(p).ok()).collect::<Vec<_>>().join("\n")
}

fn tiny_config() -> LlamaConfig {
    LlamaConfig::from_json(
        r#"{"vocab_size":256,"hidden_size":128,"intermediate_size":352,"num_hidden_layers":4,
            "num_attention_heads":4,"num_key_value_heads":2,"rms_norm_eps":1e-5,
            "rope_theta":10000.0,"max_position_embeddings":256,"tie_word_embeddings":true}"#,
    )
    .expect("valid config")
}

fn log(total: usize, every: usize) -> impl FnMut(&Event) {
    move |e| match e {
        Event::Step(s) if s.step == 1 || s.step % every == 0 || s.step == total => {
            println!("  step {:>4}/{total}  train loss {:.3}  {:>6.0} tok/s", s.step, s.loss, s.tokens_per_sec);
        }
        Event::Eval { step, val_loss } => println!("  step {step:>4}/{total}  validation loss {val_loss:.3}"),
        _ => {}
    }
}

pub fn run(args: &Args) -> Result<(), String> {
    let steps: usize = args.parsed("steps", 250)?;
    let lora_steps: usize = args.parsed("lora-steps", 100)?;
    let tok = ByteTokenizer;
    let seq_len = 128;

    // --- corpora -------------------------------------------------------------
    let mut rs = Vec::new();
    collect(Path::new("crates"), "rs", &mut rs);
    if rs.is_empty() {
        return Err("run `cpuforge demo` from the repository root (no crates/**/*.rs found)".into());
    }
    let code = read_all(&rs);
    let mut other = Vec::new();
    for ext in ["md", "toml", "py"] {
        collect(Path::new("."), ext, &mut other);
    }
    let docs = read_all(&other);
    println!("pretraining text: {} Rust files, {} KB", rs.len(), code.len() / 1024);
    println!("finetuning text : {} md/toml/py files, {} KB\n", other.len(), docs.len() / 1024);

    // --- 1. pretrain from scratch ----------------------------------------------
    let mut code_set = Dataset::from_stream(&tok.encode(&code)?, seq_len);
    let code_val = code_set.split_off_val(0.05);
    let model = Model::random(tiny_config(), 256, 1, 0.05);
    println!("[1/4] pretraining a {} parameter byte-level Llama from scratch", fmt_count(model.cfg.num_params()));
    let cfg = TrainConfig { steps, batch: 4, lr: 3e-3, warmup_steps: 10, memory: MemoryMode::Full, log_every: 1, ..Default::default() };
    let mut trainer = Trainer::new(model, TrainSpec { base: true, embed: true, lora: false }, cfg);
    println!("  validation loss before: {:.3}  (a coin-flip over 256 bytes would be {:.3})", evaluate(&trainer.model, &code_val, 0), 256f32.ln());
    trainer.train(&code_set, None, None, &mut log(steps, (steps / 5).max(1)))?;
    println!("  validation loss after : {:.3}\n", evaluate(&trainer.model, &code_val, 0));

    // --- 2. save as a Hugging Face directory and load it back --------------------
    let dir = std::env::temp_dir().join(format!("cpuforge-demo-{}", std::process::id()));
    save_model(&dir, &mut trainer.model)?;
    let mut model = load_model(&dir, 256)?;
    let probe = tok.encode("fn main() {")?;
    let same = trainer.model.logits(&probe) == model.logits(&probe);
    println!("[2/4] saved to {} and reloaded; logits identical: {}\n", dir.display(), if same { "yes" } else { "NO" });
    if !same {
        return Err("reloaded model differs from the saved one".into());
    }

    // --- 3. LoRA finetune on different text -------------------------------------
    let mut doc_set = Dataset::from_stream(&tok.encode(&docs)?, seq_len);
    let doc_val = doc_set.split_off_val(0.1);
    let before = evaluate(&model, &doc_val, 0);
    let base_model_for_compare = load_model(&dir, 256)?;
    model.add_lora(16, 32.0, &["q", "k", "v", "o", "gate", "up", "down"], 5)?;
    println!("[3/4] LoRA finetune (rank 16, all projections) on docs/config text");
    println!("  validation loss on that text before: {before:.3}");
    let eval_every = (lora_steps / 4).max(1);
    let cfg = TrainConfig { steps: lora_steps, batch: 4, lr: args.parsed("lora-lr", 2e-3)?, warmup_steps: 5, log_every: 1, eval_every, ..Default::default() };
    let mut lt = Trainer::new(model, TrainSpec::LORA_ONLY, cfg);
    println!("  trainable parameters: {} of {}", fmt_count(lt.grads.numel()), fmt_count(lt.model.cfg.num_params()));
    lt.train(&doc_set, Some(&doc_val), None, &mut log(lora_steps, eval_every))?;
    let after = evaluate(&lt.model, &doc_val, 0);
    println!("  validation loss on that text after : {after:.3}\n");

    // adapter file round trip
    let adapter = dir.join("adapter.safetensors");
    save_adapter(&adapter, &mut lt.model)?;
    let size = fs::metadata(&adapter).map(|m| m.len()).unwrap_or(0);
    let mut restored = load_model(&dir, 256)?;
    load_adapter(&mut restored, &adapter)?;

    // --- 4. compare outputs -------------------------------------------------------
    println!("[4/4] same prompt, same sampling seed ({} KB adapter file):", size / 1024);
    let prompt = "[package]\nname = \"";
    for (label, m) in [("base (Rust-only)", &base_model_for_compare), ("with adapter", &restored)] {
        let out = generate(m, &tok, prompt, 80, 0.7, 20, 11, &mut |_| {})?;
        let text = tok.decode(&out)?;
        println!("  {label:<17}| {}", format!("{prompt}{text}").replace('\n', "\n                   | "));
    }
    let _ = fs::remove_dir_all(&dir);
    Ok(())
}
