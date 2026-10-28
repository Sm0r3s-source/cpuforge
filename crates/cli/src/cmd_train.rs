//! `cpuforge train --config run.toml [--resume]`

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use cf_io::{load_model, save_adapter, save_model};
use cf_model::{MemoryMode, TrainSpec};
use cf_train::{Dataset, Event, TrainConfig, Trainer};

use crate::args::Args;
use crate::util::{load_tokenizer, peak_rss_mb};

#[derive(Deserialize, Default)]
#[serde(default)]
pub struct Config {
    pub model: ModelSection,
    pub lora: LoraSection,
    pub data: DataSection,
    pub train: TrainSection,
}

#[derive(Deserialize, Default)]
#[serde(default)]
pub struct ModelSection {
    /// Hugging Face directory (config.json + safetensors).
    pub path: String,
    /// tokenizer.json path or "bytes". Default: `<model path>/tokenizer.json`.
    pub tokenizer: Option<String>,
}

#[derive(Deserialize)]
#[serde(default)]
pub struct LoraSection {
    pub rank: usize,
    pub alpha: f32,
    pub targets: Vec<String>,
    pub seed: u64,
}

#[derive(Deserialize)]
#[serde(default)]
pub struct DataSection {
    /// `.jsonl` ({"text"} or {"prompt","completion"} per line) or a plain text file.
    pub path: String,
    pub seq_len: usize,
    pub val_fraction: f32,
}

#[derive(Deserialize)]
#[serde(default)]
pub struct TrainSection {
    /// "lora" (frozen base + adapters) or "full".
    pub mode: String,
    pub steps: usize,
    pub batch: usize,
    pub lr: f32,
    pub warmup_steps: usize,
    pub min_lr_ratio: f32,
    pub weight_decay: f32,
    pub clip: f32,
    /// "full" keeps all activations; "checkpoint" recomputes them to save RAM.
    pub memory: String,
    pub seed: u64,
    pub log_every: usize,
    pub eval_every: usize,
    pub save_every: usize,
    pub out_dir: String,
}

impl Default for LoraSection {
    fn default() -> Self {
        LoraSection {
            rank: 16,
            alpha: 32.0,
            targets: ["q", "k", "v", "o", "gate", "up", "down"].iter().map(|s| s.to_string()).collect(),
            seed: 0,
        }
    }
}
impl Default for DataSection {
    fn default() -> Self {
        DataSection { path: String::new(), seq_len: 512, val_fraction: 0.05 }
    }
}
impl Default for TrainSection {
    fn default() -> Self {
        TrainSection {
            mode: "lora".into(),
            steps: 100,
            batch: 4,
            lr: 2e-4,
            warmup_steps: 10,
            min_lr_ratio: 0.1,
            weight_decay: 0.01,
            clip: 1.0,
            memory: "full".into(),
            seed: 1,
            log_every: 1,
            eval_every: 0,
            save_every: 25,
            out_dir: "runs/latest".into(),
        }
    }
}

pub fn parse_config(text: &str) -> Result<Config, String> {
    let cfg: Config = toml::from_str(text).map_err(|e| format!("bad config: {e}"))?;
    if cfg.model.path.is_empty() {
        return Err("config needs [model] path".into());
    }
    if cfg.data.path.is_empty() {
        return Err("config needs [data] path".into());
    }
    if !["lora", "full"].contains(&cfg.train.mode.as_str()) {
        return Err(format!("train.mode must be \"lora\" or \"full\", got '{}'", cfg.train.mode));
    }
    if !["full", "checkpoint"].contains(&cfg.train.memory.as_str()) {
        return Err(format!("train.memory must be \"full\" or \"checkpoint\", got '{}'", cfg.train.memory));
    }
    if cfg.train.batch == 0 || cfg.train.steps == 0 || cfg.data.seq_len < 2 {
        return Err("train.batch, train.steps must be > 0 and data.seq_len >= 2".into());
    }
    Ok(cfg)
}

pub fn load_dataset(path: &Path, tok: &dyn cf_train::Tokenizer, seq_len: usize) -> Result<Dataset, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let ds = if path.extension().is_some_and(|e| e == "jsonl") {
        Dataset::from_jsonl(&text, tok, seq_len)?
    } else {
        Dataset::from_stream(&tok.encode(&text)?, seq_len)
    };
    if ds.is_empty() {
        return Err(format!("{} produced no training examples", path.display()));
    }
    Ok(ds)
}

pub fn run(args: &Args) -> Result<(), String> {
    let cfg_path = args.require("config")?;
    let text = fs::read_to_string(cfg_path).map_err(|e| format!("cannot read {cfg_path}: {e}"))?;
    let cfg = parse_config(&text)?;
    let model_dir = PathBuf::from(&cfg.model.path);

    let tok_spec = match &cfg.model.tokenizer {
        Some(t) => t.clone(),
        None => model_dir.join("tokenizer.json").to_string_lossy().into_owned(),
    };
    let tok = load_tokenizer(&tok_spec)?;

    println!("loading {} ...", model_dir.display());
    let mut model = load_model(&model_dir, cfg.data.seq_len)?;
    if tok.vocab_size() > model.cfg.vocab_size {
        return Err(format!("tokenizer has {} tokens but the model only {}", tok.vocab_size(), model.cfg.vocab_size));
    }
    let total_params = model.cfg.num_params();

    let mut data = load_dataset(Path::new(&cfg.data.path), tok.as_ref(), cfg.data.seq_len)?;
    let val = if cfg.data.val_fraction > 0.0 { data.split_off_val(cfg.data.val_fraction) } else { Dataset::default() };

    let spec = if cfg.train.mode == "lora" {
        let targets: Vec<&str> = cfg.lora.targets.iter().map(String::as_str).collect();
        model.add_lora(cfg.lora.rank, cfg.lora.alpha, &targets, cfg.lora.seed)?;
        TrainSpec::LORA_ONLY
    } else {
        TrainSpec { base: true, embed: true, lora: false }
    };

    let tc = TrainConfig {
        steps: cfg.train.steps,
        batch: cfg.train.batch,
        lr: cfg.train.lr,
        warmup_steps: cfg.train.warmup_steps,
        min_lr_ratio: cfg.train.min_lr_ratio,
        weight_decay: cfg.train.weight_decay,
        clip: cfg.train.clip,
        memory: if cfg.train.memory == "checkpoint" { MemoryMode::Checkpoint } else { MemoryMode::Full },
        seed: cfg.train.seed,
        log_every: cfg.train.log_every,
        eval_every: if val.is_empty() { 0 } else { cfg.train.eval_every },
        save_every: cfg.train.save_every,
        ..Default::default()
    };
    let mut trainer = Trainer::new(model, spec, tc);

    let out = PathBuf::from(&cfg.train.out_dir);
    fs::create_dir_all(&out).map_err(|e| format!("cannot create {}: {e}", out.display()))?;
    let state_path = out.join("state.safetensors");
    if args.has("resume") {
        if state_path.exists() {
            trainer.resume(&state_path)?;
            println!("resumed from step {} ({} examples seen)", trainer.step, trainer.consumed);
        } else {
            println!("no state at {}; starting fresh", state_path.display());
        }
    }

    println!(
        "{} params, {} trainable ({}), {} train examples / {} tokens, {} val examples, seq_len {}, batch {}",
        fmt_count(total_params),
        fmt_count(trainer.grads.numel()),
        cfg.train.mode,
        data.len(),
        fmt_count(data.total_tokens()),
        val.len(),
        cfg.data.seq_len,
        cfg.train.batch
    );
    if !val.is_empty() {
        println!("validation loss before training: {:.4}", cf_train::evaluate(&trainer.model, &val, 0));
    }

    let mut csv = fs::File::create(out.join("log.csv")).map_err(|e| e.to_string())?;
    writeln!(csv, "step,loss,lr,grad_norm,tokens_per_sec").map_err(|e| e.to_string())?;
    let total_steps = cfg.train.steps;
    let val_ref = if val.is_empty() { None } else { Some(&val) };
    trainer.train(&data, val_ref, Some(&state_path), &mut |e| match e {
        Event::Step(s) => {
            println!(
                "step {:>5}/{}  loss {:.4}  lr {:.2e}  |g| {:.3}  {:>6.0} tok/s  epoch {:.2}",
                s.step, total_steps, s.loss, s.lr, s.grad_norm, s.tokens_per_sec, s.epoch
            );
            let _ = writeln!(csv, "{},{},{},{},{}", s.step, s.loss, s.lr, s.grad_norm, s.tokens_per_sec);
        }
        Event::Micro { .. } => {}
        Event::Eval { step, val_loss } => println!("step {step:>5}  validation loss {val_loss:.4}"),
        Event::Saved { step, path } => println!("step {step:>5}  saved {}", path.display()),
    })?;

    if !val.is_empty() {
        println!("validation loss after training: {:.4}", cf_train::evaluate(&trainer.model, &val, 0));
    }
    if cfg.train.mode == "lora" {
        let p = out.join("adapter.safetensors");
        save_adapter(&p, &mut trainer.model)?;
        println!("adapter saved to {}", p.display());
    } else {
        let p = out.join("model");
        save_model(&p, &mut trainer.model)?;
        println!("model saved to {}", p.display());
    }
    if let Some(mb) = peak_rss_mb() {
        println!("peak memory {mb:.0} MB");
    }
    Ok(())
}

pub fn fmt_count(n: usize) -> String {
    match n {
        n if n >= 1_000_000_000 => format!("{:.2}B", n as f64 / 1e9),
        n if n >= 1_000_000 => format!("{:.1}M", n as f64 / 1e6),
        n if n >= 1_000 => format!("{:.1}K", n as f64 / 1e3),
        n => n.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimal_config_gets_defaults() {
        let c = parse_config("[model]\npath = \"m\"\n[data]\npath = \"d.txt\"\n").unwrap();
        assert_eq!(c.train.mode, "lora");
        assert_eq!(c.lora.rank, 16);
        assert_eq!(c.lora.targets.len(), 7);
        assert_eq!(c.data.seq_len, 512);
    }

    #[test]
    fn overrides_and_validation() {
        let c = parse_config(
            "[model]\npath = \"m\"\n[data]\npath = \"d\"\nseq_len = 64\n[train]\nmode = \"full\"\nsteps = 3\nlr = 1e-3\nmemory = \"checkpoint\"\n[lora]\nrank = 4\n",
        )
        .unwrap();
        assert_eq!((c.train.steps, c.data.seq_len, c.lora.rank), (3, 64, 4));
        assert!(parse_config("[data]\npath = \"d\"").is_err()); // no model
        assert!(parse_config("[model]\npath = \"m\"").is_err()); // no data
        assert!(parse_config("[model]\npath=\"m\"\n[data]\npath=\"d\"\n[train]\nmode=\"weird\"").is_err());
        assert!(parse_config("[model]\npath=\"m\"\n[data]\npath=\"d\"\n[train]\nmemory=\"lots\"").is_err());
        assert!(parse_config("[model]\npath=\"m\"\n[data]\npath=\"d\"\n[train]\nbatch=0").is_err());
        assert!(parse_config("this is not toml").is_err());
    }

    #[test]
    fn counts_are_readable() {
        assert_eq!(fmt_count(999), "999");
        assert_eq!(fmt_count(1_500), "1.5K");
        assert_eq!(fmt_count(134_515_008), "134.5M");
    }
}
