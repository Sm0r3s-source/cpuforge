//! `cpuforge worker --config RUN_DIR/cpuforge.toml [--resume]`
//!
//! The training entry point Brewery drives. Brewery tokenizes its data with the model's chat
//! template and writes a `cpuforge.toml` next to the data; this runs the job, reports progress
//! through `status.json` / `metrics.jsonl` exactly like Brewery's own GPU worker, stops cleanly
//! on a `STOP` file or SIGTERM, and leaves a PEFT adapter (or a full model) in `final/`.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use cf_io::{load_model, save_model, save_peft_adapter};
use cf_model::{MemoryMode, TrainSpec};
use cf_train::{Dataset, Event, TrainConfig, Trainer};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::args::Args;
use crate::cmd_pull::{default_cache_dir, pull_model};
use crate::status::{classify_error, now_string, Status, STOP_FILE};
use crate::util::peak_rss_mb;

static SIGNALLED: AtomicBool = AtomicBool::new(false);

#[derive(Deserialize, Default)]
#[serde(default)]
pub struct WorkerConfig {
    pub run: RunSection,
    pub model: ModelSection,
    pub data: DataSection,
    pub lora: Option<LoraSection>,
    pub train: TrainSection,
}

#[derive(Deserialize, Default)]
#[serde(default)]
pub struct RunSection {
    pub job_id: String,
    /// Hugging Face id of the base model, written into the adapter's config.
    pub base_model: String,
    pub objective: String,
}

#[derive(Deserialize, Default)]
#[serde(default)]
pub struct ModelSection {
    /// Hugging Face repository to download when the model is not on disk yet.
    pub hf: String,
    /// Model directory (relative paths are relative to the run folder). Default: the cpuforge cache for `hf`.
    pub path: String,
}

#[derive(Deserialize)]
#[serde(default)]
pub struct DataSection {
    pub train: String,
    pub eval: String,
    pub seq_len: usize,
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
pub struct TrainSection {
    pub steps: usize,
    pub batch: usize,
    pub lr: f32,
    pub warmup_steps: usize,
    pub min_lr_ratio: f32,
    pub weight_decay: f32,
    pub clip: f32,
    /// "full" keeps activations; "checkpoint" recomputes them to save RAM.
    pub memory: String,
    pub seed: u64,
    pub eval_every: usize,
    pub save_every: usize,
}

impl Default for DataSection {
    fn default() -> Self {
        DataSection { train: String::new(), eval: String::new(), seq_len: 512 }
    }
}
impl Default for LoraSection {
    fn default() -> Self {
        LoraSection { rank: 16, alpha: 32.0, targets: ["q", "k", "v", "o", "gate", "up", "down"].map(String::from).to_vec(), seed: 0 }
    }
}
impl Default for TrainSection {
    fn default() -> Self {
        TrainSection {
            steps: 100,
            batch: 4,
            lr: 2e-4,
            warmup_steps: 5,
            min_lr_ratio: 0.1,
            weight_decay: 0.01,
            clip: 1.0,
            memory: "full".into(),
            seed: 42,
            eval_every: 0,
            save_every: 25,
        }
    }
}

pub fn parse_worker_config(text: &str) -> Result<WorkerConfig, String> {
    let c: WorkerConfig = toml::from_str(text).map_err(|e| format!("bad worker config: {e}"))?;
    if c.data.train.is_empty() {
        return Err("worker config needs [data] train".into());
    }
    if c.model.hf.is_empty() && c.model.path.is_empty() {
        return Err("worker config needs [model] hf or path".into());
    }
    if c.train.steps == 0 || c.train.batch == 0 || c.data.seq_len < 2 {
        return Err("train.steps and train.batch must be > 0 and data.seq_len >= 2".into());
    }
    if !["full", "checkpoint"].contains(&c.train.memory.as_str()) {
        return Err(format!("train.memory must be \"full\" or \"checkpoint\", got '{}'", c.train.memory));
    }
    if let Some(l) = &c.lora {
        if l.rank == 0 || l.targets.is_empty() {
            return Err("lora.rank must be > 0 and lora.targets must not be empty".into());
        }
    }
    Ok(c)
}

const TOKENIZER_FILES: &[&str] =
    &["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja", "chat_template.json", "tokenizer.model", "vocab.json", "merges.txt", "generation_config.json"];

fn copy_tokenizer_files(model_dir: &Path, to: &Path) {
    for f in TOKENIZER_FILES {
        let src = model_dir.join(f);
        if src.exists() {
            let _ = std::fs::copy(&src, to.join(f));
        }
    }
}

struct Outcome {
    stopped: bool,
    final_metrics: Value,
}

fn resolve(run_dir: &Path, p: &str) -> PathBuf {
    let path = Path::new(p);
    if path.is_absolute() { path.to_path_buf() } else { run_dir.join(path) }
}

fn execute(cfg: &WorkerConfig, run_dir: &Path, status: &mut Status, resume: bool) -> Result<Outcome, String> {
    status.set(json!({"state": "preparing", "job_id": cfg.run.job_id, "base_model": cfg.run.base_model,
        "method": if cfg.lora.is_some() { "lora" } else { "full" }, "objective": cfg.run.objective, "engine": "cpuforge"}));

    // --- model: on disk, or downloaded once into the cache
    let model_dir = if cfg.model.path.is_empty() { default_cache_dir(&cfg.model.hf) } else { resolve(run_dir, &cfg.model.path) };
    if !model_dir.join("config.json").exists() {
        if cfg.model.hf.is_empty() {
            return Err(format!("model folder {} has no config.json", model_dir.display()));
        }
        status.set(json!({"state": "preparing", "message": format!("downloading {}", cfg.model.hf)}));
        pull_model(&cfg.model.hf, "main", &model_dir, &mut |l| println!("{l}"))?;
    }

    // --- data
    let read = |rel: &str| std::fs::read_to_string(resolve(run_dir, rel)).map_err(|e| format!("cannot read {rel}: {e}"));
    let seq_len = cfg.data.seq_len;
    status.set(json!({"state": "loading_model", "message": format!("loading {}", model_dir.display())}));
    let mut model = load_model(&model_dir, seq_len)?;
    let total_params = model.cfg.num_params();
    let vocab = model.cfg.vocab_size as i64;

    let train = Dataset::from_tokenized_jsonl(&read(&cfg.data.train)?, seq_len)?;
    let val = if cfg.data.eval.is_empty() { Dataset::default() } else { Dataset::from_tokenized_jsonl(&read(&cfg.data.eval)?, seq_len)? };
    if train.is_empty() {
        return Err("the training file has no usable examples".into());
    }
    for (name, d) in [("training", &train), ("evaluation", &val)] {
        if let Some(m) = d.max_token_id() {
            if m >= vocab {
                return Err(format!("the {name} data contains token id {m} but the model's vocabulary has only {vocab} tokens: it was tokenized for a different model"));
            }
        }
    }

    // --- what to train
    let spec = match &cfg.lora {
        Some(l) => {
            let targets: Vec<&str> = l.targets.iter().map(|t| t.strip_suffix("_proj").unwrap_or(t)).collect();
            model.add_lora(l.rank, l.alpha, &targets, l.seed)?;
            TrainSpec::LORA_ONLY
        }
        None => TrainSpec { base: true, embed: true, lora: false },
    };
    let t = &cfg.train;
    let tc = TrainConfig {
        steps: t.steps,
        batch: t.batch,
        lr: t.lr,
        warmup_steps: t.warmup_steps,
        min_lr_ratio: t.min_lr_ratio,
        weight_decay: t.weight_decay,
        clip: t.clip,
        memory: if t.memory == "checkpoint" { MemoryMode::Checkpoint } else { MemoryMode::Full },
        seed: t.seed,
        log_every: 1,
        eval_every: if val.is_empty() { 0 } else { t.eval_every },
        save_every: t.save_every,
        ..Default::default()
    };
    let mut trainer = Trainer::new(model, spec, tc);
    let stop_path = run_dir.join(STOP_FILE);
    trainer.should_stop = Some(Box::new(move || SIGNALLED.load(Ordering::Relaxed) || stop_path.exists()));
    let state_path = run_dir.join("state.safetensors");
    if resume && state_path.exists() {
        trainer.resume(&state_path)?;
    }
    status.set(json!({"trainable_params": trainer.grads.numel(), "total_params": total_params, "num_train": train.len(), "num_eval": val.len(),
        "data": {"examples": train.len(), "tokens": train.total_tokens()}, "step": trainer.step, "max_steps": t.steps,
        "state": "training", "message": "training"}));

    // --- train
    let (started, first_step) = (Instant::now(), trainer.step);
    let (mut last_loss, mut last_eval) = (None::<f32>, None::<f32>);
    let total = t.steps;
    let val_ref = if val.is_empty() { None } else { Some(&val) };
    trainer.train(&train, val_ref, Some(&state_path), &mut |e| match e {
        Event::Micro { step, done, of } => status.set(json!({"message": format!("step {step}/{total}: sequence {done}/{of}")})),
        Event::Step(s) => {
            last_loss = Some(s.loss);
            let per_step = started.elapsed().as_secs_f64() / (s.step - first_step).max(1) as f64;
            let eta = per_step * (total - s.step) as f64;
            status.log_metrics(json!({"step": s.step, "loss": s.loss, "learning_rate": s.lr, "grad_norm": s.grad_norm, "epoch": s.epoch, "tokens_per_sec": s.tokens_per_sec}));
            status.set(json!({"state": "training", "step": s.step, "max_steps": total, "loss": s.loss, "eta_s": eta.round(), "tokens_per_sec": s.tokens_per_sec.round(),
                "epoch": s.epoch, "message": format!("step {}/{total}", s.step), "peak_rss_mb": peak_rss_mb().map(|m| m.round())}));
        }
        Event::Eval { step, val_loss } => {
            last_eval = Some(*val_loss);
            status.log_metrics(json!({"step": step, "eval_loss": val_loss}));
        }
        Event::Saved { step, .. } => status.set(json!({"last_checkpoint_step": step})),
    })?;

    let mut final_metrics = serde_json::Map::new();
    if let Some(l) = last_loss {
        final_metrics.insert("train_loss".into(), l.into());
    }
    if let Some(l) = last_eval.or_else(|| (!val.is_empty()).then(|| cf_train::evaluate(&trainer.model, &val, 0))) {
        final_metrics.insert("eval_loss".into(), l.into());
    }
    let final_metrics = Value::Object(final_metrics);

    if trainer.stopped {
        return Ok(Outcome { stopped: true, final_metrics });
    }

    // --- results
    status.set(json!({"state": "saving", "message": "saving the final model"}));
    let final_dir = run_dir.join("final");
    let _ = std::fs::remove_dir_all(&final_dir);
    if cfg.lora.is_some() {
        save_peft_adapter(&final_dir, &mut trainer.model, &cfg.run.base_model)?;
    } else {
        save_model(&final_dir, &mut trainer.model)?;
    }
    copy_tokenizer_files(&model_dir, &final_dir);
    let results = json!({"objective": cfg.run.objective, "metrics": final_metrics, "data": {"examples": train.len(), "tokens": train.total_tokens(), "eval_examples": val.len()},
        "stopped_early": false, "started_from": cfg.run.base_model, "engine": "cpuforge"});
    std::fs::write(run_dir.join("results.json"), serde_json::to_string_pretty(&results).unwrap_or_default()).map_err(|e| e.to_string())?;
    Ok(Outcome { stopped: false, final_metrics })
}

pub fn run(args: &Args) -> Result<(), String> {
    let cfg_path = PathBuf::from(args.require("config")?);
    let run_dir = cfg_path.parent().map(Path::to_path_buf).filter(|p| !p.as_os_str().is_empty()).unwrap_or_else(|| PathBuf::from("."));
    let resume = args.has("resume");
    let _ = ctrlc::set_handler(|| SIGNALLED.store(true, Ordering::Relaxed)); // SIGTERM / SIGINT / Ctrl-C: save and stop
    if resume {
        let _ = std::fs::remove_file(run_dir.join(STOP_FILE)); // the stop that paused this run is over
    }
    let mut status = Status::new(&run_dir);

    let parsed = std::fs::read_to_string(&cfg_path).map_err(|e| format!("cannot read {}: {e}", cfg_path.display())).and_then(|t| parse_worker_config(&t));
    let cfg = match parsed {
        Ok(c) => c,
        Err(e) => {
            let (kind, hint) = classify_error(&e);
            status.fail(&e, hint, kind);
            return Err(e);
        }
    };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| execute(&cfg, &run_dir, &mut status, resume)));
    let result = result.unwrap_or_else(|p| {
        let msg = p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_else(|| "internal error".into());
        Err(format!("internal error: {msg}"))
    });
    match result {
        Ok(o) => {
            let state = if o.stopped { "stopped" } else { "completed" };
            status.set(json!({"state": state, "final_metrics": o.final_metrics, "finished_at": now_string(), "message": if o.stopped { "stopped: state saved, resume with --resume" } else { "done" }}));
            println!("{state}");
            Ok(())
        }
        Err(e) => {
            let (kind, hint) = classify_error(&e);
            status.fail(&e, hint, kind);
            Err(e)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimal_config_gets_defaults_and_validation_catches_mistakes() {
        let c = parse_worker_config("[model]\nhf = \"Qwen/Qwen3-0.6B\"\n[data]\ntrain = \"data/train.tok.jsonl\"\n").unwrap();
        assert!(c.lora.is_none() && c.train.steps == 100 && c.data.seq_len == 512);
        let l = parse_worker_config("[model]\npath = \"m\"\n[data]\ntrain = \"t\"\n[lora]\nrank = 8\n").unwrap().lora.unwrap();
        assert_eq!((l.rank, l.targets.len()), (8, 7));
        assert!(parse_worker_config("[data]\ntrain = \"t\"").is_err()); // no model
        assert!(parse_worker_config("[model]\npath = \"m\"").is_err()); // no data
        assert!(parse_worker_config("[model]\npath = \"m\"\n[data]\ntrain = \"t\"\n[train]\nmemory = \"x\"").is_err());
        assert!(parse_worker_config("[model]\npath = \"m\"\n[data]\ntrain = \"t\"\n[lora]\ntargets = []").is_err());
        assert!(parse_worker_config("nonsense").is_err());
    }
}
