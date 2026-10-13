//! Drives the real binary through the workflows a user would run.

use std::path::Path;
use std::process::{Command, Output};

fn tf(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_cpuforge")).args(args).output().expect("run cpuforge")
}

fn ok(args: &[&str]) -> String {
    let o = tf(args);
    let out = String::from_utf8_lossy(&o.stdout).into_owned();
    assert!(o.status.success(), "`cpuforge {}` failed:\n{}\n{}", args.join(" "), out, String::from_utf8_lossy(&o.stderr));
    out
}

fn write(path: &Path, text: &str) {
    std::fs::write(path, text).unwrap();
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

fn config(model: &Path, data: &Path, out: &Path, mode: &str, steps: usize, extra: &str) -> String {
    format!(
        "[model]\npath = \"{}\"\ntokenizer = \"bytes\"\n[data]\npath = \"{}\"\nseq_len = 48\nval_fraction = 0.1\n\
         [lora]\nrank = 4\nalpha = 8\n[train]\nmode = \"{mode}\"\nsteps = {steps}\nbatch = 2\nlr = 3e-3\nwarmup_steps = 2\n\
         save_every = 2\neval_every = 2\nout_dir = \"{}\"\n{extra}",
        model.display(),
        data.display(),
        out.display()
    )
}

fn loss_after(log: &str) -> f32 {
    let line = log.lines().find(|l| l.contains("validation loss after")).expect("val loss line");
    line.rsplit(':').next().unwrap().trim().parse().unwrap()
}

#[test]
fn from_scratch_then_lora_then_merge_then_generate() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    let text = "the quick brown fox jumps over the lazy dog. ".repeat(60);
    write(&d.join("data.txt"), &text);

    // init -> full training from scratch
    ok(&["init", "--out", s(&d.join("base")), "--preset", "tiny"]);
    assert!(d.join("base/config.json").exists() && d.join("base/model.safetensors").exists());
    let cfg = d.join("full.toml");
    write(&cfg, &config(&d.join("base"), &d.join("data.txt"), &d.join("full"), "full", 30, ""));
    let log = ok(&["train", "--config", s(&cfg)]);
    assert!(log.contains("step    30/30"), "{log}");
    assert!(d.join("full/model/model.safetensors").exists() && d.join("full/log.csv").exists());
    let first_val: f32 = log.lines().find(|l| l.contains("validation loss before")).unwrap().rsplit(':').next().unwrap().trim().parse().unwrap();
    assert!(loss_after(&log) < first_val, "validation loss did not improve:\n{log}");

    // LoRA on top of the trained model, with a different text
    write(&d.join("other.txt"), &"pack my box with five dozen liquor jugs. ".repeat(60));
    let cfg = d.join("lora.toml");
    write(&cfg, &config(&d.join("full/model"), &d.join("other.txt"), &d.join("lora"), "lora", 20, ""));
    let log = ok(&["train", "--config", s(&cfg)]);
    assert!(log.contains("trainable"), "{log}");
    assert!(d.join("lora/adapter.safetensors").exists());
    assert!(std::fs::metadata(d.join("lora/adapter.safetensors")).unwrap().len() < std::fs::metadata(d.join("full/model/model.safetensors")).unwrap().len());

    // resume: asking for the same number of steps again does nothing but still works
    let log = ok(&["train", "--config", s(&cfg), "--resume"]);
    assert!(log.contains("resumed from step 20"), "{log}");

    // generate with and without the adapter, deterministic at temperature 0
    let model = s(&d.join("full/model")).to_string();
    let a1 = ok(&["generate", "--model", &model, "--tokenizer", "bytes", "--prompt", "the quick", "--tokens", "12", "--temp", "0"]);
    let a2 = ok(&["generate", "--model", &model, "--tokenizer", "bytes", "--prompt", "the quick", "--tokens", "12", "--temp", "0"]);
    assert_eq!(a1, a2);
    let with = ok(&["generate", "--model", &model, "--adapter", s(&d.join("lora/adapter.safetensors")), "--tokenizer", "bytes", "--prompt", "the quick", "--tokens", "12", "--temp", "0"]);
    assert!(with.starts_with("the quick"));

    // merge bakes the adapter in: the merged model without an adapter writes the same text
    ok(&["merge", "--model", &model, "--adapter", s(&d.join("lora/adapter.safetensors")), "--out", s(&d.join("merged"))]);
    let merged = ok(&["generate", "--model", s(&d.join("merged")), "--tokenizer", "bytes", "--prompt", "the quick", "--tokens", "12", "--temp", "0"]);
    assert_eq!(with, merged);
}

#[test]
fn checkpoint_memory_mode_and_jsonl_data_work() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    ok(&["init", "--out", s(&d.join("base"))]);
    let jsonl: String = (0..40).map(|i| format!("{{\"prompt\": \"q{i}: \", \"completion\": \"answer {i}\"}}\n")).collect();
    write(&d.join("sft.jsonl"), &jsonl);
    let cfg = d.join("c.toml");
    write(&cfg, &config(&d.join("base"), &d.join("sft.jsonl"), &d.join("out"), "lora", 6, "memory = \"checkpoint\"\n"));
    let log = ok(&["train", "--config", s(&cfg)]);
    assert!(log.contains("step     6/6"), "{log}");
}

#[test]
fn errors_are_clear_and_exit_nonzero() {
    let o = tf(&["train"]);
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("--config"));
    let o = tf(&["frobnicate"]);
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("unknown command"));
    let o = tf(&["generate", "--model", "/nonexistent", "--prompt", "x", "--tokenizer", "bytes"]);
    assert!(!o.status.success());
    assert!(ok(&["help"]).contains("USAGE"));
}

#[test]
fn parity_command_passes_against_its_own_logits_and_fails_on_corruption() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    ok(&["init", "--out", s(&d.join("m"))]);
    // The `generate` path and `parity` share the loader; build a reference by hand with the library.
    let model = cf_io::load_model(&d.join("m"), 6).unwrap();
    let tokens = [5u32, 9, 100, 7, 250, 3];
    let logits = model.logits(&tokens);
    std::fs::create_dir_all(d.join("ref")).unwrap();
    write(&d.join("ref/tokens.txt"), &tokens.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(" "));
    std::fs::write(d.join("ref/logits.bin"), logits.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    let out = ok(&["parity", "--model", s(&d.join("m")), "--ref", s(&d.join("ref"))]);
    assert!(out.contains("PASS"), "{out}");

    let mut bad = logits.clone();
    bad[10] += 1.0;
    std::fs::write(d.join("ref/logits.bin"), bad.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    let o = tf(&["parity", "--model", s(&d.join("m")), "--ref", s(&d.join("ref"))]);
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("FAIL"));
}
