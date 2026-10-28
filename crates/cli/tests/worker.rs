//! The Brewery worker (`cpuforge worker`) on the real binary: contract files, stop/resume, failures.

use std::path::Path;
use std::process::{Command, Output};
use std::time::{Duration, Instant};

use serde_json::Value;

fn cf(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_cpuforge")).args(args).output().unwrap()
}

fn ok(args: &[&str]) -> String {
    let o = cf(args);
    assert!(o.status.success(), "`cpuforge {}` failed:\n{}{}", args.join(" "), String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr));
    String::from_utf8_lossy(&o.stdout).into_owned()
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

/// Byte-level examples in Brewery's format: the prompt is masked with -100, the answer is learned.
fn tokenized_data(n: usize) -> String {
    (0..n)
        .map(|i| {
            let (prompt, answer) = (format!("Q{}: say the word. ", i % 7), "the quick brown fox jumps over the lazy dog.");
            let ids: Vec<i64> = prompt.bytes().chain(answer.bytes()).map(|b| b as i64).collect();
            let labels: Vec<i64> = ids.iter().enumerate().map(|(k, &t)| if k < prompt.len() { -100 } else { t }).collect();
            serde_json::json!({"input_ids": ids, "labels": labels}).to_string()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn setup(root: &Path, lora: bool, extra_train: &str) -> std::path::PathBuf {
    ok(&["init", "--out", s(&root.join("base")), "--preset", "tiny"]);
    std::fs::write(root.join("base/tokenizer_config.json"), "{}").unwrap(); // a tokenizer file that must travel with the result
    let run = root.join("runs/job1");
    std::fs::create_dir_all(run.join("data")).unwrap();
    std::fs::write(run.join("data/train.tok.jsonl"), tokenized_data(24)).unwrap();
    std::fs::write(run.join("data/eval.tok.jsonl"), tokenized_data(4)).unwrap();
    let lora_section = if lora { "[lora]\nrank = 4\nalpha = 8\ntargets = [\"q_proj\", \"v_proj\", \"down_proj\"]\n" } else { "" };
    std::fs::write(
        run.join("cpuforge.toml"),
        format!(
            "[run]\njob_id = \"job1\"\nbase_model = \"org/tiny\"\nobjective = \"sft\"\n[model]\npath = \"{}\"\n[data]\ntrain = \"data/train.tok.jsonl\"\neval = \"data/eval.tok.jsonl\"\nseq_len = 64\n{lora_section}\
             [train]\nsteps = 6\nbatch = 2\nlr = 3e-3\nwarmup_steps = 1\neval_every = 3\nsave_every = 2\n{extra_train}",
            s(&root.join("base"))
        ),
    )
    .unwrap();
    run
}

fn status(run: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(run.join("status.json")).unwrap()).unwrap()
}

#[test]
fn lora_job_follows_the_brewery_contract_and_produces_a_peft_adapter() {
    let d = tempfile::tempdir().unwrap();
    let run = setup(d.path(), true, "");
    // Long enough for a rank-4 adapter on a random byte model to memorise the answer.
    let cfg = std::fs::read_to_string(run.join("cpuforge.toml")).unwrap().replace("steps = 6", "steps = 150").replace("eval_every = 3", "eval_every = 75").replace("lr = 3e-3", "lr = 1e-2");
    std::fs::write(run.join("cpuforge.toml"), cfg).unwrap();
    ok(&["worker", "--config", s(&run.join("cpuforge.toml"))]);

    let st = status(&run);
    assert_eq!(st["state"], "completed", "{st}");
    assert_eq!((st["step"].as_u64(), st["max_steps"].as_u64()), (Some(150), Some(150)));
    assert_eq!(st["engine"], "cpuforge");
    assert!(st["trainable_params"].as_u64().unwrap() > 0 && st["trainable_params"].as_u64().unwrap() < st["total_params"].as_u64().unwrap());
    assert_eq!(st["loss_history"].as_array().unwrap().len(), 150);
    assert!(st["final_metrics"]["train_loss"].is_number() && st["final_metrics"]["eval_loss"].is_number());
    assert!(st["finished_at"].is_string() && st["pid"].is_u64());

    let metrics = std::fs::read_to_string(run.join("metrics.jsonl")).unwrap();
    let rows: Vec<Value> = metrics.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    assert_eq!(rows.iter().filter(|r| r.get("loss").is_some()).count(), 150);
    assert_eq!(rows.iter().filter(|r| r.get("eval_loss").is_some()).count(), 2); // steps 75 and 150
    let (first, last) = (rows[0]["loss"].as_f64().unwrap(), rows.iter().filter_map(|r| r["loss"].as_f64()).next_back().unwrap());
    assert!(last < first, "loss did not fall: {first} -> {last}");

    // final/ is what Brewery's export expects of a LoRA run
    let fin = run.join("final");
    for f in ["adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json"] {
        assert!(fin.join(f).exists(), "missing final/{f}");
    }
    let c: Value = serde_json::from_str(&std::fs::read_to_string(fin.join("adapter_config.json")).unwrap()).unwrap();
    assert_eq!((c["r"].as_u64(), c["base_model_name_or_path"].as_str()), (Some(4), Some("org/tiny")));
    let results: Value = serde_json::from_str(&std::fs::read_to_string(run.join("results.json")).unwrap()).unwrap();
    assert_eq!(results["stopped_early"], false);

    // the PEFT folder works directly with generate and merge
    let out = ok(&["generate", "--model", s(&d.path().join("base")), "--adapter", s(&fin), "--prompt", "Q1: say the word. ", "--tokens", "20", "--temp", "0", "--tokenizer", "bytes"]);
    assert!(out.contains("the quick"), "adapter did not teach the answer: {out}");
    ok(&["merge", "--model", s(&d.path().join("base")), "--adapter", s(&fin), "--out", s(&d.path().join("merged"))]);
    assert!(d.path().join("merged/model.safetensors").exists());
}

#[test]
fn full_finetune_job_writes_a_model_folder() {
    let d = tempfile::tempdir().unwrap();
    let run = setup(d.path(), false, "memory = \"checkpoint\"\n");
    ok(&["worker", "--config", s(&run.join("cpuforge.toml"))]);
    assert_eq!(status(&run)["state"], "completed");
    assert!(run.join("final/model.safetensors").exists() && run.join("final/config.json").exists() && run.join("final/tokenizer_config.json").exists());
    assert!(!run.join("final/adapter_config.json").exists());
}

#[test]
fn stop_file_pauses_cleanly_and_resume_finishes() {
    let d = tempfile::tempdir().unwrap();
    let run = setup(d.path(), true, "");
    // Plenty of steps so the run is still going when STOP appears.
    let cfg = std::fs::read_to_string(run.join("cpuforge.toml")).unwrap().replace("steps = 6", "steps = 400").replace("save_every = 2", "save_every = 100");
    std::fs::write(run.join("cpuforge.toml"), cfg).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_cpuforge")).args(["worker", "--config", s(&run.join("cpuforge.toml"))]).spawn().unwrap();
    let t0 = Instant::now();
    loop {
        if run.join("status.json").exists() && status(&run)["step"].as_u64().unwrap_or(0) >= 2 {
            break;
        }
        assert!(t0.elapsed() < Duration::from_secs(60), "worker never reached step 2");
        std::thread::sleep(Duration::from_millis(50));
    }
    std::fs::write(run.join("STOP"), "").unwrap();
    let code = child.wait().unwrap();
    assert!(code.success());
    let st = status(&run);
    assert_eq!(st["state"], "stopped", "{st}");
    let stopped_at = st["step"].as_u64().unwrap();
    assert!((2..400).contains(&stopped_at));
    assert!(run.join("state.safetensors").exists());
    assert!(!run.join("final").exists(), "a stopped run must not publish a half-trained result");

    // resume with fewer total steps so the test stays quick
    let cfg = std::fs::read_to_string(run.join("cpuforge.toml")).unwrap().replace("steps = 400", &format!("steps = {}", stopped_at + 3));
    std::fs::write(run.join("cpuforge.toml"), cfg).unwrap();
    ok(&["worker", "--config", s(&run.join("cpuforge.toml")), "--resume"]);
    let st = status(&run);
    assert_eq!((st["state"].as_str(), st["step"].as_u64()), (Some("completed"), Some(stopped_at + 3)));
    assert!(run.join("final/adapter_model.safetensors").exists());
    assert!(st["loss_history"].as_array().unwrap().len() as u64 >= stopped_at); // curve kept across the restart
}

#[cfg(unix)]
#[test]
fn sigterm_is_treated_like_a_stop_request() {
    let d = tempfile::tempdir().unwrap();
    let run = setup(d.path(), true, "");
    let cfg = std::fs::read_to_string(run.join("cpuforge.toml")).unwrap().replace("steps = 6", "steps = 400");
    std::fs::write(run.join("cpuforge.toml"), cfg).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_cpuforge")).args(["worker", "--config", s(&run.join("cpuforge.toml"))]).spawn().unwrap();
    let t0 = Instant::now();
    while !(run.join("status.json").exists() && status(&run)["step"].as_u64().unwrap_or(0) >= 1) {
        assert!(t0.elapsed() < Duration::from_secs(60));
        std::thread::sleep(Duration::from_millis(50));
    }
    Command::new("kill").args(["-TERM", &child.id().to_string()]).status().unwrap();
    assert!(child.wait().unwrap().success());
    assert_eq!(status(&run)["state"], "stopped");
    assert!(run.join("state.safetensors").exists());
}

#[test]
fn failures_are_written_to_status_json_with_a_hint() {
    let d = tempfile::tempdir().unwrap();
    // 1) data tokenized for a bigger vocabulary than the model has
    let run = setup(d.path(), true, "");
    std::fs::write(run.join("data/train.tok.jsonl"), r#"{"input_ids":[1,2,999],"labels":[1,2,999]}"#).unwrap();
    let o = cf(&["worker", "--config", s(&run.join("cpuforge.toml"))]);
    assert!(!o.status.success());
    let st = status(&run);
    assert_eq!((st["state"].as_str(), st["error_kind"].as_str()), (Some("failed"), Some("data")), "{st}");
    assert!(st["hint"].as_str().unwrap().contains("different model") && st["error"].as_str().unwrap().contains("999"));

    // 2) model folder that does not exist and cannot be downloaded (no hf id)
    let cfg = std::fs::read_to_string(run.join("cpuforge.toml")).unwrap().replace(s(&d.path().join("base")), s(&d.path().join("nowhere")));
    std::fs::write(run.join("cpuforge.toml"), cfg).unwrap();
    assert!(!cf(&["worker", "--config", s(&run.join("cpuforge.toml"))]).status.success());
    assert!(status(&run)["error"].as_str().unwrap().contains("no config.json"));

    // 3) a model type cpuforge cannot run
    ok(&["init", "--out", s(&d.path().join("gemma")), "--preset", "tiny"]);
    let cj = std::fs::read_to_string(d.path().join("gemma/config.json")).unwrap().replacen('{', "{\"model_type\":\"gemma3\",", 1);
    std::fs::write(d.path().join("gemma/config.json"), cj).unwrap();
    let cfg = std::fs::read_to_string(run.join("cpuforge.toml")).unwrap().replace(s(&d.path().join("nowhere")), s(&d.path().join("gemma")));
    std::fs::write(run.join("cpuforge.toml"), cfg).unwrap();
    assert!(!cf(&["worker", "--config", s(&run.join("cpuforge.toml"))]).status.success());
    let st = status(&run);
    assert_eq!(st["error_kind"], "unsupported", "{st}");

    // 4) a config file that is not valid
    std::fs::write(run.join("cpuforge.toml"), "[data]\ntrain = \"x\"\n").unwrap();
    assert!(!cf(&["worker", "--config", s(&run.join("cpuforge.toml"))]).status.success());
    assert_eq!(status(&run)["state"], "failed");
}
