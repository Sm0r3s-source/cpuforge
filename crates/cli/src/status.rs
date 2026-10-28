//! `status.json` + `metrics.jsonl` in Brewery's format, so Brewery can follow a cpuforge run the
//! same way it follows a GPU run (locally or over SSH).

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde_json::{json, Map, Value};

pub const STOP_FILE: &str = "STOP";

pub struct Status {
    dir: PathBuf,
    started: Instant,
    data: Map<String, Value>,
}

/// Local time as `2026-10-28T19:07:00+0000` (Brewery's `time.strftime("%Y-%m-%dT%H:%M:%S%z")`), computed without a time zone database.
pub fn now_string() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) as i64;
    let (days, rem) = (secs.div_euclid(86400), secs.rem_euclid(86400));
    // civil-from-days (Howard Hinnant)
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let (mut y, doy) = (yoe + era * 400, doe - (365 * yoe + yoe / 4 - yoe / 100));
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    if m <= 2 {
        y += 1;
    }
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}+0000", rem / 3600, rem % 3600 / 60, rem % 60)
}

impl Status {
    pub fn new(dir: &Path) -> Status {
        let _ = std::fs::create_dir_all(dir);
        let mut data = Map::new();
        data.insert("state".into(), "queued".into());
        data.insert("pid".into(), std::process::id().into());
        data.insert("started_at".into(), now_string().into());
        data.insert("loss_history".into(), json!([]));
        // Keep the loss curve across a resume.
        if let Some(prev) = std::fs::read_to_string(dir.join("status.json")).ok().and_then(|t| serde_json::from_str::<Value>(&t).ok()) {
            if let Some(h) = prev.get("loss_history") {
                data.insert("loss_history".into(), h.clone());
            }
        }
        let mut s = Status { dir: dir.to_path_buf(), started: Instant::now(), data };
        s.flush();
        s
    }

    pub fn set(&mut self, fields: Value) {
        if let Value::Object(m) = fields {
            for (k, v) in m {
                self.data.insert(k, v);
            }
        }
        self.data.insert("elapsed_s".into(), ((self.started.elapsed().as_secs_f64() * 10.0).round() / 10.0).into());
        self.flush();
    }

    pub fn log_metrics(&mut self, record: Value) {
        let Value::Object(mut m) = record else { return };
        m.insert("time".into(), now_string().into());
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(self.dir.join("metrics.jsonl")) {
            let _ = writeln!(f, "{}", Value::Object(m.clone()));
        }
        if let (Some(step), Some(loss)) = (m.get("step").and_then(Value::as_u64), m.get("loss").and_then(Value::as_f64)) {
            let hist = self.data.entry("loss_history").or_insert_with(|| json!([]));
            if let Some(h) = hist.as_array_mut() {
                h.push(json!([step, (loss * 1e5).round() / 1e5]));
                if h.len() > 400 {
                    let cut = h.len() - 400;
                    h.drain(..cut);
                }
            }
        }
    }

    pub fn fail(&mut self, error: &str, hint: Option<&str>, kind: &str) {
        let tail: String = error.chars().rev().take(4000).collect::<Vec<_>>().into_iter().rev().collect();
        self.data.insert("state".into(), "failed".into());
        self.data.insert("error".into(), tail.into());
        self.data.insert("error_kind".into(), kind.into());
        self.data.insert("hint".into(), hint.map_or(Value::Null, |h| h.into()));
        self.data.insert("finished_at".into(), now_string().into());
        self.flush();
    }

    #[cfg(test)]
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.data.get(key)
    }

    fn flush(&mut self) {
        self.data.insert("updated_at".into(), now_string().into());
        let tmp = self.dir.join(format!(".status.json.{}.tmp", std::process::id()));
        if std::fs::write(&tmp, serde_json::to_string_pretty(&self.data).unwrap_or_default()).is_ok() {
            let _ = std::fs::rename(&tmp, self.dir.join("status.json"));
        }
    }
}

/// Turn an error message into Brewery's (kind, hint) pair, so the agent can explain it in plain words.
pub fn classify_error(text: &str) -> (&'static str, Option<&'static str>) {
    let t = text.to_ascii_lowercase();
    if t.contains("out of memory") || t.contains("cannot allocate") || t.contains("memory allocation") {
        ("oom", Some("The model does not fit in RAM. Use LoRA instead of full fine-tuning, a smaller model, a shorter max sequence length, or the low-memory (checkpoint) mode."))
    } else if t.contains("gated") || t.contains("401") || t.contains("403") {
        ("auth", Some("The model is gated or needs a Hugging Face token. Accept its licence on the model page and set HF_TOKEN."))
    } else if t.contains("cannot reach") || t.contains("connection") || t.contains("dns") {
        ("network", Some("The machine could not reach Hugging Face. Check its internet connection and try again."))
    } else if t.contains("token id") || t.contains("vocab") {
        ("data", Some("The training data uses token ids outside the model's vocabulary: it was tokenized for a different model."))
    } else if t.contains("not supported") || t.contains("unsupported") {
        ("unsupported", Some("cpuforge does not support this model architecture yet. Train it on a GPU instead."))
    } else {
        ("error", None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamps_have_brewerys_shape_and_correct_dates() {
        let s = now_string();
        assert_eq!(s.len(), 24, "{s}");
        assert!(s.starts_with("20") && s.as_bytes()[10] == b'T' && s.ends_with("+0000"));
    }

    #[test]
    fn status_file_is_atomic_json_with_loss_history_and_survives_resume() {
        let d = tempfile::tempdir().unwrap();
        let mut s = Status::new(d.path());
        s.set(json!({"state": "training", "step": 3, "max_steps": 10}));
        s.log_metrics(json!({"step": 3, "loss": 1.234567891}));
        s.set(json!({}));
        let v: Value = serde_json::from_str(&std::fs::read_to_string(d.path().join("status.json")).unwrap()).unwrap();
        assert_eq!((v["state"].as_str(), v["step"].as_u64(), v["max_steps"].as_u64()), (Some("training"), Some(3), Some(10)));
        assert_eq!(v["loss_history"], json!([[3, 1.23457]]));
        assert!(v["pid"].is_u64() && v["started_at"].is_string() && v["updated_at"].is_string());
        let line = std::fs::read_to_string(d.path().join("metrics.jsonl")).unwrap();
        assert!(line.contains("\"loss\":1.234567891") && line.contains("\"time\""));
        assert!(!std::fs::read_dir(d.path()).unwrap().any(|e| e.unwrap().file_name().to_string_lossy().ends_with(".tmp")));

        let again = Status::new(d.path()); // a resumed worker keeps the curve
        assert_eq!(again.get("loss_history"), Some(&json!([[3, 1.23457]])));
    }

    #[test]
    fn failures_carry_kind_and_hint() {
        let d = tempfile::tempdir().unwrap();
        let mut s = Status::new(d.path());
        let (k, h) = classify_error("cannot reach https://huggingface.co/x: connection refused");
        s.fail("boom", h, k);
        let v: Value = serde_json::from_str(&std::fs::read_to_string(d.path().join("status.json")).unwrap()).unwrap();
        assert_eq!((v["state"].as_str(), v["error"].as_str(), v["error_kind"].as_str()), (Some("failed"), Some("boom"), Some("network")));
        assert!(v["hint"].as_str().unwrap().contains("internet"));
        assert_eq!(classify_error("tensor shape mismatch").0, "error");
        assert_eq!(classify_error("model_type 'gemma3' is not supported").0, "unsupported");
    }
}
