//! Starts the real `cpuforge ui` and drives it over HTTP like the browser page does.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

struct Ui {
    child: Child,
    port: u16,
}

impl Drop for Ui {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn start(dir: &std::path::Path) -> Ui {
    let mut child = Command::new(env!("CARGO_BIN_EXE_cpuforge"))
        .args(["ui", "--port", "0", "--no-open"])
        .current_dir(dir)
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut line = String::new();
    BufReader::new(child.stdout.as_mut().unwrap()).read_line(&mut line).unwrap();
    let port = line.split("127.0.0.1:").nth(1).unwrap().split(|c: char| !c.is_ascii_digit()).next().unwrap().parse().unwrap();
    Ui { child, port }
}

/// Returns (status code, body).
fn http(ui: &Ui, method: &str, path: &str, body: Option<&str>, extra: &str) -> (u16, String) {
    let mut s = TcpStream::connect(("127.0.0.1", ui.port)).unwrap();
    let b = body.unwrap_or("");
    let ct = if body.is_some() { "Content-Type: application/json\r\n" } else { "" };
    write!(s, "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{}\r\n{ct}{extra}Content-Length: {}\r\n\r\n{b}", ui.port, b.len()).unwrap();
    let mut out = String::new();
    s.read_to_string(&mut out).unwrap();
    let code = out.split_whitespace().nth(1).unwrap().parse().unwrap();
    (code, out.split("\r\n\r\n").nth(1).unwrap_or("").to_string())
}

fn json(ui: &Ui, method: &str, path: &str, body: Option<&str>) -> (u16, serde_json::Value) {
    let (c, b) = http(ui, method, path, body, "");
    (c, serde_json::from_str(&b).unwrap_or_else(|_| panic!("not json: {b}")))
}

/// Wait for the current job to end; returns (exit code, whole log).
fn wait_job(ui: &Ui) -> (i64, String) {
    let (mut since, mut log) = (0, String::new());
    let t0 = Instant::now();
    loop {
        let (_, j) = json(ui, "GET", &format!("/api/job?since={since}"), None);
        log.push_str(j["text"].as_str().unwrap());
        since = j["next"].as_u64().unwrap();
        if !j["running"].as_bool().unwrap() {
            return (j["exit"].as_i64().unwrap(), log);
        }
        assert!(t0.elapsed() < Duration::from_secs(120), "job timed out:\n{log}");
        std::thread::sleep(Duration::from_millis(100));
    }
}

#[test]
fn serves_page_and_rejects_foreign_requests() {
    let d = tempfile::tempdir().unwrap();
    let ui = start(d.path());
    let (c, body) = http(&ui, "GET", "/", None, "");
    assert_eq!(c, 200);
    assert!(body.contains("<title>cpuforge</title>"));
    // DNS rebinding style Host, foreign Origin, and a form-style POST are all refused
    let mut s = TcpStream::connect(("127.0.0.1", ui.port)).unwrap();
    write!(s, "GET /api/files HTTP/1.1\r\nHost: evil.example\r\n\r\n").unwrap();
    let mut out = String::new();
    s.read_to_string(&mut out).unwrap();
    assert!(out.starts_with("HTTP/1.1 403"), "{out}");
    assert_eq!(http(&ui, "GET", "/api/files", None, "Origin: https://evil.example\r\n").0, 403);
    assert_eq!(http(&ui, "POST", "/api/cancel", None, "").0, 403); // no JSON content type
    assert_eq!(http(&ui, "GET", "/nope", None, "").0, 404);
    let (c, v) = json(&ui, "GET", "/api/version", None);
    assert_eq!((c, v["build"].as_u64()), (200, Some(0)));
    // path escapes are rejected before any process starts
    let (c, j) = json(&ui, "POST", "/api/run/init", Some(r#"{"out":"../escape"}"#));
    assert_eq!(c, 400);
    assert!(j["error"].as_str().unwrap().contains(".."));
    assert!(!d.path().join("../escape").exists());
}

#[test]
fn full_workflow_from_the_page() {
    let d = tempfile::tempdir().unwrap();
    let ui = start(d.path());

    let (c, j) = json(&ui, "POST", "/api/data", Some(&serde_json::json!({"name": "t.txt", "text": "the quick brown fox jumps over the lazy dog. ".repeat(50)}).to_string()));
    assert_eq!((c, j["path"].as_str().unwrap()), (200, "data/t.txt"));
    assert_eq!(json(&ui, "POST", "/api/data", Some(r#"{"name":"../x.txt","text":"a"}"#)).0, 400);

    assert_eq!(json(&ui, "POST", "/api/run/init", Some(r#"{"out":"models/m","preset":"tiny"}"#)).0, 200);
    assert_eq!(wait_job(&ui).0, 0);
    let (_, f) = json(&ui, "GET", "/api/files", None);
    assert_eq!(f["models"], serde_json::json!(["models/m"]));
    assert_eq!(f["data"], serde_json::json!(["data/t.txt"]));

    let train = serde_json::json!({"model":"models/m","data":"data/t.txt","mode":"lora","steps":4,"batch":2,"lr":0.003,"seq_len":48,
        "rank":4,"alpha":8,"eval_every":2,"save_every":2,"out_dir":"runs/ui"});
    let (c, _) = json(&ui, "POST", "/api/run/train", Some(&train.to_string()));
    assert_eq!(c, 200);
    // a second job while one runs is refused
    let (c2, e) = json(&ui, "POST", "/api/run/init", Some(r#"{"out":"models/other"}"#));
    assert_eq!(c2, 400, "{e}");
    let (code, log) = wait_job(&ui);
    assert_eq!(code, 0, "{log}");
    assert!(log.contains("step     4/4  loss"), "{log}");
    assert!(log.contains("adapter saved"), "{log}");
    assert!(d.path().join("runs/ui/adapter.safetensors").exists());
    assert!(d.path().join("runs/ui/ui-config.toml").exists());

    let gen = serde_json::json!({"model":"models/m","adapter":"runs/ui/adapter.safetensors","prompt":"the quick","tokens":12,"temp":0});
    assert_eq!(json(&ui, "POST", "/api/run/generate", Some(&gen.to_string())).0, 200);
    let (code, log) = wait_job(&ui);
    assert_eq!(code, 0, "{log}");
    assert!(log.starts_with("the quick"), "{log}");

    let merge = serde_json::json!({"model":"models/m","adapter":"runs/ui/adapter.safetensors","out":"models/merged"});
    assert_eq!(json(&ui, "POST", "/api/run/merge", Some(&merge.to_string())).0, 200);
    assert_eq!(wait_job(&ui).0, 0);
    assert!(d.path().join("models/merged/model.safetensors").exists());

    // a failing job reports a non-zero exit and its error text
    let bad = serde_json::json!({"model":"models/missing","data":"data/t.txt"});
    assert_eq!(json(&ui, "POST", "/api/run/train", Some(&bad.to_string())).0, 200);
    let (code, log) = wait_job(&ui);
    assert_ne!(code, 0);
    assert!(log.contains("error"), "{log}");
}

#[test]
fn cancel_stops_a_running_job() {
    let d = tempfile::tempdir().unwrap();
    let ui = start(d.path());
    std::fs::write(d.path().join("t.txt"), "abc def ghi ".repeat(400)).unwrap();
    json(&ui, "POST", "/api/run/init", Some(r#"{"out":"m","preset":"small"}"#));
    assert_eq!(wait_job(&ui).0, 0);
    let train = r#"{"model":"m","data":"t.txt","mode":"full","steps":100000,"batch":4,"seq_len":128,"out_dir":"r"}"#;
    assert_eq!(json(&ui, "POST", "/api/run/train", Some(train)).0, 200);
    std::thread::sleep(Duration::from_millis(1500));
    assert_eq!(json(&ui, "POST", "/api/cancel", Some("{}")).1["cancelled"], true);
    let (code, _) = wait_job(&ui);
    assert_ne!(code, 0);
    assert_eq!(json(&ui, "POST", "/api/cancel", Some("{}")).1["cancelled"], false);
}
