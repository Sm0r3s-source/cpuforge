//! `cpuforge pull` against a fake Hugging Face server served from this test.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::process::{Command, Output};
use std::sync::{Arc, Mutex};

type Log = Arc<Mutex<Vec<String>>>;

/// `files`: (name, bytes, advertised size). `status`: forces the model-lookup answer (e.g. 401).
fn fake_hf(files: Vec<(&'static str, Vec<u8>, u64)>, status: u16) -> (String, Log) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://127.0.0.1:{}", listener.local_addr().unwrap().port());
    let log: Log = Arc::default();
    let l2 = log.clone();
    std::thread::spawn(move || {
        for mut s in listener.incoming().flatten() {
            let mut buf = [0u8; 8192];
            let n = s.read(&mut buf).unwrap_or(0);
            let req = String::from_utf8_lossy(&buf[..n]).to_string();
            let line = req.lines().next().unwrap_or("").to_string();
            let auth = req.lines().find(|l| l.to_ascii_lowercase().starts_with("authorization:")).unwrap_or("").to_string();
            l2.lock().unwrap().push(format!("{line} | {auth}"));
            let (code, body): (u16, Vec<u8>) = if line.contains("/api/models/org/m/revision/main") {
                if status != 200 {
                    (status, b"gated".to_vec())
                } else {
                    let sib: Vec<String> = files.iter().map(|(n, _, sz)| format!(r#"{{"rfilename":"{n}","size":{sz}}}"#)).collect();
                    (200, format!(r#"{{"siblings":[{}]}}"#, sib.join(",")).into_bytes())
                }
            } else if let Some((_, b, _)) = files.iter().find(|(n, _, _)| line.contains(&format!("/org/m/resolve/main/{n} "))) {
                (200, b.clone())
            } else {
                (404, b"nope".to_vec())
            };
            let _ = write!(s, "HTTP/1.1 {code} X\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len());
            let _ = s.write_all(&body);
        }
    });
    (base, log)
}

fn pull(base: &str, out: &std::path::Path, token: Option<&str>) -> Output {
    let mut c = Command::new(env!("CARGO_BIN_EXE_cpuforge"));
    c.args(["pull", "--repo", "org/m", "--out", out.to_str().unwrap()]).env("CPUFORGE_HF", base).env_remove("HF_TOKEN").env_remove("HUGGING_FACE_HUB_TOKEN");
    if let Some(t) = token {
        c.env("HF_TOKEN", t);
    }
    c.output().unwrap()
}

fn text(o: &Output) -> String {
    format!("{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr))
}

fn weights() -> Vec<u8> {
    (0..5000u32).map(|i| (i % 251) as u8).collect()
}

#[test]
fn downloads_only_needed_files_sends_the_token_and_skips_complete_files_on_rerun() {
    let w = weights();
    let files = vec![
        ("config.json", b"{}".to_vec(), 2),
        ("model.safetensors", w.clone(), w.len() as u64),
        ("tokenizer.json", b"{\"t\":1}".to_vec(), 7),
        ("pytorch_model.bin", vec![1; 10], 10), // never fetched
        ("README.md", vec![2; 10], 10),
    ];
    let (base, log) = fake_hf(files, 200);
    let d = tempfile::tempdir().unwrap();
    let out = d.path().join("m");

    let o = pull(&base, &out, Some("hf_secret"));
    assert!(o.status.success(), "{}", text(&o));
    assert_eq!(std::fs::read(out.join("model.safetensors")).unwrap(), w);
    assert!(out.join("config.json").exists() && out.join("tokenizer.json").exists());
    assert!(!out.join("pytorch_model.bin").exists() && !out.join("README.md").exists());
    assert!(std::fs::read_dir(&out).unwrap().all(|e| !e.unwrap().file_name().to_string_lossy().ends_with(".part")));
    {
        let l = log.lock().unwrap();
        assert!(l.iter().all(|r| r.contains("Bearer hf_secret")), "token missing on a request: {l:?}");
        assert!(!l.iter().any(|r| r.contains("pytorch_model.bin") || r.contains("README")));
    }

    // second run: nothing is downloaded again, only the lookup happens
    let before = log.lock().unwrap().len();
    let o = pull(&base, &out, None);
    assert!(o.status.success() && text(&o).contains("already complete"), "{}", text(&o));
    assert_eq!(log.lock().unwrap().len(), before + 1);

    // a truncated file is fetched again
    std::fs::write(out.join("model.safetensors"), b"short").unwrap();
    assert!(pull(&base, &out, None).status.success());
    assert_eq!(std::fs::read(out.join("model.safetensors")).unwrap(), w);
}

#[test]
fn gated_models_explain_what_to_do() {
    let (base, _) = fake_hf(vec![], 401);
    let d = tempfile::tempdir().unwrap();
    let o = pull(&base, &d.path().join("m"), None);
    assert!(!o.status.success());
    let t = text(&o);
    assert!(t.contains("gated") && t.contains("HF_TOKEN"), "{t}");
}

#[test]
fn unknown_repo_and_unusable_repos_give_clear_errors() {
    let d = tempfile::tempdir().unwrap();
    let (base, _) = fake_hf(vec![], 404);
    assert!(text(&pull(&base, &d.path().join("a"), None)).contains("404"));

    let (base, _) = fake_hf(vec![("config.json", b"{}".to_vec(), 2), ("model.bin", vec![0], 1)], 200);
    assert!(text(&pull(&base, &d.path().join("b"), None)).contains("no .safetensors"));

    let (base, _) = fake_hf(vec![("model.safetensors", vec![0; 4], 4)], 200);
    assert!(text(&pull(&base, &d.path().join("c"), None)).contains("no config.json"));
}

#[test]
fn suspicious_file_names_from_the_server_are_refused() {
    let (base, _) = fake_hf(vec![("config.json", b"{}".to_vec(), 2), ("..evil.safetensors", vec![0; 4], 4)], 200);
    let d = tempfile::tempdir().unwrap();
    let out = d.path().join("m");
    let o = pull(&base, &out, None);
    assert!(!o.status.success() && text(&o).contains("suspicious"), "{}", text(&o));
    assert!(!d.path().join("evil.safetensors").exists());
}

#[test]
fn a_short_download_is_not_kept() {
    // The server advertises 5000 bytes but sends 10.
    let (base, _) = fake_hf(vec![("config.json", b"{}".to_vec(), 2), ("model.safetensors", vec![7; 10], 5000)], 200);
    let d = tempfile::tempdir().unwrap();
    let out = d.path().join("m");
    let o = pull(&base, &out, None);
    assert!(!o.status.success() && text(&o).contains("incomplete"), "{}", text(&o));
    assert!(!out.join("model.safetensors").exists() && !out.join("model.safetensors.part").exists());
}
