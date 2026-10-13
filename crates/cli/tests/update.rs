//! The updater against a fake GitHub API served from this test.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::process::Command;

fn sha(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

/// Serves `/repos/o/r/releases/latest` plus the asset and checksum; returns the API base URL.
fn fake_github(tag: &'static str, payload: Vec<u8>, sha_text: String) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://127.0.0.1:{}", listener.local_addr().unwrap().port());
    let b = base.clone();
    std::thread::spawn(move || {
        for mut s in listener.incoming().flatten() {
            let mut buf = [0u8; 4096];
            let n = s.read(&mut buf).unwrap_or(0);
            let line = String::from_utf8_lossy(&buf[..n]).lines().next().unwrap_or("").to_string();
            let (ctype, body): (&str, Vec<u8>) = if line.contains("/releases/latest") {
                let asset = cpuforge_asset();
                let json = format!(
                    r#"{{"tag_name":"{tag}","assets":[{{"name":"{asset}","url":"{b}/dl/bin"}},{{"name":"{asset}.sha256","url":"{b}/dl/sha"}}]}}"#
                );
                ("application/json", json.into_bytes())
            } else if line.contains("/dl/bin") {
                ("application/octet-stream", payload.clone())
            } else if line.contains("/dl/sha") {
                ("text/plain", sha_text.clone().into_bytes())
            } else {
                ("text/plain", b"nope".to_vec())
            };
            let _ = write!(s, "HTTP/1.1 200 OK\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len());
            let _ = s.write_all(&body);
        }
    });
    base
}

fn cpuforge_asset() -> &'static str {
    if cfg!(windows) {
        "cpuforge-windows-x64.exe"
    } else if cfg!(target_os = "macos") {
        "cpuforge-macos-arm64"
    } else {
        "cpuforge-linux-x64"
    }
}

/// Copy of the real binary in a temp dir, so the test can overwrite it.
fn copy_of_binary(d: &std::path::Path) -> std::path::PathBuf {
    let p = d.join(if cfg!(windows) { "cpuforge.exe" } else { "cpuforge" });
    std::fs::copy(env!("CARGO_BIN_EXE_cpuforge"), &p).unwrap();
    p
}

fn run(exe: &std::path::Path, api: &str, extra: &[&str]) -> (bool, String) {
    let o = Command::new(exe).args(["update", "--api", api, "--repo", "o/r"]).args(extra).output().unwrap();
    (o.status.success(), format!("{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr)))
}

#[test]
fn installs_a_newer_release_and_verifies_it() {
    let d = tempfile::tempdir().unwrap();
    let exe = copy_of_binary(d.path());
    let payload = b"#!/bin/sh\necho new version\n".to_vec();

    // --check reports but changes nothing
    let api = fake_github("build-5", payload.clone(), sha(&payload));
    let before = std::fs::read(&exe).unwrap();
    let (ok, out) = run(&exe, &api, &["--check"]);
    assert!(ok && out.contains("update available"), "{out}");
    assert_eq!(std::fs::read(&exe).unwrap(), before);

    // a wrong checksum is refused and leaves the program untouched
    let bad = fake_github("build-5", payload.clone(), "0".repeat(64));
    let (ok, out) = run(&exe, &bad, &[]);
    assert!(!ok && out.contains("checksum mismatch"), "{out}");
    assert_eq!(std::fs::read(&exe).unwrap(), before);

    // the real install
    let (ok, out) = run(&exe, &api, &[]);
    assert!(ok && out.contains("updated to build-5"), "{out}");
    assert_eq!(std::fs::read(&exe).unwrap(), payload);
    assert!(!d.path().join("cpuforge.new").exists());
}

#[test]
fn does_nothing_when_already_current() {
    let d = tempfile::tempdir().unwrap();
    let exe = copy_of_binary(d.path());
    let payload = b"x".to_vec();
    let api = fake_github("build-0", payload.clone(), sha(&payload)); // this test binary is build 0
    let before = std::fs::read(&exe).unwrap();
    let (ok, out) = run(&exe, &api, &[]);
    assert!(ok && out.contains("already up to date"), "{out}");
    assert_eq!(std::fs::read(&exe).unwrap(), before);
}

#[test]
fn network_errors_are_reported_cleanly() {
    let d = tempfile::tempdir().unwrap();
    let exe = copy_of_binary(d.path());
    let (ok, out) = run(&exe, "http://127.0.0.1:1", &[]);
    assert!(!ok && out.contains("cannot reach"), "{out}");
}

/// The UI's "Install" button: download, swap, and exec into the new binary with the same arguments.
#[cfg(unix)]
#[test]
fn ui_update_restarts_into_the_new_binary() {
    use std::io::BufRead;
    let d = tempfile::tempdir().unwrap();
    let exe = copy_of_binary(d.path());
    let payload = b"#!/bin/sh\necho \"$@\" > marker\nsleep 3\n".to_vec();
    let api = fake_github("build-9", payload.clone(), sha(&payload));
    let mut child = Command::new(&exe)
        .args(["ui", "--port", "0", "--no-open"])
        .env("CPUFORGE_API", &api)
        .env("CPUFORGE_REPO", "o/r")
        .current_dir(d.path())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let mut line = String::new();
    std::io::BufReader::new(child.stdout.as_mut().unwrap()).read_line(&mut line).unwrap();
    let port: u16 = line.split("127.0.0.1:").nth(1).unwrap().split(|c: char| !c.is_ascii_digit()).next().unwrap().parse().unwrap();

    let ask = |method: &str, path: &str| {
        let mut s = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
        write!(s, "{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{{}}").unwrap();
        let mut out = String::new();
        s.read_to_string(&mut out).unwrap();
        out
    };
    let out = ask("GET", "/api/update/check");
    assert!(out.contains(r#""available":true"#), "{out}");
    let out = ask("POST", "/api/update");
    assert!(out.contains("build-9"), "{out}");

    let marker = d.path().join("marker");
    for _ in 0..50 {
        if marker.exists() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    let text = std::fs::read_to_string(&marker).expect("new binary was started");
    assert_eq!(text.trim(), "ui --port 0 --no-open"); // same arguments, same process
    let _ = child.kill();
    let _ = child.wait();
}
