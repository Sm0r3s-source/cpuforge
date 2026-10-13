//! `cpuforge ui [--port 7860] [--bind IP] [--no-open] [--auto-update]`
//!
//! A local web app: pick a model and data, press Train, watch the loss curve,
//! chat with the result, merge adapters. It is a thin shell around the same
//! commands as the CLI: every action runs `cpuforge <command>` as a child
//! process (one at a time), so the UI and the CLI cannot disagree.
//!
//! Security: listens on 127.0.0.1 unless `--bind IP` is given (then no login exists, so bind
//! only to a private interface such as Tailscale); requests must carry a loopback or bound `Host`
//! header (blocks DNS rebinding) and no foreign `Origin`; POST bodies must be
//! `application/json` (cross-site pages cannot send that without a preflight,
//! and we never answer preflights); every path from the browser must be relative
//! to the folder the UI was started in and may not contain `..`.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Component, Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{json, Value};

use crate::args::Args;
use crate::cmd_train::parse_config;
use crate::cmd_update as update;

const PAGE: &str = include_str!("ui.html");
const MAX_BODY: usize = 32 << 20;

// ---------------------------------------------------------------- http

pub struct Request {
    pub method: String,
    pub path: String,
    pub query: HashMap<String, String>,
    pub headers: HashMap<String, String>,
    pub body: Vec<u8>,
}

pub fn read_request(stream: &mut impl Read) -> Result<Request, String> {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    let header_end = loop {
        if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break p;
        }
        if buf.len() > 64 * 1024 {
            return Err("headers too large".into());
        }
        let n = stream.read(&mut chunk).map_err(|e| e.to_string())?;
        if n == 0 {
            return Err("connection closed".into());
        }
        buf.extend_from_slice(&chunk[..n]);
    };
    let head = String::from_utf8_lossy(&buf[..header_end]).into_owned();
    let mut lines = head.split("\r\n");
    let mut first = lines.next().unwrap_or("").split_whitespace();
    let (method, target) = (first.next().unwrap_or(""), first.next().unwrap_or(""));
    if method.is_empty() || target.is_empty() {
        return Err("bad request line".into());
    }
    let headers: HashMap<String, String> = lines
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_string()))
        .collect();
    let len: usize = headers.get("content-length").map(|v| v.parse().map_err(|_| "bad content-length")).transpose()?.unwrap_or(0);
    if len > MAX_BODY {
        return Err("body too large".into());
    }
    let mut body = buf[header_end + 4..].to_vec();
    while body.len() < len {
        let n = stream.read(&mut chunk).map_err(|e| e.to_string())?;
        if n == 0 {
            return Err("body truncated".into());
        }
        body.extend_from_slice(&chunk[..n]);
    }
    body.truncate(len);
    let (path, q) = target.split_once('?').unwrap_or((target, ""));
    let query = q.split('&').filter_map(|kv| kv.split_once('=')).map(|(k, v)| (k.to_string(), v.to_string())).collect();
    Ok(Request { method: method.to_string(), path: path.to_string(), query, headers, body })
}

fn respond(stream: &mut TcpStream, status: &str, ctype: &str, body: &[u8]) {
    let head = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(body);
}

fn host_allowed(host: &str, extra: Option<&str>) -> bool {
    let h = host.rsplit_once(':').map(|(h, _)| h).unwrap_or(host);
    matches!(h, "127.0.0.1" | "localhost" | "[::1]") || extra.is_some_and(|e| e == h)
}

/// Reject requests that did not come from our own page.
pub fn check_origin(req: &Request, extra: Option<&str>) -> Result<(), String> {
    match req.headers.get("host") {
        Some(h) if host_allowed(h, extra) => {}
        _ => return Err("bad Host header".into()),
    }
    if let Some(o) = req.headers.get("origin") {
        let host = o.split("://").nth(1).unwrap_or("");
        if !host_allowed(host, extra) {
            return Err("foreign Origin".into());
        }
    }
    if req.method == "POST" {
        let ct = req.headers.get("content-type").map(|s| s.to_ascii_lowercase()).unwrap_or_default();
        if !ct.starts_with("application/json") {
            return Err("POST needs Content-Type: application/json".into());
        }
    }
    Ok(())
}

// ---------------------------------------------------------------- paths

/// A path from the browser: non-empty, relative, no `..`, no NUL.
pub fn safe_rel(p: &str) -> Result<String, String> {
    let path = Path::new(p);
    if p.is_empty() || p.contains('\0') || path.is_absolute() {
        return Err(format!("path '{p}' must be relative to the folder the UI runs in"));
    }
    if path.components().any(|c| !matches!(c, Component::Normal(_) | Component::CurDir)) {
        return Err(format!("path '{p}' may not contain '..'"));
    }
    Ok(p.to_string())
}

fn safe_name(n: &str) -> Result<String, String> {
    let ok = !n.is_empty()
        && n.len() <= 80
        && n.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
        && !n.starts_with('.')
        && (n.ends_with(".txt") || n.ends_with(".jsonl"));
    if ok { Ok(n.to_string()) } else { Err("file name must be letters/digits/._- and end in .txt or .jsonl".into()) }
}

fn s<'a>(v: &'a Value, key: &str) -> Result<&'a str, String> {
    v.get(key).and_then(Value::as_str).filter(|x| !x.is_empty()).ok_or_else(|| format!("missing '{key}'"))
}

fn num(v: &Value, key: &str, default: f64) -> Result<f64, String> {
    match v.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Number(n)) => n.as_f64().ok_or_else(|| format!("'{key}' is not a number")),
        Some(Value::String(t)) if !t.trim().is_empty() => t.trim().parse().map_err(|_| format!("'{key}': cannot parse '{t}'")),
        Some(Value::String(_)) => Ok(default),
        _ => Err(format!("'{key}' is not a number")),
    }
}

/// Tokenizer spec: explicit choice, else the model's tokenizer.json, else bytes.
fn tokenizer_for(root: &Path, model: &str, explicit: Option<&str>) -> String {
    match explicit {
        Some(t) if !t.is_empty() => t.to_string(),
        _ if root.join(model).join("tokenizer.json").exists() => "auto".into(),
        _ => "bytes".into(),
    }
}

// ---------------------------------------------------------------- requests -> commands

/// Build the TOML for a train run from the form.
pub fn train_toml(root: &Path, v: &Value) -> Result<(String, String), String> {
    let model = safe_rel(s(v, "model")?)?;
    let data = safe_rel(s(v, "data")?)?;
    let out = safe_rel(v.get("out_dir").and_then(Value::as_str).filter(|x| !x.is_empty()).unwrap_or("runs/ui"))?;
    let mode = v.get("mode").and_then(Value::as_str).unwrap_or("lora");
    let tok = tokenizer_for(root, &model, v.get("tokenizer").and_then(Value::as_str));

    let mut m = toml::Table::new();
    m.insert("path".into(), model.into());
    if tok != "auto" {
        m.insert("tokenizer".into(), tok.into());
    }
    let mut d = toml::Table::new();
    d.insert("path".into(), data.into());
    d.insert("seq_len".into(), (num(v, "seq_len", 256.0)? as i64).into());
    d.insert("val_fraction".into(), num(v, "val_fraction", 0.05)?.into());
    let mut l = toml::Table::new();
    l.insert("rank".into(), (num(v, "rank", 16.0)? as i64).into());
    l.insert("alpha".into(), num(v, "alpha", 32.0)?.into());
    let mut t = toml::Table::new();
    t.insert("mode".into(), mode.to_string().into());
    t.insert("steps".into(), (num(v, "steps", 100.0)? as i64).into());
    t.insert("batch".into(), (num(v, "batch", 4.0)? as i64).into());
    t.insert("lr".into(), num(v, "lr", 2e-4)?.into());
    t.insert("memory".into(), v.get("memory").and_then(Value::as_str).unwrap_or("full").to_string().into());
    t.insert("eval_every".into(), (num(v, "eval_every", 0.0)? as i64).into());
    t.insert("save_every".into(), (num(v, "save_every", 25.0)? as i64).into());
    t.insert("out_dir".into(), out.clone().into());
    let mut top = toml::Table::new();
    top.insert("model".into(), m.into());
    top.insert("data".into(), d.into());
    top.insert("lora".into(), l.into());
    top.insert("train".into(), t.into());
    let text = toml::to_string(&top).map_err(|e| e.to_string())?;
    parse_config(&text)?; // same validation as the CLI
    Ok((text, out))
}

/// Argument list (after the program name) for an action.
/// Optional file (path, contents) to write before the command runs.
type Pending = Option<(PathBuf, String)>;

pub fn command_for(root: &Path, action: &str, v: &Value) -> Result<(Vec<String>, Pending), String> {
    let mut a: Vec<String> = Vec::new();
    let mut write = None;
    let mut push = |items: &[&str]| a.extend(items.iter().map(|x| x.to_string()));
    match action {
        "train" => {
            let (text, out) = train_toml(root, v)?;
            let cfg = Path::new(&out).join("ui-config.toml");
            push(&["train", "--config", &cfg.to_string_lossy()]);
            if v.get("resume").and_then(Value::as_bool) == Some(true) {
                push(&["--resume"]);
            }
            write = Some((root.join(&cfg), text));
        }
        "init" => {
            let out = safe_rel(s(v, "out")?)?;
            let preset = s(v, "preset").unwrap_or("tiny");
            push(&["init", "--out", &out, "--preset", preset]);
        }
        "generate" => {
            let model = safe_rel(s(v, "model")?)?;
            let tok = tokenizer_for(root, &model, v.get("tokenizer").and_then(Value::as_str));
            push(&["generate", "--model", &model, "--prompt", s(v, "prompt")?]);
            if tok != "auto" {
                push(&["--tokenizer", &tok]);
            }
            if let Some(ad) = v.get("adapter").and_then(Value::as_str).filter(|x| !x.is_empty()) {
                push(&["--adapter", &safe_rel(ad)?]);
            }
            let (n, t, k) = (num(v, "tokens", 64.0)?, num(v, "temp", 0.8)?, num(v, "top_k", 40.0)?);
            push(&["--tokens", &(n as i64).to_string(), "--temp", &t.to_string(), "--top-k", &(k as i64).to_string()]);
        }
        "merge" => {
            let (model, ad, out) = (safe_rel(s(v, "model")?)?, safe_rel(s(v, "adapter")?)?, safe_rel(s(v, "out")?)?);
            push(&["merge", "--model", &model, "--adapter", &ad, "--out", &out]);
        }
        other => return Err(format!("unknown action '{other}'")),
    }
    Ok((a, write))
}

/// What is on disk that the UI can offer: models, adapters, data files.
pub fn scan(root: &Path) -> Value {
    fn walk(dir: &Path, depth: usize, out: &mut [Vec<String>; 3], root: &Path) {
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        let mut entries: Vec<_> = rd.flatten().collect();
        entries.sort_by_key(|e| e.file_name());
        for e in entries {
            let name = e.file_name().to_string_lossy().into_owned();
            let p = e.path();
            let rel = p.strip_prefix(root).unwrap_or(&p).to_string_lossy().into_owned();
            if p.is_dir() {
                if matches!(name.as_str(), "target" | ".git" | "node_modules") || name.starts_with('.') {
                    continue;
                }
                if p.join("config.json").exists() {
                    out[0].push(rel.clone());
                }
                if depth < 4 {
                    walk(&p, depth + 1, out, root);
                }
            } else if name.ends_with(".safetensors") && name.starts_with("adapter") {
                out[1].push(rel);
            } else if (name.ends_with(".jsonl") || name.ends_with(".txt")) && !matches!(name.as_str(), "tokens.txt" | "requirements.txt") {
                out[2].push(rel);
            }
        }
    }
    let mut out: [Vec<String>; 3] = Default::default();
    walk(root, 0, &mut out, root);
    json!({"models": out[0], "adapters": out[1], "data": out[2], "cwd": root.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()})
}

// ---------------------------------------------------------------- jobs

#[derive(Default)]
struct JobState {
    id: u64,
    kind: String,
    running: bool,
    exit: Option<i32>,
    log: Vec<u8>,
    child: Option<Child>,
}

#[derive(Clone)]
pub struct Server {
    root: PathBuf,
    exe: PathBuf,
    /// Extra address accepted in `Host` (the `--bind` address).
    bind: Option<String>,
    /// Our own command line, to restart with after an update.
    argv: Vec<String>,
    job: Arc<Mutex<JobState>>,
}

impl Server {
    pub fn new(root: PathBuf, exe: PathBuf, bind: Option<String>, argv: Vec<String>) -> Server {
        Server { root, exe, bind, argv, job: Arc::new(Mutex::new(JobState::default())) }
    }

    fn busy(&self) -> bool {
        self.job.lock().unwrap().running
    }

    /// Install the latest release over our binary and restart into it.
    fn update_and_restart(&self, rel: &update::Release, src: &update::Source) -> Result<(), String> {
        if self.busy() {
            return Err("a job is running; wait for it to finish first".into());
        }
        src.install(rel, &self.exe)?;
        let (exe, argv) = (self.exe.clone(), self.argv.clone());
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(400)); // let the HTTP response go out
            update::restart_self(&exe, &argv)
        });
        Ok(())
    }

    /// Background loop for `--auto-update`: check now and then, install when idle.
    fn auto_update_loop(&self) {
        let me = self.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_secs(30));
            loop {
                let src = update::Source::new(None, None);
                match src.latest() {
                    Ok(rel) if rel.build > update::build() && !me.busy() => {
                        println!("auto-update: installing {}", rel.tag);
                        if let Err(e) = me.update_and_restart(&rel, &src) {
                            eprintln!("auto-update failed: {e}");
                        }
                    }
                    Ok(_) => {}
                    Err(e) => eprintln!("auto-update check failed: {e}"),
                }
                std::thread::sleep(Duration::from_secs(6 * 3600));
            }
        });
    }

    fn start(&self, kind: &str, args: Vec<String>, write: Pending) -> Result<u64, String> {
        let mut j = self.job.lock().unwrap();
        if j.running {
            return Err(format!("a {} job is still running; wait or cancel it", j.kind));
        }
        if let Some((path, text)) = write {
            if let Some(dir) = path.parent() {
                std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
            }
            std::fs::write(&path, text).map_err(|e| e.to_string())?;
        }
        let mut cmd = Command::new(&self.exe);
        #[cfg(windows)]
        std::os::windows::process::CommandExt::creation_flags(&mut cmd, 0x0800_0000); // no extra console window
        let mut child = cmd
            .args(&args)
            .current_dir(&self.root)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("cannot start: {e}"))?;
        let (out, err) = (child.stdout.take().unwrap(), child.stderr.take().unwrap());
        j.id += 1;
        let id = j.id;
        *j = JobState { id, kind: kind.to_string(), running: true, exit: None, log: Vec::new(), child: Some(child) };
        drop(j);

        let pumps: Vec<_> = [Box::new(out) as Box<dyn Read + Send>, Box::new(err)]
            .into_iter()
            .map(|mut r| {
                let job = self.job.clone();
                std::thread::spawn(move || {
                    let mut buf = [0u8; 4096];
                    while let Ok(n) = r.read(&mut buf) {
                        if n == 0 {
                            break;
                        }
                        let mut j = job.lock().unwrap();
                        if j.id != id {
                            break;
                        }
                        j.log.extend_from_slice(&buf[..n]);
                    }
                })
            })
            .collect();
        let job = self.job.clone();
        std::thread::spawn(move || {
            loop {
                {
                    let mut j = job.lock().unwrap();
                    if j.id != id {
                        return;
                    }
                    if let Some(c) = j.child.as_mut() {
                        if let Ok(Some(st)) = c.try_wait() {
                            j.exit = Some(st.code().unwrap_or(-1));
                            break;
                        }
                    }
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            for p in pumps {
                let _ = p.join();
            }
            let mut j = job.lock().unwrap();
            if j.id == id {
                j.running = false;
                j.child = None;
            }
        });
        Ok(id)
    }

    fn cancel(&self) -> bool {
        let mut j = self.job.lock().unwrap();
        let running = j.running;
        match j.child.as_mut() {
            Some(c) if running => c.kill().is_ok(),
            _ => false,
        }
    }

    fn job_json(&self, since: usize) -> Value {
        let j = self.job.lock().unwrap();
        let since = since.min(j.log.len());
        let tail = &j.log[since..];
        let valid = match std::str::from_utf8(tail) {
            Ok(t) => t.len(),
            Err(e) if j.running => e.valid_up_to(),
            Err(e) => e.valid_up_to().max(tail.len()), // finished: emit lossy rest
        };
        let text = String::from_utf8_lossy(&tail[..valid]).into_owned();
        json!({"id": j.id, "kind": j.kind, "running": j.running, "exit": j.exit, "text": text, "next": since + valid})
    }

    fn handle(&self, req: &Request) -> (String, String, Vec<u8>) {
        let ok = |v: Value| ("200 OK".to_string(), "application/json".to_string(), v.to_string().into_bytes());
        let err = |code: &str, m: String| (code.to_string(), "application/json".to_string(), json!({"error": m}).to_string().into_bytes());
        if let Err(e) = check_origin(req, self.bind.as_deref()) {
            return err("403 Forbidden", e);
        }
        let body = || -> Result<Value, String> { serde_json::from_slice(&req.body).map_err(|e| format!("bad JSON: {e}")) };
        match (req.method.as_str(), req.path.as_str()) {
            ("GET", "/") => ("200 OK".into(), "text/html; charset=utf-8".into(), PAGE.as_bytes().to_vec()),
            ("GET", "/api/files") => ok(scan(&self.root)),
            ("GET", "/api/version") => ok(json!({"version": update::version(), "build": update::build()})),
            ("GET", "/api/update/check") => {
                let src = update::Source::new(None, None);
                match src.latest() {
                    Ok(r) => ok(json!({"current": update::build(), "latest": r.build, "tag": r.tag, "available": r.build > update::build()})),
                    Err(e) => err("502 Bad Gateway", e),
                }
            }
            ("POST", "/api/update") => {
                let src = update::Source::new(None, None);
                match src.latest().and_then(|r| self.update_and_restart(&r, &src).map(|_| r.tag)) {
                    Ok(tag) => ok(json!({"updated": tag})),
                    Err(e) => err("400 Bad Request", e),
                }
            }
            ("GET", "/api/job") => ok(self.job_json(req.query.get("since").and_then(|x| x.parse().ok()).unwrap_or(0))),
            ("POST", "/api/cancel") => ok(json!({"cancelled": self.cancel()})),
            ("POST", "/api/data") => {
                let r = body().and_then(|v| {
                    let name = safe_name(s(&v, "name")?)?;
                    let text = v.get("text").and_then(Value::as_str).ok_or("missing 'text'")?;
                    let dir = self.root.join("data");
                    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
                    std::fs::write(dir.join(&name), text).map_err(|e| e.to_string())?;
                    Ok(format!("data/{name}"))
                });
                match r {
                    Ok(p) => ok(json!({"path": p})),
                    Err(e) => err("400 Bad Request", e),
                }
            }
            ("POST", p) if p.starts_with("/api/run/") => {
                let action = &p["/api/run/".len()..];
                let r = body().and_then(|v| {
                    let (args, write) = command_for(&self.root, action, &v)?;
                    self.start(action, args, write)
                });
                match r {
                    Ok(id) => ok(json!({"id": id})),
                    Err(e) => err("400 Bad Request", e),
                }
            }
            ("GET", _) | ("POST", _) => err("404 Not Found", "not found".into()),
            _ => err("405 Method Not Allowed", "method not allowed".into()),
        }
    }
}

pub fn serve(listener: TcpListener, server: Server) {
    for conn in listener.incoming().flatten() {
        let server = server.clone();
        std::thread::spawn(move || {
            let mut conn = conn;
            let _ = conn.set_read_timeout(Some(Duration::from_secs(30)));
            match read_request(&mut conn) {
                Ok(req) => {
                    let (status, ctype, body) = server.handle(&req);
                    respond(&mut conn, &status, &ctype, &body);
                }
                Err(e) => respond(&mut conn, "400 Bad Request", "application/json", json!({"error": e}).to_string().as_bytes()),
            }
        });
    }
}

/// Open the page in the default browser (best effort; the URL is printed anyway).
fn open_browser(url: &str) {
    #[cfg(target_os = "windows")]
    let r = Command::new("rundll32").args(["url.dll,FileProtocolHandler", url]).spawn();
    #[cfg(target_os = "macos")]
    let r = Command::new("open").arg(url).spawn();
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    let r = Command::new("xdg-open").arg(url).stdout(Stdio::null()).stderr(Stdio::null()).spawn();
    let _ = r;
}

pub fn run(args: &Args) -> Result<(), String> {
    let port: u16 = args.parsed("port", 7860)?;
    let bind = args.get("bind").unwrap_or("127.0.0.1");
    let listener = TcpListener::bind((bind, port)).map_err(|e| format!("cannot listen on {bind}:{port}: {e}"))?;
    let addr = listener.local_addr().map_err(|e| e.to_string())?;
    let root = std::env::current_dir().map_err(|e| e.to_string())?;
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let url = format!("http://{bind}:{}", addr.port());
    println!("cpuforge ui: {url}  (folder: {}; close this window or press Ctrl-C to stop)", root.display());
    for d in ["models", "data"] {
        let _ = std::fs::create_dir_all(root.join(d));
    }
    if !args.has("no-open") {
        open_browser(&url);
    }
    update::cleanup_old(&exe);
    let extra = (bind != "127.0.0.1" && bind != "0.0.0.0").then(|| bind.to_string());
    let server = Server::new(root, exe, extra, std::env::args().skip(1).collect());
    if args.has("auto-update") {
        if update::build() == 0 {
            eprintln!("--auto-update ignored: this is a development build");
        } else {
            server.auto_update_loop();
        }
    }
    serve(listener, server);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(raw: &str) -> Request {
        read_request(&mut raw.as_bytes()).unwrap()
    }

    #[test]
    fn parses_requests_with_body_and_query() {
        let r = req("POST /api/x?since=12&a=b HTTP/1.1\r\nHost: localhost:7860\r\nContent-Length: 5\r\nContent-Type: application/json\r\n\r\nhello");
        assert_eq!((r.method.as_str(), r.path.as_str()), ("POST", "/api/x"));
        assert_eq!(r.query["since"], "12");
        assert_eq!(r.body, b"hello");
        assert!(read_request(&mut "GARBAGE".as_bytes()).is_err());
        assert!(read_request(&mut "POST / HTTP/1.1\r\nContent-Length: 99999999999\r\n\r\n".as_bytes()).is_err());
    }

    #[test]
    fn origin_checks_block_rebinding_and_cross_site_posts() {
        let good = req("GET / HTTP/1.1\r\nHost: 127.0.0.1:7860\r\n\r\n");
        assert!(check_origin(&good, None).is_ok());
        assert!(check_origin(&req("GET / HTTP/1.1\r\nHost: evil.example:7860\r\n\r\n"), None).is_err());
        assert!(check_origin(&req("GET / HTTP/1.1\r\n\r\n"), None).is_err());
        assert!(check_origin(&req("GET / HTTP/1.1\r\nHost: localhost\r\nOrigin: https://evil.example\r\n\r\n"), None).is_err());
        assert!(check_origin(&req("GET / HTTP/1.1\r\nHost: localhost:1\r\nOrigin: http://localhost:1\r\n\r\n"), None).is_ok());
        assert!(check_origin(&req("POST / HTTP/1.1\r\nHost: localhost\r\nContent-Type: text/plain\r\nContent-Length: 0\r\n\r\n"), None).is_err());
        assert!(check_origin(&req("POST / HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 0\r\n\r\n"), None).is_ok());
        // an explicitly chosen --bind address is accepted as Host, others still are not
        let tail = req("GET / HTTP/1.1\r\nHost: 100.1.2.3:7860\r\n\r\n");
        assert!(check_origin(&tail, Some("100.1.2.3")).is_ok() && check_origin(&tail, None).is_err());
    }

    #[test]
    fn paths_must_stay_inside_the_folder() {
        for bad in ["", "/etc/passwd", "../x", "a/../../x", "a\0b"] {
            assert!(safe_rel(bad).is_err(), "{bad:?}");
        }
        for good in ["models/m", "./data/a.txt", "runs/ui"] {
            assert!(safe_rel(good).is_ok(), "{good:?}");
        }
        assert!(safe_name("notes.txt").is_ok() && safe_name("a b.txt").is_err() && safe_name("../a.txt").is_err() && safe_name("x.exe").is_err() && safe_name(".h.txt").is_err());
    }

    #[test]
    fn train_form_becomes_a_valid_config() {
        let root = Path::new("/nonexistent");
        let v = json!({"model": "models/m", "data": "data/a.txt", "steps": "7", "lr": 0.001, "mode": "full", "seq_len": 64});
        let (text, out) = train_toml(root, &v).unwrap();
        assert_eq!(out, "runs/ui");
        let c = parse_config(&text).unwrap();
        assert_eq!((c.train.steps, c.data.seq_len, c.train.mode.as_str()), (7, 64, "full"));
        assert_eq!(c.model.tokenizer.as_deref(), Some("bytes")); // no tokenizer.json next to the model
        assert!(train_toml(root, &json!({"model": "../m", "data": "d"})).is_err());
        assert!(train_toml(root, &json!({"model": "m", "data": "d", "mode": "weird"})).is_err());
        assert!(train_toml(root, &json!({"model": "m", "data": "d", "steps": "abc"})).is_err());
        assert!(train_toml(root, &json!({"data": "d"})).is_err());
    }

    #[test]
    fn commands_are_built_from_validated_fields_only() {
        let root = Path::new("/nonexistent");
        let (a, _) = command_for(root, "generate", &json!({"model": "m", "prompt": "--evil", "adapter": "a.safetensors"})).unwrap();
        assert_eq!(&a[..5], ["generate", "--model", "m", "--prompt", "--evil"]); // prompt stays one argument
        assert!(a.contains(&"--adapter".to_string()));
        assert!(command_for(root, "generate", &json!({"model": "m", "prompt": "x", "adapter": "/abs"})).is_err());
        assert!(command_for(root, "rm-rf", &json!({})).is_err());
        assert!(command_for(root, "merge", &json!({"model": "m", "adapter": "a", "out": ".."})).is_err());
    }

    #[test]
    fn scan_finds_models_adapters_and_data() {
        let d = tempfile::tempdir().unwrap();
        let r = d.path();
        std::fs::create_dir_all(r.join("models/m")).unwrap();
        std::fs::write(r.join("models/m/config.json"), "{}").unwrap();
        std::fs::create_dir_all(r.join("runs/x")).unwrap();
        std::fs::write(r.join("runs/x/adapter.safetensors"), "").unwrap();
        std::fs::write(r.join("a.jsonl"), "").unwrap();
        std::fs::create_dir_all(r.join("target/models")).unwrap();
        std::fs::write(r.join("target/models/config.json"), "{}").unwrap();
        let v = scan(r);
        assert_eq!(v["models"], json!(["models/m"]));
        assert_eq!(v["adapters"], json!(["runs/x/adapter.safetensors"]));
        assert_eq!(v["data"], json!(["a.jsonl"]));
    }
}
