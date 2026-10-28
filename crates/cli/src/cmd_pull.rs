//! `cpuforge pull --repo ORG/NAME [--out DIR] [--revision main]`
//!
//! Downloads a model from Hugging Face (weights in safetensors, config and tokenizer files), so a
//! headless box needs neither Python nor `huggingface-cli`. Gated models need `HF_TOKEN`.
//! Files that are already complete are skipped, so an interrupted pull simply continues.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::Value;

use crate::args::Args;

const DEFAULT_HF: &str = "https://huggingface.co";

/// Files worth downloading: weights as safetensors, config, tokenizer and template files.
pub fn wanted(name: &str) -> bool {
    let base = name.rsplit('/').next().unwrap_or(name);
    if name.contains('/') {
        return false; // sub-folders (onnx, original/, ...) are never needed
    }
    base.ends_with(".safetensors")
        || base == "model.safetensors.index.json"
        || matches!(
            base,
            "config.json" | "generation_config.json" | "tokenizer.json" | "tokenizer_config.json" | "special_tokens_map.json" | "vocab.json" | "merges.txt" | "tokenizer.model"
        )
        || base.starts_with("chat_template")
}

/// A file name from the server must be a plain name: no separators, no `..`.
fn safe_file_name(name: &str) -> bool {
    !name.is_empty() && !name.contains(['/', '\\', '\0']) && name != "." && name != ".." && !name.starts_with("..")
}

pub fn default_cache_dir(repo: &str) -> PathBuf {
    let home = std::env::var_os("CPUFORGE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")).map(|h| PathBuf::from(h).join(".cache").join("cpuforge")))
        .unwrap_or_else(|| PathBuf::from(".cpuforge"));
    home.join("models").join(repo.replace('/', "__"))
}

pub fn hf_token() -> Option<String> {
    ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"].iter().find_map(|k| std::env::var(k).ok()).filter(|t| !t.is_empty())
}

fn agent() -> ureq::Agent {
    ureq::Agent::config_builder().timeout_connect(Some(Duration::from_secs(30))).timeout_recv_body(Some(Duration::from_secs(120))).build().into()
}

fn describe(e: ureq::Error, what: &str, repo: &str) -> String {
    match e {
        ureq::Error::StatusCode(401) | ureq::Error::StatusCode(403) => {
            format!("{what}: access denied (HTTP 401/403). {repo} is gated or private: accept its licence on huggingface.co and set HF_TOKEN")
        }
        ureq::Error::StatusCode(404) => format!("{what}: not found (HTTP 404). Check the repository name '{repo}'"),
        other => format!("cannot reach Hugging Face for {what}: {other}"),
    }
}

/// Download the needed files of `repo` into `dest`. Returns the number of files fetched or skipped.
pub fn pull_model(repo: &str, revision: &str, dest: &Path, say: &mut dyn FnMut(&str)) -> Result<usize, String> {
    let base = std::env::var("CPUFORGE_HF").unwrap_or_else(|_| DEFAULT_HF.into());
    let token = hf_token();
    let agent = agent();
    let auth = |req: ureq::RequestBuilder<ureq::typestate::WithoutBody>| match &token {
        Some(t) => req.header("Authorization", format!("Bearer {t}")),
        None => req,
    };

    let info_url = format!("{base}/api/models/{repo}/revision/{revision}?blobs=true");
    let mut resp = auth(agent.get(&info_url).header("User-Agent", "cpuforge")).call().map_err(|e| describe(e, "model lookup", repo))?;
    let info: Value = serde_json::from_str(&resp.body_mut().read_to_string().map_err(|e| e.to_string())?).map_err(|e| format!("bad answer from Hugging Face: {e}"))?;
    let mut files: Vec<(String, Option<u64>)> = Vec::new();
    for s in info["siblings"].as_array().ok_or("Hugging Face returned no file list")? {
        let Some(name) = s["rfilename"].as_str() else { continue };
        if name.contains('/') || !wanted(name) {
            continue;
        }
        if !safe_file_name(name) {
            return Err(format!("refusing suspicious file name '{name}' from the server"));
        }
        files.push((name.to_string(), s["size"].as_u64()));
    }
    if !files.iter().any(|(n, _)| n == "config.json") {
        return Err(format!("{repo} has no config.json: not a transformers model"));
    }
    if !files.iter().any(|(n, _)| n.ends_with(".safetensors")) {
        return Err(format!("{repo} has no .safetensors weights (only safetensors models are supported)"));
    }
    std::fs::create_dir_all(dest).map_err(|e| format!("cannot create {}: {e}", dest.display()))?;

    let total = files.len();
    for (i, (name, size)) in files.iter().enumerate() {
        let target = dest.join(name);
        if let (Some(want), Ok(meta)) = (size, std::fs::metadata(&target)) {
            if meta.len() == *want {
                say(&format!("[{}/{total}] {name}: already complete", i + 1));
                continue;
            }
        }
        say(&format!("[{}/{total}] downloading {name}{}", i + 1, size.map_or(String::new(), |s| format!(" ({:.1} MB)", s as f64 / 1e6))));
        let url = format!("{base}/{repo}/resolve/{revision}/{name}");
        let mut r = auth(agent.get(&url).header("User-Agent", "cpuforge")).call().map_err(|e| describe(e, name, repo))?;
        let part = dest.join(format!("{name}.part"));
        let mut out = std::fs::File::create(&part).map_err(|e| format!("cannot write {}: {e}", part.display()))?;
        let mut reader = r.body_mut().as_reader();
        let mut buf = vec![0u8; 1 << 20];
        let mut got = 0u64;
        loop {
            let n = reader.read(&mut buf).map_err(|e| format!("download of {name} failed: {e}"))?;
            if n == 0 {
                break;
            }
            out.write_all(&buf[..n]).map_err(|e| format!("cannot write {}: {e}", part.display()))?;
            got += n as u64;
        }
        out.sync_all().map_err(|e| e.to_string())?;
        if let Some(want) = size {
            if got != *want {
                let _ = std::fs::remove_file(&part);
                return Err(format!("download of {name} is incomplete ({got} of {want} bytes)"));
            }
        }
        std::fs::rename(&part, &target).map_err(|e| format!("cannot move {} into place: {e}", part.display()))?;
    }
    Ok(total)
}

pub fn run(args: &Args) -> Result<(), String> {
    let repo = args.require("repo")?;
    let dest = args.get("out").map(PathBuf::from).unwrap_or_else(|| default_cache_dir(repo));
    let n = pull_model(repo, args.get("revision").unwrap_or("main"), &dest, &mut |l| println!("{l}"))?;
    println!("{repo}: {n} files in {}", dest.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_needed_files_are_wanted() {
        for ok in ["model.safetensors", "model-00001-of-00002.safetensors", "model.safetensors.index.json", "config.json", "tokenizer.json", "chat_template.jinja", "merges.txt"] {
            assert!(wanted(ok), "{ok}");
        }
        for no in ["pytorch_model.bin", "model.gguf", "README.md", "onnx/model.onnx", "original/consolidated.00.pth", ".gitattributes", "model.onnx"] {
            assert!(!wanted(no), "{no}");
        }
    }

    #[test]
    fn file_names_cannot_escape_the_folder() {
        for bad in ["", "..", "../x", "a/b", "a\\b", "..config.json"] {
            assert!(!safe_file_name(bad), "{bad:?}");
        }
        assert!(safe_file_name("model-00001-of-00002.safetensors"));
    }

    #[test]
    fn cache_dir_names_are_flat() {
        let d = default_cache_dir("Qwen/Qwen3-0.6B");
        assert!(d.ends_with("models/Qwen__Qwen3-0.6B") || d.ends_with("models\\Qwen__Qwen3-0.6B"), "{d:?}");
    }
}
