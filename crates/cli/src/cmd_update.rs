//! `cpuforge update [--check] [--repo OWNER/REPO]`
//!
//! Self-update from the latest GitHub release. Release tags end in the build
//! number (`build-42`); the running binary knows its own build number from
//! `CPUFORGE_BUILD`, set by the release workflow. The asset's `.sha256` file
//! is checked before anything on disk is touched, and the swap is atomic.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::args::Args;

pub const DEFAULT_REPO: &str = "Sm0r3s-source/cpuforge";
const DEFAULT_API: &str = "https://api.github.com";

/// Build number of this binary (0 for local development builds).
pub fn build() -> u64 {
    option_env!("CPUFORGE_BUILD").and_then(|s| s.parse().ok()).unwrap_or(0)
}

pub fn version() -> String {
    match option_env!("CPUFORGE_BUILD") {
        Some(b) if !b.is_empty() => format!("0.{b}"),
        _ => format!("{}-dev", env!("CARGO_PKG_VERSION")),
    }
}

/// Release asset name for this platform.
pub fn asset_name() -> Option<&'static str> {
    if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        Some("cpuforge-linux-x64")
    } else if cfg!(all(target_os = "windows", target_arch = "x86_64")) {
        Some("cpuforge-windows-x64.exe")
    } else if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        Some("cpuforge-macos-arm64")
    } else {
        None
    }
}

#[derive(Debug, PartialEq)]
pub struct Release {
    pub build: u64,
    pub tag: String,
    pub asset_url: String,
    pub sha_url: String,
}

/// `build-42` / `v42` / `0.42` -> 42.
pub fn parse_build(tag: &str) -> Option<u64> {
    let digits: String = tag.chars().rev().take_while(char::is_ascii_digit).collect::<Vec<_>>().into_iter().rev().collect();
    digits.parse().ok()
}

pub fn parse_release(json: &str, asset: &str) -> Result<Release, String> {
    let v: Value = serde_json::from_str(json).map_err(|e| format!("bad release JSON: {e}"))?;
    let tag = v["tag_name"].as_str().ok_or("release has no tag_name")?.to_string();
    let build = parse_build(&tag).ok_or_else(|| format!("cannot read a build number from tag '{tag}'"))?;
    let url_of = |name: &str| {
        v["assets"].as_array().and_then(|a| a.iter().find(|x| x["name"] == name)).and_then(|x| x["url"].as_str()).map(String::from)
    };
    let asset_url = url_of(asset).ok_or_else(|| format!("release {tag} has no asset '{asset}'"))?;
    let sha_url = url_of(&format!("{asset}.sha256")).ok_or_else(|| format!("release {tag} has no '{asset}.sha256' (refusing to install unchecked)"))?;
    Ok(Release { build, tag, asset_url, sha_url })
}

pub struct Source {
    api: String,
    repo: String,
    token: Option<String>,
}

impl Source {
    pub fn new(repo: Option<&str>, api: Option<&str>) -> Source {
        Source {
            api: api.map(String::from).or_else(|| std::env::var("CPUFORGE_API").ok()).unwrap_or_else(|| DEFAULT_API.into()),
            repo: repo.map(String::from).or_else(|| std::env::var("CPUFORGE_REPO").ok()).unwrap_or_else(|| DEFAULT_REPO.into()),
            token: std::env::var("CPUFORGE_TOKEN").ok().filter(|t| !t.is_empty()),
        }
    }

    fn get(&self, url: &str, accept: &str) -> Result<Vec<u8>, String> {
        let agent: ureq::Agent = ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(600))).build().into();
        let mut req = agent.get(url).header("User-Agent", "cpuforge-updater").header("Accept", accept);
        if let Some(t) = &self.token {
            req = req.header("Authorization", format!("Bearer {t}"));
        }
        let mut resp = req.call().map_err(|e| format!("cannot reach {url}: {e}"))?;
        resp.body_mut().with_config().limit(400 << 20).read_to_vec().map_err(|e| format!("download failed: {e}"))
    }

    pub fn latest(&self) -> Result<Release, String> {
        let asset = asset_name().ok_or("no prebuilt release for this platform")?;
        let url = format!("{}/repos/{}/releases/latest", self.api, self.repo);
        let body = self.get(&url, "application/vnd.github+json")?;
        parse_release(&String::from_utf8_lossy(&body), asset)
    }

    /// Download, verify and install the release over `exe`.
    pub fn install(&self, rel: &Release, exe: &Path) -> Result<(), String> {
        let sha = self.get(&rel.sha_url, "application/octet-stream")?;
        let want = String::from_utf8_lossy(&sha).split_whitespace().next().unwrap_or("").to_ascii_lowercase();
        let bytes = self.get(&rel.asset_url, "application/octet-stream")?;
        install_bytes(&bytes, &want, exe)
    }
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

/// Check the hash, then swap the file in. Nothing is changed on a mismatch.
pub fn install_bytes(bytes: &[u8], want_sha: &str, exe: &Path) -> Result<(), String> {
    let got = sha256_hex(bytes);
    if got != want_sha {
        return Err(format!("checksum mismatch (expected {want_sha}, got {got}); not installing"));
    }
    let mut new = exe.as_os_str().to_owned();
    new.push(".new");
    let new = PathBuf::from(new);
    let write = || -> std::io::Result<()> {
        let mut f = std::fs::File::create(&new)?;
        f.write_all(bytes)?;
        f.sync_all()?;
        #[cfg(unix)]
        std::fs::set_permissions(&new, std::os::unix::fs::PermissionsExt::from_mode(0o755))?;
        Ok(())
    };
    write().map_err(|e| format!("cannot write {}: {e}", new.display()))?;
    swap(&new, exe).inspect_err(|_| {
        let _ = std::fs::remove_file(&new);
    })
}

#[cfg(unix)]
fn swap(new: &Path, exe: &Path) -> Result<(), String> {
    std::fs::rename(new, exe).map_err(|e| format!("cannot replace {}: {e}", exe.display()))
}

/// Windows cannot overwrite a running exe but can rename it out of the way.
#[cfg(not(unix))]
fn swap(new: &Path, exe: &Path) -> Result<(), String> {
    let mut old = exe.as_os_str().to_owned();
    old.push(".old");
    let old = PathBuf::from(old);
    let _ = std::fs::remove_file(&old);
    std::fs::rename(exe, &old).map_err(|e| format!("cannot move {} aside: {e}", exe.display()))?;
    std::fs::rename(new, exe).map_err(|e| {
        let _ = std::fs::rename(&old, exe);
        format!("cannot replace {}: {e}", exe.display())
    })
}

/// Remove the leftover from a previous Windows update.
pub fn cleanup_old(exe: &Path) {
    let mut old = exe.as_os_str().to_owned();
    old.push(".old");
    let _ = std::fs::remove_file(PathBuf::from(old));
}

/// Start the freshly installed binary in place of this process.
pub fn restart_self(exe: &Path, argv: &[String]) -> ! {
    #[cfg(unix)]
    {
        let err = std::os::unix::process::CommandExt::exec(std::process::Command::new(exe).args(argv));
        eprintln!("restart failed: {err}");
        std::process::exit(1);
    }
    #[cfg(not(unix))]
    {
        let _ = std::process::Command::new(exe).args(argv).spawn();
        std::process::exit(0);
    }
}

pub fn run(args: &Args) -> Result<(), String> {
    let src = Source::new(args.get("repo"), args.get("api"));
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let rel = src.latest()?;
    println!("installed: {} (build {})   latest: {} (build {})", version(), build(), rel.tag, rel.build);
    if rel.build <= build() {
        println!("already up to date");
        return Ok(());
    }
    if args.has("check") {
        println!("update available (run `cpuforge update` to install)");
        return Ok(());
    }
    src.install(&rel, &exe)?;
    println!("updated to {}; restart cpuforge to use it", rel.tag);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const REL: &str = r#"{"tag_name":"build-42","assets":[
        {"name":"cpuforge-linux-x64","url":"https://x/a1"},{"name":"cpuforge-linux-x64.sha256","url":"https://x/a2"},
        {"name":"cpuforge-windows-x64.exe","url":"https://x/b1"},{"name":"cpuforge-windows-x64.exe.sha256","url":"https://x/b2"}]}"#;

    #[test]
    fn build_numbers_are_read_from_tags() {
        assert_eq!(parse_build("build-42"), Some(42));
        assert_eq!(parse_build("v7"), Some(7));
        assert_eq!(parse_build("0.130"), Some(130));
        assert_eq!(parse_build("latest"), None);
    }

    #[test]
    fn picks_the_asset_and_its_checksum() {
        let r = parse_release(REL, "cpuforge-windows-x64.exe").unwrap();
        assert_eq!((r.build, r.asset_url.as_str(), r.sha_url.as_str()), (42, "https://x/b1", "https://x/b2"));
        assert!(parse_release(REL, "cpuforge-macos-arm64").is_err());
        assert!(parse_release(r#"{"tag_name":"build-1","assets":[{"name":"cpuforge-linux-x64","url":"u"}]}"#, "cpuforge-linux-x64").is_err()); // no checksum
        assert!(parse_release("{}", "x").is_err());
    }

    #[test]
    fn sha256_matches_a_known_value() {
        assert_eq!(sha256_hex(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }

    #[test]
    fn install_swaps_atomically_and_refuses_bad_checksums() {
        let d = tempfile::tempdir().unwrap();
        let exe = d.path().join("cpuforge");
        std::fs::write(&exe, b"old").unwrap();
        assert!(install_bytes(b"new", "00", &exe).is_err());
        assert_eq!(std::fs::read(&exe).unwrap(), b"old");
        assert!(!d.path().join("cpuforge.new").exists());
        install_bytes(b"new", &sha256_hex(b"new"), &exe).unwrap();
        assert_eq!(std::fs::read(&exe).unwrap(), b"new");
        #[cfg(unix)]
        assert_eq!(std::os::unix::fs::PermissionsExt::mode(&std::fs::metadata(&exe).unwrap().permissions()) & 0o111, 0o111);
    }
}
