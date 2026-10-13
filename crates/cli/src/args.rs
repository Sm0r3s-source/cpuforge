//! Tiny flag parser: `cpuforge <command> --key value --switch`.

use std::collections::HashMap;
use std::str::FromStr;

pub struct Args {
    pub command: String,
    flags: HashMap<String, Option<String>>,
}

impl Args {
    pub fn parse<I: IntoIterator<Item = String>>(argv: I) -> Result<Args, String> {
        let mut it = argv.into_iter().peekable();
        let command = it.next().unwrap_or_default();
        let mut flags = HashMap::new();
        while let Some(a) = it.next() {
            let Some(key) = a.strip_prefix("--") else {
                return Err(format!("unexpected argument '{a}' (flags look like --name value)"));
            };
            let value = match it.peek() {
                Some(next) if !next.starts_with("--") => it.next(),
                _ => None,
            };
            flags.insert(key.to_string(), value);
        }
        Ok(Args { command, flags })
    }

    pub fn has(&self, key: &str) -> bool {
        self.flags.contains_key(key)
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.flags.get(key).and_then(|v| v.as_deref())
    }

    pub fn require(&self, key: &str) -> Result<&str, String> {
        self.get(key).ok_or_else(|| format!("missing required --{key}"))
    }

    pub fn parsed<T: FromStr>(&self, key: &str, default: T) -> Result<T, String> {
        match self.get(key) {
            None => Ok(default),
            Some(v) => v.parse().map_err(|_| format!("--{key}: cannot parse '{v}'")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(s: &str) -> Result<Args, String> {
        Args::parse(s.split_whitespace().map(String::from))
    }

    #[test]
    fn values_and_switches() {
        let a = parse("train --config run.toml --resume --steps 5").unwrap();
        assert_eq!(a.command, "train");
        assert_eq!(a.get("config"), Some("run.toml"));
        assert!(a.has("resume") && a.get("resume").is_none());
        assert_eq!(a.parsed("steps", 1usize).unwrap(), 5);
        assert_eq!(a.parsed("missing", 7usize).unwrap(), 7);
        assert!(a.require("nope").is_err());
    }

    #[test]
    fn bad_input_is_reported() {
        assert!(parse("train stray").is_err());
        assert!(parse("train --steps x").unwrap().parsed("steps", 1usize).is_err());
        assert_eq!(parse("").unwrap().command, "");
    }
}
