//! Small shared helpers for the commands.

use std::path::Path;

use cf_train::{ByteTokenizer, HfTokenizer, Tokenizer};

/// Peak resident memory of this process in MB (Linux only).
pub fn peak_rss_mb() -> Option<f64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = s.lines().find(|l| l.starts_with("VmHWM:"))?;
    let kb: f64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb / 1024.0)
}

/// Attach an adapter that is either a cpuforge `.safetensors` file or a PEFT-style folder (as Brewery uses).
pub fn load_any_adapter(model: &mut cf_model::Model, path: &Path) -> Result<(), String> {
    if path.is_dir() { cf_io::load_peft_adapter(model, path) } else { cf_io::load_adapter(model, path) }
}

/// `"bytes"` gives the byte-level tokenizer; anything else is a tokenizer.json path.
pub fn load_tokenizer(spec: &str) -> Result<Box<dyn Tokenizer>, String> {
    if spec == "bytes" {
        Ok(Box::new(ByteTokenizer))
    } else {
        Ok(Box::new(HfTokenizer::from_file(Path::new(spec))?))
    }
}

/// Pick a token from logits: greedy when `temp <= 0`, otherwise sample from the
/// `top_k` most likely tokens at the given temperature.
pub fn sample(logits: &[f32], temp: f32, top_k: usize, rng: &mut cf_core::rng::Rng) -> u32 {
    if temp <= 0.0 {
        return logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i as u32).unwrap_or(0);
    }
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    let k = if top_k == 0 { logits.len() } else { top_k.min(logits.len()) };
    idx.select_nth_unstable_by(k - 1, |&a, &b| logits[b].total_cmp(&logits[a]));
    idx.truncate(k);
    let max = idx.iter().map(|&i| logits[i]).fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<f32> = idx.iter().map(|&i| ((logits[i] - max) / temp).exp()).collect();
    let total: f32 = weights.iter().sum();
    let mut r = rng.uniform() * total;
    for (&i, &w) in idx.iter().zip(&weights) {
        if r < w {
            return i as u32;
        }
        r -= w;
    }
    idx[idx.len() - 1] as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use cf_core::rng::Rng;

    #[test]
    fn greedy_picks_the_argmax() {
        assert_eq!(sample(&[0.1, 2.0, -1.0, 1.9], 0.0, 0, &mut Rng::new(1)), 1);
    }

    #[test]
    fn top_k_never_leaves_the_top_set() {
        let logits = [5.0, 4.0, 3.0, -10.0, -10.0, -10.0];
        let mut rng = Rng::new(2);
        for _ in 0..200 {
            assert!(sample(&logits, 1.0, 3, &mut rng) < 3);
        }
    }

    #[test]
    fn low_temperature_approaches_greedy_and_high_spreads_out() {
        let logits = [3.0, 2.0, 1.0, 0.0];
        let mut rng = Rng::new(3);
        let cold = (0..100).filter(|_| sample(&logits, 0.05, 0, &mut rng) == 0).count();
        assert!(cold > 95, "{cold}");
        let hot: std::collections::HashSet<u32> = (0..200).map(|_| sample(&logits, 5.0, 0, &mut rng)).collect();
        assert!(hot.len() >= 3);
    }
}
