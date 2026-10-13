//! `cpuforge parity --model DIR --ref DIR [--tol 5e-3]`
//!
//! Compares our logits with Hugging Face transformers on the same tokens.
//! Produce the reference with `python scripts/parity.py MODEL_DIR REF_DIR`,
//! which writes `tokens.txt` and `logits.bin` (float32, [tokens, vocab]).

use std::path::Path;

use cf_io::load_model;

use crate::args::Args;

#[derive(Debug)]
pub struct Report {
    pub max_abs_diff: f32,
    pub mean_abs_diff: f32,
    pub argmax_matches: usize,
    pub positions: usize,
    /// Per-position max abs diff.
    pub per_position: Vec<f32>,
}

impl Report {
    pub fn passed(&self, tol: f32) -> bool {
        self.max_abs_diff <= tol && self.argmax_matches == self.positions
    }
}

/// Compare two `[positions, vocab]` logit arrays.
pub fn compare(ours: &[f32], theirs: &[f32], vocab: usize) -> Result<Report, String> {
    if ours.len() != theirs.len() || vocab == 0 || !ours.len().is_multiple_of(vocab) {
        return Err(format!("size mismatch: {} vs {} values (vocab {vocab})", ours.len(), theirs.len()));
    }
    let positions = ours.len() / vocab;
    let argmax = |row: &[f32]| row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i).unwrap_or(0);
    let (mut max_abs, mut sum_abs, mut matches) = (0.0f32, 0.0f64, 0usize);
    let mut per_position = Vec::with_capacity(positions);
    for p in 0..positions {
        let (a, b) = (&ours[p * vocab..(p + 1) * vocab], &theirs[p * vocab..(p + 1) * vocab]);
        let mut row_max = 0.0f32;
        for (x, y) in a.iter().zip(b) {
            let d = (x - y).abs();
            row_max = row_max.max(d);
            sum_abs += d as f64;
        }
        max_abs = max_abs.max(row_max);
        per_position.push(row_max);
        if argmax(a) == argmax(b) {
            matches += 1;
        }
    }
    Ok(Report {
        max_abs_diff: max_abs,
        mean_abs_diff: (sum_abs / ours.len() as f64) as f32,
        argmax_matches: matches,
        positions,
        per_position,
    })
}

pub fn read_f32_le(path: &Path) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    if !bytes.len().is_multiple_of(4) {
        return Err(format!("{} is not a float32 array", path.display()));
    }
    Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

pub fn run(args: &Args) -> Result<(), String> {
    let model_dir = Path::new(args.require("model")?);
    let ref_dir = Path::new(args.require("ref")?);
    let tol: f32 = args.parsed("tol", 5e-3)?;

    let tokens: Vec<u32> = std::fs::read_to_string(ref_dir.join("tokens.txt"))
        .map_err(|e| format!("cannot read tokens.txt in {}: {e}", ref_dir.display()))?
        .split_whitespace()
        .map(|t| t.parse::<u32>().map_err(|_| format!("bad token id '{t}'")))
        .collect::<Result<_, _>>()?;
    let reference = read_f32_le(&ref_dir.join("logits.bin"))?;

    let model = load_model(model_dir, tokens.len())?;
    let ours = model.logits(&tokens);
    let r = compare(&ours, &reference, model.cfg.vocab_size)?;

    println!("{} tokens, vocab {}", r.positions, model.cfg.vocab_size);
    println!("max abs logit difference : {:.3e}", r.max_abs_diff);
    println!("mean abs logit difference: {:.3e}", r.mean_abs_diff);
    println!("argmax agreement         : {}/{}", r.argmax_matches, r.positions);
    if r.passed(tol) {
        println!("PASS (tolerance {tol:.0e})");
        Ok(())
    } else {
        Err(format!("FAIL: differences exceed {tol:.0e} or predictions disagree; first positions: {:?}", &r.per_position[..r.per_position.len().min(8)]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logits(positions: usize, vocab: usize) -> Vec<f32> {
        (0..positions * vocab).map(|i| ((i * 7919) % 1000) as f32 / 100.0 - 5.0).collect()
    }

    #[test]
    fn identical_logits_pass() {
        let a = logits(4, 50);
        let r = compare(&a, &a, 50).unwrap();
        assert_eq!(r.max_abs_diff, 0.0);
        assert_eq!(r.argmax_matches, 4);
        assert!(r.passed(1e-6));
    }

    #[test]
    fn small_noise_passes_but_large_noise_or_flipped_argmax_fails() {
        let a = logits(4, 50);
        let noisy: Vec<f32> = a.iter().enumerate().map(|(i, v)| v + if i % 2 == 0 { 1e-4 } else { -1e-4 }).collect();
        let r = compare(&a, &noisy, 50).unwrap();
        assert!(r.passed(1e-3), "{r:?}");
        assert!(!r.passed(1e-5));

        // Flip the winner in position 2 while keeping the difference "small" in other places.
        let mut flipped = a.clone();
        let row = &mut flipped[2 * 50..3 * 50];
        let best = row.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).unwrap().0;
        row[best] -= 20.0;
        let r = compare(&a, &flipped, 50).unwrap();
        assert!(r.argmax_matches < r.positions);
        assert!(!r.passed(1e9));
    }

    #[test]
    fn shape_mismatch_is_an_error() {
        assert!(compare(&[0.0; 10], &[0.0; 12], 5).is_err());
        assert!(compare(&[0.0; 10], &[0.0; 10], 3).is_err());
    }

    #[test]
    fn reads_little_endian_floats() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("l.bin");
        let vals = [1.5f32, -2.0, 0.25];
        std::fs::write(&p, vals.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
        assert_eq!(read_f32_le(&p).unwrap(), vals);
        std::fs::write(&p, [1u8, 2, 3]).unwrap();
        assert!(read_f32_le(&p).is_err());
    }
}
