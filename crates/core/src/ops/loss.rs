//! Cross-entropy over a vocabulary, fused with its gradient.

/// Targets equal to this value are skipped (prompt tokens, padding).
pub const IGNORE_INDEX: i64 = -100;

/// Mean cross-entropy over non-ignored rows. Writes `dlogits` (**overwrites**,
/// ignored rows get zeros) and returns the mean loss. If every row is ignored
/// the loss and gradient are zero.
pub fn cross_entropy_fwd_bwd(
    logits: &[f32],
    targets: &[i64],
    dlogits: &mut [f32],
    rows: usize,
    vocab: usize,
) -> f32 {
    assert_eq!(logits.len(), rows * vocab);
    assert_eq!(dlogits.len(), rows * vocab);
    assert_eq!(targets.len(), rows);

    let n_valid = targets.iter().filter(|&&t| t != IGNORE_INDEX).count();
    if n_valid == 0 {
        dlogits.fill(0.0);
        return 0.0;
    }
    let inv_n = 1.0 / n_valid as f32;
    let mut total = 0.0f64;

    for r in 0..rows {
        let lr = &logits[r * vocab..(r + 1) * vocab];
        let dr = &mut dlogits[r * vocab..(r + 1) * vocab];
        let t = targets[r];
        if t == IGNORE_INDEX {
            dr.fill(0.0);
            continue;
        }
        assert!(t >= 0 && (t as usize) < vocab, "target {t} out of range for vocab {vocab}");
        let t = t as usize;

        let max = lr.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for (d, &l) in dr.iter_mut().zip(lr) {
            let e = (l - max).exp();
            *d = e;
            sum += e;
        }
        total += (sum.ln() + max - lr[t]) as f64;

        let inv_sum = 1.0 / sum;
        for d in dr.iter_mut() {
            *d *= inv_sum * inv_n;
        }
        dr[t] -= inv_n;
    }
    (total / n_valid as f64) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gradcheck::*;
    use crate::rng::Rng;

    fn ref_loss(logits: &[f64], targets: &[i64], rows: usize, vocab: usize) -> f64 {
        let mut total = 0.0;
        let mut n = 0;
        for r in 0..rows {
            if targets[r] == IGNORE_INDEX {
                continue;
            }
            let lr = &logits[r * vocab..(r + 1) * vocab];
            let max = lr.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let lse = lr.iter().map(|l| (l - max).exp()).sum::<f64>().ln() + max;
            total += lse - lr[targets[r] as usize];
            n += 1;
        }
        total / n as f64
    }

    fn logits(rows: usize, vocab: usize, seed: u64) -> Vec<f32> {
        let mut r = Rng::new(seed);
        (0..rows * vocab).map(|_| r.normal() * 3.0).collect()
    }

    #[test]
    fn uniform_logits_give_ln_vocab() {
        let (rows, vocab) = (4, 50);
        let l = vec![0.7; rows * vocab];
        let t = vec![3, 10, 0, 49];
        let mut d = vec![0.0; rows * vocab];
        let loss = cross_entropy_fwd_bwd(&l, &t, &mut d, rows, vocab);
        assert!((loss - (vocab as f32).ln()).abs() < 1e-5);
    }

    #[test]
    fn loss_matches_f64_reference() {
        let (rows, vocab) = (6, 23);
        let l = logits(rows, vocab, 1);
        let t = vec![0, 5, IGNORE_INDEX, 22, 7, 7];
        let mut d = vec![0.0; rows * vocab];
        let loss = cross_entropy_fwd_bwd(&l, &t, &mut d, rows, vocab);
        let want = ref_loss(&to_f64(&l), &t, rows, vocab);
        assert!((loss as f64 - want).abs() < 1e-5, "{loss} vs {want}");
    }

    #[test]
    fn gradient_matches_finite_differences() {
        let (rows, vocab) = (5, 11);
        let l = logits(rows, vocab, 2);
        let t = vec![1, IGNORE_INDEX, 4, 10, 0];
        let mut d = vec![0.0; rows * vocab];
        cross_entropy_fwd_bwd(&l, &t, &mut d, rows, vocab);
        let n = numeric_grad(|ls| ref_loss(ls, &t, rows, vocab), &to_f64(&l), 1e-6);
        assert_grad_close("cross-entropy dlogits", &d, &n, 5e-4);
        // Ignored row gets exactly zero gradient.
        assert!(d[vocab..2 * vocab].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn gradient_rows_sum_to_zero() {
        let (rows, vocab) = (3, 17);
        let l = logits(rows, vocab, 3);
        let t = vec![2, 3, 4];
        let mut d = vec![0.0; rows * vocab];
        cross_entropy_fwd_bwd(&l, &t, &mut d, rows, vocab);
        for r in 0..rows {
            let s: f32 = d[r * vocab..(r + 1) * vocab].iter().sum();
            assert!(s.abs() < 1e-6, "row {r} sums to {s}");
        }
    }

    #[test]
    fn stable_for_huge_logits() {
        let (rows, vocab) = (2, 8);
        let mut l = logits(rows, vocab, 4);
        for v in l.iter_mut() {
            *v += 1000.0;
        }
        let t = vec![1, 2];
        let mut d = vec![0.0; rows * vocab];
        let loss = cross_entropy_fwd_bwd(&l, &t, &mut d, rows, vocab);
        assert!(loss.is_finite());
        assert!(d.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn all_ignored_is_zero() {
        let l = logits(2, 5, 5);
        let mut d = vec![1.0; 10];
        let loss = cross_entropy_fwd_bwd(&l, &[IGNORE_INDEX, IGNORE_INDEX], &mut d, 2, 5);
        assert_eq!(loss, 0.0);
        assert!(d.iter().all(|&v| v == 0.0));
    }
}
