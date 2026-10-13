//! Rotary position embeddings, Llama/HF "rotate half" layout: dimension `i` is
//! paired with `i + head_dim/2` and rotated by `pos * theta^(-2i/head_dim)`.
//!
//! A rotation's backward is the inverse rotation, so one routine serves both.

pub struct RopeCache {
    head_dim: usize,
    half: usize,
    max_pos: usize,
    cos: Vec<f32>, // [max_pos, half]
    sin: Vec<f32>,
}

impl RopeCache {
    pub fn new(head_dim: usize, max_pos: usize, theta: f64) -> Self {
        assert!(head_dim.is_multiple_of(2), "head_dim must be even");
        let half = head_dim / 2;
        let mut cos = Vec::with_capacity(max_pos * half);
        let mut sin = Vec::with_capacity(max_pos * half);
        for pos in 0..max_pos {
            for i in 0..half {
                let inv_freq = theta.powf(-(2.0 * i as f64) / head_dim as f64);
                let angle = pos as f64 * inv_freq;
                cos.push(angle.cos() as f32);
                sin.push(angle.sin() as f32);
            }
        }
        Self { head_dim, half, max_pos, cos, sin }
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Rotate `x` in place. `x` is `[seq, heads, head_dim]`; token `t` uses
    /// position `pos0 + t`. `inverse = true` applies the transpose (backward).
    pub fn apply(&self, x: &mut [f32], seq: usize, heads: usize, pos0: usize, inverse: bool) {
        assert_eq!(x.len(), seq * heads * self.head_dim);
        assert!(pos0 + seq <= self.max_pos, "position {} exceeds cache size {}", pos0 + seq, self.max_pos);
        let sign = if inverse { -1.0f32 } else { 1.0 };
        for t in 0..seq {
            let cs = &self.cos[(pos0 + t) * self.half..(pos0 + t + 1) * self.half];
            let sn = &self.sin[(pos0 + t) * self.half..(pos0 + t + 1) * self.half];
            for h in 0..heads {
                let base = (t * heads + h) * self.head_dim;
                let (lo, hi) = x[base..base + self.head_dim].split_at_mut(self.half);
                for i in 0..self.half {
                    let (c, s) = (cs[i], sign * sn[i]);
                    let (x1, x2) = (lo[i], hi[i]);
                    lo[i] = x1 * c - x2 * s;
                    hi[i] = x2 * c + x1 * s;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gradcheck::*;
    use crate::rng::Rng;

    fn rope_ref(x: &[f64], seq: usize, heads: usize, hd: usize, pos0: usize, theta: f64) -> Vec<f64> {
        let half = hd / 2;
        let mut y = x.to_vec();
        for t in 0..seq {
            for h in 0..heads {
                let base = (t * heads + h) * hd;
                for i in 0..half {
                    let ang = (pos0 + t) as f64 * theta.powf(-(2.0 * i as f64) / hd as f64);
                    let (c, s) = (ang.cos(), ang.sin());
                    let (x1, x2) = (x[base + i], x[base + half + i]);
                    y[base + i] = x1 * c - x2 * s;
                    y[base + half + i] = x2 * c + x1 * s;
                }
            }
        }
        y
    }

    fn rand(n: usize, seed: u64) -> Vec<f32> {
        let mut r = Rng::new(seed);
        (0..n).map(|_| r.normal()).collect()
    }

    #[test]
    fn forward_matches_f64_reference() {
        let (seq, heads, hd, theta) = (6, 3, 16, 10000.0);
        let cache = RopeCache::new(hd, 32, theta);
        let x = rand(seq * heads * hd, 1);
        let mut y = x.clone();
        cache.apply(&mut y, seq, heads, 4, false);
        let want = rope_ref(&to_f64(&x), seq, heads, hd, 4, theta);
        assert_close("rope fwd", &y, &want, 1e-5);
    }

    #[test]
    fn position_zero_is_identity() {
        let cache = RopeCache::new(8, 4, 10000.0);
        let x = rand(2 * 8, 2);
        let mut y = x.clone();
        cache.apply(&mut y, 1, 2, 0, false);
        assert_eq!(x, y);
    }

    #[test]
    fn rotation_preserves_norm_and_inverse_roundtrips() {
        let (seq, heads, hd) = (10, 2, 32);
        let cache = RopeCache::new(hd, 64, 500000.0);
        let x = rand(seq * heads * hd, 3);
        let mut y = x.clone();
        cache.apply(&mut y, seq, heads, 7, false);
        let n0: f32 = x.iter().map(|v| v * v).sum();
        let n1: f32 = y.iter().map(|v| v * v).sum();
        assert!((n0 - n1).abs() / n0 < 1e-5);
        cache.apply(&mut y, seq, heads, 7, true);
        for (a, b) in x.iter().zip(&y) {
            assert!((a - b).abs() < 1e-4);
        }
    }

    #[test]
    fn backward_matches_finite_differences() {
        let (seq, heads, hd, theta) = (4, 2, 8, 10000.0);
        let cache = RopeCache::new(hd, 16, theta);
        let x = rand(seq * heads * hd, 4);
        let g = rand(seq * heads * hd, 5);

        // Analytic: dx = R^T g
        let mut dx = g.clone();
        cache.apply(&mut dx, seq, heads, 2, true);

        let g64 = to_f64(&g);
        let nx = numeric_grad(
            |xs| rope_ref(xs, seq, heads, hd, 2, theta).iter().zip(&g64).map(|(a, b)| a * b).sum(),
            &to_f64(&x),
            1e-6,
        );
        assert_grad_close("rope dx", &dx, &nx, 5e-4);
    }
}
