//! AdamW (decoupled weight decay) and the learning-rate schedule.

use std::collections::HashMap;

use cf_model::{visit_params, Grads, Model};

/// One AdamW update of a single tensor. `t` is the 1-based step count.
#[allow(clippy::too_many_arguments)]
pub fn adamw_update(
    p: &mut [f32],
    g: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    t: u64,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
) {
    let bc1 = 1.0 - beta1.powi(t as i32);
    let bc2 = 1.0 - beta2.powi(t as i32);
    for i in 0..p.len() {
        m[i] = beta1 * m[i] + (1.0 - beta1) * g[i];
        v[i] = beta2 * v[i] + (1.0 - beta2) * g[i] * g[i];
        let mhat = m[i] / bc1;
        let vhat = v[i] / bc2;
        p[i] -= lr * (mhat / (vhat.sqrt() + eps) + weight_decay * p[i]);
    }
}

pub struct AdamW {
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    /// Number of optimizer steps taken so far.
    pub step: u64,
    /// First and second moments, keyed by parameter name.
    pub state: HashMap<String, (Vec<f32>, Vec<f32>)>,
}

impl AdamW {
    pub fn new(beta1: f32, beta2: f32, eps: f32, weight_decay: f32) -> Self {
        Self { beta1, beta2, eps, weight_decay, step: 0, state: HashMap::new() }
    }

    /// Update every parameter that has a gradient. Norm weights get no weight decay.
    pub fn step(&mut self, model: &mut Model, grads: &Grads, lr: f32) {
        self.step += 1;
        let (t, b1, b2, eps, wd) = (self.step, self.beta1, self.beta2, self.eps, self.weight_decay);
        let state = &mut self.state;
        visit_params(model, Some(grads), &mut |name, _shape, p, g| {
            if g.is_empty() {
                return;
            }
            let (m, v) = state.entry(name.to_string()).or_insert_with(|| (vec![0.0; g.len()], vec![0.0; g.len()]));
            let decay = if name.contains("norm") { 0.0 } else { wd };
            adamw_update(p, g, m, v, t, lr, b1, b2, eps, decay);
        });
    }
}

/// Linear warmup to `peak`, then cosine decay to `peak * min_ratio` at `total` steps.
/// `step` is 0-based: the learning rate to use for the step about to run.
pub fn lr_at(step: usize, total: usize, warmup: usize, peak: f32, min_ratio: f32) -> f32 {
    if warmup > 0 && step < warmup {
        return peak * (step + 1) as f32 / warmup as f32;
    }
    let decay_steps = total.saturating_sub(warmup).max(1);
    let progress = ((step - warmup.min(step)) as f32 / decay_steps as f32).min(1.0);
    let cosine = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
    peak * (min_ratio + (1.0 - min_ratio) * cosine)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_step_moves_by_about_lr_in_the_gradient_direction() {
        let mut p = [1.0f32, 1.0];
        let g = [0.5f32, -3.0];
        let (mut m, mut v) = ([0.0; 2], [0.0; 2]);
        adamw_update(&mut p, &g, &mut m, &mut v, 1, 0.1, 0.9, 0.999, 1e-8, 0.0);
        // At t=1 the bias-corrected update is lr * g/|g|, independent of gradient scale.
        assert!((p[0] - 0.9).abs() < 1e-5, "{}", p[0]);
        assert!((p[1] - 1.1).abs() < 1e-5, "{}", p[1]);
    }

    #[test]
    fn matches_a_hand_computed_second_step() {
        let (b1, b2, eps, lr) = (0.9f32, 0.999f32, 1e-8f32, 0.01f32);
        let mut p = [2.0f32];
        let (mut m, mut v) = ([0.0f32], [0.0f32]);
        adamw_update(&mut p, &[1.0], &mut m, &mut v, 1, lr, b1, b2, eps, 0.0);
        adamw_update(&mut p, &[0.5], &mut m, &mut v, 2, lr, b1, b2, eps, 0.0);
        // Reference in f64.
        let (mut p64, mut m64, mut v64) = (2.0f64, 0.0f64, 0.0f64);
        for (t, g) in [(1, 1.0f64), (2, 0.5)] {
            m64 = 0.9 * m64 + 0.1 * g;
            v64 = 0.999 * v64 + 0.001 * g * g;
            let (mh, vh) = (m64 / (1.0 - 0.9f64.powi(t)), v64 / (1.0 - 0.999f64.powi(t)));
            p64 -= 0.01 * mh / (vh.sqrt() + 1e-8);
        }
        assert!((p[0] as f64 - p64).abs() < 1e-6, "{} vs {p64}", p[0]);
    }

    #[test]
    fn weight_decay_is_decoupled_from_the_gradient() {
        let mut p = [10.0f32];
        let (mut m, mut v) = ([0.0], [0.0]);
        adamw_update(&mut p, &[0.0], &mut m, &mut v, 1, 0.1, 0.9, 0.999, 1e-8, 0.5);
        assert!((p[0] - 9.5).abs() < 1e-5, "{}", p[0]); // p -= lr * wd * p
    }

    #[test]
    fn schedule_warms_up_then_decays_to_floor() {
        let (total, warmup, peak) = (100, 10, 1e-3f32);
        assert!((lr_at(0, total, warmup, peak, 0.1) - peak / 10.0).abs() < 1e-9);
        assert!((lr_at(9, total, warmup, peak, 0.1) - peak).abs() < 1e-9);
        assert!((lr_at(10, total, warmup, peak, 0.1) - peak).abs() < 1e-9);
        let mid = lr_at(55, total, warmup, peak, 0.1);
        assert!(mid < peak && mid > 0.1 * peak);
        assert!((lr_at(100, total, warmup, peak, 0.1) - 0.1 * peak).abs() < 1e-9);
        assert!((lr_at(500, total, warmup, peak, 0.1) - 0.1 * peak).abs() < 1e-9);
        // Monotone non-increasing after warmup.
        let mut last = f32::MAX;
        for s in 10..=100 {
            let lr = lr_at(s, total, warmup, peak, 0.1);
            assert!(lr <= last + 1e-12);
            last = lr;
        }
    }

    #[test]
    fn no_warmup_starts_at_peak() {
        assert!((lr_at(0, 50, 0, 2e-4, 0.0) - 2e-4).abs() < 1e-9);
    }
}
