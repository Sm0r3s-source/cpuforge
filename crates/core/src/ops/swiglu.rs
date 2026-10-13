//! SwiGLU gating: h = silu(g) * u, with silu(g) = g * sigmoid(g).
//! The surrounding linear layers (gate/up/down projections) are plain matmuls.

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn silu_mul_forward(g: &[f32], u: &[f32], h: &mut [f32]) {
    assert_eq!(g.len(), u.len());
    assert_eq!(g.len(), h.len());
    for ((hv, &gv), &uv) in h.iter_mut().zip(g).zip(u) {
        *hv = gv * sigmoid(gv) * uv;
    }
}

/// Accumulates into `dg` and `du`.
pub fn silu_mul_backward(dh: &[f32], g: &[f32], u: &[f32], dg: &mut [f32], du: &mut [f32]) {
    let n = g.len();
    assert!(dh.len() == n && u.len() == n && dg.len() == n && du.len() == n);
    for i in 0..n {
        let s = sigmoid(g[i]);
        let silu = g[i] * s;
        // d/dg [g * s(g)] = s + g * s * (1 - s)
        let dsilu = s * (1.0 + g[i] * (1.0 - s));
        dg[i] += dh[i] * u[i] * dsilu;
        du[i] += dh[i] * silu;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gradcheck::*;
    use crate::rng::Rng;

    fn ref_fwd(g: &[f64], u: &[f64]) -> Vec<f64> {
        g.iter().zip(u).map(|(g, u)| g / (1.0 + (-g).exp()) * u).collect()
    }

    fn data(n: usize, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut r = Rng::new(seed);
        let g = (0..n).map(|_| r.normal() * 2.0).collect();
        let u = (0..n).map(|_| r.normal()).collect();
        let d = (0..n).map(|_| r.normal()).collect();
        (g, u, d)
    }

    #[test]
    fn forward_matches_f64_reference() {
        let (g, u, _) = data(100, 1);
        let mut h = vec![0.0; 100];
        silu_mul_forward(&g, &u, &mut h);
        assert_close("swiglu fwd", &h, &ref_fwd(&to_f64(&g), &to_f64(&u)), 1e-5);
    }

    #[test]
    fn backward_matches_finite_differences() {
        let n = 40;
        let (g, u, dh) = data(n, 2);
        let mut dg = vec![0.0; n];
        let mut du = vec![0.0; n];
        silu_mul_backward(&dh, &g, &u, &mut dg, &mut du);

        let (g64, u64, d64) = (to_f64(&g), to_f64(&u), to_f64(&dh));
        let loss = |h: &[f64]| h.iter().zip(&d64).map(|(a, b)| a * b).sum::<f64>();
        let ng = numeric_grad(|gs| loss(&ref_fwd(gs, &u64)), &g64, 1e-6);
        let nu = numeric_grad(|us| loss(&ref_fwd(&g64, us)), &u64, 1e-6);
        assert_grad_close("swiglu dg", &dg, &ng, 5e-4);
        assert_grad_close("swiglu du", &du, &nu, 5e-4);
    }
}
