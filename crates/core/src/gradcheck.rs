//! Helpers for verifying hand-written backward passes against central finite
//! differences computed in f64. The f32 analytic gradient is compared with a
//! numeric gradient of an independent f64 reference forward, so both the
//! backward formula and the f32 implementation get checked.

/// Central-difference gradient of `f` at `x`.
pub fn numeric_grad<F>(f: F, x: &[f64], eps: f64) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64,
{
    let mut xp = x.to_vec();
    let mut g = vec![0.0; x.len()];
    for i in 0..x.len() {
        let orig = xp[i];
        xp[i] = orig + eps;
        let hi = f(&xp);
        xp[i] = orig - eps;
        let lo = f(&xp);
        xp[i] = orig;
        g[i] = (hi - lo) / (2.0 * eps);
    }
    g
}

pub fn to_f64(x: &[f32]) -> Vec<f64> {
    x.iter().map(|&v| v as f64).collect()
}

/// Panics with a useful message if any element differs by more than
/// `tol * (1 + |numeric|)`.
pub fn assert_grad_close(name: &str, analytic: &[f32], numeric: &[f64], tol: f64) {
    assert_eq!(analytic.len(), numeric.len(), "{name}: length mismatch");
    let mut worst = 0.0f64;
    let mut worst_i = 0;
    for (i, (a, n)) in analytic.iter().zip(numeric).enumerate() {
        let err = (*a as f64 - n).abs() / (1.0 + n.abs());
        if err > worst {
            worst = err;
            worst_i = i;
        }
    }
    assert!(
        worst <= tol,
        "{name}: worst relative error {worst:.3e} at index {worst_i} (analytic {}, numeric {}), tol {tol:.1e}",
        analytic[worst_i],
        numeric[worst_i]
    );
}

/// Panics if two f32 slices differ by more than `tol * (1 + |want|)`.
pub fn assert_close(name: &str, got: &[f32], want: &[f64], tol: f64) {
    assert_grad_close(name, got, want, tol)
}
