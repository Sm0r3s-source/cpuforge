//! RMSNorm: y = x * rsqrt(mean(x²) + eps) * w, applied per row.

/// Forward. `rstd` receives the per-row 1/rms, which backward needs.
pub fn rmsnorm_forward(
    x: &[f32],
    w: &[f32],
    y: &mut [f32],
    rstd: &mut [f32],
    rows: usize,
    dim: usize,
    eps: f32,
) {
    assert_eq!(x.len(), rows * dim);
    assert_eq!(y.len(), rows * dim);
    assert_eq!(w.len(), dim);
    assert_eq!(rstd.len(), rows);
    for r in 0..rows {
        let xr = &x[r * dim..(r + 1) * dim];
        let ms = xr.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        rstd[r] = inv;
        for ((yv, &xv), &wv) in y[r * dim..(r + 1) * dim].iter_mut().zip(xr).zip(w) {
            *yv = xv * inv * wv;
        }
    }
}

/// Backward. Accumulates into `dx` and `dw`.
///
/// With x̂ = x·r, dx̂ = dy·w:  dx = r·(dx̂ − x̂·mean(dx̂·x̂)),  dw = dy·x̂.
#[allow(clippy::too_many_arguments)]
pub fn rmsnorm_backward(
    dy: &[f32],
    x: &[f32],
    w: &[f32],
    rstd: &[f32],
    dx: &mut [f32],
    dw: &mut [f32],
    rows: usize,
    dim: usize,
) {
    assert_eq!(dy.len(), rows * dim);
    assert_eq!(x.len(), rows * dim);
    assert_eq!(dx.len(), rows * dim);
    assert_eq!(w.len(), dim);
    assert_eq!(dw.len(), dim);
    assert_eq!(rstd.len(), rows);
    for r in 0..rows {
        let xr = &x[r * dim..(r + 1) * dim];
        let dyr = &dy[r * dim..(r + 1) * dim];
        let inv = rstd[r];

        // mean(dx̂ · x̂) over the row
        let mut dot = 0.0f32;
        for i in 0..dim {
            dot += dyr[i] * w[i] * xr[i] * inv;
        }
        let mean = dot / dim as f32;

        let dxr = &mut dx[r * dim..(r + 1) * dim];
        for i in 0..dim {
            let xhat = xr[i] * inv;
            dw[i] += dyr[i] * xhat;
            dxr[i] += inv * (dyr[i] * w[i] - xhat * mean);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gradcheck::*;
    use crate::rng::Rng;

    fn rmsnorm_ref(x: &[f64], w: &[f64], rows: usize, dim: usize, eps: f64) -> Vec<f64> {
        let mut y = vec![0.0; rows * dim];
        for r in 0..rows {
            let xr = &x[r * dim..(r + 1) * dim];
            let ms = xr.iter().map(|v| v * v).sum::<f64>() / dim as f64;
            let inv = 1.0 / (ms + eps).sqrt();
            for i in 0..dim {
                y[r * dim + i] = xr[i] * inv * w[i];
            }
        }
        y
    }

    fn setup(rows: usize, dim: usize, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut rng = Rng::new(seed);
        let x: Vec<f32> = (0..rows * dim).map(|_| rng.normal() * 1.5 + 0.3).collect();
        let w: Vec<f32> = (0..dim).map(|_| 1.0 + 0.5 * rng.normal()).collect();
        let g: Vec<f32> = (0..rows * dim).map(|_| rng.normal()).collect();
        (x, w, g)
    }

    #[test]
    fn forward_matches_f64_reference() {
        let (rows, dim, eps) = (5, 37, 1e-5f32);
        let (x, w, _) = setup(rows, dim, 1);
        let mut y = vec![0.0; rows * dim];
        let mut rstd = vec![0.0; rows];
        rmsnorm_forward(&x, &w, &mut y, &mut rstd, rows, dim, eps);
        let want = rmsnorm_ref(&to_f64(&x), &to_f64(&w), rows, dim, eps as f64);
        assert_close("rmsnorm y", &y, &want, 1e-5);
    }

    #[test]
    fn output_has_unit_rms_when_weight_is_one() {
        let (rows, dim) = (4, 64);
        let (x, _, _) = setup(rows, dim, 2);
        let w = vec![1.0; dim];
        let mut y = vec![0.0; rows * dim];
        let mut rstd = vec![0.0; rows];
        rmsnorm_forward(&x, &w, &mut y, &mut rstd, rows, dim, 1e-8);
        for r in 0..rows {
            let ms = y[r * dim..(r + 1) * dim].iter().map(|v| v * v).sum::<f32>() / dim as f32;
            assert!((ms - 1.0).abs() < 1e-4, "row {r} mean-square {ms}");
        }
    }

    #[test]
    fn backward_matches_finite_differences() {
        let (rows, dim, eps) = (3, 16, 1e-5f32);
        let (x, w, g) = setup(rows, dim, 3);

        // Analytic
        let mut y = vec![0.0; rows * dim];
        let mut rstd = vec![0.0; rows];
        rmsnorm_forward(&x, &w, &mut y, &mut rstd, rows, dim, eps);
        let mut dx = vec![0.0; rows * dim];
        let mut dw = vec![0.0; dim];
        rmsnorm_backward(&g, &x, &w, &rstd, &mut dx, &mut dw, rows, dim);

        // Numeric: L = sum(y * g)
        let g64 = to_f64(&g);
        let (x64, w64) = (to_f64(&x), to_f64(&w));
        let loss = |y: &[f64]| y.iter().zip(&g64).map(|(a, b)| a * b).sum::<f64>();

        let nx = numeric_grad(|xs| loss(&rmsnorm_ref(xs, &w64, rows, dim, eps as f64)), &x64, 1e-6);
        let nw = numeric_grad(|ws| loss(&rmsnorm_ref(&x64, ws, rows, dim, eps as f64)), &w64, 1e-6);

        assert_grad_close("rmsnorm dx", &dx, &nx, 5e-4);
        assert_grad_close("rmsnorm dw", &dw, &nw, 5e-4);
    }

    #[test]
    fn backward_accumulates() {
        let (rows, dim, eps) = (2, 8, 1e-5f32);
        let (x, w, g) = setup(rows, dim, 4);
        let mut y = vec![0.0; rows * dim];
        let mut rstd = vec![0.0; rows];
        rmsnorm_forward(&x, &w, &mut y, &mut rstd, rows, dim, eps);

        let mut dx1 = vec![0.0; rows * dim];
        let mut dw1 = vec![0.0; dim];
        rmsnorm_backward(&g, &x, &w, &rstd, &mut dx1, &mut dw1, rows, dim);

        let mut dx2 = dx1.clone();
        let mut dw2 = dw1.clone();
        rmsnorm_backward(&g, &x, &w, &rstd, &mut dx2, &mut dw2, rows, dim);

        for i in 0..dx1.len() {
            assert!((dx2[i] - 2.0 * dx1[i]).abs() < 1e-5);
        }
        for i in 0..dw1.len() {
            assert!((dw2[i] - 2.0 * dw1[i]).abs() < 1e-5);
        }
    }
}
