//! Row-major f32 matmul on top of the `gemm` crate (packed, SIMD-blocked,
//! multi-threaded microkernels).
//!
//! Three layouts cover everything a linear layer needs:
//!   y  = x  · Wᵀ   -> `matmul_nt`  (forward,  W stored [out, in])
//!   dx = dy · W    -> `matmul_nn`  (backward, input grad)
//!   dW = dyᵀ · x   -> `matmul_tn`  (backward, weight grad)
//!
//! All kernels take an `acc` flag: `false` overwrites C, `true` adds into it.
//! Transposes are free: they are expressed as strides, never as copies.

use gemm::Parallelism;

/// A strided view of a matrix: element (i, j) lives at `data[i * rs + j * cs]`.
#[derive(Clone, Copy)]
pub struct Mat<'a> {
    pub data: &'a [f32],
    pub rs: isize,
    pub cs: isize,
}

impl<'a> Mat<'a> {
    pub fn new(data: &'a [f32], rs: isize, cs: isize) -> Self {
        Self { data, rs, cs }
    }

    /// Row-major contiguous [_, cols].
    pub fn row_major(data: &'a [f32], cols: usize) -> Self {
        Self { data, rs: cols as isize, cs: 1 }
    }
}

/// C[m,n] (+)= A[m,k] · B[k,n] where A, B and C are arbitrary strided views.
/// This is what lets attention work on per-head slices of [T, heads, head_dim]
/// tensors without copying. Strides must be non-negative.
#[allow(clippy::too_many_arguments)]
pub fn gemm_strided(
    a: Mat,
    b: Mat,
    c: &mut [f32],
    c_rs: isize,
    c_cs: isize,
    m: usize,
    k: usize,
    n: usize,
    acc: bool,
) {
    if m == 0 || n == 0 {
        return;
    }
    assert!(
        a.rs >= 0 && a.cs >= 0 && b.rs >= 0 && b.cs >= 0 && c_rs >= 0 && c_cs >= 0,
        "negative strides are not supported"
    );
    // Every logical element must land inside its slice, which keeps the unsafe call sound.
    let c_max = (m as isize - 1) * c_rs + (n as isize - 1) * c_cs;
    assert!((c_max as usize) < c.len(), "C too small for {m}x{n}");
    if k == 0 {
        if !acc {
            for i in 0..m {
                for j in 0..n {
                    c[i * c_rs as usize + j * c_cs as usize] = 0.0;
                }
            }
        }
        return;
    }
    let a_max = (m as isize - 1) * a.rs + (k as isize - 1) * a.cs;
    let b_max = (k as isize - 1) * b.rs + (n as isize - 1) * b.cs;
    assert!((a_max as usize) < a.data.len(), "A too small for {m}x{k}");
    assert!((b_max as usize) < b.data.len(), "B too small for {k}x{n}");

    // SAFETY: all three views were bounds-checked above. `c` is an exclusive borrow, so it
    // cannot alias `a` or `b`, which are shared borrows.
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            c.as_mut_ptr(),
            c_cs,
            c_rs,
            acc,
            a.data.as_ptr(),
            a.cs,
            a.rs,
            b.data.as_ptr(),
            b.cs,
            b.rs,
            1.0,
            1.0,
            false,
            false,
            false,
            Parallelism::Rayon(0), // gemm applies its own size threshold
        );
    }
}

/// Contiguous row-major C with strided A and B.
#[allow(clippy::too_many_arguments)]
fn gemm_f32(
    a: &[f32],
    a_strides: (isize, isize),
    b: &[f32],
    b_strides: (isize, isize),
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    acc: bool,
) {
    assert_eq!(c.len(), m * n);
    gemm_strided(
        Mat::new(a, a_strides.0, a_strides.1),
        Mat::new(b, b_strides.0, b_strides.1),
        c,
        n as isize,
        1,
        m,
        k,
        n,
        acc,
    );
}

/// C[m,n] (+)= A[m,k] · B[k,n]
pub fn matmul_nn(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize, acc: bool) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    gemm_f32(a, (k as isize, 1), b, (n as isize, 1), c, m, k, n, acc);
}

/// C[m,n] (+)= A[m,k] · B[n,k]ᵀ   (B stored row-major as [n,k])
pub fn matmul_nt(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize, acc: bool) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    gemm_f32(a, (k as isize, 1), b, (1, k as isize), c, m, k, n, acc);
}

/// C[m,n] (+)= A[k,m]ᵀ · B[k,n]   (A stored row-major as [k,m])
pub fn matmul_tn(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize, acc: bool) {
    assert_eq!(a.len(), k * m);
    assert_eq!(b.len(), k * n);
    gemm_f32(a, (1, m as isize), b, (n as isize, 1), c, m, k, n, acc);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn rand_vec(n: usize, rng: &mut Rng) -> Vec<f32> {
        (0..n).map(|_| rng.normal()).collect()
    }

    /// f64 reference: C = op(A) · op(B).
    fn reference(
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
        a_t: bool,
        b_t: bool,
    ) -> Vec<f64> {
        let mut c = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f64;
                for p in 0..k {
                    let av = if a_t { a[p * m + i] } else { a[i * k + p] };
                    let bv = if b_t { b[j * k + p] } else { b[p * n + j] };
                    s += av as f64 * bv as f64;
                }
                c[i * n + j] = s;
            }
        }
        c
    }

    fn check(c: &[f32], r: &[f64], tol: f64) {
        for (x, y) in c.iter().zip(r) {
            let err = (*x as f64 - y).abs();
            assert!(err <= tol * (1.0 + y.abs()), "got {x}, want {y}");
        }
    }

    // Includes sizes that are not multiples of the SIMD width and large ones that go multi-threaded.
    const SHAPES: &[(usize, usize, usize)] = &[
        (1, 1, 1),
        (3, 5, 7),
        (17, 33, 9),
        (64, 64, 64),
        (130, 70, 90),
        (256, 257, 255),
        (512, 300, 400),
    ];

    #[test]
    fn nn_matches_reference() {
        let mut rng = Rng::new(11);
        for &(m, k, n) in SHAPES {
            let a = rand_vec(m * k, &mut rng);
            let b = rand_vec(k * n, &mut rng);
            let mut c = vec![0.0; m * n];
            matmul_nn(&a, &b, &mut c, m, k, n, false);
            check(&c, &reference(&a, &b, m, k, n, false, false), 1e-4);
        }
    }

    #[test]
    fn nt_matches_reference() {
        let mut rng = Rng::new(12);
        for &(m, k, n) in SHAPES {
            let a = rand_vec(m * k, &mut rng);
            let b = rand_vec(n * k, &mut rng);
            let mut c = vec![0.0; m * n];
            matmul_nt(&a, &b, &mut c, m, k, n, false);
            check(&c, &reference(&a, &b, m, k, n, false, true), 1e-4);
        }
    }

    #[test]
    fn tn_matches_reference() {
        let mut rng = Rng::new(13);
        for &(m, k, n) in SHAPES {
            let a = rand_vec(k * m, &mut rng);
            let b = rand_vec(k * n, &mut rng);
            let mut c = vec![0.0; m * n];
            matmul_tn(&a, &b, &mut c, m, k, n, false);
            check(&c, &reference(&a, &b, m, k, n, true, false), 1e-4);
        }
    }

    #[test]
    fn overwrite_ignores_garbage_in_c() {
        let mut rng = Rng::new(15);
        let (m, k, n) = (20, 30, 25);
        let a = rand_vec(m * k, &mut rng);
        let b = rand_vec(k * n, &mut rng);
        let mut c = vec![f32::NAN; m * n];
        matmul_nn(&a, &b, &mut c, m, k, n, false);
        check(&c, &reference(&a, &b, m, k, n, false, false), 1e-4);
    }

    #[test]
    fn accumulate_adds_instead_of_overwriting() {
        let mut rng = Rng::new(14);
        for &(m, k, n) in &[(9, 11, 13), (200, 150, 180)] {
            let a = rand_vec(m * k, &mut rng);
            let b = rand_vec(k * n, &mut rng);
            let base = rand_vec(m * n, &mut rng);
            let r = reference(&a, &b, m, k, n, false, false);
            let want: Vec<f64> = r.iter().zip(&base).map(|(x, y)| x + *y as f64).collect();

            let mut c = base.clone();
            matmul_nn(&a, &b, &mut c, m, k, n, true);
            check(&c, &want, 1e-4);

            // Same product through the nt kernel, with B laid out as [n,k].
            let mut bt = vec![0.0; n * k];
            for p in 0..k {
                for j in 0..n {
                    bt[j * k + p] = b[p * n + j];
                }
            }
            let mut c2 = base.clone();
            matmul_nt(&a, &bt, &mut c2, m, k, n, true);
            check(&c2, &want, 1e-4);
        }
    }

    #[test]
    fn zero_inner_dimension() {
        let mut c = vec![5.0; 6];
        matmul_nn(&[], &[], &mut c, 2, 0, 3, false);
        assert!(c.iter().all(|&v| v == 0.0));
        let mut c = vec![5.0; 6];
        matmul_nn(&[], &[], &mut c, 2, 0, 3, true);
        assert!(c.iter().all(|&v| v == 5.0));
    }
}
