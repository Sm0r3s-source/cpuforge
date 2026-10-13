//! Linear layer (y = x·Wᵀ) with an optional LoRA adapter:
//!   y = x·Wᵀ + (alpha/r) · (x·Aᵀ)·Bᵀ,   W:[out,in]  A:[r,in]  B:[out,r]
//!
//! Which tensors receive gradients is decided by `LinGrad`: an empty Vec means
//! "frozen, skip the work". Frozen base weights cost no weight-gradient matmul.

use cf_core::matmul::{matmul_nn, matmul_nt, matmul_tn};
use cf_core::rng::Rng;
use cf_core::Tensor;

pub struct Lora {
    pub a: Tensor,
    pub b: Tensor,
    pub scale: f32,
}

pub struct Linear {
    pub w: Tensor,
    pub lora: Option<Lora>,
}

#[derive(Default, Clone)]
pub struct LinGrad {
    pub w: Vec<f32>,
    pub a: Vec<f32>,
    pub b: Vec<f32>,
}

impl Linear {
    pub fn new(w: Tensor) -> Self {
        assert_eq!(w.shape().len(), 2);
        Self { w, lora: None }
    }

    pub fn out_dim(&self) -> usize {
        self.w.rows()
    }

    pub fn in_dim(&self) -> usize {
        self.w.cols()
    }

    /// Attach a fresh adapter. B starts at zero so the model is unchanged at step 0.
    pub fn add_lora(&mut self, rank: usize, alpha: f32, rng: &mut Rng) {
        let (out, inn) = (self.out_dim(), self.in_dim());
        let bound = 1.0 / (inn as f32).sqrt();
        let mut a = Tensor::zeros(&[rank, inn]);
        for v in a.data_mut() {
            *v = (rng.uniform() * 2.0 - 1.0) * bound;
        }
        self.lora = Some(Lora { a, b: Tensor::zeros(&[out, rank]), scale: alpha / rank as f32 });
    }

    /// y[t,out] = forward(x[t,in]). `xs` receives the scaled LoRA bottleneck
    /// (scale·x·Aᵀ, [t,r]) that backward needs; it is cleared when there is no adapter.
    pub fn forward(&self, x: &[f32], t: usize, y: &mut [f32], xs: &mut Vec<f32>) {
        let (inn, out) = (self.in_dim(), self.out_dim());
        matmul_nt(x, self.w.data(), y, t, inn, out, false);
        match &self.lora {
            Some(l) => {
                let r = l.a.rows();
                xs.resize(t * r, 0.0);
                matmul_nt(x, l.a.data(), xs, t, inn, r, false);
                xs.iter_mut().for_each(|v| *v *= l.scale);
                matmul_nt(xs, l.b.data(), y, t, r, out, true);
            }
            None => xs.clear(),
        }
    }

    /// Accumulates into `dx` and into whichever tensors `g` has room for.
    pub fn backward(&self, x: &[f32], xs: &[f32], dy: &[f32], t: usize, dx: &mut [f32], g: &mut LinGrad) {
        let (inn, out) = (self.in_dim(), self.out_dim());
        matmul_nn(dy, self.w.data(), dx, t, out, inn, true);
        if !g.w.is_empty() {
            matmul_tn(dy, x, &mut g.w, out, t, inn, true);
        }
        if let Some(l) = &self.lora {
            let r = l.a.rows();
            if !g.b.is_empty() {
                matmul_tn(dy, xs, &mut g.b, out, t, r, true);
            }
            let mut dxa = vec![0.0f32; t * r];
            matmul_nn(dy, l.b.data(), &mut dxa, t, out, r, false);
            dxa.iter_mut().for_each(|v| *v *= l.scale);
            if !g.a.is_empty() {
                matmul_tn(&dxa, x, &mut g.a, r, t, inn, true);
            }
            matmul_nn(&dxa, l.a.data(), dx, t, r, inn, true);
        }
    }
}
