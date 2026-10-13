//! One Llama transformer block with hand-written forward and backward.
//!
//!   x_mid = x + Wo · attention(rope(Wq·n1), rope(Wk·n1), Wv·n1),   n1 = rmsnorm(x)
//!   out   = x_mid + Wdown · (silu(Wgate·n2) ⊙ Wup·n2),             n2 = rmsnorm(x_mid)

use cf_core::ops::attention::{attention_backward, attention_forward, AttnShape};
use cf_core::ops::rmsnorm::{rmsnorm_backward, rmsnorm_forward};
use cf_core::ops::rope::RopeCache;
use cf_core::ops::swiglu::{silu_mul_backward, silu_mul_forward};

use crate::linear::{LinGrad, Linear};

/// Derived dimensions shared by every layer.
#[derive(Clone, Copy, Debug)]
pub struct Dims {
    pub d: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub eps: f32,
}

impl Dims {
    pub fn q_dim(&self) -> usize {
        self.heads * self.head_dim
    }
    pub fn kv_dim(&self) -> usize {
        self.kv_heads * self.head_dim
    }
}

pub struct Layer {
    pub attn_norm: Vec<f32>,
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub mlp_norm: Vec<f32>,
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
}

#[derive(Default, Clone)]
pub struct LayerGrads {
    pub attn_norm: Vec<f32>,
    pub mlp_norm: Vec<f32>,
    pub wq: LinGrad,
    pub wk: LinGrad,
    pub wv: LinGrad,
    pub wo: LinGrad,
    pub gate: LinGrad,
    pub up: LinGrad,
    pub down: LinGrad,
}

/// Everything backward needs from the forward pass (except the block input).
#[derive(Default)]
pub struct LayerCache {
    n1: Vec<f32>,
    rstd1: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn_o: Vec<f32>,
    lse: Vec<f32>,
    x_mid: Vec<f32>,
    n2: Vec<f32>,
    rstd2: Vec<f32>,
    g: Vec<f32>,
    u: Vec<f32>,
    h: Vec<f32>,
    /// LoRA bottlenecks for q, k, v, o, gate, up, down.
    xs: [Vec<f32>; 7],
}

impl Layer {
    /// Forward over `t` tokens. `out` is resized and overwritten.
    pub fn forward(&self, dm: &Dims, rope: &RopeCache, x_in: &[f32], t: usize, c: &mut LayerCache, out: &mut Vec<f32>) {
        let (d, qd, kvd, f) = (dm.d, dm.q_dim(), dm.kv_dim(), dm.ffn);
        assert_eq!(x_in.len(), t * d);

        c.n1.resize(t * d, 0.0);
        c.rstd1.resize(t, 0.0);
        rmsnorm_forward(x_in, &self.attn_norm, &mut c.n1, &mut c.rstd1, t, d, dm.eps);

        c.q.resize(t * qd, 0.0);
        c.k.resize(t * kvd, 0.0);
        c.v.resize(t * kvd, 0.0);
        self.wq.forward(&c.n1, t, &mut c.q, &mut c.xs[0]);
        self.wk.forward(&c.n1, t, &mut c.k, &mut c.xs[1]);
        self.wv.forward(&c.n1, t, &mut c.v, &mut c.xs[2]);
        rope.apply(&mut c.q, t, dm.heads, 0, false);
        rope.apply(&mut c.k, t, dm.kv_heads, 0, false);

        c.attn_o.resize(t * qd, 0.0);
        c.lse.resize(dm.heads * t, 0.0);
        let shape = AttnShape { seq: t, n_heads: dm.heads, n_kv: dm.kv_heads, head_dim: dm.head_dim };
        attention_forward(&c.q, &c.k, &c.v, &mut c.attn_o, &mut c.lse, shape);

        let mut ao = vec![0.0f32; t * d];
        self.wo.forward(&c.attn_o, t, &mut ao, &mut c.xs[3]);
        c.x_mid.clear();
        c.x_mid.extend(x_in.iter().zip(&ao).map(|(a, b)| a + b));

        c.n2.resize(t * d, 0.0);
        c.rstd2.resize(t, 0.0);
        rmsnorm_forward(&c.x_mid, &self.mlp_norm, &mut c.n2, &mut c.rstd2, t, d, dm.eps);

        c.g.resize(t * f, 0.0);
        c.u.resize(t * f, 0.0);
        c.h.resize(t * f, 0.0);
        self.gate.forward(&c.n2, t, &mut c.g, &mut c.xs[4]);
        self.up.forward(&c.n2, t, &mut c.u, &mut c.xs[5]);
        silu_mul_forward(&c.g, &c.u, &mut c.h);

        let mut dn = vec![0.0f32; t * d];
        self.down.forward(&c.h, t, &mut dn, &mut c.xs[6]);
        out.clear();
        out.extend(c.x_mid.iter().zip(&dn).map(|(a, b)| a + b));
    }

    /// Backward. Accumulates parameter gradients into `lg` and returns dL/dx_in.
    #[allow(clippy::too_many_arguments)]
    pub fn backward(
        &self,
        dm: &Dims,
        rope: &RopeCache,
        x_in: &[f32],
        d_out: &[f32],
        t: usize,
        c: &LayerCache,
        lg: &mut LayerGrads,
    ) -> Vec<f32> {
        let (d, qd, kvd, f) = (dm.d, dm.q_dim(), dm.kv_dim(), dm.ffn);
        assert_eq!(d_out.len(), t * d);

        // --- MLP branch ---------------------------------------------------
        let mut dh = vec![0.0f32; t * f];
        self.down.backward(&c.h, &c.xs[6], d_out, t, &mut dh, &mut lg.down);

        let mut dg = vec![0.0f32; t * f];
        let mut du = vec![0.0f32; t * f];
        silu_mul_backward(&dh, &c.g, &c.u, &mut dg, &mut du);
        drop(dh);

        let mut dn2 = vec![0.0f32; t * d];
        self.gate.backward(&c.n2, &c.xs[4], &dg, t, &mut dn2, &mut lg.gate);
        self.up.backward(&c.n2, &c.xs[5], &du, t, &mut dn2, &mut lg.up);

        // The residual passes d_out straight to x_mid; the norm adds its share.
        let mut dx_mid = d_out.to_vec();
        let mut scratch = Vec::new();
        let dw = norm_grad(&mut lg.mlp_norm, &mut scratch, d);
        rmsnorm_backward(&dn2, &c.x_mid, &self.mlp_norm, &c.rstd2, &mut dx_mid, dw, t, d);

        // --- Attention branch ----------------------------------------------
        let mut d_attn_o = vec![0.0f32; t * qd];
        self.wo.backward(&c.attn_o, &c.xs[3], &dx_mid, t, &mut d_attn_o, &mut lg.wo);

        let mut dq = vec![0.0f32; t * qd];
        let mut dk = vec![0.0f32; t * kvd];
        let mut dv = vec![0.0f32; t * kvd];
        let shape = AttnShape { seq: t, n_heads: dm.heads, n_kv: dm.kv_heads, head_dim: dm.head_dim };
        attention_backward(&d_attn_o, &c.q, &c.k, &c.v, &c.attn_o, &c.lse, &mut dq, &mut dk, &mut dv, shape);
        drop(d_attn_o);

        // Rotation backward = inverse rotation.
        rope.apply(&mut dq, t, dm.heads, 0, true);
        rope.apply(&mut dk, t, dm.kv_heads, 0, true);

        let mut dn1 = vec![0.0f32; t * d];
        self.wq.backward(&c.n1, &c.xs[0], &dq, t, &mut dn1, &mut lg.wq);
        self.wk.backward(&c.n1, &c.xs[1], &dk, t, &mut dn1, &mut lg.wk);
        self.wv.backward(&c.n1, &c.xs[2], &dv, t, &mut dn1, &mut lg.wv);

        let mut dx_in = dx_mid; // residual path into x_in
        let mut scratch1 = Vec::new();
        let dw1 = norm_grad(&mut lg.attn_norm, &mut scratch1, d);
        rmsnorm_backward(&dn1, x_in, &self.attn_norm, &c.rstd1, &mut dx_in, dw1, t, d);
        dx_in
    }
}

/// Gradient target for a norm weight: the real buffer if trainable, else a throwaway.
fn norm_grad<'a>(g: &'a mut [f32], scratch: &'a mut Vec<f32>, d: usize) -> &'a mut [f32] {
    if g.is_empty() {
        *scratch = vec![0.0; d];
        scratch
    } else {
        g
    }
}
