//! Full Llama-style causal LM: embedding, blocks, final norm, LM head, with
//! forward, loss, and hand-written backward.

use std::collections::HashMap;

use cf_core::ops::loss::cross_entropy_fwd_bwd;
use cf_core::ops::rmsnorm::{rmsnorm_backward, rmsnorm_forward};
use cf_core::ops::rope::RopeCache;
use cf_core::matmul::{matmul_nn, matmul_nt, matmul_tn};
use cf_core::rng::Rng;
use cf_core::Tensor;

use crate::config::LlamaConfig;
use crate::layer::{Dims, Layer, LayerCache, LayerGrads};
use crate::linear::{LinGrad, Linear};

/// What to keep from the forward pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryMode {
    /// Keep every activation: fastest, most RAM.
    Full,
    /// Keep only each block's input and recompute activations in backward:
    /// roughly one extra forward pass of compute for a large RAM saving.
    Checkpoint,
}

/// Which parameters receive gradients.
#[derive(Clone, Copy, Debug)]
pub struct TrainSpec {
    /// Linear-layer weights and RMSNorm weights.
    pub base: bool,
    /// Token embeddings (and the LM head, when untied).
    pub embed: bool,
    /// LoRA adapters.
    pub lora: bool,
}

impl TrainSpec {
    pub const LORA_ONLY: TrainSpec = TrainSpec { base: false, embed: false, lora: true };
    pub const FULL: TrainSpec = TrainSpec { base: true, embed: true, lora: true };
}

/// How adapters were attached (needed to save/restore them).
#[derive(Clone, Debug, PartialEq)]
pub struct LoraConfig {
    pub rank: usize,
    pub alpha: f32,
    pub targets: Vec<String>,
}

pub struct Model {
    pub cfg: LlamaConfig,
    pub lora_cfg: Option<LoraConfig>,
    pub dims: Dims,
    pub max_seq: usize,
    pub embed: Tensor,
    pub layers: Vec<Layer>,
    pub final_norm: Vec<f32>,
    pub lm_head: Option<Tensor>,
    rope: RopeCache,
}

/// Gradient buffers mirroring the model. An empty Vec means "not trained".
#[derive(Default, Clone)]
pub struct Grads {
    pub embed: Vec<f32>,
    pub layers: Vec<LayerGrads>,
    pub final_norm: Vec<f32>,
    pub lm_head: Vec<f32>,
}

fn zeros_if(cond: bool, n: usize) -> Vec<f32> {
    if cond {
        vec![0.0; n]
    } else {
        Vec::new()
    }
}

fn lin_grad(l: &Linear, spec: TrainSpec) -> LinGrad {
    let mut g = LinGrad { w: zeros_if(spec.base, l.w.numel()), ..Default::default() };
    if let (Some(lo), true) = (&l.lora, spec.lora) {
        g.a = vec![0.0; lo.a.numel()];
        g.b = vec![0.0; lo.b.numel()];
    }
    g
}

impl Grads {
    pub fn new(m: &Model, spec: TrainSpec) -> Self {
        let d = m.dims.d;
        Grads {
            embed: zeros_if(spec.embed, m.embed.numel()),
            final_norm: zeros_if(spec.base, d),
            lm_head: match &m.lm_head {
                Some(h) => zeros_if(spec.embed, h.numel()),
                None => Vec::new(),
            },
            layers: m
                .layers
                .iter()
                .map(|l| LayerGrads {
                    attn_norm: zeros_if(spec.base, d),
                    q_norm: zeros_if(spec.base && l.q_norm.is_some(), m.dims.head_dim),
                    k_norm: zeros_if(spec.base && l.k_norm.is_some(), m.dims.head_dim),
                    mlp_norm: zeros_if(spec.base, d),
                    wq: lin_grad(&l.wq, spec),
                    wk: lin_grad(&l.wk, spec),
                    wv: lin_grad(&l.wv, spec),
                    wo: lin_grad(&l.wo, spec),
                    gate: lin_grad(&l.gate, spec),
                    up: lin_grad(&l.up, spec),
                    down: lin_grad(&l.down, spec),
                })
                .collect(),
        }
    }

    fn vecs_mut(&mut self) -> Vec<&mut Vec<f32>> {
        let mut v = vec![&mut self.embed, &mut self.final_norm, &mut self.lm_head];
        for l in &mut self.layers {
            v.push(&mut l.attn_norm);
            v.push(&mut l.q_norm);
            v.push(&mut l.k_norm);
            v.push(&mut l.mlp_norm);
            for g in [&mut l.wq, &mut l.wk, &mut l.wv, &mut l.wo, &mut l.gate, &mut l.up, &mut l.down] {
                v.push(&mut g.w);
                v.push(&mut g.a);
                v.push(&mut g.b);
            }
        }
        v
    }

    pub fn zero(&mut self) {
        for v in self.vecs_mut() {
            v.fill(0.0);
        }
    }

    pub fn scale(&mut self, s: f32) {
        for v in self.vecs_mut() {
            v.iter_mut().for_each(|x| *x *= s);
        }
    }

    /// L2 norm over every trained gradient.
    pub fn global_norm(&mut self) -> f32 {
        let mut sum = 0.0f64;
        for v in self.vecs_mut() {
            sum += v.iter().map(|&x| x as f64 * x as f64).sum::<f64>();
        }
        sum.sqrt() as f32
    }

    pub fn numel(&mut self) -> usize {
        self.vecs_mut().iter().map(|v| v.len()).sum()
    }
}

/// Callback for `visit_params`: (name, shape, parameter data, gradient or empty slice).
pub type ParamVisitor<'a> = dyn FnMut(&str, &[usize], &mut [f32], &[f32]) + 'a;

fn visit_linear(name: &str, lin: &mut Linear, g: Option<&LinGrad>, f: &mut ParamVisitor) {
    let e: &[f32] = &[];
    let shape = lin.w.shape().to_vec();
    f(&format!("{name}.weight"), &shape, lin.w.data_mut(), g.map_or(e, |g| &g.w));
    if let Some(l) = &mut lin.lora {
        let (sa, sb) = (l.a.shape().to_vec(), l.b.shape().to_vec());
        f(&format!("{name}.lora_A"), &sa, l.a.data_mut(), g.map_or(e, |g| &g.a));
        f(&format!("{name}.lora_B"), &sb, l.b.data_mut(), g.map_or(e, |g| &g.b));
    }
}

/// Visit every parameter in a fixed order with its (possibly empty) gradient.
/// Names follow Hugging Face conventions; LoRA tensors are `<proj>.lora_A/B`.
/// This one traversal serves the optimizer, checkpointing and tests, so the
/// parameter/gradient pairing cannot drift.
pub fn visit_params(m: &mut Model, grads: Option<&Grads>, f: &mut ParamVisitor) {
    let e: &[f32] = &[];
    let (d, hd) = (m.dims.d, m.dims.head_dim);
    let embed_shape = m.embed.shape().to_vec();
    f("model.embed_tokens.weight", &embed_shape, m.embed.data_mut(), grads.map_or(e, |g| &g.embed));
    for (i, l) in m.layers.iter_mut().enumerate() {
        let lg = grads.map(|g| &g.layers[i]);
        let p = format!("model.layers.{i}");
        f(&format!("{p}.input_layernorm.weight"), &[d], &mut l.attn_norm, lg.map_or(e, |g| &g.attn_norm));
        visit_linear(&format!("{p}.self_attn.q_proj"), &mut l.wq, lg.map(|g| &g.wq), f);
        if let Some(w) = &mut l.q_norm {
            f(&format!("{p}.self_attn.q_norm.weight"), &[hd], w, lg.map_or(e, |g| &g.q_norm));
        }
        visit_linear(&format!("{p}.self_attn.k_proj"), &mut l.wk, lg.map(|g| &g.wk), f);
        if let Some(w) = &mut l.k_norm {
            f(&format!("{p}.self_attn.k_norm.weight"), &[hd], w, lg.map_or(e, |g| &g.k_norm));
        }
        visit_linear(&format!("{p}.self_attn.v_proj"), &mut l.wv, lg.map(|g| &g.wv), f);
        visit_linear(&format!("{p}.self_attn.o_proj"), &mut l.wo, lg.map(|g| &g.wo), f);
        f(&format!("{p}.post_attention_layernorm.weight"), &[d], &mut l.mlp_norm, lg.map_or(e, |g| &g.mlp_norm));
        visit_linear(&format!("{p}.mlp.gate_proj"), &mut l.gate, lg.map(|g| &g.gate), f);
        visit_linear(&format!("{p}.mlp.up_proj"), &mut l.up, lg.map(|g| &g.up), f);
        visit_linear(&format!("{p}.mlp.down_proj"), &mut l.down, lg.map(|g| &g.down), f);
    }
    f("model.norm.weight", &[d], &mut m.final_norm, grads.map_or(e, |g| &g.final_norm));
    if let Some(h) = &mut m.lm_head {
        let shape = h.shape().to_vec();
        f("lm_head.weight", &shape, h.data_mut(), grads.map_or(e, |g| &g.lm_head));
    }
}

fn take(map: &mut HashMap<String, Tensor>, name: &str, shape: &[usize]) -> Result<Tensor, String> {
    let t = map.remove(name).ok_or_else(|| format!("missing tensor '{name}'"))?;
    if t.shape() != shape {
        return Err(format!("tensor '{name}' has shape {:?}, expected {:?}", t.shape(), shape));
    }
    Ok(t)
}

impl Model {
    fn dims_of(cfg: &LlamaConfig) -> Dims {
        Dims {
            d: cfg.hidden_size,
            heads: cfg.num_attention_heads,
            kv_heads: cfg.kv_heads(),
            head_dim: cfg.head_dim(),
            ffn: cfg.intermediate_size,
            eps: cfg.rms_norm_eps,
        }
    }

    fn assemble(
        cfg: LlamaConfig,
        max_seq: usize,
        embed: Tensor,
        layers: Vec<Layer>,
        final_norm: Vec<f32>,
        lm_head: Option<Tensor>,
    ) -> Self {
        let dims = Self::dims_of(&cfg);
        let scaling = cfg.llama3_scaling().expect("validated config");
        let rope = RopeCache::with_scaling(dims.head_dim, max_seq, cfg.rope_theta, scaling);
        Model { cfg, lora_cfg: None, dims, max_seq, embed, layers, final_norm, lm_head, rope }
    }

    /// Build from Hugging Face-named tensors (already converted to f32).
    pub fn from_named(cfg: LlamaConfig, max_seq: usize, mut t: HashMap<String, Tensor>) -> Result<Self, String> {
        cfg.validate()?;
        let dm = Self::dims_of(&cfg);
        let (d, qd, kvd, f, v) = (dm.d, dm.q_dim(), dm.kv_dim(), dm.ffn, cfg.vocab_size);

        let embed = take(&mut t, "model.embed_tokens.weight", &[v, d])?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{i}");
            layers.push(Layer {
                attn_norm: take(&mut t, &format!("{p}.input_layernorm.weight"), &[d])?.data().to_vec(),
                wq: Linear::new(take(&mut t, &format!("{p}.self_attn.q_proj.weight"), &[qd, d])?),
                wk: Linear::new(take(&mut t, &format!("{p}.self_attn.k_proj.weight"), &[kvd, d])?),
                wv: Linear::new(take(&mut t, &format!("{p}.self_attn.v_proj.weight"), &[kvd, d])?),
                wo: Linear::new(take(&mut t, &format!("{p}.self_attn.o_proj.weight"), &[d, qd])?),
                q_norm: if cfg.qk_norm() { Some(take(&mut t, &format!("{p}.self_attn.q_norm.weight"), &[dm.head_dim])?.data().to_vec()) } else { None },
                k_norm: if cfg.qk_norm() { Some(take(&mut t, &format!("{p}.self_attn.k_norm.weight"), &[dm.head_dim])?.data().to_vec()) } else { None },
                mlp_norm: take(&mut t, &format!("{p}.post_attention_layernorm.weight"), &[d])?.data().to_vec(),
                gate: Linear::new(take(&mut t, &format!("{p}.mlp.gate_proj.weight"), &[f, d])?),
                up: Linear::new(take(&mut t, &format!("{p}.mlp.up_proj.weight"), &[f, d])?),
                down: Linear::new(take(&mut t, &format!("{p}.mlp.down_proj.weight"), &[d, f])?),
            });
        }
        let final_norm = take(&mut t, "model.norm.weight", &[d])?.data().to_vec();
        let lm_head = if cfg.tie_word_embeddings {
            t.remove("lm_head.weight"); // identical to the embedding in tied checkpoints
            None
        } else {
            Some(take(&mut t, "lm_head.weight", &[v, d])?)
        };
        if let Some(name) = t.keys().find(|k| k.ends_with(".bias")) {
            return Err(format!("checkpoint has bias tensor '{name}', which this engine does not support yet"));
        }
        Ok(Self::assemble(cfg, max_seq, embed, layers, final_norm, lm_head))
    }

    /// Random init, mainly for tests and from-scratch experiments.
    pub fn random(cfg: LlamaConfig, max_seq: usize, seed: u64, std: f32) -> Self {
        let mut rng = Rng::new(seed);
        let dm = Self::dims_of(&cfg);
        let (d, qd, kvd, f, v) = (dm.d, dm.q_dim(), dm.kv_dim(), dm.ffn, cfg.vocab_size);
        let mut lin = |o: usize, i: usize| Linear::new(Tensor::randn(&[o, i], std, &mut rng));
        let layers: Vec<Layer> = (0..cfg.num_hidden_layers)
            .map(|_| Layer {
                attn_norm: vec![1.0; d],
                wq: lin(qd, d),
                wk: lin(kvd, d),
                wv: lin(kvd, d),
                wo: lin(d, qd),
                q_norm: cfg.qk_norm().then(|| vec![1.0; dm.head_dim]),
                k_norm: cfg.qk_norm().then(|| vec![1.0; dm.head_dim]),
                mlp_norm: vec![1.0; d],
                gate: lin(f, d),
                up: lin(f, d),
                down: lin(d, f),
            })
            .collect();
        let embed = Tensor::randn(&[v, d], std, &mut rng);
        let lm_head = if cfg.tie_word_embeddings { None } else { Some(Tensor::randn(&[v, d], std, &mut rng)) };
        Self::assemble(cfg, max_seq, embed, layers, vec![1.0; d], lm_head)
    }

    /// Attach LoRA adapters to the named projections: any of
    /// `q`, `k`, `v`, `o`, `gate`, `up`, `down`.
    pub fn add_lora(&mut self, rank: usize, alpha: f32, targets: &[&str], seed: u64) -> Result<(), String> {
        for t in targets {
            if !["q", "k", "v", "o", "gate", "up", "down"].contains(t) {
                return Err(format!("unknown LoRA target '{t}'"));
            }
        }
        if self.lora_cfg.is_some() {
            return Err("model already has LoRA adapters".into());
        }
        let mut rng = Rng::new(seed);
        for l in &mut self.layers {
            for (name, lin) in [
                ("q", &mut l.wq),
                ("k", &mut l.wk),
                ("v", &mut l.wv),
                ("o", &mut l.wo),
                ("gate", &mut l.gate),
                ("up", &mut l.up),
                ("down", &mut l.down),
            ] {
                if targets.contains(&name) {
                    lin.add_lora(rank, alpha, &mut rng);
                }
            }
        }
        self.lora_cfg = Some(LoraConfig { rank, alpha, targets: targets.iter().map(|s| s.to_string()).collect() });
        Ok(())
    }

    /// Fold adapters into the base weights (W += scale·B·A) and drop them, giving
    /// a plain model that any Llama runtime can load.
    pub fn merge_lora(&mut self) {
        for l in &mut self.layers {
            for lin in [&mut l.wq, &mut l.wk, &mut l.wv, &mut l.wo, &mut l.gate, &mut l.up, &mut l.down] {
                if let Some(lo) = lin.lora.take() {
                    let (out, inn, r) = (lin.w.rows(), lin.w.cols(), lo.a.rows());
                    let mut delta = vec![0.0f32; out * inn];
                    matmul_nn(lo.b.data(), lo.a.data(), &mut delta, out, r, inn, false);
                    for (w, d) in lin.w.data_mut().iter_mut().zip(&delta) {
                        *w += lo.scale * d;
                    }
                }
            }
        }
        self.lora_cfg = None;
    }

    fn head(&self) -> &Tensor {
        self.lm_head.as_ref().unwrap_or(&self.embed)
    }

    fn embed_tokens(&self, tokens: &[u32]) -> Vec<f32> {
        let d = self.dims.d;
        let mut x = vec![0.0f32; tokens.len() * d];
        for (t, &tok) in tokens.iter().enumerate() {
            assert!((tok as usize) < self.cfg.vocab_size, "token {tok} out of vocabulary");
            x[t * d..(t + 1) * d].copy_from_slice(self.embed.row(tok as usize));
        }
        x
    }

    /// Run all blocks. Returns the final hidden state plus whatever `keep` asks for.
    fn run_layers(&self, x0: Vec<f32>, t: usize, keep: Option<MemoryMode>) -> (Vec<f32>, Vec<Vec<f32>>, Vec<LayerCache>) {
        assert!(t > 0 && t <= self.max_seq, "sequence length {t} outside 1..={}", self.max_seq);
        let mut cur = x0;
        let mut inputs = Vec::new();
        let mut caches = Vec::new();
        let mut scratch = LayerCache::default();
        for layer in &self.layers {
            let mut next = Vec::new();
            if keep == Some(MemoryMode::Full) {
                let mut c = LayerCache::default();
                layer.forward(&self.dims, &self.rope, &cur, t, &mut c, &mut next);
                caches.push(c);
            } else {
                layer.forward(&self.dims, &self.rope, &cur, t, &mut scratch, &mut next);
            }
            if keep.is_some() {
                inputs.push(cur);
            }
            cur = next;
        }
        (cur, inputs, caches)
    }

    /// Forward only: logits `[t, vocab]`.
    pub fn logits(&self, tokens: &[u32]) -> Vec<f32> {
        let (t, d, v) = (tokens.len(), self.dims.d, self.cfg.vocab_size);
        let (x, _, _) = self.run_layers(self.embed_tokens(tokens), t, None);
        let mut hn = vec![0.0f32; t * d];
        let mut rstd = vec![0.0f32; t];
        rmsnorm_forward(&x, &self.final_norm, &mut hn, &mut rstd, t, d, self.dims.eps);
        let mut logits = vec![0.0f32; t * v];
        matmul_nt(&hn, self.head().data(), &mut logits, t, d, v, false);
        logits
    }

    /// Forward only: logits for the final position, `[vocab]`. Skips the
    /// `[t, vocab]` head matmul, which matters for generation with big vocabularies.
    pub fn last_logits(&self, tokens: &[u32]) -> Vec<f32> {
        let (t, d, v) = (tokens.len(), self.dims.d, self.cfg.vocab_size);
        let (x, _, _) = self.run_layers(self.embed_tokens(tokens), t, None);
        let mut hn = vec![0.0f32; d];
        let mut rstd = [0.0f32; 1];
        rmsnorm_forward(&x[(t - 1) * d..], &self.final_norm, &mut hn, &mut rstd, 1, d, self.dims.eps);
        let mut logits = vec![0.0f32; v];
        matmul_nt(&hn, self.head().data(), &mut logits, 1, d, v, false);
        logits
    }

    /// Mean cross-entropy over non-ignored targets (forward only).
    pub fn loss(&self, tokens: &[u32], targets: &[i64]) -> f32 {
        let (t, v) = (tokens.len(), self.cfg.vocab_size);
        let logits = self.logits(tokens);
        let mut dl = vec![0.0f32; t * v];
        cross_entropy_fwd_bwd(&logits, targets, &mut dl, t, v)
    }

    /// Forward + loss + backward for one sequence. `targets[i]` is the label for
    /// the output at position i (the caller shifts). Gradients are accumulated
    /// into `grads`; returns the mean loss.
    pub fn forward_backward(&self, tokens: &[u32], targets: &[i64], grads: &mut Grads, mode: MemoryMode) -> f32 {
        let (t, d, v) = (tokens.len(), self.dims.d, self.cfg.vocab_size);
        assert_eq!(targets.len(), t);

        // ---- forward ----
        let (x_last, inputs, mut caches) = self.run_layers(self.embed_tokens(tokens), t, Some(mode));
        let mut hn = vec![0.0f32; t * d];
        let mut rstd = vec![0.0f32; t];
        rmsnorm_forward(&x_last, &self.final_norm, &mut hn, &mut rstd, t, d, self.dims.eps);
        let mut dlogits = vec![0.0f32; t * v];
        let mut logits = vec![0.0f32; t * v];
        matmul_nt(&hn, self.head().data(), &mut logits, t, d, v, false);
        let loss = cross_entropy_fwd_bwd(&logits, targets, &mut dlogits, t, v);
        drop(logits);

        // ---- backward ----
        self.backward_from(tokens, loss, &dlogits, &x_last, &hn, &rstd, inputs, &mut caches, grads, mode)
    }

    #[allow(clippy::too_many_arguments)]
    fn backward_from(
        &self,
        tokens: &[u32],
        loss: f32,
        dlogits: &[f32],
        x_last: &[f32],
        hn: &[f32],
        rstd: &[f32],
        inputs: Vec<Vec<f32>>,
        caches: &mut [LayerCache],
        grads: &mut Grads,
        mode: MemoryMode,
    ) -> f32 {
        let (t, d, v) = (tokens.len(), self.dims.d, self.cfg.vocab_size);

        // LM head: logits = hn · headᵀ
        let head = self.head();
        let mut dhn = vec![0.0f32; t * d];
        matmul_nn(dlogits, head.data(), &mut dhn, t, v, d, false);
        let head_grad = if self.lm_head.is_some() { &mut grads.lm_head } else { &mut grads.embed };
        if !head_grad.is_empty() {
            matmul_tn(dlogits, hn, head_grad, v, t, d, true);
        }

        // Final norm
        let mut dx = vec![0.0f32; t * d];
        let mut scratch = vec![0.0f32; d];
        let dw = if grads.final_norm.is_empty() { &mut scratch } else { &mut grads.final_norm };
        rmsnorm_backward(&dhn, x_last, &self.final_norm, rstd, &mut dx, dw, t, d);
        drop(dhn);

        // Blocks, last to first
        for (i, layer) in self.layers.iter().enumerate().rev() {
            let lg = &mut grads.layers[i];
            dx = match mode {
                MemoryMode::Full => layer.backward(&self.dims, &self.rope, &inputs[i], &dx, t, &caches[i], lg),
                MemoryMode::Checkpoint => {
                    let mut c = LayerCache::default();
                    let mut out = Vec::new();
                    layer.forward(&self.dims, &self.rope, &inputs[i], t, &mut c, &mut out);
                    layer.backward(&self.dims, &self.rope, &inputs[i], &dx, t, &c, lg)
                }
            };
            if mode == MemoryMode::Full {
                caches[i] = LayerCache::default(); // free as we go
            }
        }

        // Embedding scatter
        if !grads.embed.is_empty() {
            for (pos, &tok) in tokens.iter().enumerate() {
                let dst = &mut grads.embed[tok as usize * d..(tok as usize + 1) * d];
                for (a, b) in dst.iter_mut().zip(&dx[pos * d..(pos + 1) * d]) {
                    *a += *b;
                }
            }
        }
        loss
    }
}
