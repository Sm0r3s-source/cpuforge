//! End-to-end checks on a tiny random model: the whole backward pass
//! (embedding -> blocks -> norm -> head -> loss) against finite differences.

#![allow(clippy::err_expect)]
use cf_core::ops::loss::IGNORE_INDEX;
use cf_core::rng::Rng;

use crate::*;

fn tiny_cfg(tie: bool) -> LlamaConfig {
    LlamaConfig::from_json(&format!(
        r#"{{"vocab_size":13,"hidden_size":16,"intermediate_size":24,"num_hidden_layers":2,
            "num_attention_heads":4,"num_key_value_heads":2,"rms_norm_eps":1e-5,
            "rope_theta":10000.0,"max_position_embeddings":32,"tie_word_embeddings":{tie}}}"#
    ))
    .unwrap()
}

const TOKENS: [u32; 8] = [3, 1, 4, 1, 5, 9, 2, 6];
const TARGETS: [i64; 8] = [1, 4, 1, 5, IGNORE_INDEX, 2, 6, 3];

fn model(tie: bool, seed: u64) -> Model {
    let mut m = Model::random(tiny_cfg(tie), 32, seed, 0.3);
    // Non-trivial norm weights so their gradients are exercised.
    let mut rng = Rng::new(seed + 1);
    for l in &mut m.layers {
        l.attn_norm.iter_mut().for_each(|w| *w = 1.0 + 0.2 * rng.normal());
        l.mlp_norm.iter_mut().for_each(|w| *w = 1.0 + 0.2 * rng.normal());
    }
    m.final_norm.iter_mut().for_each(|w| *w = 1.0 + 0.2 * rng.normal());
    m
}

fn all_lora(m: &mut Model, seed: u64) {
    m.add_lora(3, 6.0, &["q", "k", "v", "o", "gate", "up", "down"], seed).unwrap();
    // B starts at zero; make it non-zero so the adapters actually matter.
    let mut rng = Rng::new(seed + 7);
    for l in &mut m.layers {
        for lin in [&mut l.wq, &mut l.wk, &mut l.wv, &mut l.wo, &mut l.gate, &mut l.up, &mut l.down] {
            if let Some(lo) = &mut lin.lora {
                lo.b.data_mut().iter_mut().for_each(|v| *v = rng.normal() * 0.2);
            }
        }
    }
}

/// Compare analytic gradients against central differences of the f32 loss at a
/// sample of elements from every trained tensor.
fn check_against_finite_differences(m: &mut Model, spec: TrainSpec, label: &str) {
    let mut grads = Grads::new(m, spec);
    let loss = m.forward_backward(&TOKENS, &TARGETS, &mut grads, MemoryMode::Full);
    assert!(loss.is_finite() && loss > 0.0, "{label}: loss {loss}");

    // Collect (name, len, gradient) of everything trained.
    let mut trained: Vec<(String, Vec<f32>)> = Vec::new();
    visit_params(m, Some(&grads), &mut |name, _s, _p, g| {
        if !g.is_empty() {
            trained.push((name.to_string(), g.to_vec()));
        }
    });
    assert!(!trained.is_empty(), "{label}: nothing trained");

    let eps = 2e-3f32;
    let mut rng = Rng::new(99);
    let mut checked = 0;
    for (name, g) in &trained {
        for _ in 0..6 {
            let idx = rng.below(g.len());
            let eval = |delta: f32, m: &mut Model| -> f32 {
                let mut hit = false;
                visit_params(m, None, &mut |n, _s, p, _| {
                    if n == name {
                        p[idx] += delta;
                        hit = true;
                    }
                });
                assert!(hit);
                let l = m.loss(&TOKENS, &TARGETS);
                visit_params(m, None, &mut |n, _s, p, _| {
                    if n == name {
                        p[idx] -= delta;
                    }
                });
                l
            };
            let hi = eval(eps, m);
            let lo = eval(-eps, m);
            let numeric = (hi - lo) / (2.0 * eps);
            let analytic = g[idx];
            let err = (analytic - numeric).abs();
            assert!(
                err <= 2.5e-3 + 3e-2 * numeric.abs(),
                "{label}: {name}[{idx}] analytic {analytic} vs numeric {numeric}"
            );
            checked += 1;
        }
    }
    assert!(checked > 20);
}

#[test]
fn full_finetune_gradients_untied() {
    let mut m = model(false, 1);
    check_against_finite_differences(&mut m, TrainSpec { base: true, embed: true, lora: false }, "full/untied");
}

#[test]
fn full_finetune_gradients_tied_embeddings() {
    let mut m = model(true, 2);
    check_against_finite_differences(&mut m, TrainSpec { base: true, embed: true, lora: false }, "full/tied");
}

#[test]
fn lora_gradients_with_frozen_base() {
    let mut m = model(false, 3);
    all_lora(&mut m, 30);
    check_against_finite_differences(&mut m, TrainSpec::LORA_ONLY, "lora-only");
}

#[test]
fn everything_trainable_together() {
    let mut m = model(true, 4);
    all_lora(&mut m, 40);
    check_against_finite_differences(&mut m, TrainSpec::FULL, "full+lora");
}

#[test]
fn lora_only_leaves_base_gradients_empty() {
    let mut m = model(false, 5);
    all_lora(&mut m, 50);
    let mut g = Grads::new(&m, TrainSpec::LORA_ONLY);
    assert!(g.embed.is_empty() && g.lm_head.is_empty() && g.final_norm.is_empty());
    assert!(g.layers[0].wq.w.is_empty() && !g.layers[0].wq.a.is_empty() && !g.layers[0].wq.b.is_empty());
    m.forward_backward(&TOKENS, &TARGETS, &mut g, MemoryMode::Full);
    assert!(g.global_norm() > 0.0);
}

#[test]
fn lora_with_zero_b_does_not_change_the_model() {
    let mut m = model(false, 6);
    let before = m.logits(&TOKENS);
    m.add_lora(4, 8.0, &["q", "v", "down"], 60).unwrap();
    let after = m.logits(&TOKENS);
    assert_eq!(before, after);
}

#[test]
fn checkpoint_mode_gives_identical_gradients() {
    let mut m = model(false, 7);
    all_lora(&mut m, 70);
    let mut full = Grads::new(&m, TrainSpec::FULL);
    let mut ckpt = Grads::new(&m, TrainSpec::FULL);
    let l1 = m.forward_backward(&TOKENS, &TARGETS, &mut full, MemoryMode::Full);
    let l2 = m.forward_backward(&TOKENS, &TARGETS, &mut ckpt, MemoryMode::Checkpoint);
    assert_eq!(l1, l2);
    let (mut a, mut b) = (Vec::new(), Vec::new());
    visit_params(&mut m, Some(&full), &mut |_, _, _, g| a.extend_from_slice(g));
    visit_params(&mut m, Some(&ckpt), &mut |_, _, _, g| b.extend_from_slice(g));
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(&b) {
        assert!((x - y).abs() <= 1e-6 * (1.0 + x.abs()), "{x} vs {y}");
    }
}

#[test]
fn gradients_accumulate_across_calls() {
    let m = model(false, 8);
    let mut g1 = Grads::new(&m, TrainSpec::FULL);
    m.forward_backward(&TOKENS, &TARGETS, &mut g1, MemoryMode::Full);
    let n1 = g1.global_norm();
    m.forward_backward(&TOKENS, &TARGETS, &mut g1, MemoryMode::Full);
    let n2 = g1.global_norm();
    assert!((n2 / n1 - 2.0).abs() < 1e-4, "{n1} -> {n2}");
}

#[test]
fn model_is_causal() {
    let m = model(false, 9);
    let v = m.cfg.vocab_size;
    let a = m.logits(&[3, 1, 4, 1, 5, 9]);
    let b = m.logits(&[3, 1, 4, 7, 7, 7]); // differs from position 3 on
    for i in 0..3 * v {
        assert!((a[i] - b[i]).abs() < 1e-5, "logit {i} leaked future information");
    }
    assert!((a[3 * v] - b[3 * v]).abs() > 1e-4);
}

#[test]
fn sgd_on_the_gradients_reduces_the_loss() {
    let mut m = model(false, 10);
    let start = m.loss(&TOKENS, &TARGETS);
    for _ in 0..40 {
        let mut g = Grads::new(&m, TrainSpec::FULL);
        m.forward_backward(&TOKENS, &TARGETS, &mut g, MemoryMode::Full);
        visit_params(&mut m, Some(&g), &mut |_, _, p, g| {
            for (w, d) in p.iter_mut().zip(g) {
                *w -= 0.1 * d;
            }
        });
    }
    let end = m.loss(&TOKENS, &TARGETS);
    assert!(end < start * 0.5, "loss {start} -> {end}");
}

#[test]
fn from_named_roundtrip_and_validation() {
    use std::collections::HashMap;
    let mut m = model(false, 11);
    let mut map = HashMap::new();
    let mut shapes = HashMap::new();
    shapes.insert("model.embed_tokens.weight".to_string(), vec![13, 16]);
    visit_params(&mut m, None, &mut |name, _, p, _| {
        map.insert(name.to_string(), p.to_vec());
    });
    let cfg = m.cfg.clone();
    let build = |map: &HashMap<String, Vec<f32>>| -> HashMap<String, cf_core::Tensor> {
        let (d, qd, kvd, f, v) = (16, 16, 8, 24, 13);
        map.iter()
            .map(|(k, v_)| {
                let shape: Vec<usize> = if k.ends_with("embed_tokens.weight") || k == "lm_head.weight" {
                    vec![v, d]
                } else if k.contains("layernorm") || k == "model.norm.weight" {
                    vec![d]
                } else if k.contains("q_proj") {
                    vec![qd, d]
                } else if k.contains("k_proj") || k.contains("v_proj") {
                    vec![kvd, d]
                } else if k.contains("o_proj") {
                    vec![d, qd]
                } else if k.contains("down_proj") {
                    vec![d, f]
                } else {
                    vec![f, d]
                };
                (k.clone(), cf_core::Tensor::from_vec(v_.clone(), &shape))
            })
            .collect()
    };
    let rebuilt = Model::from_named(cfg.clone(), 32, build(&map)).unwrap();
    assert_eq!(m.logits(&TOKENS), rebuilt.logits(&TOKENS));

    // Missing tensor
    let mut broken = build(&map);
    broken.remove("model.norm.weight");
    assert!(Model::from_named(cfg.clone(), 32, broken).err().expect("should fail").contains("model.norm.weight"));

    // Wrong shape
    let mut broken = build(&map);
    broken.insert("model.norm.weight".into(), cf_core::Tensor::zeros(&[5]));
    assert!(Model::from_named(cfg.clone(), 32, broken).err().expect("should fail").contains("shape"));

    // Bias tensors must be refused rather than silently dropped
    let mut with_bias = build(&map);
    with_bias.insert("model.layers.0.self_attn.q_proj.bias".into(), cf_core::Tensor::zeros(&[16]));
    assert!(Model::from_named(cfg, 32, with_bias).err().expect("should fail").contains("bias"));
}

#[test]
fn merging_lora_preserves_the_function() {
    let mut m = model(false, 12);
    all_lora(&mut m, 80);
    let before = m.logits(&TOKENS);
    m.merge_lora();
    assert!(m.lora_cfg.is_none() && m.layers[0].wq.lora.is_none());
    let after = m.logits(&TOKENS);
    for (a, b) in before.iter().zip(&after) {
        assert!((a - b).abs() < 1e-4, "{a} vs {b}");
    }
}

#[test]
fn second_add_lora_is_rejected() {
    let mut m = model(false, 13);
    m.add_lora(2, 4.0, &["q"], 1).unwrap();
    assert!(m.add_lora(2, 4.0, &["v"], 2).is_err());
    assert!(model(false, 14).add_lora(2, 4.0, &["nope"], 1).is_err());
}

#[test]
fn last_logits_equals_the_last_row_of_full_logits() {
    let m = model(false, 15);
    let v = m.cfg.vocab_size;
    let full = m.logits(&TOKENS);
    let last = m.last_logits(&TOKENS);
    assert_eq!(last.len(), v);
    for (a, b) in last.iter().zip(&full[(TOKENS.len() - 1) * v..]) {
        assert!((a - b).abs() < 1e-5);
    }
}

// ---- Qwen3 (per-head q/k norm, head_dim != hidden/heads) and Llama-3 rope scaling ----

fn qwen_cfg(extra: &str) -> LlamaConfig {
    LlamaConfig::from_json(&format!(
        r#"{{"model_type":"qwen3","vocab_size":13,"hidden_size":16,"intermediate_size":24,"num_hidden_layers":2,
            "num_attention_heads":4,"num_key_value_heads":2,"head_dim":6,"rms_norm_eps":1e-6,
            "rope_theta":1000000.0,"max_position_embeddings":32,"tie_word_embeddings":true{extra}}}"#
    ))
    .unwrap()
}

fn qwen_model(seed: u64, extra: &str) -> Model {
    let mut m = Model::random(qwen_cfg(extra), 32, seed, 0.3);
    let mut rng = Rng::new(seed + 1);
    for l in &mut m.layers {
        l.attn_norm.iter_mut().for_each(|w| *w = 1.0 + 0.2 * rng.normal());
        l.mlp_norm.iter_mut().for_each(|w| *w = 1.0 + 0.2 * rng.normal());
        for w in [l.q_norm.as_mut().unwrap(), l.k_norm.as_mut().unwrap()] {
            w.iter_mut().for_each(|x| *x = 1.0 + 0.3 * rng.normal());
        }
    }
    m.final_norm.iter_mut().for_each(|w| *w = 1.0 + 0.2 * rng.normal());
    m
}

#[test]
fn qwen3_shapes_follow_head_dim_not_hidden_over_heads() {
    let m = qwen_model(1, "");
    assert_eq!((m.dims.q_dim(), m.dims.kv_dim(), m.dims.d), (24, 12, 16));
    assert_eq!(m.layers[0].wq.w.shape(), &[24, 16]);
    assert_eq!(m.layers[0].wo.w.shape(), &[16, 24]);
    assert_eq!(m.layers[0].q_norm.as_ref().unwrap().len(), 6);
}

#[test]
fn qwen3_full_finetune_gradients_including_qk_norm_weights() {
    let mut m = qwen_model(2, "");
    check_against_finite_differences(&mut m, TrainSpec { base: true, embed: true, lora: false }, "qwen3/full");
    // The check only visits tensors with gradients, so make sure the qk norms were among them.
    let g = Grads::new(&m, TrainSpec { base: true, embed: true, lora: false });
    assert_eq!((g.layers[0].q_norm.len(), g.layers[0].k_norm.len()), (6, 6));
}

#[test]
fn qwen3_lora_gradients_pass_through_the_norms_and_leave_them_frozen() {
    let mut m = qwen_model(3, "");
    all_lora(&mut m, 30);
    check_against_finite_differences(&mut m, TrainSpec::LORA_ONLY, "qwen3/lora");
    let g = Grads::new(&m, TrainSpec::LORA_ONLY);
    assert!(g.layers[0].q_norm.is_empty() && g.layers[0].k_norm.is_empty());
}

#[test]
fn qwen3_checkpoint_mode_gives_identical_gradients() {
    let mut m = qwen_model(4, "");
    all_lora(&mut m, 40);
    let mut full = Grads::new(&m, TrainSpec::FULL);
    let mut ckpt = Grads::new(&m, TrainSpec::FULL);
    let l1 = m.forward_backward(&TOKENS, &TARGETS, &mut full, MemoryMode::Full);
    let l2 = m.forward_backward(&TOKENS, &TARGETS, &mut ckpt, MemoryMode::Checkpoint);
    assert_eq!(l1, l2);
    let (mut a, mut b) = (Vec::new(), Vec::new());
    visit_params(&mut m, Some(&full), &mut |_, _, _, g| a.extend_from_slice(g));
    visit_params(&mut m, Some(&ckpt), &mut |_, _, _, g| b.extend_from_slice(g));
    assert_eq!(a.len(), b.len());
    assert!(a.iter().zip(&b).all(|(x, y)| (x - y).abs() <= 1e-6 * (1.0 + x.abs())));
}

#[test]
fn qk_norm_changes_the_function() {
    // Same weights, once with the norms as identity-ish (weights 1) and once perturbed: outputs must differ.
    let a = qwen_model(5, "");
    let mut b = qwen_model(5, "");
    b.layers[0].q_norm.as_mut().unwrap()[0] += 0.5;
    assert_ne!(a.logits(&TOKENS), b.logits(&TOKENS));
}

#[test]
fn qwen3_from_named_requires_the_norm_tensors_and_roundtrips() {
    use std::collections::HashMap;
    let mut m = qwen_model(6, "");
    let mut map: HashMap<String, cf_core::Tensor> = HashMap::new();
    visit_params(&mut m, None, &mut |name, shape, p, _| {
        map.insert(name.to_string(), cf_core::Tensor::from_vec(p.to_vec(), shape));
    });
    assert!(map.contains_key("model.layers.1.self_attn.q_norm.weight") && map.contains_key("model.layers.1.self_attn.k_norm.weight"));
    let rebuilt = Model::from_named(m.cfg.clone(), 32, map.clone()).unwrap();
    assert_eq!(m.logits(&TOKENS), rebuilt.logits(&TOKENS));
    map.remove("model.layers.0.self_attn.k_norm.weight");
    assert!(Model::from_named(m.cfg.clone(), 32, map).err().expect("should fail").contains("k_norm"));
}

#[test]
fn llama3_rope_scaling_trains_correctly_and_differs_from_plain_rope() {
    let scaling = r#","rope_scaling":{"rope_type":"llama3","factor":8.0,"low_freq_factor":1.0,"high_freq_factor":4.0,"original_max_position_embeddings":16}"#;
    let mut scaled = qwen_model(7, scaling);
    check_against_finite_differences(&mut scaled, TrainSpec { base: true, embed: true, lora: false }, "llama3-rope");
    let plain = qwen_model(7, "");
    assert_ne!(plain.logits(&TOKENS), scaled.logits(&TOKENS));
}
