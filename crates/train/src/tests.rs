//! Does training actually train? Small byte-level models on a learnable pattern.

#![allow(clippy::err_expect)]
use cf_model::{visit_params, LlamaConfig, MemoryMode, Model, TrainSpec};

use crate::*;

fn cfg() -> LlamaConfig {
    LlamaConfig::from_json(
        r#"{"vocab_size":256,"hidden_size":32,"intermediate_size":64,"num_hidden_layers":2,
            "num_attention_heads":4,"num_key_value_heads":2,"max_position_embeddings":64,
            "tie_word_embeddings":true}"#,
    )
    .unwrap()
}

fn data() -> Dataset {
    let text = "the quick brown fox jumps over the lazy dog. ".repeat(40);
    Dataset::from_stream(&ByteTokenizer.encode(&text).unwrap(), 32)
}

fn train_cfg(steps: usize) -> TrainConfig {
    TrainConfig { steps, batch: 2, lr: 3e-3, warmup_steps: 3, log_every: 1, ..Default::default() }
}

fn params(m: &mut Model) -> Vec<f32> {
    let mut all = Vec::new();
    visit_params(m, None, &mut |_, _, p, _| all.extend_from_slice(p));
    all
}

fn run(trainer: &mut Trainer, d: &Dataset) -> Vec<f32> {
    let mut losses = Vec::new();
    trainer
        .train(d, None, None, &mut |e| {
            if let Event::Step(s) = e {
                losses.push(s.loss);
            }
        })
        .unwrap();
    losses
}

#[test]
fn full_finetune_learns_the_pattern() {
    let d = data();
    let model = Model::random(cfg(), 64, 1, 0.1);
    let mut t = Trainer::new(model, TrainSpec::FULL, train_cfg(60));
    let losses = run(&mut t, &d);
    let (first, last) = (losses[0], *losses.last().unwrap());
    assert!(first > 4.5, "should start near ln(256)=5.5, got {first}");
    assert!(last < 0.5 * first, "loss {first} -> {last}");
    assert!(losses.iter().all(|l| l.is_finite()));
}

#[test]
fn lora_learns_and_never_touches_the_base() {
    let d = data();
    let mut model = Model::random(cfg(), 64, 2, 0.1);
    model.add_lora(8, 16.0, &["q", "k", "v", "o", "gate", "up", "down"], 5).unwrap();
    let mut base_before = Vec::new();
    visit_params(&mut model, None, &mut |n, _, p, _| {
        if !n.contains(".lora_") {
            base_before.extend_from_slice(p);
        }
    });
    let mut t = Trainer::new(model, TrainSpec::LORA_ONLY, TrainConfig { lr: 1e-2, ..train_cfg(80) });
    let losses = run(&mut t, &d);
    assert!(*losses.last().unwrap() < 0.85 * losses[0], "loss {} -> {}", losses[0], losses.last().unwrap());

    let mut base_after = Vec::new();
    visit_params(&mut t.model, None, &mut |n, _, p, _| {
        if !n.contains(".lora_") {
            base_after.extend_from_slice(p);
        }
    });
    assert_eq!(base_before, base_after, "frozen weights changed");
}

#[test]
fn checkpoint_memory_mode_trains_identically() {
    let d = data();
    let mut a = Trainer::new(Model::random(cfg(), 64, 3, 0.1), TrainSpec::FULL, train_cfg(5));
    let mut b = Trainer::new(
        Model::random(cfg(), 64, 3, 0.1),
        TrainSpec::FULL,
        TrainConfig { memory: MemoryMode::Checkpoint, ..train_cfg(5) },
    );
    let la = run(&mut a, &d);
    let lb = run(&mut b, &d);
    for (x, y) in la.iter().zip(&lb) {
        assert!((x - y).abs() < 1e-4, "{x} vs {y}");
    }
}

#[test]
fn resume_continues_exactly_where_it_stopped() {
    let d = data();
    let dir = tempfile::tempdir().unwrap();
    let state = dir.path().join("state.safetensors");
    let mk = || {
        let mut m = Model::random(cfg(), 64, 4, 0.1);
        m.add_lora(4, 8.0, &["q", "v", "down"], 9).unwrap();
        m
    };

    // Straight through: 8 steps.
    let mut straight = Trainer::new(mk(), TrainSpec::LORA_ONLY, train_cfg(8));
    let straight_losses = run(&mut straight, &d);

    // 4 steps, save, then a brand-new trainer (model without adapters) resumes to 8.
    let mut first = Trainer::new(mk(), TrainSpec::LORA_ONLY, train_cfg(4));
    run(&mut first, &d);
    first.save(&state).unwrap();

    let fresh_base = Model::random(cfg(), 64, 4, 0.1); // same base weights, no adapter yet
    let mut second = Trainer::new(fresh_base, TrainSpec::LORA_ONLY, train_cfg(8));
    // Grads were sized before the adapter existed, so build the trainer after attaching it:
    second.model.add_lora(4, 8.0, &["q", "v", "down"], 123).unwrap();
    second.grads = cf_model::Grads::new(&second.model, TrainSpec::LORA_ONLY);
    second.resume(&state).unwrap();
    assert_eq!(second.step, 4);
    let resumed_losses = run(&mut second, &d);

    assert_eq!(resumed_losses.len(), 4);
    for (x, y) in straight_losses[4..].iter().zip(&resumed_losses) {
        assert!((x - y).abs() < 1e-5, "{x} vs {y}");
    }
    let (pa, pb) = (params(&mut straight.model), params(&mut second.model));
    let worst = pa.iter().zip(&pb).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(worst < 1e-5, "parameters diverged by {worst}");
}

#[test]
fn resume_rejects_a_state_from_a_different_setup() {
    let d = data();
    let dir = tempfile::tempdir().unwrap();
    let state = dir.path().join("s.safetensors");
    let mut m = Model::random(cfg(), 64, 5, 0.1);
    m.add_lora(4, 8.0, &["q"], 1).unwrap();
    let mut t = Trainer::new(m, TrainSpec::LORA_ONLY, train_cfg(2));
    run(&mut t, &d);
    t.save(&state).unwrap();

    let mut other = Model::random(cfg(), 64, 5, 0.1);
    other.add_lora(8, 8.0, &["q"], 1).unwrap(); // different rank
    let mut t2 = Trainer::new(other, TrainSpec::LORA_ONLY, train_cfg(4));
    assert!(t2.resume(&state).err().expect("rank mismatch").contains("differs"));
}

#[test]
fn saved_state_includes_optimizer_moments_and_progress() {
    let d = data();
    let dir = tempfile::tempdir().unwrap();
    let state = dir.path().join("s.safetensors");
    let mut t = Trainer::new(Model::random(cfg(), 64, 6, 0.1), TrainSpec::FULL, train_cfg(3));
    run(&mut t, &d);
    t.save(&state).unwrap();
    let loaded = cf_io::read_safetensors(&state).unwrap();
    assert_eq!(loaded.metadata["step"], "3");
    assert_eq!(loaded.metadata["consumed"], "6"); // 3 steps x batch 2
    assert!(loaded.tensors.contains_key("p.model.embed_tokens.weight"));
    assert!(loaded.tensors.contains_key("m.model.layers.0.mlp.up_proj.weight"));
    assert!(loaded.tensors.contains_key("v.model.norm.weight"));
}

#[test]
fn evaluation_and_events_fire_on_schedule() {
    let mut d = data();
    let val = d.split_off_val(0.2);
    let dir = tempfile::tempdir().unwrap();
    let state = dir.path().join("s.safetensors");
    let mut t = Trainer::new(
        Model::random(cfg(), 64, 7, 0.1),
        TrainSpec::FULL,
        TrainConfig { eval_every: 2, save_every: 3, log_every: 2, ..train_cfg(6) },
    );
    let (mut steps, mut evals, mut saves) = (Vec::new(), Vec::new(), Vec::new());
    t.train(&d, Some(&val), Some(&state), &mut |e| match e {
        Event::Step(s) => steps.push(s.step),
        Event::Eval { step, val_loss } => {
            assert!(val_loss.is_finite());
            evals.push(*step)
        }
        Event::Saved { step, .. } => saves.push(*step),
    })
    .unwrap();
    assert_eq!(steps, vec![2, 4, 6]);
    assert_eq!(evals, vec![2, 4, 6]);
    assert_eq!(saves, vec![3, 6]);
    assert!(state.exists());
}

#[test]
fn empty_dataset_is_an_error() {
    let mut t = Trainer::new(Model::random(cfg(), 64, 8, 0.1), TrainSpec::FULL, train_cfg(1));
    assert!(t.train(&Dataset::default(), None, None, &mut |_| {}).is_err());
}
