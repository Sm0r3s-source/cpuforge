//! `cpuforge bench`        matmul kernels at SmolLM-135M-like shapes
//! `cpuforge bench-train`  full train-step speed and memory on a random SmolLM-135M-shaped model

use std::time::Instant;

use cf_core::matmul::{matmul_nn, matmul_nt, matmul_tn};
use cf_core::rng::Rng;
use cf_model::{Grads, LlamaConfig, MemoryMode, Model, TrainSpec};

use crate::args::Args;
use crate::util::peak_rss_mb;

fn time_gflops(label: &str, flops: f64, mut f: impl FnMut()) {
    f(); // warm-up
    let mut best = f64::MAX;
    for _ in 0..5 {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64());
    }
    println!("{label:<36} {:>8.2} ms   {:>7.2} GFLOP/s", best * 1e3, flops / best / 1e9);
}

pub fn run_matmul() {
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    println!("threads: {threads}");

    // 512 tokens, hidden 576, FFN 1536.
    let (t, d, ffn) = (512usize, 576usize, 1536usize);
    let mut rng = Rng::new(1);
    let mut rand = |n: usize| -> Vec<f32> { (0..n).map(|_| rng.normal()).collect() };
    let (x, w, dy) = (rand(t * d), rand(ffn * d), rand(t * ffn));
    let flops = 2.0 * (t * d * ffn) as f64;

    let mut y = vec![0.0; t * ffn];
    time_gflops("forward   x·Wᵀ   [512x576]·[576x1536]", flops, || matmul_nt(&x, &w, &mut y, t, d, ffn, false));
    let mut dx = vec![0.0; t * d];
    time_gflops("input grad dy·W  [512x1536]·[1536x576]", flops, || matmul_nn(&dy, &w, &mut dx, t, ffn, d, false));
    let mut dw = vec![0.0; ffn * d];
    time_gflops("weight grad dyᵀ·x [1536x512]·[512x576]", flops, || matmul_tn(&dy, &x, &mut dw, ffn, t, d, false));
}

pub fn smollm_135m_config(layers: usize) -> LlamaConfig {
    LlamaConfig::from_json(&format!(
        r#"{{"vocab_size":49152,"hidden_size":576,"intermediate_size":1536,"num_hidden_layers":{layers},
            "num_attention_heads":9,"num_key_value_heads":3,"rms_norm_eps":1e-5,"rope_theta":10000.0,
            "max_position_embeddings":2048,"tie_word_embeddings":true}}"#
    ))
    .expect("valid config")
}

pub fn run_train(args: &Args) -> Result<(), String> {
    let layers: usize = args.parsed("layers", 30)?;
    let seq: usize = args.parsed("seq", 512)?;
    let rank: usize = args.parsed("rank", 16)?;
    let steps: usize = args.parsed("steps", 2)?;

    let cfg = smollm_135m_config(layers);
    println!("model: SmolLM-135M shape, {layers} layers, {:.1}M params, random weights", cfg.num_params() as f64 / 1e6);
    println!("sequence length {seq}, LoRA rank {rank} on all projections\n");

    let started = Instant::now();
    let mut model = Model::random(cfg.clone(), seq, 1, 0.02);
    println!("built in {:.1}s", started.elapsed().as_secs_f32());

    let mut rng = Rng::new(7);
    let v = cfg.vocab_size;
    let tokens: Vec<u32> = (0..seq).map(|_| rng.below(v) as u32).collect();
    let targets: Vec<i64> = (0..seq).map(|_| rng.below(v) as i64).collect();

    // Inference
    let t = Instant::now();
    model.last_logits(&tokens);
    let secs = t.elapsed().as_secs_f32();
    println!("forward only (prompt processing): {:.0} tok/s\n", seq as f32 / secs);

    let bench = |label: &str, model: &Model, spec: TrainSpec, mode: MemoryMode| {
        let mut grads = Grads::new(model, spec);
        model.forward_backward(&tokens, &targets, &mut grads, mode); // warm-up
        let t = Instant::now();
        for _ in 0..steps {
            grads.zero();
            model.forward_backward(&tokens, &targets, &mut grads, mode);
        }
        let per = t.elapsed().as_secs_f32() / steps as f32;
        println!("{label:<34} {:>6.2} s/step   {:>6.0} tok/s", per, seq as f32 / per);
    };

    model.add_lora(rank, 2.0 * rank as f32, &["q", "k", "v", "o", "gate", "up", "down"], 3)?;
    bench("LoRA, all activations kept", &model, TrainSpec::LORA_ONLY, MemoryMode::Full);
    bench("LoRA, gradient checkpointing", &model, TrainSpec::LORA_ONLY, MemoryMode::Checkpoint);
    model.merge_lora();
    bench("full finetune, all activations kept", &model, TrainSpec { base: true, embed: true, lora: false }, MemoryMode::Full);

    if let Some(mb) = peak_rss_mb() {
        println!("\npeak memory {mb:.0} MB");
    }
    Ok(())
}
