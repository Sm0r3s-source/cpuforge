//! `cpuforge init --out DIR [--preset tiny|small] [--seed N]`
//!
//! Writes a randomly initialised byte-level Llama (vocab 256) as a Hugging Face
//! directory, ready for from-scratch training with `train` (set
//! `tokenizer = "bytes"` and `mode = "full"`).

use std::path::Path;

use cf_io::save_model;
use cf_model::{LlamaConfig, Model};

use crate::args::Args;
use crate::cmd_train::fmt_count;

pub fn preset(name: &str) -> Result<LlamaConfig, String> {
    let (hidden, ffn, layers, heads, kv) = match name {
        "tiny" => (128, 352, 4, 4, 2),
        "small" => (256, 704, 6, 8, 4),
        other => return Err(format!("unknown preset '{other}' (try tiny or small)")),
    };
    LlamaConfig::from_json(&format!(
        r#"{{"vocab_size":256,"hidden_size":{hidden},"intermediate_size":{ffn},"num_hidden_layers":{layers},
            "num_attention_heads":{heads},"num_key_value_heads":{kv},"rms_norm_eps":1e-5,
            "rope_theta":10000.0,"max_position_embeddings":1024,"tie_word_embeddings":true}}"#
    ))
}

pub fn run(args: &Args) -> Result<(), String> {
    let out = Path::new(args.require("out")?);
    let cfg = preset(args.get("preset").unwrap_or("tiny"))?;
    let seed: u64 = args.parsed("seed", 1)?;
    let mut model = Model::random(cfg.clone(), 16, seed, 0.05);
    save_model(out, &mut model)?;
    println!("wrote a {} parameter model to {}", fmt_count(cfg.num_params()), out.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_exist_and_unknown_is_rejected() {
        assert!(preset("tiny").unwrap().num_params() < 1_000_000);
        assert!(preset("small").unwrap().num_params() > 3_000_000);
        assert!(preset("huge").is_err());
    }
}
