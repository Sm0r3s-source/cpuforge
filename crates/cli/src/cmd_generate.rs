//! `cpuforge generate --model DIR [--adapter FILE] --prompt TEXT [--tokens N] [--temp T] [--top-k K] [--tokenizer PATH|bytes]`
//!
//! Recomputes the whole sequence for every new token (no KV cache yet), which is
//! fine for short completions and checking that a finetune did what you wanted.

use std::io::Write;
use std::path::{Path, PathBuf};

use cf_core::rng::Rng;
use cf_io::{load_adapter, load_model};
use cf_model::Model;
use cf_train::Tokenizer;

use crate::args::Args;
use crate::util::{load_tokenizer, sample};

/// Generate up to `max_new` tokens after `prompt`, calling `on_token` with each new id.
#[allow(clippy::too_many_arguments)]
pub fn generate(
    model: &Model,
    tok: &dyn Tokenizer,
    prompt: &str,
    max_new: usize,
    temp: f32,
    top_k: usize,
    seed: u64,
    on_token: &mut dyn FnMut(u32),
) -> Result<Vec<u32>, String> {
    let mut ids = tok.encode(prompt)?;
    if ids.is_empty() {
        return Err("prompt encodes to zero tokens".into());
    }
    let mut rng = Rng::new(seed);
    let mut new = Vec::new();
    for _ in 0..max_new {
        if ids.len() >= model.max_seq {
            break;
        }
        let logits = model.last_logits(&ids);
        let next = sample(&logits, temp, top_k, &mut rng);
        if Some(next) == tok.eos_id() {
            break;
        }
        ids.push(next);
        new.push(next);
        on_token(next);
    }
    Ok(new)
}

pub fn run(args: &Args) -> Result<(), String> {
    let dir = PathBuf::from(args.require("model")?);
    let prompt = args.require("prompt")?.to_string();
    let max_new: usize = args.parsed("tokens", 64)?;
    let temp: f32 = args.parsed("temp", 0.8)?;
    let top_k: usize = args.parsed("top-k", 40)?;
    let seed: u64 = args.parsed("seed", 1)?;

    let tok_spec = args.get("tokenizer").map(String::from).unwrap_or_else(|| dir.join("tokenizer.json").to_string_lossy().into_owned());
    let tok = load_tokenizer(&tok_spec)?;

    let prompt_len = tok.encode(&prompt)?.len();
    let mut model = load_model(&dir, prompt_len + max_new + 1)?;
    if let Some(a) = args.get("adapter") {
        load_adapter(&mut model, Path::new(a))?;
    }

    print!("{prompt}");
    std::io::stdout().flush().ok();
    let mut all = Vec::new();
    let mut printed = String::new();
    generate(&model, tok.as_ref(), &prompt, max_new, temp, top_k, seed, &mut |t| {
        all.push(t);
        // Decode everything so far so multi-token characters come out right.
        if let Ok(text) = tok.decode(&all) {
            if let Some(tail) = text.strip_prefix(&printed) {
                print!("{tail}");
                std::io::stdout().flush().ok();
            }
            printed = text;
        }
    })?;
    println!();
    Ok(())
}
