//! `cpuforge merge --model DIR --adapter FILE --out DIR`
//!
//! Bakes a LoRA adapter into the base weights and writes a plain Hugging Face
//! directory (tokenizer files are copied along) that other runtimes can load.

use std::path::{Path, PathBuf};

use cf_io::{load_model, save_model};

use crate::args::Args;
use crate::util::load_any_adapter;

const SIDE_FILES: &[&str] = &["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json"];

pub fn run(args: &Args) -> Result<(), String> {
    let model_dir = PathBuf::from(args.require("model")?);
    let adapter = args.require("adapter")?;
    let out = PathBuf::from(args.require("out")?);

    let mut model = load_model(&model_dir, 16)?;
    load_any_adapter(&mut model, Path::new(adapter))?;
    model.merge_lora();
    save_model(&out, &mut model)?;

    let mut copied = 0;
    for f in SIDE_FILES {
        let src = model_dir.join(f);
        if src.exists() {
            std::fs::copy(&src, out.join(f)).map_err(|e| format!("cannot copy {f}: {e}"))?;
            copied += 1;
        }
    }
    println!("merged model written to {} ({copied} tokenizer/config files copied)", out.display());
    Ok(())
}
