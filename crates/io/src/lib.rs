//! Reading and writing models.
//!
//! - `read_safetensors` / `write_safetensors`: mmap-backed reader (f32, bf16, f16 all
//!   become f32 in memory) and an atomic writer.
//! - `load_model` / `save_model`: Hugging Face directories (`config.json` plus
//!   `model.safetensors` or a sharded index).
//! - `save_adapter` / `load_adapter`: just the LoRA tensors, a few MB instead of the model.
//! - `save_peft_adapter` / `load_peft_adapter`: the same adapter in Hugging Face PEFT layout.

mod dtype;
mod hf;
mod st;

pub use hf::{load_adapter, load_model, load_peft_adapter, save_adapter, save_model, save_peft_adapter};
pub use st::{read_safetensors, write_safetensors, Loaded};
pub use cf_core;
pub use cf_model;
