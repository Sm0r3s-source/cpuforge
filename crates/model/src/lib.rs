//! Llama-style model definition: RMSNorm, RoPE, GQA attention, SwiGLU MLP, with
//! hand-written forward and backward and optional LoRA adapters.

pub mod config;
pub mod layer;
pub mod linear;
pub mod model;

#[cfg(test)]
mod tests;

pub use config::LlamaConfig;
pub use linear::{LinGrad, Linear, Lora};
pub use model::{visit_params, Grads, LoraConfig, MemoryMode, Model, ParamVisitor, TrainSpec};
pub use cf_core;
