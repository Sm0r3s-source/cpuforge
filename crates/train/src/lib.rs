//! Training: AdamW, LR schedule, datasets and tokenizers, the training loop,
//! and resumable state.

pub mod data;
pub mod optim;
pub mod state;
pub mod trainer;

#[cfg(test)]
mod tests;

pub use data::{ByteTokenizer, Dataset, Example, HfTokenizer, Tokenizer};
pub use optim::{lr_at, AdamW};
pub use trainer::{evaluate, Event, StepStats, TrainConfig, Trainer};

pub use cf_core;
pub use cf_io;
pub use cf_model;
