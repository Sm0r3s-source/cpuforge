//! Core building blocks: a minimal f32 tensor, matmul, and hand-written
//! forward/backward ops for Llama-style transformers.

pub mod gradcheck;
pub mod matmul;
pub mod ops;
pub mod rng;
pub mod tensor;

pub use tensor::Tensor;
