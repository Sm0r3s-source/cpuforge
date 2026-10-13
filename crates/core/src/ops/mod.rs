//! Hand-written forward/backward ops.
//!
//! Convention: every `*_backward` function **accumulates** (`+=`) into its
//! gradient outputs, so callers zero gradient buffers once and can sum
//! contributions from several consumers (e.g. the residual stream). The one
//! exception is `loss::cross_entropy_fwd_bwd`, which starts the backward pass
//! and overwrites `dlogits`.

pub mod attention;
pub mod loss;
pub mod rmsnorm;
pub mod rope;
pub mod swiglu;
