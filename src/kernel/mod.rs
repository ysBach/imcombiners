//! Pure-Rust kernel layer. No PyO3 here — only `ndarray`/`rayon`.
//!
//! This split makes the kernels callable from `cargo test` and from any future
//! Rust consumer, and isolates the FFI surface in `crate::pyapi`.

pub mod combine;
pub mod mask;
pub mod reject;
pub mod utils;
