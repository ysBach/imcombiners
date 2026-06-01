//! PyO3 binding surface. Each submodule registers its own functions.
//!
//! Conventions (mirroring `astroapers::pyapi`):
//!   - Pure-Rust math lives in `crate::kernel`.
//!   - Each PyO3 function does numpy dtype dispatch + argument shaping, then
//!     calls into `crate::kernel`.
//!   - One PyO3 function per *operation* — no string `method=...` dispatch
//!     at the FFI boundary. Different rejection algorithms get different
//!     functions with only their own kwargs.

use pyo3::prelude::*;

pub mod combine;
pub mod reject;
mod support;

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    combine::register(m)?;
    reject::register(m)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
