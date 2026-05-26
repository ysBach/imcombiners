//! Python extension entrypoint for `imcombiners`.

use pyo3::prelude::*;

mod kernel;
mod pyapi;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyapi::register(m)
}
