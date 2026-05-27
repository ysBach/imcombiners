//! PyO3 wrappers for parallel execution controls.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::kernel::utils::{
    minmax_1d_parallel_threshold, parallel_threshold,
    set_minmax_1d_parallel_threshold as k_set_minmax_1d_parallel_threshold,
    set_parallel_threshold as k_set_parallel_threshold,
};

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_rayon_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(set_rayon_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(get_parallel_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(set_parallel_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(get_minmax_1d_parallel_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(set_minmax_1d_parallel_threshold, m)?)?;
    Ok(())
}

#[pyfunction]
fn get_rayon_num_threads() -> usize {
    rayon::current_num_threads()
}

#[pyfunction]
fn set_rayon_num_threads(num_threads: usize) -> PyResult<()> {
    if num_threads == 0 {
        return Err(PyValueError::new_err("Rayon thread count must be positive"));
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .map_err(|err| {
            PyRuntimeError::new_err(format!(
                "failed to set Rayon thread count; call this before any Rayon \
                 use, including get_rayon_num_threads(): {err}"
            ))
        })?;
    Ok(())
}

#[pyfunction]
fn get_parallel_threshold() -> usize {
    parallel_threshold()
}

#[pyfunction]
fn set_parallel_threshold(threshold: usize) -> PyResult<()> {
    if threshold == 0 {
        return Err(PyValueError::new_err("parallel threshold must be positive"));
    }
    k_set_parallel_threshold(threshold);
    Ok(())
}

#[pyfunction]
fn get_minmax_1d_parallel_threshold() -> usize {
    minmax_1d_parallel_threshold()
}

#[pyfunction]
fn set_minmax_1d_parallel_threshold(threshold: usize) -> PyResult<()> {
    if threshold == 0 {
        return Err(PyValueError::new_err(
            "1-D min/max parallel threshold must be positive",
        ));
    }
    k_set_minmax_1d_parallel_threshold(threshold);
    Ok(())
}
