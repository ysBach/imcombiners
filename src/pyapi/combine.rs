//! PyO3 wrappers for combine kernels. One function per method (no string dispatch).

use numpy::{IntoPyArray, PyArray3, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::kernel::combine::{combine_axis0, lmedian_axis0_ord, CombineKind};

use super::support::dispatch_combine;

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mean, m)?)?;
    m.add_function(wrap_pyfunction!(median, m)?)?;
    m.add_function(wrap_pyfunction!(lmedian, m)?)?;
    m.add_function(wrap_pyfunction!(summation, m)?)?;
    m.add_function(wrap_pyfunction!(minimum, m)?)?;
    m.add_function(wrap_pyfunction!(maximum, m)?)?;
    m.add_function(wrap_pyfunction!(variance, m)?)?;
    m.add_function(wrap_pyfunction!(weighted_average, m)?)?;
    // Compat shim with string dispatch (used by IRAF-style `ndcombine`).
    m.add_function(wrap_pyfunction!(combine, m)?)?;
    Ok(())
}

// ---- per-method PyO3 functions -----------------------------------------------------

macro_rules! simple_combine {
    ($name:ident, $kind:expr) => {
        #[pyfunction]
        #[pyo3(signature = (arr))]
        fn $name<'py>(py: Python<'py>, arr: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
            dispatch_combine(
                py,
                arr,
                |v| combine_axis0::<f32>(&v, $kind, None, 0),
                |v| combine_axis0::<f64>(&v, $kind, None, 0),
            )
        }
    };
}

simple_combine!(mean, CombineKind::Mean);
simple_combine!(median, CombineKind::Median);
simple_combine!(summation, CombineKind::Sum);
simple_combine!(minimum, CombineKind::Min);
simple_combine!(maximum, CombineKind::Max);

#[pyfunction]
#[pyo3(signature = (arr, *, ddof = 0))]
fn variance<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    ddof: usize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_combine(
        py,
        arr,
        |v| combine_axis0::<f32>(&v, CombineKind::Variance, None, ddof),
        |v| combine_axis0::<f64>(&v, CombineKind::Variance, None, ddof),
    )
}

#[pyfunction]
#[pyo3(signature = (arr))]
fn lmedian<'py>(py: Python<'py>, arr: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(a) = arr.cast::<PyArray3<u8>>() {
        let out = lmedian_axis0_ord::<u8>(&a.readonly().as_array());
        Ok(out.into_pyarray(py).into_any())
    } else if let Ok(a) = arr.cast::<PyArray3<u16>>() {
        let out = lmedian_axis0_ord::<u16>(&a.readonly().as_array());
        Ok(out.into_pyarray(py).into_any())
    } else if let Ok(a) = arr.cast::<PyArray3<i16>>() {
        let out = lmedian_axis0_ord::<i16>(&a.readonly().as_array());
        Ok(out.into_pyarray(py).into_any())
    } else if let Ok(a) = arr.cast::<PyArray3<i32>>() {
        let out = lmedian_axis0_ord::<i32>(&a.readonly().as_array());
        Ok(out.into_pyarray(py).into_any())
    } else {
        dispatch_combine(
            py,
            arr,
            |v| combine_axis0::<f32>(&v, CombineKind::LMedian, None, 0),
            |v| combine_axis0::<f64>(&v, CombineKind::LMedian, None, 0),
        )
    }
}

#[pyfunction]
#[pyo3(signature = (arr, weights, *, validate = true))]
fn weighted_average<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    weights: PyReadonlyArray1<'py, f64>,
    validate: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let w: Vec<f64> = weights.as_slice().unwrap().to_vec();
    if validate {
        validate_weights_len(arr, w.len())?;
    }
    dispatch_combine(
        py,
        arr,
        |v| combine_axis0::<f32>(&v, CombineKind::WeightedAverage, Some(&w), 0),
        |v| combine_axis0::<f64>(&v, CombineKind::WeightedAverage, Some(&w), 0),
    )
}

// ---- string-dispatch compatibility shim --------------------------------------------

#[pyfunction]
#[pyo3(signature = (arr, method, weights=None, *, ddof = 0, validate = true))]
fn combine<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    method: &str,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    ddof: usize,
    validate: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let kind = CombineKind::parse(method)
        .ok_or_else(|| PyValueError::new_err(format!("unknown combine method: {method}")))?;
    let w_vec: Option<Vec<f64>> = weights.map(|w| w.as_slice().unwrap().to_vec());
    if validate && matches!(kind, CombineKind::WeightedAverage) {
        let weights_len = w_vec
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("weighted average requires weights"))?
            .len();
        validate_weights_len(arr, weights_len)?;
    }
    dispatch_combine(
        py,
        arr,
        |v| combine_axis0::<f32>(&v, kind, w_vec.as_deref(), ddof),
        |v| combine_axis0::<f64>(&v, kind, w_vec.as_deref(), ddof),
    )
}

fn stack_size(arr: &Bound<'_, PyAny>) -> PyResult<usize> {
    if let Ok(a) = arr.cast::<numpy::PyArray3<f32>>() {
        Ok(a.readonly().as_array().shape()[0])
    } else if let Ok(a) = arr.cast::<numpy::PyArray3<f64>>() {
        Ok(a.readonly().as_array().shape()[0])
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "arr must be a 3-D float32 or float64 NumPy array",
        ))
    }
}

fn validate_weights_len(arr: &Bound<'_, PyAny>, weights_len: usize) -> PyResult<()> {
    let n = stack_size(arr)?;
    if n == 0 {
        return Err(PyValueError::new_err(
            "arr must contain at least one image along axis 0",
        ));
    }
    if weights_len != n {
        return Err(PyValueError::new_err(format!(
            "weights length must match stack size N={n}"
        )));
    }
    Ok(())
}
