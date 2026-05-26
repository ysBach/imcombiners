//! PyO3 wrapper for offset/padding helpers.

use ndarray::ArrayViewD;
use numpy::{IntoPyArray, PyArrayDyn, PyArrayMethods, PyReadonlyArray2, PyReadonlyArrayDyn};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PySequence;

use crate::kernel::offset::place_into_padded as k_place_into_padded;

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(place_into_padded, m)?)?;
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (images, offsets, fill = None))]
fn place_into_padded<'py>(
    py: Python<'py>,
    images: &Bound<'py, PyAny>,
    offsets: PyReadonlyArray2<'py, i64>,
    fill: Option<f64>,
) -> PyResult<Bound<'py, PyAny>> {
    let sequence = images
        .cast::<PySequence>()
        .map_err(|_| PyTypeError::new_err("images must be a sequence of NumPy arrays"))?;
    let n = sequence.len()?;
    if n == 0 {
        return Err(PyValueError::new_err(
            "images must contain at least one 2-D array",
        ));
    }

    let offsets_arr = offsets.as_array();
    let offsets_vec: Vec<Vec<i64>> = offsets_arr
        .outer_iter()
        .map(|row| row.iter().copied().collect())
        .collect();

    let first = sequence.get_item(0)?;
    if first.cast::<PyArrayDyn<f32>>().is_ok() {
        let arrays: Vec<PyReadonlyArrayDyn<'py, f32>> = collect_images(sequence, n)?;
        let views: Vec<ArrayViewD<'_, f32>> = arrays.iter().map(|array| array.as_array()).collect();
        validate_offsets_shape(offsets_arr.shape(), n, views[0].ndim())?;
        let out = k_place_into_padded(&views, &offsets_vec, fill.map_or(f32::NAN, |v| v as f32))
            .map_err(PyValueError::new_err)?;
        Ok(out.into_pyarray(py).into_any())
    } else if first.cast::<PyArrayDyn<f64>>().is_ok() {
        let arrays: Vec<PyReadonlyArrayDyn<'py, f64>> = collect_images(sequence, n)?;
        let views: Vec<ArrayViewD<'_, f64>> = arrays.iter().map(|array| array.as_array()).collect();
        validate_offsets_shape(offsets_arr.shape(), n, views[0].ndim())?;
        let out = k_place_into_padded(&views, &offsets_vec, fill.unwrap_or(f64::NAN))
            .map_err(PyValueError::new_err)?;
        Ok(out.into_pyarray(py).into_any())
    } else {
        Err(PyTypeError::new_err(
            "images must contain float32 or float64 NumPy arrays",
        ))
    }
}

fn validate_offsets_shape(shape: &[usize], n: usize, ndim: usize) -> PyResult<()> {
    if shape != [n, ndim] {
        return Err(PyValueError::new_err(format!(
            "offsets shape must be ({n}, {ndim}); got {shape:?}"
        )));
    }
    Ok(())
}

fn collect_images<'py, T>(
    sequence: &Bound<'py, PySequence>,
    n: usize,
) -> PyResult<Vec<PyReadonlyArrayDyn<'py, T>>>
where
    T: numpy::Element,
{
    let mut arrays = Vec::with_capacity(n);
    for idx in 0..n {
        let item = sequence.get_item(idx)?;
        let array = item
            .cast::<PyArrayDyn<T>>()
            .map_err(|_| PyTypeError::new_err("all images must have the same float dtype"))?;
        arrays.push(array.readonly());
    }
    Ok(arrays)
}
