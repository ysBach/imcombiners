//! Shared helpers for the PyO3 layer (dtype dispatch + tuple builders).

use ndarray::{Array2, ArrayView3};
use numpy::{IntoPyArray, PyArray3, PyArrayMethods};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::kernel::reject::RejectOutput;
use crate::kernel::utils::Float;

/// Run a generic kernel `f` against `arr` after dispatching on its dtype.
/// `f` takes `(view3<T>, optional mask view)` and returns an `Array2<T>`.
pub(crate) fn dispatch_combine<'py, F32, F64>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    f32_kernel: F32,
    f64_kernel: F64,
) -> PyResult<Bound<'py, PyAny>>
where
    F32: FnOnce(ArrayView3<f32>) -> Array2<f32>,
    F64: FnOnce(ArrayView3<f64>) -> Array2<f64>,
{
    if let Ok(a) = arr.cast::<PyArray3<f32>>() {
        let a = a.readonly();
        let out = f32_kernel(a.as_array());
        Ok(out.into_pyarray(py).into_any())
    } else if let Ok(a) = arr.cast::<PyArray3<f64>>() {
        let a = a.readonly();
        let out = f64_kernel(a.as_array());
        Ok(out.into_pyarray(py).into_any())
    } else {
        Err(PyTypeError::new_err(
            "arr must be a 3-D float32 or float64 NumPy array",
        ))
    }
}

/// Pack a `RejectOutput<T>` into the standard 6-tuple
/// `(mask_rej, std, low, upp, nit, output_flags)`.
pub(crate) fn reject_to_tuple<T: Float>(
    py: Python<'_>,
    out: RejectOutput<T>,
) -> Bound<'_, PyTuple> {
    let mask = out.mask.into_pyarray(py).into_any();
    let low = out.low.into_pyarray(py).into_any();
    let upp = out.upp.into_pyarray(py).into_any();
    let nit = out.nit.into_pyarray(py).into_any();
    let output_flags = out.output_flags.into_pyarray(py).into_any();
    let std = out.std.into_pyarray(py).into_any();
    PyTuple::new(py, [mask, std, low, upp, nit, output_flags]).unwrap()
}
