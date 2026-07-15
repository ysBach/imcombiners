//! PyO3 wrappers for rejection kernels.
//!
//! One function per algorithm — each accepts only the kwargs that apply to it.
//! This is intentional: the string-dispatch `reject(method=...)` mega-signature
//! is a smell that we explicitly avoid here (it hides the type system and lets
//! irrelevant kwargs through silently).

use numpy::{
    IntoPyArray, PyArray1, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray3,
    PyReadonlyArrayDyn,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::kernel::mask::grow_mask as k_grow_mask;
use crate::kernel::reject::{
    ccdclip_mean as k_ccdclip_mean, ccdclip_mean_1d as k_ccdclip_mean_1d,
    ccdclip_median as k_ccdclip_median, ccdclip_median_1d as k_ccdclip_median_1d,
    linearclip as k_linearclip, linearclip_1d as k_linearclip_1d,
    linearclip_restored_flags as k_linearclip_restored_flags, minmax as k_minmax,
    minmax_1d as k_minmax_1d, minmax_mask as k_minmax_mask, minmax_mask_1d as k_minmax_mask_1d,
    minmax_mean as k_minmax_mean, minmax_mean_1d as k_minmax_mean_1d,
    minmax_median as k_minmax_median, minmax_median_1d as k_minmax_median_1d, pclip as k_pclip,
    pclip_1d as k_pclip_1d, pclip_mask as k_pclip_mask, pclip_mask_1d as k_pclip_mask_1d,
    pclip_mean as k_pclip_mean, pclip_mean_1d as k_pclip_mean_1d, pclip_median as k_pclip_median,
    pclip_median_1d as k_pclip_median_1d, sigclip as k_sigclip, sigclip_1d as k_sigclip_1d,
    sigclip_mask as k_sigclip_mask, sigclip_mask_1d as k_sigclip_mask_1d,
    sigclip_mean as k_sigclip_mean, sigclip_mean_1d as k_sigclip_mean_1d,
    sigclip_median as k_sigclip_median, sigclip_median_1d as k_sigclip_median_1d,
    sigclip_restored_flags as k_sigclip_restored_flags, CenFunc, ClipCenter, LinearClipParams,
    SigClipParams, StdFunc,
};
use crate::kernel::utils::Float;

use super::support::{reject_1d_to_tuple, reject_to_tuple};

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sigclip, m)?)?;
    m.add_function(wrap_pyfunction!(sigclip_1d, m)?)?;
    m.add_function(wrap_pyfunction!(ccdclip, m)?)?;
    m.add_function(wrap_pyfunction!(ccdclip_1d, m)?)?;
    m.add_function(wrap_pyfunction!(sigclip_restored_flags, m)?)?;
    m.add_function(wrap_pyfunction!(ccdclip_restored_flags, m)?)?;
    m.add_function(wrap_pyfunction!(linearclip, m)?)?;
    m.add_function(wrap_pyfunction!(linearclip_1d, m)?)?;
    m.add_function(wrap_pyfunction!(linearclip_restored_flags, m)?)?;
    m.add_function(wrap_pyfunction!(minmax, m)?)?;
    m.add_function(wrap_pyfunction!(minmax_1d, m)?)?;
    m.add_function(wrap_pyfunction!(pclip, m)?)?;
    m.add_function(wrap_pyfunction!(pclip_1d, m)?)?;
    m.add_function(wrap_pyfunction!(sigclip_mask, m)?)?;
    m.add_function(wrap_pyfunction!(sigclip_mask_1d, m)?)?;
    m.add_function(wrap_pyfunction!(sigclip_mean, m)?)?;
    m.add_function(wrap_pyfunction!(sigclip_mean_1d, m)?)?;
    m.add_function(wrap_pyfunction!(sigclip_median, m)?)?;
    m.add_function(wrap_pyfunction!(sigclip_median_1d, m)?)?;
    m.add_function(wrap_pyfunction!(ccdclip_mask, m)?)?;
    m.add_function(wrap_pyfunction!(ccdclip_mask_1d, m)?)?;
    m.add_function(wrap_pyfunction!(ccdclip_mean, m)?)?;
    m.add_function(wrap_pyfunction!(ccdclip_mean_1d, m)?)?;
    m.add_function(wrap_pyfunction!(ccdclip_median, m)?)?;
    m.add_function(wrap_pyfunction!(ccdclip_median_1d, m)?)?;
    m.add_function(wrap_pyfunction!(minmax_mask, m)?)?;
    m.add_function(wrap_pyfunction!(minmax_mask_1d, m)?)?;
    m.add_function(wrap_pyfunction!(minmax_mean, m)?)?;
    m.add_function(wrap_pyfunction!(minmax_mean_1d, m)?)?;
    m.add_function(wrap_pyfunction!(minmax_median, m)?)?;
    m.add_function(wrap_pyfunction!(minmax_median_1d, m)?)?;
    m.add_function(wrap_pyfunction!(pclip_mask, m)?)?;
    m.add_function(wrap_pyfunction!(pclip_mask_1d, m)?)?;
    m.add_function(wrap_pyfunction!(pclip_mean, m)?)?;
    m.add_function(wrap_pyfunction!(pclip_mean_1d, m)?)?;
    m.add_function(wrap_pyfunction!(pclip_median, m)?)?;
    m.add_function(wrap_pyfunction!(pclip_median_1d, m)?)?;
    m.add_function(wrap_pyfunction!(grow_mask, m)?)?;
    Ok(())
}

// ---- helpers -----------------------------------------------------------------------

/// Dispatch a sigclip-family kernel on dtype + run it. The closure receives the
/// fully-built `SigClipParams` and the array views.
fn run_sigclip_kind<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArray3<'py, bool>>,
    params: SigClipParams,
    validate: bool,
) -> PyResult<Bound<'py, PyTuple>> {
    if let Ok(a) = arr.cast::<PyArray3<f32>>() {
        let a = a.readonly();
        let view = a.as_array();
        let mview = mask.as_ref().map(|m| m.as_array());
        if validate {
            validate_reject_inputs(view.shape(), mview.as_ref().map(|m| m.shape()))?;
        }
        let out = k_sigclip::<f32>(&view, mview.as_ref(), &params);
        Ok(reject_to_tuple::<f32>(py, out))
    } else if let Ok(a) = arr.cast::<PyArray3<f64>>() {
        let a = a.readonly();
        let view = a.as_array();
        let mview = mask.as_ref().map(|m| m.as_array());
        if validate {
            validate_reject_inputs(view.shape(), mview.as_ref().map(|m| m.shape()))?;
        }
        let out = k_sigclip::<f64>(&view, mview.as_ref(), &params);
        Ok(reject_to_tuple::<f64>(py, out))
    } else {
        Err(PyTypeError::new_err(
            "arr must be a 3-D float32 or float64 NumPy array",
        ))
    }
}

fn run_sigclip_mask_kind<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArray3<'py, bool>>,
    params: SigClipParams,
    validate: bool,
) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(a) = arr.cast::<PyArray3<f32>>() {
        let a = a.readonly();
        let view = a.as_array();
        let mview = mask.as_ref().map(|m| m.as_array());
        if validate {
            validate_reject_inputs(view.shape(), mview.as_ref().map(|m| m.shape()))?;
        }
        let out = k_sigclip_mask::<f32>(&view, mview.as_ref(), &params);
        Ok(out.into_pyarray(py).into_any())
    } else if let Ok(a) = arr.cast::<PyArray3<f64>>() {
        let a = a.readonly();
        let view = a.as_array();
        let mview = mask.as_ref().map(|m| m.as_array());
        if validate {
            validate_reject_inputs(view.shape(), mview.as_ref().map(|m| m.shape()))?;
        }
        let out = k_sigclip_mask::<f64>(&view, mview.as_ref(), &params);
        Ok(out.into_pyarray(py).into_any())
    } else {
        Err(PyTypeError::new_err(
            "arr must be a 3-D float32 or float64 NumPy array",
        ))
    }
}

fn run_sigclip_mask_1d_kind<'py>(
    py: Python<'py>,
    values: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArray1<'py, bool>>,
    params: SigClipParams,
    validate: bool,
) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(a) = values.cast::<PyArray1<f32>>() {
        let a = a.readonly();
        let values = a.as_slice().unwrap();
        let mask_slice = mask.as_ref().map(|m| m.as_slice().unwrap());
        if validate {
            validate_1d_inputs(values.len(), mask_slice.map(<[bool]>::len))?;
        }
        let out = k_sigclip_mask_1d::<f32>(values, mask_slice, &params);
        Ok(out.into_pyarray(py).into_any())
    } else if let Ok(a) = values.cast::<PyArray1<f64>>() {
        let a = a.readonly();
        let values = a.as_slice().unwrap();
        let mask_slice = mask.as_ref().map(|m| m.as_slice().unwrap());
        if validate {
            validate_1d_inputs(values.len(), mask_slice.map(<[bool]>::len))?;
        }
        let out = k_sigclip_mask_1d::<f64>(values, mask_slice, &params);
        Ok(out.into_pyarray(py).into_any())
    } else {
        Err(PyTypeError::new_err(
            "values must be a 1-D float32 or float64 NumPy array",
        ))
    }
}

fn run_sigclip_restored_flags_kind<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArray3<'py, bool>>,
    params: SigClipParams,
    validate: bool,
) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(a) = arr.cast::<PyArray3<f32>>() {
        let a = a.readonly();
        let view = a.as_array();
        let mview = mask.as_ref().map(|m| m.as_array());
        if validate {
            validate_reject_inputs(view.shape(), mview.as_ref().map(|m| m.shape()))?;
        }
        let out = k_sigclip_restored_flags::<f32>(&view, mview.as_ref(), &params);
        Ok(out.into_pyarray(py).into_any())
    } else if let Ok(a) = arr.cast::<PyArray3<f64>>() {
        let a = a.readonly();
        let view = a.as_array();
        let mview = mask.as_ref().map(|m| m.as_array());
        if validate {
            validate_reject_inputs(view.shape(), mview.as_ref().map(|m| m.shape()))?;
        }
        let out = k_sigclip_restored_flags::<f64>(&view, mview.as_ref(), &params);
        Ok(out.into_pyarray(py).into_any())
    } else {
        Err(PyTypeError::new_err(
            "arr must be a 3-D float32 or float64 NumPy array",
        ))
    }
}

fn run_sigclip_mean_kind<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArray3<'py, bool>>,
    params: SigClipParams,
    validate: bool,
) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(a) = arr.cast::<PyArray3<f32>>() {
        let a = a.readonly();
        let view = a.as_array();
        let mview = mask.as_ref().map(|m| m.as_array());
        if validate {
            validate_reject_inputs(view.shape(), mview.as_ref().map(|m| m.shape()))?;
        }
        let out = k_sigclip_mean::<f32>(&view, mview.as_ref(), &params);
        Ok(out.into_pyarray(py).into_any())
    } else if let Ok(a) = arr.cast::<PyArray3<f64>>() {
        let a = a.readonly();
        let view = a.as_array();
        let mview = mask.as_ref().map(|m| m.as_array());
        if validate {
            validate_reject_inputs(view.shape(), mview.as_ref().map(|m| m.shape()))?;
        }
        let out = k_sigclip_mean::<f64>(&view, mview.as_ref(), &params);
        Ok(out.into_pyarray(py).into_any())
    } else {
        Err(PyTypeError::new_err(
            "arr must be a 3-D float32 or float64 NumPy array",
        ))
    }
}

fn run_sigclip_median_kind<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArray3<'py, bool>>,
    params: SigClipParams,
    validate: bool,
) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(a) = arr.cast::<PyArray3<f32>>() {
        let a = a.readonly();
        let view = a.as_array();
        let mview = mask.as_ref().map(|m| m.as_array());
        if validate {
            validate_reject_inputs(view.shape(), mview.as_ref().map(|m| m.shape()))?;
        }
        let out = k_sigclip_median::<f32>(&view, mview.as_ref(), &params);
        Ok(out.into_pyarray(py).into_any())
    } else if let Ok(a) = arr.cast::<PyArray3<f64>>() {
        let a = a.readonly();
        let view = a.as_array();
        let mview = mask.as_ref().map(|m| m.as_array());
        if validate {
            validate_reject_inputs(view.shape(), mview.as_ref().map(|m| m.shape()))?;
        }
        let out = k_sigclip_median::<f64>(&view, mview.as_ref(), &params);
        Ok(out.into_pyarray(py).into_any())
    } else {
        Err(PyTypeError::new_err(
            "arr must be a 3-D float32 or float64 NumPy array",
        ))
    }
}

fn parse_cenfunc(s: &str) -> PyResult<CenFunc> {
    CenFunc::parse(s).ok_or_else(|| PyValueError::new_err(format!("unknown cenfunc: {s}")))
}

fn parse_clip_center(s: &str) -> PyResult<ClipCenter> {
    ClipCenter::parse(s).ok_or_else(|| PyValueError::new_err(format!("unknown clip_cen: {s}")))
}

fn parse_stdfunc(s: &str) -> PyResult<StdFunc> {
    StdFunc::parse(s).ok_or_else(|| PyValueError::new_err(format!("unknown stdfunc: {s}")))
}

fn validate_sigma_thresholds(sigma_lower: f64, sigma_upper: f64) -> PyResult<()> {
    if sigma_lower.is_finite()
        && sigma_upper.is_finite()
        && sigma_lower >= 0.0
        && sigma_upper >= 0.0
    {
        Ok(())
    } else {
        Err(PyValueError::new_err(
            "sigma thresholds must be finite and non-negative",
        ))
    }
}

#[allow(clippy::too_many_arguments)]
fn make_sigclip_params(
    n: usize,
    sigma_lower: f64,
    sigma_upper: f64,
    maxiters: usize,
    ddof: usize,
    nkeep: usize,
    maxrej: Option<usize>,
    cenfunc: &str,
    clip_cen: &str,
    stdfunc: &str,
    revert_on_nkeep: bool,
) -> PyResult<SigClipParams> {
    validate_sigma_thresholds(sigma_lower, sigma_upper)?;
    Ok(SigClipParams {
        sigma_lower,
        sigma_upper,
        maxiters,
        ddof,
        nkeep,
        maxrej: maxrej.unwrap_or(n),
        cenfunc: parse_cenfunc(cenfunc)?,
        clip_cen: parse_clip_center(clip_cen)?,
        stdfunc: parse_stdfunc(stdfunc)?,
        revert_on_nkeep,
        ccdclip: false,
        rdnoise_ref: 0.0,
        snoise_ref: 0.0,
        scales: Vec::new(),
        zeros: Vec::new(),
        gain_ref: 1.0,
    })
}

#[allow(clippy::too_many_arguments)]
fn make_ccdclip_params(
    n: usize,
    sigma_lower: f64,
    sigma_upper: f64,
    maxiters: usize,
    ddof: usize,
    nkeep: usize,
    maxrej: Option<usize>,
    cenfunc: &str,
    clip_cen: &str,
    revert_on_nkeep: bool,
    rdnoise_ref: f64,
    snoise_ref: f64,
    scales: Option<Vec<f64>>,
    zeros: Option<Vec<f64>>,
    gain: f64,
    validate: bool,
) -> PyResult<SigClipParams> {
    validate_sigma_thresholds(sigma_lower, sigma_upper)?;
    if validate && (!gain.is_finite() || gain <= 0.0) {
        return Err(PyValueError::new_err("gain must be finite and positive"));
    }
    let scales = scales.unwrap_or_else(|| vec![1.0; n]);
    let zeros = zeros.unwrap_or_else(|| vec![0.0; n]);
    if validate {
        if scales.len() != n || zeros.len() != n {
            return Err(PyValueError::new_err(
                "scales/zeros must have one value per image plane",
            ));
        }
        if scales
            .iter()
            .any(|&value| !value.is_finite() || value <= 0.0)
        {
            return Err(PyValueError::new_err("scales must be finite and positive"));
        }
        if zeros.iter().any(|&value| !value.is_finite()) {
            return Err(PyValueError::new_err("zeros must be finite"));
        }
    }
    let use_scaling = scales
        .iter()
        .zip(&zeros)
        .any(|(&scale, &zero)| scale != 1.0 || zero != 0.0);
    Ok(SigClipParams {
        sigma_lower,
        sigma_upper,
        maxiters,
        ddof,
        nkeep,
        maxrej: maxrej.unwrap_or(n),
        cenfunc: parse_cenfunc(cenfunc)?,
        clip_cen: parse_clip_center(clip_cen)?,
        stdfunc: StdFunc::Std,
        revert_on_nkeep,
        ccdclip: true,
        rdnoise_ref,
        snoise_ref,
        scales: if use_scaling { scales } else { Vec::new() },
        zeros: if use_scaling { zeros } else { Vec::new() },
        gain_ref: gain,
    })
}

// ---- public PyO3 functions ---------------------------------------------------------

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    arr,
    *,
    mask = None,
    sigma_lower = 3.0,
    sigma_upper = 3.0,
    maxiters = 5,
    ddof = 0,
    nkeep = 1,
    maxrej = None,
    cenfunc = "median",
    clip_cen = "mean",
    stdfunc = "std",
    revert_on_nkeep = true,
    validate = true,
))]
fn sigclip<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArray3<'py, bool>>,
    sigma_lower: f64,
    sigma_upper: f64,
    maxiters: usize,
    ddof: usize,
    nkeep: usize,
    maxrej: Option<usize>,
    cenfunc: &str,
    clip_cen: &str,
    stdfunc: &str,
    revert_on_nkeep: bool,
    validate: bool,
) -> PyResult<Bound<'py, PyTuple>> {
    let n = stack_size(arr)?;
    let params = make_sigclip_params(
        n,
        sigma_lower,
        sigma_upper,
        maxiters,
        ddof,
        nkeep,
        maxrej,
        cenfunc,
        clip_cen,
        stdfunc,
        revert_on_nkeep,
    )?;
    run_sigclip_kind(py, arr, mask, params, validate)
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    values,
    *,
    mask = None,
    sigma_lower = 3.0,
    sigma_upper = 3.0,
    maxiters = 5,
    ddof = 0,
    nkeep = 1,
    maxrej = None,
    cenfunc = "median",
    clip_cen = "mean",
    stdfunc = "std",
    revert_on_nkeep = true,
    validate = true,
))]
fn sigclip_1d<'py>(
    py: Python<'py>,
    values: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArray1<'py, bool>>,
    sigma_lower: f64,
    sigma_upper: f64,
    maxiters: usize,
    ddof: usize,
    nkeep: usize,
    maxrej: Option<usize>,
    cenfunc: &str,
    clip_cen: &str,
    stdfunc: &str,
    revert_on_nkeep: bool,
    validate: bool,
) -> PyResult<Bound<'py, PyTuple>> {
    let n = values_len_1d(values)?;
    let params = make_sigclip_params(
        n,
        sigma_lower,
        sigma_upper,
        maxiters,
        ddof,
        nkeep,
        maxrej,
        cenfunc,
        clip_cen,
        stdfunc,
        revert_on_nkeep,
    )?;
    dispatch_rejection_1d(
        py,
        values,
        mask,
        |v, m| k_sigclip_1d(v, m, &params),
        |v, m| k_sigclip_1d(v, m, &params),
        validate,
    )
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    arr,
    *,
    mask = None,
    sigma_lower = 3.0,
    sigma_upper = 3.0,
    maxiters = 5,
    ddof = 0,
    nkeep = 1,
    maxrej = None,
    cenfunc = "median",
    clip_cen = "mean",
    revert_on_nkeep = true,
    rdnoise_ref = 0.0,
    snoise_ref = 0.0,
    scales = None,
    zeros = None,
    gain = 1.0,
    validate = true,
))]
fn ccdclip<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArray3<'py, bool>>,
    sigma_lower: f64,
    sigma_upper: f64,
    maxiters: usize,
    ddof: usize,
    nkeep: usize,
    maxrej: Option<usize>,
    cenfunc: &str,
    clip_cen: &str,
    revert_on_nkeep: bool,
    rdnoise_ref: f64,
    snoise_ref: f64,
    scales: Option<Vec<f64>>,
    zeros: Option<Vec<f64>>,
    gain: f64,
    validate: bool,
) -> PyResult<Bound<'py, PyTuple>> {
    let n = stack_size(arr)?;
    let params = make_ccdclip_params(
        n,
        sigma_lower,
        sigma_upper,
        maxiters,
        ddof,
        nkeep,
        maxrej,
        cenfunc,
        clip_cen,
        revert_on_nkeep,
        rdnoise_ref,
        snoise_ref,
        scales,
        zeros,
        gain,
        validate,
    )?;
    run_sigclip_kind(py, arr, mask, params, validate)
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    values,
    *,
    mask = None,
    sigma_lower = 3.0,
    sigma_upper = 3.0,
    maxiters = 5,
    ddof = 0,
    nkeep = 1,
    maxrej = None,
    cenfunc = "median",
    clip_cen = "mean",
    revert_on_nkeep = true,
    rdnoise_ref = 0.0,
    snoise_ref = 0.0,
    scales = None,
    zeros = None,
    gain = 1.0,
    validate = true,
))]
fn ccdclip_1d<'py>(
    py: Python<'py>,
    values: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArray1<'py, bool>>,
    sigma_lower: f64,
    sigma_upper: f64,
    maxiters: usize,
    ddof: usize,
    nkeep: usize,
    maxrej: Option<usize>,
    cenfunc: &str,
    clip_cen: &str,
    revert_on_nkeep: bool,
    rdnoise_ref: f64,
    snoise_ref: f64,
    scales: Option<Vec<f64>>,
    zeros: Option<Vec<f64>>,
    gain: f64,
    validate: bool,
) -> PyResult<Bound<'py, PyTuple>> {
    let n = values_len_1d(values)?;
    let params = make_ccdclip_params(
        n,
        sigma_lower,
        sigma_upper,
        maxiters,
        ddof,
        nkeep,
        maxrej,
        cenfunc,
        clip_cen,
        revert_on_nkeep,
        rdnoise_ref,
        snoise_ref,
        scales,
        zeros,
        gain,
        validate,
    )?;
    dispatch_rejection_1d(
        py,
        values,
        mask,
        |v, m| k_sigclip_1d(v, m, &params),
        |v, m| k_sigclip_1d(v, m, &params),
        validate,
    )
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    arr,
    *,
    mask = None,
    sigma_lower = 3.0,
    sigma_upper = 3.0,
    maxiters = 5,
    ddof = 0,
    nkeep = 1,
    maxrej = None,
    cenfunc = "median",
    clip_cen = "mean",
    stdfunc = "std",
    revert_on_nkeep = true,
    validate = true,
))]
fn sigclip_restored_flags<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArray3<'py, bool>>,
    sigma_lower: f64,
    sigma_upper: f64,
    maxiters: usize,
    ddof: usize,
    nkeep: usize,
    maxrej: Option<usize>,
    cenfunc: &str,
    clip_cen: &str,
    stdfunc: &str,
    revert_on_nkeep: bool,
    validate: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let n = stack_size(arr)?;
    let params = make_sigclip_params(
        n,
        sigma_lower,
        sigma_upper,
        maxiters,
        ddof,
        nkeep,
        maxrej,
        cenfunc,
        clip_cen,
        stdfunc,
        revert_on_nkeep,
    )?;
    run_sigclip_restored_flags_kind(py, arr, mask, params, validate)
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    arr,
    *,
    mask = None,
    sigma_lower = 3.0,
    sigma_upper = 3.0,
    maxiters = 5,
    ddof = 0,
    nkeep = 1,
    maxrej = None,
    cenfunc = "median",
    clip_cen = "mean",
    revert_on_nkeep = true,
    rdnoise_ref = 0.0,
    snoise_ref = 0.0,
    scales = None,
    zeros = None,
    gain = 1.0,
    validate = true,
))]
fn ccdclip_restored_flags<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArray3<'py, bool>>,
    sigma_lower: f64,
    sigma_upper: f64,
    maxiters: usize,
    ddof: usize,
    nkeep: usize,
    maxrej: Option<usize>,
    cenfunc: &str,
    clip_cen: &str,
    revert_on_nkeep: bool,
    rdnoise_ref: f64,
    snoise_ref: f64,
    scales: Option<Vec<f64>>,
    zeros: Option<Vec<f64>>,
    gain: f64,
    validate: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let n = stack_size(arr)?;
    let params = make_ccdclip_params(
        n,
        sigma_lower,
        sigma_upper,
        maxiters,
        ddof,
        nkeep,
        maxrej,
        cenfunc,
        clip_cen,
        revert_on_nkeep,
        rdnoise_ref,
        snoise_ref,
        scales,
        zeros,
        gain,
        validate,
    )?;
    run_sigclip_restored_flags_kind(py, arr, mask, params, validate)
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    arr,
    *,
    mask = None,
    sigma_lower = 3.0,
    sigma_upper = 3.0,
    maxiters = 5,
    ddof = 0,
    nkeep = 1,
    maxrej = None,
    cenfunc = "median",
    clip_cen = "mean",
    stdfunc = "std",
    revert_on_nkeep = true,
    validate = true,
))]
fn sigclip_mask<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArray3<'py, bool>>,
    sigma_lower: f64,
    sigma_upper: f64,
    maxiters: usize,
    ddof: usize,
    nkeep: usize,
    maxrej: Option<usize>,
    cenfunc: &str,
    clip_cen: &str,
    stdfunc: &str,
    revert_on_nkeep: bool,
    validate: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let n = stack_size(arr)?;
    let params = make_sigclip_params(
        n,
        sigma_lower,
        sigma_upper,
        maxiters,
        ddof,
        nkeep,
        maxrej,
        cenfunc,
        clip_cen,
        stdfunc,
        revert_on_nkeep,
    )?;
    run_sigclip_mask_kind(py, arr, mask, params, validate)
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    values,
    *,
    mask = None,
    sigma_lower = 3.0,
    sigma_upper = 3.0,
    maxiters = 5,
    ddof = 0,
    nkeep = 1,
    maxrej = None,
    cenfunc = "median",
    clip_cen = "mean",
    stdfunc = "std",
    revert_on_nkeep = true,
    validate = true,
))]
fn sigclip_mask_1d<'py>(
    py: Python<'py>,
    values: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArray1<'py, bool>>,
    sigma_lower: f64,
    sigma_upper: f64,
    maxiters: usize,
    ddof: usize,
    nkeep: usize,
    maxrej: Option<usize>,
    cenfunc: &str,
    clip_cen: &str,
    stdfunc: &str,
    revert_on_nkeep: bool,
    validate: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let n = values_len_1d(values)?;
    let params = make_sigclip_params(
        n,
        sigma_lower,
        sigma_upper,
        maxiters,
        ddof,
        nkeep,
        maxrej,
        cenfunc,
        clip_cen,
        stdfunc,
        revert_on_nkeep,
    )?;
    run_sigclip_mask_1d_kind(py, values, mask, params, validate)
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    arr,
    *,
    mask = None,
    sigma_lower = 3.0,
    sigma_upper = 3.0,
    maxiters = 5,
    ddof = 0,
    nkeep = 1,
    maxrej = None,
    cenfunc = "median",
    clip_cen = "mean",
    stdfunc = "std",
    revert_on_nkeep = true,
    validate = true,
))]
fn sigclip_mean<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArray3<'py, bool>>,
    sigma_lower: f64,
    sigma_upper: f64,
    maxiters: usize,
    ddof: usize,
    nkeep: usize,
    maxrej: Option<usize>,
    cenfunc: &str,
    clip_cen: &str,
    stdfunc: &str,
    revert_on_nkeep: bool,
    validate: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let n = stack_size(arr)?;
    let params = make_sigclip_params(
        n,
        sigma_lower,
        sigma_upper,
        maxiters,
        ddof,
        nkeep,
        maxrej,
        cenfunc,
        clip_cen,
        stdfunc,
        revert_on_nkeep,
    )?;
    run_sigclip_mean_kind(py, arr, mask, params, validate)
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    values,
    *,
    mask = None,
    sigma_lower = 3.0,
    sigma_upper = 3.0,
    maxiters = 5,
    ddof = 0,
    nkeep = 1,
    maxrej = None,
    cenfunc = "median",
    clip_cen = "mean",
    stdfunc = "std",
    revert_on_nkeep = true,
    validate = true,
))]
fn sigclip_mean_1d<'py>(
    values: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArray1<'py, bool>>,
    sigma_lower: f64,
    sigma_upper: f64,
    maxiters: usize,
    ddof: usize,
    nkeep: usize,
    maxrej: Option<usize>,
    cenfunc: &str,
    clip_cen: &str,
    stdfunc: &str,
    revert_on_nkeep: bool,
    validate: bool,
) -> PyResult<f64> {
    let n = values_len_1d(values)?;
    let params = make_sigclip_params(
        n,
        sigma_lower,
        sigma_upper,
        maxiters,
        ddof,
        nkeep,
        maxrej,
        cenfunc,
        clip_cen,
        stdfunc,
        revert_on_nkeep,
    )?;
    dispatch_rejection_combine_1d(
        values,
        mask,
        |v, m| k_sigclip_mean_1d(v, m, &params),
        |v, m| k_sigclip_mean_1d(v, m, &params),
        validate,
    )
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    arr,
    *,
    mask = None,
    sigma_lower = 3.0,
    sigma_upper = 3.0,
    maxiters = 5,
    ddof = 0,
    nkeep = 1,
    maxrej = None,
    cenfunc = "median",
    clip_cen = "mean",
    stdfunc = "std",
    revert_on_nkeep = true,
    validate = true,
))]
fn sigclip_median<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArray3<'py, bool>>,
    sigma_lower: f64,
    sigma_upper: f64,
    maxiters: usize,
    ddof: usize,
    nkeep: usize,
    maxrej: Option<usize>,
    cenfunc: &str,
    clip_cen: &str,
    stdfunc: &str,
    revert_on_nkeep: bool,
    validate: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let n = stack_size(arr)?;
    let params = make_sigclip_params(
        n,
        sigma_lower,
        sigma_upper,
        maxiters,
        ddof,
        nkeep,
        maxrej,
        cenfunc,
        clip_cen,
        stdfunc,
        revert_on_nkeep,
    )?;
    run_sigclip_median_kind(py, arr, mask, params, validate)
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    values,
    *,
    mask = None,
    sigma_lower = 3.0,
    sigma_upper = 3.0,
    maxiters = 5,
    ddof = 0,
    nkeep = 1,
    maxrej = None,
    cenfunc = "median",
    clip_cen = "mean",
    stdfunc = "std",
    revert_on_nkeep = true,
    validate = true,
))]
fn sigclip_median_1d<'py>(
    values: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArray1<'py, bool>>,
    sigma_lower: f64,
    sigma_upper: f64,
    maxiters: usize,
    ddof: usize,
    nkeep: usize,
    maxrej: Option<usize>,
    cenfunc: &str,
    clip_cen: &str,
    stdfunc: &str,
    revert_on_nkeep: bool,
    validate: bool,
) -> PyResult<f64> {
    let n = values_len_1d(values)?;
    let params = make_sigclip_params(
        n,
        sigma_lower,
        sigma_upper,
        maxiters,
        ddof,
        nkeep,
        maxrej,
        cenfunc,
        clip_cen,
        stdfunc,
        revert_on_nkeep,
    )?;
    dispatch_rejection_combine_1d(
        values,
        mask,
        |v, m| k_sigclip_median_1d(v, m, &params),
        |v, m| k_sigclip_median_1d(v, m, &params),
        validate,
    )
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    arr,
    *,
    mask = None,
    sigma_lower = 3.0,
    sigma_upper = 3.0,
    maxiters = 5,
    ddof = 0,
    nkeep = 1,
    maxrej = None,
    cenfunc = "median",
    clip_cen = "mean",
    revert_on_nkeep = true,
    rdnoise_ref = 0.0,
    snoise_ref = 0.0,
    scales = None,
    zeros = None,
    gain = 1.0,
    validate = true,
))]
fn ccdclip_mask<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArray3<'py, bool>>,
    sigma_lower: f64,
    sigma_upper: f64,
    maxiters: usize,
    ddof: usize,
    nkeep: usize,
    maxrej: Option<usize>,
    cenfunc: &str,
    clip_cen: &str,
    revert_on_nkeep: bool,
    rdnoise_ref: f64,
    snoise_ref: f64,
    scales: Option<Vec<f64>>,
    zeros: Option<Vec<f64>>,
    gain: f64,
    validate: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let n = stack_size(arr)?;
    let params = make_ccdclip_params(
        n,
        sigma_lower,
        sigma_upper,
        maxiters,
        ddof,
        nkeep,
        maxrej,
        cenfunc,
        clip_cen,
        revert_on_nkeep,
        rdnoise_ref,
        snoise_ref,
        scales,
        zeros,
        gain,
        validate,
    )?;
    run_sigclip_mask_kind(py, arr, mask, params, validate)
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    values,
    *,
    mask = None,
    sigma_lower = 3.0,
    sigma_upper = 3.0,
    maxiters = 5,
    ddof = 0,
    nkeep = 1,
    maxrej = None,
    cenfunc = "median",
    clip_cen = "mean",
    revert_on_nkeep = true,
    rdnoise_ref = 0.0,
    snoise_ref = 0.0,
    scales = None,
    zeros = None,
    gain = 1.0,
    validate = true,
))]
fn ccdclip_mask_1d<'py>(
    py: Python<'py>,
    values: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArray1<'py, bool>>,
    sigma_lower: f64,
    sigma_upper: f64,
    maxiters: usize,
    ddof: usize,
    nkeep: usize,
    maxrej: Option<usize>,
    cenfunc: &str,
    clip_cen: &str,
    revert_on_nkeep: bool,
    rdnoise_ref: f64,
    snoise_ref: f64,
    scales: Option<Vec<f64>>,
    zeros: Option<Vec<f64>>,
    gain: f64,
    validate: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let n = values_len_1d(values)?;
    let params = make_ccdclip_params(
        n,
        sigma_lower,
        sigma_upper,
        maxiters,
        ddof,
        nkeep,
        maxrej,
        cenfunc,
        clip_cen,
        revert_on_nkeep,
        rdnoise_ref,
        snoise_ref,
        scales,
        zeros,
        gain,
        validate,
    )?;
    dispatch_rejection_mask_1d(
        py,
        values,
        mask,
        |v, m| k_sigclip_mask_1d(v, m, &params),
        |v, m| k_sigclip_mask_1d(v, m, &params),
        validate,
    )
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    arr,
    *,
    mask = None,
    sigma_lower = 3.0,
    sigma_upper = 3.0,
    maxiters = 5,
    ddof = 0,
    nkeep = 1,
    maxrej = None,
    cenfunc = "median",
    clip_cen = "mean",
    revert_on_nkeep = true,
    rdnoise_ref = 0.0,
    snoise_ref = 0.0,
    scales = None,
    zeros = None,
    gain = 1.0,
    validate = true,
))]
fn ccdclip_mean<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArray3<'py, bool>>,
    sigma_lower: f64,
    sigma_upper: f64,
    maxiters: usize,
    ddof: usize,
    nkeep: usize,
    maxrej: Option<usize>,
    cenfunc: &str,
    clip_cen: &str,
    revert_on_nkeep: bool,
    rdnoise_ref: f64,
    snoise_ref: f64,
    scales: Option<Vec<f64>>,
    zeros: Option<Vec<f64>>,
    gain: f64,
    validate: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let n = stack_size(arr)?;
    let params = make_ccdclip_params(
        n,
        sigma_lower,
        sigma_upper,
        maxiters,
        ddof,
        nkeep,
        maxrej,
        cenfunc,
        clip_cen,
        revert_on_nkeep,
        rdnoise_ref,
        snoise_ref,
        scales,
        zeros,
        gain,
        validate,
    )?;
    dispatch_rejection_combine(
        py,
        arr,
        mask,
        |v, m| k_ccdclip_mean(&v, m, &params),
        |v, m| k_ccdclip_mean(&v, m, &params),
        validate,
    )
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    values,
    *,
    mask = None,
    sigma_lower = 3.0,
    sigma_upper = 3.0,
    maxiters = 5,
    ddof = 0,
    nkeep = 1,
    maxrej = None,
    cenfunc = "median",
    clip_cen = "mean",
    revert_on_nkeep = true,
    rdnoise_ref = 0.0,
    snoise_ref = 0.0,
    scales = None,
    zeros = None,
    gain = 1.0,
    validate = true,
))]
fn ccdclip_mean_1d<'py>(
    values: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArray1<'py, bool>>,
    sigma_lower: f64,
    sigma_upper: f64,
    maxiters: usize,
    ddof: usize,
    nkeep: usize,
    maxrej: Option<usize>,
    cenfunc: &str,
    clip_cen: &str,
    revert_on_nkeep: bool,
    rdnoise_ref: f64,
    snoise_ref: f64,
    scales: Option<Vec<f64>>,
    zeros: Option<Vec<f64>>,
    gain: f64,
    validate: bool,
) -> PyResult<f64> {
    let n = values_len_1d(values)?;
    let params = make_ccdclip_params(
        n,
        sigma_lower,
        sigma_upper,
        maxiters,
        ddof,
        nkeep,
        maxrej,
        cenfunc,
        clip_cen,
        revert_on_nkeep,
        rdnoise_ref,
        snoise_ref,
        scales,
        zeros,
        gain,
        validate,
    )?;
    dispatch_rejection_combine_1d(
        values,
        mask,
        |v, m| k_ccdclip_mean_1d(v, m, &params),
        |v, m| k_ccdclip_mean_1d(v, m, &params),
        validate,
    )
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    arr,
    *,
    mask = None,
    sigma_lower = 3.0,
    sigma_upper = 3.0,
    maxiters = 5,
    ddof = 0,
    nkeep = 1,
    maxrej = None,
    cenfunc = "median",
    clip_cen = "mean",
    revert_on_nkeep = true,
    rdnoise_ref = 0.0,
    snoise_ref = 0.0,
    scales = None,
    zeros = None,
    gain = 1.0,
    validate = true,
))]
fn ccdclip_median<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArray3<'py, bool>>,
    sigma_lower: f64,
    sigma_upper: f64,
    maxiters: usize,
    ddof: usize,
    nkeep: usize,
    maxrej: Option<usize>,
    cenfunc: &str,
    clip_cen: &str,
    revert_on_nkeep: bool,
    rdnoise_ref: f64,
    snoise_ref: f64,
    scales: Option<Vec<f64>>,
    zeros: Option<Vec<f64>>,
    gain: f64,
    validate: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let n = stack_size(arr)?;
    let params = make_ccdclip_params(
        n,
        sigma_lower,
        sigma_upper,
        maxiters,
        ddof,
        nkeep,
        maxrej,
        cenfunc,
        clip_cen,
        revert_on_nkeep,
        rdnoise_ref,
        snoise_ref,
        scales,
        zeros,
        gain,
        validate,
    )?;
    dispatch_rejection_combine(
        py,
        arr,
        mask,
        |v, m| k_ccdclip_median(&v, m, &params),
        |v, m| k_ccdclip_median(&v, m, &params),
        validate,
    )
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    values,
    *,
    mask = None,
    sigma_lower = 3.0,
    sigma_upper = 3.0,
    maxiters = 5,
    ddof = 0,
    nkeep = 1,
    maxrej = None,
    cenfunc = "median",
    clip_cen = "mean",
    revert_on_nkeep = true,
    rdnoise_ref = 0.0,
    snoise_ref = 0.0,
    scales = None,
    zeros = None,
    gain = 1.0,
    validate = true,
))]
fn ccdclip_median_1d<'py>(
    values: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArray1<'py, bool>>,
    sigma_lower: f64,
    sigma_upper: f64,
    maxiters: usize,
    ddof: usize,
    nkeep: usize,
    maxrej: Option<usize>,
    cenfunc: &str,
    clip_cen: &str,
    revert_on_nkeep: bool,
    rdnoise_ref: f64,
    snoise_ref: f64,
    scales: Option<Vec<f64>>,
    zeros: Option<Vec<f64>>,
    gain: f64,
    validate: bool,
) -> PyResult<f64> {
    let n = values_len_1d(values)?;
    let params = make_ccdclip_params(
        n,
        sigma_lower,
        sigma_upper,
        maxiters,
        ddof,
        nkeep,
        maxrej,
        cenfunc,
        clip_cen,
        revert_on_nkeep,
        rdnoise_ref,
        snoise_ref,
        scales,
        zeros,
        gain,
        validate,
    )?;
    dispatch_rejection_combine_1d(
        values,
        mask,
        |v, m| k_ccdclip_median_1d(v, m, &params),
        |v, m| k_ccdclip_median_1d(v, m, &params),
        validate,
    )
}

#[pyfunction]
#[pyo3(signature = (
    arr,
    *,
    mask = None,
    low_scale = 1.0,
    low = 0.0,
    upp_scale = 1.0,
    upp = 0.0,
    maxiters = 1,
    nkeep = 1,
    maxrej = None,
    cenfunc = "median",
    revert_on_nkeep = true,
    validate = true,
))]
#[allow(clippy::too_many_arguments)]
fn linearclip<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArray3<'py, bool>>,
    low_scale: f64,
    low: f64,
    upp_scale: f64,
    upp: f64,
    maxiters: usize,
    nkeep: usize,
    maxrej: Option<usize>,
    cenfunc: &str,
    revert_on_nkeep: bool,
    validate: bool,
) -> PyResult<Bound<'py, PyTuple>> {
    if validate {
        if !low_scale.is_finite() || !low.is_finite() || !upp_scale.is_finite() || !upp.is_finite()
        {
            return Err(PyValueError::new_err(
                "linearclip parameters must be finite",
            ));
        }
        if low_scale < 0.0 {
            return Err(PyValueError::new_err("low_scale must be >= 0"));
        }
        if upp_scale < 0.0 {
            return Err(PyValueError::new_err("upp_scale must be >= 0"));
        }
        if low < 0.0 {
            return Err(PyValueError::new_err("low must be >= 0"));
        }
        if upp < 0.0 {
            return Err(PyValueError::new_err("upp must be >= 0"));
        }
    }
    let n = stack_size(arr)?;
    let params = LinearClipParams {
        low_scale,
        low,
        upp_scale,
        upp,
        maxiters,
        nkeep,
        maxrej: maxrej.unwrap_or(n),
        cenfunc: parse_cenfunc(cenfunc)?,
        revert_on_nkeep,
    };
    dispatch_minmax_or_pclip(
        py,
        arr,
        mask,
        |v, m| k_linearclip(&v, m, &params),
        |v, m| k_linearclip(&v, m, &params),
        validate,
    )
}

#[pyfunction]
#[pyo3(signature = (
    values,
    *,
    mask = None,
    low_scale = 1.0,
    low = 0.0,
    upp_scale = 1.0,
    upp = 0.0,
    maxiters = 1,
    nkeep = 1,
    maxrej = None,
    cenfunc = "median",
    revert_on_nkeep = true,
    validate = true,
))]
#[allow(clippy::too_many_arguments)]
fn linearclip_1d<'py>(
    py: Python<'py>,
    values: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArray1<'py, bool>>,
    low_scale: f64,
    low: f64,
    upp_scale: f64,
    upp: f64,
    maxiters: usize,
    nkeep: usize,
    maxrej: Option<usize>,
    cenfunc: &str,
    revert_on_nkeep: bool,
    validate: bool,
) -> PyResult<Bound<'py, PyTuple>> {
    if validate {
        if !low_scale.is_finite() || !low.is_finite() || !upp_scale.is_finite() || !upp.is_finite()
        {
            return Err(PyValueError::new_err(
                "linearclip parameters must be finite",
            ));
        }
        if low_scale < 0.0 {
            return Err(PyValueError::new_err("low_scale must be >= 0"));
        }
        if upp_scale < 0.0 {
            return Err(PyValueError::new_err("upp_scale must be >= 0"));
        }
        if low < 0.0 {
            return Err(PyValueError::new_err("low must be >= 0"));
        }
        if upp < 0.0 {
            return Err(PyValueError::new_err("upp must be >= 0"));
        }
    }
    let n = values_len_1d(values)?;
    let params = LinearClipParams {
        low_scale,
        low,
        upp_scale,
        upp,
        maxiters,
        nkeep,
        maxrej: maxrej.unwrap_or(n),
        cenfunc: parse_cenfunc(cenfunc)?,
        revert_on_nkeep,
    };
    dispatch_rejection_1d(
        py,
        values,
        mask,
        |v, m| k_linearclip_1d(v, m, &params),
        |v, m| k_linearclip_1d(v, m, &params),
        validate,
    )
}

#[pyfunction]
#[pyo3(signature = (
    arr,
    *,
    mask = None,
    low_scale = 1.0,
    low = 0.0,
    upp_scale = 1.0,
    upp = 0.0,
    maxiters = 1,
    nkeep = 1,
    maxrej = None,
    cenfunc = "median",
    revert_on_nkeep = true,
    validate = true,
))]
#[allow(clippy::too_many_arguments)]
fn linearclip_restored_flags<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArray3<'py, bool>>,
    low_scale: f64,
    low: f64,
    upp_scale: f64,
    upp: f64,
    maxiters: usize,
    nkeep: usize,
    maxrej: Option<usize>,
    cenfunc: &str,
    revert_on_nkeep: bool,
    validate: bool,
) -> PyResult<Bound<'py, PyAny>> {
    if validate {
        if !low_scale.is_finite() || !low.is_finite() || !upp_scale.is_finite() || !upp.is_finite()
        {
            return Err(PyValueError::new_err(
                "linearclip parameters must be finite",
            ));
        }
        if low_scale < 0.0 {
            return Err(PyValueError::new_err("low_scale must be >= 0"));
        }
        if upp_scale < 0.0 {
            return Err(PyValueError::new_err("upp_scale must be >= 0"));
        }
        if low < 0.0 {
            return Err(PyValueError::new_err("low must be >= 0"));
        }
        if upp < 0.0 {
            return Err(PyValueError::new_err("upp must be >= 0"));
        }
    }
    let n = stack_size(arr)?;
    let params = LinearClipParams {
        low_scale,
        low,
        upp_scale,
        upp,
        maxiters,
        nkeep,
        maxrej: maxrej.unwrap_or(n),
        cenfunc: parse_cenfunc(cenfunc)?,
        revert_on_nkeep,
    };
    dispatch_rejection_flags(
        py,
        arr,
        mask,
        |v, m| k_linearclip_restored_flags(&v, m, &params),
        |v, m| k_linearclip_restored_flags(&v, m, &params),
        validate,
    )
}

#[pyfunction]
#[pyo3(signature = (arr, *, mask = None, n_min = 1, n_max = 1, validate = true))]
fn minmax<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArray3<'py, bool>>,
    n_min: usize,
    n_max: usize,
    validate: bool,
) -> PyResult<Bound<'py, PyTuple>> {
    let n = stack_size(arr)?;
    let q_low = n_min as f64 / n as f64;
    let q_upp = n_max as f64 / n as f64;
    dispatch_minmax_or_pclip(
        py,
        arr,
        mask,
        |v, m| k_minmax(&v, m, q_low, q_upp),
        |v, m| k_minmax(&v, m, q_low, q_upp),
        validate,
    )
}

#[pyfunction]
#[pyo3(signature = (values, *, mask = None, n_min = 1, n_max = 1, validate = true))]
fn minmax_1d<'py>(
    py: Python<'py>,
    values: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArray1<'py, bool>>,
    n_min: usize,
    n_max: usize,
    validate: bool,
) -> PyResult<Bound<'py, PyTuple>> {
    let n = values_len_1d(values)?;
    let q_low = n_min as f64 / n as f64;
    let q_upp = n_max as f64 / n as f64;
    dispatch_rejection_1d(
        py,
        values,
        mask,
        |v, m| k_minmax_1d(v, m, q_low, q_upp),
        |v, m| k_minmax_1d(v, m, q_low, q_upp),
        validate,
    )
}

#[pyfunction]
#[pyo3(signature = (
    arr,
    *,
    mask = None,
    pclip = -0.5,
    sigma_lower = 3.0,
    sigma_upper = 3.0,
    nkeep = 1,
    validate = true,
))]
#[allow(clippy::too_many_arguments)]
fn pclip<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArray3<'py, bool>>,
    pclip: f64,
    sigma_lower: f64,
    sigma_upper: f64,
    nkeep: usize,
    validate: bool,
) -> PyResult<Bound<'py, PyTuple>> {
    if validate && (pclip == 0.0 || !pclip.is_finite()) {
        return Err(PyValueError::new_err("pclip must be finite and non-zero"));
    }
    dispatch_minmax_or_pclip(
        py,
        arr,
        mask,
        |v, m| k_pclip(&v, m, pclip, sigma_lower, sigma_upper, nkeep),
        |v, m| k_pclip(&v, m, pclip, sigma_lower, sigma_upper, nkeep),
        validate,
    )
}

#[pyfunction]
#[pyo3(signature = (
    values,
    *,
    mask = None,
    pclip = -0.5,
    sigma_lower = 3.0,
    sigma_upper = 3.0,
    nkeep = 1,
    validate = true,
))]
#[allow(clippy::too_many_arguments)]
fn pclip_1d<'py>(
    py: Python<'py>,
    values: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArray1<'py, bool>>,
    pclip: f64,
    sigma_lower: f64,
    sigma_upper: f64,
    nkeep: usize,
    validate: bool,
) -> PyResult<Bound<'py, PyTuple>> {
    if validate && (pclip == 0.0 || !pclip.is_finite()) {
        return Err(PyValueError::new_err("pclip must be finite and non-zero"));
    }
    dispatch_rejection_1d(
        py,
        values,
        mask,
        |v, m| k_pclip_1d(v, m, pclip, sigma_lower, sigma_upper, nkeep),
        |v, m| k_pclip_1d(v, m, pclip, sigma_lower, sigma_upper, nkeep),
        validate,
    )
}

#[pyfunction]
#[pyo3(signature = (arr, *, mask = None, n_min = 1, n_max = 1, validate = true))]
fn minmax_mask<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArray3<'py, bool>>,
    n_min: usize,
    n_max: usize,
    validate: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let n = stack_size(arr)?;
    let q_low = n_min as f64 / n as f64;
    let q_upp = n_max as f64 / n as f64;
    dispatch_rejection_mask(
        py,
        arr,
        mask,
        |v, m| k_minmax_mask(&v, m, q_low, q_upp),
        |v, m| k_minmax_mask(&v, m, q_low, q_upp),
        validate,
    )
}

#[pyfunction]
#[pyo3(signature = (values, *, mask = None, n_min = 1, n_max = 1, validate = true))]
fn minmax_mask_1d<'py>(
    py: Python<'py>,
    values: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArray1<'py, bool>>,
    n_min: usize,
    n_max: usize,
    validate: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let n = values_len_1d(values)?;
    let q_low = n_min as f64 / n as f64;
    let q_upp = n_max as f64 / n as f64;
    dispatch_rejection_mask_1d(
        py,
        values,
        mask,
        |v, m| k_minmax_mask_1d(v, m, q_low, q_upp),
        |v, m| k_minmax_mask_1d(v, m, q_low, q_upp),
        validate,
    )
}

#[pyfunction]
#[pyo3(signature = (arr, *, mask = None, n_min = 1, n_max = 1, validate = true))]
fn minmax_mean<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArray3<'py, bool>>,
    n_min: usize,
    n_max: usize,
    validate: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let n = stack_size(arr)?;
    let q_low = n_min as f64 / n as f64;
    let q_upp = n_max as f64 / n as f64;
    dispatch_rejection_combine(
        py,
        arr,
        mask,
        |v, m| k_minmax_mean(&v, m, q_low, q_upp),
        |v, m| k_minmax_mean(&v, m, q_low, q_upp),
        validate,
    )
}

#[pyfunction]
#[pyo3(signature = (values, *, mask = None, n_min = 1, n_max = 1, validate = true))]
fn minmax_mean_1d<'py>(
    values: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArray1<'py, bool>>,
    n_min: usize,
    n_max: usize,
    validate: bool,
) -> PyResult<f64> {
    let n = values_len_1d(values)?;
    let q_low = n_min as f64 / n as f64;
    let q_upp = n_max as f64 / n as f64;
    dispatch_rejection_combine_1d(
        values,
        mask,
        |v, m| k_minmax_mean_1d(v, m, q_low, q_upp),
        |v, m| k_minmax_mean_1d(v, m, q_low, q_upp),
        validate,
    )
}

#[pyfunction]
#[pyo3(signature = (arr, *, mask = None, n_min = 1, n_max = 1, validate = true))]
fn minmax_median<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArray3<'py, bool>>,
    n_min: usize,
    n_max: usize,
    validate: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let n = stack_size(arr)?;
    let q_low = n_min as f64 / n as f64;
    let q_upp = n_max as f64 / n as f64;
    dispatch_rejection_combine(
        py,
        arr,
        mask,
        |v, m| k_minmax_median(&v, m, q_low, q_upp),
        |v, m| k_minmax_median(&v, m, q_low, q_upp),
        validate,
    )
}

#[pyfunction]
#[pyo3(signature = (values, *, mask = None, n_min = 1, n_max = 1, validate = true))]
fn minmax_median_1d<'py>(
    values: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArray1<'py, bool>>,
    n_min: usize,
    n_max: usize,
    validate: bool,
) -> PyResult<f64> {
    let n = values_len_1d(values)?;
    let q_low = n_min as f64 / n as f64;
    let q_upp = n_max as f64 / n as f64;
    dispatch_rejection_combine_1d(
        values,
        mask,
        |v, m| k_minmax_median_1d(v, m, q_low, q_upp),
        |v, m| k_minmax_median_1d(v, m, q_low, q_upp),
        validate,
    )
}

#[pyfunction]
#[pyo3(signature = (
    arr,
    *,
    mask = None,
    pclip = -0.5,
    sigma_lower = 3.0,
    sigma_upper = 3.0,
    nkeep = 1,
    validate = true,
))]
#[allow(clippy::too_many_arguments)]
fn pclip_mask<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArray3<'py, bool>>,
    pclip: f64,
    sigma_lower: f64,
    sigma_upper: f64,
    nkeep: usize,
    validate: bool,
) -> PyResult<Bound<'py, PyAny>> {
    if validate && (pclip == 0.0 || !pclip.is_finite()) {
        return Err(PyValueError::new_err("pclip must be finite and non-zero"));
    }
    dispatch_rejection_mask(
        py,
        arr,
        mask,
        |v, m| k_pclip_mask(&v, m, pclip, sigma_lower, sigma_upper, nkeep),
        |v, m| k_pclip_mask(&v, m, pclip, sigma_lower, sigma_upper, nkeep),
        validate,
    )
}

#[pyfunction]
#[pyo3(signature = (
    values,
    *,
    mask = None,
    pclip = -0.5,
    sigma_lower = 3.0,
    sigma_upper = 3.0,
    nkeep = 1,
    validate = true,
))]
#[allow(clippy::too_many_arguments)]
fn pclip_mask_1d<'py>(
    py: Python<'py>,
    values: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArray1<'py, bool>>,
    pclip: f64,
    sigma_lower: f64,
    sigma_upper: f64,
    nkeep: usize,
    validate: bool,
) -> PyResult<Bound<'py, PyAny>> {
    if validate && (pclip == 0.0 || !pclip.is_finite()) {
        return Err(PyValueError::new_err("pclip must be finite and non-zero"));
    }
    dispatch_rejection_mask_1d(
        py,
        values,
        mask,
        |v, m| k_pclip_mask_1d(v, m, pclip, sigma_lower, sigma_upper, nkeep),
        |v, m| k_pclip_mask_1d(v, m, pclip, sigma_lower, sigma_upper, nkeep),
        validate,
    )
}

#[pyfunction]
#[pyo3(signature = (
    arr,
    *,
    mask = None,
    pclip = -0.5,
    sigma_lower = 3.0,
    sigma_upper = 3.0,
    nkeep = 1,
    validate = true,
))]
#[allow(clippy::too_many_arguments)]
fn pclip_mean<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArray3<'py, bool>>,
    pclip: f64,
    sigma_lower: f64,
    sigma_upper: f64,
    nkeep: usize,
    validate: bool,
) -> PyResult<Bound<'py, PyAny>> {
    if validate && (pclip == 0.0 || !pclip.is_finite()) {
        return Err(PyValueError::new_err("pclip must be finite and non-zero"));
    }
    dispatch_rejection_combine(
        py,
        arr,
        mask,
        |v, m| k_pclip_mean(&v, m, pclip, sigma_lower, sigma_upper, nkeep),
        |v, m| k_pclip_mean(&v, m, pclip, sigma_lower, sigma_upper, nkeep),
        validate,
    )
}

#[pyfunction]
#[pyo3(signature = (
    values,
    *,
    mask = None,
    pclip = -0.5,
    sigma_lower = 3.0,
    sigma_upper = 3.0,
    nkeep = 1,
    validate = true,
))]
#[allow(clippy::too_many_arguments)]
fn pclip_mean_1d<'py>(
    values: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArray1<'py, bool>>,
    pclip: f64,
    sigma_lower: f64,
    sigma_upper: f64,
    nkeep: usize,
    validate: bool,
) -> PyResult<f64> {
    if validate && (pclip == 0.0 || !pclip.is_finite()) {
        return Err(PyValueError::new_err("pclip must be finite and non-zero"));
    }
    dispatch_rejection_combine_1d(
        values,
        mask,
        |v, m| k_pclip_mean_1d(v, m, pclip, sigma_lower, sigma_upper, nkeep),
        |v, m| k_pclip_mean_1d(v, m, pclip, sigma_lower, sigma_upper, nkeep),
        validate,
    )
}

#[pyfunction]
#[pyo3(signature = (
    arr,
    *,
    mask = None,
    pclip = -0.5,
    sigma_lower = 3.0,
    sigma_upper = 3.0,
    nkeep = 1,
    validate = true,
))]
#[allow(clippy::too_many_arguments)]
fn pclip_median<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArray3<'py, bool>>,
    pclip: f64,
    sigma_lower: f64,
    sigma_upper: f64,
    nkeep: usize,
    validate: bool,
) -> PyResult<Bound<'py, PyAny>> {
    if validate && (pclip == 0.0 || !pclip.is_finite()) {
        return Err(PyValueError::new_err("pclip must be finite and non-zero"));
    }
    dispatch_rejection_combine(
        py,
        arr,
        mask,
        |v, m| k_pclip_median(&v, m, pclip, sigma_lower, sigma_upper, nkeep),
        |v, m| k_pclip_median(&v, m, pclip, sigma_lower, sigma_upper, nkeep),
        validate,
    )
}

#[pyfunction]
#[pyo3(signature = (
    values,
    *,
    mask = None,
    pclip = -0.5,
    sigma_lower = 3.0,
    sigma_upper = 3.0,
    nkeep = 1,
    validate = true,
))]
#[allow(clippy::too_many_arguments)]
fn pclip_median_1d<'py>(
    values: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArray1<'py, bool>>,
    pclip: f64,
    sigma_lower: f64,
    sigma_upper: f64,
    nkeep: usize,
    validate: bool,
) -> PyResult<f64> {
    if validate && (pclip == 0.0 || !pclip.is_finite()) {
        return Err(PyValueError::new_err("pclip must be finite and non-zero"));
    }
    dispatch_rejection_combine_1d(
        values,
        mask,
        |v, m| k_pclip_median_1d(v, m, pclip, sigma_lower, sigma_upper, nkeep),
        |v, m| k_pclip_median_1d(v, m, pclip, sigma_lower, sigma_upper, nkeep),
        validate,
    )
}

#[pyfunction]
#[pyo3(signature = (mask, grow))]
fn grow_mask<'py>(
    py: Python<'py>,
    mask: PyReadonlyArrayDyn<'py, bool>,
    grow: f64,
) -> PyResult<Bound<'py, PyAny>> {
    let out = k_grow_mask(&mask.as_array(), grow).map_err(PyValueError::new_err)?;
    Ok(out.into_pyarray(py).into_any())
}

// ---- shared dtype dispatch for the non-iterative rejections ------------------------

fn dispatch_minmax_or_pclip<'py, F32, F64>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArray3<'py, bool>>,
    f32_kernel: F32,
    f64_kernel: F64,
    validate: bool,
) -> PyResult<Bound<'py, PyTuple>>
where
    F32: FnOnce(
        ndarray::ArrayView3<f32>,
        Option<&ndarray::ArrayView3<bool>>,
    ) -> crate::kernel::reject::RejectOutput<f32>,
    F64: FnOnce(
        ndarray::ArrayView3<f64>,
        Option<&ndarray::ArrayView3<bool>>,
    ) -> crate::kernel::reject::RejectOutput<f64>,
{
    if let Ok(a) = arr.cast::<PyArray3<f32>>() {
        let a = a.readonly();
        let view = a.as_array();
        let mview = mask.as_ref().map(|m| m.as_array());
        if validate {
            validate_reject_inputs(view.shape(), mview.as_ref().map(|m| m.shape()))?;
        }
        let out = f32_kernel(view, mview.as_ref());
        Ok(reject_to_tuple::<f32>(py, out))
    } else if let Ok(a) = arr.cast::<PyArray3<f64>>() {
        let a = a.readonly();
        let view = a.as_array();
        let mview = mask.as_ref().map(|m| m.as_array());
        if validate {
            validate_reject_inputs(view.shape(), mview.as_ref().map(|m| m.shape()))?;
        }
        let out = f64_kernel(view, mview.as_ref());
        Ok(reject_to_tuple::<f64>(py, out))
    } else {
        Err(PyTypeError::new_err(
            "arr must be a 3-D float32 or float64 NumPy array",
        ))
    }
}

fn dispatch_rejection_mask<'py, F32, F64>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArray3<'py, bool>>,
    f32_kernel: F32,
    f64_kernel: F64,
    validate: bool,
) -> PyResult<Bound<'py, PyAny>>
where
    F32: FnOnce(
        ndarray::ArrayView3<f32>,
        Option<&ndarray::ArrayView3<bool>>,
    ) -> ndarray::Array3<bool>,
    F64: FnOnce(
        ndarray::ArrayView3<f64>,
        Option<&ndarray::ArrayView3<bool>>,
    ) -> ndarray::Array3<bool>,
{
    if let Ok(a) = arr.cast::<PyArray3<f32>>() {
        let a = a.readonly();
        let view = a.as_array();
        let mview = mask.as_ref().map(|m| m.as_array());
        if validate {
            validate_reject_inputs(view.shape(), mview.as_ref().map(|m| m.shape()))?;
        }
        let out = f32_kernel(view, mview.as_ref());
        Ok(out.into_pyarray(py).into_any())
    } else if let Ok(a) = arr.cast::<PyArray3<f64>>() {
        let a = a.readonly();
        let view = a.as_array();
        let mview = mask.as_ref().map(|m| m.as_array());
        if validate {
            validate_reject_inputs(view.shape(), mview.as_ref().map(|m| m.shape()))?;
        }
        let out = f64_kernel(view, mview.as_ref());
        Ok(out.into_pyarray(py).into_any())
    } else {
        Err(PyTypeError::new_err(
            "arr must be a 3-D float32 or float64 NumPy array",
        ))
    }
}

fn dispatch_rejection_flags<'py, F32, F64>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArray3<'py, bool>>,
    f32_kernel: F32,
    f64_kernel: F64,
    validate: bool,
) -> PyResult<Bound<'py, PyAny>>
where
    F32:
        FnOnce(ndarray::ArrayView3<f32>, Option<&ndarray::ArrayView3<bool>>) -> ndarray::Array3<u8>,
    F64:
        FnOnce(ndarray::ArrayView3<f64>, Option<&ndarray::ArrayView3<bool>>) -> ndarray::Array3<u8>,
{
    if let Ok(a) = arr.cast::<PyArray3<f32>>() {
        let a = a.readonly();
        let view = a.as_array();
        let mview = mask.as_ref().map(|m| m.as_array());
        if validate {
            validate_reject_inputs(view.shape(), mview.as_ref().map(|m| m.shape()))?;
        }
        let out = f32_kernel(view, mview.as_ref());
        Ok(out.into_pyarray(py).into_any())
    } else if let Ok(a) = arr.cast::<PyArray3<f64>>() {
        let a = a.readonly();
        let view = a.as_array();
        let mview = mask.as_ref().map(|m| m.as_array());
        if validate {
            validate_reject_inputs(view.shape(), mview.as_ref().map(|m| m.shape()))?;
        }
        let out = f64_kernel(view, mview.as_ref());
        Ok(out.into_pyarray(py).into_any())
    } else {
        Err(PyTypeError::new_err(
            "arr must be a 3-D float32 or float64 NumPy array",
        ))
    }
}

fn dispatch_rejection_combine<'py, F32, F64>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArray3<'py, bool>>,
    f32_kernel: F32,
    f64_kernel: F64,
    validate: bool,
) -> PyResult<Bound<'py, PyAny>>
where
    F32: FnOnce(
        ndarray::ArrayView3<f32>,
        Option<&ndarray::ArrayView3<bool>>,
    ) -> ndarray::Array2<f32>,
    F64: FnOnce(
        ndarray::ArrayView3<f64>,
        Option<&ndarray::ArrayView3<bool>>,
    ) -> ndarray::Array2<f64>,
{
    if let Ok(a) = arr.cast::<PyArray3<f32>>() {
        let a = a.readonly();
        let view = a.as_array();
        let mview = mask.as_ref().map(|m| m.as_array());
        if validate {
            validate_reject_inputs(view.shape(), mview.as_ref().map(|m| m.shape()))?;
        }
        let out = f32_kernel(view, mview.as_ref());
        Ok(out.into_pyarray(py).into_any())
    } else if let Ok(a) = arr.cast::<PyArray3<f64>>() {
        let a = a.readonly();
        let view = a.as_array();
        let mview = mask.as_ref().map(|m| m.as_array());
        if validate {
            validate_reject_inputs(view.shape(), mview.as_ref().map(|m| m.shape()))?;
        }
        let out = f64_kernel(view, mview.as_ref());
        Ok(out.into_pyarray(py).into_any())
    } else {
        Err(PyTypeError::new_err(
            "arr must be a 3-D float32 or float64 NumPy array",
        ))
    }
}

fn dispatch_rejection_1d<'py, F32, F64>(
    py: Python<'py>,
    values: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArray1<'py, bool>>,
    f32_kernel: F32,
    f64_kernel: F64,
    validate: bool,
) -> PyResult<Bound<'py, PyTuple>>
where
    F32: FnOnce(&[f32], Option<&[bool]>) -> crate::kernel::reject::RejectOutput1d<f32>,
    F64: FnOnce(&[f64], Option<&[bool]>) -> crate::kernel::reject::RejectOutput1d<f64>,
{
    if let Ok(a) = values.cast::<PyArray1<f32>>() {
        let a = a.readonly();
        let values = a.as_slice().unwrap();
        let mask_slice = mask.as_ref().map(|m| m.as_slice().unwrap());
        if validate {
            validate_1d_inputs(values.len(), mask_slice.map(<[bool]>::len))?;
        }
        let out = f32_kernel(values, mask_slice);
        Ok(reject_1d_to_tuple::<f32>(py, out))
    } else if let Ok(a) = values.cast::<PyArray1<f64>>() {
        let a = a.readonly();
        let values = a.as_slice().unwrap();
        let mask_slice = mask.as_ref().map(|m| m.as_slice().unwrap());
        if validate {
            validate_1d_inputs(values.len(), mask_slice.map(<[bool]>::len))?;
        }
        let out = f64_kernel(values, mask_slice);
        Ok(reject_1d_to_tuple::<f64>(py, out))
    } else {
        Err(PyTypeError::new_err(
            "values must be a 1-D float32 or float64 NumPy array",
        ))
    }
}

fn dispatch_rejection_mask_1d<'py, F32, F64>(
    py: Python<'py>,
    values: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArray1<'py, bool>>,
    f32_kernel: F32,
    f64_kernel: F64,
    validate: bool,
) -> PyResult<Bound<'py, PyAny>>
where
    F32: FnOnce(&[f32], Option<&[bool]>) -> Vec<bool>,
    F64: FnOnce(&[f64], Option<&[bool]>) -> Vec<bool>,
{
    if let Ok(a) = values.cast::<PyArray1<f32>>() {
        let a = a.readonly();
        let values = a.as_slice().unwrap();
        let mask_slice = mask.as_ref().map(|m| m.as_slice().unwrap());
        if validate {
            validate_1d_inputs(values.len(), mask_slice.map(<[bool]>::len))?;
        }
        Ok(f32_kernel(values, mask_slice).into_pyarray(py).into_any())
    } else if let Ok(a) = values.cast::<PyArray1<f64>>() {
        let a = a.readonly();
        let values = a.as_slice().unwrap();
        let mask_slice = mask.as_ref().map(|m| m.as_slice().unwrap());
        if validate {
            validate_1d_inputs(values.len(), mask_slice.map(<[bool]>::len))?;
        }
        Ok(f64_kernel(values, mask_slice).into_pyarray(py).into_any())
    } else {
        Err(PyTypeError::new_err(
            "values must be a 1-D float32 or float64 NumPy array",
        ))
    }
}

fn dispatch_rejection_combine_1d<'py, F32, F64>(
    values: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArray1<'py, bool>>,
    f32_kernel: F32,
    f64_kernel: F64,
    validate: bool,
) -> PyResult<f64>
where
    F32: FnOnce(&[f32], Option<&[bool]>) -> f32,
    F64: FnOnce(&[f64], Option<&[bool]>) -> f64,
{
    if let Ok(a) = values.cast::<PyArray1<f32>>() {
        let a = a.readonly();
        let values = a.as_slice().unwrap();
        let mask_slice = mask.as_ref().map(|m| m.as_slice().unwrap());
        if validate {
            validate_1d_inputs(values.len(), mask_slice.map(<[bool]>::len))?;
        }
        Ok(f32_kernel(values, mask_slice).to_f64())
    } else if let Ok(a) = values.cast::<PyArray1<f64>>() {
        let a = a.readonly();
        let values = a.as_slice().unwrap();
        let mask_slice = mask.as_ref().map(|m| m.as_slice().unwrap());
        if validate {
            validate_1d_inputs(values.len(), mask_slice.map(<[bool]>::len))?;
        }
        Ok(f64_kernel(values, mask_slice).to_f64())
    } else {
        Err(PyTypeError::new_err(
            "values must be a 1-D float32 or float64 NumPy array",
        ))
    }
}

/// Peek the stack size (N) from either f32 or f64 typed array.
fn stack_size(arr: &Bound<'_, PyAny>) -> PyResult<usize> {
    if let Ok(a) = arr.cast::<PyArray3<f32>>() {
        Ok(a.readonly().as_array().shape()[0])
    } else if let Ok(a) = arr.cast::<PyArray3<f64>>() {
        Ok(a.readonly().as_array().shape()[0])
    } else {
        Err(PyTypeError::new_err(
            "arr must be a 3-D float32 or float64 NumPy array",
        ))
    }
}

fn values_len_1d(values: &Bound<'_, PyAny>) -> PyResult<usize> {
    if let Ok(a) = values.cast::<PyArray1<f32>>() {
        Ok(a.readonly().as_array().shape()[0])
    } else if let Ok(a) = values.cast::<PyArray1<f64>>() {
        Ok(a.readonly().as_array().shape()[0])
    } else {
        Err(PyTypeError::new_err(
            "values must be a 1-D float32 or float64 NumPy array",
        ))
    }
}

fn validate_stack_size(n: usize) -> PyResult<usize> {
    if n == 0 {
        Err(PyValueError::new_err(
            "arr must contain at least one image along axis 0",
        ))
    } else {
        Ok(n)
    }
}

fn validate_1d_inputs(values_len: usize, mask_len: Option<usize>) -> PyResult<()> {
    validate_stack_size(values_len)?;
    if let Some(mask_len) = mask_len {
        if mask_len != values_len {
            return Err(PyValueError::new_err(format!(
                "mask length {mask_len} does not match values length {values_len}",
            )));
        }
    }
    Ok(())
}

fn validate_reject_inputs(arr_shape: &[usize], mask_shape: Option<&[usize]>) -> PyResult<()> {
    validate_stack_size(arr_shape[0])?;
    if let Some(mask_shape) = mask_shape {
        if mask_shape != arr_shape {
            return Err(PyValueError::new_err(format!(
                "mask shape {:?} does not match arr shape {:?}",
                mask_shape, arr_shape
            )));
        }
    }
    Ok(())
}
