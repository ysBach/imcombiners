"""Focused, single-purpose combine + rejection kernels.

This module mirrors the pattern used by focused kernel namespaces: one Python
function per *operation*, each with only the kwargs that apply to it. These are
direct kernel calls for callers building their own pipeline layer; most
user-facing workflows should start with Standard Combiner or
compact ndcombine() wrapper.

Most functions accept an image stack of shape ``(N, *spatial)`` and normalize
supported integer dtypes to a floating workspace when validation is enabled.
Inputs with more than 3 dimensions are flattened to ``(N, prod(spatial), 1)``
internally; outputs are reshaped back to match the input trailing dimensions.
Accepted public dtypes are ``uint8``, ``uint16``, ``int16``, ``int32``,
``float32``, and ``float64``. Other dtypes, including ``int64`` and
``float128``, are not silently cast. Pure stack reductions delegate to
``reducers`` with finite-only semantics: both ``NaN`` and ``inf`` are skipped.
Combine kernels return arrays of shape ``(*spatial,)``. Rejection kernels
return the 6-tuple ``(mask_rej, std, low, upp, nit, output_flags)``.
`mask_rej` has shape ``(N, *spatial)``. `low`, `upp`, `nit`, `output_flags`, and
sigma/CCD `std` arrays have shape ``(*spatial,)``; `std` is `None` for
rejection algorithms without a spread diagnostic.
``mask_rej.sum(axis=0)`` is the number rejected by that rejection kernel. The
number actually used for a later combine is the count of finite values after
input masks and rejection masks are applied.
When `grow` is used by a rejection kernel, `mask_rej` is the final grown
rejection mask. The `low`, `upp`, `nit`, and `std` diagnostics still describe
the underlying clipping calculation; `output_flags` bit ``16`` marks output elements
where growth added at least one rejected sample.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import reducers as rd

from . import _core, _doc
from ._typing import RejectionResult
from ._validation import (
    validate_lmedian_stack,
    validate_mask,
    validate_stack,
    validate_values_1d,
    validate_weights,
)

__all__ = [
    # Weighted combine
    "nanaverage",
    # Reject
    "sigclip",
    "sigclip_1d",
    "sigclip_mask",
    "sigclip_mask_1d",
    "sigclip_combine",
    "sigclip_combine_1d",
    "ccdclip",
    "ccdclip_1d",
    "ccdclip_mask",
    "ccdclip_mask_1d",
    "ccdclip_combine",
    "ccdclip_combine_1d",
    "linearclip",
    "linearclip_1d",
    "minmax",
    "minmax_1d",
    "minmax_mask",
    "minmax_mask_1d",
    "minmax_combine",
    "minmax_combine_1d",
    "pclip",
    "pclip_1d",
    "pclip_mask",
    "pclip_mask_1d",
    "pclip_combine",
    "pclip_combine_1d",
    "grow_mask",
]


def grow_mask(mask: np.ndarray, grow: float, *, validate: bool = True) -> np.ndarray:
    """Dilate a stack mask over spatial axes by an exact Euclidean radius.

    Axis 0 is the stack axis and is never grown across. For each input plane,
    every `True` sample marks all spatial samples whose integer-grid Euclidean
    distance is less than or equal to `grow`.

    Parameters
    ----------
    mask : ndarray of bool, shape (N, *spatial)
        Rejection mask to grow. `True` values are expanded within each plane.
    grow : float
        Non-negative radius in pixels. `0` returns an unchanged copy. Values
        below `1` grow no additional integer-grid samples.
    validate : bool, optional
        If `True`, validate dimensionality, dtype, finiteness, and contiguity
        before entering the Rust kernel. If `False`, callers must provide a
        C-contiguous boolean array with at least two dimensions and a finite
        non-negative radius.

    Returns
    -------
    grown : ndarray of bool, shape (N, *spatial)
        Grown mask with the same shape as `mask`.
    """
    if validate:
        grow = float(grow)
        if not np.isfinite(grow) or grow < 0.0:
            raise ValueError("grow must be a finite non-negative radius")
        mask = np.asarray(mask)
        if mask.ndim < 2:
            raise ValueError(
                f"grow mask must have shape (N, *spatial); got {mask.shape}"
            )
        if mask.dtype != np.bool_:
            raise TypeError(f"grow mask must have dtype bool; got {mask.dtype}")
        mask = np.ascontiguousarray(mask)
    return _core.grow_mask(mask, grow)


def _grow_rejection_mask(
    mask_rej: np.ndarray,
    output_flags: np.ndarray,
    grow: float | None,
    *,
    validate: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Grow a rejection mask and set output_flags bit 16 where growth added samples."""
    if grow is None:
        return mask_rej, output_flags
    grown = grow_mask(mask_rej, grow, validate=validate)
    added = grown & ~mask_rej
    if np.any(added):
        output_flags = output_flags.copy()
        output_flags[np.any(added, axis=0)] |= np.uint8(16)
    return grown, output_flags


def _grow_rejection_mask_only(
    mask_rej: np.ndarray,
    grow: float | None,
    *,
    validate: bool,
) -> np.ndarray:
    """Grow a rejection mask without allocating diagnostic output_flags maps."""
    if grow is None:
        return mask_rej
    return grow_mask(mask_rej, grow, validate=validate)


# ---- combine ------------------------------------------------------------------------


_STACK_FUNCS = {
    "mean": rd.nanmean,
    "average": rd.nanmean,
    "avg": rd.nanmean,
    "median": rd.nanmedian,
    "med": rd.nanmedian,
    "lmedian": rd.lmedian,
    "lmed": rd.lmedian,
    "sum": rd.nansum,
    "min": rd.nanmin,
    "max": rd.nanmax,
    "variance": rd.nanvar,
    "var": rd.nanvar,
}


def _stack(
    arr: np.ndarray,
    fun,
    *,
    validate: bool = True,
    ignore_inf: bool = True,
    **kwargs,
) -> np.ndarray:
    """Return a finite-only axis-0 stack reduction using a `reducers` function."""
    trailing = arr.shape[1:]
    dtype = np.asarray(arr).dtype

    if fun is rd.lmedian:
        if validate:
            arr = validate_lmedian_stack(arr)
        return _core.lmedian(arr).reshape(trailing)

    if dtype in (np.dtype(np.float32), np.dtype(np.float64)):
        if fun is rd.nanmedian:
            # Keep median/lmedian stack reductions on the Rust adapter path:
            # current reducers axis median can be slower than imc's former
            # image-stack loop for shallow stacks, so the fix belongs in
            # reducers::axis rather than in Python dispatch fan-out here.
            if validate:
                arr = validate_stack(arr)
            return _core.median(arr).reshape(trailing)
        if fun is rd.nanvar:
            if validate:
                arr = validate_stack(arr)
            return _core.variance(arr, ddof=int(kwargs.get("ddof", 0))).reshape(
                trailing
            )

    if validate:
        arr = validate_lmedian_stack(arr)
        validate = False
    return fun(
        arr,
        axis=0,
        ignore_inf=ignore_inf,
        validate=validate,
        **kwargs,
    ).reshape(trailing)


def _variance_stack(
    arr: np.ndarray,
    *,
    ddof: int = 0,
    return_mean: bool = False,
    validate: bool = True,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Return NaN-aware variance along the stack axis for imc internals."""
    if return_mean:
        trailing = arr.shape[1:]
        if np.asarray(arr).dtype in (np.dtype(np.float32), np.dtype(np.float64)):
            if validate:
                arr = validate_stack(arr)
            var, mean = _core.variance(arr, ddof=int(ddof), return_mean=True)
            return var.reshape(trailing), mean.reshape(trailing)
        if validate:
            arr = validate_lmedian_stack(arr)
            validate = False
        var, mean = rd.nanvar(
            arr,
            axis=0,
            ddof=int(ddof),
            return_mean=True,
            ignore_inf=True,
            validate=validate,
        )
        return var.reshape(trailing), mean.reshape(trailing)
    return _stack(arr, rd.nanvar, ddof=int(ddof), validate=validate)


def _prepare_percentile_q(q: object) -> tuple[np.ndarray, bool]:
    """Validate scalar or 1-D percentile positions."""
    q_arr = np.asarray(q, dtype=np.float64)
    scalar = q_arr.ndim == 0
    if scalar:
        q_arr = q_arr.reshape(1)
    elif q_arr.ndim != 1:
        raise ValueError(f"q must be a scalar or 1-D; got shape {q_arr.shape}")
    if np.any(~np.isfinite(q_arr)) or np.any((q_arr < 0.0) | (q_arr > 100.0)):
        raise ValueError("q must be in [0, 100]")
    return np.ascontiguousarray(q_arr), scalar


def _percentiles_stack(
    arr: np.ndarray,
    q: object,
    *,
    validate: bool = True,
) -> np.ndarray:
    """Return NaN-aware percentiles along the stack axis for imc internals."""
    trailing = arr.shape[1:]
    q_arr, scalar = _prepare_percentile_q(q)
    if np.asarray(arr).dtype in (np.dtype(np.float32), np.dtype(np.float64)):
        if validate:
            arr = validate_stack(arr)
        result = _core.percentiles(arr, q_arr)
        if scalar:
            return result[..., 0].reshape(trailing)
        return np.moveaxis(result, -1, 0).reshape((q_arr.size, *trailing))
    if validate:
        arr = validate_lmedian_stack(arr)
        validate = False
    result = rd.nanpercentile(
        arr,
        q_arr,
        axis=0,
        ignore_inf=True,
        validate=validate,
    )
    if scalar:
        return result[0].reshape(trailing)
    return result.reshape((q_arr.size, *trailing))


def nanaverage(
    arr: np.ndarray,
    weights: np.ndarray,
    *,
    validate: bool = True,
) -> np.ndarray:
    """Return the NaN-aware weighted average along the stack axis."""
    trailing = arr.shape[1:]
    if validate:
        arr = validate_lmedian_stack(arr)
        weights = validate_weights(weights, arr.shape[0])
        validate = False
    else:
        weights = np.ascontiguousarray(weights, dtype=np.float64).reshape(-1)
    return rd.nanaverage(
        arr,
        weights=weights,
        axis=0,
        ignore_inf=True,
        validate=validate,
    ).reshape(trailing)


def _prepare_1d_values(values: np.ndarray, *, validate: bool) -> np.ndarray:
    """Validate the public 1-D rejection contract and return Rust input."""
    if validate:
        return validate_values_1d(values)

    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError(f"values must be 1-D; got shape {values.shape}")
    return np.ascontiguousarray(values)


def _prepare_1d_rejection_inputs(
    values: np.ndarray, mask: np.ndarray | None, *, validate: bool
) -> tuple[np.ndarray, np.ndarray | None]:
    """Validate a public 1-D vector/mask pair for direct Rust slice kernels."""
    values = _prepare_1d_values(values, validate=validate)
    mask = validate_mask(mask, values.shape)
    return values, mask


def _scalar(value: object) -> object:
    """Return a Python/NumPy scalar from a scalar-like result."""
    if isinstance(value, np.ndarray):
        return value.reshape(-1)[0]
    return value


def _rejection_1d_result(
    result: RejectionResult,
) -> tuple[np.ndarray, object, object, object, object, object]:
    """Unwrap a stack rejection result to the 1-D public shape."""
    mask_rej, std, low, upp, nit, output_flags = result
    return (
        mask_rej.reshape(-1),
        _scalar(std),
        _scalar(low),
        _scalar(upp),
        _scalar(nit),
        _scalar(output_flags),
    )


# ---- reject -------------------------------------------------------------------------


def _sigma_pair(sigma: float | tuple[float, float]) -> tuple[float, float]:
    """Normalize scalar or asymmetric clipping thresholds."""
    if isinstance(sigma, (int, float)):
        return float(sigma), float(sigma)
    return float(sigma[0]), float(sigma[1])


def sigclip(
    arr: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    sigma: float | tuple[float, float] = (3.0, 3.0),
    maxiters: int = 5,
    ddof: int = 0,
    nkeep: int = 1,
    maxrej: int | None = None,
    cenfunc: str = "median",
    clip_cen: str | None = None,
    stdfunc: str = "std",
    revert_on_nkeep: bool = True,
    grow: float | None = None,
    validate: bool = True,
) -> RejectionResult:
    """Sigma-clipping rejection."""
    orig_shape = arr.shape
    trailing = arr.shape[1:]
    if validate:
        arr = validate_stack(arr)
        if mask is not None and mask.shape != arr.shape and mask.shape == orig_shape:
            mask = mask.reshape(arr.shape)
        mask = validate_mask(mask, arr.shape)
    sigma_lower, sigma_upper = _sigma_pair(sigma)
    _clip_cen = cenfunc if clip_cen is None else clip_cen
    mask_rej, std, low, upp, nit, output_flags = _core.sigclip(
        arr,
        mask=mask,
        sigma_lower=sigma_lower,
        sigma_upper=sigma_upper,
        maxiters=int(maxiters),
        ddof=int(ddof),
        nkeep=int(nkeep),
        maxrej=maxrej,
        cenfunc=str(cenfunc),
        clip_cen=str(_clip_cen),
        stdfunc=str(stdfunc),
        revert_on_nkeep=bool(revert_on_nkeep),
        validate=bool(validate),
    )
    mask_rej = mask_rej.reshape(orig_shape)
    output_flags = output_flags.reshape(trailing)
    mask_rej, output_flags = _grow_rejection_mask(
        mask_rej, output_flags, grow, validate=validate
    )
    return (
        mask_rej,
        std.reshape(trailing),
        low.reshape(trailing),
        upp.reshape(trailing),
        nit.reshape(trailing),
        output_flags,
    )


def sigclip_1d(
    values: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    sigma: float | tuple[float, float] = (3.0, 3.0),
    maxiters: int = 5,
    ddof: int = 0,
    nkeep: int = 1,
    maxrej: int | None = None,
    cenfunc: str = "median",
    clip_cen: str | None = None,
    stdfunc: str = "std",
    revert_on_nkeep: bool = True,
    validate: bool = True,
) -> tuple[np.ndarray, object, object, object, object, object]:
    """Sigma-clipping rejection for a 1-D value vector."""
    values, mask = _prepare_1d_rejection_inputs(values, mask, validate=validate)
    sigma_lower, sigma_upper = _sigma_pair(sigma)
    _clip_cen = cenfunc if clip_cen is None else clip_cen
    return _rejection_1d_result(
        _core.sigclip_1d(
            values,
            mask=mask,
            sigma_lower=sigma_lower,
            sigma_upper=sigma_upper,
            maxiters=int(maxiters),
            ddof=int(ddof),
            nkeep=int(nkeep),
            maxrej=maxrej,
            cenfunc=str(cenfunc),
            clip_cen=str(_clip_cen),
            stdfunc=str(stdfunc),
            revert_on_nkeep=bool(revert_on_nkeep),
            validate=False,
        )
    )


def sigclip_mask(
    arr: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    sigma: float | tuple[float, float] = (3.0, 3.0),
    maxiters: int = 5,
    ddof: int = 0,
    nkeep: int = 1,
    maxrej: int | None = None,
    cenfunc: str = "median",
    clip_cen: str | None = None,
    stdfunc: str = "std",
    revert_on_nkeep: bool = True,
    grow: float | None = None,
    validate: bool = True,
) -> np.ndarray:
    """Return only the sigma-clipping rejection mask."""
    orig_shape = arr.shape
    if validate:
        arr = validate_stack(arr)
        if mask is not None and mask.shape != arr.shape and mask.shape == orig_shape:
            mask = mask.reshape(arr.shape)
        mask = validate_mask(mask, arr.shape)
    sigma_lower, sigma_upper = _sigma_pair(sigma)
    _clip_cen = cenfunc if clip_cen is None else clip_cen
    mask_rej = _core.sigclip_mask(
        arr,
        mask=mask,
        sigma_lower=sigma_lower,
        sigma_upper=sigma_upper,
        maxiters=int(maxiters),
        ddof=int(ddof),
        nkeep=int(nkeep),
        maxrej=maxrej,
        cenfunc=str(cenfunc),
        clip_cen=str(_clip_cen),
        stdfunc=str(stdfunc),
        revert_on_nkeep=bool(revert_on_nkeep),
        validate=bool(validate),
    ).reshape(orig_shape)
    return _grow_rejection_mask_only(mask_rej, grow, validate=validate)


def sigclip_mask_1d(
    values: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    sigma: float | tuple[float, float] = (3.0, 3.0),
    maxiters: int = 5,
    ddof: int = 0,
    nkeep: int = 1,
    maxrej: int | None = None,
    cenfunc: str = "median",
    clip_cen: str | None = None,
    stdfunc: str = "std",
    revert_on_nkeep: bool = True,
    validate: bool = True,
) -> np.ndarray:
    """Return only the sigma-clipping rejection mask for a 1-D value vector."""
    values = _prepare_1d_values(values, validate=validate)
    mask = validate_mask(mask, values.shape)
    sigma_lower, sigma_upper = _sigma_pair(sigma)
    _clip_cen = cenfunc if clip_cen is None else clip_cen
    return _core.sigclip_mask_1d(
        values,
        mask=mask,
        sigma_lower=sigma_lower,
        sigma_upper=sigma_upper,
        maxiters=int(maxiters),
        ddof=int(ddof),
        nkeep=int(nkeep),
        maxrej=maxrej,
        cenfunc=str(cenfunc),
        clip_cen=str(_clip_cen),
        stdfunc=str(stdfunc),
        revert_on_nkeep=bool(revert_on_nkeep),
        validate=bool(validate),
    )


def _sigclip_restored_flags(
    arr: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    sigma: float | tuple[float, float] = (3.0, 3.0),
    maxiters: int = 5,
    ddof: int = 0,
    nkeep: int = 1,
    maxrej: int | None = None,
    cenfunc: str = "median",
    clip_cen: str | None = None,
    stdfunc: str = "std",
    revert_on_nkeep: bool = True,
    validate: bool = True,
) -> np.ndarray:
    """Return per-sample restored-candidate flags for sigma clipping."""
    orig_shape = arr.shape
    if validate:
        arr = validate_stack(arr)
        if mask is not None and mask.shape != arr.shape and mask.shape == orig_shape:
            mask = mask.reshape(arr.shape)
        mask = validate_mask(mask, arr.shape)
    sigma_lower, sigma_upper = _sigma_pair(sigma)
    _clip_cen = cenfunc if clip_cen is None else clip_cen
    return _core.sigclip_restored_flags(
        arr,
        mask=mask,
        sigma_lower=sigma_lower,
        sigma_upper=sigma_upper,
        maxiters=int(maxiters),
        ddof=int(ddof),
        nkeep=int(nkeep),
        maxrej=maxrej,
        cenfunc=str(cenfunc),
        clip_cen=str(_clip_cen),
        stdfunc=str(stdfunc),
        revert_on_nkeep=bool(revert_on_nkeep),
        validate=bool(validate),
    ).reshape(orig_shape)


def sigclip_combine(
    arr: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    combine: str,
    sigma: float | tuple[float, float] = (3.0, 3.0),
    maxiters: int = 5,
    ddof: int = 0,
    nkeep: int = 1,
    maxrej: int | None = None,
    cenfunc: str = "median",
    clip_cen: str | None = None,
    stdfunc: str = "std",
    revert_on_nkeep: bool = True,
    validate: bool = True,
) -> np.ndarray:
    """Return an output-only sigma-clipped mean or median."""
    cb = combine.lower()
    if cb not in ("mean", "average", "avg", "median", "med"):
        raise NotImplementedError("fused sigclip currently supports mean and median")
    orig_shape = arr.shape
    trailing = orig_shape[1:]
    if validate:
        arr = validate_stack(arr)
        if mask is not None and mask.shape != arr.shape and mask.shape == orig_shape:
            mask = mask.reshape(arr.shape)
        mask = validate_mask(mask, arr.shape)
    sigma_lower, sigma_upper = _sigma_pair(sigma)
    _clip_cen = cenfunc if clip_cen is None else clip_cen
    kernel = _core.sigclip_median if cb in ("median", "med") else _core.sigclip_mean
    out = kernel(
        arr,
        mask=mask,
        sigma_lower=sigma_lower,
        sigma_upper=sigma_upper,
        maxiters=int(maxiters),
        ddof=int(ddof),
        nkeep=int(nkeep),
        maxrej=maxrej,
        cenfunc=str(cenfunc),
        clip_cen=str(_clip_cen),
        stdfunc=str(stdfunc),
        revert_on_nkeep=bool(revert_on_nkeep),
        validate=bool(validate),
    )
    return out.reshape(trailing)


def sigclip_combine_1d(
    values: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    combine: str,
    sigma: float | tuple[float, float] = (3.0, 3.0),
    maxiters: int = 5,
    ddof: int = 0,
    nkeep: int = 1,
    maxrej: int | None = None,
    cenfunc: str = "median",
    clip_cen: str | None = None,
    stdfunc: str = "std",
    revert_on_nkeep: bool = True,
    validate: bool = True,
) -> object:
    """Return a 1-D sigma-clipped mean or median."""
    cb = combine.lower()
    if cb not in ("mean", "average", "avg", "median", "med"):
        raise NotImplementedError("fused sigclip currently supports mean and median")
    values, mask = _prepare_1d_rejection_inputs(values, mask, validate=validate)
    sigma_lower, sigma_upper = _sigma_pair(sigma)
    _clip_cen = cenfunc if clip_cen is None else clip_cen
    kernel = (
        _core.sigclip_median_1d if cb in ("median", "med") else _core.sigclip_mean_1d
    )
    return kernel(
        values,
        mask=mask,
        sigma_lower=sigma_lower,
        sigma_upper=sigma_upper,
        maxiters=int(maxiters),
        ddof=int(ddof),
        nkeep=int(nkeep),
        maxrej=maxrej,
        cenfunc=str(cenfunc),
        clip_cen=str(_clip_cen),
        stdfunc=str(stdfunc),
        revert_on_nkeep=bool(revert_on_nkeep),
        validate=False,
    )


def _ccd_vectors(
    scales: npt.ArrayLike | None, zeros: npt.ArrayLike | None
) -> tuple[list[float] | None, list[float] | None]:
    """Convert optional CCDClip plane vectors for the compiled boundary."""
    scale_values = (
        None
        if scales is None
        else np.asarray(scales, dtype=np.float64).ravel().tolist()
    )
    zero_values = (
        None if zeros is None else np.asarray(zeros, dtype=np.float64).ravel().tolist()
    )
    return scale_values, zero_values


def ccdclip(
    arr: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    sigma: float | tuple[float, float] = (3.0, 3.0),
    maxiters: int = 5,
    ddof: int = 0,
    nkeep: int = 1,
    maxrej: int | None = None,
    cenfunc: str = "median",
    clip_cen: str | None = None,
    revert_on_nkeep: bool = True,
    rdnoise: float = 0.0,
    gain: float = 1.0,
    snoise: float = 0.0,
    scales: npt.ArrayLike | None = None,
    zeros: npt.ArrayLike | None = None,
    grow: float | None = None,
    validate: bool = True,
) -> RejectionResult:
    """CCD noise-model clipping."""
    orig_shape = arr.shape
    trailing = arr.shape[1:]
    if validate:
        arr = validate_stack(arr)
        if mask is not None and mask.shape != arr.shape and mask.shape == orig_shape:
            mask = mask.reshape(arr.shape)
        mask = validate_mask(mask, arr.shape)
    scales, zeros = _ccd_vectors(scales, zeros)
    sigma_lower, sigma_upper = _sigma_pair(sigma)
    _clip_cen = cenfunc if clip_cen is None else clip_cen
    mask_rej, std, low, upp, nit, output_flags = _core.ccdclip(
        arr,
        mask=mask,
        sigma_lower=sigma_lower,
        sigma_upper=sigma_upper,
        maxiters=int(maxiters),
        ddof=int(ddof),
        nkeep=int(nkeep),
        maxrej=maxrej,
        cenfunc=str(cenfunc),
        clip_cen=str(_clip_cen),
        revert_on_nkeep=bool(revert_on_nkeep),
        rdnoise_ref=float(rdnoise),
        snoise_ref=float(snoise),
        scales=scales,
        zeros=zeros,
        gain=float(gain),
        validate=bool(validate),
    )
    mask_rej = mask_rej.reshape(orig_shape)
    output_flags = output_flags.reshape(trailing)
    mask_rej, output_flags = _grow_rejection_mask(
        mask_rej, output_flags, grow, validate=validate
    )
    return (
        mask_rej,
        std.reshape(trailing),
        low.reshape(trailing),
        upp.reshape(trailing),
        nit.reshape(trailing),
        output_flags,
    )


def ccdclip_1d(
    values: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    sigma: float | tuple[float, float] = (3.0, 3.0),
    maxiters: int = 5,
    ddof: int = 0,
    nkeep: int = 1,
    maxrej: int | None = None,
    cenfunc: str = "median",
    clip_cen: str | None = None,
    revert_on_nkeep: bool = True,
    rdnoise: float = 0.0,
    gain: float = 1.0,
    snoise: float = 0.0,
    scales: npt.ArrayLike | None = None,
    zeros: npt.ArrayLike | None = None,
    validate: bool = True,
) -> tuple[np.ndarray, object, object, object, object, object]:
    """CCD noise-model clipping for a 1-D value vector."""
    values, mask = _prepare_1d_rejection_inputs(values, mask, validate=validate)
    scales, zeros = _ccd_vectors(scales, zeros)
    sigma_lower, sigma_upper = _sigma_pair(sigma)
    _clip_cen = cenfunc if clip_cen is None else clip_cen
    return _rejection_1d_result(
        _core.ccdclip_1d(
            values,
            mask=mask,
            sigma_lower=sigma_lower,
            sigma_upper=sigma_upper,
            maxiters=int(maxiters),
            ddof=int(ddof),
            nkeep=int(nkeep),
            maxrej=maxrej,
            cenfunc=str(cenfunc),
            clip_cen=str(_clip_cen),
            revert_on_nkeep=bool(revert_on_nkeep),
            rdnoise_ref=float(rdnoise),
            snoise_ref=float(snoise),
            scales=scales,
            zeros=zeros,
            gain=float(gain),
            validate=False,
        )
    )


def ccdclip_mask(
    arr: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    sigma: float | tuple[float, float] = (3.0, 3.0),
    maxiters: int = 5,
    ddof: int = 0,
    nkeep: int = 1,
    maxrej: int | None = None,
    cenfunc: str = "median",
    clip_cen: str | None = None,
    revert_on_nkeep: bool = True,
    rdnoise: float = 0.0,
    gain: float = 1.0,
    snoise: float = 0.0,
    scales: npt.ArrayLike | None = None,
    zeros: npt.ArrayLike | None = None,
    grow: float | None = None,
    validate: bool = True,
) -> np.ndarray:
    """Return only the CCD-clipping rejection mask."""
    orig_shape = arr.shape
    if validate:
        arr = validate_stack(arr)
        if mask is not None and mask.shape != arr.shape and mask.shape == orig_shape:
            mask = mask.reshape(arr.shape)
        mask = validate_mask(mask, arr.shape)
    scales, zeros = _ccd_vectors(scales, zeros)
    sigma_lower, sigma_upper = _sigma_pair(sigma)
    _clip_cen = cenfunc if clip_cen is None else clip_cen
    mask_rej = _core.ccdclip_mask(
        arr,
        mask=mask,
        sigma_lower=sigma_lower,
        sigma_upper=sigma_upper,
        maxiters=int(maxiters),
        ddof=int(ddof),
        nkeep=int(nkeep),
        maxrej=maxrej,
        cenfunc=str(cenfunc),
        clip_cen=str(_clip_cen),
        revert_on_nkeep=bool(revert_on_nkeep),
        rdnoise_ref=float(rdnoise),
        snoise_ref=float(snoise),
        scales=scales,
        zeros=zeros,
        gain=float(gain),
        validate=bool(validate),
    ).reshape(orig_shape)
    return _grow_rejection_mask_only(mask_rej, grow, validate=validate)


def ccdclip_mask_1d(
    values: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    sigma: float | tuple[float, float] = (3.0, 3.0),
    maxiters: int = 5,
    ddof: int = 0,
    nkeep: int = 1,
    maxrej: int | None = None,
    cenfunc: str = "median",
    clip_cen: str | None = None,
    revert_on_nkeep: bool = True,
    rdnoise: float = 0.0,
    gain: float = 1.0,
    snoise: float = 0.0,
    scales: npt.ArrayLike | None = None,
    zeros: npt.ArrayLike | None = None,
    validate: bool = True,
) -> np.ndarray:
    """Return only the CCD-clipping rejection mask for a 1-D value vector."""
    values, mask = _prepare_1d_rejection_inputs(values, mask, validate=validate)
    scales, zeros = _ccd_vectors(scales, zeros)
    sigma_lower, sigma_upper = _sigma_pair(sigma)
    _clip_cen = cenfunc if clip_cen is None else clip_cen
    return _core.ccdclip_mask_1d(
        values,
        mask=mask,
        sigma_lower=sigma_lower,
        sigma_upper=sigma_upper,
        maxiters=int(maxiters),
        ddof=int(ddof),
        nkeep=int(nkeep),
        maxrej=maxrej,
        cenfunc=str(cenfunc),
        clip_cen=str(_clip_cen),
        revert_on_nkeep=bool(revert_on_nkeep),
        rdnoise_ref=float(rdnoise),
        snoise_ref=float(snoise),
        scales=scales,
        zeros=zeros,
        gain=float(gain),
        validate=False,
    )


def _ccdclip_restored_flags(
    arr: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    sigma: float | tuple[float, float] = (3.0, 3.0),
    maxiters: int = 5,
    ddof: int = 0,
    nkeep: int = 1,
    maxrej: int | None = None,
    cenfunc: str = "median",
    clip_cen: str | None = None,
    revert_on_nkeep: bool = True,
    rdnoise: float = 0.0,
    gain: float = 1.0,
    snoise: float = 0.0,
    scales: npt.ArrayLike | None = None,
    zeros: npt.ArrayLike | None = None,
    validate: bool = True,
) -> np.ndarray:
    """Return per-sample restored-candidate flags for CCD clipping."""
    orig_shape = arr.shape
    if validate:
        arr = validate_stack(arr)
        if mask is not None and mask.shape != arr.shape and mask.shape == orig_shape:
            mask = mask.reshape(arr.shape)
        mask = validate_mask(mask, arr.shape)
    scales, zeros = _ccd_vectors(scales, zeros)
    sigma_lower, sigma_upper = _sigma_pair(sigma)
    _clip_cen = cenfunc if clip_cen is None else clip_cen
    return _core.ccdclip_restored_flags(
        arr,
        mask=mask,
        sigma_lower=sigma_lower,
        sigma_upper=sigma_upper,
        maxiters=int(maxiters),
        ddof=int(ddof),
        nkeep=int(nkeep),
        maxrej=maxrej,
        cenfunc=str(cenfunc),
        clip_cen=str(_clip_cen),
        revert_on_nkeep=bool(revert_on_nkeep),
        rdnoise_ref=float(rdnoise),
        snoise_ref=float(snoise),
        scales=scales,
        zeros=zeros,
        gain=float(gain),
        validate=bool(validate),
    ).reshape(orig_shape)


def ccdclip_combine(
    arr: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    combine: str,
    sigma: float | tuple[float, float] = (3.0, 3.0),
    maxiters: int = 5,
    ddof: int = 0,
    nkeep: int = 1,
    maxrej: int | None = None,
    cenfunc: str = "median",
    clip_cen: str | None = None,
    revert_on_nkeep: bool = True,
    rdnoise: float = 0.0,
    gain: float = 1.0,
    snoise: float = 0.0,
    scales: npt.ArrayLike | None = None,
    zeros: npt.ArrayLike | None = None,
    validate: bool = True,
) -> np.ndarray:
    """Return an output-only CCD-clipped mean or median."""
    cb = combine.lower()
    if cb not in ("mean", "average", "avg", "median", "med"):
        raise NotImplementedError("fused ccdclip currently supports mean and median")
    orig_shape = arr.shape
    trailing = orig_shape[1:]
    if validate:
        arr = validate_stack(arr)
        if mask is not None and mask.shape != arr.shape and mask.shape == orig_shape:
            mask = mask.reshape(arr.shape)
        mask = validate_mask(mask, arr.shape)
    scales, zeros = _ccd_vectors(scales, zeros)
    sigma_lower, sigma_upper = _sigma_pair(sigma)
    _clip_cen = cenfunc if clip_cen is None else clip_cen
    kernel = _core.ccdclip_median if cb in ("median", "med") else _core.ccdclip_mean
    out = kernel(
        arr,
        mask=mask,
        sigma_lower=sigma_lower,
        sigma_upper=sigma_upper,
        maxiters=int(maxiters),
        ddof=int(ddof),
        nkeep=int(nkeep),
        maxrej=maxrej,
        cenfunc=str(cenfunc),
        clip_cen=str(_clip_cen),
        revert_on_nkeep=bool(revert_on_nkeep),
        rdnoise_ref=float(rdnoise),
        snoise_ref=float(snoise),
        scales=scales,
        zeros=zeros,
        gain=float(gain),
        validate=bool(validate),
    )
    return out.reshape(trailing)


def ccdclip_combine_1d(
    values: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    combine: str,
    sigma: float | tuple[float, float] = (3.0, 3.0),
    maxiters: int = 5,
    ddof: int = 0,
    nkeep: int = 1,
    maxrej: int | None = None,
    cenfunc: str = "median",
    clip_cen: str | None = None,
    revert_on_nkeep: bool = True,
    rdnoise: float = 0.0,
    gain: float = 1.0,
    snoise: float = 0.0,
    scales: npt.ArrayLike | None = None,
    zeros: npt.ArrayLike | None = None,
    validate: bool = True,
) -> object:
    """Return a 1-D CCD-clipped mean or median."""
    cb = combine.lower()
    if cb not in ("mean", "average", "avg", "median", "med"):
        raise NotImplementedError("fused ccdclip currently supports mean and median")
    values, mask = _prepare_1d_rejection_inputs(values, mask, validate=validate)
    scales, zeros = _ccd_vectors(scales, zeros)
    sigma_lower, sigma_upper = _sigma_pair(sigma)
    _clip_cen = cenfunc if clip_cen is None else clip_cen
    kernel = (
        _core.ccdclip_median_1d if cb in ("median", "med") else _core.ccdclip_mean_1d
    )
    return kernel(
        values,
        mask=mask,
        sigma_lower=sigma_lower,
        sigma_upper=sigma_upper,
        maxiters=int(maxiters),
        ddof=int(ddof),
        nkeep=int(nkeep),
        maxrej=maxrej,
        cenfunc=str(cenfunc),
        clip_cen=str(_clip_cen),
        revert_on_nkeep=bool(revert_on_nkeep),
        rdnoise_ref=float(rdnoise),
        snoise_ref=float(snoise),
        scales=scales,
        zeros=zeros,
        gain=float(gain),
        validate=False,
    )


def _resolve_minmax_count(n: int | float, name: str, N: int) -> int:
    """Convert an n_min / n_max value to a frame count for the Rust core.

    Values >= 1 are used directly (truncated to int).
    Values in [0, 1) are treated as a fraction of N and converted via
    ``int(N * n + 0.001)``, matching IRAF's internal fraction arithmetic.
    Negative values raise ``ValueError``.
    """
    if n < 1:
        if n < 0:
            raise ValueError(f"{name} must be >= 0; got {n!r}")
        return int(N * n + 0.001)
    return int(n)


def _pclip_value(frac: float) -> float:
    """Validate IRAF-style scalar pclip."""
    if not isinstance(frac, (int, float)):
        raise ValueError("frac must be a scalar for IRAF-style pclip")
    pclip = float(frac)
    if not np.isfinite(pclip) or pclip == 0.0:
        raise ValueError("frac must be finite and non-zero")
    return pclip


def linearclip(
    arr: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    low_scale: float = 1.0,
    low: float = 0.0,
    upp_scale: float = 1.0,
    upp: float = 0.0,
    maxiters: int = 1,
    nkeep: int = 1,
    maxrej: int | None = None,
    cenfunc: str = "median",
    revert_on_nkeep: bool = True,
    grow: float | None = None,
    validate: bool = True,
) -> RejectionResult:
    """Reject values outside center-relative linear bounds."""
    orig_shape = arr.shape
    trailing = arr.shape[1:]
    if validate:
        arr = validate_stack(arr)
        if mask is not None and mask.shape != arr.shape and mask.shape == orig_shape:
            mask = mask.reshape(arr.shape)
        mask = validate_mask(mask, arr.shape)
    low_scale = float(low_scale)
    low = float(low)
    upp_scale = float(upp_scale)
    upp = float(upp)
    if (low_scale, low, upp_scale, upp) == (1.0, 0.0, 1.0, 0.0):
        mask_rej = np.zeros(arr.shape, dtype=np.bool_).reshape(orig_shape)
        low_arr = np.full(arr.shape[1:], np.nan, dtype=arr.dtype).reshape(trailing)
        upp_arr = np.full(arr.shape[1:], np.nan, dtype=arr.dtype).reshape(trailing)
        nit = np.zeros(arr.shape[1:], dtype=np.uint8).reshape(trailing)
        output_flags = np.zeros(arr.shape[1:], dtype=np.uint8).reshape(trailing)
        return mask_rej, None, low_arr, upp_arr, nit, output_flags
    mask_rej, _std, low_arr, upp_arr, nit, output_flags = _core.linearclip(
        arr,
        mask=mask,
        low_scale=low_scale,
        low=low,
        upp_scale=upp_scale,
        upp=upp,
        maxiters=int(maxiters),
        nkeep=int(nkeep),
        maxrej=maxrej,
        cenfunc=str(cenfunc),
        revert_on_nkeep=bool(revert_on_nkeep),
        validate=bool(validate),
    )
    mask_rej = mask_rej.reshape(orig_shape)
    output_flags = output_flags.reshape(trailing)
    mask_rej, output_flags = _grow_rejection_mask(
        mask_rej, output_flags, grow, validate=validate
    )
    return (
        mask_rej,
        None,
        low_arr.reshape(trailing),
        upp_arr.reshape(trailing),
        nit.reshape(trailing),
        output_flags,
    )


def linearclip_1d(
    values: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    low_scale: float = 1.0,
    low: float = 0.0,
    upp_scale: float = 1.0,
    upp: float = 0.0,
    maxiters: int = 1,
    nkeep: int = 1,
    maxrej: int | None = None,
    cenfunc: str = "median",
    revert_on_nkeep: bool = True,
    validate: bool = True,
) -> tuple[np.ndarray, object, object, object, object, object]:
    """Reject values outside center-relative linear bounds in a 1-D vector."""
    values, mask = _prepare_1d_rejection_inputs(values, mask, validate=validate)
    mask_rej, _std, low_arr, upp_arr, nit, output_flags = _rejection_1d_result(
        _core.linearclip_1d(
            values,
            mask=mask,
            low_scale=float(low_scale),
            low=float(low),
            upp_scale=float(upp_scale),
            upp=float(upp),
            maxiters=int(maxiters),
            nkeep=int(nkeep),
            maxrej=maxrej,
            cenfunc=str(cenfunc),
            revert_on_nkeep=bool(revert_on_nkeep),
            validate=False,
        )
    )
    return mask_rej, None, low_arr, upp_arr, nit, output_flags


def _linearclip_restored_flags(
    arr: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    low_scale: float = 1.0,
    low: float = 0.0,
    upp_scale: float = 1.0,
    upp: float = 0.0,
    maxiters: int = 1,
    nkeep: int = 1,
    maxrej: int | None = None,
    cenfunc: str = "median",
    revert_on_nkeep: bool = True,
    validate: bool = True,
) -> np.ndarray:
    """Return per-sample restored-candidate flags for linear clipping."""
    orig_shape = arr.shape
    if validate:
        arr = validate_stack(arr)
        if mask is not None and mask.shape != arr.shape and mask.shape == orig_shape:
            mask = mask.reshape(arr.shape)
        mask = validate_mask(mask, arr.shape)
    if (
        float(low_scale),
        float(low),
        float(upp_scale),
        float(upp),
    ) == (1.0, 0.0, 1.0, 0.0):
        return np.zeros(arr.shape, dtype=np.uint8).reshape(orig_shape)
    return _core.linearclip_restored_flags(
        arr,
        mask=mask,
        low_scale=float(low_scale),
        low=float(low),
        upp_scale=float(upp_scale),
        upp=float(upp),
        maxiters=int(maxiters),
        nkeep=int(nkeep),
        maxrej=maxrej,
        cenfunc=str(cenfunc),
        revert_on_nkeep=bool(revert_on_nkeep),
        validate=bool(validate),
    ).reshape(orig_shape)


def minmax(
    arr: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    n_min: int | float = 1,
    n_max: int | float = 1,
    grow: float | None = None,
    validate: bool = True,
) -> RejectionResult:
    """Reject tail-ranked unmasked values at each pixel."""
    orig_shape = arr.shape
    trailing = arr.shape[1:]
    if validate:
        arr = validate_stack(arr)
        if mask is not None and mask.shape != arr.shape and mask.shape == orig_shape:
            mask = mask.reshape(arr.shape)
        mask = validate_mask(mask, arr.shape)
    N = arr.shape[0]
    mask_rej, _std, low, upp, nit, output_flags = _core.minmax(
        arr,
        mask=mask,
        n_min=_resolve_minmax_count(n_min, "n_min", N),
        n_max=_resolve_minmax_count(n_max, "n_max", N),
        validate=bool(validate),
    )
    mask_rej = mask_rej.reshape(orig_shape)
    output_flags = output_flags.reshape(trailing)
    mask_rej, output_flags = _grow_rejection_mask(
        mask_rej, output_flags, grow, validate=validate
    )
    return (
        mask_rej,
        None,
        low.reshape(trailing),
        upp.reshape(trailing),
        nit.reshape(trailing),
        output_flags,
    )


def minmax_1d(
    values: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    n_min: int | float = 1,
    n_max: int | float = 1,
    validate: bool = True,
) -> tuple[np.ndarray, object, object, object, object, object]:
    """Reject tail-ranked unmasked values in a 1-D value vector."""
    values, mask = _prepare_1d_rejection_inputs(values, mask, validate=validate)
    N = values.shape[0]
    mask_rej, _std, low, upp, nit, output_flags = _rejection_1d_result(
        _core.minmax_1d(
            values,
            mask=mask,
            n_min=_resolve_minmax_count(n_min, "n_min", N),
            n_max=_resolve_minmax_count(n_max, "n_max", N),
            validate=False,
        )
    )
    return mask_rej, None, low, upp, nit, output_flags


def minmax_mask(
    arr: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    n_min: int | float = 1,
    n_max: int | float = 1,
    grow: float | None = None,
    validate: bool = True,
) -> np.ndarray:
    """Return only the minmax rejection mask."""
    orig_shape = arr.shape
    if validate:
        arr = validate_stack(arr)
        if mask is not None and mask.shape != arr.shape and mask.shape == orig_shape:
            mask = mask.reshape(arr.shape)
        mask = validate_mask(mask, arr.shape)
    N = arr.shape[0]
    mask_rej = _core.minmax_mask(
        arr,
        mask=mask,
        n_min=_resolve_minmax_count(n_min, "n_min", N),
        n_max=_resolve_minmax_count(n_max, "n_max", N),
        validate=bool(validate),
    ).reshape(orig_shape)
    return _grow_rejection_mask_only(mask_rej, grow, validate=validate)


def minmax_mask_1d(
    values: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    n_min: int | float = 1,
    n_max: int | float = 1,
    validate: bool = True,
) -> np.ndarray:
    """Return only the minmax rejection mask for a 1-D value vector."""
    values, mask = _prepare_1d_rejection_inputs(values, mask, validate=validate)
    N = values.shape[0]
    return _core.minmax_mask_1d(
        values,
        mask=mask,
        n_min=_resolve_minmax_count(n_min, "n_min", N),
        n_max=_resolve_minmax_count(n_max, "n_max", N),
        validate=False,
    )


def minmax_combine(
    arr: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    combine: str,
    n_min: int | float = 1,
    n_max: int | float = 1,
    validate: bool = True,
) -> np.ndarray:
    """Return an output-only minmax-rejected mean or median."""
    cb = combine.lower()
    if cb not in ("mean", "average", "avg", "median", "med"):
        raise NotImplementedError("fused minmax currently supports mean and median")
    orig_shape = arr.shape
    trailing = orig_shape[1:]
    if validate:
        arr = validate_stack(arr)
        if mask is not None and mask.shape != arr.shape and mask.shape == orig_shape:
            mask = mask.reshape(arr.shape)
        mask = validate_mask(mask, arr.shape)
    N = arr.shape[0]
    kernel = _core.minmax_median if cb in ("median", "med") else _core.minmax_mean
    out = kernel(
        arr,
        mask=mask,
        n_min=_resolve_minmax_count(n_min, "n_min", N),
        n_max=_resolve_minmax_count(n_max, "n_max", N),
        validate=bool(validate),
    )
    return out.reshape(trailing)


def minmax_combine_1d(
    values: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    combine: str,
    n_min: int | float = 1,
    n_max: int | float = 1,
    validate: bool = True,
) -> object:
    """Return a 1-D minmax-rejected mean or median."""
    cb = combine.lower()
    if cb not in ("mean", "average", "avg", "median", "med"):
        raise NotImplementedError("fused minmax currently supports mean and median")
    values, mask = _prepare_1d_rejection_inputs(values, mask, validate=validate)
    N = values.shape[0]
    kernel = _core.minmax_median_1d if cb in ("median", "med") else _core.minmax_mean_1d
    return kernel(
        values,
        mask=mask,
        n_min=_resolve_minmax_count(n_min, "n_min", N),
        n_max=_resolve_minmax_count(n_max, "n_max", N),
        validate=False,
    )


def pclip(
    arr: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    frac: float = -0.5,
    sigma: float | tuple[float, float] = 3.0,
    nkeep: int = 1,
    grow: float | None = None,
    validate: bool = True,
) -> RejectionResult:
    """IRAF-style percentile clipping at each pixel."""
    orig_shape = arr.shape
    trailing = arr.shape[1:]
    if validate:
        arr = validate_stack(arr)
        if mask is not None and mask.shape != arr.shape and mask.shape == orig_shape:
            mask = mask.reshape(arr.shape)
        mask = validate_mask(mask, arr.shape)
    pclip_value = _pclip_value(frac)
    sigma_lower, sigma_upper = _sigma_pair(sigma)
    mask_rej, _std, low, upp, nit, output_flags = _core.pclip(
        arr,
        mask=mask,
        pclip=pclip_value,
        sigma_lower=sigma_lower,
        sigma_upper=sigma_upper,
        nkeep=int(nkeep),
        validate=bool(validate),
    )
    mask_rej = mask_rej.reshape(orig_shape)
    output_flags = output_flags.reshape(trailing)
    mask_rej, output_flags = _grow_rejection_mask(
        mask_rej, output_flags, grow, validate=validate
    )
    return (
        mask_rej,
        None,
        low.reshape(trailing),
        upp.reshape(trailing),
        nit.reshape(trailing),
        output_flags,
    )


def pclip_1d(
    values: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    frac: float = -0.5,
    sigma: float | tuple[float, float] = 3.0,
    nkeep: int = 1,
    validate: bool = True,
) -> tuple[np.ndarray, object, object, object, object, object]:
    """IRAF-style percentile clipping for a 1-D value vector."""
    values, mask = _prepare_1d_rejection_inputs(values, mask, validate=validate)
    pclip_value = _pclip_value(frac)
    sigma_lower, sigma_upper = _sigma_pair(sigma)
    mask_rej, _std, low, upp, nit, output_flags = _rejection_1d_result(
        _core.pclip_1d(
            values,
            mask=mask,
            pclip=pclip_value,
            sigma_lower=sigma_lower,
            sigma_upper=sigma_upper,
            nkeep=int(nkeep),
            validate=False,
        )
    )
    return mask_rej, None, low, upp, nit, output_flags


def pclip_mask(
    arr: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    frac: float = -0.5,
    sigma: float | tuple[float, float] = 3.0,
    nkeep: int = 1,
    grow: float | None = None,
    validate: bool = True,
) -> np.ndarray:
    """Return only the IRAF-style pclip rejection mask."""
    orig_shape = arr.shape
    if validate:
        arr = validate_stack(arr)
        if mask is not None and mask.shape != arr.shape and mask.shape == orig_shape:
            mask = mask.reshape(arr.shape)
        mask = validate_mask(mask, arr.shape)
    pclip_value = _pclip_value(frac)
    sigma_lower, sigma_upper = _sigma_pair(sigma)
    mask_rej = _core.pclip_mask(
        arr,
        mask=mask,
        pclip=pclip_value,
        sigma_lower=sigma_lower,
        sigma_upper=sigma_upper,
        nkeep=int(nkeep),
        validate=bool(validate),
    ).reshape(orig_shape)
    return _grow_rejection_mask_only(mask_rej, grow, validate=validate)


def pclip_mask_1d(
    values: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    frac: float = -0.5,
    sigma: float | tuple[float, float] = 3.0,
    nkeep: int = 1,
    validate: bool = True,
) -> np.ndarray:
    """Return only the IRAF-style pclip rejection mask for a 1-D value vector."""
    values, mask = _prepare_1d_rejection_inputs(values, mask, validate=validate)
    pclip_value = _pclip_value(frac)
    sigma_lower, sigma_upper = _sigma_pair(sigma)
    return _core.pclip_mask_1d(
        values,
        mask=mask,
        pclip=pclip_value,
        sigma_lower=sigma_lower,
        sigma_upper=sigma_upper,
        nkeep=int(nkeep),
        validate=False,
    )


def pclip_combine(
    arr: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    combine: str,
    frac: float = -0.5,
    sigma: float | tuple[float, float] = 3.0,
    nkeep: int = 1,
    validate: bool = True,
) -> np.ndarray:
    """Return an output-only pclip-rejected mean or median."""
    cb = combine.lower()
    if cb not in ("mean", "average", "avg", "median", "med"):
        raise NotImplementedError("fused pclip currently supports mean and median")
    orig_shape = arr.shape
    trailing = orig_shape[1:]
    if validate:
        arr = validate_stack(arr)
        if mask is not None and mask.shape != arr.shape and mask.shape == orig_shape:
            mask = mask.reshape(arr.shape)
        mask = validate_mask(mask, arr.shape)
    pclip_value = _pclip_value(frac)
    sigma_lower, sigma_upper = _sigma_pair(sigma)
    kernel = _core.pclip_median if cb in ("median", "med") else _core.pclip_mean
    out = kernel(
        arr,
        mask=mask,
        pclip=pclip_value,
        sigma_lower=sigma_lower,
        sigma_upper=sigma_upper,
        nkeep=int(nkeep),
        validate=bool(validate),
    )
    return out.reshape(trailing)


def pclip_combine_1d(
    values: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    combine: str,
    frac: float = -0.5,
    sigma: float | tuple[float, float] = 3.0,
    nkeep: int = 1,
    validate: bool = True,
) -> object:
    """Return a 1-D pclip-rejected mean or median."""
    cb = combine.lower()
    if cb not in ("mean", "average", "avg", "median", "med"):
        raise NotImplementedError("fused pclip currently supports mean and median")
    values, mask = _prepare_1d_rejection_inputs(values, mask, validate=validate)
    pclip_value = _pclip_value(frac)
    sigma_lower, sigma_upper = _sigma_pair(sigma)
    kernel = _core.pclip_median_1d if cb in ("median", "med") else _core.pclip_mean_1d
    return kernel(
        values,
        mask=mask,
        pclip=pclip_value,
        sigma_lower=sigma_lower,
        sigma_upper=sigma_upper,
        nkeep=int(nkeep),
        validate=False,
    )


_sigclip_mask = sigclip_mask
_sigclip_combine = sigclip_combine
_ccdclip_mask = ccdclip_mask
_ccdclip_combine = ccdclip_combine
_minmax_mask = minmax_mask
_minmax_combine = minmax_combine
_pclip_mask = pclip_mask
_pclip_combine = pclip_combine


_doc.install_kernel_docstrings(globals())
