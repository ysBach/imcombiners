"""Compact ``ndcombine`` compatibility wrapper."""

from __future__ import annotations

from typing import Literal, overload

import numpy as np

from . import kernels
from ._diagnostics import Diagnostics, normalize_diagnostics, sample_flags_array
from ._typing import (
    FullCombineResult,
    FullDiagnosticCombineResult,
    PlaneVectorLike,
    SigclipStatKwargs,
)
from ._validation import (
    mask_as_nan,
    resolve_zero_scale,
    threshold_mask,
    validate_mask,
    validate_positive_scalar,
    validate_stack,
    validate_thresholds,
    validate_weights,
)

__all__ = ["ndcombine"]


def _apply_zero_scale(
    arr: np.ndarray,
    zero: PlaneVectorLike,
    scale: PlaneVectorLike,
    *,
    zero_to_0th: bool = True,
    scale_to_0th: bool = True,
    zero_sigclip_kwargs: SigclipStatKwargs = None,
    scale_sigclip_kwargs: SigclipStatKwargs = None,
    mask: np.ndarray | None = None,
    validate: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Apply normalization and return its resolved zero/scale vectors."""
    out = arr
    copied = False
    arr_stat = mask_as_nan(arr, mask) if mask is not None else arr
    if zero is not None:
        z = resolve_zero_scale(
            "zero",
            zero,
            arr_stat,
            to_0th=zero_to_0th,
            sigclip_kwargs=dict(zero_sigclip_kwargs)
            if zero_sigclip_kwargs is not None
            else None,
            validate=validate,
        )
        out = arr - z
        copied = True
    if scale is not None:
        s = resolve_zero_scale(
            "scale",
            scale,
            arr_stat,
            nonzero=True,
            to_0th=scale_to_0th,
            sigclip_kwargs=dict(scale_sigclip_kwargs)
            if scale_sigclip_kwargs is not None
            else None,
            validate=validate,
        )
        out = (out if copied else arr.copy()) / s
        copied = True
    return (
        out if copied else arr.copy(),
        z if zero is not None else None,
        s if scale is not None else None,
    )


def _validate_grow_value(grow: float | None, validate: bool) -> float | None:
    """Validate an optional grow radius at the public wrapper boundary."""
    if grow is None or not validate:
        return grow
    grow = float(grow)
    if not np.isfinite(grow) or grow < 0.0:
        raise ValueError("grow must be a finite non-negative radius")
    return grow


@overload
def ndcombine(
    arr: np.ndarray,
    mask: np.ndarray | None = ...,
    thresholds: tuple[float, float] | list[float] | None = ...,
    combine: str = ...,
    reject: str | None = ...,
    sigma: float | tuple[float, float] = ...,
    cenfunc: str = ...,
    clip_cen: str | None = ...,
    maxiters: int = ...,
    ddof: int = ...,
    nkeep: int = ...,
    maxrej: int | None = ...,
    n_minmax: tuple[int | float, int | float] = ...,
    rdnoise: float = ...,
    gain: float = ...,
    snoise: float = ...,
    pclip: float = ...,
    zero: PlaneVectorLike = ...,
    scale: PlaneVectorLike = ...,
    zero_to_0th: bool = ...,
    scale_to_0th: bool = ...,
    zero_sigclip_kwargs: SigclipStatKwargs = ...,
    scale_sigclip_kwargs: SigclipStatKwargs = ...,
    weight: np.ndarray | None = ...,
    revert_on_nkeep: bool = ...,
    grow: float | None = ...,
    diagnostics: None = ...,
    sigscale: float = ...,
    validate: bool = ...,
) -> np.ndarray: ...


@overload
def ndcombine(
    arr: np.ndarray,
    mask: np.ndarray | None = ...,
    thresholds: tuple[float, float] | list[float] | None = ...,
    combine: str = ...,
    reject: str | None = ...,
    sigma: float | tuple[float, float] = ...,
    cenfunc: str = ...,
    clip_cen: str | None = ...,
    maxiters: int = ...,
    ddof: int = ...,
    nkeep: int = ...,
    maxrej: int | None = ...,
    n_minmax: tuple[int | float, int | float] = ...,
    rdnoise: float = ...,
    gain: float = ...,
    snoise: float = ...,
    pclip: float = ...,
    zero: PlaneVectorLike = ...,
    scale: PlaneVectorLike = ...,
    zero_to_0th: bool = ...,
    scale_to_0th: bool = ...,
    zero_sigclip_kwargs: SigclipStatKwargs = ...,
    scale_sigclip_kwargs: SigclipStatKwargs = ...,
    weight: np.ndarray | None = ...,
    revert_on_nkeep: bool = ...,
    grow: float | None = ...,
    diagnostics: Literal["simple"] = ...,
    sigscale: float = ...,
    validate: bool = ...,
) -> FullCombineResult: ...


@overload
def ndcombine(
    arr: np.ndarray,
    mask: np.ndarray | None = ...,
    thresholds: tuple[float, float] | list[float] | None = ...,
    combine: str = ...,
    reject: str | None = ...,
    sigma: float | tuple[float, float] = ...,
    cenfunc: str = ...,
    clip_cen: str | None = ...,
    maxiters: int = ...,
    ddof: int = ...,
    nkeep: int = ...,
    maxrej: int | None = ...,
    n_minmax: tuple[int | float, int | float] = ...,
    rdnoise: float = ...,
    gain: float = ...,
    snoise: float = ...,
    pclip: float = ...,
    zero: PlaneVectorLike = ...,
    scale: PlaneVectorLike = ...,
    zero_to_0th: bool = ...,
    scale_to_0th: bool = ...,
    zero_sigclip_kwargs: SigclipStatKwargs = ...,
    scale_sigclip_kwargs: SigclipStatKwargs = ...,
    weight: np.ndarray | None = ...,
    revert_on_nkeep: bool = ...,
    grow: float | None = ...,
    diagnostics: Literal["full"] = ...,
    sigscale: float = ...,
    validate: bool = ...,
) -> FullDiagnosticCombineResult: ...


def ndcombine(
    arr: np.ndarray,
    mask: np.ndarray | None = None,
    thresholds: tuple[float, float] | list[float] | None = None,
    combine: str = "mean",
    reject: str | None = None,
    sigma: float | tuple[float, float] = (3.0, 3.0),
    cenfunc: str = "median",
    clip_cen: str | None = None,
    maxiters: int = 5,
    ddof: int = 0,
    nkeep: int = 1,
    maxrej: int | None = None,
    n_minmax: tuple[int | float, int | float] = (1, 1),
    rdnoise: float = 0.0,
    gain: float = 1.0,
    snoise: float = 0.0,
    pclip: float = -0.5,
    zero: PlaneVectorLike = None,
    scale: PlaneVectorLike = None,
    zero_to_0th: bool = True,
    scale_to_0th: bool = True,
    zero_sigclip_kwargs: SigclipStatKwargs = None,
    scale_sigclip_kwargs: SigclipStatKwargs = None,
    weight: np.ndarray | None = None,
    revert_on_nkeep: bool = True,
    grow: float | None = None,
    diagnostics: Diagnostics = None,
    sigscale: float = 0.1,
    validate: bool = True,
) -> np.ndarray | FullCombineResult | FullDiagnosticCombineResult:
    """Combine an image stack with optional normalization and rejection.

    This is the compact ndcombine() wrapper for call sites that prefer one
    function over an explicit :class:`Combiner` workflow. A pure
    ``combine="lmedian"`` or
    ``combine="lmed"`` call with no mask, rejection, zero, scale, or weight
    preserves accepted integer input dtypes. Any path that needs masking,
    rejection, normalization, or weights uses the normal floating workspace.

    Parameters
    ----------
    arr : ndarray, shape (N, *spatial)
        Image stack. Accepted dtypes are `uint8`, `uint16`, `int16`, `int32`,
        `float32`, and `float64`. `uint8`, `uint16`, and `int16` are promoted
        to `float32`; `int32` is promoted to `float64`; `float32` and
        `float64` are preserved. Other dtypes, including `int64` and
        `float128`, are not silently cast. If your data use an unsupported
        dtype, cast them explicitly before calling `ndcombine`; there is no
        full-precision `int64` combine path because the package does not
        provide a `float128` workspace.
        Inputs with more than 3 dimensions are flattened internally; all
        outputs are reshaped back to the original trailing spatial dimensions.
    mask : ndarray of bool, optional
        Input mask with the same shape as `arr`; `True` means masked.
    thresholds : tuple of float or None, optional
        Inclusive lower and upper retained-value bounds applied before
        zero/scale, rejection, and combination. Values below the lower bound
        or above the upper bound are recorded in `mask_thresh` when
        diagnostics are requested and are treated as masked for later stages.
        For CCD images, a typical choice is ``thresholds=(0, satlevel)`` to
        remove bad/cold pixels and saturated pixels, or ``thresholds=(0, 0.99 *
        satlevel)`` to also exclude near-saturated pixels that are usually not
        useful for normalization, rejection, or final combination.
    combine : str, optional
        Combine method. Supported aliases are `"mean"`, `"average"`, `"avg"`,
        `"median"`, `"med"`, `"lmedian"`, `"lmed"`, `"sum"`, `"min"`,
        `"max"`, `"variance"`, and `"var"`.
    reject : {"minmax", "pclip", "sigclip", "ccdclip"} or None, optional
        Rejection algorithm to apply before combining. `None` disables
        rejection.
    sigma : float or tuple of float, optional
        Lower and upper sigma thresholds for `"sigclip"` and `"ccdclip"`.
    cenfunc : {"median", "lmedian", "mean"}, optional
        Center estimator used by iterative rejection.
    clip_cen : {"median", "lmedian", "mean"} or None, optional
        Center used to compute the spread for iterative rejection. `None` uses
        `cenfunc`. `lmedian` accepts the alias `lmed`.
    maxiters : int, optional
        Maximum number of rejection iterations.
    ddof : int, optional
        Delta degrees of freedom for spread estimates and for
        ``combine="variance"``. Variance is computed from final valid values
        after input mask and rejection are applied.
    nkeep : int, optional
        Minimum number of unmasked values to preserve at each pixel. The
        default, `1`, prevents clipping from rejecting every finite sample in
        a per-output stack. Set `0` only when all samples may be rejected.
    maxrej : int or None, optional
        Maximum number of values that may be rejected at each pixel. `None`
        means no limit.
    n_minmax : tuple of int or float, optional
        Number or fraction of low and high values rejected by `"minmax"`.
        Values greater than or equal to 1 are frame counts; values in `[0, 1)`
        are fractions of the original stack size.
    rdnoise : float, optional
        CCD read noise in electrons, used by `"ccdclip"`.
    gain : float, optional
        CCD gain in electrons per DN (ADU), used by `"ccdclip"`. Must be
        finite and positive when validation is enabled.
    snoise : float, optional
        Fractional sensitivity-noise coefficient used by `"ccdclip"`. It
        models multiplicative uncertainty, such as flat-field noise, through
        the ``(snoise * signal)^2`` variance term.
    pclip : float, optional
        IRAF-style pclip rank offset. Values with ``abs(pclip) < 1`` are
        converted to an integer offset using half of the input image count.
    zero : array-like, str, callable, or None, optional
        Per-plane values to subtract before scaling. Strings select per-plane
        statistics: plain (`"mean"`, `"median"`, `"sum"`, `"min"`, `"max"`
        and common aliases) or sigma-clipped (`"sigclip_mean"` / `"mean_sc"`
        and `"sigclip_median"` / `"median_sc"` / `"med_sc"`, using
        ``sigma=3``, ``maxiters=5`` by default). Callables are evaluated once
        per image plane.
    scale : array-like, str, callable, or None, optional
        Per-plane values used as divisors after zero subtraction. The same
        string and callable rules as `zero` apply, including the sigma-clipped
        variants. Scale values must be finite and non-zero when validation is
        enabled.
    zero_to_0th : bool, optional
        If `True`, subtract the first zero value from all zero values. `True`
        is the default behavior to follow IRAF. Setting this to `True` makes
        only a small subtraction happen, so real pixel values change only
        slightly and Poisson-noise calculations remain well behaved.
    scale_to_0th : bool, optional
        If `True`, divide all scale values by the first scale value. `True` is
        the default behavior to follow IRAF. Setting this to `True` makes only
        a small division happen, so real pixel values change only slightly and
        Poisson-noise calculations remain well behaved.
    zero_sigclip_kwargs, scale_sigclip_kwargs : dict or None, optional
        Keyword arguments used only when `zero` or `scale` is a sigma-clipped
        statistic alias such as `"mean_sc"` or `"med_sc"`. Supported keys are
        `sigma`, `maxiters`, `cenfunc`, `clip_cen`, and `ddof`.
    weight : ndarray, shape (N,), optional
        Optional per-plane weights for mean/average combine.
    revert_on_nkeep : bool, optional
        If `True`, an iteration that would leave fewer than `nkeep` usable
        samples at a pixel is reverted in full for that pixel. Input masks,
        threshold masks, and non-finite samples remain excluded. The default is
        `True`. If `False`, clipping is strict and may leave fewer than
        `nkeep` samples.
    grow : float or None, optional
        Non-negative radius in pixels used to grow `mask_rej` spatially after
        rejection. Axis 0 is the stack axis and is never grown across. `None`
        disables growth and skips the extra calculation. Growth expands all
        entries in the returned rejection mask, including input masks,
        threshold exclusions, and non-finite samples.
    diagnostics : {None, "simple", "full"}, optional
        Diagnostic product level. `None` returns only the combined image and
        keeps the fused output-only fast paths. `"simple"` returns the combined
        image plus the existing per-output-element diagnostics. `"full"` returns the
        `"simple"` products plus a stack-shaped `uint8` `sample_flags` array.
    sigscale : float, optional
        IRAF scale gate for `"ccdclip"` noise corrections. Resolved per-image
        zero/scale vectors are used only when a scale differs from one by more
        than `sigscale`; `0.0` disables the correction.
    validate : bool, optional
        If `True`, validate public-boundary inputs. If `False`, callers must
        provide arrays with correct dimensionality, contiguity, dtype, mask
        shape, finite scale values, and valid rejection parameters.

    Returns
    -------
    combined : ndarray, shape (*spatial)
        Combined image, returned when `diagnostics` is `None`.
    combined, mask_rej, mask_thresh, std, low, upp, nit, output_flags : tuple
        Returned when ``diagnostics="simple"``.
        Rejection diagnostics are `None` when `reject` is `None`; `mask_thresh`
        is `None` when `thresholds` is
        `None`. `mask_rej` and `mask_thresh` have shape ``(N, *spatial)``;
        `low`, `upp`, `nit`, and `output_flags` have shape ``(*spatial,)``. `std`
        has shape ``(*spatial,)`` for sigma/CCD clipping and is `None` for
        rejection algorithms without a spread diagnostic. `low` and `upp` are
        retained extrema before growth, not the clipping thresholds.
        ``mask_rej.sum(axis=0)`` counts all marked samples, including prior
        masks, non-finite inputs, and growth. Use `sample_flags` with full
        diagnostics to distinguish causes. The number actually used
        for combining is the count of finite samples after input mask and
        rejection mask are applied. `output_flags` bit ``16`` marks output elements
        where growth added at least one rejected sample; `low`, `upp`, `nit`,
        and `std` still describe the underlying clipping calculation. `std`
        is the clipping spread for sigma clipping and the reference noise
        for CCD clipping, not the uncertainty of the combined image.
    combined, ..., sample_flags : tuple
        Returned when ``diagnostics="full"``. `sample_flags` has dtype `uint8`
        and shape ``(N, *spatial)``. Its bits are `1=input mask/BPM`,
        `2=non-finite`, `4=threshold mask`, `8=final algorithm rejection`,
        `16=added by grow`, and `32=already masked before this rejection
        step`; iterative sigma/CCD clipping can also set `64=tentatively
        rejected then restored by nkeep` and `128=tentatively rejected then
        restored by maxrej`. The per-output-element `output_flags` map uses a
        separate bit namespace.
    """
    orig_shape = arr.shape
    trailing = arr.shape[1:]
    grow = _validate_grow_value(grow, validate)
    diagnostic_level = normalize_diagnostics(diagnostics)
    want_diagnostics = diagnostic_level is not None
    want_sample_flags = diagnostic_level == "full"
    cb = combine.lower()

    if (
        reject is None
        and mask is None
        and thresholds is None
        and zero is None
        and scale is None
        and weight is None
        and grow is None
        and cb in kernels._STACK_FUNCS
    ):
        fun = kernels._STACK_FUNCS[cb]
        kwargs = {"ddof": ddof} if fun is kernels.rd.nanvar else {}
        out = kernels._stack(arr, fun, validate=validate, **kwargs)
        if want_sample_flags:
            sample_flags = sample_flags_array(np.asarray(arr)).reshape(orig_shape)
            return out, None, None, None, None, None, None, None, sample_flags
        return (
            (out, None, None, None, None, None, None, None) if want_diagnostics else out
        )

    arr = validate_stack(arr) if validate else np.ascontiguousarray(arr)
    if validate:
        if mask is not None and mask.shape != arr.shape and mask.shape == orig_shape:
            mask = mask.reshape(arr.shape)
        mask = validate_mask(mask, arr.shape)
        validate_thresholds(thresholds)
    else:
        mask = None if mask is None else np.ascontiguousarray(mask)

    mask_thresh = threshold_mask(arr, thresholds)
    mask_pre = (
        mask_thresh
        if mask is None
        else (mask | mask_thresh if mask_thresh is not None else mask)
    )
    if zero is not None or scale is not None:
        arr_zs, z_vec, s_vec = _apply_zero_scale(
            arr,
            zero,
            scale,
            zero_to_0th=zero_to_0th,
            scale_to_0th=scale_to_0th,
            zero_sigclip_kwargs=zero_sigclip_kwargs,
            scale_sigclip_kwargs=scale_sigclip_kwargs,
            mask=mask_pre,
            validate=validate,
        )
    else:
        arr_zs, z_vec, s_vec = arr, None, None
    # (when validate=False the caller is responsible for supplying compatible inputs)

    mask_rej = low = upp = nit = output_flags = std = sample_flags = None
    sample_mask_rej = None
    sample_restored_flags = None
    if reject is not None:
        rj = reject.lower()
        if rj == "minmax":
            if want_diagnostics:
                mask_rej, std, low, upp, nit, output_flags = kernels.minmax(
                    arr_zs,
                    mask=mask_pre,
                    n_min=n_minmax[0],
                    n_max=n_minmax[1],
                    validate=validate,
                )
            else:
                if (
                    cb in ("mean", "average", "avg", "median", "med")
                    and grow is None
                    and weight is None
                ):
                    return kernels.minmax_combine(
                        arr_zs,
                        mask=mask_pre,
                        combine=combine,
                        n_min=n_minmax[0],
                        n_max=n_minmax[1],
                        validate=validate,
                    ).reshape(trailing)
                mask_rej = kernels.minmax_mask(
                    arr_zs,
                    mask=mask_pre,
                    n_min=n_minmax[0],
                    n_max=n_minmax[1],
                    grow=grow,
                    validate=validate,
                )
        elif rj == "pclip":
            if want_diagnostics:
                mask_rej, std, low, upp, nit, output_flags = kernels.pclip(
                    arr_zs,
                    mask=mask_pre,
                    frac=pclip,
                    sigma=sigma,
                    nkeep=nkeep,
                    validate=validate,
                )
            else:
                if (
                    cb in ("mean", "average", "avg", "median", "med")
                    and grow is None
                    and weight is None
                ):
                    return kernels.pclip_combine(
                        arr_zs,
                        mask=mask_pre,
                        combine=combine,
                        frac=pclip,
                        sigma=sigma,
                        nkeep=nkeep,
                        validate=validate,
                    ).reshape(trailing)
                mask_rej = kernels.pclip_mask(
                    arr_zs,
                    mask=mask_pre,
                    frac=pclip,
                    sigma=sigma,
                    nkeep=nkeep,
                    grow=grow,
                    validate=validate,
                )
        elif rj == "sigclip":
            if want_diagnostics:
                mask_rej, std, low, upp, nit, output_flags = kernels.sigclip(
                    arr_zs,
                    mask=mask_pre,
                    sigma=sigma,
                    maxiters=maxiters,
                    ddof=ddof,
                    nkeep=nkeep,
                    maxrej=maxrej,
                    cenfunc=cenfunc,
                    clip_cen=clip_cen,
                    revert_on_nkeep=revert_on_nkeep,
                    validate=validate,
                )
                if want_sample_flags:
                    sample_restored_flags = kernels._sigclip_restored_flags(
                        arr_zs,
                        mask=mask_pre,
                        sigma=sigma,
                        maxiters=maxiters,
                        ddof=ddof,
                        nkeep=nkeep,
                        maxrej=maxrej,
                        cenfunc=cenfunc,
                        clip_cen=clip_cen,
                        revert_on_nkeep=revert_on_nkeep,
                        validate=validate,
                    )
            else:
                if (
                    cb in ("mean", "average", "avg", "median", "med")
                    and grow is None
                    and weight is None
                ):
                    return kernels.sigclip_combine(
                        arr_zs,
                        mask=mask_pre,
                        combine=combine,
                        sigma=sigma,
                        maxiters=maxiters,
                        ddof=ddof,
                        nkeep=nkeep,
                        maxrej=maxrej,
                        cenfunc=cenfunc,
                        clip_cen=clip_cen,
                        revert_on_nkeep=revert_on_nkeep,
                        validate=validate,
                    ).reshape(trailing)
                mask_rej = kernels.sigclip_mask(
                    arr_zs,
                    mask=mask_pre,
                    sigma=sigma,
                    maxiters=maxiters,
                    ddof=ddof,
                    nkeep=nkeep,
                    maxrej=maxrej,
                    cenfunc=cenfunc,
                    clip_cen=clip_cen,
                    revert_on_nkeep=revert_on_nkeep,
                    grow=grow,
                    validate=validate,
                )
        elif rj == "ccdclip":
            gain = validate_positive_scalar("gain", gain) if validate else float(gain)
            if validate and (not np.isfinite(sigscale) or sigscale < 0.0):
                raise ValueError("sigscale must be finite and non-negative")
            ccd_scales = ccd_zeros = None
            if z_vec is not None or s_vec is not None:
                n = arr_zs.shape[0]
                scales = (
                    np.ones(n, dtype=np.float64)
                    if s_vec is None
                    else (
                        np.full(
                            n,
                            np.asarray(s_vec, dtype=np.float64).item(),
                            dtype=np.float64,
                        )
                        if np.asarray(s_vec).size == 1
                        else np.asarray(s_vec, dtype=np.float64).reshape(n, -1)[:, 0]
                    )
                )
                zeros = (
                    np.zeros(n, dtype=np.float64)
                    if z_vec is None
                    else (
                        np.full(
                            n,
                            np.asarray(z_vec, dtype=np.float64).item(),
                            dtype=np.float64,
                        )
                        if np.asarray(z_vec).size == 1
                        else np.asarray(z_vec, dtype=np.float64).reshape(n, -1)[:, 0]
                    )
                )
                doscale = bool(np.any(scales != scales[0]))
                doscale1 = bool(
                    doscale
                    and sigscale != 0.0
                    and np.any(np.abs(scales - 1.0) > sigscale)
                )
                if doscale1:
                    ccd_scales = scales
                    ccd_zeros = zeros / scales
            if (
                not want_diagnostics
                and cb in ("mean", "average", "avg", "median", "med")
                and grow is None
                and weight is None
            ):
                return kernels.ccdclip_combine(
                    arr_zs,
                    mask=mask_pre,
                    combine=combine,
                    sigma=sigma,
                    maxiters=maxiters,
                    ddof=ddof,
                    nkeep=nkeep,
                    maxrej=maxrej,
                    cenfunc=cenfunc,
                    clip_cen=clip_cen,
                    revert_on_nkeep=revert_on_nkeep,
                    rdnoise=rdnoise,
                    gain=gain,
                    snoise=snoise,
                    scales=ccd_scales,
                    zeros=ccd_zeros,
                    validate=validate,
                ).reshape(trailing)
            if want_diagnostics:
                mask_rej, std, low, upp, nit, output_flags = kernels.ccdclip(
                    arr_zs,
                    mask=mask_pre,
                    sigma=sigma,
                    maxiters=maxiters,
                    ddof=ddof,
                    nkeep=nkeep,
                    maxrej=maxrej,
                    cenfunc=cenfunc,
                    clip_cen=clip_cen,
                    revert_on_nkeep=revert_on_nkeep,
                    rdnoise=rdnoise,
                    gain=gain,
                    snoise=snoise,
                    scales=ccd_scales,
                    zeros=ccd_zeros,
                    validate=validate,
                )
                if want_sample_flags:
                    sample_restored_flags = kernels._ccdclip_restored_flags(
                        arr_zs,
                        mask=mask_pre,
                        sigma=sigma,
                        maxiters=maxiters,
                        ddof=ddof,
                        nkeep=nkeep,
                        maxrej=maxrej,
                        cenfunc=cenfunc,
                        clip_cen=clip_cen,
                        revert_on_nkeep=revert_on_nkeep,
                        rdnoise=rdnoise,
                        gain=gain,
                        snoise=snoise,
                        scales=ccd_scales,
                        zeros=ccd_zeros,
                        validate=validate,
                    )
            else:
                mask_rej = kernels.ccdclip_mask(
                    arr_zs,
                    mask=mask_pre,
                    sigma=sigma,
                    maxiters=maxiters,
                    ddof=ddof,
                    nkeep=nkeep,
                    maxrej=maxrej,
                    cenfunc=cenfunc,
                    clip_cen=clip_cen,
                    revert_on_nkeep=revert_on_nkeep,
                    rdnoise=rdnoise,
                    gain=gain,
                    snoise=snoise,
                    scales=ccd_scales,
                    zeros=ccd_zeros,
                    grow=grow,
                    validate=validate,
                )
        else:
            raise ValueError(f"unknown reject method: {reject}")

        sample_mask_rej = None if mask_rej is None else mask_rej.copy()

        if want_diagnostics and grow is not None:
            mask_rej, output_flags = kernels._grow_rejection_mask(
                mask_rej.reshape(orig_shape),
                output_flags.reshape(trailing),
                grow,
                validate=validate,
            )
            mask_rej = mask_rej.reshape(arr.shape)
            output_flags = output_flags.reshape(arr.shape[1:])

    if want_sample_flags:
        sample_flags = sample_flags_array(
            arr_zs,
            input_mask=mask,
            threshold_mask=mask_thresh,
            mask_rej=sample_mask_rej,
            grown_mask=mask_rej if grow is not None else None,
            restored_flags=sample_restored_flags,
        )

    # Apply rejection + input mask via NaN-fill so combine kernels skip them.
    if mask_rej is not None or mask_pre is not None:
        m_total = (
            mask_rej
            if mask_pre is None
            else (mask_rej | mask_pre if mask_rej is not None else mask_pre)
        )
        arr_eff = mask_as_nan(arr_zs, m_total)
    else:
        arr_eff = arr_zs

    if weight is not None:
        if cb not in ("mean", "average", "avg"):
            raise ValueError("weight can only be used with mean combine")
        out = kernels.nanaverage(
            arr_eff,
            validate_weights(weight, arr_eff.shape[0])
            if validate
            else np.asarray(weight, dtype=np.float64),
            validate=validate,
        )
    elif cb in kernels._STACK_FUNCS:
        fun = kernels._STACK_FUNCS[cb]
        kwargs = {"ddof": ddof} if fun is kernels.rd.nanvar else {}
        out = kernels._stack(arr_eff, fun, validate=validate, **kwargs)
    else:
        raise ValueError(f"unknown combine method: {combine}")

    # Reshape all outputs back to the original trailing spatial dimensions.
    out = out.reshape(trailing)
    if want_diagnostics:
        if mask_rej is not None:
            mask_rej = mask_rej.reshape(orig_shape)
        if mask_thresh is not None:
            mask_thresh = mask_thresh.reshape(orig_shape)
        if low is not None:
            low = low.reshape(trailing)
            upp = upp.reshape(trailing)
            nit = nit.reshape(trailing)
            output_flags = output_flags.reshape(trailing)
            if std is not None:
                std = std.reshape(trailing)
        if sample_flags is not None:
            sample_flags = sample_flags.reshape(orig_shape)
            return (
                out,
                mask_rej,
                mask_thresh,
                std,
                low,
                upp,
                nit,
                output_flags,
                sample_flags,
            )
        return out, mask_rej, mask_thresh, std, low, upp, nit, output_flags
    return out
