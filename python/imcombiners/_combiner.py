"""Fluent :class:`Combiner` object.

Wraps an image stack and supports two public styles:

- Standard Combiner:
  ``Combiner(arr).combine("median", rejectors=SigClip(...), diagnostics=None)``;
- Chained Combiner:
  ``Combiner(arr).threshold(...).zero_scale(...).reject(SigClip(...)).combine("median")``.

The combiner is **mutable** for ergonomics: each chained call updates internal
state and returns ``self``. If you need to fork a pipeline mid-way, use
:meth:`Combiner.copy`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Literal

import numpy as np

from . import kernels
from ._diagnostics import (
    Diagnostics,
    normalize_diagnostics,
    normalize_reject_diagnostics,
    sample_flags_array,
)
from ._rejectors import CcdClip, LinearClip, MinMaxClip, PClip, Rejector, SigClip
from ._typing import PlaneVectorLike, RejectionResult, SigclipStatKwargs
from ._validation import (
    mask_as_nan,
    resolve_zero_scale,
    threshold_mask,
    validate_mask,
    validate_stack,
    validate_thresholds,
    validate_weights,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["Combiner"]


_COMBINE_METHODS = kernels._STACK_FUNCS


def _is_threshold_bounds(value: object) -> bool:
    """Return whether `value` looks like one `(low, upp)` threshold pair."""
    return (
        isinstance(value, Sequence)
        and len(value) == 2
        and np.isscalar(value[0])
        and np.isscalar(value[1])
    )


def _normalize_threshold_steps(
    thresholds: tuple[float, float] | Sequence[tuple[float, float]] | None,
) -> list[tuple[float, float]]:
    """Return combine-call threshold steps as validated float pairs."""
    if thresholds is None:
        return []
    if _is_threshold_bounds(thresholds):
        low, upp = thresholds
        return [(float(low), float(upp))]

    steps: list[tuple[float, float]] = []
    for item in thresholds:
        if not _is_threshold_bounds(item):
            raise ValueError(
                "thresholds must be a (low, upp) pair or a sequence of pairs"
            )
        low, upp = item
        steps.append((float(low), float(upp)))
    return steps


def _normalize_rejector_steps(
    rejectors: Rejector | Sequence[Rejector] | None,
) -> list[Rejector]:
    """Return combine-call rejector steps as a list."""
    if rejectors is None:
        return []
    if isinstance(rejectors, Rejector):
        return [rejectors]
    steps = list(rejectors)
    for item in steps:
        if not isinstance(item, Rejector):
            raise TypeError(
                "rejectors must be a Rejector or a sequence of Rejector "
                f"instances; got {type(item).__name__}"
            )
    return steps


def _combine_array(
    arr: NDArray,
    method: str,
    *,
    mask: NDArray | None,
    weight: NDArray | None,
    ddof: int,
    validate: bool,
) -> NDArray:
    """Combine `arr` with an optional mask without mutating a Combiner."""
    m = method.lower()
    arr_eff = mask_as_nan(arr, mask) if mask is not None else arr

    if weight is not None:
        if m not in ("mean", "average", "avg"):
            raise ValueError("weight can only be used with mean combine")
        result = kernels.nanaverage(
            arr_eff,
            validate_weights(weight, arr_eff.shape[0])
            if validate
            else np.asarray(weight, dtype=np.float64),
            validate=validate,
        )
        return result

    canonical = _COMBINE_METHODS.get(m)
    if canonical is None:
        raise ValueError(f"unknown combine method: {method}")
    kwargs = {"ddof": ddof} if canonical is kernels.rd.nanvar else {}
    return kernels._stack(arr_eff, canonical, validate=validate, **kwargs)


def _apply_zero_scale(
    arr: NDArray,
    zero: PlaneVectorLike,
    scale: PlaneVectorLike,
    *,
    mask: NDArray | None,
    zero_to_0th: bool,
    scale_to_0th: bool,
    zero_sigclip_kwargs: SigclipStatKwargs,
    scale_sigclip_kwargs: SigclipStatKwargs,
    validate: bool,
) -> NDArray:
    """Return `arr` after optional zero subtraction and scale division."""
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
    return out if copied else arr


def _fused_reject_combine(
    arr: NDArray,
    rejector: Rejector,
    *,
    combine: str,
    mask: NDArray | None,
    validate: bool,
) -> NDArray | None:
    """Return fused output-only result when the rejector/method supports it."""
    if not isinstance(rejector, Rejector):
        raise TypeError(
            f"rejectors must contain Rejector instances; got {type(rejector).__name__}"
        )
    if getattr(rejector, "grow", None) is not None:
        return None

    if isinstance(rejector, SigClip):
        return kernels.sigclip_combine(
            arr,
            mask=mask,
            combine=combine,
            sigma=rejector._sigma_pair(),
            maxiters=rejector.maxiters,
            ddof=rejector.ddof,
            nkeep=rejector.nkeep,
            maxrej=rejector.maxrej,
            cenfunc=rejector.cenfunc,
            clip_cen=rejector.clip_cen,
            revert_on_nkeep=rejector.revert_on_nkeep,
            validate=validate,
        )
    if isinstance(rejector, CcdClip):
        return kernels.ccdclip_combine(
            arr,
            mask=mask,
            combine=combine,
            sigma=rejector._sigma_pair(),
            maxiters=rejector.maxiters,
            ddof=rejector.ddof,
            nkeep=rejector.nkeep,
            maxrej=rejector.maxrej,
            cenfunc=rejector.cenfunc,
            clip_cen=rejector.clip_cen,
            revert_on_nkeep=rejector.revert_on_nkeep,
            rdnoise=rejector.rdnoise,
            gain=rejector.gain,
            snoise=rejector.snoise,
            scale_ref=rejector.scale_ref,
            zero_ref=rejector.zero_ref,
            validate=validate,
        )
    if isinstance(rejector, MinMaxClip):
        return kernels.minmax_combine(
            arr,
            mask=mask,
            combine=combine,
            n_min=rejector.n_min,
            n_max=rejector.n_max,
            validate=validate,
        )
    if isinstance(rejector, PClip):
        return kernels.pclip_combine(
            arr,
            mask=mask,
            combine=combine,
            frac=rejector.frac,
            sigma=rejector.sigma,
            nkeep=rejector.nkeep,
            validate=validate,
        )
    if isinstance(rejector, LinearClip):
        return None
    return None


class Combiner:
    """Stateful pipeline holder for `(scale/zero, reject, combine)` workflows.

    Use Standard Combiner with ``rejectors=...`` and
    ``diagnostics=None`` for production reductions. Use Chained Combiner calls
    when you need retained masks, rejection diagnostics, or step-by-step
    inspection.

    Parameters
    ----------
    arr : ndarray, shape (N, *spatial)
        Image stack. Accepted public input dtypes are `uint8`, `uint16`,
        `int16`, `int32`, `float32`, and `float64`. `uint8`, `uint16`, and
        `int16` are promoted to `float32`; `int32` is promoted to `float64`;
        `float32` and `float64` are preserved. Other dtypes, including
        `int64` and `float128`, are not silently cast. ``Combiner`` is
        intended for workflows that may apply masking, rejection, zero, or
        scale. Inputs with more than 3 dimensions are flattened to
        ``(N, prod(spatial), 1)`` for internal processing; all outputs (from
        :meth:`combine` and :attr:`last_reject`) are reshaped back to the
        original trailing dimensions.
    mask : ndarray of bool, optional
        Input mask with the same shape as `arr`; ``True`` = masked. ``None``
        means no mask.
    copy : bool, optional
        If `True`, copy `arr` and `mask` into owned workspace arrays. If
        `False`, keep references to normalized inputs when possible.
    validate : bool, optional
        If `True`, validate shape, dtype, mask compatibility, and contiguity.
        If `False`, callers must provide a 3-D contiguous stack with a dtype
        accepted by the compiled kernels and a same-shape boolean mask.

    Attributes
    ----------
    arr : ndarray
        Working stack (after any ``zero_scale``), always 3-D internally.
        Owned, may be modified.
    mask : ndarray of bool or None
        Combined mask (union of input, threshold, and rejection masks), 3-D.
        ``None`` until one is set.
    mask_thresh : ndarray of bool or None
        Threshold mask from :meth:`threshold`, shaped like the original input
        stack. ``None`` until thresholding has been applied.
    last_reject : tuple or None
        ``(mask_rej, std, low, upp, nit, output_flags)`` from the most recent
        :meth:`reject` call, shaped to match the original trailing spatial
        dimensions. If `grow` was used, `mask_rej` is the grown rejection
        mask; `low`, `upp`, `nit`, and `std` still describe the underlying
        clipping calculation, and `output_flags` bit ``16`` marks output elements where
        growth added at least one rejected sample. ``None`` until
        :meth:`reject` has been called.
    last_sample_flags : ndarray of uint8 or None
        Stack-shaped per-sample flags from the most recent :meth:`reject` call
        made with ``diagnostics="full"``. This is stage-local like
        :attr:`last_reject`; samples already masked before the rejection step
        are marked with sample bit ``32`` rather than replaying earlier-stage
        causes. Iterative sigma/CCD/linear clipping can also mark samples
        tentatively rejected and then restored by `nkeep` or `maxrej`.
    """

    def __init__(
        self,
        arr: NDArray,
        mask: NDArray | None = None,
        *,
        copy: bool = True,
        validate: bool = True,
    ) -> None:
        """Initialize a combiner workspace.

        Parameters are documented on :class:`Combiner`.
        """
        orig_shape = arr.shape
        self._trailing_shape: tuple[int, ...] = arr.shape[1:]
        if validate:
            arr = validate_stack(arr)  # reshapes to (N, P, 1) for ndim > 3
            if (
                mask is not None
                and mask.shape != arr.shape
                and mask.shape == orig_shape
            ):
                mask = mask.reshape(arr.shape)
            mask = validate_mask(mask, arr.shape)
        else:
            arr = np.ascontiguousarray(arr)
            mask = None if mask is None else np.ascontiguousarray(mask)
        self._validate = bool(validate)
        self.arr: NDArray = arr.copy() if copy else arr
        self.mask: NDArray | None = (
            None if mask is None else (mask.copy() if copy else mask)
        )
        self._mask_thresh: NDArray | None = None
        self.last_reject: RejectionResult | None = None
        self.last_sample_flags: NDArray[np.uint8] | None = None
        self.reject_history: list[RejectionResult] = []

    # ---- chainable transforms ----------------------------------------------------

    def threshold(self, low: float = -np.inf, upp: float = np.inf) -> Combiner:
        """Mask values outside inclusive threshold bounds.

        Thresholding is a pre-rejection mask stage. Values survive when
        ``low <= value <= upp``; values below `low` or above `upp` are recorded
        in :attr:`mask_thresh` and OR-merged into the running mask. Later
        ``zero_scale`` statistics, rejection, and combination ignore these
        thresholded pixels.

        A common CCD-image pattern is ``low=0`` to remove bad/cold pixels and
        ``upp=satlevel`` or ``upp=0.99 * satlevel`` to remove saturated or
        near-saturated pixels before they influence normalization or rejection.

        Parameters
        ----------
        low, upp : float, optional
            Inclusive lower and upper retained-value bounds.

        Returns
        -------
        combiner : Combiner
            This object, after updating `mask` and `mask_thresh`.
        """
        validate_thresholds((low, upp))
        mask_thresh = threshold_mask(self.arr, (low, upp))
        if mask_thresh is None:
            return self
        self._mask_thresh = (
            mask_thresh
            if self._mask_thresh is None
            else (self._mask_thresh | mask_thresh)
        )
        self.mask = mask_thresh if self.mask is None else (self.mask | mask_thresh)
        return self

    def zero_scale(
        self,
        zero: PlaneVectorLike = None,
        scale: PlaneVectorLike = None,
        *,
        zero_to_0th: bool = True,
        scale_to_0th: bool = True,
        zero_sigclip_kwargs: SigclipStatKwargs = None,
        scale_sigclip_kwargs: SigclipStatKwargs = None,
    ) -> Combiner:
        """Apply per-image offset / scale: ``(arr - zero) / scale``.

        Either or both may be `None`. Modifies `self.arr` in place and returns
        `self` for chaining.

        Parameters
        ----------
        zero : array-like, str, callable, or None, optional
            Per-plane values to subtract before scaling. Array-like inputs must
            be scalar or length `N`. Strings select a per-plane statistic:
            plain (`"mean"`, `"median"`, `"sum"`, `"min"`, `"max"` plus
            aliases) or sigma-clipped (`"sigclip_mean"` / `"mean_sc"` and
            `"sigclip_median"` / `"median_sc"` / `"med_sc"`, using
            ``sigma=3``, ``maxiters=5`` by default). Callables are evaluated
            once per image plane and must return a scalar. Current masks,
            including threshold masks, are applied before string statistics are
            computed.
        scale : array-like, str, callable, or None, optional
            Per-plane values used as divisors after zero subtraction. The same
            string and callable rules as `zero` apply, including the
            sigma-clipped variants. Scale values must be finite and non-zero
            when validation is enabled.
        zero_to_0th : bool, optional
            If `True`, subtract the first zero value from all zero values
            before applying them. `True` is the default behavior to follow
            IRAF. Setting this to `True` makes only a small subtraction happen,
            so real pixel values change only slightly and Poisson-noise
            calculations remain well behaved.
        scale_to_0th : bool, optional
            If `True`, divide all scale values by the first scale value before
            applying them. `True` is the default behavior to follow IRAF.
            Setting this to `True` makes only a small division happen, so real
            pixel values change only slightly and Poisson-noise calculations
            remain well behaved.
        zero_sigclip_kwargs, scale_sigclip_kwargs : dict or None, optional
            Keyword arguments used only when `zero` or `scale` is a
            sigma-clipped statistic alias such as `"mean_sc"` or `"med_sc"`.
            Supported keys are `sigma`, `maxiters`, `cenfunc`, `clip_cen`, and
            `ddof`.

        Returns
        -------
        combiner : Combiner
            This object, after updating its working stack.
        """
        arr_stat = (
            mask_as_nan(self.arr, self.mask) if self.mask is not None else self.arr
        )
        z = (
            resolve_zero_scale(
                "zero",
                zero,
                arr_stat,
                to_0th=zero_to_0th,
                sigclip_kwargs=dict(zero_sigclip_kwargs)
                if zero_sigclip_kwargs is not None
                else None,
                validate=self._validate,
            )
            if zero is not None
            else None
        )
        s = (
            resolve_zero_scale(
                "scale",
                scale,
                arr_stat,
                nonzero=True,
                to_0th=scale_to_0th,
                sigclip_kwargs=dict(scale_sigclip_kwargs)
                if scale_sigclip_kwargs is not None
                else None,
                validate=self._validate,
            )
            if scale is not None
            else None
        )
        if z is not None:
            self.arr = self.arr - z
        if s is not None:
            self.arr = self.arr / s
        return self

    def reject(
        self,
        rejector: Rejector,
        *,
        grow: float | None = None,
        diagnostics: Literal["simple", "full"] | None = "simple",
    ) -> Combiner:
        """Apply a rejector, OR-merging its mask into ``self.mask``.

        ``rejector`` is any :class:`~imcombiners._rejectors.Rejector` subclass
        (``SigClip``, ``CcdClip``, ``LinearClip``, ``MinMaxClip``, ``PClip``).
        If `grow` is not `None`, the rejection mask is grown spatially within
        each input plane before it is merged into the running mask.
        After a rejection step, ``self.mask_rej.sum(axis=0)`` is the number
        rejected by that step at each output element, including any grown
        samples. The number actually used by later combine calls is
        ``np.sum(np.isfinite(self.arr) & ~self.mask, axis=0)`` after all input
        and rejection masks have been merged.

        Parameters
        ----------
        rejector : Rejector
            Rejection object to apply.
        grow : float or None, optional
            Non-negative radius in pixels used to grow this rejection mask over
            the original spatial axes. Axis 0 is the stack axis and is never
            grown across. `None` disables mask growth and avoids the extra
            calculation.
        diagnostics : {None, "simple", "full"}, optional
            Rejection diagnostic level. `None` and `"simple"` keep the current
            `last_reject` products. `"full"` also records stack-shaped
            `sample_flags` for this rejection step.

        Returns
        -------
        combiner : Combiner
            This object, after updating `mask` and `last_reject`.
        """
        if not isinstance(rejector, Rejector):
            raise TypeError(
                "reject() expects a Rejector "
                "(SigClip/CcdClip/LinearClip/MinMaxClip/PClip); "
                f"got {type(rejector).__name__}"
            )
        diagnostic_level = normalize_reject_diagnostics(diagnostics)
        effective_grow = grow if grow is not None else getattr(rejector, "grow", None)
        applied_rejector = (
            replace(rejector, grow=None)
            if effective_grow is not None and hasattr(rejector, "grow")
            else rejector
        )
        mask_before = None if self.mask is None else self.mask.copy()
        mask_rej, std, low, upp, nit, output_flags = applied_rejector.apply(
            self.arr, mask=self.mask, validate=self._validate
        )
        mask_rej_before_grow = mask_rej.copy()
        restored_flags = None
        if diagnostic_level == "full":
            if isinstance(applied_rejector, SigClip):
                restored_flags = kernels._sigclip_restored_flags(
                    self.arr,
                    mask=self.mask,
                    sigma=applied_rejector.sigma,
                    maxiters=applied_rejector.maxiters,
                    ddof=applied_rejector.ddof,
                    nkeep=applied_rejector.nkeep,
                    maxrej=applied_rejector.maxrej,
                    cenfunc=applied_rejector.cenfunc,
                    clip_cen=applied_rejector.clip_cen,
                    revert_on_nkeep=applied_rejector.revert_on_nkeep,
                    validate=self._validate,
                )
            elif isinstance(applied_rejector, CcdClip):
                restored_flags = kernels._ccdclip_restored_flags(
                    self.arr,
                    mask=self.mask,
                    sigma=applied_rejector.sigma,
                    maxiters=applied_rejector.maxiters,
                    ddof=applied_rejector.ddof,
                    nkeep=applied_rejector.nkeep,
                    maxrej=applied_rejector.maxrej,
                    cenfunc=applied_rejector.cenfunc,
                    clip_cen=applied_rejector.clip_cen,
                    revert_on_nkeep=applied_rejector.revert_on_nkeep,
                    rdnoise=applied_rejector.rdnoise,
                    gain=applied_rejector.gain,
                    snoise=applied_rejector.snoise,
                    scale_ref=applied_rejector.scale_ref,
                    zero_ref=applied_rejector.zero_ref,
                    validate=self._validate,
                )
            elif isinstance(applied_rejector, LinearClip):
                restored_flags = kernels._linearclip_restored_flags(
                    self.arr,
                    mask=self.mask,
                    low_scale=applied_rejector.low_scale,
                    low=applied_rejector.low,
                    upp_scale=applied_rejector.upp_scale,
                    upp=applied_rejector.upp,
                    maxiters=applied_rejector.maxiters,
                    nkeep=applied_rejector.nkeep,
                    maxrej=applied_rejector.maxrej,
                    cenfunc=applied_rejector.cenfunc,
                    revert_on_nkeep=applied_rejector.revert_on_nkeep,
                    validate=self._validate,
                )
        n = self.arr.shape[0]
        ts = self._trailing_shape
        if effective_grow is not None:
            mask_rej, output_flags = kernels._grow_rejection_mask(
                mask_rej.reshape(n, *ts),
                output_flags.reshape(ts),
                effective_grow,
                validate=self._validate,
            )
            mask_rej = mask_rej.reshape(self.arr.shape)
            output_flags = output_flags.reshape(self.arr.shape[1:])
        self.last_sample_flags = (
            sample_flags_array(
                self.arr,
                previous_mask=mask_before,
                mask_rej=mask_rej_before_grow,
                grown_mask=mask_rej if effective_grow is not None else None,
                restored_flags=restored_flags,
            ).reshape(n, *ts)
            if diagnostic_level == "full"
            else None
        )
        self.mask = mask_rej if self.mask is None else (self.mask | mask_rej)
        self.last_reject = (
            mask_rej.reshape(n, *ts),
            None if std is None else std.reshape(ts),
            low.reshape(ts),
            upp.reshape(ts),
            nit.reshape(ts),
            output_flags.reshape(ts),
        )
        self.reject_history.append(self.last_reject)
        return self

    # ---- combine call -----------------------------------------------------------

    def combine(
        self,
        method: str = "mean",
        *,
        weight: NDArray | None = None,
        ddof: int = 0,
        zero: PlaneVectorLike = None,
        scale: PlaneVectorLike = None,
        zero_to_0th: bool = True,
        scale_to_0th: bool = True,
        zero_sigclip_kwargs: SigclipStatKwargs = None,
        scale_sigclip_kwargs: SigclipStatKwargs = None,
        thresholds: tuple[float, float] | Sequence[tuple[float, float]] | None = None,
        rejectors: Rejector | Sequence[Rejector] | None = None,
        full: bool = False,
        diagnostics: Diagnostics = None,
    ) -> NDArray:
        """Combine the (possibly masked) stack along axis 0 and return the result.

        Masked pixels are NaN-filled before combining so the kernels skip them.
        Passing combine-call ``thresholds`` or ``rejectors`` with
        ``diagnostics=None`` is the Standard Combiner style; supported
        single-rejector mean/median calls can use fused kernels without
        storing rejection diagnostics. Calling :meth:`threshold`,
        :meth:`zero_scale`, and :meth:`reject` before :meth:`combine` is the
        Chained Combiner style; it preserves masks and rejection diagnostics for
        inspection.
        Since ``Combiner`` works on its normalized workspace, ``lmedian`` here
        follows that workspace dtype. Integer-preserving pure lower median is
        available through ``combine="lmedian"`` or pure
        :func:`imcombiners.ndcombine` calls.

        Parameters
        ----------
        method : str
            Combine method. Supported aliases are `"mean"`, `"average"`,
            `"avg"`, `"median"`, `"med"`, `"lmedian"`, `"lmed"`,
            `"sum"`, `"min"`, `"max"`, `"variance"`, `"var"`.
        weight : ndarray of shape (N,), optional
            Optional weights for ``method="mean"`` / ``"average"`` / ``"avg"``.
        ddof : int, optional
            Delta degrees of freedom for ``method="variance"``. Ignored by
            other combine methods.
        zero : array-like, str, callable, or None, optional
            Per-plane values to subtract before scaling, after any thresholds
            passed to this ``combine`` call and before combine-call rejection.
            Accepts the same forms
            as :meth:`zero_scale`.
        scale : array-like, str, callable, or None, optional
            Per-plane values used as divisors after zero subtraction. Accepts
            the same forms as :meth:`zero_scale`.
        zero_to_0th, scale_to_0th : bool, optional
            Whether per-plane zero or scale values are rebased to the zeroth
            frame before application, matching :meth:`zero_scale` defaults.
        zero_sigclip_kwargs, scale_sigclip_kwargs : dict or None, optional
            Keyword arguments used only when `zero` or `scale` is a
            sigma-clipped statistic alias.
        thresholds : tuple or sequence of tuple, optional
            Inclusive threshold bounds to apply before any rejectors passed to
            this ``combine`` call.
            Passing these to ``combine`` does not mutate the combiner unless a
            diagnostics level is requested. Combine-call zero/scale follows these
            thresholds.
        rejectors : Rejector or sequence of Rejector, optional
            Rejection steps to apply as part of this ``combine`` call.
            With ``diagnostics=None``, a single supported rejector may use a
            fused output-only kernel and no rejection diagnostics are stored.
        full : bool, optional
            Legacy alias for ``diagnostics="simple"``.
        diagnostics : {None, "simple", "full"}, optional
            Diagnostic level for this ``combine`` call. `None` returns only the
            combined image without mutating state. `"simple"` applies
            thresholds/rejectors to this combiner and retains `last_reject`.
            `"full"` also records `sample_flags` for the most recent
            rejection step.

        Returns
        -------
        combined : ndarray, shape (*spatial)
            Combined image, shaped to match the trailing spatial dimensions of
            the array passed to :meth:`__init__`.
        """
        threshold_steps = _normalize_threshold_steps(thresholds)
        rejector_steps = _normalize_rejector_steps(rejectors)
        diagnostic_level = normalize_diagnostics(diagnostics, full)
        if threshold_steps or rejector_steps or zero is not None or scale is not None:
            return self._combine_call_pipeline(
                method,
                weight=weight,
                ddof=ddof,
                zero=zero,
                scale=scale,
                zero_to_0th=zero_to_0th,
                scale_to_0th=scale_to_0th,
                zero_sigclip_kwargs=zero_sigclip_kwargs,
                scale_sigclip_kwargs=scale_sigclip_kwargs,
                thresholds=threshold_steps,
                rejectors=rejector_steps,
                diagnostics=diagnostic_level,
            )

        m = method.lower()

        if self.mask is not None:
            arr_eff = mask_as_nan(self.arr, self.mask)
        else:
            arr_eff = self.arr

        if weight is not None:
            if m not in ("mean", "average", "avg"):
                raise ValueError("weight can only be used with mean combine")
            result = kernels.nanaverage(
                arr_eff,
                validate_weights(weight, arr_eff.shape[0])
                if self._validate
                else np.asarray(weight, dtype=np.float64),
                validate=self._validate,
            )
            return result.reshape(self._trailing_shape)

        canonical = _COMBINE_METHODS.get(m)
        if canonical is None:
            raise ValueError(f"unknown combine method: {method}")
        if canonical is kernels.rd.nanvar:
            return self.variance(ddof=ddof)
        result = kernels._stack(arr_eff, canonical, validate=self._validate)
        return result.reshape(self._trailing_shape)

    def _combine_call_pipeline(
        self,
        method: str,
        *,
        weight: NDArray | None,
        ddof: int,
        zero: PlaneVectorLike,
        scale: PlaneVectorLike,
        zero_to_0th: bool,
        scale_to_0th: bool,
        zero_sigclip_kwargs: SigclipStatKwargs,
        scale_sigclip_kwargs: SigclipStatKwargs,
        thresholds: list[tuple[float, float]],
        rejectors: list[Rejector],
        diagnostics: Diagnostics,
    ) -> NDArray:
        """Run threshold/rejection steps supplied to ``combine``."""
        if diagnostics is not None:
            for low, upp in thresholds:
                self.threshold(low, upp)
            if zero is not None or scale is not None:
                self.zero_scale(
                    zero=zero,
                    scale=scale,
                    zero_to_0th=zero_to_0th,
                    scale_to_0th=scale_to_0th,
                    zero_sigclip_kwargs=zero_sigclip_kwargs,
                    scale_sigclip_kwargs=scale_sigclip_kwargs,
                )
            for rejector in rejectors:
                self.reject(rejector, diagnostics=diagnostics)
            return self.combine(method, weight=weight, ddof=ddof)

        mask = None if self.mask is None else self.mask.copy()
        for bounds in thresholds:
            validate_thresholds(bounds)
            mask_thresh = threshold_mask(self.arr, bounds)
            if mask_thresh is not None:
                mask = mask_thresh if mask is None else (mask | mask_thresh)

        arr = _apply_zero_scale(
            self.arr,
            zero,
            scale,
            mask=mask,
            zero_to_0th=zero_to_0th,
            scale_to_0th=scale_to_0th,
            zero_sigclip_kwargs=zero_sigclip_kwargs,
            scale_sigclip_kwargs=scale_sigclip_kwargs,
            validate=self._validate,
        )

        if len(rejectors) == 1 and weight is None:
            try:
                fused = _fused_reject_combine(
                    arr,
                    rejectors[0],
                    combine=method,
                    mask=mask,
                    validate=self._validate,
                )
            except NotImplementedError:
                fused = None
            if fused is not None:
                return fused.reshape(self._trailing_shape)

        for rejector in rejectors:
            if not isinstance(rejector, Rejector):
                raise TypeError(
                    "rejectors must contain Rejector instances; "
                    f"got {type(rejector).__name__}"
                )
            mask_rej, *_ = rejector.apply(
                arr,
                mask=mask,
                validate=self._validate,
            )
            mask = mask_rej if mask is None else (mask | mask_rej)

        return _combine_array(
            arr,
            method,
            mask=mask,
            weight=weight,
            ddof=ddof,
            validate=self._validate,
        ).reshape(self._trailing_shape)

    def variance(
        self, *, ddof: int = 1, return_mean: bool = False
    ) -> NDArray | tuple[NDArray, NDArray]:
        """Return variance of final valid values along axis 0.

        Valid values are finite inputs not masked by the input mask and not
        rejected by previous :meth:`reject` calls. This is equivalent to::

            arr_eff = np.where(mask | rejected, np.nan, stack)
            var = np.nanvar(arr_eff, axis=0, ddof=ddof)

        Parameters
        ----------
        ddof : int, optional
            Delta degrees of freedom. Pixels with ``nvalid <= ddof`` return
            `NaN`.
            Default is ``1``, i.e., Sample variance. Use ``ddof=0`` for
            population variance.
        return_mean : bool, optional
            If `True`, also return the per-pixel mean computed from the same
            final valid values.

        Returns
        -------
        variance : ndarray, shape (*spatial)
            Per-pixel variance. Use ``np.sqrt(var)`` if an error or
            standard-deviation map is needed.
        mean : ndarray, shape (*spatial)
            Returned only when `return_mean=True`.
        """
        if self.mask is not None:
            arr_eff = mask_as_nan(self.arr, self.mask)
        else:
            arr_eff = self.arr
        if return_mean:
            result = kernels._variance_stack(
                arr_eff,
                ddof=ddof,
                return_mean=True,
                validate=self._validate,
            )
        else:
            result = kernels._variance_stack(
                arr_eff, ddof=ddof, validate=self._validate
            )
        if return_mean:
            var, mean = result
            return var.reshape(self._trailing_shape), mean.reshape(self._trailing_shape)
        return result.reshape(self._trailing_shape)

    # ---- utilities --------------------------------------------------------------

    def copy(self) -> Combiner:
        """Return an independent copy of this pipeline state.

        Returns
        -------
        combiner : Combiner
            New combiner with copied `arr` and `mask`. `last_reject` is shared
            by reference because it is diagnostic data from a completed step.
        """
        new = Combiner.__new__(Combiner)
        new._validate = self._validate
        new._trailing_shape = self._trailing_shape
        new.arr = self.arr.copy()
        new.mask = None if self.mask is None else self.mask.copy()
        new._mask_thresh = (
            None if self._mask_thresh is None else self._mask_thresh.copy()
        )
        new.last_reject = self.last_reject  # tuple of arrays; safe to share refs
        new.last_sample_flags = self.last_sample_flags
        new.reject_history = list(self.reject_history)
        return new

    @property
    def mask_thresh(self) -> NDArray | None:
        """Threshold mask, if :meth:`threshold` has been called."""
        if self._mask_thresh is None:
            return None
        return self._mask_thresh.reshape(self.arr.shape[0], *self._trailing_shape)

    @property
    def mask_rej(self) -> NDArray | None:
        """Rejection mask from the most recent rejection call, if any.

        ``mask_rej.sum(axis=0)`` is the count rejected by that rejection step.
        The count used by combine is the finite, unmasked count after all masks
        are merged: ``np.sum(np.isfinite(self.arr) & ~self.mask, axis=0)``.
        """
        return None if self.last_reject is None else self.last_reject[0]

    @property
    def low(self) -> NDArray | None:
        """Inclusive lower rejection bound from the most recent rejection call."""
        return None if self.last_reject is None else self.last_reject[2]

    @property
    def upp(self) -> NDArray | None:
        """Inclusive upper rejection bound from the most recent rejection call."""
        return None if self.last_reject is None else self.last_reject[3]

    @property
    def nit(self) -> NDArray | None:
        """Iteration-count map from the most recent rejection call, if any."""
        return None if self.last_reject is None else self.last_reject[4]

    @property
    def output_flags(self) -> NDArray | None:
        """Rejection status flag map from the most recent rejection call, if any."""
        return None if self.last_reject is None else self.last_reject[5]

    @property
    def std(self) -> NDArray | None:
        """Per-pixel std diagnostic from the most recent rejection call, if any."""
        return None if self.last_reject is None else self.last_reject[1]

    @property
    def sample_flags(self) -> NDArray | None:
        """Per-sample flags from the most recent full-diagnostics rejection."""
        return self.last_sample_flags

    @property
    def n_rejected(self) -> int:
        """Total number of currently masked elements (input + rejection).

        This is a scalar count of the current combined mask, not the number of
        samples used for combining at each output element.
        """
        return 0 if self.mask is None else int(self.mask.sum())

    def __repr__(self) -> str:
        """Return a compact representation of stack shape and mask count."""
        n = self.arr.shape[0]
        rej = self.n_rejected
        ts = self._trailing_shape
        if len(ts) == 2:
            return (
                f"Combiner(N={n}, H={ts[0]}, W={ts[1]}, "
                f"dtype={self.arr.dtype}, n_masked={rej})"
            )
        return f"Combiner(N={n}, shape={ts}, dtype={self.arr.dtype}, n_masked={rej})"
