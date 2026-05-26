"""Shared public-boundary validation helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._typing import PlaneVectorLike

if TYPE_CHECKING:
    from numpy.typing import NDArray

_FLOAT_DTYPES = (np.float32, np.float64)
_PROMOTE_TO_FLOAT32_DTYPES = (np.uint8, np.uint16, np.int16)
_PROMOTE_TO_FLOAT64_DTYPES = (np.int32,)
_INTEGER_DTYPES = (np.uint8, np.uint16, np.int16, np.int32)
_PLANE_STATISTICS = {
    "mean": np.nanmean,
    "average": np.nanmean,
    "avg": np.nanmean,
    "median": np.nanmedian,
    "med": np.nanmedian,
    "sum": np.nansum,
    "min": np.nanmin,
    "max": np.nanmax,
}

# Sigma-clipped variants: sigma=3, maxiters=5, cenfunc=median by default.
# Maps string alias → which statistic ("mean" or "median") to return from survivors.
_SIGCLIP_STATS: dict[str, str] = {
    "sigclip_mean": "mean",
    "mean_sc": "mean",
    "sigclip_median": "median",
    "median_sc": "median",
    "med_sc": "median",
}


def _zero_scale_broadcast_shape(size: int, ndim: int) -> tuple[int, ...]:
    """Return the broadcast shape for one scalar or one value per plane."""
    return (size,) + (1,) * max(ndim - 1, 0)


def _sigma_pair(sigma: float | tuple[float, float]) -> tuple[float, float]:
    """Return lower/upper sigma thresholds."""
    if isinstance(sigma, (int, float)):
        return float(sigma), float(sigma)
    return float(sigma[0]), float(sigma[1])


def _center(values: NDArray, method: str) -> float:
    """Return a center value for sigma-clipped plane statistics."""
    key = method.lower()
    if key in ("median", "med"):
        return float(np.median(values))
    if key in ("lmedian", "lmed", "lower_median", "lower-median", "lower median"):
        sorted_values = np.sort(values)
        return float(sorted_values[(sorted_values.size - 1) // 2])
    if key in ("mean", "average", "avg"):
        return float(np.mean(values))
    raise ValueError(f"unknown sigma-clipped center: {method!r}")


def _sigclip_plane_stat(
    plane: NDArray,
    *,
    stat: str,
    sigma: float | tuple[float, float] = 3.0,
    maxiters: int = 5,
    cenfunc: str = "median",
    clip_cen: str | None = None,
    ddof: int = 0,
) -> float:
    """Return sigma-clipped mean or median of a single image plane.

    Clips iteratively using configurable sigma-clipping parameters. Returns
    `NaN` if all values are clipped.

    Parameters
    ----------
    plane : ndarray
        Single image plane (any shape); values are flattened internally.
    stat : {"mean", "median"}
        Statistic to compute from the surviving (unclipped) values.
    sigma : float or tuple of float, optional
        Lower and upper clipping thresholds. A scalar applies to both tails.
    maxiters : int, optional
        Maximum number of clipping iterations.
    cenfunc : {"median", "lmedian", "mean"}, optional
        Center estimator used each iteration.
    clip_cen : {"median", "lmedian", "mean"} or None, optional
        Center used for the spread calculation. `None` uses `cenfunc`.
    ddof : int, optional
        Delta degrees of freedom for the standard deviation.
    """
    sigma_lower, sigma_upper = _sigma_pair(sigma)
    maxiters = int(maxiters)
    ddof = int(ddof)
    if maxiters < 0:
        raise ValueError("maxiters must be >= 0")
    if ddof < 0:
        raise ValueError("ddof must be >= 0")
    data = np.asarray(plane, dtype=np.float64).ravel()
    mask = ~np.isfinite(data)
    spread_center = cenfunc if clip_cen is None else clip_cen
    for _ in range(maxiters):
        good = data[~mask]
        if good.size == 0:
            break
        cen = _center(good, cenfunc)
        std_cen = _center(good, spread_center)
        if good.size <= ddof:
            break
        std = np.sqrt(np.sum((good - std_cen) ** 2) / (good.size - ddof))
        if std == 0.0:
            break
        new_mask = (data < cen - sigma_lower * std) | (data > cen + sigma_upper * std)
        new_mask |= ~np.isfinite(data)
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask
    survivors = data[~mask]
    if survivors.size == 0:
        return float("nan")
    return float(np.mean(survivors) if stat == "mean" else np.median(survivors))


def validate_stack(arr: NDArray) -> NDArray:
    """Validate and normalize a stack for the floating-workspace kernels.

    Parameters
    ----------
    arr : ndarray
        Candidate image stack. Must have at least 2 dimensions with shape
        ``(N, *spatial)`` and ``N > 0``. Inputs with more than 3 dimensions
        are reshaped to ``(N, prod(spatial), 1)`` before being passed to the
        Rust kernels; callers should reshape the output back to the original
        trailing shape using ``result.reshape(arr.shape[1:])``.

    Returns
    -------
    arr : ndarray
        C-contiguous 3-D kernel workspace. Native 3-D inputs keep their
        original trailing axes; inputs with more than 3 dimensions are
        represented as ``(N, prod(spatial), 1)``. `uint8`, `uint16`, and
        `int16` are promoted to `float32`; `int32` is promoted to `float64`;
        `float32` and `float64` are preserved.

    Raises
    ------
    ValueError
        If `arr` has fewer than 2 dimensions or has no image planes.
    TypeError
        If `arr` has an unsupported dtype.
    """
    if arr.ndim < 2:
        raise ValueError(
            f"arr must have at least 2 dimensions (N, *spatial); got shape {arr.shape}"
        )
    if arr.shape[0] == 0:
        raise ValueError("arr must contain at least one image along axis 0")
    if arr.ndim != 3:
        arr = arr.reshape(arr.shape[0], -1, 1)
    if arr.dtype in _PROMOTE_TO_FLOAT32_DTYPES:
        return np.ascontiguousarray(arr, dtype=np.float32)
    if arr.dtype in _PROMOTE_TO_FLOAT64_DTYPES:
        return np.ascontiguousarray(arr, dtype=np.float64)
    if arr.dtype not in _FLOAT_DTYPES:
        raise TypeError(
            "arr must be uint8, uint16, int16, int32, float32, "
            f"or float64; got {arr.dtype}"
        )
    return np.ascontiguousarray(arr)


def validate_lmedian_stack(arr: NDArray) -> NDArray:
    """Validate a pure lower-median stack while preserving integer dtypes.

    Parameters
    ----------
    arr : ndarray
        Candidate image stack. Must have at least 2 dimensions with shape
        ``(N, *spatial)`` and ``N > 0``. Inputs with more than 3 dimensions
        are reshaped to ``(N, prod(spatial), 1)``; callers should reshape the
        output back using ``result.reshape(arr.shape[1:])``.

    Returns
    -------
    arr : ndarray
        C-contiguous 3-D stack with original dtype preserved for `uint8`,
        `uint16`, `int16`, `int32`, `float32`, and `float64`.
    """
    if arr.ndim < 2:
        raise ValueError(
            f"arr must have at least 2 dimensions (N, *spatial); got shape {arr.shape}"
        )
    if arr.shape[0] == 0:
        raise ValueError("arr must contain at least one image along axis 0")
    if arr.ndim != 3:
        arr = arr.reshape(arr.shape[0], -1, 1)
    if arr.dtype not in (*_INTEGER_DTYPES, *_FLOAT_DTYPES):
        raise TypeError(
            "arr must be uint8, uint16, int16, int32, float32, "
            f"or float64; got {arr.dtype}"
        )
    return np.ascontiguousarray(arr)


def validate_mask(mask: NDArray | None, shape: tuple[int, ...]) -> NDArray | None:
    """Validate and normalize a boolean mask.

    Parameters
    ----------
    mask : ndarray or None
        Candidate mask. `True` means masked.
    shape : tuple of int
        Required mask shape.

    Returns
    -------
    mask : ndarray of bool or None
        C-contiguous boolean mask, or `None` if no mask was supplied.
    """
    if mask is None:
        return None
    mask = np.ascontiguousarray(mask.astype(bool, copy=False))
    if mask.shape != shape:
        raise ValueError(f"mask shape {mask.shape} does not match arr shape {shape}")
    return mask


def validate_thresholds(
    thresholds: tuple[float, float] | list[float] | None,
) -> tuple[float, float] | None:
    """Validate lower/upper threshold bounds."""
    if thresholds is None:
        return None
    if len(thresholds) != 2:
        raise ValueError("thresholds must be a (low, upp) pair")
    low = float(thresholds[0])
    upp = float(thresholds[1])
    if np.isnan(low) or np.isnan(upp):
        raise ValueError("thresholds cannot contain NaN")
    if low > upp:
        raise ValueError("threshold lower bound must be <= upper bound")
    return low, upp


def threshold_mask(
    arr: NDArray,
    thresholds: tuple[float, float] | list[float] | None,
) -> NDArray | None:
    """Return `True` where values fall outside inclusive thresholds."""
    bounds = validate_thresholds(thresholds)
    if bounds is None:
        return None
    low, upp = bounds
    return np.ascontiguousarray((arr < low) | (arr > upp), dtype=bool)


def mask_as_nan(arr: NDArray, mask: NDArray) -> NDArray:
    """Return a copy of `arr` with `mask` positions filled by `NaN`.

    Parameters
    ----------
    arr : ndarray
        Floating image stack.
    mask : ndarray of bool
        Mask with the same shape as `arr`; `True` values become `NaN`.

    Returns
    -------
    masked : ndarray
        Copy of `arr` with masked values replaced by `NaN`.
    """
    if not np.any(mask):
        return arr.copy()
    out = arr.copy()
    out[mask] = np.nan
    return out


def validate_plane_vector(
    name: str,
    values: NDArray | None,
    n: int,
    dtype: np.dtype,
    *,
    nonzero: bool = False,
) -> NDArray | None:
    """Validate scalar-or-per-plane zero/scale style inputs.

    Parameters
    ----------
    name : str
        Name used in error messages.
    values : ndarray or None
        Candidate scalar or length-`n` vector.
    n : int
        Number of image planes.
    dtype : dtype
        Output dtype.
    nonzero : bool, optional
        If `True`, reject zero values.

    Returns
    -------
    values : ndarray or None
        Shape `(1, 1, 1)` or `(N, 1, 1)` array ready for broadcasting.
    """
    if values is None:
        return None

    out = np.asarray(values, dtype=dtype).reshape(-1)
    if out.size not in (1, n):
        raise ValueError(f"{name} length must be 1 or match stack size N={n}")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be finite")
    if nonzero and np.any(out == 0):
        raise ValueError(f"{name} must be finite and non-zero")
    return out.reshape(-1, 1, 1)


def resolve_zero_scale(
    name: str,
    values: PlaneVectorLike,
    arr: NDArray,
    *,
    nonzero: bool = False,
    to_0th: bool = False,
    sigclip_kwargs: dict[str, object] | None = None,
    validate: bool = True,
) -> NDArray | None:
    """Resolve a numeric, string, or callable zero/scale input.

    This helper resolves one `zero` or `scale` argument. It is useful for
    advanced callers that want to inspect or reuse
    the per-plane vector that `Combiner.zero_scale`, `Combiner.combine`, and
    `ndcombine` would apply internally.

    Parameters
    ----------
    name : str
        Name used in error messages.
    values : array-like, str, callable, or None
        Scalar/vector values, a named per-plane statistic, or a callable
        evaluated once for each image plane.

        Plain statistics (computed on all finite pixels):
        `"mean"`, `"average"`, `"avg"`, `"median"`, `"med"`,
        `"sum"`, `"min"`, `"max"`.

        Sigma-clipped statistics (``sigma=3``, ``maxiters=5``,
        ``cenfunc="median"`` by default):
        `"sigclip_mean"` / `"mean_sc"` and `"sigclip_median"` /
        `"median_sc"` / `"med_sc"`. These clip per-plane before computing
        the statistic, making the normalization robust to bright stars or
        cosmic rays. `sigclip_kwargs` can tune their clipping parameters.
    arr : ndarray, shape (N, *spatial)
        Workspace used for statistics and dtype normalization.
    nonzero : bool, optional
        If `True`, reject zero values after resolving.
    to_0th : bool, optional
        If `True`, normalize values relative to the first plane: subtract the
        first value for zero offsets or divide by the first value for scales.
    sigclip_kwargs : dict or None, optional
        Optional keyword arguments for sigma-clipped statistics. Supported
        keys are `sigma`, `maxiters`, `cenfunc`, `clip_cen`, and `ddof`.
    validate : bool, optional
        If `True`, validate length, finiteness, and nonzero constraints. If
        `False`, callers are responsible for providing broadcast-compatible,
        finite values.

    Returns
    -------
    values : ndarray or None
        Broadcast-ready array with shape `(1, 1, ...)` for scalar input or
        `(N, 1, ...)` for one value per input plane.
    """
    if values is None:
        return None

    if isinstance(values, str):
        key = values.lower()
        if key in _SIGCLIP_STATS:
            stat = _SIGCLIP_STATS[key]
            kwargs = {} if sigclip_kwargs is None else dict(sigclip_kwargs)
            allowed = {"sigma", "maxiters", "cenfunc", "clip_cen", "ddof"}
            unknown = set(kwargs) - allowed
            if unknown:
                raise ValueError(
                    "unknown sigma-clipped statistic kwargs: "
                    + ", ".join(sorted(unknown))
                )
            out = np.array(
                [_sigclip_plane_stat(plane, stat=stat, **kwargs) for plane in arr]
            )
        elif key in _PLANE_STATISTICS:
            out = _PLANE_STATISTICS[key](arr, axis=tuple(range(1, arr.ndim)))
        else:
            allowed = ", ".join(sorted(_PLANE_STATISTICS) + sorted(_SIGCLIP_STATS))
            raise ValueError(
                f"unknown {name} statistic: {values!r}; expected one of: {allowed}"
            )
    elif callable(values):
        out = [values(plane) for plane in arr]
    elif validate:
        out = validate_plane_vector(
            name, values, arr.shape[0], arr.dtype, nonzero=nonzero
        ).reshape(-1)
    else:
        out = np.asarray(values, dtype=arr.dtype).reshape(-1)

    is_scalar_value = np.asarray(out).reshape(-1).size == 1
    if validate:
        out = validate_plane_vector(
            name, out, arr.shape[0], arr.dtype, nonzero=nonzero
        ).reshape(-1)
    else:
        out = out.reshape(-1)
    if to_0th and not is_scalar_value:
        out = out.copy()
        if nonzero:
            out /= out[0]
        else:
            out -= out[0]
    return out.reshape(_zero_scale_broadcast_shape(out.size, arr.ndim))


def validate_weights(weights: NDArray, n: int) -> NDArray:
    """Validate weighted-mean weights.

    Parameters
    ----------
    weights : ndarray
        Candidate weight vector.
    n : int
        Required number of weights.

    Returns
    -------
    weights : ndarray, shape (N,)
        C-contiguous `float64` weight vector.
    """
    out = np.ascontiguousarray(weights, dtype=np.float64).reshape(-1)
    if out.size != n:
        raise ValueError(f"weights length must match stack size N={n}")
    if not np.all(np.isfinite(out)):
        raise ValueError("weights must be finite")
    return out


def validate_positive_scalar(name: str, value: float) -> float:
    """Validate a finite positive scalar.

    Parameters
    ----------
    name : str
        Name used in error messages.
    value : float
        Candidate scalar.

    Returns
    -------
    value : float
        Finite positive value.
    """
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value
