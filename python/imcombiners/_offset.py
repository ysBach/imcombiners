"""Offset/padding helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
from astro_ndslice.offset import offseted_shape, offsets2slice

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

_FLOAT_DTYPES = (np.float32, np.float64)
_PROMOTE_TO_FLOAT32_DTYPES = (np.uint8, np.uint16, np.int16)
_PROMOTE_TO_FLOAT64_DTYPES = (np.int32,)


def _normalize_offset_image(image: ArrayLike) -> NDArray:
    """Validate one N-D array for offset padding."""
    arr = np.asarray(image)
    if arr.ndim < 1:
        raise ValueError(
            f"images must have at least 1 dimension; got shape {arr.shape}"
        )
    if 0 in arr.shape:
        raise ValueError(f"images must have non-zero axes; got shape {arr.shape}")
    if arr.dtype in _PROMOTE_TO_FLOAT32_DTYPES:
        return np.ascontiguousarray(arr, dtype=np.float32)
    if arr.dtype in _PROMOTE_TO_FLOAT64_DTYPES:
        return np.ascontiguousarray(arr, dtype=np.float64)
    if arr.dtype not in _FLOAT_DTYPES:
        raise TypeError(
            "images must be uint8, uint16, int16, int32, float32, "
            f"or float64; got {arr.dtype}"
        )
    return np.ascontiguousarray(arr)


def _normalize_offset_images(
    images: Iterable[ArrayLike],
) -> list[NDArray]:
    """Validate offset images and promote them to one floating dtype."""
    arrays = [_normalize_offset_image(image) for image in images]
    if not arrays:
        raise ValueError("images must contain at least one array")
    ndim = arrays[0].ndim
    for arr in arrays:
        if arr.ndim != ndim:
            raise ValueError("images must all have the same number of dimensions")
    dtype = np.result_type(*(arr.dtype for arr in arrays))
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        dtype = np.dtype(np.float64)
    return [np.ascontiguousarray(arr, dtype=dtype) for arr in arrays]


def place_into_padded(
    images: Iterable[ArrayLike],
    offsets: ArrayLike,
    *,
    fill: float = np.nan,
    validate: bool = True,
) -> NDArray:
    """Place N-D arrays into a padded stack using integer offsets.

    Offsets are in Python/NumPy axis order. For 2-D images, `(dy, dx)` means
    `(row offset, column offset)`: positive `dy` places an image at a larger
    row index, and positive `dx` places it at a larger column index. Raw
    offsets are normalized by subtracting the per-axis minimum offset, matching
    `astro-ndslice` outer-shape convention for offsets already in Python axis
    order.

    Parameters
    ----------
    images : sequence of array-like, each shape (*spatial)
        Input arrays. Arrays may have different shapes, but must all have the
        same number of dimensions. Accepted dtypes are
        `uint8`, `uint16`, `int16`, `int32`, `float32`, and `float64`; integer
        images are promoted to a floating workspace. Other dtypes, including
        `int64` and `float128`, are not silently cast. Cast unsupported arrays
        explicitly before padding.
    offsets : array-like, shape (N, ndim)
        Integer offsets in NumPy axis order. `N` must match the number of
        arrays, and `ndim` must match each input array.
    fill : float, optional
        Value assigned to uncovered pixels in the padded stack. The default is
        `NaN`, which makes the combine kernels ignore uncovered regions.
    validate : bool, optional
        If `True`, validate image dimensionality, dtype, offset shape, and
        offset dtype. If `False`, callers must provide same-rank contiguous
        arrays with compatible dtypes and an integer offset array of shape
        `(N, ndim)`.

    Returns
    -------
    padded : ndarray, shape (N, *outer_shape)
        Stack containing all arrays placed into the common padded frame.
    """
    if validate:
        images = _normalize_offset_images(images)
        offsets = np.asarray(offsets)
        if not np.issubdtype(offsets.dtype, np.integer):
            raise TypeError(f"offsets must have integer dtype; got {offsets.dtype}")
        offsets = np.ascontiguousarray(offsets, dtype=np.int64)
    else:
        images = list(images)
        offsets = np.ascontiguousarray(offsets, dtype=np.int64)
    ndim = images[0].ndim
    if offsets.shape != (len(images), ndim):
        raise ValueError(
            f"offsets shape must be ({len(images)}, {ndim}); got {offsets.shape}"
        )

    shapes = np.array([image.shape for image in images], dtype=np.int64)
    int64_max = np.iinfo(np.int64).max
    intp_max = np.iinfo(np.intp).max
    min_offsets = offsets.min(axis=0)
    for image_shape, offset in zip(shapes, offsets, strict=True):
        for axis, (axis_len, raw_offset, min_offset) in enumerate(
            zip(image_shape, offset, min_offsets, strict=True)
        ):
            label = "y" if axis == 0 else "x" if axis == 1 else f"axis {axis}"
            start = int(raw_offset) - int(min_offset)
            if start > int64_max:
                raise ValueError(f"normalized {label} offset overflow")
            if start > intp_max:
                raise ValueError(f"normalized {label} offset does not fit usize")
            if start + int(axis_len) > intp_max:
                raise ValueError(f"padded output {label} overflow")

    _, outer_shape = offseted_shape(
        shapes,
        offsets,
        method="outer",
        offset_order_xyz=False,
        intify_offsets=True,
        pythonize_offsets=True,
    )
    slices = offsets2slice(
        shapes,
        offsets,
        method="outer",
        shape_order_xyz=False,
        offset_order_xyz=False,
        outer_for_stack=False,
        fits_convention=False,
    )
    padded = np.full((len(images), *outer_shape), fill, dtype=images[0].dtype)
    for image_index, (image, image_slices) in enumerate(
        zip(images, slices, strict=True)
    ):
        padded[(image_index, *image_slices)] = image
    return padded
