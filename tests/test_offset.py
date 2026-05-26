"""Tests for offset padding helpers."""

from __future__ import annotations

import numpy as np
import pytest
from imcombiners import ndcombine, place_into_padded


def test_place_into_padded_normalizes_python_order_offsets():
    img0 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    img1 = np.array([[10.0, 11.0, 12.0]], dtype=np.float32)
    offsets = np.array([[2, -1], [-1, 1]], dtype=np.int64)

    out = place_into_padded([img0, img1], offsets)

    expected = np.full((2, 5, 5), np.nan, dtype=np.float32)
    expected[0, 3:5, 0:2] = img0
    expected[1, 0:1, 2:5] = img1
    np.testing.assert_array_equal(out, expected)


def test_place_into_padded_output_combines_with_nan_aware_kernels():
    img0 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    img1 = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)

    stack = place_into_padded([img0, img1], np.array([[0, 0], [1, 1]]))
    out = ndcombine(stack, combine="mean")

    expected = np.array(
        [
            [1.0, 2.0, np.nan],
            [3.0, 7.0, 20.0],
            [np.nan, 30.0, 40.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(out, expected, equal_nan=True)


def test_place_into_padded_accepts_3d_arrays():
    img0 = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    img1 = np.arange(100, 104, dtype=np.float32).reshape(1, 2, 2)
    offsets = np.array([[1, -1, 0], [-1, 1, 2]], dtype=np.int64)

    out = place_into_padded([img0, img1], offsets)

    expected = np.full((2, 4, 4, 4), np.nan, dtype=np.float32)
    expected[0, 2:4, 0:2, 0:2] = img0
    expected[1, 0:1, 2:4, 2:4] = img1
    np.testing.assert_array_equal(out, expected)


def test_place_into_padded_accepts_1d_arrays():
    arr0 = np.array([1.0, 2.0], dtype=np.float32)
    arr1 = np.array([10.0, 11.0, 12.0], dtype=np.float32)

    out = place_into_padded([arr0, arr1], np.array([[2], [-1]], dtype=np.int64))

    expected = np.full((2, 5), np.nan, dtype=np.float32)
    expected[0, 3:5] = arr0
    expected[1, 0:3] = arr1
    np.testing.assert_array_equal(out, expected)


def test_place_into_padded_promotes_integer_inputs_to_float_workspace():
    img0 = np.array([[1, 2]], dtype=np.uint16)
    img1 = np.array([[3, 4]], dtype=np.int16)

    out = place_into_padded([img0, img1], np.array([[0, 0], [1, 0]]))

    assert out.dtype == np.float32
    assert np.isnan(out[0, 1, 0])
    np.testing.assert_array_equal(out[0, 0], np.array([1.0, 2.0], dtype=np.float32))
    np.testing.assert_array_equal(out[1, 1], np.array([3.0, 4.0], dtype=np.float32))


def test_place_into_padded_preserves_float64_workspace():
    img0 = np.array([[1.0]], dtype=np.float32)
    img1 = np.array([[2.0]], dtype=np.float64)

    out = place_into_padded([img0, img1], np.array([[0, 0], [0, 1]]))

    assert out.dtype == np.float64
    np.testing.assert_array_equal(out[0, 0, :1], np.array([1.0], dtype=np.float64))
    np.testing.assert_array_equal(out[1, 0, 1:], np.array([2.0], dtype=np.float64))


def test_place_into_padded_rejects_offset_count_mismatch():
    img = np.ones((2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="offsets shape"):
        place_into_padded([img], np.array([[0, 0], [1, 1]]))


def test_place_into_padded_rejects_offset_normalization_overflow():
    images = [np.ones((1, 1), dtype=np.float32), np.ones((1, 1), dtype=np.float32)]
    offsets = np.array(
        [
            [np.iinfo(np.int64).max, 0],
            [np.iinfo(np.int64).min, 0],
        ],
        dtype=np.int64,
    )

    with pytest.raises(ValueError, match="normalized y offset overflow"):
        place_into_padded(images, offsets)


def test_place_into_padded_rejects_mixed_ndim_images():
    img0 = np.ones((1, 2, 2), dtype=np.float32)
    img1 = np.ones((2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="same number of dimensions"):
        place_into_padded([img0, img1], np.array([[0, 0, 0], [0, 0, 0]]))


def test_place_into_padded_rejects_wrong_offset_rank_for_nd_arrays():
    img = np.ones((1, 2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="offsets shape"):
        place_into_padded([img], np.array([[0, 0]], dtype=np.int64))
