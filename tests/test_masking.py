"""Tests for internal mask-to-NaN materialization helpers."""

from __future__ import annotations

import numpy as np
from imcombiners import Combiner, ndcombine
from imcombiners._validation import mask_as_nan


def test_mask_as_nan_materializes_copy_without_mutating_input():
    arr = np.arange(6, dtype=np.float32).reshape(3, 1, 2)
    mask = np.array(
        [
            [[False, True]],
            [[False, False]],
            [[True, False]],
        ],
        dtype=bool,
    )

    out = mask_as_nan(arr, mask)

    assert out is not arr
    np.testing.assert_array_equal(arr, np.arange(6, dtype=np.float32).reshape(3, 1, 2))
    assert np.isnan(out[0, 0, 1])
    assert np.isnan(out[2, 0, 0])
    np.testing.assert_array_equal(out[~mask], arr[~mask])


def test_mask_as_nan_returns_copy_for_all_false_mask():
    arr = np.arange(6, dtype=np.float32).reshape(3, 1, 2)
    mask = np.zeros_like(arr, dtype=bool)

    out = mask_as_nan(arr, mask)

    assert out is not arr
    np.testing.assert_array_equal(out, arr)
    out[0, 0, 0] = 99
    assert arr[0, 0, 0] == 0


def test_ndcombine_masked_path_still_ignores_unmasked_nonfinite_values():
    arr = np.array(
        [
            [[1.0, np.nan], [np.inf, 4.0]],
            [[3.0, 5.0], [7.0, -np.inf]],
            [[100.0, 9.0], [11.0, 12.0]],
        ],
        dtype=np.float32,
    )
    mask = np.zeros_like(arr, dtype=bool)
    mask[2, 0, 0] = True

    out = ndcombine(arr, mask=mask, combine="mean")

    expected = np.array([[2.0, 7.0], [9.0, 8.0]], dtype=np.float32)
    np.testing.assert_allclose(out, expected)


def test_combiner_masked_combine_does_not_mutate_workspace():
    arr = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
    mask = np.zeros_like(arr, dtype=bool)
    mask[0, 0, 0] = True
    combiner = Combiner(arr, mask=mask)

    combiner.combine("mean")

    np.testing.assert_array_equal(combiner.arr, arr)
