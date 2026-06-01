"""Public-boundary validation tests."""

from __future__ import annotations

import numpy as np
import pytest
import reducers as rd
from imcombiners import CcdClip, Combiner, kernels, ndcombine, resolve_zero_scale
from imcombiners._validation import _sigclip_plane_stat


def test_kernel_validate_false_matches_checked_path():
    arr = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
    weights = np.ones(arr.shape[0], dtype=np.float64)

    checked = kernels.nanaverage(arr, weights)
    unchecked = kernels.nanaverage(arr, weights, validate=False)

    np.testing.assert_allclose(unchecked, checked)


def test_stack_helper_accepts_reducer_callable_and_preserves_integer_min_dtype():
    arr = np.arange(24, dtype=np.uint16).reshape(3, 2, 4)

    out = kernels._stack(arr, rd.nanmin)

    assert out.dtype == np.uint16
    np.testing.assert_array_equal(out, arr.min(axis=0))


@pytest.mark.parametrize(
    ("dtype", "out_dtype"),
    [
        (np.uint8, np.float64),
        (np.uint16, np.float64),
        (np.int16, np.float64),
        (np.int32, np.float64),
    ],
)
def test_ndcombine_accepts_common_integer_image_dtypes(dtype, out_dtype):
    arr = np.arange(12, dtype=dtype).reshape(3, 2, 2)

    out = ndcombine(arr, combine="mean")

    assert out.dtype == out_dtype
    np.testing.assert_allclose(out, arr.astype(out_dtype).mean(axis=0))


def test_int32_public_path_preserves_values_above_float32_exact_range():
    arr = np.arange(12, dtype=np.int32).reshape(3, 2, 2) + 2**24 + 1

    out = ndcombine(arr, combine="mean")

    assert out.dtype == np.float64
    np.testing.assert_allclose(out, arr.astype(np.float64).mean(axis=0))


def test_ndcombine_accepts_common_integer_image_dtype():
    arr = np.arange(12, dtype=np.int16).reshape(3, 2, 2)

    out = ndcombine(arr, combine="median")

    assert out.dtype == np.float64
    np.testing.assert_allclose(out, np.median(arr.astype(np.float64), axis=0))


def test_ndcombine_accepts_uint16_image_dtype():
    arr = np.arange(12, dtype=np.uint16).reshape(3, 2, 2)

    out = ndcombine(arr, combine="mean")

    assert out.dtype == np.float64
    np.testing.assert_allclose(out, arr.astype(np.float64).mean(axis=0))


def test_kernel_validate_false_skips_python_stack_validation(monkeypatch):
    arr = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
    weights = np.ones(arr.shape[0], dtype=np.float64)

    def fail_validation(_arr):
        raise AssertionError("validation should be skipped")

    monkeypatch.setattr(kernels, "validate_stack", fail_validation)

    out = kernels.nanaverage(arr, weights, validate=False)

    np.testing.assert_allclose(out, arr.mean(axis=0))


def test_ndcombine_validate_false_matches_checked_path():
    arr = np.arange(12, dtype=np.float32).reshape(3, 2, 2)

    checked = ndcombine(arr, combine="mean")
    unchecked = ndcombine(arr, combine="mean", validate=False)

    np.testing.assert_allclose(unchecked, checked)


def test_combiner_validate_false_matches_checked_path():
    arr = np.arange(12, dtype=np.float32).reshape(3, 2, 2)

    checked = Combiner(arr).combine("mean")
    unchecked = Combiner(arr, validate=False).combine("mean")

    np.testing.assert_allclose(unchecked, checked)


def test_kernel_reject_rejects_wrong_mask_shape():
    arr = np.ones((3, 2, 2), dtype=np.float32)
    mask = np.zeros((3, 2, 1), dtype=bool)

    with pytest.raises(ValueError, match="mask shape"):
        kernels.sigclip(arr, mask=mask)


def test_sigclip_accepts_scalar_sigma():
    arr = np.ones((5, 2, 2), dtype=np.float32)

    scalar = kernels.sigclip(arr, sigma=3.0)
    pair = kernels.sigclip(arr, sigma=(3.0, 3.0))

    for left, right in zip(scalar, pair, strict=True):
        np.testing.assert_array_equal(left, right)


def test_sigclip_rejects_removed_std_center_alias():
    arr = np.ones((5, 2, 2), dtype=np.float32)

    with pytest.raises(TypeError, match="std_center"):
        kernels.sigclip(arr, sigma=3.0, std_center="median")


def test_sigclip_accepts_clip_cen():
    arr = np.ones((5, 2, 2), dtype=np.float32)

    out = kernels.sigclip(arr, sigma=3.0, clip_cen="median")

    assert len(out) == 6


def test_sigclip_plane_stat_accepts_lower_median_center():
    arr = np.array([8.0, 10.0, 10.6, 40.0], dtype=np.float32)

    out = _sigclip_plane_stat(
        arr,
        stat="mean",
        sigma=(10.0, 0.021),
        maxiters=1,
        cenfunc="lmed",
        clip_cen="lmed",
    )

    np.testing.assert_allclose(out, 9.0)


def test_kernel_nanaverage_rejects_wrong_weight_length():
    arr = np.ones((3, 2, 2), dtype=np.float32)
    weights = np.ones(2, dtype=np.float64)

    with pytest.raises(ValueError, match="weights length"):
        kernels.nanaverage(arr, weights)


def test_ndcombine_rejects_wrong_zero_length():
    arr = np.ones((3, 2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="zero length"):
        ndcombine(arr, zero=np.ones(2, dtype=np.float32))


def test_ndcombine_rejects_zero_scale():
    arr = np.ones((3, 2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="scale must be finite and non-zero"):
        ndcombine(arr, scale=np.array([1.0, 0.0, 1.0], dtype=np.float32))


def test_combiner_zero_scale_rejects_zero_scale():
    arr = np.ones((3, 2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="scale must be finite and non-zero"):
        Combiner(arr).zero_scale(scale=np.array([1.0, 0.0, 1.0], dtype=np.float32))


def test_ndcombine_rejects_computed_zero_scale():
    arr = np.zeros((3, 2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="scale must be finite and non-zero"):
        ndcombine(arr, scale="mean")


def test_combiner_zero_scale_accepts_string_statistic():
    arr = np.array(
        [
            [[1.0, 3.0], [5.0, 7.0]],
            [[2.0, 4.0], [6.0, 8.0]],
            [[10.0, 10.0], [10.0, 10.0]],
        ],
        dtype=np.float32,
    )

    out = Combiner(arr).zero_scale(zero="mean", zero_to_0th=False).combine("mean")
    expected = np.mean(arr - np.mean(arr, axis=(1, 2)).reshape(-1, 1, 1), axis=0)

    np.testing.assert_allclose(out, expected)


@pytest.mark.parametrize("alias", ["sigclip_mean", "mean_sc"])
def test_combiner_zero_scale_sigclip_mean_aliases_are_equivalent(alias):
    rng = np.random.default_rng(20250311)
    arr = rng.normal(100.0, 5.0, (8, 4, 4)).astype(np.float32)
    arr[0, 0, 0] = 1_000.0  # outlier that plain mean would inflate

    out = Combiner(arr).zero_scale(zero=alias).combine("mean")
    ref = Combiner(arr).zero_scale(zero="sigclip_mean").combine("mean")

    np.testing.assert_allclose(out, ref, rtol=1e-5)


@pytest.mark.parametrize("alias", ["sigclip_median", "median_sc", "med_sc"])
def test_combiner_zero_scale_sigclip_median_aliases_are_equivalent(alias):
    rng = np.random.default_rng(20250311)
    arr = rng.normal(100.0, 5.0, (8, 4, 4)).astype(np.float32)

    out = Combiner(arr).zero_scale(zero=alias).combine("mean")
    ref = Combiner(arr).zero_scale(zero="sigclip_median").combine("mean")

    np.testing.assert_allclose(out, ref, rtol=1e-5)


def test_combiner_zero_scale_sigclip_robust_to_outlier():
    """Sigma-clipped zero should be closer to the true level than plain mean."""
    rng = np.random.default_rng(20250311)
    # One bright star in every plane drives the plain mean up significantly.
    arr = rng.normal(100.0, 2.0, (5, 16, 16)).astype(np.float32)
    for i in range(arr.shape[0]):
        arr[i, 8, 8] = 5_000.0  # bright star

    zero_plain = np.mean(arr, axis=(1, 2))
    from imcombiners._validation import _sigclip_plane_stat

    zero_sc = np.array(
        [_sigclip_plane_stat(arr[i], stat="mean") for i in range(arr.shape[0])]
    )

    # Plain mean is pulled away from 100; sigclip stays close.
    assert np.all(np.abs(zero_plain - 100.0) > 10.0)
    np.testing.assert_allclose(zero_sc, 100.0, atol=2.0)


def test_combiner_zero_scale_sigclip_kwargs_tune_clipping():
    arr = np.ones((3, 2, 2), dtype=np.float32) * 10
    arr[:, 0, 0] = 100

    loose = (
        Combiner(arr)
        .zero_scale(
            zero="mean_sc",
            zero_sigclip_kwargs={"sigma": 100.0, "maxiters": 2},
            zero_to_0th=False,
        )
        .combine("mean")
    )
    tight = (
        Combiner(arr)
        .zero_scale(
            zero="mean_sc",
            zero_sigclip_kwargs={"sigma": 1.0, "maxiters": 2},
            zero_to_0th=False,
        )
        .combine("mean")
    )

    assert not np.allclose(loose, tight)


def test_ndcombine_scale_sigclip_kwargs_accepted():
    rng = np.random.default_rng(20250311)
    arr = rng.normal(50.0, 3.0, (6, 4, 4)).astype(np.float32)

    out = ndcombine(
        arr,
        scale="med_sc",
        scale_sigclip_kwargs={"sigma": (2.5, 4.0), "maxiters": 3},
        combine="mean",
    )

    assert out.shape == (4, 4)
    assert np.isfinite(out).all()


def test_sigclip_stat_kwargs_reject_unknown_key():
    arr = np.ones((3, 2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="unknown sigma-clipped statistic kwargs"):
        Combiner(arr).zero_scale(zero="mean_sc", zero_sigclip_kwargs={"unknown": 1})


def test_ndcombine_zero_sigclip_string_accepted():
    rng = np.random.default_rng(20250311)
    arr = rng.normal(50.0, 3.0, (6, 4, 4)).astype(np.float32)

    out = ndcombine(arr, zero="sigclip_median", combine="mean")

    assert out.shape == (4, 4)
    assert np.isfinite(out).all()


def test_ndcombine_scale_sigclip_string_accepted():
    rng = np.random.default_rng(20250311)
    arr = rng.normal(50.0, 3.0, (6, 4, 4)).astype(np.float32)

    out = ndcombine(arr, scale="mean_sc", combine="mean")

    assert out.shape == (4, 4)
    assert np.isfinite(out).all()


def test_resolve_zero_scale_rejects_unknown_statistic():
    arr = np.ones((3, 2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="unknown zero statistic"):
        ndcombine(arr, zero="bad_stat_name")


def test_resolve_zero_scale_is_public_helper():
    arr = np.arange(24, dtype=np.float32).reshape(4, 2, 3) + 1.0

    resolved = resolve_zero_scale("scale", "median", arr, nonzero=True)

    assert "resolve_zero_scale" in __import__("imcombiners").__all__
    assert resolved.shape == (4, 1, 1)
    expected = np.median(arr, axis=(1, 2)).reshape(4, 1, 1)
    np.testing.assert_allclose(resolved, expected)


def test_resolve_zero_scale_handles_nd_spatial_axes():
    arr = np.arange(120, dtype=np.float32).reshape(5, 4, 3, 2) + 1.0

    resolved = resolve_zero_scale("zero", "mean", arr)

    assert resolved.shape == (5, 1, 1, 1)
    expected = np.mean(arr, axis=(1, 2, 3)).reshape(5, 1, 1, 1)
    np.testing.assert_allclose(resolved, expected)


@pytest.mark.parametrize(
    ("statistic", "expected"),
    [
        ("sum", np.sum),
        ("min", np.min),
        ("max", np.max),
    ],
)
def test_resolve_zero_scale_accepts_short_reduction_string_names(statistic, expected):
    arr = np.arange(24, dtype=np.float32).reshape(4, 2, 3) + 1.0
    plane_values = expected(arr, axis=(1, 2)).reshape(-1, 1, 1)

    out_zero = ndcombine(arr, zero=statistic, zero_to_0th=False, combine="mean")
    expected_zero = arr - plane_values
    np.testing.assert_allclose(out_zero, np.mean(expected_zero, axis=0), rtol=1e-5)

    c = Combiner(arr).zero_scale(scale=statistic, scale_to_0th=False)
    expected_scale = arr / plane_values
    np.testing.assert_allclose(c.arr, expected_scale, rtol=1e-5)


@pytest.mark.parametrize("statistic", ["summation", "minimum", "maximum"])
def test_resolve_zero_scale_rejects_long_reduction_string_names(statistic):
    arr = np.ones((3, 2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="unknown zero statistic"):
        ndcombine(arr, zero=statistic)

    with pytest.raises(ValueError, match="unknown scale statistic"):
        Combiner(arr).zero_scale(scale=statistic)


def test_ccdclip_spec_rejects_nonpositive_gain():
    arr = np.ones((3, 2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="gain must be finite and positive"):
        Combiner(arr).reject(CcdClip(gain=0.0))
