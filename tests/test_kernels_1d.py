"""1-D kernel utility API tests."""

from __future__ import annotations

import imcombiners.kernels as imck
import numpy as np
import pytest


def _stack(values: np.ndarray) -> np.ndarray:
    return values.reshape(-1, 1, 1)


def _stack_mask(mask: np.ndarray | None) -> np.ndarray | None:
    if mask is None:
        return None
    return mask.reshape(-1, 1, 1)


@pytest.mark.parametrize(
    ("name", "reference"),
    [
        ("mean_1d", np.nanmean),
        ("median_1d", np.nanmedian),
        ("sum_1d", np.nansum),
        ("min_1d", np.nanmin),
        ("max_1d", np.nanmax),
        ("var_1d", np.nanvar),
    ],
)
def test_1d_combine_functions_match_numpy(name, reference):
    values = np.array([1.0, 2.0, np.nan, 5.0], dtype=np.float32)

    out = getattr(imck, name)(values)

    np.testing.assert_allclose(out, reference(values), rtol=1e-6)
    assert np.isscalar(out)


def test_lmedian_1d_returns_lower_middle_and_preserves_integer_dtype():
    values = np.array([4, 1, 3, 2], dtype=np.uint16)

    out = imck.lmedian_1d(values)

    assert out == np.uint16(2)
    assert isinstance(out, np.uint16)


def test_wvg_1d_matches_stack_kernel():
    values = np.array([1.0, np.nan, 5.0, 7.0], dtype=np.float32)
    weights = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

    out = imck.wvg_1d(values, weights)
    expected = imck.weighted_average(_stack(values), weights)[0, 0]

    np.testing.assert_allclose(out, expected, rtol=1e-6)
    assert np.isscalar(out)


def test_1d_combine_functions_reject_non_1d_inputs():
    values = np.ones((2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="1-D"):
        imck.median_1d(values)


def test_1d_validate_false_skips_python_value_preparation(monkeypatch):
    values = np.array([1.0, 2.0, np.nan, 4.0], dtype=np.float32)

    def fail_prepare(*args, **kwargs):
        raise AssertionError("validate=False should bypass Python preparation")

    monkeypatch.setattr(imck, "_prepare_1d_values", fail_prepare)

    np.testing.assert_allclose(imck.mean_1d(values, validate=False), 7.0 / 3.0)


def _assert_rejection_1d_matches_stack(name: str, values: np.ndarray, **kwargs):
    mask = kwargs.pop("mask", None)
    out_1d = getattr(imck, f"{name}_1d")(values, mask=mask, **kwargs)
    out_stack = getattr(imck, name)(
        _stack(values), mask=_stack_mask(mask), grow=None, **kwargs
    )

    np.testing.assert_array_equal(out_1d[0], out_stack[0].reshape(-1))
    for got, expected in zip(out_1d[1:], out_stack[1:], strict=True):
        if expected is None:
            assert got is None
        else:
            np.testing.assert_allclose(got, expected[0, 0], rtol=1e-6)
            assert np.isscalar(got)


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("sigclip", {"sigma": 1.0, "maxiters": 3, "cenfunc": "median"}),
        ("ccdclip", {"sigma": 1.0, "rdnoise": 2.0, "gain": 1.5}),
        ("linearclip", {"low_scale": 1.0, "low": 1.0, "upp_scale": 1.0, "upp": 1.0}),
        ("minmax", {"n_min": 1, "n_max": 1}),
        ("pclip", {"frac": -0.5}),
    ],
)
def test_1d_rejection_functions_match_stack_kernels(name, kwargs):
    values = np.array([1.0, 10.0, 11.0, 12.0, 30.0], dtype=np.float32)
    mask = np.array([False, False, True, False, False])

    _assert_rejection_1d_matches_stack(name, values, mask=mask, **kwargs)


@pytest.mark.parametrize(
    ("name", "stack_name", "kwargs"),
    [
        ("sigclip_mask_1d", "sigclip_mask", {"sigma": 1.0}),
        ("ccdclip_mask_1d", "ccdclip_mask", {"sigma": 1.0, "rdnoise": 2.0}),
        ("minmax_mask_1d", "minmax_mask", {"n_min": 1, "n_max": 1}),
        ("pclip_mask_1d", "pclip_mask", {"frac": -0.5}),
    ],
)
def test_1d_mask_functions_match_stack_kernels(name, stack_name, kwargs):
    values = np.array([1.0, 10.0, 11.0, 12.0, 30.0], dtype=np.float32)
    mask = np.array([False, False, True, False, False])

    out = getattr(imck, name)(values, mask=mask, **kwargs)
    expected = getattr(imck, stack_name)(
        _stack(values), mask=_stack_mask(mask), **kwargs
    )

    np.testing.assert_array_equal(out, expected.reshape(-1))


@pytest.mark.parametrize(
    ("name", "stack_name", "kwargs"),
    [
        ("sigclip_combine_1d", "sigclip_combine", {"sigma": 1.0, "combine": "median"}),
        (
            "ccdclip_combine_1d",
            "ccdclip_combine",
            {"sigma": 1.0, "rdnoise": 2.0, "combine": "mean"},
        ),
        (
            "minmax_combine_1d",
            "minmax_combine",
            {"n_min": 1, "n_max": 1, "combine": "median"},
        ),
        ("pclip_combine_1d", "pclip_combine", {"frac": -0.5, "combine": "mean"}),
    ],
)
def test_1d_reject_combine_functions_match_stack_kernels(name, stack_name, kwargs):
    values = np.array([1.0, 10.0, 11.0, 12.0, 30.0], dtype=np.float32)
    mask = np.array([False, False, True, False, False])

    out = getattr(imck, name)(values, mask=mask, **kwargs)
    expected = getattr(imck, stack_name)(
        _stack(values), mask=_stack_mask(mask), **kwargs
    )

    np.testing.assert_allclose(out, expected[0, 0], rtol=1e-6)
    assert np.isscalar(out)
