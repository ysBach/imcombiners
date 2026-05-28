"""Combine-kernel unit tests."""

from __future__ import annotations

import imcombiners as imc
import numpy as np
import pytest
from imcombiners import _core, ndcombine


@pytest.fixture
def stack():
    rng = np.random.default_rng(20250311)
    return rng.normal(100.0, 5.0, (10, 8, 8)).astype(np.float32)


def test_mean_no_nan(stack):
    out = _core.combine(stack, "mean")
    np.testing.assert_allclose(out, stack.mean(axis=0), rtol=1e-5)


def test_sum_no_nan(stack):
    out = _core.combine(stack, "sum")
    np.testing.assert_allclose(out, stack.sum(axis=0), rtol=1e-5)


def test_median_no_nan(stack):
    out = _core.combine(stack, "median")
    np.testing.assert_allclose(out, np.median(stack, axis=0), rtol=1e-5)


def test_min_max(stack):
    np.testing.assert_allclose(_core.combine(stack, "min"), stack.min(axis=0))
    np.testing.assert_allclose(_core.combine(stack, "max"), stack.max(axis=0))


@pytest.mark.parametrize("method", ["summation", "minimum", "maximum"])
def test_legacy_long_string_reductions_are_rejected(stack, method):
    with pytest.raises(ValueError, match="unknown combine method"):
        _core.combine(stack, method)

    with pytest.raises(ValueError, match="unknown combine method"):
        ndcombine(stack, combine=method)

    with pytest.raises(ValueError, match="unknown combine method"):
        imc.Combiner(stack).combine(method)


def test_variance_no_nan(stack):
    out = _core.variance(stack, ddof=1)
    np.testing.assert_allclose(out, np.var(stack, axis=0, ddof=1), rtol=1e-5)


def test_ndcombine_variance_matches_nanvar():
    rng = np.random.default_rng(20250311)
    arr = rng.normal(0, 1, (8, 3, 4)).astype(np.float32)
    arr[0, 1, 2] = np.nan

    out = ndcombine(arr, combine="variance", ddof=1)

    np.testing.assert_allclose(out, np.nanvar(arr, axis=0, ddof=1), rtol=1e-5)


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("sum", np.sum),
        ("min", np.min),
        ("max", np.max),
        ("var", np.var),
    ],
)
def test_ndcombine_short_aliases(alias, expected):
    arr = np.arange(24, dtype=np.float32).reshape(4, 2, 3)

    out = ndcombine(arr, combine=alias, ddof=0)

    np.testing.assert_allclose(out, expected(arr, axis=0), rtol=1e-5)


def test_top_level_variance_matches_nanvar():
    rng = np.random.default_rng(20250311)
    arr = rng.normal(0, 1, (8, 3, 4)).astype(np.float32)

    out = imc.variance(arr, ddof=0)

    np.testing.assert_allclose(out, np.nanvar(arr, axis=0, ddof=0), rtol=1e-5)


def test_kernel_variance_can_return_mean_from_same_pass():
    rng = np.random.default_rng(20250311)
    arr = rng.normal(0, 1, (8, 3, 4)).astype(np.float32)
    arr[0, 1, 2] = np.nan

    var, mean = imc.variance(arr, ddof=1, return_mean=True)

    np.testing.assert_allclose(var, np.nanvar(arr, axis=0, ddof=1), rtol=1e-5)
    np.testing.assert_allclose(mean, np.nanmean(arr, axis=0), rtol=1e-5)


def test_kernel_variance_return_mean_keeps_mean_when_variance_is_undefined():
    arr = np.array([1.0, np.nan], dtype=np.float32).reshape(2, 1, 1)

    var, mean = imc.variance(arr, ddof=1, return_mean=True)

    assert np.isnan(var[0, 0])
    np.testing.assert_allclose(mean[0, 0], 1.0)


def test_ndcombine_variance_after_rejection_matches_final_mask():
    arr = np.array([1.0, 2.0, 100.0, 4.0, 5.0], dtype=np.float32).reshape(5, 1, 1)
    out, mask_rej, mask_thresh, *_ = ndcombine(
        arr,
        combine="variance",
        reject="minmax",
        n_minmax=(0, 1),
        ddof=1,
        full=True,
    )
    assert mask_thresh is None
    arr_eff = np.where(mask_rej, np.nan, arr)

    np.testing.assert_allclose(out, np.nanvar(arr_eff, axis=0, ddof=1), rtol=1e-5)


def test_ndcombine_thresholds_return_mask_thresh_and_precede_combine():
    arr = np.array([1.0, 2.0, 100.0, 4.0, 5.0], dtype=np.float32).reshape(5, 1, 1)

    out, mask_rej, mask_thresh, std, low, upp, nit, output_flags = ndcombine(
        arr,
        combine="mean",
        thresholds=(0.0, 10.0),
        full=True,
    )

    expected_mask = arr > 10.0
    np.testing.assert_allclose(out, 3.0, atol=1e-6)
    assert mask_rej is None
    assert std is None
    np.testing.assert_array_equal(mask_thresh, expected_mask)
    assert low is upp is nit is output_flags is None


def test_ndcombine_diagnostics_simple_matches_legacy_full_tuple():
    arr = np.array([1.0, 2.0, 100.0, 4.0, 5.0], dtype=np.float32).reshape(5, 1, 1)

    simple = ndcombine(
        arr,
        combine="mean",
        reject="minmax",
        n_minmax=(0, 1),
        diagnostics="simple",
    )
    legacy = ndcombine(
        arr,
        combine="mean",
        reject="minmax",
        n_minmax=(0, 1),
        full=True,
    )

    assert len(simple) == 8
    assert len(legacy) == 8
    for got, expected in zip(simple, legacy, strict=True):
        if got is None:
            assert expected is None
        else:
            np.testing.assert_array_equal(got, expected)


def test_ndcombine_diagnostics_full_returns_sample_flags():
    arr = np.array([np.nan, 1.0, -5.0, 2.0, 100.0], dtype=np.float32).reshape(5, 1, 1)
    mask = np.zeros_like(arr, dtype=bool)
    mask[1, 0, 0] = True

    result = ndcombine(
        arr,
        mask=mask,
        thresholds=(0.0, np.inf),
        combine="mean",
        reject="minmax",
        n_minmax=(0, 3),
        diagnostics="full",
    )

    out, mask_rej, mask_thresh, std, low, upp, nit, output_flags, sample_flags = result
    np.testing.assert_allclose(out, 2.0)
    assert mask_rej.shape == arr.shape
    assert sample_flags.shape == arr.shape
    assert sample_flags.dtype == np.uint8
    assert std is None
    assert low.shape == upp.shape == nit.shape == output_flags.shape == (1, 1)
    np.testing.assert_array_equal(mask_thresh, arr < 0.0)
    assert sample_flags[:, 0, 0].tolist() == [2, 1, 4, 0, 8]


def test_public_diagnostic_flag_enums_match_documented_values():
    assert int(imc.SampleFlags.INPUT_MASK) == 1
    assert int(imc.SampleFlags.NONFINITE) == 2
    assert int(imc.SampleFlags.THRESHOLD) == 4
    assert int(imc.SampleFlags.ALGORITHM) == 8
    assert int(imc.SampleFlags.GROW) == 16
    assert int(imc.SampleFlags.PREVIOUS) == 32
    assert int(imc.SampleFlags.RESTORED_NKEEP) == 64
    assert int(imc.SampleFlags.RESTORED_MAXREJ) == 128

    assert int(imc.OutputFlags.PREMASKED) == 1
    assert int(imc.OutputFlags.MAXITERS) == 2
    assert int(imc.OutputFlags.NKEEP) == 4
    assert int(imc.OutputFlags.MAXREJ) == 8
    assert int(imc.OutputFlags.GROW) == 16


def test_public_diagnostic_flag_enums_do_not_export_legacy_aliases():
    assert not hasattr(imc, "Sample\x43ode")
    assert not hasattr(imc, "Rejection\x43ode")


def test_ndcombine_diagnostics_full_marks_grow_only_samples():
    arr = np.ones((10, 5, 5), dtype=np.float32)
    arr[0, 2, 2] = 100.0

    *_, sample_flags = ndcombine(
        arr,
        reject="sigclip",
        sigma=3.0,
        maxiters=5,
        ddof=0,
        nkeep=0,
        combine="mean",
        grow=1,
        diagnostics="full",
    )

    assert sample_flags[0, 2, 2] == 8
    assert sample_flags[0, 1, 2] == 16
    assert sample_flags[0, 1, 1] == 0


def test_ndcombine_diagnostics_full_marks_reverted_nkeep_rejects():
    arr = np.array([0.0, 1.0, 100.0], dtype=np.float32).reshape(3, 1, 1)

    _, mask_rej, _, _, _, _, _, output_flags, sample_flags = ndcombine(
        arr,
        reject="sigclip",
        sigma=(10.0, 0.5),
        maxiters=1,
        ddof=0,
        nkeep=3,
        cenfunc="median",
        clip_cen="median",
        revert_on_nkeep=True,
        diagnostics="full",
    )

    assert not mask_rej.any()
    assert output_flags[0, 0] & 4
    assert sample_flags[:, 0, 0].tolist() == [0, 0, 64]


def test_ndcombine_diagnostics_full_marks_reverted_maxrej_rejects():
    arr = np.array([0.0, 1.0, 100.0], dtype=np.float32).reshape(3, 1, 1)

    _, mask_rej, _, _, _, _, _, output_flags, sample_flags = ndcombine(
        arr,
        reject="sigclip",
        sigma=(10.0, 0.5),
        maxiters=1,
        ddof=0,
        nkeep=0,
        maxrej=0,
        cenfunc="median",
        clip_cen="median",
        revert_on_nkeep=True,
        diagnostics="full",
    )

    assert not mask_rej.any()
    assert output_flags[0, 0] & 8
    assert sample_flags[:, 0, 0].tolist() == [0, 0, 128]


def test_ndcombine_rejects_unknown_diagnostics_level():
    arr = np.ones((3, 1, 1), dtype=np.float32)

    with pytest.raises(ValueError, match="diagnostics"):
        ndcombine(arr, diagnostics="verbose")


def test_ndcombine_rejects_full_and_diagnostics_together():
    arr = np.ones((3, 1, 1), dtype=np.float32)

    with pytest.raises(ValueError, match="either diagnostics or full"):
        ndcombine(arr, full=True, diagnostics="simple")


def test_ndcombine_thresholds_precede_zero_scale_statistics():
    arr = np.array(
        [
            [[1.0, 2.0, 1000.0]],
            [[10.0, 20.0, 1000.0]],
        ],
        dtype=np.float32,
    )

    out = ndcombine(
        arr,
        combine="mean",
        thresholds=(-np.inf, 100.0),
        zero="mean",
        zero_to_0th=False,
    )
    expected = np.array([[-2.75, 2.75, np.nan]], dtype=np.float32)

    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_ndcombine_full_false_sigclip_mean_matches_full_true():
    rng = np.random.default_rng(20250311)
    arr = rng.normal(100.0, 5.0, (31, 8, 9)).astype(np.float32)
    arr[0, 1, 2] = 1000.0
    arr[30, 3, 4] = -500.0
    mask = np.zeros_like(arr, dtype=bool)
    mask[1, 0, 0] = True

    out_fused = ndcombine(
        arr,
        mask=mask,
        combine="mean",
        reject="sigclip",
        full=False,
        sigma=3.0,
        maxiters=5,
    )
    out_reference = ndcombine(
        arr,
        mask=mask,
        combine="mean",
        reject="sigclip",
        full=True,
        sigma=3.0,
        maxiters=5,
    )[0]

    np.testing.assert_allclose(out_fused, out_reference, rtol=1e-6, atol=1e-6)


def test_ndcombine_full_false_fused_sigclip_mean_matches_after_preprocessing():
    rng = np.random.default_rng(20250311)
    arr = rng.normal(100.0, 5.0, (11, 2, 3, 4)).astype(np.float64)
    arr[0, 0, 1, 2] = 1000.0
    arr[3, 1, 2, 0] = -500.0
    arr[4, 0, 0, 0] = np.inf
    mask = np.zeros_like(arr, dtype=bool)
    mask[1, 0, 0, 1] = True

    kwargs = {
        "mask": mask,
        "thresholds": (-200.0, 500.0),
        "zero": "median",
        "scale": np.linspace(0.95, 1.05, arr.shape[0]),
        "combine": "mean",
        "reject": "sigclip",
        "sigma": (2.5, 3.0),
        "maxiters": 4,
        "ddof": 1,
        "nkeep": 2,
        "revert_on_nkeep": True,
    }

    out_fused = ndcombine(arr, full=False, **kwargs)
    out_reference = ndcombine(arr, full=True, **kwargs)[0]

    assert out_fused.shape == arr.shape[1:]
    np.testing.assert_allclose(out_fused, out_reference, rtol=1e-6, atol=1e-6)


def test_ndcombine_full_false_fused_sigclip_mean_matches_promoted_integer_input():
    arr = np.array(
        [
            [[100, 102], [99, 101]],
            [[101, 103], [98, 100]],
            [[100, 104], [97, 102]],
            [[300, 105], [96, 103]],
            [[99, 106], [95, 104]],
        ],
        dtype=np.int16,
    )

    out_fused = ndcombine(
        arr,
        combine="mean",
        reject="sigclip",
        sigma=2.0,
        maxiters=3,
        full=False,
    )
    out_reference = ndcombine(
        arr,
        combine="mean",
        reject="sigclip",
        sigma=2.0,
        maxiters=3,
        full=True,
    )[0]

    np.testing.assert_allclose(out_fused, out_reference, rtol=1e-6, atol=1e-6)


def test_ndcombine_full_false_sigclip_median_matches_full_true():
    rng = np.random.default_rng(20250311)
    arr = rng.normal(100.0, 5.0, (30, 5, 6)).astype(np.float32)
    arr[0, 1, 2] = 1000.0
    arr[29, 3, 4] = -500.0
    mask = np.zeros_like(arr, dtype=bool)
    mask[2, 0, 0] = True

    out_fused = ndcombine(
        arr,
        mask=mask,
        combine="median",
        reject="sigclip",
        full=False,
        sigma=(2.5, 3.0),
        maxiters=5,
    )
    out_reference = ndcombine(
        arr,
        mask=mask,
        combine="median",
        reject="sigclip",
        full=True,
        sigma=(2.5, 3.0),
        maxiters=5,
    )[0]

    np.testing.assert_allclose(out_fused, out_reference, rtol=1e-6, atol=1e-6)


def test_ndcombine_full_false_fused_sigclip_median_matches_after_preprocessing():
    rng = np.random.default_rng(20250311)
    arr = rng.normal(100.0, 5.0, (12, 2, 3, 4)).astype(np.float64)
    arr[0, 0, 1, 2] = 1000.0
    arr[11, 1, 2, 0] = -500.0
    arr[3, 0, 0, 0] = np.inf
    mask = np.zeros_like(arr, dtype=bool)
    mask[1, 0, 0, 1] = True

    kwargs = {
        "mask": mask,
        "thresholds": (-200.0, 500.0),
        "zero": "median",
        "scale": np.linspace(0.95, 1.05, arr.shape[0]),
        "combine": "median",
        "reject": "sigclip",
        "sigma": (2.5, 3.0),
        "maxiters": 4,
        "ddof": 1,
        "nkeep": 2,
        "revert_on_nkeep": True,
    }

    out_fused = ndcombine(arr, full=False, **kwargs)
    out_reference = ndcombine(arr, full=True, **kwargs)[0]

    assert out_fused.shape == arr.shape[1:]
    np.testing.assert_allclose(out_fused, out_reference, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("combine", "kwargs"),
    [
        ("median", {"cenfunc": "mean", "clip_cen": "mean"}),
        ("mean", {"cenfunc": "median", "clip_cen": "median"}),
        ("mean", {"maxrej": 0}),
        ("median", {"nkeep": 6}),
        ("mean", {"maxiters": 1, "sigma": 1.0}),
    ],
)
def test_ndcombine_full_false_sigclip_fused_fallback_recompute_branches(
    combine, kwargs
):
    rng = np.random.default_rng(20250311)
    arr = rng.normal(100.0, 5.0, (6, 4, 5)).astype(np.float32)
    arr[0, 1, 2] = 1000.0
    arr[5, 3, 4] = -500.0

    base_kwargs = {
        "combine": combine,
        "reject": "sigclip",
        "sigma": 2.0,
        "maxiters": 5,
    }
    base_kwargs.update(kwargs)

    out_fused = ndcombine(arr, full=False, **base_kwargs)
    out_reference = ndcombine(arr, full=True, **base_kwargs)[0]

    np.testing.assert_allclose(
        out_fused, out_reference, rtol=1e-6, atol=1e-6, equal_nan=True
    )


@pytest.mark.parametrize("combine", ["mean", "median"])
def test_ndcombine_full_false_sigclip_fused_all_masked_columns_are_nan(combine):
    arr = np.arange(24, dtype=np.float32).reshape(6, 2, 2)
    mask = np.zeros_like(arr, dtype=bool)
    mask[:, 0, 1] = True
    arr[:, 1, 0] = np.nan

    out_fused = ndcombine(
        arr,
        mask=mask,
        combine=combine,
        reject="sigclip",
        full=False,
    )
    out_reference = ndcombine(
        arr,
        mask=mask,
        combine=combine,
        reject="sigclip",
        full=True,
    )[0]

    np.testing.assert_allclose(
        out_fused, out_reference, rtol=1e-6, atol=1e-6, equal_nan=True
    )
    assert np.isnan(out_fused[0, 1])
    assert np.isnan(out_fused[1, 0])


@pytest.mark.parametrize(
    ("reject", "combine", "kwargs"),
    [
        ("ccdclip", "mean", {"sigma": 2.0, "rdnoise": 5.0, "gain": 2.0}),
        ("ccdclip", "median", {"sigma": 2.0, "rdnoise": 5.0, "gain": 2.0}),
        ("minmax", "mean", {"n_minmax": (1, 1)}),
        ("minmax", "median", {"n_minmax": (1, 1)}),
        ("pclip", "mean", {"pclip": 0.25}),
        ("pclip", "median", {"pclip": 0.25}),
    ],
)
def test_ndcombine_full_false_fused_reject_combine_matches_full_true(
    reject, combine, kwargs
):
    rng = np.random.default_rng(20250311)
    arr = rng.normal(100.0, 5.0, (9, 4, 5)).astype(np.float32)
    arr[0, 1, 2] = 1_000.0
    arr[1, 2, 3] = -500.0
    mask = np.zeros_like(arr, dtype=bool)
    mask[2, 0, 0] = True

    out_fused = ndcombine(
        arr,
        mask=mask,
        combine=combine,
        reject=reject,
        full=False,
        **kwargs,
    )
    out_reference = ndcombine(
        arr,
        mask=mask,
        combine=combine,
        reject=reject,
        full=True,
        **kwargs,
    )[0]

    np.testing.assert_allclose(
        out_fused, out_reference, rtol=1e-6, atol=1e-6, equal_nan=True
    )


@pytest.mark.parametrize(
    ("reject", "mask_name", "combine_name", "kwargs", "nd_kwargs"),
    [
        ("sigclip", "sigclip_mask", "sigclip_combine", {}, {}),
        ("ccdclip", "ccdclip_mask", "ccdclip_combine", {}, {}),
        ("minmax", "minmax_mask", "minmax_combine", {}, {}),
        (
            "pclip",
            "pclip_mask",
            "pclip_combine",
            {"frac": 0.25},
            {"pclip": 0.25},
        ),
    ],
)
def test_public_output_only_reject_kernels_match_ndcombine(
    reject, mask_name, combine_name, kwargs, nd_kwargs
):
    rng = np.random.default_rng(20250311)
    arr = rng.normal(100.0, 5.0, (9, 4, 5)).astype(np.float32)
    arr[0, 1, 2] = 1_000.0

    mask_func = getattr(imc.kernels, mask_name)
    combine_func = getattr(imc.kernels, combine_name)
    mask = mask_func(arr, validate=False, **kwargs)
    out_kernel = combine_func(arr, combine="median", validate=False, **kwargs)
    out_nd = ndcombine(
        arr,
        combine="median",
        reject=reject,
        full=False,
        validate=False,
        **nd_kwargs,
    )

    assert mask.shape == arr.shape
    np.testing.assert_allclose(out_kernel, out_nd, rtol=1e-6, atol=1e-6)


def test_ndcombine_ccdclip_mask_path_passes_gain_without_python_predivide(monkeypatch):
    arr = np.array([18.0, 20.0, 22.0, 200.0], dtype=np.float32).reshape(4, 1, 1)
    called = {}

    def fake_ccdclip_mask(arr_in, **kwargs):
        called["values"] = arr_in.copy()
        called["gain"] = kwargs["gain"]
        return np.zeros(arr_in.shape, dtype=bool)

    monkeypatch.setattr(imc.kernels, "ccdclip_mask", fake_ccdclip_mask)

    imc.ndcombine(
        arr,
        combine="sum",
        reject="ccdclip",
        gain=2.0,
        full=False,
    )

    assert called["gain"] == 2.0
    np.testing.assert_array_equal(called["values"], arr)


@pytest.mark.parametrize(
    ("reject", "combine", "kwargs"),
    [
        ("ccdclip", "mean", {"sigma": 2.0, "rdnoise": 5.0, "gain": 1.7}),
        ("ccdclip", "median", {"sigma": 2.0, "rdnoise": 5.0, "gain": 1.7}),
        ("minmax", "mean", {"n_minmax": (1, 1)}),
        ("minmax", "median", {"n_minmax": (1, 1)}),
        ("pclip", "mean", {"pclip": 0.25}),
        ("pclip", "median", {"pclip": 0.25}),
    ],
)
def test_ndcombine_fused_reject_matches_full_true_preprocessed_integer(
    reject, combine, kwargs
):
    rng = np.random.default_rng(20250312)
    arr = rng.normal(1000.0, 25.0, (9, 2, 3, 4)).astype(np.int16)
    arr[0, 0, 1, 2] = 5000
    arr[1, 1, 2, 0] = -500
    mask = np.zeros_like(arr, dtype=bool)
    mask[:, 0, 0, 0] = True
    mask[2, 1, 0, 1] = True

    call_kwargs = {
        "mask": mask,
        "thresholds": (850.0, 1150.0),
        "zero": np.linspace(-3.0, 3.0, arr.shape[0]),
        "scale": np.linspace(0.98, 1.02, arr.shape[0]),
        "combine": combine,
        "reject": reject,
        **kwargs,
    }

    out_fused = ndcombine(arr, full=False, **call_kwargs)
    out_reference = ndcombine(arr, full=True, **call_kwargs)[0]

    assert out_fused.shape == arr.shape[1:]
    assert np.isnan(out_fused[0, 0, 0])
    np.testing.assert_allclose(
        out_fused, out_reference, rtol=1e-6, atol=1e-6, equal_nan=True
    )


def test_ndcombine_full_false_uses_fused_sigclip_mean_when_available(monkeypatch):
    rng = np.random.default_rng(20250311)
    arr = rng.normal(100.0, 5.0, (9, 4, 5)).astype(np.float32)
    called = {}

    def fake_fused(arr_in, **kwargs):
        called["shape"] = arr_in.shape
        called["combine"] = kwargs["combine"]
        return np.zeros(arr_in.shape[1:], dtype=np.float32)

    def old_mask_path(*args, **kwargs):
        raise AssertionError("mask-only rejection path should not be used")

    monkeypatch.setattr(imc.kernels, "sigclip_combine", fake_fused, raising=False)
    monkeypatch.setattr(imc.kernels, "sigclip_mask", old_mask_path)

    out = ndcombine(arr, combine="mean", reject="sigclip", full=False)

    assert called == {"shape": arr.shape, "combine": "mean"}
    assert out.shape == arr.shape[1:]


def test_ndcombine_full_false_uses_fused_sigclip_median_when_available(monkeypatch):
    rng = np.random.default_rng(20250311)
    arr = rng.normal(100.0, 5.0, (10, 4, 5)).astype(np.float32)
    called = {}

    def fake_fused(arr_in, **kwargs):
        called["shape"] = arr_in.shape
        called["combine"] = kwargs["combine"]
        return np.zeros(arr_in.shape[1:], dtype=np.float32)

    def old_mask_path(*args, **kwargs):
        raise AssertionError("mask-only rejection path should not be used")

    monkeypatch.setattr(imc.kernels, "sigclip_combine", fake_fused, raising=False)
    monkeypatch.setattr(imc.kernels, "sigclip_mask", old_mask_path)

    out = ndcombine(arr, combine="median", reject="sigclip", full=False)

    assert called == {"shape": arr.shape, "combine": "median"}
    assert out.shape == arr.shape[1:]


@pytest.mark.parametrize(
    ("reject", "combine", "fused_name", "mask_name", "kwargs"),
    [
        ("ccdclip", "mean", "ccdclip_combine", "ccdclip_mask", {"gain": 2.0}),
        ("ccdclip", "median", "ccdclip_combine", "ccdclip_mask", {"gain": 2.0}),
        ("minmax", "mean", "minmax_combine", "minmax_mask", {"n_minmax": (1, 1)}),
        (
            "minmax",
            "median",
            "minmax_combine",
            "minmax_mask",
            {"n_minmax": (1, 1)},
        ),
        ("pclip", "mean", "pclip_combine", "pclip_mask", {"pclip": 0.25}),
        ("pclip", "median", "pclip_combine", "pclip_mask", {"pclip": 0.25}),
    ],
)
def test_ndcombine_full_false_uses_fused_reject_combine_when_available(
    monkeypatch, reject, combine, fused_name, mask_name, kwargs
):
    rng = np.random.default_rng(20250311)
    arr = rng.normal(100.0, 5.0, (9, 4, 5)).astype(np.float32)
    called = {}

    def fake_fused(arr_in, **call_kwargs):
        called["shape"] = arr_in.shape
        called["combine"] = call_kwargs["combine"]
        if reject == "ccdclip":
            called["gain"] = call_kwargs["gain"]
        return np.zeros(arr_in.shape[1:], dtype=np.float32)

    def old_mask_path(*args, **call_kwargs):
        raise AssertionError("mask-only rejection path should not be used")

    monkeypatch.setattr(imc.kernels, fused_name, fake_fused, raising=False)
    monkeypatch.setattr(imc.kernels, mask_name, old_mask_path)

    out = ndcombine(arr, combine=combine, reject=reject, full=False, **kwargs)

    expected = {"shape": arr.shape, "combine": combine}
    if reject == "ccdclip":
        expected["gain"] = kwargs["gain"]
    assert called == expected
    assert out.shape == arr.shape[1:]


@pytest.mark.parametrize(
    ("reject", "fused_name", "kwargs"),
    [
        ("sigclip", "sigclip_combine", {}),
        ("ccdclip", "ccdclip_combine", {"gain": 2.0}),
        ("minmax", "minmax_combine", {"n_minmax": (1, 1)}),
        ("pclip", "pclip_combine", {"pclip": 0.25}),
    ],
)
def test_ndcombine_full_false_falls_back_when_grow_requested(
    monkeypatch, reject, fused_name, kwargs
):
    rng = np.random.default_rng(20250311)
    arr = rng.normal(100.0, 5.0, (9, 4, 5)).astype(np.float32)

    def fused_should_not_run(*args, **kwargs):
        raise AssertionError("fused path should not run when grow is requested")

    monkeypatch.setattr(imc.kernels, fused_name, fused_should_not_run, raising=False)

    out = ndcombine(
        arr,
        combine="mean",
        reject=reject,
        grow=1,
        full=False,
        **kwargs,
    )

    assert out.shape == arr.shape[1:]


@pytest.mark.parametrize(
    ("reject", "fused_name", "kwargs"),
    [
        ("sigclip", "sigclip_combine", {}),
        ("ccdclip", "ccdclip_combine", {"gain": 2.0}),
        ("minmax", "minmax_combine", {"n_minmax": (1, 1)}),
        ("pclip", "pclip_combine", {"pclip": 0.25}),
    ],
)
def test_ndcombine_full_true_never_uses_fused_path(
    monkeypatch, reject, fused_name, kwargs
):
    rng = np.random.default_rng(20250311)
    arr = rng.normal(100.0, 5.0, (9, 4, 5)).astype(np.float32)

    def fused_should_not_run(*args, **kwargs):
        raise AssertionError("fused path should not run for full=True")

    monkeypatch.setattr(imc.kernels, fused_name, fused_should_not_run, raising=False)

    out, mask_rej, _, std, low, upp, nit, output_flags = ndcombine(
        arr,
        combine="mean",
        reject=reject,
        full=True,
        **kwargs,
    )

    assert out.shape == arr.shape[1:]
    assert mask_rej.shape == arr.shape
    assert low.shape == upp.shape == nit.shape == output_flags.shape == arr.shape[1:]
    if reject in ("sigclip", "ccdclip"):
        assert std.shape == arr.shape[1:]
    else:
        assert std is None


@pytest.mark.parametrize("reject", ["sigclip", "ccdclip", "minmax", "pclip"])
def test_ndcombine_full_false_does_not_use_diagnostic_rejection_path(
    monkeypatch, reject
):
    rng = np.random.default_rng(20250311)
    arr = rng.normal(100.0, 5.0, (7, 3, 4)).astype(np.float32)
    arr[0, 1, 2] = 1_000.0

    def diagnostic_path_used(*args, **kwargs):
        raise AssertionError("diagnostic rejection path should not be used")

    monkeypatch.setattr(imc.kernels, reject, diagnostic_path_used)

    kwargs = {"reject": reject, "combine": "mean", "sigma": 3.0}
    if reject == "ccdclip":
        kwargs["rdnoise"] = 5.0
    elif reject == "minmax":
        kwargs["n_minmax"] = (0, 1)
    elif reject == "pclip":
        kwargs["pclip"] = 0.25

    out = ndcombine(arr, **kwargs)

    assert out.shape == arr.shape[1:]
    assert np.isfinite(out).all()


@pytest.mark.parametrize("reject", ["sigclip", "ccdclip", "minmax", "pclip"])
def test_ndcombine_mask_only_rejection_matches_diagnostic_path(reject):
    rng = np.random.default_rng(20250311)
    arr = rng.normal(100.0, 5.0, (9, 4, 5)).astype(np.float32)
    arr[0, 1, 2] = 1_000.0
    arr[1, 2, 3] = -500.0

    kwargs = {"reject": reject, "combine": "mean", "sigma": 3.0, "grow": 1}
    if reject == "ccdclip":
        kwargs["rdnoise"] = 5.0
    elif reject == "minmax":
        kwargs["n_minmax"] = (1, 1)
    elif reject == "pclip":
        kwargs["pclip"] = 0.25

    out_mask_only = ndcombine(arr, **kwargs)
    out_diag = ndcombine(arr, full=True, **kwargs)[0]

    np.testing.assert_allclose(out_mask_only, out_diag, rtol=1e-6, atol=1e-6)


def test_ndcombine_median_sigclip_full_reports_kernel_rejection_mask():
    arr = np.array([970, 1002, 1004, 1006, 1008, 1010, 1012], dtype=np.float32)
    arr = arr.reshape(7, 1, 1)

    out, mask_rej, _, _, low, upp, _, _ = ndcombine(
        arr,
        combine="median",
        reject="sigclip",
        sigma=(2.0, 2.0),
        cenfunc="median",
        maxiters=5,
        nkeep=1,
        revert_on_nkeep=True,
        full=True,
    )

    assert out[0, 0] == 1007.0
    assert mask_rej[:, 0, 0].tolist() == [
        True,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    assert low[0, 0] == 1002.0
    assert upp[0, 0] == 1012.0


def test_ndcombine_median_pclip_full_reports_kernel_rejection_mask():
    arr = np.array([970, 1002, 1004, 1006, 1008, 1010, 1012], dtype=np.float32)
    arr = arr.reshape(7, 1, 1)

    out, mask_rej, _, _, low, upp, _, _ = ndcombine(
        arr,
        combine="median",
        reject="pclip",
        pclip=0.25,
        full=True,
    )

    assert out[0, 0] == 1006.0
    assert mask_rej[:, 0, 0].tolist() == [
        True,
        False,
        False,
        False,
        False,
        False,
        True,
    ]
    assert low[0, 0] == 1002.0
    assert upp[0, 0] == 1010.0


def test_nanmean_with_nan(stack):
    s = stack.copy()
    s[0, 0, 0] = np.nan
    s[2, 3, 4] = np.nan
    out = _core.combine(s, "mean")
    np.testing.assert_allclose(out, np.nanmean(s, axis=0), rtol=1e-5)


def test_lmedian_even_n():
    # For even N, lmedian returns the lower of the two middle values.
    arr = np.arange(6, dtype=np.float32).reshape(6, 1, 1)
    # sorted: 0,1,2,3,4,5; lower-middle = 2
    out = _core.combine(arr, "lmedian")
    assert out[0, 0] == 2.0
    # Standard median = (2+3)/2 = 2.5
    out2 = _core.combine(arr, "median")
    assert out2[0, 0] == 2.5


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int16, np.int32])
def test_lmedian_integer_input_preserves_dtype(dtype):
    arr = np.arange(24, dtype=dtype).reshape(6, 2, 2)

    out = _core.lmedian(arr)

    assert out.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(out, np.sort(arr, axis=0)[2])


def test_ndcombine_lmedian_integer_input_preserves_dtype():
    arr = np.arange(24, dtype=np.uint16).reshape(6, 2, 2)

    out = ndcombine(arr, combine="lmedian")

    assert out.dtype == np.uint16
    np.testing.assert_array_equal(out, np.sort(arr, axis=0)[2])


def test_ndcombine_lmedian_with_zero_uses_float_workspace():
    arr = np.arange(24, dtype=np.uint16).reshape(6, 2, 2)

    out = ndcombine(arr, combine="lmedian", zero=1)

    assert out.dtype == np.float32
    np.testing.assert_allclose(out, np.sort(arr.astype(np.float32) - 1, axis=0)[2])


def test_weighted_average():
    arr = np.ones((3, 2, 2), dtype=np.float32)
    arr[0] *= 10
    arr[1] *= 20
    arr[2] *= 30
    w = np.array([1.0, 1.0, 2.0])
    out = _core.combine(arr, "weighted_average", weights=w)
    expected = (10 * 1 + 20 * 1 + 30 * 2) / 4
    np.testing.assert_allclose(out, expected)
    np.testing.assert_allclose(_core.combine(arr, "wvg", weights=w), expected)


def test_weighted_average_wavg_alias_is_rejected():
    arr = np.ones((3, 2, 2), dtype=np.float32)
    arr[0] *= 10
    arr[1] *= 20
    arr[2] *= 30
    w = np.array([1.0, 1.0, 2.0])

    with pytest.raises(ValueError, match="unknown combine method"):
        _core.combine(arr, "wavg", weights=w)
    with pytest.raises(ValueError, match="unknown combine method"):
        ndcombine(arr, combine="wavg", weight=w)
    with pytest.raises(ValueError, match="unknown combine method"):
        imc.Combiner(arr).combine("wavg", weight=w)


def test_float64():
    rng = np.random.default_rng(20250311)
    arr = rng.normal(0, 1, (5, 4, 4)).astype(np.float64)
    out = _core.combine(arr, "mean")
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, arr.mean(axis=0))


def test_ndcombine_top_level(stack):
    out = ndcombine(stack, combine="mean")
    np.testing.assert_allclose(out, stack.mean(axis=0), rtol=1e-5)


def test_ndcombine_zero_scale():
    arr = np.ones((3, 2, 2), dtype=np.float32) * 10
    arr[1] = 20
    arr[2] = 30
    # zero[i] equals arr[i]'s pixel value, so every plane becomes 0.
    out = ndcombine(
        arr,
        combine="mean",
        zero=np.array([10.0, 20.0, 30.0]),
        zero_to_0th=False,
    )
    np.testing.assert_allclose(out, 0.0, atol=1e-6)


def test_ndcombine_zero_scale_to_0th():
    arr = np.ones((3, 2, 2), dtype=np.float32) * 10
    arr[1] = 20
    arr[2] = 40

    out = ndcombine(
        arr,
        combine="mean",
        zero=np.array([10.0, 20.0, 40.0]),
        scale=np.array([1.0, 2.0, 4.0]),
        zero_to_0th=True,
        scale_to_0th=True,
    )
    expected = np.mean(
        (arr - np.array([0.0, 10.0, 30.0]).reshape(-1, 1, 1))
        / np.array([1.0, 2.0, 4.0]).reshape(-1, 1, 1),
        axis=0,
    )

    np.testing.assert_allclose(out, expected)


def test_ndcombine_zero_scale_to_0th_defaults():
    arr = np.ones((3, 2, 2), dtype=np.float32) * 10
    arr[1] = 20
    arr[2] = 40

    out = ndcombine(
        arr,
        combine="mean",
        zero=np.array([10.0, 20.0, 40.0]),
        scale=np.array([1.0, 2.0, 4.0]),
    )
    expected = np.mean(
        (arr - np.array([0.0, 10.0, 30.0]).reshape(-1, 1, 1))
        / np.array([1.0, 2.0, 4.0]).reshape(-1, 1, 1),
        axis=0,
    )

    np.testing.assert_allclose(out, expected)


def test_ndcombine_zero_string_computes_per_plane_statistic():
    arr = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[10.0, 20.0], [30.0, 40.0]],
            [[-2.0, 0.0], [2.0, 4.0]],
        ],
        dtype=np.float32,
    )

    out = ndcombine(arr, combine="mean", zero="median", zero_to_0th=False)
    expected = np.mean(arr - np.median(arr, axis=(1, 2)).reshape(-1, 1, 1), axis=0)

    np.testing.assert_allclose(out, expected)


def test_ndcombine_scale_callable_computes_per_plane_statistic():
    arr = np.array(
        [
            [[2.0, 4.0], [6.0, 8.0]],
            [[1.0, 3.0], [5.0, 7.0]],
            [[10.0, 10.0], [10.0, 10.0]],
        ],
        dtype=np.float32,
    )

    out = ndcombine(arr, combine="mean", scale=np.mean, scale_to_0th=False)
    expected = np.mean(arr / np.mean(arr, axis=(1, 2)).reshape(-1, 1, 1), axis=0)

    np.testing.assert_allclose(out, expected)
