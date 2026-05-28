"""Tests for the fluent Combiner object + Rejector dataclasses."""

from __future__ import annotations

import imcombiners as imc
import numpy as np
import pytest
from imcombiners import CcdClip, Combiner, LinearClip, MinMaxClip, PClip, SigClip


def test_combiner_repr():
    arr = np.zeros((3, 4, 5), dtype=np.float32)
    c = Combiner(arr)
    r = repr(c)
    assert "N=3" in r and "H=4" in r and "W=5" in r


def test_combiner_basic_pipeline():
    rng = np.random.default_rng(20250311)
    arr = rng.normal(0, 1, (20, 4, 4)).astype(np.float32)
    arr[0] = 1e3  # outlier plane
    out = Combiner(arr).reject(SigClip(sigma=3.0, maxiters=5)).combine("mean")
    # Outlier rejected; mean near 0, not ~50.
    assert np.abs(out).max() < 5.0


def test_combiner_zero_scale_then_combine():
    arr = np.ones((3, 2, 2), dtype=np.float32) * 10
    arr[1] = 20
    arr[2] = 30
    out = (
        Combiner(arr)
        .zero_scale(zero=np.array([10.0, 20.0, 30.0]), zero_to_0th=False)
        .combine("mean")
    )
    np.testing.assert_allclose(out, 0.0, atol=1e-6)


def test_combiner_combine_accepts_inline_zero_scale_without_mutation():
    arr = np.ones((3, 2, 2), dtype=np.float32) * 10
    arr[1] = 20
    arr[2] = 40
    c = Combiner(arr)

    out = c.combine(
        "mean",
        zero=np.array([10.0, 20.0, 40.0]),
        scale=np.array([1.0, 2.0, 4.0]),
        full=False,
    )
    expected = (
        Combiner(arr)
        .zero_scale(
            zero=np.array([10.0, 20.0, 40.0]),
            scale=np.array([1.0, 2.0, 4.0]),
        )
        .combine("mean")
    )

    np.testing.assert_allclose(out, expected)
    np.testing.assert_array_equal(c.arr, arr)
    assert c.mask is None
    assert c.mask_thresh is None
    assert c.last_reject is None


def test_combiner_inline_zero_scale_statistics_follow_thresholds():
    arr = np.array(
        [
            [[1.0, 2.0, 1000.0]],
            [[10.0, 20.0, 1000.0]],
        ],
        dtype=np.float32,
    )
    c = Combiner(arr)

    out = c.combine(
        "mean",
        thresholds=(-np.inf, 100.0),
        zero="mean",
        zero_to_0th=False,
        full=False,
    )
    expected = (
        Combiner(arr)
        .threshold(-np.inf, 100.0)
        .zero_scale(zero="mean", zero_to_0th=False)
        .combine("mean")
    )

    np.testing.assert_allclose(out, expected, atol=1e-6)
    assert c.mask is None
    assert c.mask_thresh is None


def test_combiner_inline_zero_scale_full_true_mutates_like_chain():
    arr = np.ones((3, 2, 2), dtype=np.float32) * 10
    arr[1] = 20
    arr[2] = 40
    c = Combiner(arr)

    out = c.combine(
        "mean",
        zero=np.array([10.0, 20.0, 40.0]),
        scale=np.array([1.0, 2.0, 4.0]),
        full=True,
    )
    explicit = Combiner(arr).zero_scale(
        zero=np.array([10.0, 20.0, 40.0]),
        scale=np.array([1.0, 2.0, 4.0]),
    )
    expected = explicit.combine("mean")

    np.testing.assert_allclose(out, expected)
    np.testing.assert_allclose(c.arr, explicit.arr)


def test_combiner_inline_zero_scale_preserves_fused_reject_path(monkeypatch):
    rng = np.random.default_rng(20250311)
    arr = rng.normal(100.0, 5.0, (9, 4, 5)).astype(np.float32)
    c = Combiner(arr)
    called = {}

    def fake_fused(arr_in, **kwargs):
        called["first_plane_mean"] = float(arr_in[0].mean())
        called["second_plane_mean"] = float(arr_in[1].mean())
        called["combine"] = kwargs["combine"]
        return np.zeros(arr_in.shape[1:], dtype=np.float32)

    monkeypatch.setattr(imc.kernels, "sigclip_combine", fake_fused, raising=False)

    out = c.combine(
        "median",
        scale=np.arange(1.0, arr.shape[0] + 1.0, dtype=np.float32),
        scale_to_0th=False,
        rejectors=SigClip(sigma=3.0),
        full=False,
    )

    assert called == {
        "first_plane_mean": float(arr[0].mean()),
        "second_plane_mean": float((arr[1] / 2.0).mean()),
        "combine": "median",
    }
    assert out.shape == arr.shape[1:]
    assert c.mask is None
    assert c.last_reject is None


def test_combiner_chains_multiple_rejections():
    rng = np.random.default_rng(20250311)
    arr = rng.normal(0, 1, (20, 3, 3)).astype(np.float32)
    arr[0] = 1e3  # extreme positive
    arr[1] = -1e3  # extreme negative
    c = Combiner(arr).reject(MinMaxClip(n_min=1, n_max=1))
    n_after_minmax = c.n_rejected
    c.reject(SigClip(sigma=3.0, maxiters=5))
    assert c.n_rejected >= n_after_minmax
    # last_reject reflects the *most recent* call (SigClip).
    assert c.last_reject is not None


def test_minmaxclip_uses_min_max_count_names():
    arr = np.arange(10, dtype=np.float32).reshape(10, 1, 1)

    c = Combiner(arr).reject(MinMaxClip(n_min=1, n_max=2))

    assert c.mask_rej[:, 0, 0].tolist() == [
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
    ]


def test_minmaxclip_rejects_old_low_high_keyword_names():
    with pytest.raises(TypeError, match="n_low"):
        MinMaxClip(n_low=1)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="n_high"):
        MinMaxClip(n_high=1)  # type: ignore[call-arg]


def test_combiner_exposes_last_reject_diagnostics():
    arr = np.arange(10, dtype=np.float32).reshape(10, 1, 1)
    c = Combiner(arr).reject(MinMaxClip(n_min=1, n_max=1))

    assert c.mask_rej is c.last_reject[0]
    assert c.std is c.last_reject[1]
    assert c.low is c.last_reject[2]
    assert c.upp is c.last_reject[3]
    assert c.nit is c.last_reject[4]
    assert c.output_flags is c.last_reject[5]


def test_combiner_sigclip_exposes_pixel_std_diagnostic():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32).reshape(3, 1, 1)

    c = Combiner(arr).reject(
        SigClip(sigma=10.0, maxiters=5, cenfunc="median", clip_cen="median")
    )

    np.testing.assert_allclose(c.std[0, 0], np.std([1.0, 2.0, 3.0], ddof=0))


def test_combiner_non_sigma_rejector_std_is_none():
    arr = np.arange(10, dtype=np.float32).reshape(10, 1, 1)

    c = Combiner(arr).reject(MinMaxClip(n_min=1, n_max=1))

    assert c.std is None
    assert c.last_reject[1] is None


def test_combiner_respects_input_mask():
    arr = np.array([1.0, 2.0, 100.0, 4.0, 5.0], dtype=np.float32).reshape(5, 1, 1)
    m_in = np.zeros((5, 1, 1), dtype=bool)
    m_in[2] = True
    out = (
        Combiner(arr, mask=m_in)
        .reject(SigClip(sigma=3.0, maxiters=5, nkeep=1, maxrej=4))
        .combine("mean")
    )
    # 100 is masked, so mean is (1+2+4+5)/4 = 3.0
    np.testing.assert_allclose(out, 3.0, atol=1e-6)


def test_linearclip_default_is_noop():
    arr = np.array([1.0, 2.0, 100.0, 4.0], dtype=np.float32).reshape(4, 1, 1)

    c = Combiner(arr).reject(LinearClip())

    assert c.n_rejected == 0
    np.testing.assert_array_equal(c.mask_rej, np.zeros_like(arr, dtype=bool))
    assert c.nit[0, 0] == 0
    assert c.output_flags[0, 0] == 0


def test_linearclip_rejects_outside_affine_median_bounds():
    arr = np.array([8.0, 10.0, 11.0, 12.0, 40.0], dtype=np.float32).reshape(5, 1, 1)

    c = Combiner(arr).reject(LinearClip(low_scale=1.0, low=1.0, upp_scale=1.0, upp=2.0))

    # median center is 11, so retained interval is [10, 13].
    assert c.mask_rej[:, 0, 0].tolist() == [True, False, False, False, True]
    assert c.low[0, 0] == 10.0
    assert c.upp[0, 0] == 13.0
    assert c.nit[0, 0] == 1
    assert c.output_flags[0, 0] == 0


def test_linearclip_respects_input_mask_when_estimating_center():
    arr = np.array([1.0, 10.0, 11.0, 12.0, 100.0], dtype=np.float32).reshape(5, 1, 1)
    mask = np.zeros_like(arr, dtype=bool)
    mask[4] = True

    c = Combiner(arr, mask=mask).reject(
        LinearClip(low_scale=1.0, low=1.0, upp_scale=1.0, upp=2.0)
    )

    # Center is median([1, 10, 11, 12]) = 10.5, so 1 is rejected and the
    # pre-masked 100 is not counted as this rejection step's mask.
    assert c.mask_rej[:, 0, 0].tolist() == [True, False, False, False, False]
    assert c.output_flags[0, 0] & 1


def test_linearclip_accepts_lower_median_center():
    arr = np.array([8.0, 10.0, 10.6, 40.0], dtype=np.float32).reshape(4, 1, 1)

    c = Combiner(arr).reject(
        LinearClip(
            low_scale=1.0,
            low=1.0,
            upp_scale=1.0,
            upp=0.4,
            cenfunc="lmed",
        )
    )

    # Lower median center is 10.0, so the retained interval is [9.0, 10.4].
    assert c.mask_rej[:, 0, 0].tolist() == [True, False, True, True]
    assert c.low[0, 0] == 9.0
    assert c.upp[0, 0] == 10.4
    assert c.nit[0, 0] == 1


def test_linearclip_iterative_guards_restore_full_diagnostic_candidates():
    arr = np.array([0.0, 1.0, 100.0], dtype=np.float32).reshape(3, 1, 1)

    c_nkeep = Combiner(arr).reject(
        LinearClip(
            low_scale=1.0,
            low=10.0,
            upp_scale=1.0,
            upp=0.5,
            maxiters=1,
            nkeep=3,
            cenfunc="median",
            revert_on_nkeep=True,
        ),
        diagnostics="full",
    )
    assert not c_nkeep.mask_rej.any()
    assert c_nkeep.output_flags[0, 0] & 4
    assert c_nkeep.sample_flags[:, 0, 0].tolist() == [0, 0, 64]

    c_maxrej = Combiner(arr).reject(
        LinearClip(
            low_scale=1.0,
            low=10.0,
            upp_scale=1.0,
            upp=0.5,
            maxiters=1,
            nkeep=0,
            maxrej=0,
            cenfunc="median",
            revert_on_nkeep=True,
        ),
        diagnostics="full",
    )
    assert not c_maxrej.mask_rej.any()
    assert c_maxrej.output_flags[0, 0] & 8
    assert c_maxrej.sample_flags[:, 0, 0].tolist() == [0, 0, 128]


@pytest.mark.parametrize(
    "kwargs,name",
    [
        ({"low_scale": -1.0}, "low_scale"),
        ({"upp_scale": -1.0}, "upp_scale"),
        ({"low": -1.0}, "low"),
        ({"upp": -1.0}, "upp"),
        ({"low": np.nan}, "finite"),
    ],
)
def test_linearclip_rejects_invalid_parameters(kwargs, name):
    arr = np.ones((3, 1, 1), dtype=np.float32)

    with pytest.raises(ValueError, match=name):
        Combiner(arr).reject(LinearClip(**kwargs))


def test_combiner_threshold_records_mask_and_excludes_from_combine():
    arr = np.array([1.0, 2.0, 100.0, 4.0, 5.0], dtype=np.float32).reshape(5, 1, 1)

    c = Combiner(arr).threshold(0.0, 10.0)
    out = c.combine("mean")

    expected_mask = arr > 10.0
    np.testing.assert_array_equal(c.mask_thresh, expected_mask)
    np.testing.assert_array_equal(c.mask, expected_mask)
    np.testing.assert_allclose(out, 3.0, atol=1e-6)


def test_combiner_threshold_precedes_zero_scale_statistics():
    arr = np.array(
        [
            [[1.0, 2.0, 1000.0]],
            [[10.0, 20.0, 1000.0]],
        ],
        dtype=np.float32,
    )

    out = (
        Combiner(arr)
        .threshold(-np.inf, 100.0)
        .zero_scale(zero="mean", zero_to_0th=False)
        .combine("mean")
    )
    expected = np.array([[-2.75, 2.75, np.nan]], dtype=np.float32)

    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_combiner_does_not_mutate_input():
    arr = np.ones((3, 2, 2), dtype=np.float32)
    arr_orig = arr.copy()
    Combiner(arr).zero_scale(zero=np.array([0.0, 0.5, 1.0])).combine("mean")
    np.testing.assert_array_equal(arr, arr_orig)


def test_combiner_copy_false_can_share_float_workspace():
    arr = np.ones((3, 2, 2), dtype=np.float32)

    c = Combiner(arr, copy=False)
    c.arr[0, 0, 0] = 42

    assert arr[0, 0, 0] == 42


def test_combiner_copy_false_still_promotes_integer_input():
    arr = np.ones((3, 2, 2), dtype=np.uint16)

    c = Combiner(arr, copy=False)
    c.arr[0, 0, 0] = 42

    assert c.arr.dtype == np.float32
    assert arr[0, 0, 0] == 1


def test_combiner_copy_independent():
    arr = np.ones((3, 2, 2), dtype=np.float32)
    c1 = Combiner(arr).reject(MinMaxClip(n_min=1, n_max=0))
    c2 = c1.copy()
    # Mutating c1.arr doesn't change c2.arr.
    c1.arr[:] = 999
    assert not np.allclose(c1.arr, c2.arr)


def test_reject_spec_type_error():
    arr = np.zeros((3, 2, 2), dtype=np.float32)
    with pytest.raises(TypeError, match="Rejector"):
        Combiner(arr).reject("sigclip")  # type: ignore[arg-type]


def test_spec_reuse_across_stacks():
    spec = SigClip(sigma=3.0, maxiters=5)
    rng = np.random.default_rng(20250311)
    for _ in range(3):
        arr = rng.normal(0, 1, (10, 4, 4)).astype(np.float32)
        out = Combiner(arr).reject(spec).combine("mean")
        assert out.shape == (4, 4)


def test_weighted_average_via_combiner():
    arr = np.ones((3, 2, 2), dtype=np.float32)
    arr[0] *= 10
    arr[1] *= 20
    arr[2] *= 30
    out = Combiner(arr).combine("weighted_average", weight=np.array([1.0, 1.0, 2.0]))
    expected = (10 * 1 + 20 * 1 + 30 * 2) / 4
    np.testing.assert_allclose(out, expected)
    np.testing.assert_allclose(
        Combiner(arr).combine("wvg", weight=np.array([1.0, 1.0, 2.0])),
        expected,
    )


def test_combiner_inline_rejectors_full_false_uses_fused(monkeypatch):
    rng = np.random.default_rng(20250311)
    arr = rng.normal(100.0, 5.0, (9, 4, 5)).astype(np.float32)
    c = Combiner(arr)
    called = {}

    def fake_fused(arr_in, **kwargs):
        called["shape"] = arr_in.shape
        called["combine"] = kwargs["combine"]
        return np.zeros(arr_in.shape[1:], dtype=np.float32)

    def old_mask_path(*args, **kwargs):
        raise AssertionError("mask-only rejection path should not be used")

    monkeypatch.setattr(imc.kernels, "sigclip_combine", fake_fused, raising=False)
    monkeypatch.setattr(imc.kernels, "sigclip_mask", old_mask_path)

    out = c.combine("median", rejectors=SigClip(sigma=3.0), full=False)

    assert called == {"shape": arr.shape, "combine": "median"}
    assert out.shape == arr.shape[1:]
    assert c.mask is None
    assert c.last_reject is None


def test_combiner_inline_rejectors_full_true_records_state():
    rng = np.random.default_rng(20250311)
    arr = rng.normal(0, 1, (20, 3, 3)).astype(np.float32)
    arr[0] = 1e3
    c = Combiner(arr)

    out = c.combine("mean", rejectors=[MinMaxClip(n_min=0, n_max=1)], full=True)
    expected = Combiner(arr).reject(MinMaxClip(n_min=0, n_max=1)).combine("mean")

    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)
    assert c.mask is not None
    assert c.last_reject is not None
    assert len(c.reject_history) == 1


def test_combiner_reject_diagnostics_full_records_stage_local_sample_flags():
    arr = np.array([1.0, 2.0, 100.0, 4.0, 5.0], dtype=np.float32).reshape(5, 1, 1)
    c = Combiner(arr).threshold(-np.inf, 10.0)

    c.reject(MinMaxClip(n_min=2, n_max=0), diagnostics="full")

    assert c.sample_flags.shape == arr.shape
    assert c.sample_flags.dtype == np.uint8
    # The thresholded outlier was already unavailable when this rejection
    # stage began; the low-side minmax rejection is the only algorithm reject.
    assert c.sample_flags[:, 0, 0].tolist() == [8, 0, 32, 0, 0]


def test_combiner_diagnostic_flags_available():
    arr = np.arange(5, dtype=np.float32).reshape(5, 1, 1)
    c = Combiner(arr).reject(MinMaxClip(n_min=1, n_max=0), diagnostics="full")

    assert c.sample_flags is not None
    assert c.output_flags is not None
    assert not hasattr(c, "sample_\x63ode")
    assert not hasattr(c, "last_sample_\x63ode")
    assert not hasattr(c, "rej\x63ode")


def test_combiner_full_sample_flags_mark_reverted_rejections():
    arr = np.array([0.0, 1.0, 100.0], dtype=np.float32).reshape(3, 1, 1)

    c_nkeep = Combiner(arr).reject(
        SigClip(
            sigma=(10.0, 0.5),
            maxiters=1,
            ddof=0,
            nkeep=3,
            cenfunc="median",
            clip_cen="median",
            revert_on_nkeep=True,
        ),
        diagnostics="full",
    )
    assert not c_nkeep.mask_rej.any()
    assert c_nkeep.output_flags[0, 0] & 4
    assert c_nkeep.sample_flags[:, 0, 0].tolist() == [0, 0, 64]

    c_maxrej = Combiner(arr).reject(
        SigClip(
            sigma=(10.0, 0.5),
            maxiters=1,
            ddof=0,
            nkeep=0,
            maxrej=0,
            cenfunc="median",
            clip_cen="median",
            revert_on_nkeep=True,
        ),
        diagnostics="full",
    )
    assert not c_maxrej.mask_rej.any()
    assert c_maxrej.output_flags[0, 0] & 8
    assert c_maxrej.sample_flags[:, 0, 0].tolist() == [0, 0, 128]


def test_combiner_inline_pipeline_chains_thresholds_and_rejectors_without_mutation():
    arr = np.array(
        [1.0, 2.0, 1000.0, 4.0, 5.0, -100.0, 6.0],
        dtype=np.float32,
    ).reshape(7, 1, 1)
    c = Combiner(arr)

    out = c.combine(
        "mean",
        thresholds=[(0.0, np.inf), (-np.inf, 100.0)],
        rejectors=[MinMaxClip(n_min=0, n_max=1), SigClip(sigma=3.0)],
        full=False,
    )
    expected = (
        Combiner(arr)
        .threshold(0.0, np.inf)
        .threshold(-np.inf, 100.0)
        .reject(MinMaxClip(n_min=0, n_max=1))
        .reject(SigClip(sigma=3.0))
        .combine("mean")
    )

    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)
    assert c.mask is None
    assert c.mask_thresh is None
    assert c.last_reject is None


def test_combiner_inline_rejectors_full_false_falls_back_for_non_fused_combine():
    arr = np.array([1.0, 2.0, 100.0, 4.0, 5.0], dtype=np.float32).reshape(5, 1, 1)

    out = Combiner(arr).combine(
        "sum",
        rejectors=SigClip(sigma=3.0, maxiters=5, nkeep=1),
        full=False,
    )
    expected = (
        Combiner(arr).reject(SigClip(sigma=3.0, maxiters=5, nkeep=1)).combine("sum")
    )

    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)


def test_variance_via_combiner_matches_final_valid_values():
    arr = np.array([1.0, 2.0, 100.0, 4.0, 5.0], dtype=np.float32).reshape(5, 1, 1)
    c = Combiner(arr).reject(MinMaxClip(n_min=0, n_max=1))

    out_method = c.variance(ddof=1)
    out_combine = c.combine("variance", ddof=1)
    arr_eff = np.where(c.mask, np.nan, c.arr)
    expected = np.nanvar(arr_eff, axis=0, ddof=1)

    np.testing.assert_allclose(out_method, expected, rtol=1e-5)
    np.testing.assert_allclose(out_combine, expected, rtol=1e-5)


def test_combiner_variance_can_return_mean_from_final_valid_values():
    arr = np.array([1.0, 2.0, 100.0, 4.0, 5.0], dtype=np.float32).reshape(5, 1, 1)
    c = Combiner(arr).reject(MinMaxClip(n_min=0, n_max=1))

    var, mean = c.variance(ddof=1, return_mean=True)
    arr_eff = np.where(c.mask, np.nan, c.arr)

    np.testing.assert_allclose(var, np.nanvar(arr_eff, axis=0, ddof=1), rtol=1e-5)
    np.testing.assert_allclose(mean, np.nanmean(arr_eff, axis=0), rtol=1e-5)


def test_combiner_variance_owns_implementation(monkeypatch):
    arr = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
    called = {}

    def fake_variance(arr_eff, *, ddof=0, validate=True):
        called["ddof"] = ddof
        called["validate"] = validate
        return np.full(arr_eff.shape[1:], 7.0, dtype=np.float32)

    monkeypatch.setattr("imcombiners.kernels.variance", fake_variance)

    out = Combiner(arr).combine("var", ddof=2)

    assert called == {"ddof": 2, "validate": True}
    np.testing.assert_array_equal(out, np.full((2, 2), 7.0, dtype=np.float32))


@pytest.mark.parametrize("method", ["sum", "min", "max"])
def test_combiner_accepts_short_reduction_aliases(method):
    arr = np.arange(24, dtype=np.float32).reshape(4, 2, 3)
    expected = {
        "sum": np.sum(arr, axis=0),
        "min": np.min(arr, axis=0),
        "max": np.max(arr, axis=0),
    }[method]

    out = Combiner(arr).combine(method)

    np.testing.assert_allclose(out, expected)


def test_combiner_zero_scale_to_0th():
    arr = np.ones((3, 2, 2), dtype=np.float32) * 10
    arr[1] = 20
    arr[2] = 40

    out = (
        Combiner(arr)
        .zero_scale(
            zero=np.array([10.0, 20.0, 40.0]),
            scale=np.array([1.0, 2.0, 4.0]),
            zero_to_0th=True,
            scale_to_0th=True,
        )
        .combine("mean")
    )
    expected = np.mean(
        (arr - np.array([0.0, 10.0, 30.0]).reshape(-1, 1, 1))
        / np.array([1.0, 2.0, 4.0]).reshape(-1, 1, 1),
        axis=0,
    )

    np.testing.assert_allclose(out, expected)


def test_combiner_zero_scale_to_0th_defaults():
    arr = np.ones((3, 2, 2), dtype=np.float32) * 10
    arr[1] = 20
    arr[2] = 40

    out = (
        Combiner(arr)
        .zero_scale(
            zero=np.array([10.0, 20.0, 40.0]),
            scale=np.array([1.0, 2.0, 4.0]),
        )
        .combine("mean")
    )
    expected = np.mean(
        (arr - np.array([0.0, 10.0, 30.0]).reshape(-1, 1, 1))
        / np.array([1.0, 2.0, 4.0]).reshape(-1, 1, 1),
        axis=0,
    )

    np.testing.assert_allclose(out, expected)


def test_combiner_zero_scale_to_0th_defaults_preserve_scalars():
    arr = np.ones((3, 2, 2), dtype=np.float32) * 10

    out = Combiner(arr).zero_scale(zero=1.0, scale=3.0).combine("mean")

    np.testing.assert_allclose(out, 3.0)


def test_pclip_in_pipeline():
    rng = np.random.default_rng(20250311)
    arr = rng.normal(0, 1, (50, 2, 2)).astype(np.float32)
    out = Combiner(arr).reject(PClip(frac=0.1)).combine("median")
    assert out.shape == (2, 2)


def test_ccdclip_spec_handles_gain():
    rng = np.random.default_rng(20250311)
    arr = rng.normal(100, 10, (20, 4, 4)).astype(np.float32)
    arr[0] = 200
    # Two pipelines with different gain; both should run cleanly.
    out_g1 = Combiner(arr.copy()).reject(CcdClip(gain=1.0, rdnoise=5.0)).combine("mean")
    out_g2 = Combiner(arr.copy()).reject(CcdClip(gain=2.0, rdnoise=5.0)).combine("mean")
    assert out_g1.shape == out_g2.shape == (4, 4)


def test_ccdclip_apply_passes_gain_without_python_predivide(monkeypatch):
    arr = np.array([18.0, 20.0, 22.0, 200.0], dtype=np.float32).reshape(4, 1, 1)
    called = {}

    def fake_ccdclip(arr_in, **kwargs):
        called["same_object"] = arr_in is arr
        called["values"] = arr_in.copy()
        called["gain"] = kwargs["gain"]
        shape = arr_in.shape
        trailing = shape[1:]
        return (
            np.zeros(shape, dtype=bool),
            np.zeros(trailing, dtype=arr_in.dtype),
            np.zeros(trailing, dtype=arr_in.dtype),
            np.zeros(trailing, dtype=arr_in.dtype),
            np.zeros(trailing, dtype=np.uint8),
            np.zeros(trailing, dtype=np.uint8),
        )

    monkeypatch.setattr(imc.kernels, "ccdclip", fake_ccdclip)

    CcdClip(gain=2.0).apply(arr)

    assert called["same_object"] is True
    assert called["gain"] == 2.0
    np.testing.assert_array_equal(called["values"], arr)
