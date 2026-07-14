"""Rejection-kernel unit tests."""

from __future__ import annotations

import numpy as np
import pytest
from imcombiners import Combiner, Rejector, SigClip, kernels, ndcombine


def test_minmax_basic():
    arr = np.arange(10, dtype=np.float32).reshape(10, 1, 1)
    mask, std, low, upp, nit, output_flags = kernels.minmax(arr, n_min=2, n_max=3)
    rej = mask[:, 0, 0]
    assert rej.tolist() == [
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
    ]
    assert low[0, 0] == 2
    assert upp[0, 0] == 6
    assert nit[0, 0] == 1
    assert std is None


def test_minmax_uses_min_max_count_names():
    arr = np.arange(10, dtype=np.float32).reshape(10, 1, 1)
    mask, std, low, upp, nit, output_flags = kernels.minmax(arr, n_min=2, n_max=3)
    rej = mask[:, 0, 0]

    assert rej.tolist() == [
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
    ]
    assert low[0, 0] == 2
    assert upp[0, 0] == 6
    assert nit[0, 0] == 1
    assert output_flags[0, 0] == 0
    assert std is None


def test_minmax_rejects_old_low_high_keyword_names():
    arr = np.arange(10, dtype=np.float32).reshape(10, 1, 1)

    with pytest.raises(TypeError, match="n_low"):
        kernels.minmax(arr, n_low=1)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="n_high"):
        kernels.minmax(arr, n_high=1)  # type: ignore[call-arg]


def test_linearclip_kernel_is_rust_backed_and_rejects_affine_bounds():
    arr = np.array([1.0, 10.0, 11.0, 12.0, 30.0], dtype=np.float32).reshape(5, 1, 1)

    mask, std, low, upp, nit, output_flags = kernels.linearclip(
        arr,
        low_scale=1.0,
        low=1.0,
        upp_scale=1.0,
        upp=2.0,
    )

    assert hasattr(kernels._core, "linearclip")
    assert mask[:, 0, 0].tolist() == [True, False, False, False, True]
    assert low[0, 0] == 10.0
    assert upp[0, 0] == 13.0
    assert nit[0, 0] == 1
    assert output_flags[0, 0] == 0
    assert std is None


def test_linearclip_kernel_default_is_noop():
    arr = np.array([1.0, 10.0, 11.0, 12.0, 30.0], dtype=np.float32).reshape(5, 1, 1)

    mask, std, low, upp, nit, output_flags = kernels.linearclip(arr)

    assert not mask.any()
    assert np.isnan(low[0, 0])
    assert np.isnan(upp[0, 0])
    assert nit[0, 0] == 0
    assert output_flags[0, 0] == 0
    assert std is None


def test_linearclip_kernel_masks_nonfinite_without_counting_input_mask():
    arr = np.array([np.nan, 10.0, 11.0, 12.0, 100.0], dtype=np.float32).reshape(5, 1, 1)
    input_mask = np.zeros_like(arr, dtype=bool)
    input_mask[4] = True

    mask, std, low, upp, nit, output_flags = kernels.linearclip(
        arr,
        mask=input_mask,
        low_scale=1.0,
        low=1.0,
        upp_scale=1.0,
        upp=2.0,
    )

    assert mask[:, 0, 0].tolist() == [True, False, False, False, False]
    assert low[0, 0] == 10.0
    assert upp[0, 0] == 13.0
    assert nit[0, 0] == 1
    assert output_flags[0, 0] & 1
    assert std is None


def test_linearclip_kernel_accepts_lower_median_center():
    arr = np.array([8.0, 10.0, 10.6, 40.0], dtype=np.float32).reshape(4, 1, 1)

    mask, std, low, upp, nit, output_flags = kernels.linearclip(
        arr,
        low_scale=1.0,
        low=1.0,
        upp_scale=1.0,
        upp=0.4,
        cenfunc="lmedian",
    )

    assert mask[:, 0, 0].tolist() == [True, False, True, True]
    assert low[0, 0] == 9.0
    assert upp[0, 0] == 10.4
    assert nit[0, 0] == 1
    assert output_flags[0, 0] == 0
    assert std is None


def test_linearclip_kernel_iterates_until_stable():
    arr = np.array([0.0, 10.0, 11.0, 12.0, 13.0, 100.0], dtype=np.float32).reshape(
        6, 1, 1
    )

    one_pass, *_ = kernels.linearclip(
        arr,
        low_scale=1.0,
        low=10.0,
        upp_scale=1.0,
        upp=0.75,
        maxiters=1,
        cenfunc="median",
    )
    mask, std, low, upp, nit, output_flags = kernels.linearclip(
        arr,
        low_scale=1.0,
        low=10.0,
        upp_scale=1.0,
        upp=0.75,
        maxiters=5,
        cenfunc="median",
    )

    assert one_pass[:, 0, 0].tolist() == [True, False, False, False, True, True]
    assert mask[:, 0, 0].tolist() == [True, False, False, True, True, True]
    assert low[0, 0] == 0.5
    assert upp[0, 0] == 11.25
    assert nit[0, 0] == 3
    assert output_flags[0, 0] == 0
    assert std is None


def test_sigclip_returns_pixel_std_diagnostic():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32).reshape(3, 1, 1)

    mask, std, low, upp, nit, output_flags = kernels.sigclip(
        arr,
        sigma=10.0,
        maxiters=5,
        ddof=0,
        nkeep=1,
        cenfunc="median",
        clip_cen="median",
    )

    assert not mask.any()
    assert (
        low.shape == upp.shape == nit.shape == output_flags.shape == std.shape == (1, 1)
    )
    np.testing.assert_allclose(std[0, 0], np.std([1.0, 2.0, 3.0], ddof=0))


def test_sigclip_mad_stdfunc_uses_scaled_median_absolute_deviation():
    arr = np.array([8.0, 9.0, 10.0, 11.0, 12.0, 100.0], dtype=np.float32).reshape(
        6, 1, 1
    )

    mask, std, low, upp, nit, output_flags = kernels.sigclip(
        arr,
        sigma=(10.0, 3.0),
        maxiters=1,
        ddof=0,
        nkeep=0,
        cenfunc="median",
        clip_cen="median",
        stdfunc="mad",
    )

    mad_sigma = 1.4826 * np.median(np.abs(arr[:, 0, 0] - np.median(arr[:, 0, 0])))
    assert mask[:, 0, 0].tolist() == [False, False, False, False, False, True]
    np.testing.assert_allclose(std[0, 0], mad_sigma, rtol=1e-6)
    np.testing.assert_allclose(low[0, 0], 8.0)
    np.testing.assert_allclose(upp[0, 0], 12.0)
    assert nit[0, 0] == 2
    assert output_flags[0, 0] == 2


def test_sigclip_mad_stdfunc_ignores_ddof():
    arr = np.array([8.0, 9.0, 10.0, 11.0, 12.0, 100.0], dtype=np.float32).reshape(
        6, 1, 1
    )

    _, std, *_ = kernels.sigclip(
        arr,
        sigma=100.0,
        maxiters=1,
        ddof=1,
        nkeep=0,
        cenfunc="median",
        clip_cen="median",
        stdfunc="mad",
    )

    base = 1.4826 * np.median(np.abs(arr[:, 0, 0] - np.median(arr[:, 0, 0])))
    np.testing.assert_allclose(std[0, 0], base, rtol=1e-6)


def test_sigclip_rejects_unknown_stdfunc():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32).reshape(3, 1, 1)

    with pytest.raises(ValueError, match="unknown stdfunc"):
        kernels.sigclip(arr, stdfunc="biweight")


def test_sigclip_rejects_negative_sigma_threshold():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32).reshape(3, 1, 1)

    with pytest.raises(ValueError, match="sigma thresholds"):
        kernels.sigclip(arr, sigma=(-1.0, 3.0))


def test_ccdclip_rejects_negative_sigma_threshold():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32).reshape(3, 1, 1)

    with pytest.raises(ValueError, match="sigma thresholds"):
        kernels.ccdclip(arr, sigma=(3.0, -1.0))


def test_sigclip_combine_1d_accepts_mad_stdfunc():
    values = np.array([8.0, 9.0, 10.0, 11.0, 12.0, 100.0], dtype=np.float32)

    out = kernels.sigclip_combine_1d(
        values,
        combine="mean",
        sigma=(10.0, 3.0),
        maxiters=1,
        nkeep=0,
        cenfunc="median",
        clip_cen="median",
        stdfunc="mad",
    )

    np.testing.assert_allclose(out, 10.0)


def test_sigclip_rejector_accepts_mad_stdfunc():
    arr = np.array([8.0, 9.0, 10.0, 11.0, 12.0, 100.0], dtype=np.float32).reshape(
        6, 1, 1
    )

    rejector = SigClip(
        sigma=(10.0, 3.0),
        maxiters=1,
        nkeep=0,
        cenfunc="median",
        clip_cen="median",
        stdfunc="mad",
    )

    mask, *_ = rejector.apply(arr)

    assert mask[:, 0, 0].tolist() == [False, False, False, False, False, True]


def test_sigclip_mask_1d_matches_single_column_sigclip_mask():
    values = np.array(
        [np.nan, 100.0, 101.0, 99.5, 500.0, 98.5, 100.5],
        dtype=np.float64,
    )
    kwargs = {
        "sigma": (3.0, 3.0),
        "maxiters": 5,
        "ddof": 0,
        "nkeep": 0,
        "cenfunc": "median",
        "clip_cen": None,
    }

    got = kernels.sigclip_mask_1d(values, **kwargs)
    expected = kernels.sigclip_mask(values[:, None], **kwargs)[:, 0]

    assert got.shape == values.shape
    np.testing.assert_array_equal(got, expected)


def test_sigclip_lower_median_center_differs_from_standard_median_for_even_stack():
    arr = np.array([8.0, 10.0, 10.6, 40.0], dtype=np.float32).reshape(4, 1, 1)

    median_mask, *_ = kernels.sigclip(
        arr,
        sigma=(10.0, 0.021),
        maxiters=1,
        ddof=0,
        nkeep=1,
        cenfunc="median",
        clip_cen="median",
    )
    lmedian_mask, *_ = kernels.sigclip(
        arr,
        sigma=(10.0, 0.021),
        maxiters=1,
        ddof=0,
        nkeep=1,
        cenfunc="lmedian",
        clip_cen="lmedian",
    )
    lmed_mask, *_ = kernels.sigclip(
        arr,
        sigma=(10.0, 0.021),
        maxiters=1,
        ddof=0,
        nkeep=1,
        cenfunc="lmed",
        clip_cen="lmed",
    )

    assert median_mask[:, 0, 0].tolist() == [False, False, False, True]
    assert lmedian_mask[:, 0, 0].tolist() == [False, False, True, True]
    np.testing.assert_array_equal(lmed_mask, lmedian_mask)


def test_ccdclip_returns_noise_model_std_diagnostic():
    arr = np.array([9.0, 10.0, 11.0], dtype=np.float32).reshape(3, 1, 1)

    _, std, *_ = kernels.ccdclip(
        arr,
        sigma=10.0,
        maxiters=5,
        rdnoise=3.0,
        snoise=0.0,
        scale_ref=1.0,
        zero_ref=0.0,
        cenfunc="median",
        clip_cen="median",
    )

    np.testing.assert_allclose(std[0, 0], np.sqrt(10.0 + 3.0**2))


def test_ccdclip_accepts_lower_median_center_alias():
    arr = np.array([8.0, 10.0, 10.6, 40.0], dtype=np.float32).reshape(4, 1, 1)

    lmedian_mask, *_ = kernels.ccdclip(
        arr,
        sigma=(10.0, 0.25),
        maxiters=1,
        nkeep=1,
        cenfunc="lmedian",
        clip_cen="lmedian",
        rdnoise=0.1,
        gain=1.0,
    )
    lmed_mask, *_ = kernels.ccdclip(
        arr,
        sigma=(10.0, 0.25),
        maxiters=1,
        nkeep=1,
        cenfunc="lmed",
        clip_cen="lmed",
        rdnoise=0.1,
        gain=1.0,
    )

    np.testing.assert_array_equal(lmed_mask, lmedian_mask)


def test_ccdclip_gain_matches_explicit_gain_corrected_input():
    arr = np.array([18.0, 20.0, 22.0, 200.0], dtype=np.float32).reshape(4, 1, 1)
    kwargs = {
        "sigma": 1.0,
        "maxiters": 3,
        "rdnoise": 3.0,
        "snoise": 0.0,
        "scale_ref": 1.0,
        "zero_ref": 0.0,
        "cenfunc": "median",
        "clip_cen": "median",
        "nkeep": 1,
    }

    got = kernels.ccdclip(arr, gain=2.0, **kwargs)
    expected = kernels.ccdclip(arr / 2.0, **kwargs)

    for got_item, expected_item in zip(got, expected, strict=True):
        np.testing.assert_allclose(got_item, expected_item)


def test_ccdclip_mask_gain_matches_explicit_gain_corrected_input():
    arr = np.array([18.0, 20.0, 22.0, 200.0], dtype=np.float32).reshape(4, 1, 1)
    kwargs = {
        "sigma": 1.0,
        "maxiters": 3,
        "rdnoise": 3.0,
        "cenfunc": "median",
        "clip_cen": "median",
        "nkeep": 1,
    }

    got = kernels.ccdclip_mask(arr, gain=2.0, **kwargs)
    expected = kernels.ccdclip_mask(arr / 2.0, **kwargs)

    np.testing.assert_array_equal(got, expected)


def test_grow_mask_radius_one_uses_euclidean_neighbors():
    mask = np.zeros((1, 3, 3), dtype=bool)
    mask[0, 1, 1] = True

    grown = kernels.grow_mask(mask, 1)

    expected = np.array(
        [
            [
                [False, True, False],
                [True, True, True],
                [False, True, False],
            ]
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(grown, expected)


def test_grow_mask_sqrt_two_includes_diagonals():
    mask = np.zeros((1, 3, 3), dtype=bool)
    mask[0, 1, 1] = True

    grown = kernels.grow_mask(mask, np.sqrt(2))

    np.testing.assert_array_equal(grown, np.ones((1, 3, 3), dtype=bool))


def test_grow_mask_preserves_nd_spatial_geometry():
    mask = np.zeros((1, 2, 3, 3), dtype=bool)
    mask[0, 0, 1, 1] = True

    grown = kernels.grow_mask(mask, 1)

    expected = np.zeros_like(mask)
    expected[0, 0, 1, 1] = True
    expected[0, 0, 0, 1] = True
    expected[0, 0, 2, 1] = True
    expected[0, 0, 1, 0] = True
    expected[0, 0, 1, 2] = True
    expected[0, 1, 1, 1] = True
    np.testing.assert_array_equal(grown, expected)


class CenterRejector(Rejector):
    def apply(self, arr, mask=None, *, validate=True):
        mask_rej = np.zeros_like(arr, dtype=bool)
        mask_rej[0, 1, 1] = True
        low = np.zeros(arr.shape[1:], dtype=arr.dtype)
        upp = np.ones(arr.shape[1:], dtype=arr.dtype)
        nit = np.ones(arr.shape[1:], dtype=np.uint8)
        output_flags = np.zeros(arr.shape[1:], dtype=np.uint8)
        std = np.full(arr.shape[1:], np.nan, dtype=arr.dtype)
        return mask_rej, std, low, upp, nit, output_flags


def test_combiner_reject_grow_dilates_only_rejection_mask():
    arr = np.ones((3, 3, 3), dtype=np.float32)
    arr[0, 0, 0] = 10.0
    input_mask = np.zeros_like(arr, dtype=bool)
    input_mask[1, 2, 2] = True

    c = Combiner(arr, mask=input_mask).reject(CenterRejector(), grow=1)

    expected_rej = np.zeros_like(arr, dtype=bool)
    expected_rej[0, 1, 1] = True
    expected_rej[0, 0, 1] = True
    expected_rej[0, 2, 1] = True
    expected_rej[0, 1, 0] = True
    expected_rej[0, 1, 2] = True
    np.testing.assert_array_equal(c.mask_rej, expected_rej)
    assert c.output_flags[1, 1] == 0
    assert c.output_flags[0, 1] & 16
    assert c.mask[1, 2, 2]
    assert not c.mask[1, 2, 1]


def test_rejector_object_can_carry_grow_radius():
    arr = np.ones((10, 5, 5), dtype=np.float32)
    arr[0, 2, 2] = 100.0

    c = Combiner(arr).reject(SigClip(sigma=3.0, ddof=0, nkeep=0, grow=1))

    assert c.mask_rej[0, 2, 2]
    assert c.mask_rej[0, 1, 2]
    assert c.output_flags[1, 2] & 16


def test_ndcombine_grow_changes_only_neighbors_of_rejected_samples():
    arr = np.ones((10, 5, 5), dtype=np.float32)
    arr[0, 2, 2] = 100.0
    ordinary_series = np.linspace(1.4, 0.5, 10, dtype=np.float32)
    arr[:, 1, 2] = ordinary_series
    arr[:, 2, 1] = ordinary_series
    arr[:, 2, 3] = ordinary_series
    arr[:, 3, 2] = ordinary_series

    out_no_grow, mask_no_grow, *_ = ndcombine(
        arr,
        reject="sigclip",
        sigma=3.0,
        maxiters=5,
        ddof=0,
        nkeep=0,
        combine="mean",
        diagnostics="simple",
    )
    out_grow, mask_grow, _, _, _, _, _, code_grow = ndcombine(
        arr,
        reject="sigclip",
        sigma=3.0,
        maxiters=5,
        ddof=0,
        nkeep=0,
        combine="mean",
        grow=1,
        diagnostics="simple",
    )

    assert mask_no_grow[:, 2, 2].sum() == 1
    assert mask_no_grow[:, 1, 2].sum() == 0
    assert mask_grow[:, 1, 2].sum() == 1
    assert mask_grow[:, 1, 1].sum() == 0
    assert code_grow[2, 2] == 0
    assert code_grow[1, 2] & 16
    assert out_no_grow[1, 2] > out_grow[1, 2]
    np.testing.assert_allclose(out_grow[1, 2], ordinary_series[1:].mean(), rtol=1e-6)


def test_direct_rejection_kernel_grow_sets_code_bit():
    arr = np.ones((10, 5, 5), dtype=np.float32)
    arr[0, 2, 2] = 100.0

    mask_rej, _, _, _, _, output_flags = kernels.sigclip(
        arr,
        sigma=3.0,
        maxiters=5,
        ddof=0,
        nkeep=0,
        grow=1,
    )

    assert mask_rej[0, 2, 2]
    assert mask_rej[0, 1, 2]
    assert not (output_flags[2, 2] & 16)
    assert output_flags[1, 2] & 16


@pytest.mark.parametrize(
    "func,kwargs",
    [
        (kernels.sigclip, {"sigma": 3.0, "ddof": 0, "nkeep": 0}),
        (kernels.ccdclip, {"sigma": 3.0, "ddof": 0, "nkeep": 0, "rdnoise": 1.0}),
        (kernels.minmax, {"n_min": 0, "n_max": 1}),
        (kernels.pclip, {"frac": 0.4}),
    ],
)
def test_rejection_grow_none_and_zero_preserve_kernel_results(func, kwargs):
    rng = np.random.default_rng(20250311)
    arr = rng.normal(0.0, 1.0, (8, 4, 4)).astype(np.float32)
    arr[0, 2, 2] = 100.0

    out_default = func(arr, **kwargs)
    out_none = func(arr, grow=None, **kwargs)
    out_zero = func(arr, grow=0, **kwargs)

    for got, expected in zip(out_none, out_default, strict=True):
        np.testing.assert_array_equal(got, expected)
    for got, expected in zip(out_zero, out_default, strict=True):
        np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("grow", [-1, np.nan, np.inf])
def test_grow_rejects_invalid_radius(grow):
    mask = np.zeros((1, 2, 2), dtype=bool)

    with pytest.raises(ValueError, match="grow"):
        kernels.grow_mask(mask, grow)


def test_minmax_zero_rejection():
    arr = np.arange(10, dtype=np.float32).reshape(10, 1, 1)
    mask, _, _, _, _, output_flags = kernels.minmax(arr, n_min=0, n_max=0)
    assert mask.sum() == 0
    assert output_flags[0, 0] == 0


def test_minmax_one_sided_rejection_has_normal_code_without_premask():
    arr = np.arange(10, dtype=np.float32).reshape(10, 1, 1)
    mask, _, _, _, _, output_flags = kernels.minmax(arr, n_min=0, n_max=1)
    assert mask.sum() == 1
    assert not (output_flags[0, 0] & 1)


def test_minmax_fraction_equals_integer_equivalent():
    # n_min=0.1 with N=10 → int(10*0.1+0.001)=1, same as n_min=1.
    arr = np.arange(10, dtype=np.float32).reshape(10, 1, 1)
    mask_int, *_ = kernels.minmax(arr, n_min=1, n_max=1)
    mask_frac, *_ = kernels.minmax(arr, n_min=0.1, n_max=0.1)
    np.testing.assert_array_equal(mask_int, mask_frac)


def test_minmax_fraction_scales_with_n_finite():
    # With n_min=0.1 and N=20, the effective count is int(20*0.1+0.001)=2.
    # The Rust core then uses fraction=2/20=0.1 per pixel, so an unmasked
    # pixel with all 20 values present gets 2 low values rejected.
    arr = np.arange(20, dtype=np.float32).reshape(20, 1, 1)
    mask_frac, *_ = kernels.minmax(arr, n_min=0.1, n_max=0.1)
    assert mask_frac[:, 0, 0].sum() == 4  # 2 low + 2 high


def test_minmax_fraction_negative_raises():
    arr = np.ones((5, 1, 1), dtype=np.float32)
    import pytest

    with pytest.raises(ValueError, match="n_min"):
        kernels.minmax(arr, n_min=-0.1)


def test_pclip_uses_iraf_rank_offset_sigma():
    arr = np.array([970, 1002, 1004, 1006, 1008, 1010, 1012], dtype=np.float32)
    arr = arr.reshape(7, 1, 1)

    mask, _, low, upp, nit, _ = kernels.pclip(arr, frac=0.25)

    assert low[0, 0] == 1002
    assert upp[0, 0] == 1010
    assert nit[0, 0] == 1
    assert mask[:, 0, 0].tolist() == [
        True,
        False,
        False,
        False,
        False,
        False,
        True,
    ]


def test_pclip_rejects_tuple_fraction():
    arr = np.ones((5, 1, 1), dtype=np.float32)

    with pytest.raises(ValueError, match="frac"):
        kernels.pclip(arr, frac=(0.1, 0.2))


def test_pclip_rejects_zero_fraction():
    arr = np.ones((5, 1, 1), dtype=np.float32)

    with pytest.raises(ValueError, match="frac"):
        kernels.pclip(arr, frac=0.0)


def test_sigclip_gaussian_no_outliers():
    rng = np.random.default_rng(20250311)
    arr = rng.normal(0, 1, (50, 4, 4)).astype(np.float32)
    mask, *_ = kernels.sigclip(
        arr, sigma=(3.0, 3.0), maxiters=5, cenfunc="median", nkeep=1, maxrej=49
    )
    assert mask.mean() < 0.05


def test_sigclip_rejects_strong_outlier():
    rng = np.random.default_rng(20250311)
    arr = rng.normal(0, 1, (20, 3, 3)).astype(np.float32)
    arr[0] = 1e3
    mask, *_ = kernels.sigclip(
        arr, sigma=(3.0, 3.0), maxiters=10, cenfunc="median", nkeep=1, maxrej=19
    )
    assert mask[0].all()


def test_sigclip_low_upp_are_retained_extrema():
    arr = np.array([1.0, 2.0, 3.0, 100.0], dtype=np.float32).reshape(4, 1, 1)

    mask, _, low, upp, *_ = kernels.sigclip(
        arr,
        sigma=1.0,
        maxiters=2,
        cenfunc="median",
        clip_cen="median",
        nkeep=0,
    )

    kept = arr[~mask]
    assert low[0, 0] == np.min(kept)
    assert upp[0, 0] == np.max(kept)


def test_sigclip_nkeep_default_prevents_all_rejected_column():
    arr = np.array([0.0, 100.0], dtype=np.float32).reshape(2, 1, 1)

    default_mask, *_ = kernels.sigclip(
        arr,
        sigma=0.1,
        maxiters=1,
        ddof=0,
        cenfunc="median",
        clip_cen="median",
    )
    unbounded_mask, *_ = kernels.sigclip(
        arr,
        sigma=0.1,
        maxiters=1,
        ddof=0,
        nkeep=0,
        cenfunc="median",
        clip_cen="median",
    )

    assert default_mask[:, 0, 0].tolist() == [False, False]
    assert unbounded_mask[:, 0, 0].tolist() == [True, True]


def test_sigclip_recomputes_center_between_iterations():
    arr = np.array(
        [-4.8572235, -2.555665, 0.41809884, -0.5677696],
        dtype=np.float32,
    ).reshape(4, 1, 1)

    mask, *_ = kernels.sigclip(
        arr,
        sigma=1.1664820770253346,
        maxiters=5,
        ddof=0,
        nkeep=0,
        cenfunc="median",
        clip_cen="median",
    )

    assert mask[:, 0, 0].tolist() == [True, True, False, False]


def test_ccdclip_low_upp_are_retained_extrema():
    arr = np.array([100.0, 101.0, 102.0, 300.0], dtype=np.float32).reshape(4, 1, 1)

    mask, _, low, upp, *_ = kernels.ccdclip(
        arr,
        sigma=1.0,
        maxiters=2,
        rdnoise=1.0,
        nkeep=0,
    )

    kept = arr[~mask]
    assert low[0, 0] == np.min(kept)
    assert upp[0, 0] == np.max(kept)


def test_ccdclip_noise_model_honors_clip_cen():
    arr = np.array([1.0, 1.0, 5.0, 12.0], dtype=np.float32).reshape(4, 1, 1)

    mean_noise_mask, *_ = kernels.ccdclip(
        arr,
        sigma=1.0,
        maxiters=5,
        nkeep=0,
        rdnoise=0.0,
        snoise=0.0,
        scale_ref=1.0,
        zero_ref=0.0,
        cenfunc="median",
        clip_cen="mean",
    )
    median_noise_mask, *_ = kernels.ccdclip(
        arr,
        sigma=1.0,
        maxiters=5,
        nkeep=0,
        rdnoise=0.0,
        snoise=0.0,
        scale_ref=1.0,
        zero_ref=0.0,
        cenfunc="median",
        clip_cen="median",
    )

    assert mean_noise_mask[:, 0, 0].tolist() == [False, False, True, True]
    assert median_noise_mask[:, 0, 0].tolist() == [True, True, True, True]


def test_sigclip_plain_mode_keeps_cumulative_rejections():
    # clip_cen="mean" is explicit here: the assertion is sensitive to this
    # choice and would not hold for clip_cen=None (cenfunc center).
    col = np.array(
        [
            120.0,
            126.0,
            123.0,
            123.0,
            126.0,
            119.0,
            124.0,
            129.0,
            126.0,
            131.0,
            122.0,
            126.0,
            124.0,
            128.0,
            128.0,
            128.0,
            128.0,
            124.0,
            128.0,
            123.0,
            130.0,
            124.0,
            130.0,
            122.0,
            126.0,
            127.0,
            114.0,
            129.0,
            127.0,
            116.0,
            127.0,
        ],
        dtype=np.float32,
    ).reshape(31, 1, 1)

    mask, *_ = kernels.sigclip(
        col,
        sigma=(3.0, 3.0),
        maxiters=5,
        cenfunc="median",
        clip_cen="mean",
        ddof=0,
        nkeep=0,
        revert_on_nkeep=False,
    )

    assert mask[:, 0, 0].tolist() == [i == 26 for i in range(31)]


def test_sigclip_revert_on_nkeep_controls_full_iteration_rollback():
    arr = np.array([0.0, 1.0, 100.0], dtype=np.float32).reshape(3, 1, 1)

    reverted, _, _, _, _, code_reverted = kernels.sigclip(
        arr,
        sigma=(10.0, 0.5),
        maxiters=1,
        ddof=0,
        nkeep=3,
        cenfunc="median",
        clip_cen="median",
        revert_on_nkeep=True,
    )
    strict, _, _, _, _, code_strict = kernels.sigclip(
        arr,
        sigma=(10.0, 0.5),
        maxiters=1,
        ddof=0,
        nkeep=3,
        cenfunc="median",
        clip_cen="median",
        revert_on_nkeep=False,
    )

    assert not reverted[:, 0, 0].any()
    assert strict[:, 0, 0].tolist() == [False, False, True]
    assert code_reverted[0, 0] & 4
    assert not (code_strict[0, 0] & 4)


def test_sigclip_revert_on_nkeep_does_not_restore_closest_rejected_samples():
    arr = np.array([-0.1, 0.0, 1.0, 2.0], dtype=np.float32).reshape(4, 1, 1)

    mask, *_ = kernels.sigclip(
        arr,
        sigma=(0.1, 10.0),
        maxiters=1,
        ddof=0,
        nkeep=2,
        cenfunc="median",
        clip_cen="median",
        revert_on_nkeep=True,
    )

    assert mask[:, 0, 0].tolist() == [True, True, False, False]


def test_sigclip_revert_on_nkeep_never_unmasks_input_mask_or_nan():
    arr = np.array([np.nan, 0.0, 1.0, 2.0], dtype=np.float32).reshape(4, 1, 1)
    mask_in = np.zeros_like(arr, dtype=bool)
    mask_in[1, 0, 0] = True

    mask, *_ = kernels.sigclip(
        arr,
        mask=mask_in,
        sigma=(0.1, 10.0),
        maxiters=1,
        ddof=0,
        nkeep=2,
        cenfunc="median",
        clip_cen="median",
        revert_on_nkeep=True,
    )

    assert mask[0, 0, 0]
    assert mask[1, 0, 0]


def test_sigclip_rejects_old_restore_nkeep_keyword():
    arr = np.array([0.0, 1.0, 100.0], dtype=np.float32).reshape(3, 1, 1)

    with pytest.raises(TypeError, match="restore_nkeep"):
        kernels.sigclip(arr, restore_nkeep=True)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="restore_nkeep"):
        SigClip(restore_nkeep=True)  # type: ignore[call-arg]


def test_sigclip_clip_cen_none_uses_cenfunc():
    # clip_cen=None (default) uses the same center as cenfunc.
    # clip_cen="mean" uses the unmasked mean as the clip center instead,
    # which produces a wider spread and may reject a borderline pixel that
    # None / cenfunc-center would keep.
    col = np.array(
        [
            120.0,
            126.0,
            123.0,
            123.0,
            126.0,
            119.0,
            124.0,
            129.0,
            126.0,
            131.0,
            122.0,
            126.0,
            124.0,
            128.0,
            128.0,
            128.0,
            128.0,
            124.0,
            128.0,
            123.0,
            130.0,
            124.0,
            130.0,
            122.0,
            126.0,
            127.0,
            114.0,
            129.0,
            127.0,
            116.0,
            127.0,
        ],
        dtype=np.float32,
    ).reshape(31, 1, 1)

    mask_mean, *_ = kernels.sigclip(
        col,
        sigma=(3.0, 3.0),
        maxiters=5,
        cenfunc="median",
        clip_cen="mean",
        ddof=0,
        nkeep=0,
        revert_on_nkeep=False,
    )
    # clip_cen=None → uses cenfunc ("median") as the clip center
    mask_none, *_ = kernels.sigclip(
        col,
        sigma=(3.0, 3.0),
        maxiters=5,
        cenfunc="median",
        clip_cen=None,
        ddof=0,
        nkeep=0,
        revert_on_nkeep=False,
    )

    assert mask_mean[26, 0, 0]
    assert not mask_none[26, 0, 0]


def test_sigclip_float32_compares_unrounded_thresholds():
    col = np.array(
        [
            992.9125,
            1000.275,
            1034.5599,
            1012.77594,
            995.114,
            1003.24164,
            971.7945,
            974.2783,
            976.7365,
            965.33966,
            1017.0628,
            973.20807,
            1061.7831,
            1006.59283,
            1042.946,
            1106.5452,
            982.25836,
            1018.5643,
            1016.13336,
            989.6895,
            977.3514,
            1016.6553,
            1017.8271,
            986.9952,
            955.7002,
            913.0865,
            1003.0293,
            981.4976,
            990.4585,
            997.5548,
            999.18524,
        ],
        dtype=np.float32,
    ).reshape(31, 1, 1)

    mask, *_ = kernels.sigclip(
        col,
        sigma=(3.0, 3.0),
        maxiters=5,
        cenfunc="median",
        clip_cen="mean",
        ddof=0,
        nkeep=0,
        revert_on_nkeep=False,
    )

    assert mask[:, 0, 0].tolist() == [i in (15, 25) for i in range(31)]


def test_ccdclip_high_rdnoise_rejects_less():
    rng = np.random.default_rng(20250311)
    arr = rng.normal(100, 10, (20, 4, 4)).astype(np.float32)
    arr[0] = 200

    mask_low_noise, *_ = kernels.ccdclip(
        arr,
        maxiters=5,
        rdnoise=0.0,
        snoise=0.0,
        scale_ref=1.0,
        zero_ref=0.0,
        nkeep=1,
        maxrej=19,
    )
    mask_high_noise, *_ = kernels.ccdclip(
        arr,
        maxiters=5,
        rdnoise=1e6,
        snoise=0.0,
        scale_ref=1.0,
        zero_ref=0.0,
        nkeep=1,
        maxrej=19,
    )
    assert mask_high_noise.sum() <= mask_low_noise.sum()


def test_pclip_smooth_ramp_has_no_outliers_under_iraf_sigma():
    arr = np.arange(100, dtype=np.float32).reshape(100, 1, 1)
    mask, *_ = kernels.pclip(arr, frac=0.4)
    assert mask.sum() == 0


def test_pclip_full_window_sets_normal_code():
    arr = np.arange(10, dtype=np.float32).reshape(10, 1, 1)
    mask, _, _, _, _, output_flags = kernels.pclip(arr, frac=0.5)
    assert mask.sum() == 0
    assert output_flags[0, 0] == 0


def test_sigclip_default_is_plain_cumulative_mode():
    col = np.array(
        [
            120.0,
            126.0,
            123.0,
            123.0,
            126.0,
            119.0,
            124.0,
            129.0,
            126.0,
            131.0,
            122.0,
            126.0,
            124.0,
            128.0,
            128.0,
            128.0,
            128.0,
            124.0,
            128.0,
            123.0,
            130.0,
            124.0,
            130.0,
            122.0,
            126.0,
            127.0,
            114.0,
            129.0,
            127.0,
            116.0,
            127.0,
        ],
        dtype=np.float32,
    ).reshape(31, 1, 1)

    mask, _, _, _, _, output_flags = kernels.sigclip(
        col,
        sigma=(3.0, 3.0),
        maxiters=5,
        cenfunc="median",
        clip_cen="mean",
        ddof=0,
        nkeep=0,
    )

    assert mask[:, 0, 0].tolist() == [i == 26 for i in range(31)]
    assert output_flags[0, 0] == 0


def test_sigclip_output_flags_uses_zero_for_normal_and_bit1_for_maxiters():
    normal = np.array([1.0, 2.0, 3.0], dtype=np.float32).reshape(3, 1, 1)
    _, _, _, _, _, code_normal = kernels.sigclip(normal, sigma=100.0, maxiters=5)
    assert code_normal[0, 0] == 0

    changing = np.array([0.0, 100.0, 101.0], dtype=np.float32).reshape(3, 1, 1)
    mask, _, _, _, _, code_maxiters = kernels.sigclip(
        changing,
        sigma=0.1,
        maxiters=1,
        nkeep=0,
        maxrej=None,
        cenfunc="median",
        clip_cen="mean",
    )
    assert mask.sum() > 0
    assert code_maxiters[0, 0] & 2
    assert not (code_maxiters[0, 0] & 1)

    mask_zero, _, _, _, _, code_zero = kernels.sigclip(
        changing,
        sigma=0.1,
        maxiters=0,
        nkeep=0,
        maxrej=None,
        cenfunc="median",
        clip_cen="mean",
    )
    assert mask_zero.sum() == 0
    assert code_zero[0, 0] & 2


def test_output_flags_bit0_marks_premasked_values():
    arr = np.array([1.0, 2.0, 100.0, 4.0, 5.0], dtype=np.float32).reshape(5, 1, 1)
    m_in = np.zeros((5, 1, 1), dtype=bool)
    m_in[2] = True

    _, _, _, _, _, code_sig = kernels.sigclip(arr, mask=m_in, sigma=100.0)
    _, _, _, _, _, code_mm = kernels.minmax(arr, mask=m_in, n_min=0, n_max=0)
    _, _, _, _, _, code_pc = kernels.pclip(arr, mask=m_in, frac=0.5)

    assert code_sig[0, 0] & 1
    assert code_mm[0, 0] & 1
    assert code_pc[0, 0] & 1


def test_output_flags_bit0_is_per_output_and_can_come_from_previous_rejection():
    arr = np.array(
        [
            [[-100.0, 1.0]],
            [[1.0, 2.0]],
            [[2.0, 3.0]],
            [[3.0, 100.0]],
        ],
        dtype=np.float32,
    )

    mask_first, *_ = kernels.minmax(arr, n_min=1, n_max=0)
    _, _, _, _, _, code_second = kernels.sigclip(arr, mask=mask_first, sigma=100.0)

    assert mask_first[:, 0, 0].sum() == 1
    assert mask_first[:, 0, 1].sum() == 1
    assert (code_second[0] & 1).tolist() == [True, True]


def test_reject_respects_input_mask():
    arr = np.array([1.0, 2.0, 100.0, 4.0, 5.0], dtype=np.float32).reshape(5, 1, 1)
    m_in = np.zeros((5, 1, 1), dtype=bool)
    m_in[2] = True
    mask, *_ = kernels.sigclip(
        arr,
        mask=m_in,
        sigma=(3.0, 3.0),
        maxiters=5,
        cenfunc="median",
        nkeep=1,
        maxrej=4,
    )
    assert mask[2, 0, 0]


def test_ndcombine_with_sigclip_shim():
    rng = np.random.default_rng(20250311)
    arr = rng.normal(0, 1, (20, 4, 4)).astype(np.float32)
    arr[0] = 1e3
    out = ndcombine(arr, combine="mean", reject="sigclip", sigma=(3.0, 3.0))
    assert np.abs(out).max() < 5.0
