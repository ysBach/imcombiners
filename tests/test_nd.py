"""Tests for N-D (more than 3 axis) stack support."""

from __future__ import annotations

import imcombiners as imc
import numpy as np
import pytest
from imcombiners import kernels, ndcombine

# ---- helpers -------------------------------------------------------------------


def _nd_stack(shape=(5, 4, 3, 2), dtype=np.float32, rng_seed=20250311):
    """Return a small N-D float stack and its 3-D equivalent for comparison."""
    rng = np.random.default_rng(rng_seed)
    arr_nd = rng.normal(100.0, 5.0, shape).astype(dtype)
    # equivalent 3-D view the Rust kernels see internally
    arr_3d = arr_nd.reshape(shape[0], -1, 1)
    return arr_nd, arr_3d


def _numpy_lmedian(arr, axis=0):
    """Return lower median along axis 0."""
    if axis != 0:
        raise ValueError("test helper only supports axis=0")
    return np.sort(arr, axis=0)[(arr.shape[0] - 1) // 2]


# ---- validate_stack ------------------------------------------------------------


def test_validate_stack_reshapes_4d():
    from imcombiners._validation import validate_stack

    arr = np.ones((5, 4, 3, 2), dtype=np.float32)
    out = validate_stack(arr)
    assert out.shape == (5, 24, 1)


def test_validate_stack_rejects_1d():
    from imcombiners._validation import validate_stack

    with pytest.raises(ValueError, match="at least 2 dimensions"):
        validate_stack(np.ones(5, dtype=np.float32))


# ---- combine paths -------------------------------------------------------------


@pytest.mark.parametrize(
    "method",
    ["mean", "median", "lmedian", "summation", "minimum", "maximum", "variance"],
)
def test_ndcombine_nd_output_shape(method):
    arr_nd, arr_3d = _nd_stack()
    trailing = arr_nd.shape[1:]

    combine = {"summation": "sum", "minimum": "min", "maximum": "max"}.get(
        method, method
    )
    out_nd = ndcombine(arr_nd, combine=combine)
    out_3d = ndcombine(arr_3d, combine=combine).reshape(trailing)

    assert out_nd.shape == trailing
    np.testing.assert_allclose(out_nd, out_3d, rtol=1e-5)


@pytest.mark.parametrize(
    "dtype", [np.uint8, np.uint16, np.int16, np.int32, np.float32, np.float64]
)
@pytest.mark.parametrize(
    "method,numpy_func",
    [
        ("mean", np.mean),
        ("median", np.median),
        ("lmedian", _numpy_lmedian),
        ("summation", np.sum),
        ("minimum", np.min),
        ("maximum", np.max),
        ("variance", np.var),
    ],
)
def test_ndcombine_nd_matches_numpy_for_supported_dtypes(dtype, method, numpy_func):
    arr = (np.arange(5 * 4 * 3 * 2).reshape(5, 4, 3, 2) % 127).astype(dtype)

    combine = {"summation": "sum", "minimum": "min", "maximum": "max"}.get(
        method, method
    )
    out = ndcombine(arr, combine=combine)
    expected = numpy_func(arr.astype(out.dtype, copy=False), axis=0)

    assert out.shape == arr.shape[1:]
    if method == "lmedian" and np.issubdtype(dtype, np.integer):
        assert out.dtype == dtype
    np.testing.assert_allclose(out, expected)


@pytest.mark.parametrize(
    ("method", "numpy_func"),
    [
        ("sum", np.sum),
        ("min", np.min),
        ("max", np.max),
    ],
)
def test_ndcombine_uses_short_reduction_string_names(method, numpy_func):
    arr, _ = _nd_stack()

    out = ndcombine(arr, combine=method)

    np.testing.assert_allclose(out, numpy_func(arr, axis=0), rtol=1e-5)


@pytest.mark.parametrize("method", ["summation", "minimum", "maximum"])
def test_ndcombine_rejects_long_reduction_string_names(method):
    arr, _ = _nd_stack()

    with pytest.raises(ValueError, match="unknown combine method"):
        ndcombine(arr, combine=method)


def test_nanaverage_nd_output_shape():
    arr_nd, arr_3d = _nd_stack()
    trailing = arr_nd.shape[1:]
    N = arr_nd.shape[0]
    weights = np.ones(N, dtype=np.float64)

    out_nd = kernels.nanaverage(arr_nd, weights)
    out_3d = kernels.nanaverage(arr_3d, weights).reshape(trailing)

    assert out_nd.shape == trailing
    np.testing.assert_allclose(out_nd, out_3d, rtol=1e-5)


def test_nanaverage_nd_matches_numpy_with_nan():
    arr_nd, _ = _nd_stack(shape=(6, 4, 3, 2))
    arr_nd[0, 0, 0, 0] = np.nan
    weights = np.arange(1, arr_nd.shape[0] + 1, dtype=np.float64)

    out = kernels.nanaverage(arr_nd, weights)
    valid = np.isfinite(arr_nd)
    expected = np.nansum(arr_nd * weights.reshape(-1, 1, 1, 1), axis=0) / np.sum(
        np.where(valid, weights.reshape(-1, 1, 1, 1), 0.0), axis=0
    )

    np.testing.assert_allclose(out, expected, rtol=1e-5)


# ---- reject kernels ------------------------------------------------------------


@pytest.mark.parametrize(
    "reject_fn,kwargs",
    [
        ("sigclip", {"sigma": 3.0}),
        ("minmax", {"n_min": 1, "n_max": 1}),
        ("pclip", {"frac": 0.1}),
    ],
)
def test_reject_kernel_nd_output_shapes(reject_fn, kwargs):
    arr_nd, _ = _nd_stack()
    orig_shape = arr_nd.shape
    trailing = arr_nd.shape[1:]

    fn = getattr(kernels, reject_fn)
    mask_rej, std, low, upp, nit, output_flags = fn(arr_nd, **kwargs)

    assert mask_rej.shape == orig_shape
    assert low.shape == trailing
    assert upp.shape == trailing
    assert nit.shape == trailing
    assert output_flags.shape == trailing
    if reject_fn == "sigclip":
        assert std.shape == trailing
    else:
        assert std is None


def test_ndcombine_non_sigma_full_std_is_none():
    arr_nd, _ = _nd_stack()

    std = ndcombine(
        arr_nd,
        combine="median",
        reject="minmax",
        n_minmax=(1, 1),
        diagnostics="simple",
    )[3]

    assert std is None


def test_reject_kernel_nd_mask_must_match_arr():
    arr_nd, _ = _nd_stack(shape=(5, 4, 3, 2))
    bad_mask = np.zeros((5, 4, 3, 9), dtype=bool)  # incompatible trailing dims
    with pytest.raises(ValueError, match="mask shape"):
        kernels.sigclip(arr_nd, mask=bad_mask)


def test_reject_kernel_nd_mask_matching_shape_accepted():
    arr_nd, _ = _nd_stack(shape=(5, 4, 3, 2))
    mask = np.zeros(arr_nd.shape, dtype=bool)
    mask[0] = True  # mask first frame
    mask_rej, *_ = kernels.sigclip(arr_nd, mask=mask, sigma=3.0)
    assert mask_rej.shape == arr_nd.shape


@pytest.mark.parametrize(
    "reject_fn,kwargs",
    [
        ("sigclip", {"sigma": 2.5, "cenfunc": "mean", "clip_cen": "mean"}),
        ("ccdclip", {"sigma": 3.0, "rdnoise": 4.0, "snoise": 0.01}),
        ("linearclip", {"low": 5.0, "upp": 5.0, "maxiters": 2, "cenfunc": "lmed"}),
        ("minmax", {"n_min": 1, "n_max": 1}),
        ("pclip", {"frac": 0.2}),
    ],
)
def test_reject_kernel_nd_matches_3d_values(reject_fn, kwargs):
    arr_nd, arr_3d = _nd_stack(shape=(7, 4, 3, 2))
    arr_nd[0, 0, 0, 0] = 1_000.0
    arr_3d = arr_nd.reshape(arr_nd.shape[0], -1, 1)
    trailing = arr_nd.shape[1:]

    fn = getattr(kernels, reject_fn)
    out_nd = fn(arr_nd, **kwargs)
    out_3d = fn(arr_3d, **kwargs)

    np.testing.assert_array_equal(out_nd[0], out_3d[0].reshape(arr_nd.shape))
    for got, expected in zip(out_nd[1:], out_3d[1:], strict=True):
        if expected is None:
            assert got is None
        else:
            np.testing.assert_allclose(got, expected.reshape(trailing), rtol=1e-5)


# ---- Combiner ------------------------------------------------------------------


def test_combiner_nd_combine_output_shape():
    arr_nd, _ = _nd_stack()
    trailing = arr_nd.shape[1:]
    out = imc.Combiner(arr_nd).combine("median")
    assert out.shape == trailing


def test_combiner_nd_matches_3d_equivalent():
    arr_nd, arr_3d = _nd_stack()
    trailing = arr_nd.shape[1:]
    out_nd = imc.Combiner(arr_nd).combine("mean")
    out_3d = imc.Combiner(arr_3d).combine("mean").reshape(trailing)
    np.testing.assert_allclose(out_nd, out_3d, rtol=1e-5)


def test_combiner_nd_reject_last_reject_shapes():
    arr_nd, _ = _nd_stack()
    orig_shape = arr_nd.shape
    trailing = arr_nd.shape[1:]

    c = imc.Combiner(arr_nd).reject(imc.SigClip(sigma=3.0))

    assert c.mask_rej.shape == orig_shape
    assert c.low.shape == trailing
    assert c.nit.shape == trailing


def test_combiner_nd_mask_input():
    arr_nd, _ = _nd_stack(shape=(5, 4, 3))
    mask = np.zeros(arr_nd.shape, dtype=bool)
    mask[0] = True  # mask first frame everywhere
    out = imc.Combiner(arr_nd, mask=mask).combine("mean")
    assert out.shape == arr_nd.shape[1:]


def test_combiner_nd_repr_shows_shape():
    arr_nd, _ = _nd_stack(shape=(5, 4, 3, 2))
    r = repr(imc.Combiner(arr_nd))
    assert "shape=(4, 3, 2)" in r


def test_combiner_nd_copy_preserves_trailing_shape():
    arr_nd, _ = _nd_stack()
    c = imc.Combiner(arr_nd)
    c2 = c.copy()
    assert c2._trailing_shape == c._trailing_shape
    assert c2.combine("mean").shape == arr_nd.shape[1:]


def test_combiner_nd_copy_false_reuses_contiguous_float_workspace():
    arr_nd, _ = _nd_stack(shape=(5, 4, 3, 2), dtype=np.float32)

    c = imc.Combiner(arr_nd, copy=False)
    c.arr[0, 0, 0] = 1234.0

    assert np.shares_memory(c.arr, arr_nd)
    assert arr_nd.reshape(5, -1, 1)[0, 0, 0] == 1234.0


def test_combiner_nd_integer_copy_false_promotes_and_does_not_share():
    arr_nd = np.ones((5, 4, 3, 2), dtype=np.uint16)

    c = imc.Combiner(arr_nd, copy=False)
    c.arr[0, 0, 0] = 42.0

    assert c.arr.dtype == np.float32
    assert not np.shares_memory(c.arr, arr_nd)
    assert arr_nd.reshape(5, -1, 1)[0, 0, 0] == 1


# ---- ndcombine -----------------------------------------------------------------


def test_ndcombine_median_output_shape():
    arr_nd, _ = _nd_stack()
    out = ndcombine(arr_nd, combine="median")
    assert out.shape == arr_nd.shape[1:]


def test_ndcombine_nd_matches_3d_equivalent():
    arr_nd, arr_3d = _nd_stack()
    trailing = arr_nd.shape[1:]
    out_nd = ndcombine(arr_nd, combine="mean")
    out_3d = ndcombine(arr_3d, combine="mean").reshape(trailing)
    np.testing.assert_allclose(out_nd, out_3d, rtol=1e-5)


def test_ndcombine_nd_full_output_shapes():
    arr_nd, _ = _nd_stack()
    orig_shape = arr_nd.shape
    trailing = arr_nd.shape[1:]

    out, mask_rej, mask_thresh, std, low, upp, nit, output_flags = ndcombine(
        arr_nd,
        combine="median",
        reject="sigclip",
        sigma=3.0,
        diagnostics="simple",
    )

    assert out.shape == trailing
    assert mask_rej.shape == orig_shape
    assert mask_thresh is None
    assert low.shape == trailing
    assert nit.shape == trailing
    assert output_flags.shape == trailing
    assert std.shape == trailing


def test_ndcombine_nd_mask_input():
    arr_nd, _ = _nd_stack(shape=(5, 4, 3))
    mask = np.zeros(arr_nd.shape, dtype=bool)
    mask[0] = True
    out = ndcombine(arr_nd, mask=mask, combine="mean")
    assert out.shape == arr_nd.shape[1:]


def test_ndcombine_nd_lmedian_integer_dtype():
    arr_nd = np.arange(60, dtype=np.int16).reshape(5, 4, 3)
    out = ndcombine(arr_nd, combine="lmedian")
    assert out.shape == (4, 3)
    assert out.dtype == np.int16


def test_ndcombine_nd_zero_scale_string_and_callable_match_numpy():
    arr_nd, _ = _nd_stack(shape=(5, 4, 3, 2))

    out = ndcombine(arr_nd, combine="mean", zero="median", scale=np.mean)
    zero = np.median(arr_nd, axis=(1, 2, 3)).reshape(-1, 1, 1, 1)
    scale = np.mean(arr_nd, axis=(1, 2, 3)).reshape(-1, 1, 1, 1)
    zero = zero - zero[0]
    scale = scale / scale[0]
    expected = np.mean((arr_nd - zero) / scale, axis=0)

    np.testing.assert_allclose(out, expected, rtol=1e-5)


def test_combiner_nd_zero_scale_statistics_match_ndcombine():
    arr_nd, _ = _nd_stack(shape=(5, 4, 3, 2))

    out_combiner = (
        imc.Combiner(arr_nd).zero_scale(zero="median", scale=np.mean).combine("mean")
    )
    out_function = ndcombine(arr_nd, combine="mean", zero="median", scale=np.mean)

    np.testing.assert_allclose(out_combiner, out_function, rtol=1e-5)


def test_ndcombine_validate_false_accepts_nd_pure_stack_reduction():
    arr_nd, _ = _nd_stack(shape=(5, 4, 3, 2))

    out = ndcombine(arr_nd, combine="mean", validate=False)

    np.testing.assert_allclose(out, np.mean(arr_nd, axis=0), rtol=1e-5)


def test_3d_mean_and_median_match_ccdproc_when_available():
    ccdproc = pytest.importorskip("ccdproc")
    astropy_units = pytest.importorskip("astropy.units")
    ccddata = pytest.importorskip("astropy.nddata").CCDData
    arr, _ = _nd_stack(shape=(5, 4, 3), dtype=np.float32)
    ccds = [ccddata(frame, unit=astropy_units.adu) for frame in arr]

    combiner = ccdproc.Combiner(ccds)
    np.testing.assert_allclose(
        ndcombine(arr, combine="mean"),
        combiner.average_combine().data,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        ndcombine(arr, combine="median"),
        combiner.median_combine().data,
        rtol=1e-5,
    )
