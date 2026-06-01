"""Opt-in performance regression checks."""

from __future__ import annotations

import os
import statistics
import time

import imcombiners as imc
import numpy as np
import pytest
from imcombiners._validation import mask_as_nan

RUN_PERF = os.environ.get("IMCOMBINERS_PERF_TEST") == "1"


def _time_ms(func, repeats: int = 7) -> float:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = func()
        if result.size == 0:
            raise AssertionError("benchmark operation returned empty result")
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def _stack() -> np.ndarray:
    rng = np.random.default_rng(20260525)
    arr = rng.normal(1000.0, 30.0, (7, 256, 256)).astype(np.float32)
    arr[0, ::32, ::32] += 800.0
    arr[1, 16::32, 16::32] -= 500.0
    return arr


def _combine_masked(arr: np.ndarray, mask: np.ndarray, combine: str) -> np.ndarray:
    import reducers as rd

    masked = mask_as_nan(arr, mask)
    if combine == "mean":
        return rd.nanmean(masked, axis=0, validate=False)
    if combine == "median":
        return rd.nanmedian(masked, axis=0, validate=False)
    raise AssertionError(f"unsupported combine method in test: {combine}")


def _fallback(arr: np.ndarray, reject: str, combine: str) -> np.ndarray:
    if reject == "sigclip":
        mask = imc.kernels.sigclip_mask(
            arr,
            sigma=(2.0, 2.0),
            maxiters=5,
            nkeep=1,
            cenfunc="median",
            revert_on_nkeep=True,
            validate=False,
        )
    elif reject == "ccdclip":
        arr_gc = arr / 2.0
        mask = imc.kernels.ccdclip_mask(
            arr_gc,
            sigma=(2.0, 2.0),
            maxiters=5,
            nkeep=1,
            cenfunc="median",
            revert_on_nkeep=True,
            rdnoise=5.0,
            validate=False,
        )
    elif reject == "minmax":
        mask = imc.kernels.minmax_mask(arr, n_min=1, n_max=1, validate=False)
    elif reject == "pclip":
        mask = imc.kernels.pclip_mask(
            arr, frac=0.25, sigma=(3.0, 3.0), nkeep=1, validate=False
        )
    else:
        raise AssertionError(f"unsupported reject method in test: {reject}")
    return _combine_masked(arr, mask, combine)


def _fused(arr: np.ndarray, reject: str, combine: str) -> np.ndarray:
    kwargs = {
        "combine": combine,
        "reject": reject,
        "full": False,
        "validate": False,
    }
    if reject == "sigclip":
        kwargs.update(
            sigma=(2.0, 2.0),
            maxiters=5,
            nkeep=1,
            cenfunc="median",
            revert_on_nkeep=True,
        )
    elif reject == "ccdclip":
        kwargs.update(
            sigma=(2.0, 2.0),
            maxiters=5,
            nkeep=1,
            cenfunc="median",
            revert_on_nkeep=True,
            rdnoise=5.0,
            gain=2.0,
        )
    elif reject == "minmax":
        kwargs["n_minmax"] = (1, 1)
    elif reject == "pclip":
        kwargs["pclip"] = 0.25
    else:
        raise AssertionError(f"unsupported reject method in test: {reject}")
    return imc.ndcombine(arr, **kwargs)


@pytest.mark.performance
@pytest.mark.skipif(
    not RUN_PERF,
    reason="set IMCOMBINERS_PERF_TEST=1 to run performance regression checks",
)
@pytest.mark.parametrize("reject", ["sigclip", "ccdclip", "minmax", "pclip"])
@pytest.mark.parametrize("combine", ["mean", "median"])
def test_fused_reject_combine_beats_mask_fallback(reject, combine):
    arr = _stack()

    np.testing.assert_allclose(
        _fused(arr, reject, combine),
        _fallback(arr, reject, combine),
        rtol=1e-6,
        atol=1e-6,
        equal_nan=True,
    )

    fused_ms = _time_ms(lambda: _fused(arr, reject, combine))
    fallback_ms = _time_ms(lambda: _fallback(arr, reject, combine))
    speedup = fallback_ms / fused_ms

    assert speedup >= 1.15, (
        f"{reject} fused path too close to fallback: "
        f"fused={fused_ms:.3f} ms fallback={fallback_ms:.3f} ms "
        f"speedup={speedup:.2f}x"
    )
