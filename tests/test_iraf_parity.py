"""Parity checks against temporary FITS files produced by IRAF imcombine.

When enabled, this module creates a temporary IRAF workspace, generates
deterministic inputs, runs IRAF's generated CL script there, and verifies the
matching ``imcombiners`` arguments against those temporary outputs.
"""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import imcombiners as imc
import numpy as np
import pytest

RUN_IRAF_PARITY = os.environ.get("IMC_RUN_IRAF_PARITY") == "1"
if RUN_IRAF_PARITY:
    from astropy.io import fits
else:
    fits = pytest.importorskip("astropy.io.fits")

REPO_ROOT = Path(__file__).resolve().parents[1]
IRAF_TEMPLATE_DIR = REPO_ROOT / "benchmarks" / "IRAF"
IRAF_SOURCE_ROOT = Path(os.environ.get("IMC_IRAF_ROOT", IRAF_TEMPLATE_DIR))
IRAF_ECL = Path(os.environ.get("IMC_IRAF_ECL", IRAF_TEMPLATE_DIR / "ecl.e"))
REQUIRED_DTYPES = {"uint16", "int16", "float32"}
REQUIRED_REJECTS = {"sigclip", "ccdclip", "pclip", "minmax"}
REQUIRED_COMBINES = {"mean", "median"}
REQUIRED_VARIANTS = {"baseline", "threshold", "input_bpm", "snoise", "scaled"}
IRAF_PARITY_REASON = (
    "set IMC_RUN_IRAF_PARITY=1 to generate temporary IRAF references and compare them"
)
_IRAF_TMP: tempfile.TemporaryDirectory[str] | None = None


def _prepare_iraf_workspace() -> Path:
    global _IRAF_TMP
    ecl = IRAF_ECL
    if not ecl.exists():
        legacy_ecl = IRAF_SOURCE_ROOT / "bin.macos64" / "ecl.e"
        ecl = legacy_ecl if legacy_ecl.exists() else ecl
    if not ecl.exists():
        raise FileNotFoundError(
            "IRAF ecl.e not found. Put it at benchmarks/IRAF/ecl.e or set IMC_IRAF_ECL."
        )

    _IRAF_TMP = tempfile.TemporaryDirectory(prefix="imc-iraf-parity-")
    atexit.register(_IRAF_TMP.cleanup)
    workspace = Path(_IRAF_TMP.name)

    sys.path.insert(0, str(IRAF_TEMPLATE_DIR / "scripts"))
    import make_inputs  # noqa: PLC0415

    make_inputs.generate(workspace, make_inputs.DEFAULT_SHAPE)

    env = os.environ.copy()
    env.update(
        {
            "iraf": str(IRAF_SOURCE_ROOT) + "/",
            "IRAFARCH": "macos64",
            "arch": ".macos64",
        }
    )
    result = subprocess.run(
        [str(ecl), "-f", str(workspace / "scripts" / "run_imcombine.cl")],
        cwd=workspace,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    cases = json.loads((workspace / "cases.json").read_text())["cases"]
    expected_outputs = {
        workspace / "outputs" / name
        for case in cases
        for name in (
            case["output"],
            *(case["diagnostics"][key] for key in ("nrejmask", "sigmas", "rejmask")),
        )
    }
    missing = sorted(str(path) for path in expected_outputs if not path.exists())
    if result.returncode != 0 or missing:
        raise RuntimeError(
            f"IRAF reference generation failed (exit {result.returncode}). "
            "Check IMC_IRAF_ROOT and IMC_IRAF_ECL.\n"
            f"Missing outputs: {', '.join(missing)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return workspace


if RUN_IRAF_PARITY:
    IRAF_DIR = _prepare_iraf_workspace()
else:
    IRAF_DIR = IRAF_TEMPLATE_DIR

MANIFEST = IRAF_DIR / "cases.json"


def _cases() -> list[dict[str, object]]:
    if not MANIFEST.exists():
        return []
    return json.loads(MANIFEST.read_text())["cases"]


def _load_stack(case: dict[str, object]) -> np.ndarray:
    arrays = []
    dtype = np.dtype(str(case["dtype"]))
    for name in case["inputs"]:
        with fits.open(IRAF_DIR / "inputs" / str(name), memmap=False) as hdul:
            arrays.append(np.asarray(hdul[0].data, dtype=dtype))
    return np.stack(arrays, axis=0)


def _load_input_mask(case: dict[str, object]) -> np.ndarray:
    arrays = []
    for name in case["input_masks"]:
        with fits.open(IRAF_DIR / "inputs" / str(name), memmap=False) as hdul:
            arrays.append(np.asarray(hdul[0].data) != 0)
    return np.stack(arrays, axis=0)


def _read_fits(path: str | Path) -> np.ndarray:
    with fits.open(path, memmap=False) as hdul:
        return np.asarray(hdul[0].data)


def _run_imcombiners(
    case: dict[str, object], stack: np.ndarray
) -> tuple[
    np.ndarray,
    np.ndarray,
    object,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    kwargs = dict(case["imcombiners"])
    mask_from = kwargs.pop("mask_from", None)
    if mask_from == "input_masks":
        kwargs["mask"] = _load_input_mask(case)
    elif mask_from is not None:
        raise ValueError(f"unknown mask source: {mask_from}")
    return imc.ndcombine(stack, **kwargs)


def _atol_for(dtype: np.dtype) -> float:
    dtype = np.dtype(dtype)
    return float(np.finfo(dtype).eps) if np.issubdtype(dtype, np.floating) else 0.0


def _atol_for_array(arr: np.ndarray) -> float:
    arr = np.asarray(arr)
    if not np.issubdtype(arr.dtype, np.floating):
        return 0.0
    finite = np.abs(arr[np.isfinite(arr)])
    if finite.size == 0:
        return _atol_for(arr.dtype)
    return float(2 * np.max(np.spacing(finite)))


def _load_iraf_rejection_mask(case: dict[str, object]) -> np.ndarray:
    diagnostics = case["diagnostics"]
    return np.asarray(
        _read_fits(IRAF_DIR / "outputs" / diagnostics["rejmask"]) != 0, dtype=bool
    )


def _threshold_mask(case: dict[str, object], stack: np.ndarray) -> np.ndarray:
    thresholds = case["imcombiners"].get("thresholds")
    if thresholds is None:
        return np.zeros_like(stack, dtype=bool)
    lower, upper = thresholds
    return (stack < lower) | (stack > upper)


def _pre_rejection_mask(case: dict[str, object], stack: np.ndarray) -> np.ndarray:
    mask = _threshold_mask(case, stack)
    if case["imcombiners"].get("mask_from") == "input_masks":
        mask = mask | _load_input_mask(case)
    return mask


def _diagnostic_stack(case: dict[str, object], stack: np.ndarray) -> np.ndarray:
    arr = stack.astype(np.float32, copy=False)
    kwargs = case["imcombiners"]
    zero = kwargs.get("zero")
    scale = kwargs.get("scale")
    if zero is not None:
        arr = arr - np.asarray(zero, dtype=np.float32).reshape(-1, 1, 1)
    if scale is not None:
        arr = arr / np.asarray(scale, dtype=np.float32).reshape(-1, 1, 1)
    return arr


def _retained_bounds(
    arr: np.ndarray, mask_rej: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    retained = np.where(mask_rej, np.nan, arr)
    return np.nanmin(retained, axis=0), np.nanmax(retained, axis=0)


def _retained_sigma(
    arr: np.ndarray, mask_rej: np.ndarray, combined: np.ndarray
) -> np.ndarray:
    retained = np.where(mask_rej, np.nan, arr.astype(np.float32, copy=False))
    n_kept = np.sum(~mask_rej, axis=0)
    sumsq = np.nansum((retained - combined.astype(np.float32, copy=False)) ** 2, axis=0)
    denom = np.where(n_kept > 1, n_kept - 1, 1)
    return np.sqrt(sumsq / denom)


def _has_median_attribution_quirk(case: dict[str, object]) -> bool:
    # IRAF's sorted-window rejectors skip the retained-window reorder for
    # median combine (iccclip.x:408 and analogues); minmax (icmm.gx) compacts
    # its ids unconditionally and attributes correctly.
    return str(case["imcombiners"]["combine"]).lower() in ("median", "med") and str(
        case["imcombiners"]["reject"]
    ) in ("sigclip", "ccdclip", "pclip")


def _iraf_median_attributed_mask(
    mask_rej: np.ndarray, pre_mask: np.ndarray, diag_stack: np.ndarray
) -> np.ndarray:
    """Predict IRAF's rejection-mask attribution for median combine."""
    good = ~pre_mask & np.isfinite(diag_stack)
    nrej = (mask_rej & good).sum(axis=0)
    vals = np.where(good, diag_stack, -np.inf)
    order = np.argsort(-vals, axis=0, kind="stable")
    ranks = np.empty(order.shape, dtype=np.int64)
    np.put_along_axis(
        ranks,
        order,
        np.arange(vals.shape[0], dtype=np.int64).reshape(-1, 1, 1),
        axis=0,
    )
    predicted = (ranks < nrej[None]) & good
    return predicted | (mask_rej & pre_mask)


def _case_context(case: dict[str, object]) -> str:
    return (
        f"case={case['name']} variant={case.get('variant', 'baseline')} "
        f"dtype={case['dtype']} IRAF={case['iraf']} "
        f"imcombiners={case['imcombiners']}"
    )


def _assert_equal_mask(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    case: dict[str, object],
    label: str,
    stack: np.ndarray,
) -> None:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        mismatch = actual != expected
        n_mismatch = int(np.count_nonzero(mismatch))
        examples = []
        for idx in np.argwhere(mismatch)[:8]:
            idx_tuple = tuple(int(i) for i in idx)
            if len(idx_tuple) == stack.ndim:
                value = stack[idx_tuple]
            elif len(idx_tuple) == stack.ndim - 1:
                value = stack[(slice(None), *idx_tuple)].tolist()
            else:
                value = None
            examples.append(
                f"{idx_tuple}: imc={actual[idx_tuple]!r} IRAF={expected[idx_tuple]!r} "
                f"input={value!r}"
            )
        raise AssertionError(
            f"{label} parity failed for {_case_context(case)}; "
            f"{n_mismatch}/{actual.size} entries differ. Examples: "
            + "; ".join(examples)
        )


@pytest.mark.skipif(
    not RUN_IRAF_PARITY,
    reason=IRAF_PARITY_REASON,
)
def test_iraf_parity_manifest_covers_required_dtypes_rejectors_and_combines():
    cases = _cases()
    dtypes = {case["dtype"] for case in cases}
    rejects = {case["imcombiners"]["reject"] for case in cases}
    combines = {case["imcombiners"]["combine"] for case in cases}
    variants = {case["variant"] for case in cases}

    assert REQUIRED_DTYPES <= dtypes
    assert REQUIRED_REJECTS <= rejects
    assert REQUIRED_COMBINES <= combines
    assert REQUIRED_VARIANTS <= variants


@pytest.mark.parametrize(
    "case",
    _cases(),
    ids=[case["name"] for case in _cases()],
)
@pytest.mark.skipif(not RUN_IRAF_PARITY, reason=IRAF_PARITY_REASON)
def test_imcombiners_exactly_matches_iraf_imcombine_outputs(case):
    diagnostics = case["diagnostics"]
    expected_paths = [
        IRAF_DIR / "outputs" / case["output"],
        IRAF_DIR / "outputs" / diagnostics["nrejmask"],
        IRAF_DIR / "outputs" / diagnostics["sigmas"],
        IRAF_DIR / "outputs" / diagnostics["rejmask"],
    ]
    missing = [str(path) for path in expected_paths if not path.exists()]
    if missing:
        pytest.fail(
            "IRAF imcombine reference outputs are absent: " + ", ".join(missing)
        )

    stack = _load_stack(case)
    got, mask_rej, _, _, low, upp, _, _ = _run_imcombiners(case, stack)
    expected = _read_fits(IRAF_DIR / "outputs" / case["output"])
    atol = _atol_for_array(expected)

    assert got.dtype.kind == expected.dtype.kind
    assert got.dtype.itemsize == expected.dtype.itemsize
    np.testing.assert_allclose(
        got,
        expected.astype(got.dtype, copy=False),
        rtol=0.0,
        atol=atol,
        err_msg=f"IRAF imcombine output parity failed for {_case_context(case)}",
    )

    iraf_mask_rej = _load_iraf_rejection_mask(case)
    pre_mask = _pre_rejection_mask(case, stack)
    diag_stack = _diagnostic_stack(case, stack)
    if _has_median_attribution_quirk(case):
        _assert_equal_mask(
            _iraf_median_attributed_mask(mask_rej, pre_mask, diag_stack),
            iraf_mask_rej,
            case=case,
            label="IRAF rejection mask (median attribution)",
            stack=stack,
        )
    else:
        _assert_equal_mask(
            mask_rej,
            iraf_mask_rej,
            case=case,
            label="IRAF rejection mask",
            stack=stack,
        )

    nrej = _read_fits(IRAF_DIR / "outputs" / case["diagnostics"]["nrejmask"])
    _assert_equal_mask(
        mask_rej.sum(axis=0).astype(nrej.dtype, copy=False),
        nrej,
        case=case,
        label="IRAF nrejmask",
        stack=stack,
    )

    if _has_median_attribution_quirk(case):
        total_excluded = mask_rej | pre_mask
    else:
        total_excluded = iraf_mask_rej | pre_mask
    iraf_low, iraf_upp = _retained_bounds(diag_stack, total_excluded)
    np.testing.assert_allclose(
        low,
        iraf_low.astype(low.dtype, copy=False),
        rtol=0.0,
        atol=_atol_for_array(iraf_low.astype(low.dtype, copy=False)),
        err_msg=f"IRAF lower retained bound parity failed for {_case_context(case)}",
    )
    np.testing.assert_allclose(
        upp,
        iraf_upp.astype(upp.dtype, copy=False),
        rtol=0.0,
        atol=_atol_for_array(iraf_upp.astype(upp.dtype, copy=False)),
        err_msg=f"IRAF upper retained bound parity failed for {_case_context(case)}",
    )

    if case.get("variant") == "scaled":
        # IRAF's sigma image applies its own scale correction in icsigma.x;
        # this test covers CCD rejection and retained values, not that helper.
        return

    iraf_sigmas = _read_fits(IRAF_DIR / "outputs" / case["diagnostics"]["sigmas"])
    # IRAF's sigma image uses the same first-n1 value-sorted window as its
    # rejection-mask attribution, not the true retained sample identities.
    imc_sigmas = _retained_sigma(stack, iraf_mask_rej | pre_mask, got)
    np.testing.assert_allclose(
        imc_sigmas.astype(iraf_sigmas.dtype, copy=False),
        iraf_sigmas,
        rtol=0.0,
        atol=_atol_for_array(iraf_sigmas),
        err_msg=f"IRAF sigma diagnostic parity failed for {_case_context(case)}",
    )
