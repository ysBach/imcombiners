"""Benchmark imcombiners against Astropy (with bottleneck) and ccdproc.

The benchmark validates numerical agreement before reporting timings.  It uses
deterministic fake image stacks with shape ``(N, 512, 512)`` by default.

Run examples
------------
Quick smoke run::

    uv run --extra bench python benchmarks/benchmark_combine.py --quick

Full requested matrix::

    uv run --extra bench python benchmarks/benchmark_combine.py

Rows
----
``imc``
    Normal public one-shot ``Combiner(...).combine(...)`` path on the original
    input dtype.
``imc_chain``
    Stateful ``Combiner(...).reject(...).combine(...)`` path. This is mainly
    useful for rejection rows because it retains diagnostics and pipeline state.
``imc_opt``
    Optimized lower-level path for callers that already prepared a contiguous
    floating workspace and disable validation. Integer stacks follow the same
    workspace policy as the public API: `uint8`, `uint16`, and `int16` use
    `float32`; `int32` uses `float64`. Simple generic mean/median rows use the
    optimized imcombiners stack dispatcher; rejection rows use imcombiners
    fused output-only kernels when available.
"""

from __future__ import annotations

import argparse
import gc
import random
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

import imcombiners as imc
import imcombiners.kernels as imck
import numpy as np
import reducers as rd
from astropy import units as u
from astropy.nddata import CCDData
from astropy.stats import sigma_clip

try:
    from benchmarks._environment import format_environment_markdown
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from _environment import format_environment_markdown

try:
    import ccdproc
except ImportError:  # pragma: no cover - optional benchmark dependency
    ccdproc = None


DTYPES = {
    "uint8": np.uint8,
    "uint16": np.uint16,
    "int16": np.int16,
    "int32": np.int32,
    "float32": np.float32,
}

OPS = ("mean", "median", "sigclip_mean", "sigclip_median")
DEFAULT_N = (5, 31)
DEFAULT_SHAPE = (512, 512)
MASK_FRAC = 0.02  # fraction of pixels masked in the masked-case section


@dataclass(frozen=True)
class Case:
    dtype_name: str
    n: int
    op: str
    shape: tuple[int, int]
    masked: bool = False

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(DTYPES[self.dtype_name])

    @property
    def label(self) -> str:
        h, w = self.shape
        tag = " mask" if self.masked else ""
        return f"{self.op:14s} {self.dtype_name:7s} n={self.n:2d} {h}x{w}{tag}"


@dataclass
class CaseResult:
    case: Case
    timings: dict[str, float]
    max_abs: dict[str, float]
    median_abs: dict[str, float]
    frac_bad: dict[str, float]


def make_stack(case: Case) -> np.ndarray:
    """Return a deterministic fake image stack for one benchmark case."""
    rng = np.random.default_rng(20260523 + case.n)
    n, (h, w) = case.n, case.shape
    base = rng.normal(1000.0, 30.0, size=(n, h, w)).astype(np.float32)

    if case.op.startswith("sigclip"):
        high = 800.0
        low = 500.0
        base[0, ::32, ::32] += high
        if n > 1:
            base[1, 16::32, 16::32] -= low

    if case.dtype == np.uint8:
        return np.clip(base / 8.0, 0, 255).round().astype(case.dtype)
    if np.issubdtype(case.dtype, np.integer):
        info = np.iinfo(case.dtype)
        return np.clip(base, info.min, info.max).round().astype(case.dtype)
    return base.astype(case.dtype)


def make_mask(case: Case) -> np.ndarray | None:
    """Return a deterministic boolean mask for masked cases, else None."""
    if not case.masked:
        return None
    n, (h, w) = case.n, case.shape
    rng = np.random.default_rng(77 + case.n)
    return rng.random((n, h, w)) < MASK_FRAC


def np_reference(
    stack: np.ndarray, op: str, mask: np.ndarray | None = None
) -> np.ndarray:
    """Return the NumPy/Astropy reference result."""
    arr = stack.astype(np.float32, copy=True if mask is not None else False)
    if mask is not None:
        arr[mask] = np.nan
    if op == "mean":
        return np.nanmean(arr, axis=0)
    if op == "median":
        return np.nanmedian(arr, axis=0).astype(np.float32)

    clipped = sigma_clip(
        arr,
        sigma_lower=3.0,
        sigma_upper=3.0,
        maxiters=5,
        cenfunc="median",
        stdfunc="std",
        axis=0,
        masked=True,
        copy=True,
    )
    filled = clipped.filled(np.nan).astype(np.float32, copy=False)
    if op == "sigclip_mean":
        return np.nanmean(filled, axis=0)
    if op == "sigclip_median":
        return np.nanmedian(filled, axis=0).astype(np.float32)
    raise ValueError(f"unknown op: {op}")


def _sigclip_spec() -> imc.SigClip:
    """Return the benchmark sigma-clipping rejector spec."""
    return imc.SigClip(
        sigma=(3.0, 3.0),
        maxiters=5,
        ddof=0,
        nkeep=0,
        cenfunc="median",
        clip_cen="mean",
    )


def imcombiners_public(
    stack: np.ndarray, op: str, mask: np.ndarray | None = None
) -> np.ndarray:
    """Run imc through the public one-shot Combiner path."""
    if op == "mean":
        return imc.Combiner(stack, mask=mask).combine("mean")
    if op == "median":
        return imc.Combiner(stack, mask=mask).combine("median")
    if op == "sigclip_mean":
        return imc.Combiner(stack, mask=mask).combine(
            "mean",
            rejectors=_sigclip_spec(),
            diagnostics=None,
        )
    if op == "sigclip_median":
        return imc.Combiner(stack, mask=mask).combine(
            "median",
            rejectors=_sigclip_spec(),
            diagnostics=None,
        )
    raise ValueError(f"unknown op: {op}")


def imcombiners_chain(
    stack: np.ndarray, op: str, mask: np.ndarray | None = None
) -> np.ndarray:
    """Run imc through the stateful Combiner chain."""
    if op == "mean":
        return imc.Combiner(stack, mask=mask).combine("mean")
    if op == "median":
        return imc.Combiner(stack, mask=mask).combine("median")
    if op == "sigclip_mean":
        return imc.Combiner(stack, mask=mask).reject(_sigclip_spec()).combine("mean")
    if op == "sigclip_median":
        return imc.Combiner(stack, mask=mask).reject(_sigclip_spec()).combine("median")
    raise ValueError(f"unknown op: {op}")


def optimized_workspace(stack: np.ndarray) -> np.ndarray:
    """Return the fastest valid workspace dtype for `validate=False` calls."""
    dtype = np.float64 if stack.dtype == np.dtype(np.int32) else np.float32
    return np.ascontiguousarray(stack, dtype=dtype)


def imcombiners_optimized(
    stack_opt: np.ndarray, op: str, mask: np.ndarray | None = None
) -> np.ndarray:
    """Run the optimized direct path with validation disabled."""
    if op == "mean":
        arr = stack_opt if mask is None else np.where(mask, np.nan, stack_opt)
        return imck._stack(arr, rd.nanmean, validate=False)
    if op == "median":
        arr = stack_opt if mask is None else np.where(mask, np.nan, stack_opt)
        return imck._stack(arr, rd.nanmedian, validate=False)
    if op == "sigclip_mean":
        return imck.sigclip_combine(
            stack_opt,
            mask=mask,
            combine="mean",
            sigma=(3.0, 3.0),
            maxiters=5,
            ddof=0,
            nkeep=0,
            cenfunc="median",
            clip_cen="mean",
            validate=False,
        )
    if op == "sigclip_median":
        return imck.sigclip_combine(
            stack_opt,
            mask=mask,
            combine="median",
            sigma=(3.0, 3.0),
            maxiters=5,
            ddof=0,
            nkeep=0,
            cenfunc="median",
            clip_cen="mean",
            validate=False,
        )
    raise ValueError(f"unknown op: {op}")


def ccdproc_result(
    stack: np.ndarray, op: str, mask: np.ndarray | None = None
) -> np.ndarray | None:
    """Return ccdproc result, or ``None`` when ccdproc is unavailable."""
    if ccdproc is None:
        return None

    combiner = ccdproc.Combiner(
        [
            CCDData(
                frame,
                unit=u.dimensionless_unscaled,
                mask=None if mask is None else mask[i],
            )
            for i, frame in enumerate(stack)
        ],
        dtype=np.float32,
    )
    if op.startswith("sigclip"):
        combiner.sigma_clipping(
            low_thresh=3.0,
            high_thresh=3.0,
            func="median",
            dev_func="std",
        )

    if op.endswith("mean"):
        return np.asarray(combiner.average_combine().data, dtype=np.float32)
    if op.endswith("median"):
        return np.asarray(combiner.median_combine().data, dtype=np.float32)
    raise ValueError(f"unknown op: {op}")


def time_functions(
    functions: dict[str, Callable[[], np.ndarray]],
    *,
    repeats: int,
    warmups: int,
) -> dict[str, float]:
    """Return median wall times from global round-robin timing."""
    for _ in range(warmups):
        for fn in functions.values():
            result = fn()
            if result.size == 0:
                raise RuntimeError("benchmark function returned empty result")

    samples = {name: [] for name in functions}
    names = list(functions)
    rng = random.Random(12345)
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            rng.shuffle(names)
            for name in names:
                start = time.perf_counter()
                result = functions[name]()
                elapsed = time.perf_counter() - start
                if result.size == 0:
                    raise RuntimeError("benchmark function returned empty result")
                samples[name].append(elapsed)
    finally:
        if gc_was_enabled:
            gc.enable()

    return {name: statistics.median(vals) for name, vals in samples.items()}


def summarize_difference(
    candidate: np.ndarray, reference: np.ndarray, *, rtol: float, atol: float
) -> tuple[float, float, float]:
    """Return max abs, median abs, and fraction above tolerance."""
    diff = np.abs(candidate.astype(np.float64) - reference.astype(np.float64))
    tol = atol + rtol * np.abs(reference.astype(np.float64))
    return (
        float(np.nanmax(diff)),
        float(np.nanmedian(diff)),
        float(np.count_nonzero(diff > tol) / diff.size),
    )


def assert_matches(
    name: str,
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    rtol: float,
    atol: float,
    max_frac_bad: float,
) -> None:
    """Fail on systematic mismatch, allowing rare threshold-boundary pixels."""
    max_abs, median_abs, frac_bad = summarize_difference(
        candidate, reference, rtol=rtol, atol=atol
    )
    if frac_bad <= max_frac_bad:
        return
    raise AssertionError(
        f"{name} differs from reference: max_abs={max_abs:.6g}, "
        f"median_abs={median_abs:.6g}, "
        f"frac_bad={frac_bad:.6g} > {max_frac_bad:.6g}"
    )


def run_case(case: Case, args: argparse.Namespace) -> CaseResult:
    """Validate and time one benchmark case."""
    stack = make_stack(case)
    mask = make_mask(case)
    stack_opt = optimized_workspace(stack)
    ref = np_reference(stack, case.op, mask)

    functions: dict[str, Callable[[], np.ndarray]] = {
        "numpy_astropy": lambda: np_reference(stack, case.op, mask),
        "imc": lambda: imcombiners_public(stack, case.op, mask),
        "imc_chain": lambda: imcombiners_chain(stack, case.op, mask),
        "imc_opt": lambda: imcombiners_optimized(stack_opt, case.op, mask),
    }
    ccd = ccdproc_result(stack, case.op, mask)
    if ccd is not None:
        functions["ccdproc"] = lambda: ccdproc_result(stack, case.op, mask)

    imc_out = functions["imc"]()
    assert_matches(
        "imc",
        imc_out,
        ref,
        rtol=args.rtol,
        atol=args.atol,
        max_frac_bad=args.max_frac_bad,
    )
    imc_opt_out = functions["imc_opt"]()
    assert_matches(
        "imc_opt",
        imc_opt_out,
        ref,
        rtol=args.rtol,
        atol=args.atol,
        max_frac_bad=args.max_frac_bad,
    )
    imc_chain_out = functions["imc_chain"]()
    assert_matches(
        "imc_chain",
        imc_chain_out,
        ref,
        rtol=args.rtol,
        atol=args.atol,
        max_frac_bad=args.max_frac_bad,
    )

    max_abs: dict[str, float] = {}
    median_abs: dict[str, float] = {}
    frac_bad: dict[str, float] = {}
    for name, fn in functions.items():
        out = fn()
        ma, med, frac = summarize_difference(out, ref, rtol=args.rtol, atol=args.atol)
        max_abs[name] = ma
        median_abs[name] = med
        frac_bad[name] = frac

    timings = time_functions(functions, repeats=args.repeats, warmups=args.warmups)
    return CaseResult(case, timings, max_abs, median_abs, frac_bad)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick", action="store_true", help="run a reduced smoke matrix"
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument(
        "--max-frac-bad",
        type=float,
        default=1e-5,
        help="maximum tolerated fraction of pixels outside tolerance for imcombiners",
    )
    parser.add_argument("--height", type=int, default=DEFAULT_SHAPE[0])
    parser.add_argument("--width", type=int, default=DEFAULT_SHAPE[1])
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=tuple(DTYPES),
        default=list(DTYPES),
    )
    parser.add_argument("--n", nargs="+", type=int, default=list(DEFAULT_N))
    parser.add_argument("--ops", nargs="+", choices=OPS, default=list(OPS))
    return parser.parse_args()


_SIGCLIP_NOTE = (
    '`sigma=(3, 3)`, `maxiters=5`, `ddof=0`, `cenfunc="median"`, '
    '`clip_cen="mean"`. ccdproc uses `maxiters=1` †.'
)

_COL_HEADERS = [
    "op",
    "dtype",
    "N",
    "imc",
    "imc_chain",
    "imc_opt",
    "ap_bn",
    "ccdproc",
    "imc/opt",
    "chain/opt",
    "ap/opt",
    "ccd/opt",
]
# "<" = left-aligned, ">" = right-aligned (GFM convention)
_COL_ALIGN = ["<", "<", ">", ">", ">", ">", ">", ">", ">", ">", ">", ">"]


def _dw(cell: str) -> int:
    """Display width: strip markdown bold markers."""
    return len(cell.replace("**", ""))


def _row_cells(result: CaseResult) -> list[str]:
    case = result.case
    t = result.timings
    imc_ms = t["imc"] * 1e3
    chain_ms = t["imc_chain"] * 1e3
    opt_ms = t["imc_opt"] * 1e3
    ap_ms = t["numpy_astropy"] * 1e3
    ccd_ms = t.get("ccdproc", float("nan")) * 1e3

    is_sigclip = case.op.startswith("sigclip")
    ratio_fmt = ".2f" if is_sigclip else ".1f"

    imc_ratio = imc_ms / opt_ms
    chain_ratio = chain_ms / opt_ms
    ap_ratio = ap_ms / opt_ms
    ccd_ratio = ccd_ms / opt_ms

    bold = case.dtype_name == "float32" and case.op in ("median", "sigclip_median")
    ap_str = f"{ap_ratio:{ratio_fmt}}"
    ccd_str = f"{ccd_ratio:{ratio_fmt}}"
    if bold:
        ap_str = f"**{ap_str}**"
        ccd_str = f"**{ccd_str}**"

    ccd_dagger = "†" if result.frac_bad.get("ccdproc", 0) > 0 else ""

    return [
        case.op,
        case.dtype_name,
        str(case.n),
        f"{imc_ms:.2f}",
        f"{chain_ms:.2f}",
        f"{opt_ms:.2f}",
        f"{ap_ms:.2f}",
        f"{ccd_ms:.2f}{ccd_dagger}",
        f"{imc_ratio:{ratio_fmt}}",
        f"{chain_ratio:{ratio_fmt}}",
        ap_str,
        ccd_str,
    ]


def _pad(cell: str, width: int, align: str) -> str:
    """Pad cell to `width` display chars, accounting for bold markers."""
    extra = len(cell) - _dw(cell)  # chars added by ** markers
    target = width + extra
    return cell.rjust(target) if align == ">" else cell.ljust(target)


def _print_table(results: list[CaseResult]) -> None:
    if not results:
        print("_No cases selected for this section._")
        return

    all_rows = [_row_cells(r) for r in results]
    ncols = len(_COL_HEADERS)

    widths = [
        max(_dw(_COL_HEADERS[i]), max(_dw(row[i]) for row in all_rows))
        for i in range(ncols)
    ]

    def _fmt_row(cells: list[str]) -> str:
        padded = [_pad(cells[i], widths[i], _COL_ALIGN[i]) for i in range(ncols)]
        return "| " + " | ".join(padded) + " |"

    def _sep_cell(w: int, align: str) -> str:
        if align == ">":
            return "-" * max(w - 1, 2) + ":"
        return "-" * max(w, 3)

    print(_fmt_row(_COL_HEADERS))
    sep_cells = [_sep_cell(widths[i], _COL_ALIGN[i]) for i in range(ncols)]
    print("|" + "|".join(f" {s} " for s in sep_cells) + "|")
    for row in all_rows:
        print(_fmt_row(row))


def main() -> None:
    args = parse_args()
    if args.quick:
        args.dtypes = ["float32", "uint16", "int16"]
        args.n = [5]
        args.ops = ["mean", "sigclip_mean"]
        args.repeats = min(args.repeats, 3)

    base_cases = [
        Case(dtype_name=dtype_name, n=n, op=op, shape=(args.height, args.width))
        for dtype_name in args.dtypes
        for n in args.n
        for op in args.ops
    ]
    masked_cases = [
        Case(
            dtype_name=dtype_name,
            n=n,
            op=op,
            shape=(args.height, args.width),
            masked=True,
        )
        for dtype_name in args.dtypes
        for n in args.n
        for op in args.ops
    ]

    print("# imcombiners dtype/combination benchmark")
    print("# Reference: NumPy mean/median and Astropy sigma_clip+bottleneck combine.")
    print("# ccdproc differences are informational and are not used as parity gates.")
    print("# imc is the normal public one-shot Combiner path on the original dtype.")
    print("# imc_chain is the stateful Combiner reject/combine path.")
    print(
        "# imc_opt is a prepared contiguous workspace with validation disabled, "
        "using imc stack dispatch for simple reductions and fused imc kernels "
        "where available."
    )
    print(
        "# Sigma clipping: sigma=(3, 3), maxiters=5, ddof=0, "
        'cenfunc="median", clip_cen="mean".'
    )
    print("# ccdproc Combiner.sigma_clipping uses its public default maxiters=1.")
    print(f"# Masked cases use a random {MASK_FRAC:.0%} pixel mask per frame.")
    print()
    print(format_environment_markdown())

    all_base: list[CaseResult] = [run_case(c, args) for c in base_cases]
    all_masked: list[CaseResult] = [run_case(c, args) for c in masked_cases]

    def _split(results: list[CaseResult]) -> tuple[list[CaseResult], list[CaseResult]]:
        simple = [r for r in results if not r.case.op.startswith("sigclip")]
        sigclip = [r for r in results if r.case.op.startswith("sigclip")]
        return simple, sigclip

    base_simple, base_sigclip = _split(all_base)
    masked_simple, masked_sigclip = _split(all_masked)

    print("\n## No Mask\n")
    print("### Simple\n")
    _print_table(base_simple)

    print("\n### Sigclip\n")
    print(_SIGCLIP_NOTE)
    print()
    _print_table(base_sigclip)

    print(f"\n## With Mask (~{MASK_FRAC:.0%} pixels masked per frame)\n")
    print("### Simple\n")
    _print_table(masked_simple)

    print("\n### Sigclip\n")
    print(_SIGCLIP_NOTE)
    print()
    _print_table(masked_sigclip)


if __name__ == "__main__":
    main()
