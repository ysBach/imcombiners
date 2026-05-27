"""Benchmark 1-D imcombiners kernels against NumPy and Bottleneck.

Examples
--------
Quick smoke run:

    uv run --extra bench python benchmarks/benchmark_1d.py --quick

Default benchmark lengths are 100, 10_000, and 10_000_000 samples.
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

import imcombiners.kernels as imck
import numpy as np

try:
    import bottleneck as bn
except ImportError:  # pragma: no cover - optional benchmark dependency
    bn = None


OPS = ("mean", "median", "sum", "min", "max", "var")
DTYPES = ("float64", "float32")
DEFAULT_LENGTHS = (100, 10_000, 10_000_000)


@dataclass(frozen=True)
class Result:
    """Timing result for one function."""

    length: int
    dtype: str
    op: str
    name: str
    median_ms: float


def make_values(length: int, dtype: str) -> np.ndarray:
    """Return deterministic benchmark values."""
    rng = np.random.default_rng(20260528 + length)
    values = rng.normal(1000.0, 30.0, size=length).astype(dtype)
    if length >= 10:
        values[0] = np.nan
        values[1] += 700.0
    if length >= 1_000:
        values[:: max(1, length // 97)] = np.nan
        values[1 :: max(1, length // 89)] += 700.0
    return np.ascontiguousarray(values)


def calls_per_sample(length: int) -> int:
    """Return inner-loop calls used to stabilize small-vector timings."""
    return max(1, min(100_000, 1_000_000 // length))


def time_function(
    func: Callable[[], float], *, repeats: int, warmups: int, inner: int
) -> float:
    """Return median milliseconds per call."""
    for _ in range(warmups):
        for _ in range(inner):
            func()

    samples: list[float] = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            start = time.perf_counter()
            for _ in range(inner):
                func()
            samples.append((time.perf_counter() - start) * 1e3 / inner)
    finally:
        if gc_was_enabled:
            gc.enable()
    return statistics.median(samples)


def functions_for(values: np.ndarray, op: str) -> dict[str, Callable[[], float]]:
    """Return benchmark functions for one operation."""
    funcs: dict[str, Callable[[], float]] = {}
    if op == "mean":
        funcs["numpy"] = lambda: np.nanmean(values)
        if bn is not None:
            funcs["bottleneck"] = lambda: bn.nanmean(values)
        funcs["imc"] = lambda: imck.mean_1d(values, validate=False)
    elif op == "median":
        funcs["numpy"] = lambda: np.nanmedian(values)
        if bn is not None:
            funcs["bottleneck"] = lambda: bn.nanmedian(values)
        funcs["imc"] = lambda: imck.median_1d(values, validate=False)
    elif op == "sum":
        funcs["numpy"] = lambda: np.nansum(values)
        if bn is not None:
            funcs["bottleneck"] = lambda: bn.nansum(values)
        funcs["imc"] = lambda: imck.sum_1d(values, validate=False)
    elif op == "min":
        funcs["numpy"] = lambda: np.nanmin(values)
        if bn is not None:
            funcs["bottleneck"] = lambda: bn.nanmin(values)
        funcs["imc"] = lambda: imck.min_1d(values, validate=False)
    elif op == "max":
        funcs["numpy"] = lambda: np.nanmax(values)
        if bn is not None:
            funcs["bottleneck"] = lambda: bn.nanmax(values)
        funcs["imc"] = lambda: imck.max_1d(values, validate=False)
    elif op == "var":
        funcs["numpy"] = lambda: np.nanvar(values)
        if bn is not None and hasattr(bn, "nanvar"):
            funcs["bottleneck"] = lambda: bn.nanvar(values)
        funcs["imc"] = lambda: imck.var_1d(values, validate=False)
    else:
        raise ValueError(f"unknown op: {op}")
    return funcs


def run_case(
    values: np.ndarray, *, op: str, repeats: int, warmups: int
) -> list[Result]:
    """Time all functions for one vector/op pair."""
    inner = calls_per_sample(values.size)
    timings: dict[str, float] = {}
    for name, func in functions_for(values, op).items():
        timings[name] = time_function(
            func, repeats=repeats, warmups=warmups, inner=inner
        )

    return [
        Result(
            length=values.size,
            dtype=str(values.dtype),
            op=op,
            name=name,
            median_ms=median_ms,
        )
        for name, median_ms in timings.items()
    ]


def print_table(results: list[Result]) -> None:
    """Print a Markdown table."""
    print("| length | dtype | op | np (us) | bn (us) | imc (us) | np/imc | bn/imc |")
    print("|---:|---|---|---:|---:|---:|---:|---:|")

    rows: dict[tuple[int, str, str], dict[str, float]] = {}
    for result in results:
        key = (result.length, result.dtype, result.op)
        rows.setdefault(key, {})[result.name] = result.median_ms

    def fmt_us(value_ms: float | None) -> str:
        if value_ms is None:
            return "-"
        return f"{value_ms * 1000:.2f}"

    def fmt_ratio(value: float | None, reference: float | None) -> str:
        if value is None or reference is None:
            return "-"
        return f"{value / reference:.1f}x"

    def fmt_length(length: int) -> str:
        labels = {100: "10^2", 10_000: "10^4", 10_000_000: "10^7"}
        return labels.get(length, f"{length:,}")

    for (length, dtype, op), timings in rows.items():
        np_ms = timings.get("numpy")
        bn_ms = timings.get("bottleneck")
        imc_ms = timings.get("imc")
        print(
            f"| {fmt_length(length)} | {dtype} | {op} | {fmt_us(np_ms)} | "
            f"{fmt_us(bn_ms)} | {fmt_us(imc_ms)} | {fmt_ratio(np_ms, imc_ms)} | "
            f"{fmt_ratio(bn_ms, imc_ms)} |"
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", nargs="+", type=int, default=list(DEFAULT_LENGTHS))
    parser.add_argument("--ops", nargs="+", choices=OPS, default=list(OPS))
    parser.add_argument("--dtypes", nargs="+", choices=DTYPES, default=list(DTYPES))
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.lengths = [100, 10_000]
        args.repeats = min(args.repeats, 3)
        args.warmups = min(args.warmups, 1)
    if any(length <= 0 for length in args.lengths):
        parser.error("--lengths must be positive")
    return args


def main() -> None:
    """Run the benchmark."""
    args = parse_args()
    results: list[Result] = []
    print("# imcombiners 1-D kernel benchmark")
    bn_status = "available" if bn is not None else "missing"
    print(f"# dtypes={','.join(args.dtypes)}; bottleneck={bn_status}")
    for dtype in args.dtypes:
        for length in args.lengths:
            values = make_values(length, dtype)
            for op in args.ops:
                results.extend(
                    run_case(values, op=op, repeats=args.repeats, warmups=args.warmups)
                )
    print_table(results)


if __name__ == "__main__":
    main()
