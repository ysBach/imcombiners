"""Tune Rayon thread counts for representative imcombiners workloads.

See ``docs/quarto/performance/max-performance.qmd`` for usage guidance,
threshold notes, and interpretation caveats.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

OPS = ("mean", "median", "sigclip_mean", "sigclip_median")
DTYPES = ("float32", "float64")
THREAD_ENV_VARS = (
    "RAYON_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@dataclass(frozen=True)
class ThreadResult:
    """Timing summary for one candidate thread count."""

    threads: int
    median_ms: float
    min_ms: float
    max_ms: float


def _json_from_stdout(stdout: str) -> dict[str, float]:
    """Parse the last non-empty stdout line as JSON."""
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if stripped:
            return json.loads(stripped)
    raise ValueError("worker produced no JSON output")


_DENSE_SWEEP_MAX = 32
_PRACTICAL_PARALLEL_THRESHOLD = 10_000


def candidate_threads(
    cpu_count: int | None = None, *, force_large_sweep: bool = False
) -> list[int]:
    """Return a default thread sweep for this machine.

    For machines with at least 2 and at most ``_DENSE_SWEEP_MAX`` logical CPUs
    every integer from 2 to ``ncpu`` is tested.  The forced serial baseline
    already covers the non-parallel case, so one-thread Rayon is kept out of
    the default sweep unless users request it explicitly with ``--threads``.

    For larger machines, return no implicit candidates unless
    ``force_large_sweep`` is set. Such hosts are commonly computing-center nodes
    where users should pass scheduler/allocation-aware candidates explicitly.
    """
    ncpu = cpu_count or (os.cpu_count() or 1)
    ncpu = max(1, int(ncpu))
    if ncpu == 1:
        return []
    if ncpu <= _DENSE_SWEEP_MAX:
        return list(range(2, ncpu + 1))
    if not force_large_sweep:
        return []
    values = {2, ncpu}
    value = 2
    while value < ncpu:
        values.add(value)
        value *= 2
    return sorted(values)


def recommended_parallel_threshold(output_elements: int) -> int:
    """Return a practical production threshold for a parallel recommendation.

    The benchmark forces threshold=1 internally to measure Rayon. User-facing
    recommendations should keep small workloads serial unless the measured
    workload is itself smaller than the practical threshold.
    """
    return max(1, min(_PRACTICAL_PARALLEL_THRESHOLD, int(output_elements)))


def selected_threads(args: argparse.Namespace) -> list[int]:
    """Return explicit or default thread-count candidates for parsed args."""
    return args.threads or candidate_threads(
        args.max_threads, force_large_sweep=args.force_large_sweep
    )


def image_count_label(n_images: int) -> str:
    """Return a concise label for a benchmark image-count workload."""
    if n_images == 5:
        return "few images"
    if n_images == 31:
        return "many images"
    return f"n={n_images} images"


def image_count_summary(args: argparse.Namespace) -> str:
    """Return a user-facing summary of the image-count workloads."""
    if args.n_values == [5, 31] and not args.quick:
        return (
            "Default image-count workloads are n=5 (few-image combinations) "
            "and n=31 (many-image combinations)."
        )
    items = ", ".join(
        f"n={n_images} ({image_count_label(n_images)})" for n_images in args.n_values
    )
    return f"Image-count workloads for this run: {items}."


def iter_workloads(args: argparse.Namespace):
    """Yield labeled argument namespaces for each requested image count."""
    for n_images in args.n_values:
        yield (
            image_count_label(n_images),
            argparse.Namespace(**{**vars(args), "n": n_images}),
        )


def best_result(
    results: list[ThreadResult], *, tie_tolerance: float = 0.05
) -> ThreadResult:
    """Return a robust recommendation, preferring fewer near-tied threads."""
    if not results:
        raise ValueError("results must not be empty")
    fastest = min(results, key=lambda item: item.median_ms)
    cutoff = fastest.median_ms * (1.0 + tie_tolerance)
    near_best = [item for item in results if item.median_ms <= cutoff]
    return min(near_best, key=lambda item: item.threads)


def fastest_result(results: list[ThreadResult]) -> ThreadResult:
    """Return the fastest observed median result."""
    if not results:
        raise ValueError("results must not be empty")
    return min(results, key=lambda item: (item.median_ms, item.threads))


def speedup(result: ThreadResult, baseline: ThreadResult) -> float:
    """Return speedup relative to the baseline median time."""
    return baseline.median_ms / result.median_ms


def make_stack(n: int, height: int, width: int, dtype: str) -> np.ndarray:
    """Create a deterministic stack with a few outliers for rejection work."""
    rng = np.random.default_rng(20260524 + n + height + width)
    stack = rng.normal(1000.0, 30.0, size=(n, height, width)).astype(np.float32)
    stack[0, ::32, ::32] += 800.0
    if n > 1:
        stack[1, 16::32, 16::32] -= 500.0
    return stack.astype(dtype, copy=False)


def run_operation(stack: np.ndarray, op: str) -> np.ndarray:
    """Run one imcombiners operation."""
    import imcombiners.kernels as imck

    if op == "mean":
        return imck.mean(stack, validate=False)
    if op == "median":
        return imck.median(stack, validate=False)
    if op == "sigclip_mean":
        return imck.sigclip_combine(
            stack,
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
            stack,
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


def worker_main(args: argparse.Namespace) -> None:
    """Run one timing worker and print JSON for the parent process."""
    stack = make_stack(args.n, args.height, args.width, args.dtype)

    for _ in range(args.warmups):
        result = run_operation(stack, args.op)
        if result.size == 0:
            raise RuntimeError("benchmark operation returned empty result")

    samples: list[float] = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(args.repeats):
            start = time.perf_counter()
            result = run_operation(stack, args.op)
            elapsed = time.perf_counter() - start
            if result.size == 0:
                raise RuntimeError("benchmark operation returned empty result")
            samples.append(elapsed * 1e3)
    finally:
        if gc_was_enabled:
            gc.enable()

    print(
        json.dumps(
            {
                "median_ms": statistics.median(samples),
                "min_ms": min(samples),
                "max_ms": max(samples),
            },
            sort_keys=True,
        )
    )


def run_candidate(threads: int, args: argparse.Namespace) -> ThreadResult:
    """Run one thread-count candidate in a fresh subprocess."""
    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = str(threads)
    env["IMCOMBINERS_PARALLEL_THRESHOLD"] = "1"
    for name in THREAD_ENV_VARS[1:]:
        env[name] = "1"

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--op",
        args.op,
        "--dtype",
        args.dtype,
        "--n",
        str(args.n),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--repeats",
        str(args.repeats),
        "--warmups",
        str(args.warmups),
    ]
    proc = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )
    payload = _json_from_stdout(proc.stdout)
    return ThreadResult(
        threads=threads,
        median_ms=float(payload["median_ms"]),
        min_ms=float(payload["min_ms"]),
        max_ms=float(payload["max_ms"]),
    )


def run_serial_candidate(args: argparse.Namespace) -> ThreadResult:
    """Run the benchmark workload with the serial kernel path forced."""
    output_elements = args.height * args.width
    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = "1"
    env["IMCOMBINERS_PARALLEL_THRESHOLD"] = str(output_elements + 1)
    for name in THREAD_ENV_VARS[1:]:
        env[name] = "1"

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--op",
        args.op,
        "--dtype",
        args.dtype,
        "--n",
        str(args.n),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--repeats",
        str(args.repeats),
        "--warmups",
        str(args.warmups),
    ]
    proc = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )
    payload = _json_from_stdout(proc.stdout)
    return ThreadResult(
        threads=0,
        median_ms=float(payload["median_ms"]),
        min_ms=float(payload["min_ms"]),
        max_ms=float(payload["max_ms"]),
    )


def run_sweep(args: argparse.Namespace) -> list[ThreadResult]:
    """Run all requested thread-count candidates."""
    threads = selected_threads(args)
    results: list[ThreadResult] = []
    order = list(dict.fromkeys(int(value) for value in threads if int(value) > 0))
    rng = random.Random(args.seed)
    rng.shuffle(order)
    for n_threads in order:
        results.append(run_candidate(n_threads, args))
    return sorted(results, key=lambda item: item.threads)


def print_results(
    results: list[ThreadResult],
    args: argparse.Namespace,
    *,
    serial_result: ThreadResult | None = None,
    workload_label: str | None = None,
) -> None:
    """Print a human-readable tuning table."""
    if not results and serial_result is None:
        raise ValueError("results must not be empty without a serial baseline")
    baseline = serial_result or next(
        (item for item in results if item.threads == 1), results[0]
    )
    fastest = fastest_result(results) if results else None
    best = best_result(results, tie_tolerance=args.tie_tolerance) if results else None
    output_elements = args.height * args.width
    measured_parallel_threshold = 1
    force_serial_threshold = output_elements + 1
    practical_threshold = recommended_parallel_threshold(output_elements)
    recommend_serial = best is None or (
        serial_result is not None
        and serial_result.median_ms <= best.median_ms * (1.0 + args.tie_tolerance)
    )

    print("# imcombiners parallel thread tuning")
    if workload_label is not None:
        print(f"# workload case: {workload_label}")
    print(
        f"# workload: op={args.op}, dtype={args.dtype}, "
        f"shape=({args.n}, {args.height}, {args.width})"
    )
    print(f"# repeats={args.repeats}, warmups={args.warmups}")
    print("# set RAYON_NUM_THREADS before importing imcombiners")
    print(
        f"# recommendation uses the lowest thread count within "
        f"{args.tie_tolerance:.0%} of the fastest median"
    )
    print()
    speedup_label = (
        "speedup_vs_serial" if serial_result is not None else "speedup_vs_baseline"
    )
    print(f"mode/threads  median_ms   min_ms   max_ms  {speedup_label}")
    if serial_result is not None:
        print(
            f"{'serial':>11s}  {serial_result.median_ms:9.2f}  "
            f"{serial_result.min_ms:7.2f}  {serial_result.max_ms:7.2f}  "
            f"{speedup(serial_result, baseline):17.2f}"
        )
    for item in results:
        print(
            f"{item.threads:11d}  {item.median_ms:9.2f}  "
            f"{item.min_ms:7.2f}  {item.max_ms:7.2f}  "
            f"{speedup(item, baseline):17.2f}"
        )
    print()
    if best is not None and fastest is not None:
        print(f"Fastest observed median: RAYON_NUM_THREADS={fastest.threads}")
        print(f"Best measured parallel setting: RAYON_NUM_THREADS={best.threads}")
    else:
        print("No parallel thread candidates were measured.")
    if best is not None and fastest is not None and fastest.threads != best.threads:
        print(
            "# fastest and recommended differ because the lower thread count "
            "is within the near-tie tolerance"
        )
    print()
    if recommend_serial:
        print("Recommended measured mode: serial")
        print("# Copy into your shell before starting Python:")
        print(f"export IMCOMBINERS_PARALLEL_THRESHOLD={force_serial_threshold}")
        print(
            "# RAYON_NUM_THREADS is not used by this serial recommendation "
            "for this workload."
        )
        threshold = force_serial_threshold
    else:
        print("Recommended measured mode: parallel")
        assert best is not None
        print(f"Recommended RAYON_NUM_THREADS={best.threads}")
        print(
            f"# parallel candidates were measured with threshold="
            f"{measured_parallel_threshold} to force Rayon"
        )
        print(
            "# Do not use threshold=1 as a global default; leave the package "
            "default or tune the threshold separately"
        )
        print(
            "# Practical threshold starting point for this output size: "
            f"{practical_threshold}"
        )
        print("# You may copy it into your shell before starting Python:")
        print(f"export RAYON_NUM_THREADS={best.threads}")
        print(f"export IMCOMBINERS_PARALLEL_THRESHOLD={practical_threshold}")
        threshold = practical_threshold
    print()
    print(
        "# Exact numbers can vary with op, dtype, image count, "
        "shape, CPU allocation, memory bandwidth, and machine load.\n"
        "# For example, (5, 64, 128) results in best threads=5 while "
        "(31, 512, 512) can result in 12 on the same machine."
    )
    print("\n# You may also set/check the threshold inside Python right after import:")
    print("import imcombiners as imc")
    if not recommend_serial:
        print(f"imc.set_rayon_num_threads({best.threads})")
    if threshold is not None:
        print(f"imc.set_parallel_threshold({threshold})")
    if not recommend_serial:
        print("print(imc.get_rayon_num_threads())")
    if threshold is not None:
        print("print(imc.get_parallel_threshold())")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--quick", action="store_true", help="run a small smoke sweep")
    parser.add_argument("--threads", nargs="+", type=int, help="thread counts to test")
    parser.add_argument(
        "--max-threads",
        type=int,
        default=None,
        help="maximum thread count for the default sweep; defaults to os.cpu_count()",
    )
    parser.add_argument(
        "--force-large-sweep",
        "--force-test",
        action="store_true",
        help=(
            "force the sparse default sweep on hosts with more than "
            f"{_DENSE_SWEEP_MAX} logical CPUs"
        ),
    )
    parser.add_argument("--op", choices=OPS, default="sigclip_median")
    parser.add_argument("--dtype", choices=DTYPES, default="float32")
    parser.add_argument(
        "--n",
        nargs="+",
        type=int,
        default=None,
        help=(
            "one or more image counts to benchmark; defaults to 5 and 31 "
            "for few-image and many-image combinations"
        ),
    )
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20250501)
    parser.add_argument(
        "--tie-tolerance",
        type=float,
        default=0.05,
        help=(
            "fractional tolerance for treating median times as near-ties; "
            "the lowest thread count within this tolerance is recommended"
        ),
    )
    args = parser.parse_args(argv)
    if args.tie_tolerance < 0:
        parser.error("--tie-tolerance must be non-negative")
    user_n_values = args.n
    args.n_values = user_n_values or [5, 31]
    if any(value <= 0 for value in args.n_values):
        parser.error("--n values must be positive")
    if args.worker and len(args.n_values) != 1:
        parser.error("--worker requires exactly one --n value")
    if args.quick:
        args.threads = args.threads or [2]
        if user_n_values is None:
            args.n_values = [5]
        else:
            args.n_values = [min(value, 5) for value in args.n_values]
        args.height = min(args.height, 128)
        args.width = min(args.width, 128)
        args.repeats = min(args.repeats, 3)
    args.n = args.n_values[0]
    return args


def main() -> None:
    """Run the thread-count sweep."""
    args = parse_args()
    if args.worker:
        worker_main(args)
        return
    candidates = selected_threads(args)
    if (
        not candidates
        and (args.max_threads or (os.cpu_count() or 1)) > _DENSE_SWEEP_MAX
    ):
        print(
            "Host appears to have more than "
            f"{_DENSE_SWEEP_MAX} logical CPUs. This is often a computing-center "
            "node where os.cpu_count() may not match your allocation."
        )
        print("Pass --threads with allocation-aware candidates, for example:")
        print("  uv run python benchmarks/benchmark_threads.py --threads 8 16 32")
        ncpu = args.max_threads or (os.cpu_count() or 1)
        if ncpu > 3:
            print(
                f"  uv run python benchmarks/benchmark_threads.py --threads "
                f"2 {max(2, ncpu - 2)} {ncpu}"
            )
        print(
            "Or pass --force-large-sweep/--force-test to run the sparse default sweep."
        )
        return
    workloads = list(iter_workloads(args))
    n_candidates = len(candidates)
    print(
        f"Running sweep over {n_candidates} thread-count candidate(s) "
        f"for {len(workloads)} image-count workload(s) "
        f"({args.warmups} warmup + {args.repeats} timed repeat(s) each). "
        "This may take several minutes depending on your machine."
    )
    print(image_count_summary(args))
    print(
        "Exact numbers may vary with operation, dtype, image count, shape, "
        "CPU allocation, memory bandwidth, and machine load."
    )
    for index, (label, workload_args) in enumerate(workloads, start=1):
        if len(workloads) > 1:
            print()
            print(
                f"## Workload {index}/{len(workloads)}: {label} (n={workload_args.n})"
            )
        print_results(
            run_sweep(workload_args),
            workload_args,
            serial_result=run_serial_candidate(workload_args),
            workload_label=label,
        )


if __name__ == "__main__":
    main()
