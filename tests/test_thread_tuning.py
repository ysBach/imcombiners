"""Thread-tuning benchmark helper tests."""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import reducers as rd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import imcombiners as imc  # noqa: E402

from benchmarks import benchmark_threads as bt  # noqa: E402


def test_candidate_threads_start_at_two_when_serial_baseline_exists():
    assert bt.candidate_threads(12) == list(range(2, 13))


def test_candidate_threads_requires_explicit_force_for_many_core_hosts():
    assert bt.candidate_threads(64) == []


def test_candidate_threads_can_force_sparse_sweep_for_many_core_hosts():
    assert bt.candidate_threads(64, force_large_sweep=True) == [2, 4, 8, 16, 32, 64]


def test_candidate_threads_still_dense_for_32_core_hosts():
    assert bt.candidate_threads(32) == list(range(2, 33))


def test_candidate_threads_keep_one_on_single_cpu_hosts():
    assert bt.candidate_threads(1) == []


def test_parse_args_defaults_to_few_and_many_image_workloads():
    args = bt.parse_args([])

    assert args.n_values == [5, 31]
    assert args.n == 5


def test_parse_args_accepts_custom_image_count_workloads():
    args = bt.parse_args(["--n", "7", "15"])

    assert args.n_values == [7, 15]
    assert args.n == 7


def test_parse_args_worker_requires_one_image_count():
    try:
        bt.parse_args(["--worker", "--n", "5", "31"])
    except SystemExit:
        pass
    else:  # pragma: no cover - should not happen once parser validation exists
        raise AssertionError("worker accepted multiple --n values")


def test_iter_workloads_labels_few_and_many_image_defaults():
    args = bt.parse_args([])

    workloads = list(bt.iter_workloads(args))

    assert [(label, workload.n) for label, workload in workloads] == [
        ("few images", 5),
        ("many images", 31),
    ]


def test_image_count_summary_reports_actual_quick_workload():
    args = bt.parse_args(["--quick"])

    assert bt.image_count_summary(args) == (
        "Image-count workloads for this run: n=5 (few images)."
    )


def test_image_count_summary_reports_default_workloads():
    args = bt.parse_args([])

    assert bt.image_count_summary(args) == (
        "Default image-count workloads are n=5 (few-image combinations) "
        "and n=31 (many-image combinations)."
    )


def test_print_results_warns_that_recommendations_are_workload_specific(capsys):
    results = [bt.ThreadResult(threads=2, median_ms=5.0, min_ms=4.8, max_ms=5.2)]
    serial = bt.ThreadResult(threads=0, median_ms=10.0, min_ms=9.8, max_ms=10.2)
    args = bt.parse_args(["--n", "5", "--height", "64", "--width", "64"])

    bt.print_results(results, args, serial_result=serial, workload_label="few images")

    out = capsys.readouterr().out
    assert "# workload case: few images" in out
    assert "Exact numbers can vary with op, dtype, image count" in out
    assert "shape, CPU allocation, memory bandwidth, and machine load" in out


def test_print_results_handles_no_parallel_candidates(capsys):
    serial = bt.ThreadResult(threads=0, median_ms=10.20, min_ms=9.9, max_ms=10.5)
    args = bt.parse_args(["--height", "64", "--width", "64"])

    bt.print_results([], args, serial_result=serial)

    out = capsys.readouterr().out
    assert "No parallel thread candidates were measured." in out
    assert "Recommended measured mode: serial" in out
    assert "export RAYON_NUM_THREADS=" not in out
    assert "IMCOMBINERS_PARALLEL_THRESHOLD" not in out


def test_best_result_prefers_lowest_median_then_lower_thread_count():
    results = [
        bt.ThreadResult(threads=1, median_ms=10.0, min_ms=9.0, max_ms=11.0),
        bt.ThreadResult(threads=2, median_ms=8.0, min_ms=7.0, max_ms=9.0),
        bt.ThreadResult(threads=4, median_ms=8.0, min_ms=7.5, max_ms=8.5),
    ]

    assert bt.best_result(results).threads == 2


def test_best_result_prefers_lower_thread_count_within_tolerance():
    results = [
        bt.ThreadResult(threads=8, median_ms=10.10, min_ms=9.8, max_ms=10.5),
        bt.ThreadResult(threads=16, median_ms=10.00, min_ms=9.7, max_ms=10.4),
        bt.ThreadResult(threads=32, median_ms=10.02, min_ms=9.6, max_ms=10.7),
    ]

    assert bt.best_result(results, tie_tolerance=0.02).threads == 8


def test_best_result_uses_fastest_when_outside_tolerance():
    results = [
        bt.ThreadResult(threads=8, median_ms=11.00, min_ms=10.8, max_ms=11.2),
        bt.ThreadResult(threads=16, median_ms=10.00, min_ms=9.8, max_ms=10.2),
    ]

    assert bt.best_result(results, tie_tolerance=0.02).threads == 16


def test_speedup_is_relative_to_single_thread():
    baseline = bt.ThreadResult(threads=1, median_ms=12.0, min_ms=11.0, max_ms=13.0)
    faster = bt.ThreadResult(threads=4, median_ms=3.0, min_ms=2.8, max_ms=3.2)

    assert math.isclose(bt.speedup(faster, baseline), 4.0)


def test_json_from_stdout_uses_last_nonempty_line():
    assert bt._json_from_stdout('warning\n\n{"median_ms": 1.0}\n') == {"median_ms": 1.0}


def test_print_results_includes_copyable_shell_and_python_snippets(capsys):
    results = [
        bt.ThreadResult(threads=8, median_ms=10.10, min_ms=9.8, max_ms=10.5),
        bt.ThreadResult(threads=16, median_ms=10.00, min_ms=9.7, max_ms=10.4),
    ]
    serial = bt.ThreadResult(threads=0, median_ms=30.0, min_ms=29.0, max_ms=31.0)
    args = bt.parse_args(["--threads", "8", "16", "--height", "64", "--width", "64"])

    bt.print_results(results, args, serial_result=serial)

    out = capsys.readouterr().out
    assert "Fastest observed parallel median: RAYON_NUM_THREADS=16" in out
    assert "Recommended measured mode: parallel" in out
    assert "Recommended RAYON_NUM_THREADS=8" in out
    assert "within the near-tie tolerance" in out
    assert "speedup_vs_serial" in out
    assert "export RAYON_NUM_THREADS=8" in out
    assert "IMCOMBINERS_PARALLEL_THRESHOLD" not in out
    assert "rejection kernels use imc's internal rejection-parallel policy" in out
    assert "import reducers as rd" in out
    assert "rd.get_num_threads()" in out
    assert "imc.set_rayon_num_threads" not in out
    assert "imc.set_parallel_threshold" not in out


def test_print_results_recommends_serial_when_serial_is_near_tied(capsys):
    results = [
        bt.ThreadResult(threads=8, median_ms=10.10, min_ms=9.8, max_ms=10.5),
        bt.ThreadResult(threads=16, median_ms=10.00, min_ms=9.7, max_ms=10.4),
    ]
    serial = bt.ThreadResult(threads=0, median_ms=10.20, min_ms=9.9, max_ms=10.5)
    args = bt.parse_args(["--threads", "8", "16", "--height", "64", "--width", "64"])

    bt.print_results(results, args, serial_result=serial)

    out = capsys.readouterr().out
    assert "Recommended measured mode: serial" in out
    assert "speedup_vs_serial" in out
    assert "export RAYON_NUM_THREADS=" not in out
    assert "IMCOMBINERS_PARALLEL_THRESHOLD" not in out
    assert "imc.set_rayon_num_threads(" not in out
    assert "imc.set_parallel_threshold" not in out


def test_reducer_parallel_grains_are_user_tunable():
    original = rd.get_parallel_grains()
    try:
        rd.set_parallel_grain("axis_scan_nan", 7)
        assert rd.get_parallel_grains()["axis_scan_nan"] == 7
    finally:
        rd.set_parallel_grains(original)


def test_reducer_thread_function_reports_rayon_pool_size():
    assert isinstance(rd.get_num_threads(), int)
    assert rd.get_num_threads() >= 1


def test_rayon_thread_count_environment_variable_is_reported_by_reducers():
    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = "3"
    snippet = "import reducers as rd; print(rd.get_num_threads())"

    result = subprocess.run(
        [sys.executable, "-c", snippet],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "3"


def test_reducer_parallel_grains_do_not_change_combine_or_reject_results():
    rng = np.random.default_rng(20250311)
    arr = rng.normal(size=(7, 8, 9)).astype(np.float64)
    arr[0, 2, 3] = np.nan
    original = rd.get_parallel_grains()
    try:
        rd.set_parallel_grains({key: 10**12 for key in original})
        serial_mean = imc.ndcombine(arr, combine="mean")
        serial_rej = imc.sigclip(arr, sigma=2.0, maxiters=2)

        rd.set_parallel_grains({key: 1 for key in original})
        parallel_mean = imc.ndcombine(arr, combine="mean")
        parallel_rej = imc.sigclip(arr, sigma=2.0, maxiters=2)

        np.testing.assert_allclose(parallel_mean, serial_mean, equal_nan=True)
        for parallel, serial in zip(parallel_rej, serial_rej, strict=True):
            np.testing.assert_array_equal(parallel, serial)
    finally:
        rd.set_parallel_grains(original)
