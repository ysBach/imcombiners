"""Thread-tuning benchmark helper tests."""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

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
    assert "export IMCOMBINERS_PARALLEL_THRESHOLD=4097" in out


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
    assert "Fastest observed median: RAYON_NUM_THREADS=16" in out
    assert "Recommended measured mode: parallel" in out
    assert "Recommended RAYON_NUM_THREADS=8" in out
    assert "within the near-tie tolerance" in out
    assert "speedup_vs_serial" in out
    assert "export RAYON_NUM_THREADS=8" in out
    assert "export IMCOMBINERS_PARALLEL_THRESHOLD=4096" in out
    assert "export IMCOMBINERS_PARALLEL_THRESHOLD=1" not in out
    assert "parallel candidates were measured with threshold=1" in out
    assert "Do not use threshold=1 as a global default" in out
    assert "Practical threshold starting point for this output size: 4096" in out
    assert "imc.set_rayon_num_threads(8)" in out
    assert "imc.set_parallel_threshold(4096)" in out
    assert "print(imc.get_rayon_num_threads())" in out
    assert "print(imc.get_parallel_threshold())" in out


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
    assert "export IMCOMBINERS_PARALLEL_THRESHOLD=4097" in out
    assert "imc.set_rayon_num_threads(" not in out
    assert "imc.set_parallel_threshold(4097)" in out


def test_parallel_threshold_is_user_tunable():
    original = imc.get_parallel_threshold()
    try:
        imc.set_parallel_threshold(7)
        assert imc.get_parallel_threshold() == 7
        assert imc.kernels.get_parallel_threshold() == 7
    finally:
        imc.set_parallel_threshold(original)


def test_minmax_1d_parallel_threshold_is_user_tunable():
    original = imc.get_minmax_1d_parallel_threshold()
    try:
        imc.set_minmax_1d_parallel_threshold(1234)
        assert imc.get_minmax_1d_parallel_threshold() == 1234
        assert imc.kernels.get_minmax_1d_parallel_threshold() == 1234
    finally:
        imc.set_minmax_1d_parallel_threshold(original)


def test_rayon_thread_functions_are_exported_and_validate_values():
    assert isinstance(imc.get_rayon_num_threads(), int)
    assert imc.get_rayon_num_threads() >= 1
    assert imc.kernels.get_rayon_num_threads() == imc.get_rayon_num_threads()

    for value in (0, -1):
        try:
            imc.set_rayon_num_threads(value)
        except ValueError:
            pass
        else:  # pragma: no cover - should not happen once API exists
            raise AssertionError("set_rayon_num_threads accepted nonpositive value")

    for value in (1.2, "8"):
        try:
            imc.set_rayon_num_threads(value)  # type: ignore[arg-type]
        except TypeError:
            pass
        else:  # pragma: no cover - should not happen once API exists
            raise AssertionError("set_rayon_num_threads accepted non-integer value")


def test_parallel_threshold_rejects_nonpositive_values():
    original = imc.get_parallel_threshold()
    try:
        for value in (0, -1):
            try:
                imc.set_parallel_threshold(value)
            except ValueError:
                pass
            else:  # pragma: no cover - should not happen once API exists
                raise AssertionError(
                    "set_parallel_threshold accepted nonpositive value"
                )
        assert imc.get_parallel_threshold() == original
    finally:
        imc.set_parallel_threshold(original)


def test_parallel_threshold_rejects_non_integer_values():
    original = imc.get_parallel_threshold()
    try:
        for value in (1.2, "8"):
            try:
                imc.set_parallel_threshold(value)  # type: ignore[arg-type]
            except TypeError:
                pass
            else:  # pragma: no cover - should not happen once API exists
                raise AssertionError(
                    "set_parallel_threshold accepted non-integer value"
                )
        assert imc.get_parallel_threshold() == original
    finally:
        imc.set_parallel_threshold(original)


def test_parallel_threshold_environment_variable_is_read_in_new_process():
    env = os.environ.copy()
    env["IMCOMBINERS_PARALLEL_THRESHOLD"] = "13"
    snippet = "import imcombiners as imc; print(imc.get_parallel_threshold())"

    result = subprocess.run(
        [sys.executable, "-c", snippet],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "13"


def test_minmax_1d_parallel_threshold_environment_variable_is_read_in_new_process():
    env = os.environ.copy()
    env["IMCOMBINERS_1D_MINMAX_PARALLEL_THRESHOLD"] = "321"
    snippet = "import imcombiners as imc; print(imc.get_minmax_1d_parallel_threshold())"

    result = subprocess.run(
        [sys.executable, "-c", snippet],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "321"


def test_rayon_thread_count_can_be_set_in_new_process_before_use():
    snippet = (
        "import imcombiners as imc; "
        "imc.set_rayon_num_threads(2); "
        "print(imc.get_rayon_num_threads())"
    )

    result = subprocess.run(
        [sys.executable, "-c", snippet],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "2"


def test_rayon_thread_count_environment_variable_is_read_in_new_process():
    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = "3"
    snippet = "import imcombiners as imc; print(imc.get_rayon_num_threads())"

    result = subprocess.run(
        [sys.executable, "-c", snippet],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "3"


def test_rayon_thread_count_set_fails_after_pool_initialization():
    snippet = (
        "import imcombiners as imc; "
        "imc.get_rayon_num_threads(); "
        "\ntry:\n"
        "    imc.set_rayon_num_threads(2)\n"
        "except RuntimeError as exc:\n"
        "    print(type(exc).__name__)\n"
        "else:\n"
        "    raise SystemExit('set_rayon_num_threads unexpectedly succeeded')\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", snippet],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "RuntimeError"


def test_parallel_threshold_does_not_change_combine_or_reject_results():
    rng = np.random.default_rng(20250311)
    arr = rng.normal(size=(7, 8, 9)).astype(np.float64)
    arr[0, 2, 3] = np.nan
    original = imc.get_parallel_threshold()
    try:
        imc.set_parallel_threshold(10_000)
        serial_mean = imc.mean(arr)
        serial_rej = imc.sigclip(arr, sigma=2.0, maxiters=2)

        imc.set_parallel_threshold(1)
        parallel_mean = imc.mean(arr)
        parallel_rej = imc.sigclip(arr, sigma=2.0, maxiters=2)

        np.testing.assert_allclose(parallel_mean, serial_mean, equal_nan=True)
        for parallel, serial in zip(parallel_rej, serial_rej, strict=True):
            np.testing.assert_array_equal(parallel, serial)
    finally:
        imc.set_parallel_threshold(original)
