"""Tests for the 1-D benchmark output contract."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_benchmark_module():
    path = Path(__file__).parents[1] / "benchmarks" / "benchmark_1d.py"
    spec = importlib.util.spec_from_file_location("benchmark_1d", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_print_table_pivots_timings_by_library(capsys):
    bench = _load_benchmark_module()
    results = [
        bench.Result(10, "float64", "median", "numpy", 4.0),
        bench.Result(10, "float64", "median", "bottleneck", 2.0),
        bench.Result(10, "float64", "median", "imc", 1.0),
    ]

    bench.print_table(results)

    out = capsys.readouterr().out.splitlines()
    assert (
        out[0]
        == "| length | dtype | op | np (us) | bn (us) | imc (us) | np/imc | bn/imc |"
    )
    assert (
        out[2]
        == "| 10 | float64 | median | 4000.00 | 2000.00 | 1000.00 | 4.0x | 2.0x |"
    )


def test_length_labels_use_powers_of_ten(capsys):
    bench = _load_benchmark_module()
    bench.print_table([bench.Result(10_000_000, "float64", "median", "imc", 1.0)])

    out = capsys.readouterr().out.splitlines()
    assert out[2].startswith("| 10^7 |")


def test_timings_and_speedups_use_fixed_precision(capsys):
    bench = _load_benchmark_module()
    results = [
        bench.Result(100, "float64", "mean", "numpy", 0.0042),
        bench.Result(100, "float64", "mean", "bottleneck", 0.000055),
        bench.Result(100, "float64", "mean", "imc", 0.0002),
    ]

    bench.print_table(results)

    out = capsys.readouterr().out.splitlines()
    assert out[2] == "| 10^2 | float64 | mean | 4.20 | 0.06 | 0.20 | 21.0x | 0.3x |"


def test_default_lengths_start_at_10_squared():
    bench = _load_benchmark_module()

    assert bench.DEFAULT_LENGTHS == (100, 10_000, 10_000_000)


def test_variance_operation_is_named_var():
    bench = _load_benchmark_module()
    values = bench.make_values(10, "float64")

    assert "var" in bench.OPS
    assert "variance" not in bench.OPS
    assert set(bench.functions_for(values, "var")) >= {"numpy", "imc"}


def test_minmax_benchmark_operations_are_available():
    bench = _load_benchmark_module()
    values = bench.make_values(10, "float64")

    assert "min" in bench.OPS
    assert "max" in bench.OPS
    assert set(bench.functions_for(values, "min")) >= {"numpy", "imc"}
    assert set(bench.functions_for(values, "max")) >= {"numpy", "imc"}
