"""Regression tests for benchmark helper semantics."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from imcombiners import _core

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks import _environment  # noqa: E402


def _load_benchmark_combine():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "benchmark_combine.py"
    spec = importlib.util.spec_from_file_location("benchmark_combine", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


bench = _load_benchmark_combine()


def test_format_environment_markdown_includes_versions_and_kernel():
    report = _environment.format_environment_markdown(
        {
            "Python": "3.13.0",
            "Kernel/OS": "macOS-26.4.1-arm64",
            "Machine": "arm64",
            "Logical CPUs": "12",
            "numpy": "2.4.6",
            "bottleneck": "1.6.0",
        }
    )

    assert "## Environment" in report
    assert "| Python | 3.13.0 |" in report
    assert "| Kernel/OS | macOS-26.4.1-arm64 |" in report
    assert "| bottleneck | 1.6.0 |" in report


def test_environment_distinguishes_compiled_and_python_reducers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The Python package may be independently upgraded after imc was built.
    monkeypatch.setattr(_environment.metadata, "version", lambda name: "999.0.0")

    environment = _environment.collect_environment(packages=("reducers",))

    assert environment["reducers"] == "999.0.0"
    assert _core.__reducers_version__ != "999.0.0"
    assert environment["reducers (Rust)"] == _core.__reducers_version__
    assert environment["reducers (Rust source)"] == _core.__reducers_source__


@pytest.mark.parametrize(
    ("op", "combine"), [("sigclip_mean", "mean"), ("sigclip_median", "median")]
)
def test_optimized_sigclip_uses_fused_kernel(monkeypatch, op, combine):
    stack = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    mask = np.zeros_like(stack, dtype=bool)
    calls: list[dict[str, object]] = []

    def fake_sigclip_combine(arr, **kwargs):
        calls.append({"arr": arr, **kwargs})
        return np.zeros(arr.shape[1:], dtype=np.float32)

    monkeypatch.setattr(bench.imck, "sigclip_combine", fake_sigclip_combine)
    monkeypatch.setattr(
        bench.imck,
        "sigclip",
        lambda *args, **kwargs: pytest.fail("diagnostic sigclip path was used"),
    )
    monkeypatch.setattr(
        bench.np,
        "where",
        lambda *args, **kwargs: pytest.fail("sigclip path materialized NaNs"),
    )

    result = bench.imcombiners_optimized(stack, op, mask)

    assert result.shape == stack.shape[1:]
    assert len(calls) == 1
    assert calls[0]["arr"] is stack
    assert calls[0]["mask"] is mask
    assert calls[0]["combine"] == combine
    assert calls[0]["validate"] is False


@pytest.mark.parametrize("op", ["mean", "median", "sigclip_mean", "sigclip_median"])
def test_combiner_benchmark_paths_match_optimized(op):
    stack = np.arange(7 * 6 * 5, dtype=np.float32).reshape(7, 6, 5)
    stack[0, ::2, ::2] += 1000.0
    mask = np.zeros_like(stack, dtype=bool)
    mask[1, 1::2, 1::2] = True

    opt = bench.imcombiners_optimized(stack, op, mask)

    np.testing.assert_allclose(
        bench.imcombiners_public(stack, op, mask),
        opt,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        bench.imcombiners_chain(stack, op, mask),
        opt,
        rtol=1e-6,
        atol=1e-6,
    )
