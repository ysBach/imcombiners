"""Environment reporting helpers for benchmark scripts."""

from __future__ import annotations

import os
import platform
import sys
from collections.abc import Iterable, Mapping
from importlib import metadata

DEFAULT_PACKAGES = (
    "imcombiners",
    "reducers",
    "numpy",
    "bottleneck",
    "astropy",
    "ccdproc",
    "fitsio",
)


def package_version(name: str) -> str:
    """Return an installed package version or a clear missing marker."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def collect_environment(packages: Iterable[str] = DEFAULT_PACKAGES) -> dict[str, str]:
    """Return benchmark environment metadata as display-ready strings."""
    processor = platform.processor() or "unknown"
    logical_cpus = os.cpu_count()
    rows = {
        "Python": sys.version.split()[0],
        "Python executable": sys.executable,
        "Kernel/OS": platform.platform(),
        "Machine": platform.machine() or "unknown",
        "Processor": processor,
        "Logical CPUs": "unknown" if logical_cpus is None else str(logical_cpus),
    }
    rows.update({name: package_version(name) for name in packages})
    try:
        from imcombiners import _core
    except ImportError:
        rows["reducers (Rust)"] = "unavailable"
        rows["reducers (Rust source)"] = "unavailable"
    else:
        rows["reducers (Rust)"] = getattr(_core, "__reducers_version__", "unknown")
        rows["reducers (Rust source)"] = getattr(
            _core, "__reducers_source__", "unknown"
        )
    return rows


def format_environment_markdown(rows: Mapping[str, str] | None = None) -> str:
    """Return a Markdown table with benchmark environment metadata."""
    rows = collect_environment() if rows is None else rows
    lines = [
        "## Environment",
        "",
        "| item | value |",
        "| --- | --- |",
    ]
    lines.extend(f"| {key} | {value} |" for key, value in rows.items())
    return "\n".join(lines)
