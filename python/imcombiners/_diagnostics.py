"""Diagnostic-level and flag helpers."""

from __future__ import annotations

from enum import IntFlag
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

Diagnostics: TypeAlias = Literal["simple", "full"] | None


class SampleFlags(IntFlag):
    """Per-sample rejection/provenance flags.

    These bits apply to stack-shaped `sample_flags` arrays, not to the
    per-output-element `output_flags` diagnostic map.

    Notes
    -----
    Values are ``INPUT_MASK=1``, ``NONFINITE=2``, ``THRESHOLD=4``,
    ``ALGORITHM=8``, ``GROW=16``, ``PREVIOUS=32``, ``RESTORED_NKEEP=64``,
    and ``RESTORED_MAXREJ=128``. Multiple causes are combined by bitwise OR.
    `ALGORITHM` marks new algorithm rejections; `GROW` marks added neighbors.
    The restoration bits record tentative rejections undone by safeguards.
    """

    INPUT_MASK = 1
    NONFINITE = 2
    THRESHOLD = 4
    ALGORITHM = 8
    GROW = 16
    PREVIOUS = 32
    RESTORED_NKEEP = 64
    RESTORED_MAXREJ = 128


class OutputFlags(IntFlag):
    """Per-output rejection status flags.

    These bits apply to spatial-shaped `output_flags` arrays, not to the
    stack-shaped `sample_flags` diagnostic arrays.

    Notes
    -----
    Values are ``PREMASKED=1``, ``MAXITERS=2``, ``NKEEP=4``, ``MAXREJ=8``,
    and ``GROW=16``. Multiple conditions are combined by bitwise OR.
    Zero means normal completion and does not imply that no samples were
    rejected. Use `sample_flags` to inspect individual sample causes.
    """

    PREMASKED = 1
    MAXITERS = 2
    NKEEP = 4
    MAXREJ = 8
    GROW = 16


def normalize_diagnostics(diagnostics: Diagnostics) -> Diagnostics:
    """Validate and return a diagnostic product level."""
    if diagnostics not in (None, "simple", "full"):
        raise ValueError("diagnostics must be None, 'simple', or 'full'")
    return diagnostics


def normalize_reject_diagnostics(diagnostics: Diagnostics) -> Literal["simple", "full"]:
    """Validate diagnostics for stateful rejection calls."""
    if diagnostics is None:
        return "simple"
    if diagnostics not in ("simple", "full"):
        raise ValueError("diagnostics must be None, 'simple', or 'full'")
    return diagnostics


def sample_flags_array(
    arr: NDArray,
    *,
    input_mask: NDArray[np.bool_] | None = None,
    threshold_mask: NDArray[np.bool_] | None = None,
    previous_mask: NDArray[np.bool_] | None = None,
    mask_rej: NDArray[np.bool_] | None = None,
    grown_mask: NDArray[np.bool_] | None = None,
    restored_flags: NDArray[np.uint8] | None = None,
) -> NDArray[np.uint8]:
    """Build a per-sample uint8 diagnostic flag array."""
    flags = np.zeros(arr.shape, dtype=np.uint8)

    if input_mask is not None:
        flags[input_mask] |= np.uint8(SampleFlags.INPUT_MASK)
    nonfinite = ~np.isfinite(arr)
    if np.any(nonfinite):
        flags[nonfinite] |= np.uint8(SampleFlags.NONFINITE)
    if threshold_mask is not None:
        flags[threshold_mask] |= np.uint8(SampleFlags.THRESHOLD)
    if previous_mask is not None:
        flags[previous_mask] |= np.uint8(SampleFlags.PREVIOUS)

    unavailable = nonfinite.copy()
    for mask in (input_mask, threshold_mask, previous_mask):
        if mask is not None:
            unavailable |= mask

    if mask_rej is not None:
        algorithm = mask_rej & ~unavailable
        if np.any(algorithm):
            flags[algorithm] |= np.uint8(SampleFlags.ALGORITHM)

    if grown_mask is not None and mask_rej is not None:
        added = grown_mask & ~mask_rej
        if np.any(added):
            flags[added] |= np.uint8(SampleFlags.GROW)

    if restored_flags is not None:
        flags |= restored_flags

    return flags
