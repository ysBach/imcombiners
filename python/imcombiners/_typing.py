"""Internal typing aliases for the Python API layer."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

PlaneStatistic: TypeAlias = str | Callable[[NDArray], float]
PlaneVectorLike: TypeAlias = ArrayLike | PlaneStatistic | None
SigclipStatKwargs: TypeAlias = Mapping[str, object] | None
RejectionResult: TypeAlias = tuple[
    NDArray[np.bool_],
    NDArray[np.floating] | None,
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.uint8],
    NDArray[np.uint8],
]
FullCombineResult: TypeAlias = tuple[
    NDArray,
    NDArray[np.bool_] | None,
    NDArray[np.bool_] | None,
    NDArray[np.floating] | None,
    NDArray[np.floating] | None,
    NDArray[np.floating] | None,
    NDArray[np.uint8] | None,
    NDArray[np.uint8] | None,
]
FullDiagnosticCombineResult: TypeAlias = tuple[
    NDArray,
    NDArray[np.bool_] | None,
    NDArray[np.bool_] | None,
    NDArray[np.floating] | None,
    NDArray[np.floating] | None,
    NDArray[np.floating] | None,
    NDArray[np.uint8] | None,
    NDArray[np.uint8] | None,
    NDArray[np.uint8],
]
