"""Rejection algorithm dataclasses.

Each rejector is an immutable bundle of the parameters for one rejection
algorithm. It knows how to apply itself to a stack via :meth:`apply`, which
calls the corresponding kernel and returns the standard 6-tuple
``(mask_rej, std, low, upp, nit, output_flags)``.

These exist for two reasons:

- They keep parameter sets *typed and reusable*: build a ``SigClip(...)`` once,
  reuse across many stacks.
- They are the natural argument to :meth:`imcombiners.Combiner.reject`, which
  dispatches purely on the rejector's type; no string matching at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import _doc
from ._typing import RejectionResult
from ._validation import validate_positive_scalar

__all__ = [
    "Rejector",
    "SigClip",
    "CcdClip",
    "LinearClip",
    "MinMaxClip",
    "PClip",
]


@dataclass(frozen=True)
class Rejector:
    """Base class for rejection objects."""

    def apply(
        self,
        arr: np.ndarray,
        mask: np.ndarray | None = None,
        *,
        validate: bool = True,
    ) -> RejectionResult:  # pragma: no cover
        """Apply this rejection algorithm to an image stack."""
        raise NotImplementedError


@dataclass(frozen=True)
class SigClip(Rejector):
    """Iterative sigma-clipping rejection."""

    sigma: tuple[float, float] | float = (3.0, 3.0)
    maxiters: int = 5
    cenfunc: str = "median"
    clip_cen: str | None = None
    stdfunc: str = "std"
    ddof: int = 0
    nkeep: int = 1
    maxrej: int | None = None
    revert_on_nkeep: bool = True
    grow: float | None = None

    def _sigma_pair(self) -> tuple[float, float]:
        """Return `(lower, upper)` sigma thresholds as floats."""
        s = self.sigma
        return (
            (float(s), float(s))
            if isinstance(s, (int, float))
            else (float(s[0]), float(s[1]))
        )

    def apply(
        self,
        arr: np.ndarray,
        mask: np.ndarray | None = None,
        *,
        validate: bool = True,
    ) -> RejectionResult:
        """Apply sigma-clipping rejection to `arr`."""
        from .kernels import sigclip

        return sigclip(
            arr,
            mask=mask,
            sigma=self._sigma_pair(),
            maxiters=self.maxiters,
            ddof=self.ddof,
            nkeep=self.nkeep,
            maxrej=self.maxrej,
            cenfunc=self.cenfunc,
            clip_cen=self.clip_cen,
            stdfunc=self.stdfunc,
            revert_on_nkeep=self.revert_on_nkeep,
            grow=self.grow,
            validate=validate,
        )


@dataclass(frozen=True)
class CcdClip(Rejector):
    """CCD noise-model clipping."""

    sigma: tuple[float, float] | float = (3.0, 3.0)
    maxiters: int = 5
    cenfunc: str = "median"
    clip_cen: str | None = None
    ddof: int = 0
    nkeep: int = 1
    maxrej: int | None = None
    revert_on_nkeep: bool = True
    rdnoise: float = 0.0
    gain: float = 1.0
    snoise: float = 0.0
    scales: tuple[float, ...] | None = None
    zeros: tuple[float, ...] | None = None
    grow: float | None = None

    def _sigma_pair(self) -> tuple[float, float]:
        """Return `(lower, upper)` sigma thresholds as floats."""
        s = self.sigma
        return (
            (float(s), float(s))
            if isinstance(s, (int, float))
            else (float(s[0]), float(s[1]))
        )

    def apply(
        self,
        arr: np.ndarray,
        mask: np.ndarray | None = None,
        *,
        validate: bool = True,
    ) -> RejectionResult:
        """Apply CCD noise-model clipping to `arr`."""
        from .kernels import ccdclip

        gain = (
            validate_positive_scalar("gain", self.gain)
            if validate
            else float(self.gain)
        )
        return ccdclip(
            arr,
            mask=mask,
            sigma=self._sigma_pair(),
            maxiters=self.maxiters,
            ddof=self.ddof,
            nkeep=self.nkeep,
            maxrej=self.maxrej,
            cenfunc=self.cenfunc,
            clip_cen=self.clip_cen,
            revert_on_nkeep=self.revert_on_nkeep,
            rdnoise=self.rdnoise,
            gain=gain,
            snoise=self.snoise,
            scales=self.scales,
            zeros=self.zeros,
            grow=self.grow,
            validate=validate,
        )


@dataclass(frozen=True)
class LinearClip(Rejector):
    """Reject values outside center-relative linear bounds."""

    low_scale: float = 1.0
    low: float = 0.0
    upp_scale: float = 1.0
    upp: float = 0.0
    maxiters: int = 1
    cenfunc: str = "median"
    nkeep: int = 1
    maxrej: int | None = None
    revert_on_nkeep: bool = True
    grow: float | None = None

    def apply(
        self,
        arr: np.ndarray,
        mask: np.ndarray | None = None,
        *,
        validate: bool = True,
    ) -> RejectionResult:
        """Apply center-relative linear clipping to `arr`."""
        from .kernels import linearclip

        return linearclip(
            arr,
            mask=mask,
            low_scale=self.low_scale,
            low=self.low,
            upp_scale=self.upp_scale,
            upp=self.upp,
            maxiters=self.maxiters,
            cenfunc=self.cenfunc,
            nkeep=self.nkeep,
            maxrej=self.maxrej,
            revert_on_nkeep=self.revert_on_nkeep,
            grow=self.grow,
            validate=validate,
        )


@dataclass(frozen=True)
class MinMaxClip(Rejector):
    """Reject the smallest and largest unmasked values at each pixel."""

    n_min: int | float = 1
    n_max: int | float = 1
    grow: float | None = None

    def apply(
        self,
        arr: np.ndarray,
        mask: np.ndarray | None = None,
        *,
        validate: bool = True,
    ) -> RejectionResult:
        """Apply min/max rejection to `arr`."""
        from .kernels import minmax

        return minmax(
            arr,
            mask=mask,
            n_min=self.n_min,
            n_max=self.n_max,
            grow=self.grow,
            validate=validate,
        )


@dataclass(frozen=True)
class PClip(Rejector):
    """IRAF-style percentile clipping."""

    frac: float = -0.5
    sigma: float | tuple[float, float] = 3.0
    nkeep: int = 1
    grow: float | None = None

    def apply(
        self,
        arr: np.ndarray,
        mask: np.ndarray | None = None,
        *,
        validate: bool = True,
    ) -> RejectionResult:
        """Apply percentile-clipping rejection to `arr`."""
        from .kernels import pclip

        return pclip(
            arr,
            mask=mask,
            frac=self.frac,
            sigma=self.sigma,
            nkeep=self.nkeep,
            grow=self.grow,
            validate=validate,
        )


_doc.install_rejector_docstrings(globals())
