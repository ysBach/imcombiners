"""Rust-backed image-stack combine + rejection kernels.

Four usage modes: pick the one that fits your call site:

1. **Standard Combiner** - recommended for ordinary Python workflows.
   ::

       cmb = Combiner(arr, mask=m)
       out = cmb.combine(
           "median",
           zero=zs,
           scale=sc,
           rejectors=SigClip(sigma=3, maxiters=5),
           diagnostics=None,
       )

2. **compact ndcombine() wrapper** - for IRAF-style or CLI-facing call sites.

3. **Chained Combiner** - use :meth:`Combiner.reject` and friends when you
   need retained diagnostics or step-by-step state.

4. **direct kernel calls** in :mod:`imcombiners.kernels` - focused primitives for
   already-prepared arrays.
"""

from __future__ import annotations

from . import _core, kernels
from ._combiner import Combiner
from ._diagnostics import (
    OutputFlags,
    SampleFlags,
)
from ._ndcombine import ndcombine
from ._offset import place_into_padded
from ._rejectors import (
    CcdClip,
    LinearClip,
    MinMaxClip,
    PClip,
    Rejector,
    SigClip,
)
from ._validation import resolve_zero_scale
from .kernels import (
    ccdclip,
    get_minmax_1d_parallel_threshold,
    get_parallel_threshold,
    get_rayon_num_threads,
    grow_mask,
    linearclip,
    lmedian,
    maximum,
    mean,
    median,
    minimum,
    minmax,
    pclip,
    percentiles,
    set_minmax_1d_parallel_threshold,
    set_parallel_threshold,
    set_rayon_num_threads,
    sigclip,
    summation,
    variance,
    weighted_average,
)

__all__ = [
    # Fluent layer
    "Combiner",
    # Rejection objects
    "Rejector",
    "SigClip",
    "CcdClip",
    "LinearClip",
    "MinMaxClip",
    "PClip",
    "SampleFlags",
    "OutputFlags",
    # Modules
    "kernels",
    # Compat
    "ndcombine",
    "place_into_padded",
    # Combine functions
    "mean",
    "median",
    "lmedian",
    "summation",
    "minimum",
    "maximum",
    "variance",
    "weighted_average",
    # Reject functions
    "sigclip",
    "ccdclip",
    "linearclip",
    "minmax",
    "pclip",
    "percentiles",
    "grow_mask",
    # Zero/scale helper
    "resolve_zero_scale",
    # Parallel controls
    "get_rayon_num_threads",
    "set_rayon_num_threads",
    "get_parallel_threshold",
    "set_parallel_threshold",
    "get_minmax_1d_parallel_threshold",
    "set_minmax_1d_parallel_threshold",
    "__version__",
]

__version__ = _core.__version__
