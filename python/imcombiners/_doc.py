"""Runtime docstring templates for public Python APIs."""

from __future__ import annotations

from collections.abc import MutableMapping

_STACK_PARAM = """arr : ndarray, shape (N, *spatial)
    Image stack. Accepted dtypes are `uint8`, `uint16`, `int16`, `int32`,
    `float32`, and `float64`. Integer inputs are promoted to the package's
    floating workspace when `validate` is `True`. Inputs with more than
    3 dimensions are flattened internally; output shapes match the trailing
    spatial dimensions of the input."""

_STACK_PARAM_SHORT = """arr : ndarray, shape (N, *spatial)
    Image stack. Inputs with more than 3 dimensions are flattened internally;
    output shapes match the trailing spatial dimensions of the input."""

_VALIDATE_PARAM = """validate : bool, optional
    If `True`, check dimensionality and normalize dtype/contiguity before
    entering the Rust kernel. If `False`, callers must provide inputs that
    satisfy the compiled kernel assumptions."""

_MASK_PARAM = """mask : ndarray of bool, optional
    Input mask; `True` means already masked. Must have the same shape as `arr`
    before any internal flattening."""

_GROW_PARAM = """grow : float or None, optional
    Optional radius in pixels used to grow `mask_rej` spatially after
    rejection. Axis 0 is the stack axis and is never grown across. `None`
    disables growth and skips the extra calculation."""

_ITERATIVE_PARAMS = """sigma : float or tuple of float
    Clipping threshold in units of the per-pixel std. A scalar applies the same
    value to both tails; a 2-tuple ``(sigma_lower, sigma_upper)`` clips
    asymmetrically.
maxiters : int
    Maximum number of clipping iterations per pixel.
ddof : int
    Delta degrees of freedom for the per-pixel std.
nkeep : int
    Minimum number of unmasked values to preserve at each pixel. An iteration
    that would drop below this count is reverted in full. The default, `1`,
    prevents clipping from rejecting every finite sample in a per-output stack.
    Set `0` only when all samples may be rejected.
maxrej : int or None
    Maximum number of values that may be rejected at each pixel. `None` means
    no limit.
cenfunc : {"median", "lmedian", "mean"}
    Center estimator used each iteration. `lmedian` accepts the alias `lmed`
    and uses the lower of the two middle samples for even valid counts.
clip_cen : {"median", "lmedian", "mean"} or None
    Center used when computing the spread for iterative rejection. `None` uses
    the same center as `cenfunc`; `lmedian` also accepts `lmed`.
revert_on_nkeep : bool
    If `True`, an iteration that would leave fewer than `nkeep` usable samples
    at a pixel is reverted in full for that pixel. Input masks, threshold
    masks, and non-finite samples remain excluded. The default is `True`.
    If `False`, clipping is strict and may leave fewer than `nkeep` samples."""

_REJECTION_RETURNS = """mask_rej : ndarray of bool, shape (N, *spatial)
    `True` where a value was rejected by this kernel.
    ``mask_rej.sum(axis=0)`` is the per-output rejected count.
std : ndarray, shape (*spatial) or None
    Per-pixel spread used by sigma/CCD clipping. `None` for rejection
    algorithms without a spread diagnostic.
low : ndarray, shape (*spatial)
    Lower retained-value bound. Bounds are inclusive: values equal to `low`
    are retained.
upp : ndarray, shape (*spatial)
    Upper retained-value bound. Bounds are inclusive: values equal to `upp`
    are retained.
nit : ndarray of uint8, shape (*spatial)
    Iteration-count map.
output_flags : ndarray of uint8, shape (*spatial)
    Bit-coded status. ``0`` means normal completion; bit ``1`` means at least
    one pre-masked sample at this output element; bit ``16`` means `grow` added
    at least one rejected sample at this output element. Iterative kernels also
    use bit ``2`` for maxiters, bit ``4`` for `nkeep`, and bit ``8`` for
    `maxrej`. Bits are OR-ed. `output_flags` is a per-output diagnostic; use
    ``mask_rej.sum(axis=0)`` to count samples rejected by this step."""


_COMBINE_SPECS = {
    "mean": (
        "Return the NaN-aware mean along the stack axis.",
        "",
        "mean : ndarray, shape (*spatial)\n"
        "    Per-pixel mean of finite values. All-NaN output elements return `NaN`.",
        "",
    ),
    "median": (
        "Return the NaN-aware median along the stack axis.",
        "",
        "median : ndarray, shape (*spatial)\n"
        "    Per-pixel median of finite values. For an even number of finite "
        "values, the two middle values are averaged. All-NaN output elements return "
        "`NaN`.",
        "",
    ),
    "lmedian": (
        "IRAF-style lower median along the stack axis.",
        "",
        "lmedian : ndarray, shape (*spatial)\n"
        "    Per-pixel lower median. For an even number of finite values, this "
        "is the lower of the two middle values rather than their average.",
        "Accepted integer dtypes (`uint8`, `uint16`, `int16`, `int32`) are "
        "passed through without promotion so the output dtype matches the "
        "input. Any path that needs NaN masking, rejection, zero, or scale "
        "should convert the stack to a floating workspace before calling this "
        "function.",
    ),
    "summation": (
        "Return the NaN-aware sum along the stack axis.",
        "",
        "sum : ndarray, shape (*spatial)\n"
        "    Per-pixel sum of finite values. All-NaN output elements return `NaN`.",
        "",
    ),
    "minimum": (
        "Return the NaN-aware minimum along the stack axis.",
        "",
        "minimum : ndarray, shape (*spatial)\n"
        "    Per-pixel minimum of finite values. All-NaN output elements return `NaN`.",
        "",
    ),
    "maximum": (
        "Return the NaN-aware maximum along the stack axis.",
        "",
        "maximum : ndarray, shape (*spatial)\n"
        "    Per-pixel maximum of finite values. All-NaN output elements return `NaN`.",
        "",
    ),
    "variance": (
        "Return the NaN-aware variance along the stack axis.",
        "ddof : int, optional\n"
        "    Delta degrees of freedom. The returned value is "
        "``sum((valid - mean)**2) / (nvalid - ddof)``. Pixels with "
        "``nvalid <= ddof`` return `NaN`.",
        "variance : ndarray, shape (*spatial)\n"
        "    Per-pixel variance of finite values. Use ``np.sqrt(var)`` if a "
        "standard-deviation or error-like map is needed.",
        "",
    ),
    "weighted_average": (
        "Return the NaN-aware weighted average along the stack axis.",
        "weights : ndarray, shape (N,)\n"
        "    Per-plane weights. The weighted sum skips non-finite pixels, so "
        "the effective denominator is the sum of weights for finite values at "
        "each output pixel.",
        "weighted_average : ndarray, shape (*spatial)\n"
        "    Per-pixel weighted average. Pixels with no finite weighted samples "
        "return `NaN`.",
        "",
    ),
}


_REJECTION_SPECS = {
    "sigclip": (
        "Sigma-clipping rejection.",
        _ITERATIVE_PARAMS,
        "",
    ),
    "ccdclip": (
        "CCD noise-model clipping.",
        _ITERATIVE_PARAMS + "\nrdnoise : float\n"
        "    Read noise in electrons.\n"
        "snoise : float\n"
        "    Sky-noise coefficient. Adds a Poisson-like contribution scaled by "
        "the local count level.\n"
        "scale_ref : float\n"
        "    Reference scale factor. Typically the inverse of the exposure-time "
        "normalization applied during reduction.\n"
        "zero_ref : float\n"
        "    Reference zero-point offset. Accounts for a known DC bias that "
        "shifts the effective noise level.",
        "The kernel evaluates CCD rejection on gain-corrected scratch values "
        "(electrons). The per-pixel noise threshold is approximately::\n\n"
        "    sqrt((1 + snoise) * abs(noise_center + zero_ref) * scale_ref + "
        "rdnoise**2)",
    ),
    "linearclip": (
        "Center-relative linear clipping (`low + low_scale * center <= value <= "
        "upp + upp_scale * center`).",
        "low_scale : float, optional\n"
        "    Multiplier for the per-pixel center used in the lower "
        "retained bound.\n"
        "low : float, optional\n"
        "    Non-negative margin subtracted from ``low_scale * center``.\n"
        "upp_scale : float, optional\n"
        "    Multiplier for the per-pixel center used in the upper "
        "retained bound.\n"
        "upp : float, optional\n"
        "    Non-negative margin added to ``upp_scale * center``.\n"
        "maxiters : int, optional\n"
        "    Maximum number of clipping iterations per pixel. `1` preserves "
        "the historical single-pass behavior.\n"
        "nkeep : int, optional\n"
        "    Minimum number of unmasked values to preserve at each pixel.\n"
        "maxrej : int or None, optional\n"
        "    Maximum number of values that may be rejected at each pixel. "
        "`None` means no limit.\n"
        'cenfunc : {"median", "lmedian", "mean"}, optional\n'
        "    Center estimator used each iteration. `lmedian` accepts the alias "
        "`lmed` and uses the lower of the two middle samples for even valid "
        "counts.\n"
        "revert_on_nkeep : bool, optional\n"
        "    If `True`, an iteration that would leave fewer than `nkeep` usable "
        "samples at a pixel is reverted in full for that pixel.",
        "Values survive when ``low_scale * center - low <= value <= "
        "upp_scale * center + upp``. Each iteration recomputes the center from "
        "finite, unmasked survivors and stops when no new samples are rejected "
        "or `maxiters` is reached. The default bounds ``(1, 0, 1, 0)`` are a "
        "no-op.",
    ),
    "minmax": (
        "Reject the `n_min` smallest and `n_max` largest unmasked values at "
        "each pixel. Single-pass.",
        "n_min : int or float\n"
        "    Number of minimum-side values to reject at each pixel. Values "
        "``>= 1`` are treated as a frame count. Values in ``[0, 1)`` are "
        "treated as a fraction of the total frame count ``N`` and converted via "
        "``int(N * n_min + 0.001)``, matching IRAF's internal fraction "
        "arithmetic.\n"
        "n_max : int or float\n"
        "    Same convention as `n_min`, applied to the maximum-side tail.",
        "",
    ),
    "pclip": (
        "IRAF-style percentile clipping at each pixel.",
        "frac : float, optional\n"
        "    IRAF `pclip` value. If ``abs(frac) < 1``, it is converted to an "
        "integer rank offset using half of the input image count. Positive "
        "values estimate sigma from the high side of the sorted median; "
        "negative values estimate sigma from the low side.\n"
        "sigma : float or tuple of float, optional\n"
        "    Lower and upper clipping thresholds applied to the pclip-estimated "
        "sigma. This maps to IRAF `lsigma` and `hsigma`.\n"
        "nkeep : int, optional\n"
        "    Minimum number of unmasked samples to retain after pclip rejection.",
        "This follows IRAF `imcombine` pclip semantics: choose a sorted rank "
        "offset from the median, use that sample's distance from the median as "
        "sigma, then apply lower and upper sigma thresholds.",
    ),
}


_REJECTOR_CLASS_DOCS = {
    "SigClip": ("Iterative sigma-clipping rejection.", _ITERATIVE_PARAMS),
    "CcdClip": (
        "CCD noise-model clipping.",
        _ITERATIVE_PARAMS + "\nrdnoise : float\n"
        "    Read noise in electrons.\n"
        "gain : float\n"
        "    CCD gain in electrons per DN. Rejection is evaluated on "
        "gain-corrected values equivalent to ``arr / gain`` without requiring "
        "callers to pre-divide the input stack.\n"
        "snoise : float\n"
        "    Sky-noise coefficient. Adds a Poisson-like contribution scaled by "
        "the local count level.\n"
        "scale_ref : float\n"
        "    Reference scale factor. Typically the inverse of the exposure-time "
        "normalization applied during reduction.\n"
        "zero_ref : float\n"
        "    Reference zero-point offset. Accounts for a known DC bias that "
        "shifts the effective noise level.",
    ),
    "LinearClip": (
        "Center-relative linear clipping.",
        "low_scale : float\n"
        "    Multiplier for the per-pixel center used in the lower "
        "retained bound.\n"
        "low : float\n"
        "    Non-negative margin subtracted from ``low_scale * center``.\n"
        "upp_scale : float\n"
        "    Multiplier for the per-pixel center used in the upper "
        "retained bound.\n"
        "upp : float\n"
        "    Non-negative margin added to ``upp_scale * center``.\n"
        "maxiters : int\n"
        "    Maximum number of clipping iterations per pixel. `1` preserves "
        "the historical single-pass behavior.\n"
        'cenfunc : {"median", "lmedian", "mean"}\n'
        "    Center estimator used each iteration. `lmedian` accepts the alias "
        "`lmed` and uses the lower of the two middle samples for even valid "
        "counts.\n"
        "nkeep : int\n"
        "    Minimum number of unmasked values to preserve at each pixel.\n"
        "maxrej : int or None\n"
        "    Maximum number of values that may be rejected at each pixel. "
        "`None` means no limit.\n"
        "revert_on_nkeep : bool\n"
        "    If `True`, an iteration that would leave fewer than `nkeep` usable "
        "samples at a pixel is reverted in full for that pixel.",
    ),
    "MinMaxClip": (
        "Reject the smallest and largest unmasked values at each pixel.",
        "n_min : int or float\n"
        "    Number of minimum-side values to reject. Values ``>= 1`` are a "
        "frame count. Values in ``[0, 1)`` are a fraction of the total frame "
        "count ``N``, converted via ``int(N * n_min + 0.001)``.\n"
        "n_max : int or float\n"
        "    Same convention as `n_min`, applied to the maximum-side tail.",
    ),
    "PClip": (
        "IRAF-style percentile clipping.",
        "frac : float\n"
        "    IRAF `pclip` value. Values with ``abs(frac) < 1`` are converted "
        "to an integer sorted-rank offset using half of the input image count.\n"
        "sigma : float or tuple of float\n"
        "    Lower and upper clipping thresholds applied to the pclip-estimated "
        "sigma.\n"
        "nkeep : int\n"
        "    Minimum number of unmasked samples to retain after pclip rejection.",
    ),
}


def install_kernel_docstrings(namespace: MutableMapping[str, object]) -> None:
    """Install shared runtime docstrings on public kernel functions."""
    for name, (summary, extra_params, returns, notes) in _COMBINE_SPECS.items():
        obj = namespace.get(name)
        if obj is not None:
            obj.__doc__ = _combine_doc(summary, extra_params, returns, notes)

    for name, (summary, params, notes) in _REJECTION_SPECS.items():
        obj = namespace.get(name)
        if obj is not None:
            obj.__doc__ = _rejection_doc(summary, params, notes)


def install_rejector_docstrings(namespace: MutableMapping[str, object]) -> None:
    """Install shared runtime docstrings on rejector dataclasses and methods."""
    base = namespace.get("Rejector")
    if base is not None:
        base.apply.__doc__ = _rejector_apply_doc("this rejection algorithm")

    for class_name, (summary, params) in _REJECTOR_CLASS_DOCS.items():
        cls = namespace.get(class_name)
        if cls is None:
            continue
        cls.__doc__ = _rejector_class_doc(summary, params)
        cls.apply.__doc__ = _rejector_apply_doc(class_name)


def _combine_doc(summary: str, extra_params: str, returns: str, notes: str) -> str:
    params = "\n".join(
        part for part in (_STACK_PARAM, extra_params, _VALIDATE_PARAM) if part
    )
    doc = f"""{summary}

Parameters
----------
{params}

Returns
-------
{returns}
"""
    if notes:
        doc += f"\nNotes\n-----\n{notes}\n"
    return doc


def _rejection_doc(summary: str, algorithm_params: str, notes: str) -> str:
    params = "\n".join(
        part
        for part in (
            _STACK_PARAM,
            _MASK_PARAM,
            algorithm_params,
            _GROW_PARAM,
            _VALIDATE_PARAM,
        )
        if part
    )
    doc = f"""{summary}

Parameters
----------
{params}

Returns
-------
{_REJECTION_RETURNS}
"""
    if notes:
        doc += f"\nNotes\n-----\n{notes}\n"
    return doc


def _rejector_class_doc(summary: str, params: str) -> str:
    return f"""{summary}

Parameters
----------
{params}
{_GROW_PARAM.replace("grow : float or None, optional", "grow : float or None")}
"""


def _rejector_apply_doc(kernel_name: str) -> str:
    kernel_targets = {
        "CcdClip": "ccdclip",
        "LinearClip": "linearclip",
        "MinMaxClip": "minmax",
        "PClip": "pclip",
        "SigClip": "sigclip",
    }
    target = (
        "this rejection algorithm"
        if kernel_name == "LinearClip" or kernel_name.startswith("this ")
        else f":func:`imcombiners.kernels.{kernel_targets[kernel_name]}`"
    )
    return f"""Apply {kernel_name} to an image stack.

Parameters
----------
{_STACK_PARAM_SHORT}
{_MASK_PARAM}
validate : bool, optional
    If `True`, validate inputs before calling the kernel.

Returns
-------
mask_rej, std, low, upp, nit, output_flags : tuple of ndarray
    Rejection mask and diagnostics from {target}. `std` is the per-pixel
    spread used by sigma/CCD clipping and `None` for rejection algorithms
    without a spread diagnostic.
"""
