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

_VALUES_PARAM = """values : ndarray, shape (N,)
    One-dimensional value vector. Accepted dtypes are `uint8`, `uint16`,
    `int16`, `int32`, `float32`, and `float64`. Integer inputs are promoted to
    the package's floating workspace when `validate` is `True`."""

_MASK_PARAM_1D = """mask : ndarray of bool, optional
    Input mask; `True` means already masked. Must have shape ``(N,)``."""

_GROW_PARAM = """grow : float or None, optional
    Optional radius in pixels used to grow `mask_rej` spatially after
    rejection. Axis 0 is the stack axis and is never grown across. `None`
    disables growth and skips the extra calculation."""

_COMBINE_PARAM = """combine : str
    Output combine method evaluated after rejection. Fused combine kernels
    support `"mean"`, `"average"`, `"avg"`, `"median"`, and `"med"`."""

_ITERATIVE_PARAMS = """sigma : float or tuple of float
    User-supplied clipping multiplier, not the measured data spread itself.
    Values are clipped when their residual from the clipping center exceeds
    ``sigma * spread``. A scalar applies the same multiplier to both tails; a
    2-tuple ``(sigma_lower, sigma_upper)`` clips asymmetrically. Thresholds
    must be finite and non-negative.
maxiters : int
    Maximum number of clipping iterations per pixel.
ddof : int
    Delta degrees of freedom for the per-pixel spread estimator.
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

_SIGMA_ITERATIVE_PARAMS = (
    _ITERATIVE_PARAMS
    + """
stdfunc : {"std", "mad"}
    Spread estimator used by sigma clipping. `"std"` uses the standard
    deviation. `"mad"` uses ``1.4826 * median(abs(x - clip_cen))`` as a robust
    sigma estimate and ignores `ddof`."""
)

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

_REJECTION_RETURNS_1D = """mask_rej : ndarray of bool, shape (N,)
    `True` where a value was rejected by this kernel.
std : scalar or None
    Spread used by sigma/CCD clipping. `None` for rejection algorithms without
    a spread diagnostic.
low : scalar
    Lower retained-value bound. Bounds are inclusive: values equal to `low`
    are retained.
upp : scalar
    Upper retained-value bound. Bounds are inclusive: values equal to `upp`
    are retained.
nit : scalar
    Iteration count.
output_flags : scalar
    Bit-coded status using the same bit meanings as the stack kernel."""

_MASK_RETURNS = """mask_rej : ndarray of bool, shape (N, *spatial)
    `True` where a value was rejected by this kernel.
    ``mask_rej.sum(axis=0)`` is the per-output rejected count."""

_MASK_RETURNS_1D = """mask_rej : ndarray of bool, shape (N,)
    `True` where a value was rejected by this kernel."""

_COMBINE_RETURNS = """combined : ndarray, shape (*spatial)
    Per-output mean or median after applying the input mask, finite-value
    filtering, and this rejection algorithm."""

_COMBINE_RETURNS_1D = """combined : scalar
    Mean or median of the surviving values after applying the input mask,
    finite-value filtering, and this rejection algorithm."""


_COMBINE_SPECS = {
    "nanaverage": (
        "Return the NaN-aware weighted average along the stack axis.",
        "weights : ndarray, shape (N,)\n"
        "    Per-plane weights. The weighted sum skips non-finite pixels, so "
        "the effective denominator is the sum of weights for finite values at "
        "each output pixel.",
        "nanaverage : ndarray, shape (*spatial)\n"
        "    Per-pixel weighted average. Pixels with no finite weighted samples "
        "return `NaN`.",
        "",
    ),
}


_REJECTION_SPECS = {
    "sigclip": (
        "Sigma-clipping rejection.",
        _SIGMA_ITERATIVE_PARAMS,
        "",
    ),
    "ccdclip": (
        "CCD noise-model clipping.",
        _ITERATIVE_PARAMS + "\nrdnoise : float\n"
        "    Read noise in electrons.\n"
        "snoise : float\n"
        "    Fractional sensitivity-noise coefficient. Its independent variance "
        "term is quadratic in the local signal.\n"
        "scale_ref : float\n"
        "    Reference scale factor. Typically the inverse of the exposure-time "
        "normalization applied during reduction.\n"
        "zero_ref : float\n"
        "    Reference zero-point offset. Accounts for a known DC bias that "
        "shifts the effective noise level.",
        "The kernel evaluates CCD rejection in the input DN units. With "
        "``signal = abs(noise_center + zero_ref) * scale_ref``, the per-pixel "
        "noise threshold is::\n\n"
        "    sqrt((rdnoise / gain)**2 + signal / gain + (snoise * signal)**2)",
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
        "    User-supplied lower and upper clipping multipliers applied to the "
        "pclip-estimated spread. This maps to IRAF `lsigma` and `hsigma`.\n"
        "nkeep : int, optional\n"
        "    Minimum number of unmasked samples to retain after pclip rejection.",
        "This follows IRAF `imcombine` pclip semantics: choose a sorted rank "
        "offset from the median, use that sample's distance from the median as "
        "sigma, then apply lower and upper sigma thresholds.",
    ),
}


_REJECTOR_CLASS_DOCS = {
    "SigClip": ("Iterative sigma-clipping rejection.", _SIGMA_ITERATIVE_PARAMS),
    "CcdClip": (
        "CCD noise-model clipping.",
        _ITERATIVE_PARAMS + "\nrdnoise : float\n"
        "    Read noise in electrons.\n"
        "gain : float\n"
        "    CCD gain in electrons per DN. Input values and rejection residuals "
        "remain in DN.\n"
        "snoise : float\n"
        "    Fractional sensitivity-noise coefficient. Its independent variance "
        "term is quadratic in the local signal.\n"
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
        "    User-supplied lower and upper clipping multipliers applied to the "
        "pclip-estimated spread.\n"
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
        obj = namespace.get(f"{name}_1d")
        if obj is not None:
            obj.__doc__ = _rejection_1d_doc(
                _variant_summary(summary, "1-D"), params, notes
            )
        obj = namespace.get(f"{name}_mask")
        if obj is not None:
            obj.__doc__ = _rejection_mask_doc(
                _variant_summary(summary, "mask-only"), params, notes
            )
        obj = namespace.get(f"{name}_mask_1d")
        if obj is not None:
            obj.__doc__ = _rejection_mask_1d_doc(
                _variant_summary(summary, "1-D mask-only"), params, notes
            )
        obj = namespace.get(f"{name}_combine")
        if obj is not None:
            obj.__doc__ = _rejection_combine_doc(
                _variant_summary(summary, "output-only"), params, notes
            )
        obj = namespace.get(f"{name}_combine_1d")
        if obj is not None:
            obj.__doc__ = _rejection_combine_1d_doc(
                _variant_summary(summary, "1-D output-only"), params, notes
            )


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


def _rejection_1d_doc(summary: str, algorithm_params: str, notes: str) -> str:
    params = "\n".join(
        part
        for part in (_VALUES_PARAM, _MASK_PARAM_1D, algorithm_params, _VALIDATE_PARAM)
        if part
    )
    doc = f"""{summary}

Parameters
----------
{params}

Returns
-------
{_REJECTION_RETURNS_1D}
"""
    if notes:
        doc += f"\nNotes\n-----\n{notes}\n"
    return doc


def _rejection_mask_doc(summary: str, algorithm_params: str, notes: str) -> str:
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
{_MASK_RETURNS}
"""
    if notes:
        doc += f"\nNotes\n-----\n{notes}\n"
    return doc


def _rejection_mask_1d_doc(summary: str, algorithm_params: str, notes: str) -> str:
    params = "\n".join(
        part
        for part in (_VALUES_PARAM, _MASK_PARAM_1D, algorithm_params, _VALIDATE_PARAM)
        if part
    )
    doc = f"""{summary}

Parameters
----------
{params}

Returns
-------
{_MASK_RETURNS_1D}
"""
    if notes:
        doc += f"\nNotes\n-----\n{notes}\n"
    return doc


def _rejection_combine_doc(summary: str, algorithm_params: str, notes: str) -> str:
    params = "\n".join(
        part
        for part in (
            _STACK_PARAM,
            _MASK_PARAM,
            _COMBINE_PARAM,
            algorithm_params,
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
{_COMBINE_RETURNS}
"""
    if notes:
        doc += f"\nNotes\n-----\n{notes}\n"
    return doc


def _rejection_combine_1d_doc(summary: str, algorithm_params: str, notes: str) -> str:
    params = "\n".join(
        part
        for part in (
            _VALUES_PARAM,
            _MASK_PARAM_1D,
            _COMBINE_PARAM,
            algorithm_params,
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
{_COMBINE_RETURNS_1D}
"""
    if notes:
        doc += f"\nNotes\n-----\n{notes}\n"
    return doc


def _variant_summary(summary: str, label: str) -> str:
    return f"{summary.rstrip('.')} ({label} variant)."


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
