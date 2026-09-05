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

_REJECTION_TUTORIAL = (
    "For derivations and worked examples, see the [Rejection Methods tutorial]"
    "(https://ysbach.github.io/imcombiners/tutorials/05-rejection-methods.html)."
)

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
clip_cen : str or None
    Center used when computing the spread for iterative rejection. `None` uses
    the same center as `cenfunc`. Choices are `"median"`, `"lmedian"`,
    `"lmed"`, and `"mean"`.
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
    `True` for excluded samples, including input masks and non-finite values.
    LinearClip excludes input-mask entries; its disabled default returns
    an all-false mask (see Notes).
    Thus ``mask_rej.sum(axis=0)`` need not count only newly rejected samples.
std : ndarray, shape (*spatial) or None
    Clipping spread for SigClip; reference noise for CCDClip (see Notes).
    `None` for rejection
    algorithms without a spread diagnostic.
low : ndarray, shape (*spatial)
    Minimum retained value before growth. LinearClip instead reports its
    last accepted lower interval bound. Use `mask_rej` to identify survivors.
upp : ndarray, shape (*spatial)
    Maximum retained value before growth. LinearClip instead reports its
    last accepted upper interval bound. Use `mask_rej` to identify survivors.
nit : ndarray of uint8, shape (*spatial)
    Iteration-count map.
output_flags : ndarray of uint8, shape (*spatial)
    Bit-coded status. ``0`` means normal completion; bit ``1`` means at least
    one pre-masked sample at this output element; bit ``16`` means `grow` added
    at least one rejected sample at this output element. Iterative kernels also
    use bit ``2`` for maxiters, bit ``4`` for `nkeep`, and bit ``8`` for
    `maxrej`. Bits are OR-ed. `output_flags` is a per-output diagnostic; use
    `mask_rej` to inspect individual samples."""

_REJECTION_RETURNS_1D = """mask_rej : ndarray of bool, shape (N,)
    `True` for excluded samples, including input masks and non-finite values.
    LinearClip excludes input-mask entries from this returned mask.
std : scalar or None
    Clipping spread for SigClip; reference noise for CCDClip (see Notes).
    `None` for rejection algorithms without a spread diagnostic.
low : scalar
    Minimum retained value. LinearClip instead reports its last accepted
    lower interval bound. Use `mask_rej` to identify survivors.
upp : scalar
    Maximum retained value. LinearClip instead reports its last accepted
    upper interval bound. Use `mask_rej` to identify survivors.
nit : scalar
    Iteration count.
output_flags : scalar
    Bit-coded status using the same bit meanings as the stack kernel."""

_MASK_RETURNS = """mask_rej : ndarray of bool, shape (N, *spatial)
    `True` for excluded samples, including input masks and non-finite values.
    ``mask_rej.sum(axis=0)`` counts all exclusions, not only new rejections."""

_MASK_RETURNS_1D = """mask_rej : ndarray of bool, shape (N,)
    `True` for excluded samples, including input masks and non-finite values."""

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
        "At each output pixel, compute ``sum(w_i * x_i) / sum(w_i)`` over "
        "finite input values only; a missing value removes its weight from "
        "both sums. An empty selection or zero total selected weight returns "
        "`NaN`. Weights apply per image plane, not per output pixel.",
    ),
}


_REJECTION_SPECS = {
    "sigclip": (
        "Sigma-clipping rejection.",
        _SIGMA_ITERATIVE_PARAMS,
        """For the current finite, unmasked samples, let `c` be the `cenfunc`
center, `a` the `clip_cen` center, and `n` the sample count. Retain::

    c - sigma_lower * d <= x <= c + sigma_upper * d

`x` is a sample value; ``sigma=(sigma_lower, sigma_upper)`` sets the two
multipliers, or a scalar uses the same multiplier for both tails.
The spread is ``d = sqrt(sum((x - a)**2) / (n - ddof))`` for
``stdfunc="std"``, or ``d = 1.4826 * median(abs(x - a))`` for
``stdfunc="mad"``. MAD ignores `ddof`; standard deviation is undefined
when ``n <= ddof``. `clip_cen=None` uses the same center as `cenfunc`.

Equality is retained. Recompute center and spread after each accepted pass;
stop when no new samples are rejected, the spread is undefined, a safeguard
reverts the pass, or `maxiters` is reached. Earlier rejections remain excluded.
When requested, `std` describes the clipping spread, not the uncertainty of
the combined output; `low` and `upp` report the final retained extrema.""",
    ),
    "ccdclip": (
        "CCD noise-model clipping.",
        _ITERATIVE_PARAMS + "\nrdnoise : float\n"
        "    Read noise in electrons.\n"
        "gain : float\n"
        "    Positive CCD gain in electrons per DN (ADU).\n"
        "snoise : float\n"
        "    Fractional sensitivity-noise coefficient for multiplicative sensitivity "
        "(for example, flat-field) uncertainty. It adds (snoise * signal)^2 to "
        "the CCD variance.\n"
        "scales, zeros : array-like or None\n"
        "    Noise-model vectors with one value per image. Scales must be "
        "finite and positive; zeros must be finite. See Notes for units.",
        """Let `c` be the `cenfunc` center of the finite, unmasked values, `g`
the gain, `r` the read noise, and `q` the fractional sensitivity noise.
For image `i`, the CCD model is::

    S_i = max(0, s_i * (c + z_i))
    d_i = sqrt(max((r/g)**2, v_floor) + S_i/g + (q*S_i)**2) / s_i

Here ``v_floor = 1e4 / 3.402823466e38`` is IRAF's read-variance floor.
The direct-kernel vectors satisfy ``raw_i = s_i * (x_i + z_i)``:
`x_i` is the supplied sample, `s_i` is `scales[i]`, and `z_i` is
`zeros[i]` in normalized units.
They describe already normalized input; this kernel does not normalize data.
For ``x_i = (raw_i - raw_zero_i) / s_i``, pass ``z_i = raw_zero_i / s_i``.
An omitted scale or zero vector defaults to ones or zeros, respectively.

Scan each sorted tail inward, rejecting while ``(c-x_i)/d_i > sigma_lower``
or ``(x_i-c)/d_i > sigma_upper``, using the two `sigma` multipliers.
Each side stops at its first retained value;
with unequal noise, this can shield an interior sample. Recompute `c` after
accepted passes, subject to `nkeep`, `maxrej`, and `maxiters`.

`ddof` does not affect this noise model. `clip_cen` selects the center for
the unscaled reference noise reported as `std`; a non-finite reference
variance stops the iteration. Rejection residuals use `cenfunc`. This diagnostic
is neither the individual `d_i` nor the combined-image uncertainty.
When requested, `low`/`upp` report retained extrema.""",
    ),
    "linearclip": (
        "Center-relative linear clipping.",
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
        """Compute `c` with `cenfunc` from the current finite, unmasked values.
Retain values within the inclusive interval::

    low_scale * c - low <= x <= upp_scale * c + upp

For example, ``low_scale=upp_scale=1, low=2, upp=3`` retains values from
``c - 2`` through ``c + 3``. Recompute `c` after each accepted pass until
no new samples are rejected, a safeguard reverts the pass, or `maxiters`
is reached. `maxiters=1` makes a single pass. No spread is estimated.

Default behavior differs by entry point: `LinearClip()` and stack
``linearclip()`` explicitly disable clipping for
``(low_scale, low, upp_scale, upp) = (1, 0, 1, 0)``; they return an all-false
rejection mask, NaN bounds, and zero iterations. ``linearclip_1d()`` applies
the equation even for these defaults, retaining only values equal to `c`
unless a safeguard reverts the pass. `low`/`upp` outputs are the last accepted
interval bounds, or NaN if no pass was accepted.""",
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
        """Sort the `M` finite, unmasked samples at each output pixel. With `N`
the original image count, a request `a` (`n_min` or `n_max`) resolves to::

    A = int(N * a + 0.001) if 0 <= a < 1 else int(a)
    k = int(M * A / N + 0.001)

Reject `k` samples from that tail, first the low tail, then the high tail
of the remaining values. Both counts use the initial `M`; available samples
limit the cuts. Thus masked/non-finite inputs reduce even count-based requests
proportionally. For example, ``N=10, M=5, n_min=2`` removes one minimum.
This is rank selection, not a value threshold: tied values may be split.
There is no spread estimate or iteration; `low`/`upp` are retained extrema.""",
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
        """Sort the finite, unmasked samples and let `m` be their median.
`frac` selects an offset rank `r` below (negative) or above (positive) the
median; it is not a percentage of samples to reject. For ``abs(frac) < 1``,
scale by ``max(N // 2, 1)`` for the original image count `N` and truncate
to an integer, with a minimum offset magnitude of one. Estimate::

    d = sign(frac) * (x_sorted[r] - m)

Reject sorted endpoints while ``(m-x)/d >= sigma_lower`` or
``(x-m)/d >= sigma_upper``, using the two `sigma` multipliers.
Unlike sigma/CCD clipping, equality is rejected.
The center and spread are computed once. If needed, restore the least-extreme
rejected endpoints to satisfy `nkeep`, including IRAF's residual-tie handling.
With fewer than three usable samples, at most `nkeep` usable samples, or
``d == 0``, no additional samples are rejected. In diagnostic calls, `std`
is `None` and `low`/`upp` report retained extrema. The tutorial specifies the
full rank convention.""",
    ),
}


_REJECTOR_CLASS_DOCS = {
    "SigClip": (
        "Iterative sigma-clipping rejection.",
        _SIGMA_ITERATIVE_PARAMS,
        _REJECTION_SPECS["sigclip"][2],
    ),
    "CcdClip": (
        "CCD noise-model clipping.",
        _ITERATIVE_PARAMS + "\nrdnoise : float\n"
        "    Read noise in electrons.\n"
        "gain : float\n"
        "    Positive CCD gain in electrons per DN (ADU).\n"
        "snoise : float\n"
        "    Fractional sensitivity-noise coefficient for multiplicative sensitivity "
        "(for example, flat-field) uncertainty. It adds (snoise * signal)^2 to "
        "the CCD variance.\n"
        "scales, zeros : tuple of float or None\n"
        "    Noise-model vectors with one value per image. Scales must be "
        "finite and positive; zeros must be finite. See Notes for units.",
        _REJECTION_SPECS["ccdclip"][2],
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
        _REJECTION_SPECS["linearclip"][2],
    ),
    "MinMaxClip": (
        "Reject the smallest and largest unmasked values at each pixel.",
        "n_min : int or float\n"
        "    Number of minimum-side values to reject. Values ``>= 1`` are a "
        "frame count. Values in ``[0, 1)`` are a fraction of the total frame "
        "count ``N``, converted via ``int(N * n_min + 0.001)``.\n"
        "n_max : int or float\n"
        "    Same convention as `n_min`, applied to the maximum-side tail.",
        _REJECTION_SPECS["minmax"][2],
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
        _REJECTION_SPECS["pclip"][2],
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

    for class_name, (summary, params, notes) in _REJECTOR_CLASS_DOCS.items():
        cls = namespace.get(class_name)
        if cls is None:
            continue
        cls.__doc__ = _rejector_class_doc(summary, params, notes)
        cls.apply.__doc__ = _rejector_apply_doc(class_name, notes)


def _combine_doc(summary: str, extra_params: str, returns: str, notes: str) -> str:
    stack_param = """arr : ndarray, shape (N, *spatial)
    Image stack. Accepted dtypes are `uint8`, `uint16`, `int16`, `int32`,
    `float32`, and `float64`. Weighted averages delegate to `reducers` and
    return `float64`. Output shapes match the trailing spatial dimensions."""
    params = "\n".join(
        part for part in (stack_param, extra_params, _VALIDATE_PARAM) if part
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
    notes = "\n".join(part for part in (notes, _REJECTION_TUTORIAL) if part)
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
    notes = "\n".join(part for part in (notes, _REJECTION_TUTORIAL) if part)
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
    notes = "\n".join(part for part in (notes, _REJECTION_TUTORIAL) if part)
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
    notes = "\n".join(part for part in (notes, _REJECTION_TUTORIAL) if part)
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
    notes = "\n".join(part for part in (notes, _REJECTION_TUTORIAL) if part)
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
    notes = "\n".join(part for part in (notes, _REJECTION_TUTORIAL) if part)
    doc += f"\nNotes\n-----\n{notes}\n"
    return doc


def _variant_summary(summary: str, label: str) -> str:
    return f"{summary.rstrip('.')} ({label} variant)."


def _rejector_class_doc(summary: str, params: str, notes: str) -> str:
    return f"""{summary}

Parameters
----------
{params}
{_GROW_PARAM.replace("grow : float or None, optional", "grow : float or None")}

Notes
-----
{notes}

{_REJECTION_TUTORIAL}
"""


def _rejector_apply_doc(kernel_name: str, notes: str = "") -> str:
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
    doc = f"""Apply {kernel_name} to an image stack.

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
    clipping spread for SigClip, reference noise for CCDClip, and `None` for algorithms
    without a spread diagnostic.
"""
    if notes:
        doc += f"\nNotes\n-----\n{notes}\n\n{_REJECTION_TUTORIAL}\n"
    return doc
