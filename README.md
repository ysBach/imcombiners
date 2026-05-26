# imcombiners

<p align="center">
  <img src="logo.png" alt="imcombiners logo" width="170">
</p>

**(Image + Combiner + Rust(rs))**

> **Status:** alpha - see the [CHANGELOG](CHANGELOG.md).

`imcombiners` was built for astronomical image-stack combination, but the core
kernels are general-purpose reductions for stack combination and pixel
rejection.

The package started as a tool for the main developer(@ysBach)'s reduction tools (ysfitsutilpy).
Now it targets a modern Python API around Rust kernels, with IRAF `IMCOMBINE` compatibility but with better speed & API.
Tests and benchmark material compare supported paths against IRAF, Astropy/NumPy, `ccdproc`, and `bottleneck` where appropriate.


## First Look
The package has four usage modes: **Standard `Combiner().combine()`** approach, compact
`ndcombine()` wrapper, **Chained `Combiner()`**, and direct kernel calls. Start
with standard `Combiner` usage for ordinary Python workflows. Use chained
`Combiner` calls when you need retained diagnostics, `ndcombine()` for compact
IRAF-like call sites (this function was made in consideration of CLI tools), and direct kernel calls for custom high-throughput layers.

```python
import numpy as np
import imcombiners as imc

rng = np.random.default_rng(20250311)
stack = rng.normal(1000, 5, (15, 256, 256)).astype("float32")

cmb = imc.Combiner(stack)
out = cmb.combine(
    "median",  # final stack-combination method
    # 1. Optional pre-rejection threshold masking
    thresholds=(0.0, 65000.0),
    # 2. Optional per-image zero/scale normalization
    zero=None,
    scale="median",
    # 3. Optional pixel rejection before final combination
    rejectors=[
        imc.MinMaxClip(n_min=1, n_max=0.1),
        imc.SigClip(sigma=3.0, maxiters=5),
    ],
    diagnostics=None,  # output-only fast path
)
```

See [docs/quarto/index.qmd](docs/quarto/index.qmd#first-look) for the detailed explanations, API-level guidance, and conventions behind this example.

## Features

- Stack combination: mean, median, lower median, sum, min, max, variance, and weighted average.
- Pixel rejection: sigma, CCD noise-model, iterative linear, min/max, and IRAF-style percentile clipping. Rejection centers accept mean, median, and lower median (`lmedian`/`lmed`).
- Pipeline helpers: threshold masking, zero/scale normalization, offset padding, masks, `diagnostics=None|"simple"|"full"`, and output-only fast paths.
- Performance docs: see [docs/quarto/performance.qmd](docs/quarto/performance.qmd).

## Documentation
TBD

## Development Install

Requires Python, `uv`, a stable Rust toolchain, and the Quarto CLI when
rendering documentation.

```bash
uv sync --extra dev
uv run maturin develop --release
uv run pytest -q
```

## Testing and Benchmarks

```bash
uv run pytest
uv run --extra bench python benchmarks/benchmark_combine.py
uv run python benchmarks/benchmark_threads.py
```

`--quick` runs the smoke benchmark matrix. Omit it to run the full table that
backs the published benchmark documentation.
