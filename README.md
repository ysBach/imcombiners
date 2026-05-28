# imcombiners
**(Image + Combiner + Rust(rs))**, made for Python.

<p align="center">
  <img src="logo.png" alt="imcombiners logo" width="170">
</p>


`imcombiners` was built for astronomical image-stack combination, but the core kernels are general-purpose reductions for stack combination, 1-D statistics, and pixel/value rejection.

Documentation: https://ysbach.github.io/imcombiners/
GitHub: https://github.com/ysBach/imcombiners

The package started as a tool for the main developer(@ysBach)'s reduction tools ([ysfitsutilpy](https://github.com/ysBach/ysfitsutilpy)). It is a result of their graduate school life, TA experience (2016-2023), and astropy image combination TF experience (2020). After years of use & trial using `numba`, I finally rewrote the core in Rust for better speed, reliability, and maintainability.

Now it targets a modern Python API around Rust kernels, with IRAF `IMCOMBINE` compatibility but with **better speed & API**.
Tests and benchmark material compare supported paths against IRAF, Astropy/NumPy, `ccdproc`, and `bottleneck` where appropriate. On a personal laptop (Apple M4 Pro), I experience a factor of few speedup over IRAF and dozens times over Python-based tools (astropy/ccdproc/bottleneck) for typical use cases. See the Documentation.


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

- Stack combination: mean, median, lower median, percentiles, sum, min, max, variance, and weighted average.
- 1-D utilities: `imcombiners.kernels` exposes `_1d` functions such as `median_1d`, `percentiles_1d`, `var_1d`, `wvg_1d`, `sigclip_mask_1d`, and `minmax_combine_1d` for generic vectors, flattened images, light curves, and detector samples.
- Pixel rejection: sigma, CCD noise-model, iterative linear, min/max, and IRAF-style percentile clipping. Rejection centers accept mean, median, and lower median (`lmedian`/`lmed`).
- Pipeline helpers: threshold masking, zero/scale normalization, offset padding, masks, `diagnostics=None|"simple"|"full"`, and output-only fast paths.
- Performance docs: see [docs/quarto/performance/max-performance.qmd](docs/quarto/performance/max-performance.qmd), [docs/quarto/performance/image-benchmarks.qmd](docs/quarto/performance/image-benchmarks.qmd), and [docs/quarto/performance/array-1d-benchmarks.qmd](docs/quarto/performance/array-1d-benchmarks.qmd).

## Development Install

```bash
# You may activate your Python environment before this, e.g.,
# source ~/.venvs/your_env/bin/activate
uv pip install -e ".[dev]"
```

## Testing and Benchmarks

```bash
uv run pytest
uv run --extra bench python benchmarks/benchmark_combine.py
uv run --extra bench python benchmarks/benchmark_1d.py
uv run python benchmarks/benchmark_threads.py
```

`--quick` runs the smoke benchmark matrix. Omit it to run the full table that
backs the published benchmark documentation.
