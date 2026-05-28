# Changelog

All notable changes to `imcombiners`. This project follows [Semantic Versioning](https://semver.org).

## [0.1.0] — Unreleased

### Added
* Rust core with pyo3 bindings (extension module `imcombiners._core`).
* `kernel/combine.rs`: NaN-aware `mean`, `median`, `lmedian`, `sum`, `min`, `max`, `variance`, `weighted_average` along axis 0; parallel via `rayon`.
* `kernel/reject.rs`: per-pixel `sigclip`, `ccdclip`, `minmax`, `pclip` with IRAF-style `nkeep` rollback support; `nkeep` / `maxrej` constraints; bit-coded `output_flags` diagnostics.
* `pyapi/`: one `#[pyfunction]` per operation — no string `method=...` dispatch at the FFI boundary.
* Python layer with **three API tiers**:
 * `Combiner` fluent class (recommended).
 * `imcombiners.kernels` focused functions.
 * `ndcombine(...)` one-call wrapper.
* Pure `lmedian` / `lmed` integer fast path: direct `kernels.lmedian(...)` and pure `ndcombine(..., combine="lmedian")` preserve accepted integer dtypes.
* `place_into_padded(...)` for dense integer-pixel offset placement into a padded stack, with Python axis-order `(dy, dx)` offsets.
* Frozen rejection-spec dataclasses: `SigClip`, `CcdClip`, `LinearClip`,
  `MinMaxClip`, `PClip`.
* Rejection diagnostics now return `(mask_rej, std, low, upp, nit, output_flags)`.
  `std` is the per-pixel spread used by sigma/CCD clipping and `None` for
  rejection algorithms without a spread diagnostic; `ndcombine(full=True)`
  returns this as its final item.
* `LinearClip`, a center-relative rejection spec that keeps values within
  `low_scale * center - low` and `upp_scale * center + upp` bounds using the
  per-pixel median center. The default `LinearClip()` is a no-op.
* `MinMaxClip` and `kernels.minmax` use `n_min` / `n_max` for minimum-side and
  maximum-side rejection counts. The older `n_low` / `n_high` names were
  removed before publication.
* Optimized mask-to-NaN materialization helper used by high-level masked combine paths.
* Quarto docs scaffold (`docs/quarto/`) with API reference and tutorials, including zero/scale, rejection, and offset-stacking examples.
* Benchmark script `benchmarks/benchmark_combine.py` with numerical checks before timing.
* `pyproject.toml` extras: `test`, `dev`, `docs`.

### Known gaps
* `pclip` now follows IRAF `imcombine` pclip semantics instead of the earlier percentile-window interpretation. This is a breaking change for `PClip`, `kernels.pclip`, and `ndcombine(..., reject="pclip")`: tuple `frac` values are rejected, and masks can differ from older `imcombiners` releases.
* Offset support is currently dense integer-pixel placement only; WCS/physical offset calculation, subpixel interpolation, chunked placement, and paired mask/variance placement remain downstream/future work.
* Variance is exposed through `combine="variance"` / `kernels.variance` / `Combiner.variance`; there is no separate `return_variance` flag on `ndcombine` (use `np.sqrt(var)` for an error/std map).
