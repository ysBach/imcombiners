# IRAF imcombine parity workspace

This directory contains the local IRAF parity and benchmark for
`imcombiners`. Generated FITS inputs, IRAF outputs, manifests, CL scripts, and
local IRAF executables are ignored by git; the parity test and benchmark use
temporary workspaces for normal runs.

## IRAF executable

This requires an IRAF `ecl.e` executable. If you do not already have
IRAF installed, get the community distribution from
<https://iraf-community.github.io/>.

Put your built executable at `benchmarks/IRAF/ecl.e` (copy or symlink is fine);
`.gitignore` excludes it. If your IRAF task tree lives somewhere else, keep that
path local by using `IMCOMBINERS_IRAF_ROOT` or the benchmark script's
`--iraf-root` option.

Manual IRAF inspection from this directory:

```bash
cd benchmarks/IRAF
env iraf="${IMCOMBINERS_IRAF_ROOT:-$(pwd)}/" IRAFARCH=macos64 arch=.macos64 ./ecl.e -f scripts/run_imcombine.cl
```

Do not pass `niterate` to this IRAF `imcombine`: the local IRAF
`pkg/images/immatch/src/imcombine/imcombine.par` does not define that parameter.

## Acknowledgement

This checks supported `imc` compatibility behavior against IRAF
`imcombine`. Some performance choices in `imcombiners` intentionally follow
IRAF's implementation strategy: skip diagnostic products for output-only runs,
avoid a duplicate final mean/median pass when rejection already has retained
samples, and preserve IRAF rank/`nkeep` rollback semantics in compatibility
modes.

The generator can still be run manually when you want to inspect the IRAF case matrix:

```bash
uv run --extra bench python benchmarks/IRAF/scripts/make_inputs.py
```

The generator writes 7-frame, 128x128 `uint16`, `int16`, and `float32` stacks by
default. It covers `sigclip`, `ccdclip`, `pclip`, and `minmax` for IRAF
`average`/`median`, with three variants:

- `baseline`: no pre-rejection threshold or input mask.
- `threshold`: IRAF `lthreshold`/`hthreshold`, mapped to `ndcombine(..., thresholds=(lower, upper))`.
- `input_bpm`: IRAF reads per-input bad-pixel masks through the FITS `BPM` header keyword with `masktype="!BPM badvalue"` and `maskvalue=1`; `imc` loads the same mask stack and passes `ndcombine(..., mask=mask)`.

For automated parity, run:

```bash
IMCOMBINERS_RUN_IRAF_PARITY=1 uv run pytest tests/test_iraf_parity.py -q
```

The test creates a temporary IRAF workspace, calls `make_inputs.generate(...)`
there, runs the generated CL script with `ecl.e`, compares the temporary
outputs, and removes the workspace when pytest exits. By default it looks for
`benchmarks/IRAF/ecl.e`; set `IMCOMBINERS_IRAF_ECL` for the executable or
`IMCOMBINERS_IRAF_ROOT` for a separate IRAF task tree.

The pytest comparison checks the combined output image, per-input rejection
masks, `nrejmasks`, `sigmas`, and lower/upper retained-value bounds derived
from IRAF's rejection masks plus any pre-rejection threshold/input-BPM masks.

## `imc` argument mapping used here

The parity manifest records the exact bridge kwargs used for each generated case. The important IRAF-to-`imc` mappings are:

- IRAF `combine="average"` -> `ndcombine(..., combine="mean")`.
- IRAF `combine="median"` -> `ndcombine(..., combine="median")`.
- IRAF `reject="sigclip"` -> `reject="sigclip"`, `sigma=(lsigma, hsigma)`, `cenfunc="median"` for `mclip=yes`, `maxiters=5`, `nkeep=1`, `revert_on_nkeep=True`.
- IRAF `reject="ccdclip"` -> the same sigma settings plus `gain`, `rdnoise`, and `snoise`.
- IRAF `reject="minmax"` -> `reject="minmax"`, `n_minmax=(nlow, nhigh)`.
- IRAF `lthreshold`/`hthreshold` -> `thresholds=(lthreshold, hthreshold)`.
- IRAF input BPM masks are not the `bpmasks` output parameter. They are input masks selected through `masktype` and a FITS header keyword. For this, IRAF uses `masktype="!BPM badvalue"` and `maskvalue=1`, while `imc` receives a boolean `mask` with `True` where the BPM image equals 1.

## Exact parameter values used in the test cases

All values are defined in `scripts/make_inputs.py`. Fixed IRAF parameters for every case: `outtype=real`, `offsets=none`, `scale=none`, `zero=none`, `weight=none`, `project=no`, `blank=0.0`.

### sigclip

| parameter | IRAF | imc |
|---|---|---|
| lsigma / sigma[0] | 2.0 | 2.0 |
| hsigma / sigma[1] | 2.0 | 2.0 |
| mclip / cenfunc | `yes` | `"median"` |
| (maxiters) | — (IRAF default 5) | `maxiters=5` |
| nkeep | 1 | `nkeep=1` |
| revert_on_nkeep | — | `True` |

### ccdclip

Same as sigclip plus:

| parameter | IRAF | imc |
|---|---|---|
| rdnoise | 5.0 | 5.0 |
| gain | 2.0 | 2.0 |
| snoise | 0.0 | 0.0 |

### pclip

| parameter | IRAF | imc |
|---|---|---|
| pclip | 0.25 | 0.25 |

### minmax

| parameter | IRAF | imc |
|---|---|---|
| nlow / n_minmax[0] | 1 | 1 |
| nhigh / n_minmax[1] | 1 | 1 |

### Variants (applied to every rejection × combine combination)

| variant | IRAF | imc |
|---|---|---|
| baseline | (no threshold or mask) | (no threshold or mask) |
| threshold | `lthreshold`/`hthreshold` (uint16: 990/1200, int16 & float32: 90/220) | `thresholds=(lthreshold, hthreshold)` |
| input_bpm | `masktype="!BPM badvalue"`, `maskvalue=1.0` | `mask` (bool array, `True` where BPM == 1) |

Dtypes tested: `uint16`, `int16`, `float32`. Combine methods: IRAF `average`→imc `mean`, IRAF `median`→imc `median`. Stack size: 7 frames per dtype at the requested `--height × --width` shape.

## Current parity status

With `IMCOMBINERS_RUN_IRAF_PARITY=1`, the local IRAF parity run has 73 checks and
all pass against current `imc`. Coverage includes `uint16`, `int16`, `float32`;
IRAF `average`/`median`; `sigclip`, `ccdclip`, `pclip`, `minmax`; and the
`baseline`, `threshold`, and `input_bpm` variants. Failures report the case,
IRAF parameters, `imc` kwargs, differing mask/count entries, and example pixels.

## Benchmark IRAF against `imc`

Run:

```bash
uv run --extra bench python benchmarks/IRAF/scripts/benchmark_iraf.py \
    --repeats 3
```

The benchmark groups the same manifest cases into batched CL throughput groups:
`all`, `sigclip`, `ccdclip`, `pclip`, and `minmax`. It also emits one one-case
CL row per manifest case by default; pass `--no-by-case` to skip those rows.

**IRAF timing method (warm-ecl):** each group or case is repeated inside one
`ecl.e -f` process, and elapsed wall time is divided by `--repeats`. By default,
the script subtracts a dtype-specific 1x1 output-only `imcombine` baseline from
each IRAF workload repeat. Use `--iraf-baseline startup` or
`--iraf-baseline none` to change that.

**`imc` timing:** reads the generated FITS inputs with `fitsio`, stacks them with NumPy, runs `ndcombine(..., diagnostics=None)`, and writes the combined FITS output to a temporary directory on each timed iteration.

Before timing, the benchmark asserts each selected output-only `imc` result against an IRAF output-only run. All benchmark inputs, IRAF outputs, and `imc` outputs are created under a Python temporary directory and deleted when the benchmark exits.

The ratio is a task-level comparison. IRAF does not expose its internal `imcombine` combine/reject kernel as a standalone callable in this, so this cannot be a pure kernel microbenchmark.

Grouped timings are batched workloads, not sums of the one-case rows. For
example, with `group=all` and `--repeats 3`, one CL script contains 216
output-only `imcombine` calls (72 cases x 3 repeats). The reported time is
`(single ecl wall time - baseline) / 3`. One-case rows use a separate CL file
per case with the same batched-repeat approach.

The script prints Markdown to stdout, including an environment table with
Python, package versions, OS/kernel details, machine type, and logical CPU
count. Pass `--output /tmp/imc-iraf-benchmark.md` if you want to preserve a
local run artifact. Public benchmark summaries belong in
`docs/quarto/benchmarks.qmd`; generated local run artifacts are intentionally
not tracked.
