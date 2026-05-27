"""Benchmark local IRAF imcombine runs against imcombiners on parity cases."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import fitsio
import imcombiners as imc
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
DEFAULT_IRAF_ROOT = ROOT
DEFAULT_ECL = ROOT / "ecl.e"


def _cases(manifest: Path) -> list[dict[str, Any]]:
    return json.loads(manifest.read_text())["cases"]


def _case_groups(cases: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    rejectors = ("sigclip", "ccdclip", "pclip", "minmax")
    groups = {"all": cases}
    groups.update(
        {
            reject: [case for case in cases if case["imcombiners"]["reject"] == reject]
            for reject in rejectors
        }
    )
    return groups


def _load_stack(case: dict[str, Any], data_root: Path) -> np.ndarray:
    arrays = []
    dtype = np.dtype(str(case["dtype"]))
    for name in case["inputs"]:
        path = data_root / "inputs" / str(name)
        arrays.append(np.asarray(fitsio.read(path), dtype=dtype))
    return np.stack(arrays, axis=0)


def _load_input_mask(case: dict[str, Any], data_root: Path) -> np.ndarray:
    arrays = []
    for name in case["input_masks"]:
        arrays.append(np.asarray(fitsio.read(data_root / "inputs" / str(name))) != 0)
    return np.stack(arrays, axis=0)


def _write_fits(path: Path, data: np.ndarray | None) -> None:
    if data is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(data)
    if arr.dtype == np.bool_:
        arr = arr.astype(np.uint8)
    fitsio.write(path, arr, clobber=True)


def _imc_output(case: dict[str, Any], data_root: Path) -> np.ndarray:
    stack = _load_stack(case, data_root)
    kwargs = dict(case["imcombiners"])
    kwargs["full"] = False
    mask_from = kwargs.pop("mask_from", None)
    if mask_from == "input_masks":
        kwargs["mask"] = _load_input_mask(case, data_root)
    elif mask_from is not None:
        raise ValueError(f"unknown mask source: {mask_from}")
    return imc.ndcombine(stack, **kwargs)


def _run_imc_case(case: dict[str, Any], data_root: Path, output_root: Path) -> None:
    path = output_root / "outputs" / str(case["output"])
    _write_fits(path, _imc_output(case, data_root))


def _median_time(func, repeats: int) -> float:
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        timings.append(time.perf_counter() - start)
    return float(np.median(timings))


def _cl_value(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, str):
        return f'"{value}"'
    return str(value)


def _write_output_only_cl(cases: list[dict[str, Any]], path: Path) -> None:
    lines = [
        'reset imtype = "fits"',
        "images",
        "immatch",
        "",
    ]
    for case in cases:
        lines.append(f'imdelete ("outputs/{case["output"]}", verify=no)')
        params = {
            "headers": "",
            "bpmasks": "",
            "rejmasks": "",
            "nrejmasks": "",
            "expmasks": "",
            "sigmas": "",
            "logfile": "",
            "combine": case["iraf"]["combine"],
            "reject": case["iraf"]["reject"],
            "project": False,
            "outtype": case["iraf"]["outtype"],
            "offsets": "none",
            "masktype": "none",
            "maskvalue": 0.0,
            "blank": 0.0,
            "scale": "none",
            "zero": "none",
            "weight": "none",
        }
        params.update(
            {
                key: value
                for key, value in case["iraf"].items()
                if key not in {"combine", "reject", "outtype"}
            }
        )
        arg_text = ", ".join(
            f"{key}={_cl_value(value)}" for key, value in params.items()
        )
        lines.extend(
            [
                f'imcombine ("@{case["dtype"]}_inputs.lis", '
                f'"outputs/{case["output"]}", {arg_text})',
                "",
            ]
        )
    lines.append("logout")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _write_startup_baseline_cl(path: Path) -> None:
    lines = [
        'reset imtype = "fits"',
        "images",
        "immatch",
        "",
        "logout",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _write_tiny_imcombine_inputs(data_root: Path, dtypes: set[str]) -> None:
    baseline_dir = data_root / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    values = {
        "uint16": np.array([[1000]], dtype=np.uint16),
        "int16": np.array([[100]], dtype=np.int16),
        "float32": np.array([[100.0]], dtype=np.float32),
    }
    for dtype_name in sorted(dtypes):
        path = baseline_dir / f"tiny_{dtype_name}.fits"
        fitsio.write(path, values[dtype_name], clobber=True)
        (data_root / f"tiny_{dtype_name}_inputs.lis").write_text(
            f"baseline/{path.name}\n"
        )


def _write_tiny_imcombine_baseline_cl(dtype_name: str, path: Path) -> None:
    params = {
        "headers": "",
        "bpmasks": "",
        "rejmasks": "",
        "nrejmasks": "",
        "expmasks": "",
        "sigmas": "",
        "logfile": "",
        "combine": "average",
        "reject": "none",
        "project": False,
        "outtype": "real",
        "offsets": "none",
        "masktype": "none",
        "maskvalue": 0.0,
        "blank": 0.0,
        "scale": "none",
        "zero": "none",
        "weight": "none",
    }
    arg_text = ", ".join(f"{key}={_cl_value(value)}" for key, value in params.items())
    output = f"baseline/tiny_{dtype_name}_out.fits"
    lines = [
        'reset imtype = "fits"',
        "images",
        "immatch",
        "",
        f'imdelete ("{output}", verify=no)',
        f'imcombine ("@tiny_{dtype_name}_inputs.lis", "{output}", {arg_text})',
        "",
        "logout",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _default_iraf_root() -> Path:
    return Path(os.environ.get("IMCOMBINERS_IRAF_ROOT", DEFAULT_IRAF_ROOT))


def _default_ecl() -> Path | None:
    value = os.environ.get("IMCOMBINERS_IRAF_ECL")
    return Path(value) if value else None


def _resolve_ecl(iraf_root: Path, ecl: Path | None) -> Path:
    if ecl is not None:
        return ecl
    for candidate in (
        DEFAULT_ECL,
        iraf_root / "ecl.e",
        iraf_root / "bin.macos64" / "ecl.e",
    ):
        if candidate.exists():
            return candidate
    return DEFAULT_ECL


def _iraf_command(ecl: Path, cl_script: Path) -> list[str]:
    return [str(ecl), "-f", str(cl_script)]


def _run_iraf(
    iraf_root: Path,
    ecl: Path,
    cl_script: Path,
    *,
    cwd: Path,
    verbose: bool,
) -> None:
    env = os.environ.copy()
    env.update(
        {
            "iraf": str(iraf_root) + "/",
            "IRAFARCH": "macos64",
            "arch": ".macos64",
        }
    )
    stdout = None if verbose else subprocess.DEVNULL
    stderr = None if verbose else subprocess.DEVNULL
    subprocess.run(
        _iraf_command(ecl, cl_script),
        cwd=cwd,
        env=env,
        check=True,
        stdout=stdout,
        stderr=stderr,
    )


def _time_iraf_startup_baseline(
    *,
    iraf_root: Path,
    ecl: Path,
    cwd: Path,
    repeats: int,
    verbose_iraf: bool,
) -> float:
    cl_script = cwd / "scripts" / "benchmark_ecl_startup.cl"
    _write_startup_baseline_cl(cl_script)
    return _median_time(
        lambda: _run_iraf(iraf_root, ecl, cl_script, cwd=cwd, verbose=verbose_iraf),
        repeats,
    )


def _time_iraf_tiny_imcombine_baselines(
    dtypes: set[str],
    *,
    iraf_root: Path,
    ecl: Path,
    cwd: Path,
    repeats: int,
    verbose_iraf: bool,
) -> dict[str, float]:
    _write_tiny_imcombine_inputs(cwd, dtypes)
    baselines = {}
    for dtype_name in sorted(dtypes):
        cl_script = cwd / "scripts" / f"benchmark_tiny_{dtype_name}.cl"
        _write_tiny_imcombine_baseline_cl(dtype_name, cl_script)
        baselines[dtype_name] = _median_time(
            lambda s=cl_script: _run_iraf(
                iraf_root, ecl, s, cwd=cwd, verbose=verbose_iraf
            ),
            repeats,
        )
    return baselines


def _time_iraf_workload(
    cases: list[dict[str, Any]],
    cl_script: Path,
    *,
    repeats: int,
    iraf_root: Path,
    ecl: Path,
    cwd: Path,
    startup_baseline_s: float,
    verbose_iraf: bool,
) -> float:
    _write_output_only_cl(cases * repeats, cl_script)
    start = time.perf_counter()
    _run_iraf(iraf_root, ecl, cl_script, cwd=cwd, verbose=verbose_iraf)
    elapsed = time.perf_counter() - start
    return max(0.0, (elapsed - startup_baseline_s) / repeats)


def _atol_for_array(arr: np.ndarray) -> float:
    arr = np.asarray(arr)
    if not np.issubdtype(arr.dtype, np.floating):
        return 0.0
    finite = np.abs(arr[np.isfinite(arr)])
    if finite.size == 0:
        return float(np.finfo(arr.dtype).eps)
    return float(2 * np.max(np.spacing(finite)))


def _validate_outputs(
    cases: list[dict[str, Any]],
    *,
    data_root: Path,
    iraf_root: Path,
    ecl: Path,
    verbose_iraf: bool,
) -> None:
    cl_script = data_root / "scripts" / "validate_output_only.cl"
    _write_output_only_cl(cases, cl_script)
    _run_iraf(iraf_root, ecl, cl_script, cwd=data_root, verbose=verbose_iraf)
    for case in cases:
        expected = np.asarray(fitsio.read(data_root / "outputs" / str(case["output"])))
        got = _imc_output(case, data_root)
        if (
            got.dtype.kind != expected.dtype.kind
            or got.dtype.itemsize != expected.dtype.itemsize
        ):
            raise AssertionError(
                f"{case['name']} dtype mismatch: imc={got.dtype} IRAF={expected.dtype}"
            )
        np.testing.assert_allclose(
            got,
            expected.astype(got.dtype, copy=False),
            rtol=0.0,
            atol=_atol_for_array(expected),
            err_msg=f"output-only IRAF parity failed for {case['name']}",
        )


_GROUP_HEADERS = ["group", "cases", "IRAF ms", "imc ms", "IRAF/imc"]
_GROUP_ALIGN = ["<", ">", ">", ">", ">"]
_CASE_HEADERS = [
    "op",
    "dtype",
    "N",
    "imc ms",
    "IRAF ms",
    "IRAF/imc (speedup)",
]
_CASE_ALIGN = ["<", "<", ">", ">", ">", ">"]


def _dw(cell: str) -> int:
    return len(cell.replace("**", ""))


def _pad(cell: str, width: int, align: str) -> str:
    extra = len(cell) - _dw(cell)
    target = width + extra
    return cell.rjust(target) if align == ">" else cell.ljust(target)


def _table(headers: list[str], align: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return ""
    widths = [
        max(_dw(headers[i]), max(_dw(row[i]) for row in rows))
        for i in range(len(headers))
    ]

    def fmt_row(cells: list[str]) -> str:
        return (
            "| "
            + " | ".join(
                _pad(cells[i], widths[i], align[i]) for i in range(len(headers))
            )
            + " |"
        )

    def sep(width: int, side: str) -> str:
        if side == ">":
            return "-" * max(width - 1, 2) + ":"
        return "-" * max(width, 3)

    lines = [fmt_row(headers)]
    separator = "|".join(f" {sep(widths[i], align[i])} " for i in range(len(headers)))
    lines.append(f"|{separator}|")
    lines.extend(fmt_row(row) for row in rows)
    return "\n".join(lines)


def _ratio(iraf_s: float | None, imc_s: float | None) -> float | None:
    if iraf_s is None or imc_s is None:
        return None
    return iraf_s / imc_s


def _fmt_ms(value: float | None) -> str:
    return "ERROR" if value is None else f"{value * 1000.0:.3f}"


def _fmt_ratio(value: float | None) -> str:
    return "ERROR" if value is None else f"{value:.1f}x"


def _fmt_baselines_ms(values: dict[str, float]) -> str:
    if set(values) == {"*"}:
        return _fmt_ms(values["*"])
    return ", ".join(f"{key}={_fmt_ms(values[key])}" for key in sorted(values))


def _markdown_group_table(rows: list[dict[str, Any]]) -> str:
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                str(row["group"]),
                str(row["cases"]),
                _fmt_ms(row["iraf_median_s"]),
                _fmt_ms(row["imc_median_s"]),
                _fmt_ratio(_ratio(row["iraf_median_s"], row["imc_median_s"])),
            ]
        )
    return _table(_GROUP_HEADERS, _GROUP_ALIGN, table_rows)


def _markdown_case_table(rows: list[dict[str, Any]]) -> str:
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                str(row["op"]),
                str(row["dtype"]),
                str(row["n"]),
                _fmt_ms(row["imc_median_s"]),
                _fmt_ms(row["iraf_median_s"]),
                _fmt_ratio(_ratio(row["iraf_median_s"], row["imc_median_s"])),
            ]
        )
    return _table(_CASE_HEADERS, _CASE_ALIGN, table_rows)


def _case_op(case: dict[str, Any]) -> str:
    combine = str(case["imcombiners"]["combine"])
    reject = str(case["imcombiners"].get("reject", "none"))
    return combine if reject == "none" else f"{reject}_{combine}"


def _benchmark_case(
    case: dict[str, Any],
    *,
    repeats: int,
    iraf_root: Path,
    ecl: Path,
    data_root: Path,
    output_root: Path,
    iraf_baseline_s: float,
    verbose_iraf: bool,
) -> dict[str, Any]:
    cl_script = data_root / "scripts" / f"benchmark_case_{case['name']}.cl"

    try:
        iraf_median = _time_iraf_workload(
            [case],
            cl_script,
            repeats=repeats,
            iraf_root=iraf_root,
            ecl=ecl,
            cwd=data_root,
            startup_baseline_s=iraf_baseline_s,
            verbose_iraf=verbose_iraf,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"IRAF ERROR for {case['name']}: {exc}", file=sys.stderr)
        iraf_median = None

    try:
        imc_median = _median_time(
            lambda: _run_imc_case(case, data_root, output_root),
            repeats,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"imc ERROR for {case['name']}: {exc}", file=sys.stderr)
        imc_median = None

    return {
        "case": case["name"],
        "op": _case_op(case),
        "dtype": case["dtype"],
        "n": len(case["inputs"]),
        "variant": case["variant"],
        "combine": case["iraf"]["combine"],
        "reject": case["iraf"]["reject"],
        "iraf_median_s": iraf_median,
        "imc_median_s": imc_median,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Time IRAF imcombine and imcombiners on the generated IRAF parity cases."
        )
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument(
        "--iraf-root",
        type=Path,
        default=_default_iraf_root(),
        help=(
            "IRAF root used for the `iraf` environment variable. Defaults to "
            "IMCOMBINERS_IRAF_ROOT or benchmarks/IRAF."
        ),
    )
    parser.add_argument(
        "--ecl",
        type=Path,
        default=None,
        help=(
            "Path to IRAF ecl.e. Defaults to IMCOMBINERS_IRAF_ECL, then "
            "benchmarks/IRAF/ecl.e, then <iraf-root>/bin.macos64/ecl.e."
        ),
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=("all", "sigclip", "ccdclip", "pclip", "minmax"),
        default=("all", "sigclip", "ccdclip", "pclip", "minmax"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional Markdown file to write. Stdout is always printed.",
    )
    parser.add_argument(
        "--by-case",
        dest="by_case",
        action="store_true",
        default=True,
        help="Benchmark each manifest case with a separate IRAF CL run (default).",
    )
    parser.add_argument(
        "--no-by-case",
        dest="by_case",
        action="store_false",
        help="Skip per-case benchmark rows.",
    )
    parser.add_argument(
        "--no-groups",
        action="store_true",
        help="Skip grouped all/rejector benchmark rows.",
    )
    parser.add_argument("--verbose-iraf", action="store_true")
    parser.add_argument(
        "--iraf-baseline",
        choices=("tiny-imcombine", "startup", "none"),
        default="tiny-imcombine",
        help=(
            "IRAF overhead to subtract. tiny-imcombine times a 1x1 FITS "
            "imcombine task matching the first workload dtype; startup times "
            "only reset/images/immatch/logout; none disables subtraction."
        ),
    )
    args = parser.parse_args()
    args.iraf_root = args.iraf_root.expanduser().resolve()
    args.ecl = _resolve_ecl(
        args.iraf_root,
        args.ecl.expanduser().resolve() if args.ecl is not None else _default_ecl(),
    )

    if not args.ecl.exists():
        raise SystemExit(
            "IRAF ecl.e not found. Put it at benchmarks/IRAF/ecl.e, set "
            "IMCOMBINERS_IRAF_ECL, or pass --ecl."
        )

    group_rows = []
    case_rows = []
    image_shape = "unknown"
    iraf_baselines: dict[str, float] = {"*": 0.0}

    def baseline_for(cases: list[dict[str, Any]]) -> float:
        if args.iraf_baseline == "tiny-imcombine":
            return iraf_baselines[str(cases[0]["dtype"])]
        return iraf_baselines["*"]

    with tempfile.TemporaryDirectory(prefix="imc-iraf-bench-") as tmp:
        tmp_root = Path(tmp)
        data_root = tmp_root / "iraf"
        imc_output_root = tmp_root / "imc"

        sys.path.insert(0, str(SCRIPTS_DIR))
        import make_inputs  # noqa: PLC0415

        make_inputs.generate(data_root, (args.height, args.width))
        cases = _cases(data_root / "cases.json")
        groups = _case_groups(cases)
        seen: dict[int, dict[str, Any]] = {}
        for group_name in args.groups:
            for c in groups[group_name]:
                seen.setdefault(id(c), c)
        selected_cases = list(seen.values())

        first_stack = _load_stack(cases[0], data_root)
        image_shape = "x".join(str(size) for size in first_stack.shape[1:])
        _validate_outputs(
            selected_cases,
            data_root=data_root,
            iraf_root=args.iraf_root,
            ecl=args.ecl,
            verbose_iraf=args.verbose_iraf,
        )
        if args.iraf_baseline == "tiny-imcombine":
            iraf_baselines = _time_iraf_tiny_imcombine_baselines(
                {str(case["dtype"]) for case in selected_cases},
                iraf_root=args.iraf_root,
                ecl=args.ecl,
                cwd=data_root,
                repeats=args.repeats,
                verbose_iraf=args.verbose_iraf,
            )
        elif args.iraf_baseline == "startup":
            iraf_baselines = {
                "*": _time_iraf_startup_baseline(
                    iraf_root=args.iraf_root,
                    ecl=args.ecl,
                    cwd=data_root,
                    repeats=args.repeats,
                    verbose_iraf=args.verbose_iraf,
                )
            }

        if not args.no_groups:
            for group_name in args.groups:
                group_cases = groups[group_name]
                cl_script = data_root / "scripts" / f"benchmark_{group_name}.cl"

                def run_imc_group(group: list[dict[str, Any]] = group_cases) -> None:
                    for case in group:
                        _run_imc_case(case, data_root, imc_output_root)

                try:
                    iraf_median = _time_iraf_workload(
                        group_cases,
                        cl_script,
                        repeats=args.repeats,
                        iraf_root=args.iraf_root,
                        ecl=args.ecl,
                        cwd=data_root,
                        startup_baseline_s=baseline_for(group_cases),
                        verbose_iraf=args.verbose_iraf,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"IRAF ERROR for group {group_name}: {exc}", file=sys.stderr)
                    iraf_median = None

                try:
                    imc_median = _median_time(run_imc_group, args.repeats)
                except Exception as exc:  # noqa: BLE001
                    print(f"imc ERROR for group {group_name}: {exc}", file=sys.stderr)
                    imc_median = None

                group_rows.append(
                    {
                        "group": group_name,
                        "cases": len(group_cases),
                        "iraf_median_s": iraf_median,
                        "imc_median_s": imc_median,
                    }
                )

        if args.by_case:
            case_rows = [
                _benchmark_case(
                    case,
                    repeats=args.repeats,
                    iraf_root=args.iraf_root,
                    ecl=args.ecl,
                    data_root=data_root,
                    output_root=imc_output_root,
                    iraf_baseline_s=baseline_for([case]),
                    verbose_iraf=args.verbose_iraf,
                )
                for case in selected_cases
            ]

    sections = [
        "# IRAF vs imcombiners benchmark",
        "",
        f"Repeats: {args.repeats}",
        f"Input shape: {image_shape}",
        f"IRAF baseline mode: `{args.iraf_baseline}`",
        f"Subtracted IRAF baseline: {_fmt_baselines_ms(iraf_baselines)} ms",
        "Validation: output-only `imc` results asserted against IRAF before timing",
        "",
    ]
    if group_rows:
        sections.extend(
            [
                "## Batched CL Throughput Timings",
                "",
                _markdown_group_table(group_rows),
                "",
            ]
        )
    if case_rows:
        sections.extend(
            [
                "## One-Case CL Timings",
                "",
                _markdown_case_table(case_rows),
                "",
            ]
        )
    if args.iraf_baseline == "tiny-imcombine":
        baseline_notes = [
            "- IRAF baseline: median time for a 1x1 output-only `imcombine` "
            "task per dtype; subtracted once per IRAF workload repeat.",
        ]
    elif args.iraf_baseline == "startup":
        baseline_notes = [
            "- IRAF baseline: median time for `reset imtype`, `images`, "
            "`immatch`, and `logout`; subtracted once per IRAF workload repeat.",
        ]
    else:
        baseline_notes = [
            "- `--iraf-baseline none` disables IRAF overhead subtraction.",
        ]
    sections.extend(
        [
            "Notes:",
            "",
            "- For IRAF setup, `ecl.e` discovery, parity cases, and full "
            "details, see `benchmarks/IRAF/README.md`.",
            *baseline_notes,
            "- Batched CL rows: all cases run in one IRAF CL process; "
            "the script contains `--repeats` copies of the case list "
            "and the total elapsed time is divided by `--repeats`.",
            "- One-case rows: one CL file per case, same batched-repeat approach.",
            "- Timings include FITS input reads and combined FITS output writes; "
            "IRAF diagnostics and `imc full=True` outputs are not timed.",
        ]
    )
    report = "\n".join(sections)
    print(report)
    if args.output is not None:
        args.output.write_text(report + "\n")


if __name__ == "__main__":
    main()
