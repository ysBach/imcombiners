"""Create deterministic FITS inputs and IRAF imcombine parity metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "inputs"
OUTPUT_DIR = ROOT / "outputs"
RUN_CL = ROOT / "scripts" / "run_imcombine.cl"

DTYPES = ("uint16", "int16", "float32")
DEFAULT_SHAPE = (128, 128)
COMBINES = (
    ("average", "mean"),
    ("median", "median"),
)
THRESHOLDS = {
    "uint16": (990.0, 1200.0),
    "int16": (90.0, 220.0),
    "float32": (90.0, 220.0),
}


def _set_root(root: Path) -> None:
    global ROOT
    global INPUT_DIR
    global LIST_DIR
    global OUTPUT_DIR
    global RUN_CL
    ROOT = root
    INPUT_DIR = ROOT / "inputs"
    OUTPUT_DIR = ROOT / "outputs"
    RUN_CL = ROOT / "scripts" / "run_imcombine.cl"


def _write_fits(
    path: Path, data: np.ndarray, header_values: dict[str, object] | None = None
) -> None:
    header = fits.Header()
    header["BUNIT"] = "adu"
    for key, value in (header_values or {}).items():
        header[key] = value
    fits.PrimaryHDU(data, header=header).writeto(path, overwrite=True)


def _stack_for_dtype(dtype_name: str, shape: tuple[int, int]) -> list[np.ndarray]:
    dtype = np.dtype(dtype_name)
    yy, xx = np.indices(shape, dtype=np.float32)
    grid = (np.mod(yy, 3.0) * 4.0 + np.mod(xx, 4.0)).astype(np.float32)
    base = 1000.0 + grid if dtype_name == "uint16" else 100.0 + grid
    if dtype_name == "float32":
        phase = np.mod(yy + 2.0 * xx, 12.0) / 11.0
        base = base + (phase * 0.33).astype(np.float32)

    planes = []
    for idx, offset in enumerate([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]):
        plane = base + offset
        if idx == 0:
            plane = plane.copy()
            plane[0, 0] -= 30.0
            plane[2, 3] += 45.0
        if idx == 6:
            plane = plane.copy()
            plane[0, 1] += 320.0
            plane[1, 2] += 180.0
        planes.append(plane.astype(dtype))
    return planes


def _mask_for_plane(idx: int, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint16)
    if idx == 2:
        mask[0, 0] = 1
        mask[1, 1] = 1
        mask[4::32, 4::32] = 1
    if idx == 5:
        mask[2, 2] = 1
        mask[12::32, 12::32] = 1
    if idx == 7:
        mask[0, 3] = 1
        mask[20::32, 20::32] = 1
    return mask


def _input_list(dtype_name: str, shape: tuple[int, int]) -> tuple[list[str], list[str]]:
    inputs = []
    masks = []
    for idx, plane in enumerate(_stack_for_dtype(dtype_name, shape), start=1):
        name = f"{dtype_name}_input_{idx:02d}.fits"
        mask_name = f"{dtype_name}_bpm_{idx:02d}.fits"
        _write_fits(INPUT_DIR / mask_name, _mask_for_plane(idx, shape))
        _write_fits(INPUT_DIR / name, plane, {"BPM": f"inputs/{mask_name}"})
        inputs.append(name)
        masks.append(mask_name)
    list_path = ROOT / f"{dtype_name}_inputs.lis"
    list_path.write_text("".join(f"inputs/{name}\n" for name in inputs))
    return inputs, masks


def _scaled_input_list(dtype_name: str, shape: tuple[int, int]) -> list[str]:
    """Write a stack with known IRAF file-scale/file-zero normalization."""
    dtype = np.dtype(dtype_name)
    scales = np.array([1.0, 2.0, 0.5, 4.0, 0.25, 2.0, 0.5], dtype=np.float32)
    file_scales = 1.0 / scales
    file_zeros = np.array([0.0, 20.0, -40.0, 50.0, 400.0, -60.0, 80.0])
    if dtype_name == "uint16":
        # Preserve representability of the unsigned stack while retaining
        # nonuniform file zeros and scale deviations larger than sigscale.
        file_zeros[4] = 50.0

    inputs = []
    for index, target in enumerate(_stack_for_dtype(dtype_name, shape), start=1):
        # IRAF applies ``(raw + file_zero) / internal_scale``.  Scaling the
        # target by four makes every integer target compatible with the exact
        # binary scale factors above, including the 0.25 plane.
        target = np.asarray(target, dtype=np.float32) * 4.0
        raw = target * scales[index - 1] - file_zeros[index - 1]
        name = f"{dtype_name}_scaled_input_{index:02d}.fits"
        _write_fits(INPUT_DIR / name, raw.astype(dtype))
        inputs.append(name)

    (ROOT / f"{dtype_name}_scaled_inputs.lis").write_text(
        "".join(f"inputs/{name}\n" for name in inputs)
    )
    (ROOT / f"{dtype_name}_scales.txt").write_text(
        "".join(f"{value:.8g}\n" for value in file_scales)
    )
    (ROOT / f"{dtype_name}_zeros.txt").write_text(
        "".join(f"{value:.8g}\n" for value in file_zeros)
    )
    return inputs


def _case(
    dtype_name: str,
    reject: str,
    iraf_combine: str,
    imc_combine: str,
    iraf: dict[str, Any],
    imc: dict[str, Any],
    variant: str,
    *,
    inputs: list[str] | None = None,
    input_list: str | None = None,
):
    suffix = "" if variant == "baseline" else f"_{variant}"
    name = f"{dtype_name}_{reject}_{iraf_combine}{suffix}"
    return {
        "name": name,
        "dtype": dtype_name,
        "variant": variant,
        "inputs": _INPUTS[dtype_name] if inputs is None else inputs,
        "input_list": f"{dtype_name}_inputs.lis" if input_list is None else input_list,
        "input_masks": _INPUT_MASKS[dtype_name],
        "output": f"{name}.fits",
        "diagnostics": {
            "rejmask": f"{name}_rej.fits",
            "rejmask_pl": f"{name}_rej.pl",
            "nrejmask": f"{name}_nrej.fits",
            "nrejmask_pl": f"{name}_nrej.pl",
            "sigmas": f"{name}_sigma.fits",
        },
        "iraf": {
            "combine": iraf_combine,
            "reject": reject,
            "outtype": "real",
            **iraf,
        },
        "imcombiners": {
            "combine": imc_combine,
            "reject": reject,
            "diagnostics": "simple",
            **imc,
        },
    }


def _cl_value(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, str):
        return f'"{value}"'
    return str(value)


def _variants(dtype_name: str) -> tuple[dict[str, Any], ...]:
    lthreshold, hthreshold = THRESHOLDS[dtype_name]
    return (
        {
            "name": "baseline",
            "iraf": {},
            "imc": {},
        },
        {
            "name": "threshold",
            "iraf": {
                "lthreshold": lthreshold,
                "hthreshold": hthreshold,
            },
            "imc": {
                "thresholds": [lthreshold, hthreshold],
            },
        },
        {
            "name": "input_bpm",
            "iraf": {
                "masktype": "!BPM badvalue",
                "maskvalue": 1.0,
            },
            "imc": {
                "mask_from": "input_masks",
            },
        },
    )


def _write_run_cl(cases: list[dict[str, Any]]) -> None:
    lines = [
        'reset imtype = "fits"',
        "images",
        "immatch",
        "",
    ]
    for case in cases:
        diag = case["diagnostics"]
        delete_targets = [
            f"outputs/{case['output']}",
            f"outputs/{diag['nrejmask']}",
            f"outputs/{diag['nrejmask_pl']}",
            f"outputs/{diag['sigmas']}",
            f"outputs/{diag['rejmask']}",
            f"outputs/{diag['rejmask_pl']}",
        ]
        for target in delete_targets:
            lines.append(f'imdelete ("{target}", verify=no)')

        params = {
            "headers": "",
            "bpmasks": "",
            "rejmasks": f"outputs/{diag['rejmask_pl']}",
            "nrejmasks": f"outputs/{diag['nrejmask_pl']}",
            "expmasks": "",
            "sigmas": f"outputs/{diag['sigmas']}",
            "logfile": f"outputs/{case['name']}.log",
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
                f'imcombine ("@{case["input_list"]}", '
                f'"outputs/{case["output"]}", {arg_text})',
                f'imcopy ("outputs/{diag["rejmask_pl"]}", "outputs/{diag["rejmask"]}")',
                f'imcopy ("outputs/{diag["nrejmask_pl"]}", '
                f'"outputs/{diag["nrejmask"]}")',
                "",
            ]
        )
    lines.append("logout")
    RUN_CL.write_text("\n".join(lines) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--height", type=int, default=DEFAULT_SHAPE[0])
    parser.add_argument("--width", type=int, default=DEFAULT_SHAPE[1])
    return parser.parse_args()


def generate(root: Path = ROOT, shape: tuple[int, int] = DEFAULT_SHAPE) -> list[dict]:
    _set_root(root)

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_CL.parent.mkdir(parents=True, exist_ok=True)

    global _INPUTS
    global _INPUT_MASKS
    inputs_and_masks = {
        dtype_name: _input_list(dtype_name, shape) for dtype_name in DTYPES
    }
    _INPUTS = {dtype_name: item[0] for dtype_name, item in inputs_and_masks.items()}
    _INPUT_MASKS = {
        dtype_name: item[1] for dtype_name, item in inputs_and_masks.items()
    }
    scaled_inputs = {
        dtype_name: _scaled_input_list(dtype_name, shape) for dtype_name in DTYPES
    }

    cases = []
    for dtype_name in DTYPES:
        for iraf_combine, imc_combine in COMBINES:
            for variant in _variants(dtype_name):
                variant_name = variant["name"]
                variant_iraf = variant["iraf"]
                variant_imc = variant["imc"]
                new_cases = [
                    _case(
                        dtype_name,
                        "sigclip",
                        iraf_combine,
                        imc_combine,
                        {
                            "lsigma": 2.0,
                            "hsigma": 2.0,
                            "mclip": True,
                            "nkeep": 1,
                        },
                        {
                            "sigma": [2.0, 2.0],
                            "cenfunc": "median",
                            "maxiters": 5,
                            "nkeep": 1,
                            "revert_on_nkeep": True,
                        },
                        variant_name,
                    ),
                    _case(
                        dtype_name,
                        "ccdclip",
                        iraf_combine,
                        imc_combine,
                        {
                            "lsigma": 2.0,
                            "hsigma": 2.0,
                            "mclip": True,
                            "nkeep": 1,
                            "rdnoise": 5.0,
                            "gain": 2.0,
                            "snoise": 0.0,
                        },
                        {
                            "sigma": [2.0, 2.0],
                            "cenfunc": "median",
                            "maxiters": 5,
                            "nkeep": 1,
                            "revert_on_nkeep": True,
                            "rdnoise": 5.0,
                            "gain": 2.0,
                            "snoise": 0.0,
                        },
                        variant_name,
                    ),
                    _case(
                        dtype_name,
                        "pclip",
                        iraf_combine,
                        imc_combine,
                        {"pclip": 0.25},
                        {"pclip": 0.25},
                        variant_name,
                    ),
                    _case(
                        dtype_name,
                        "minmax",
                        iraf_combine,
                        imc_combine,
                        {"nlow": 1, "nhigh": 1},
                        {"n_minmax": [1, 1]},
                        variant_name,
                    ),
                ]
                for case in new_cases:
                    case["iraf"].update(variant_iraf)
                    case["imcombiners"].update(variant_imc)
                cases.extend(new_cases)

            # Pin the quadratic sensitivity-noise term against IRAF without
            # changing the unscaled normalization path.
            cases.append(
                _case(
                    dtype_name,
                    "ccdclip",
                    iraf_combine,
                    imc_combine,
                    {
                        "lsigma": 2.0,
                        "hsigma": 2.0,
                        "mclip": True,
                        "nkeep": 1,
                        "rdnoise": 5.0,
                        "gain": 2.0,
                        "snoise": 0.05,
                    },
                    {
                        "sigma": [2.0, 2.0],
                        "cenfunc": "median",
                        "maxiters": 5,
                        "nkeep": 1,
                        "revert_on_nkeep": True,
                        "rdnoise": 5.0,
                        "gain": 2.0,
                        "snoise": 0.05,
                    },
                    "snoise",
                )
            )

            scaled_iraf = {
                "scale": f"@{dtype_name}_scales.txt",
                "zero": f"@{dtype_name}_zeros.txt",
            }
            scales = [1.0, 2.0, 0.5, 4.0, 0.25, 2.0, 0.5]
            file_zeros = [0.0, 20.0, -40.0, 50.0, 400.0, -60.0, 80.0]
            if dtype_name == "uint16":
                file_zeros[4] = 50.0
            scaled_imc = {
                "scale": scales,
                "zero": [-value for value in file_zeros],
                "zero_to_0th": False,
                "scale_to_0th": False,
            }
            scaled_inputs_name = f"{dtype_name}_scaled_inputs.lis"
            scaled_case_inputs = scaled_inputs[dtype_name]
            for reject, iraf_reject, imc_reject in (
                (
                    "ccdclip",
                    {
                        "lsigma": 2.0,
                        "hsigma": 2.0,
                        "mclip": True,
                        "nkeep": 1,
                        "rdnoise": 5.0,
                        "gain": 2.0,
                        "snoise": 0.05,
                    },
                    {
                        "sigma": [2.0, 2.0],
                        "cenfunc": "median",
                        "maxiters": 5,
                        "nkeep": 1,
                        "revert_on_nkeep": True,
                        "rdnoise": 5.0,
                        "gain": 2.0,
                        "snoise": 0.05,
                    },
                ),
                ("pclip", {"pclip": 0.25}, {"pclip": 0.25}),
                ("minmax", {"nlow": 1, "nhigh": 1}, {"n_minmax": [1, 1]}),
            ):
                cases.append(
                    _case(
                        dtype_name,
                        reject,
                        iraf_combine,
                        imc_combine,
                        {**iraf_reject, **scaled_iraf},
                        {**imc_reject, **scaled_imc},
                        "scaled",
                        inputs=scaled_case_inputs,
                        input_list=scaled_inputs_name,
                    )
                )

    (ROOT / "cases.json").write_text(json.dumps({"cases": cases}, indent=2) + "\n")
    _write_run_cl(cases)
    return cases


def main() -> None:
    args = _parse_args()
    generate(args.root, (args.height, args.width))


if __name__ == "__main__":
    main()
