"""The opt-in IRAF check must not report success without reference outputs."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("astropy.io.fits")


@pytest.mark.parametrize("returncode", [0, 17])
@pytest.mark.skipif(
    os.name != "posix", reason="fake IRAF executable requires a POSIX shell"
)
def test_enabled_iraf_reports_reference_generation_failure(
    tmp_path: Path, returncode: int
) -> None:
    ecl = tmp_path / "ecl"
    ecl.write_text(
        "#!/bin/sh\n"
        "printf 'reference-generation-stdout\\n'\n"
        "printf 'reference-generation-stderr\\n' >&2\n"
        f"exit {returncode}\n"
    )
    ecl.chmod(0o700)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_iraf_parity.py", "-q"],
        cwd=Path(__file__).resolve().parents[1],
        env={
            **os.environ,
            "IMC_RUN_IRAF_PARITY": "1",
            "IMC_IRAF_ROOT": str(tmp_path),
            "IMC_IRAF_ECL": str(ecl),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0, result.stdout + result.stderr
    assert "reference-generation-stdout" in result.stdout + result.stderr
    assert "reference-generation-stderr" in result.stdout + result.stderr
