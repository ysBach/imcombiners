import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_logo_generator():
    spec = importlib.util.spec_from_file_location(
        "logo_generator", ROOT / "logo_generator.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generate_imcombiners_logo_writes_png(tmp_path: Path):
    logo_generator = _load_logo_generator()

    output = tmp_path / "logo.png"

    result = logo_generator.generate_imcombiners_logo(output, dpi=80)

    assert result == output
    data = output.read_bytes()
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(data) > 10_000
    width = int.from_bytes(data[16:20], "big")
    height = int.from_bytes(data[20:24], "big")
    assert width == height
    assert width >= 300
