"""Load embed/extract helpers from dwt_dct_watermarking.ipynb."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace


def _quiet_exec(src: str, ns: dict) -> None:
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        exec(compile(src, ns.get("__file__", "<notebook>"), "exec"), ns, ns)


def load_notebook_impl(notebook_path: Path | None = None) -> SimpleNamespace:
    path = notebook_path or Path(__file__).resolve().parent / "dwt_dct_watermarking.ipynb"
    if not path.is_file():
        raise FileNotFoundError(f"DCT-DWT notebook not found: {path}")

    with open(path, encoding="utf-8") as f:
        nb = json.load(f)
    cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    ns: dict = {"__name__": "dct_dwt_impl", "__file__": str(path)}

    import_cell = next(
        ("".join(c.get("source", [])) for c in cells if "import pywt" in "".join(c.get("source", []))),
        None,
    )
    config_cell = next(
        (
            "".join(c.get("source", []))
            for c in cells
            if "HOST_SIZE" in "".join(c.get("source", [])) and "MID_POS" in "".join(c.get("source", []))
        ),
        None,
    )
    helper_cell = next(
        (
            "".join(c.get("source", []))
            for c in cells
            if "def load_grayscale(" in "".join(c.get("source", []))
            and "def image_to_bits(" in "".join(c.get("source", []))
            and "def rho(" in "".join(c.get("source", []))
        ),
        None,
    )
    core_cell = next(
        (
            "".join(c.get("source", []))
            for c in cells
            if "def embed_watermark(" in "".join(c.get("source", []))
            and "def extract_watermark_bits(" in "".join(c.get("source", []))
        ),
        None,
    )
    for src in (import_cell, config_cell, helper_cell, core_cell):
        if not src:
            raise RuntimeError(f"Notebook missing required DCT-DWT definitions: {path}")
        _quiet_exec(src, ns)

    required = (
        "load_grayscale",
        "image_to_bits",
        "rho",
        "psnr",
        "ssim",
        "embed_watermark",
        "extract_watermark_bits",
        "HOST_SIZE",
        "EMBED_WM_SIZE",
        "WAVELET",
        "SUBBAND_CHOICE",
        "ALPHA",
        "SEED",
        "PN_MODE",
        "MID_POS",
    )
    missing = [k for k in required if k not in ns]
    if missing:
        raise RuntimeError(f"Notebook missing symbols {missing} in {path}")
    return SimpleNamespace(**{k: ns[k] for k in required})
