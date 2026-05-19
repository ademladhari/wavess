from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import uuid
from types import SimpleNamespace

import numpy as np
from PIL import Image

from wmbench.watermarks.base import WatermarkAdapter


def _resolve_dct_dwt_notebook_path(user_path: str | None = None) -> str:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    candidates = [
        user_path,
        os.environ.get("WMBENCH_DCT_DWT_NOTEBOOK"),
        os.path.join(root, "dct-dwt", "dwt_dct_watermarking.ipynb"),
        os.path.abspath(os.path.join(root, "..", "dct-dwt", "dwt_dct_watermarking.ipynb")),
        os.path.join(os.path.expanduser("~"), "Desktop", "thesis", "dct-dwt", "dwt_dct_watermarking.ipynb"),
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            return os.path.abspath(c)
    listed = ", ".join(repr(c) for c in candidates if c)
    raise FileNotFoundError(
        "Could not locate DCT-DWT notebook implementation (dwt_dct_watermarking.ipynb). "
        f"Tried: {listed}. Set WMBENCH_DCT_DWT_NOTEBOOK to the notebook path."
    )


def _quiet_exec(src: str, ns: dict) -> None:
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        exec(compile(src, ns.get("__file__", "<notebook>"), "exec"), ns, ns)


def _load_notebook_dct_dwt_impl(notebook_path: str) -> SimpleNamespace:
    with open(notebook_path, encoding="utf-8") as f:
        nb = json.load(f)
    cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    ns: dict = {"__name__": "wmbench_external_dct_dwt_impl", "__file__": notebook_path}

    import_cell = next(
        ("".join(c.get("source", [])) for c in cells if "import pywt" in "".join(c.get("source", []))),
        None,
    )
    config_cell = next(
        (
            "".join(c.get("source", []))
            for c in cells
            if "HOST_SIZE" in "".join(c.get("source", []))
            and "MID_POS" in "".join(c.get("source", []))
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
            raise RuntimeError(f"Notebook is missing required DCT-DWT definitions: {notebook_path}")
        _quiet_exec(src, ns)

    required = (
        "load_grayscale",
        "image_to_bits",
        "rho",
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
        raise RuntimeError(f"Notebook missing expected symbols {missing} in {notebook_path}")
    return SimpleNamespace(**{k: ns[k] for k in required})


def _make_seeded_binary_watermark(path: str, seed: int, size: int) -> None:
    rng = np.random.default_rng(seed)
    arr = (rng.integers(0, 2, size=(size, size), dtype=np.uint8) * 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


class DCTDWTAdapter(WatermarkAdapter):
    """Adapter that directly uses dct-dwt notebook embed/extract functions."""

    def __init__(
        self,
        *,
        notebook_path: str | None = None,
        alpha: float | None = None,
        seed: int | None = None,
        wavelet: str | None = None,
        subband_choice: str | None = None,
        pn_mode: str | None = None,
        wm_seed: int = 777,
    ):
        self._scratch = tempfile.TemporaryDirectory(prefix="wmbench_dctdwt_")
        self._impl = _load_notebook_dct_dwt_impl(_resolve_dct_dwt_notebook_path(notebook_path))
        self._host_size = int(self._impl.HOST_SIZE)
        self._alpha = float(alpha if alpha is not None else self._impl.ALPHA)
        self._seed = int(seed if seed is not None else self._impl.SEED)
        self._wavelet = str(wavelet if wavelet is not None else self._impl.WAVELET)
        self._subband_choice = str(subband_choice if subband_choice is not None else self._impl.SUBBAND_CHOICE)
        self._pn_mode = str(pn_mode if pn_mode is not None else self._impl.PN_MODE)
        self._mid_pos = [tuple(x) for x in self._impl.MID_POS]

        self._wm_img_path = os.path.join(self._scratch.name, "wm_bits_seeded.png")
        _make_seeded_binary_watermark(self._wm_img_path, seed=wm_seed, size=int(self._impl.EMBED_WM_SIZE))
        wm_gray = self._impl.load_grayscale(self._wm_img_path, int(self._impl.EMBED_WM_SIZE))
        self._wm_bits = self._impl.image_to_bits(wm_gray)

        # extraction capacity at HOST_SIZE with 2-level DWT and 4x4 DCT blocks
        self._capacity = (self._host_size // 4 // 4) * (self._host_size // 4 // 4)
        self._last_payload: dict | None = None
        self._default_payload = self._make_payload()

    @property
    def name(self) -> str:
        return "dct-dwt"

    def _make_payload(self) -> dict:
        return {
            "capacity": int(self._capacity),
            "seed": int(self._seed),
            "wavelet": self._wavelet,
            "subband_choice": self._subband_choice,
            "pn_mode": self._pn_mode,
            "mid_pos": [tuple(x) for x in self._mid_pos],
            "wm_bits": np.asarray(self._wm_bits, dtype=np.uint8),
        }

    def payload_for_meta(self) -> dict | None:
        return self._last_payload

    def _load_host_from_image(self, image: Image.Image, tag: str) -> np.ndarray:
        host_path = os.path.join(self._scratch.name, f"{tag}.png")
        image.convert("RGB").save(host_path)
        host = self._impl.load_grayscale(host_path, self._host_size)
        try:
            os.remove(host_path)
        except OSError:
            pass
        return host

    def embed(self, image: Image.Image) -> Image.Image:
        host = self._load_host_from_image(image, f"host_{uuid.uuid4().hex}")
        wm_img, state = self._impl.embed_watermark(
            host,
            self._wm_bits,
            alpha=self._alpha,
            seed=self._seed,
            wavelet=self._wavelet,
            subband_choice=self._subband_choice,
            mid_pos=self._mid_pos,
            pn_mode=self._pn_mode,
        )
        self._last_payload = self._make_payload()
        self._last_payload["capacity"] = int(state.get("capacity", self._capacity))
        arr = np.clip(wm_img, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="L").convert("RGB")

    def detect(
        self,
        image: Image.Image,
        original: Image.Image | None = None,
        *,
        meta: dict | None = None,
        blind: bool = False,
    ) -> float:
        del original, blind  # notebook method is blind and key-based
        payload = meta if meta is not None else self._last_payload
        if payload is None:
            payload = self._default_payload
        host = self._load_host_from_image(image, f"det_{uuid.uuid4().hex}")
        ex_bits = self._impl.extract_watermark_bits(
            host,
            capacity=int(payload["capacity"]),
            seed=int(payload["seed"]),
            wavelet=str(payload["wavelet"]),
            subband_choice=str(payload["subband_choice"]),
            mid_pos=[tuple(x) for x in payload["mid_pos"]],
            pn_mode=str(payload["pn_mode"]),
        )
        ref_bits = np.asarray(payload["wm_bits"], dtype=np.uint8).flatten()[: len(ex_bits)]
        score_rho = float(self._impl.rho(ref_bits, ex_bits.astype(np.uint8)))
        return float(np.clip((score_rho + 1.0) * 0.5, 0.0, 1.0))

