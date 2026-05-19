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


def _resolve_svd_notebook_path(user_path: str | None = None) -> str:
    """Resolve notebook path for the user-provided SVD implementation."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    candidates = [
        user_path,
        os.environ.get("WMBENCH_SVD_NOTEBOOK"),
        os.path.join(root, "svd", "svd_Watermarking.ipynb"),
        os.path.abspath(os.path.join(root, "..", "svd", "svd", "svd_Watermarking.ipynb")),
        os.path.join(os.path.expanduser("~"), "Desktop", "thesis", "svd", "svd", "svd_Watermarking.ipynb"),
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            return os.path.abspath(c)
    listed = ", ".join(repr(c) for c in candidates if c)
    raise FileNotFoundError(
        "Could not locate SVD notebook implementation (svd_Watermarking.ipynb). "
        f"Tried: {listed}. Set WMBENCH_SVD_NOTEBOOK to the notebook path."
    )


def _load_notebook_svd_impl(notebook_path: str) -> SimpleNamespace:
    """Load embed/extract directly from the notebook code cell."""
    with open(notebook_path, encoding="utf-8") as f:
        nb = json.load(f)
    selected_cell = None
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "def embed(" in src and "def extract(" in src:
            selected_cell = src
            break
    if not selected_cell:
        raise RuntimeError(f"Notebook does not contain embed/extract functions: {notebook_path}")

    ns: dict = {"__name__": "wmbench_external_svd_impl"}
    exec(compile(selected_cell, notebook_path, "exec"), ns, ns)
    if "embed" not in ns or "extract" not in ns:
        raise RuntimeError(f"Failed to load embed/extract from notebook: {notebook_path}")
    return SimpleNamespace(embed=ns["embed"], extract=ns["extract"])


def _make_seeded_watermark(path: str, seed: int, size: tuple[int, int] = (32, 32)) -> None:
    rng = np.random.default_rng(seed)
    arr = (rng.integers(0, 2, size=size, dtype=np.uint8) * 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def _payload_from_key_npz(path: str) -> dict:
    with np.load(path, allow_pickle=True) as key:
        return {k: key[k] for k in key.files}


def _write_key_npz(path: str, payload: dict) -> None:
    np.savez(path, **payload)


def _quiet_call(fn, *args, **kwargs):
    """Call notebook function while suppressing stdout/stderr chatter."""
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        return fn(*args, **kwargs)


class SVDAdapter(WatermarkAdapter):
    """Adapter that reuses the SVD notebook implementation directly."""

    def __init__(
        self,
        *,
        alpha: float = 0.01,
        block_size: int = 2,
        threshold: int = 128,
        wm_seed: int = 42,
        notebook_path: str | None = None,
    ):
        self._alpha = float(alpha)
        self._block_size = int(block_size)
        self._threshold = int(threshold)
        self._scratch = tempfile.TemporaryDirectory(prefix="wmbench_svd_")
        self._watermark_path = os.path.join(self._scratch.name, "wm_seeded_32x32.png")
        _make_seeded_watermark(self._watermark_path, seed=wm_seed)
        nb_path = _resolve_svd_notebook_path(notebook_path)
        self._svd_impl = _load_notebook_svd_impl(nb_path)
        self._last_payload: dict | None = None

    @property
    def name(self) -> str:
        return "svd"

    def payload_for_meta(self) -> dict | None:
        return self._last_payload

    def embed(self, image: Image.Image) -> Image.Image:
        tag = uuid.uuid4().hex
        host_path = os.path.join(self._scratch.name, f"host_{tag}.png")
        out_path = os.path.join(self._scratch.name, f"wm_{tag}.png")
        image.convert("RGB").save(host_path)

        _quiet_call(
            self._svd_impl.embed,
            host_path,
            self._watermark_path,
            out_path,
            alpha=self._alpha,
            block_size=self._block_size,
            threshold=self._threshold,
        )
        key_path = out_path + ".key.npz"
        self._last_payload = _payload_from_key_npz(key_path)

        with Image.open(out_path) as im:
            out_img = im.convert("RGB").copy()

        for p in (host_path, out_path, key_path):
            try:
                os.remove(p)
            except OSError:
                pass
        return out_img

    def detect(
        self,
        image: Image.Image,
        original: Image.Image | None = None,
        *,
        meta: dict | None = None,
        blind: bool = False,
    ) -> float:
        del original, blind  # SVD notebook extraction is key-based.
        payload = meta if meta is not None else self._last_payload
        if payload is None:
            raise RuntimeError("SVDAdapter.detect requires key payload from embed sidecar")

        tag = uuid.uuid4().hex
        img_path = os.path.join(self._scratch.name, f"det_{tag}.png")
        key_path = os.path.join(self._scratch.name, f"det_{tag}.key.npz")
        extracted_path = os.path.join(self._scratch.name, f"det_{tag}.extracted.png")
        image.convert("RGB").save(img_path)
        _write_key_npz(key_path, payload)

        result = _quiet_call(
            self._svd_impl.extract,
            img_path,
            key_path,
            extracted_path,
            original_wm_path=self._watermark_path,
        )
        score = float(result.get("nc", 0.0))
        if np.isnan(score):
            score = 0.0
        score = float(np.clip(score, 0.0, 1.0))

        for p in (img_path, key_path, extracted_path):
            try:
                os.remove(p)
            except OSError:
                pass
        return score
