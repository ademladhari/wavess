from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from wmbench.watermarks.base import WatermarkAdapter


def _resolve_ssl_root(user_path: str | None = None) -> Path:
    root = Path(__file__).resolve().parents[2]
    candidates = [
        user_path,
        os.environ.get("WMBENCH_SSL_ROOT"),
        str(root / "ssl"),
        str(root.parent / "wavesPipeline" / "ssl"),
        str(Path.home() / "Desktop" / "thesis" / "wavesPipeline" / "ssl"),
    ]
    for c in candidates:
        if c:
            p = Path(c).expanduser().resolve()
            if (p / "src").is_dir():
                return p
    listed = ", ".join(repr(c) for c in candidates if c)
    raise FileNotFoundError(
        "Could not locate SSL implementation root (expects a src/ directory). "
        f"Tried: {listed}. Set WMBENCH_SSL_ROOT."
    )


def _resolve_file(user_path: str | None, env_key: str, default_path: Path, label: str) -> Path:
    candidates = [user_path, os.environ.get(env_key), str(default_path)]
    for c in candidates:
        if c:
            p = Path(c).expanduser().resolve()
            if p.is_file():
                return p
    listed = ", ".join(repr(c) for c in candidates if c)
    raise FileNotFoundError(f"Could not locate {label}. Tried: {listed}. Set {env_key}.")


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    if t.ndim == 4:
        t = t[0]
    arr = (
        t.detach()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return Image.fromarray(arr, mode="RGB")


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class SSLAdapter(WatermarkAdapter):
    """
    Adapter for Self-Supervised Latent-Space watermarking (Fernandez et al. 2021).

    Uses the project's own code from ssl/src:
      - src.embed.embed_zero_bit / embed_multi_bit
      - src.detect.detect_zero_bit / decode_multi_bit
      - src.backbone.Backbone + PCAWhitening
    """

    def __init__(
        self,
        *,
        ssl_root: str | None = None,
        mode: str | None = None,
        whitening_path: str | None = None,
        key_path: str | None = None,
        message_bits: str | None = None,
        fpr: float | None = None,
        seed: int = 1234,
        device: str | None = None,
    ):
        self._root = _resolve_ssl_root(ssl_root)
        if str(self._root) not in sys.path:
            sys.path.insert(0, str(self._root))

        from src.backbone import Backbone, PCAWhitening
        from src.embed import EmbedConfig, embed_multi_bit, embed_zero_bit
        from src.detect import decode_multi_bit, detect_zero_bit
        from src.keys import load_key, parse_message

        mode_val = (mode or os.environ.get("WMBENCH_SSL_MODE", "zero_bit")).strip().lower()
        if mode_val not in {"zero_bit", "multi_bit"}:
            raise ValueError(f"SSL mode must be 'zero_bit' or 'multi_bit', got {mode_val!r}")
        self._mode = mode_val
        self._seed = int(seed)
        self._device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        cfg_path = self._root / "configs" / f"{self._mode}.yaml"
        cfg = _load_yaml(cfg_path)
        self._cfg = cfg

        self._whitening_path = _resolve_file(
            whitening_path,
            "WMBENCH_SSL_WHITENING",
            self._root / "checkpoints" / "whitening.pt",
            "SSL whitening checkpoint",
        )
        key_default = self._root / "checkpoints" / ("key_zero.npy" if self._mode == "zero_bit" else "key_multi.npy")
        self._key_path = _resolve_file(
            key_path,
            "WMBENCH_SSL_KEY",
            key_default,
            "SSL key file",
        )

        whitening = PCAWhitening.load(self._whitening_path)
        self._backbone = Backbone(whitening=whitening, feat_dim=int(cfg["feat_dim"])).to(self._device).eval()

        self._key = load_key(self._key_path).to(self._device)
        self._fpr = float(fpr if fpr is not None else cfg.get("fpr", 1e-6))
        self._embed_cfg = EmbedConfig(
            target_psnr=float(cfg["target_psnr"]),
            n_iter=int(cfg["n_iter"]),
            lr=float(cfg["lr"]),
            lambda_w=float(cfg["lambda_w"]),
        )
        self._margin = float(cfg.get("margin", 5.0))

        self._message: torch.Tensor | None = None
        if self._mode == "multi_bit":
            k = int(cfg["n_bits"])
            msg = message_bits or os.environ.get("WMBENCH_SSL_MESSAGE")
            if msg:
                parsed = parse_message(str(msg))
                if parsed.numel() != k:
                    raise ValueError(f"SSL message length {parsed.numel()} != n_bits {k}")
                self._message = parsed.to(self._device).float()
            else:
                gen = torch.Generator(device="cpu").manual_seed(self._seed)
                self._message = torch.randint(0, 2, (k,), generator=gen).mul_(2).sub_(1).to(self._device).float()

        self._embed_zero_bit = embed_zero_bit
        self._embed_multi_bit = embed_multi_bit
        self._detect_zero_bit = detect_zero_bit
        self._decode_multi_bit = decode_multi_bit
        self._last_payload: dict | None = None

    @property
    def name(self) -> str:
        return "ssl"

    def _payload(self) -> dict:
        payload = {"mode": self._mode}
        if self._mode == "multi_bit" and self._message is not None:
            payload["message"] = self._message.detach().cpu().numpy().astype(np.int8)
        return payload

    def payload_for_meta(self) -> dict | None:
        return self._last_payload

    def embed(self, image: Image.Image) -> Image.Image:
        x = _pil_to_tensor(image).to(self._device)
        with torch.no_grad():
            if self._mode == "zero_bit":
                out = self._embed_zero_bit(
                    self._backbone,
                    x,
                    key=self._key,
                    fpr=self._fpr,
                    config=self._embed_cfg,
                    seed=self._seed,
                )
            else:
                assert self._message is not None
                out = self._embed_multi_bit(
                    self._backbone,
                    x,
                    carriers=self._key,
                    messages=self._message.unsqueeze(0),
                    margin=self._margin,
                    config=self._embed_cfg,
                    seed=self._seed,
                )
        self._last_payload = self._payload()
        return _tensor_to_pil(out)

    def detect(
        self,
        image: Image.Image,
        original: Image.Image | None = None,
        *,
        meta: dict | None = None,
        blind: bool = False,
    ) -> float:
        del original, blind
        x = _pil_to_tensor(image).to(self._device)
        payload = meta if meta is not None else self._last_payload
        if payload is None:
            payload = self._payload()

        with torch.no_grad():
            if self._mode == "zero_bit":
                _det, score = self._detect_zero_bit(self._backbone, x, key=self._key, fpr=self._fpr)
                # Raw score is unbounded; squash monotonically for wmbench's [0,1]-style reporting.
                return float(torch.sigmoid(score[0]).item())

            decoded = self._decode_multi_bit(self._backbone, x, carriers=self._key)[0].sign()
            msg = payload.get("message")
            if msg is None and self._message is not None:
                truth = self._message
            else:
                truth = torch.from_numpy(np.asarray(msg, dtype=np.int8)).to(self._device).float()
            acc = (decoded == truth.sign()).float().mean()
            return float(acc.item())
