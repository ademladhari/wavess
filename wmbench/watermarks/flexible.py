from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from wmbench.watermarks.base import WatermarkAdapter


def _resolve_flexible_root(user_path: str | None = None) -> Path:
    root = Path(__file__).resolve().parents[2]
    candidates = [
        user_path,
        os.environ.get("WMBENCH_FLEXIBLE_ROOT"),
        str(root / "flexible"),
    ]
    for c in candidates:
        if c:
            p = Path(c).expanduser().resolve()
            if (p / "src").is_dir():
                return p
    listed = ", ".join(repr(c) for c in candidates if c)
    raise FileNotFoundError(
        "Could not locate flexible implementation root (expects a src/ directory). "
        f"Tried: {listed}. Set WMBENCH_FLEXIBLE_ROOT."
    )


def _resolve_checkpoints_dir(root: Path, user_path: str | None = None) -> Path:
    candidates = [
        user_path,
        os.environ.get("WMBENCH_FLEXIBLE_CHECKPOINTS"),
        str(root / "checkpoints"),
    ]
    for c in candidates:
        if c:
            p = Path(c).expanduser().resolve()
            if p.is_dir():
                return p
    listed = ", ".join(repr(c) for c in candidates if c)
    raise FileNotFoundError(
        "Could not locate flexible checkpoints directory. "
        f"Tried: {listed}. Set WMBENCH_FLEXIBLE_CHECKPOINTS."
    )


def _to_tensor_01(image: Image.Image, size: int) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB").resize((size, size), Image.BICUBIC), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t


class FlexibleAdapter(WatermarkAdapter):
    """
    Adapter for the "flexible" SDM watermarking project.

    Uses the project's own models/code paths:
      - WatermarkEncoder + FrozenSDM for embed
      - WatermarkExtractor + WatermarkDecoder for detect
    """

    def __init__(
        self,
        *,
        flexible_root: str | None = None,
        checkpoints_dir: str | None = None,
        n_bits: int | None = None,
        prompt: str | None = None,
        seed: int = 1234,
        device: str | None = None,
    ):
        self._root = _resolve_flexible_root(flexible_root)
        if str(self._root) not in sys.path:
            sys.path.insert(0, str(self._root))

        from src.eval.extraction import bit_accuracy
        from src.models import WatermarkDecoder, WatermarkEncoder, WatermarkExtractor
        from src.models.sdm import FrozenSDM
        from src.utils import load_config

        cfg = load_config(str(self._root / "configs" / "default.yaml"))
        self._cfg = cfg
        self._n_bits = int(n_bits if n_bits is not None else cfg.watermark.primary_bits)
        self._prompt = str(prompt if prompt is not None else os.environ.get("WMBENCH_FLEXIBLE_PROMPT", "a photo"))
        self._device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._bit_accuracy = bit_accuracy

        ckpt_dir = _resolve_checkpoints_dir(self._root, checkpoints_dir)
        enc_ckpt = ckpt_dir / f"encoder_b{self._n_bits}.pt"
        ext_ckpt = ckpt_dir / f"extractor_b{self._n_bits}.pt"
        dec_ft_ckpt = ckpt_dir / f"decoder_ft_b{self._n_bits}.pt"
        dec_ckpt = ckpt_dir / f"decoder_b{self._n_bits}.pt"
        if not enc_ckpt.is_file():
            raise FileNotFoundError(f"Missing flexible encoder checkpoint: {enc_ckpt}")
        if not ext_ckpt.is_file():
            raise FileNotFoundError(f"Missing flexible extractor checkpoint: {ext_ckpt}")
        dec_use = dec_ft_ckpt if dec_ft_ckpt.is_file() else dec_ckpt
        if not dec_use.is_file():
            raise FileNotFoundError(
                f"Missing flexible decoder checkpoint: {dec_ft_ckpt} (or fallback {dec_ckpt})"
            )

        self._image_size = int(cfg.sdm.image_size)
        self._steps = int(cfg.sdm.num_inference_steps)
        self._guidance = float(cfg.sdm.guidance_scale)

        self._enc = WatermarkEncoder(
            n_bits=self._n_bits,
            latent_channels=int(cfg.sdm.latent_channels),
            latent_size=int(cfg.sdm.latent_size),
            base_channels=int(cfg.arch.encoder.base_channels),
            conv_blocks=int(cfg.arch.encoder.conv_blocks),
            feature_hw=int(cfg.arch.encoder.latent_feature_hw),
            kernel_size=int(cfg.arch.encoder.kernel_size),
        ).to(self._device)
        self._enc.load_state_dict(torch.load(enc_ckpt, map_location=self._device))
        self._enc.eval()

        self._ext = WatermarkExtractor(
            image_size=int(cfg.sdm.image_size),
            in_channels=3,
            latent_channels=int(cfg.sdm.latent_channels),
            latent_size=int(cfg.sdm.latent_size),
            in_downsample=int(cfg.arch.extractor.in_downsample),
            token_dim=int(cfg.arch.extractor.token_dim),
            transformer_layers=int(cfg.arch.extractor.transformer_layers),
            transformer_heads=int(cfg.arch.extractor.transformer_heads),
            mlp_hidden=int(cfg.arch.extractor.mlp_hidden),
            ffn_dim=int(cfg.arch.extractor.ffn_dim),
        ).to(self._device)
        self._ext.load_state_dict(torch.load(ext_ckpt, map_location=self._device))
        self._ext.eval()

        self._dec = WatermarkDecoder(
            n_bits=self._n_bits,
            latent_channels=int(cfg.sdm.latent_channels),
            latent_size=int(cfg.sdm.latent_size),
            base_channels=int(cfg.arch.decoder.base_channels),
            tconv_blocks=int(cfg.arch.decoder.tconv_blocks),
            feature_hw=int(cfg.arch.encoder.latent_feature_hw),
            kernel_size=int(cfg.arch.decoder.kernel_size),
        ).to(self._device)
        self._dec.load_state_dict(torch.load(dec_use, map_location=self._device))
        self._dec.eval()

        sdm_dtype = torch.float16 if str(cfg.sdm.dtype) == "float16" else torch.float32
        self._sdm = FrozenSDM(
            pretrained_id=str(cfg.sdm.pretrained_id),
            dtype=sdm_dtype,
            device=self._device,
            enable_attention_slicing=bool(cfg.sdm.attention_slicing),
            enable_vae_slicing=bool(cfg.sdm.vae_slicing),
            enable_xformers=bool(cfg.sdm.enable_xformers),
            cache_dir=str((self._root / str(cfg.paths.hf_cache)).resolve()),
        )

        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        self._w = torch.randint(0, 2, (1, self._n_bits), generator=gen).to(self._device, dtype=torch.float32)
        self._last_payload: dict | None = None
        self._z_init_cached: torch.Tensor | None = None

    @property
    def name(self) -> str:
        return "flexible"

    @property
    def embed_meta_shared(self) -> bool:
        return True

    def payload_for_meta(self) -> dict | None:
        return self._last_payload

    def _payload(self) -> dict:
        return {"w_bits": self._w.detach().cpu().numpy().astype(np.uint8), "n_bits": int(self._n_bits)}

    def _get_z_init(self) -> torch.Tensor:
        """Cached encoder output for fixed payload ``_w`` (same for every embed)."""
        if self._z_init_cached is None:
            with torch.no_grad():
                z_img, _, _ = self._enc(self._w)
                self._z_init_cached = z_img.to(self._sdm.dtype)
        return self._z_init_cached

    @staticmethod
    def _sdm_tensors_to_pil(image_tensor: torch.Tensor) -> list[Image.Image]:
        out: list[Image.Image] = []
        for i in range(int(image_tensor.shape[0])):
            arr = (
                image_tensor[i]
                .detach()
                .clamp(0.0, 1.0)
                .mul(255.0)
                .round()
                .to(torch.uint8)
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
            out.append(Image.fromarray(arr, mode="RGB"))
        return out

    def _generate_one(self) -> Image.Image:
        """Single SD forward (inputs ignored; payload fixed by ``_w``)."""
        with torch.no_grad():
            out = self._sdm.generate(
                z_init=self._get_z_init(),
                prompt=self._prompt,
                num_inference_steps=self._steps,
                guidance_scale=self._guidance,
            )
        self._last_payload = self._payload()
        return self._sdm_tensors_to_pil(out.image)[0]

    def embed_batch(self, images: list[Image.Image]) -> list[Image.Image]:
        """One SD generation per batch chunk; replicate for each slot (same as repeated ``embed``)."""
        if not images:
            return []
        one = self._generate_one()
        return [one.copy() for _ in images]

    def embed(self, image: Image.Image) -> Image.Image:
        del image  # Flexible embedding is performed via the method's own latent->SDM generation path.
        return self._generate_one()

    def detect(
        self,
        image: Image.Image,
        original: Image.Image | None = None,
        *,
        meta: dict | None = None,
        blind: bool = False,
    ) -> float:
        del original, blind
        payload = meta if meta is not None else self._last_payload
        if payload is None:
            payload = self._payload()
        ref_bits = torch.from_numpy(np.asarray(payload["w_bits"], dtype=np.uint8).reshape(1, -1)).to(
            self._device, dtype=torch.float32
        )
        with torch.no_grad():
            img = _to_tensor_01(image, self._image_size).to(self._device)
            z = self._ext(img)
            wd = self._dec(z)
            score = float(self._bit_accuracy(wd, ref_bits))
        return float(np.clip(score, 0.0, 1.0))
