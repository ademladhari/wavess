from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from wmbench.watermarks.base import WatermarkAdapter


def _resolve_tree_ring_root(user_path: str | None = None) -> Path:
    root = Path(__file__).resolve().parents[2]
    candidates = [
        user_path,
        os.environ.get("WMBENCH_TREE_RING_ROOT"),
        str(root / "tree-ring"),
        str(root.parent / "wavesPipeline" / "tree-ring"),
        str(Path.home() / "Desktop" / "thesis" / "wavesPipeline" / "tree-ring"),
    ]
    for c in candidates:
        if c:
            p = Path(c).expanduser().resolve()
            if (p / "src" / "treering").is_dir():
                return p
    listed = ", ".join(repr(c) for c in candidates if c)
    raise FileNotFoundError(
        "Could not locate Tree-Ring implementation root (expects src/treering). "
        f"Tried: {listed}. Set WMBENCH_TREE_RING_ROOT."
    )


def _to_score_from_pvalue(p_value: float) -> float:
    # Tree-Ring uses smaller p-value => stronger watermark evidence.
    return float(np.clip(1.0 - float(p_value), 0.0, 1.0))


class TreeRingAdapter(WatermarkAdapter):
    """
    Adapter for Tree-Ring watermarking using the project under tree-ring/src/treering.

    This wrapper calls the implementation's own components:
      - treering.generate.TreeRingGenerator (embed/generation)
      - treering.invert.DDIMInverter + treering.detect.detect_watermark (detect)
      - treering.keygen.generate_key_material + treering.fourier.circular_mask (key material)
    """

    def __init__(
        self,
        *,
        tree_ring_root: str | None = None,
        config_path: str | None = None,
        prompt: str | None = None,
    ):
        self._root = _resolve_tree_ring_root(tree_ring_root)
        src_root = self._root / "src"
        if str(src_root) not in sys.path:
            sys.path.insert(0, str(src_root))

        from treering.config import load_config
        from treering.detect import detect_watermark
        from treering.fourier import circular_mask
        from treering.generate import TreeRingGenerator
        from treering.keygen import generate_key_material
        from treering.invert import DDIMInverter

        cfg_path = Path(
            config_path
            or os.environ.get("WMBENCH_TREE_RING_CONFIG")
            or (self._root / "configs" / "base.yaml")
        ).expanduser().resolve()
        cfg = load_config(cfg_path)
        self._cfg = cfg

        self._prompt = str(prompt or os.environ.get("WMBENCH_TREE_RING_PROMPT", "a photo"))
        # Optional explicit device override, useful for multi-process multi-GPU detect workers.
        model_device = (os.environ.get("WMBENCH_TREE_RING_DEVICE") or str(cfg.model.device)).strip() or str(cfg.model.device)

        self._generator = TreeRingGenerator(
            model_id=cfg.model.model_id,
            device=model_device,
            dtype=cfg.model.dtype,
            num_inference_steps=cfg.model.num_inference_steps,
            guidance_scale=cfg.model.guidance_scale,
        )

        inv_dtype_name = os.environ.get("WMBENCH_TREE_RING_INVERT_DTYPE", str(cfg.model.dtype)).strip().lower()
        self._inv_dtype = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }.get(inv_dtype_name, torch.float16)
        self._inv_device = os.environ.get("WMBENCH_TREE_RING_INVERT_DEVICE", model_device).strip() or model_device
        # Use the same already-loaded SD pipeline for inversion to avoid a second
        # from_pretrained load (the Windows crash path).
        self._inverter = DDIMInverter(
            model_id=cfg.model.model_id,
            device=self._inv_device,
            dtype=self._inv_dtype,
            pipeline=self._generator.pipeline,
        )

        self._mask = circular_mask(
            cfg.watermark.height,
            cfg.watermark.width,
            cfg.watermark.radius,
            device=model_device,
        )
        self._key = generate_key_material(
            channels=cfg.watermark.channels,
            height=cfg.watermark.height,
            width=cfg.watermark.width,
            mask=self._mask,
            variant=cfg.watermark.key_variant,
            seed=cfg.watermark.seed,
            device=model_device,
        )
        self._detect_watermark = detect_watermark

    def _ensure_inverter(self):
        return self._inverter

    @property
    def name(self) -> str:
        return "tree-ring"

    @property
    def embed_meta_shared(self) -> bool:
        return True

    def payload_for_meta(self) -> dict | None:
        # Tree-Ring key material is deterministic from config+seed; no per-image payload needed.
        return {
            "model_id": self._cfg.model.model_id,
            "seed": int(self._cfg.watermark.seed),
            "radius": int(self._cfg.watermark.radius),
            "key_variant": str(self._cfg.watermark.key_variant),
        }

    def _generate_one(self) -> Image.Image:
        out = self._generator.generate(
            prompts=[self._prompt],
            key=self._key,
            seed=self._cfg.watermark.seed,
        )
        return out.images[0].convert("RGB")

    def embed_batch(self, images: list[Image.Image]) -> list[Image.Image]:
        """One SD generation per batch chunk; replicate (same seed/key as repeated ``embed``)."""
        if not images:
            return []
        one = self._generate_one()
        return [one.copy() for _ in images]

    def embed(self, image: Image.Image) -> Image.Image:
        del image  # Tree-Ring embedding is generation-time latent injection.
        return self._generate_one()

    def detect(
        self,
        image: Image.Image,
        original: Image.Image | None = None,
        *,
        meta: dict | None = None,
        blind: bool = False,
    ) -> float:
        del original, meta, blind
        inverter = self._ensure_inverter()
        inverted = inverter.invert(
            images=[image.convert("RGB")],
            invert_prompt=self._cfg.detection.invert_prompt,
            num_inversion_steps=self._cfg.detection.num_inversion_steps,
        )
        detection = self._detect_watermark(
            inverted_latents=inverted,
            key=self._key,
            threshold=self._cfg.detection.threshold,
            alpha=self._cfg.detection.alpha,
            variance_estimation=self._cfg.detection.variance_estimation,
            pvalue_tail=self._cfg.detection.pvalue_tail,
        )[0]
        return _to_score_from_pvalue(detection.p_value)
