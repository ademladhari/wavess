from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from wmbench.watermarks.base import WatermarkAdapter


def _resolve_robin_root(user_path: str | None = None) -> Path:
    root = Path(__file__).resolve().parents[2]
    candidates = [
        user_path,
        os.environ.get("WMBENCH_ROBIN_ROOT"),
        str(root / "ROBIN"),
        str(root.parent / "wavesPipeline" / "ROBIN"),
        str(Path.home() / "Desktop" / "thesis" / "wavesPipeline" / "ROBIN"),
    ]
    for c in candidates:
        if not c:
            continue
        p = Path(c).expanduser().resolve()
        if (p / "stable_diffusion_robin.py").is_file() and (p / "inverse_stable_diffusion.py").is_file():
            return p
    listed = ", ".join(repr(c) for c in candidates if c)
    raise FileNotFoundError(
        "Could not locate ROBIN implementation root (expects stable_diffusion_robin.py and "
        f"inverse_stable_diffusion.py). Tried: {listed}. Set WMBENCH_ROBIN_ROOT."
    )


def _resolve_robin_wm_path(root: Path, user_path: str | None = None) -> Path:
    candidates = [
        user_path,
        os.environ.get("WMBENCH_ROBIN_WM_PATH"),
    ]
    for c in candidates:
        if not c:
            continue
        p = Path(c).expanduser().resolve()
        if p.is_file():
            return p
        if p.is_dir():
            pts = sorted(p.rglob("*.pt"))
            if pts:
                return pts[-1]

    # Best-effort fallback: most ROBIN runs save optimized wm .pt checkpoints.
    fallback_globs = [
        root / "ckpts",
        root / "checkpoints",
        root,
    ]
    for base in fallback_globs:
        if not base.exists():
            continue
        pts = sorted(base.rglob("*.pt"))
        if pts:
            return pts[-1]

    raise FileNotFoundError(
        "Could not locate ROBIN watermark checkpoint (.pt). Set WMBENCH_ROBIN_WM_PATH to a file path "
        "produced by ROBIN optimization (contains opt_wm and optionally opt_acond)."
    )


def _seed_from_image(image: Image.Image, *, base_seed: int) -> int:
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    digest = hashlib.sha1(arr.tobytes(), usedforsecurity=False).hexdigest()
    return (int(digest[:8], 16) + int(base_seed)) % (2**31 - 1)


def _score_from_distance(distance: float) -> float:
    # Smaller distance to target watermark => stronger watermark evidence.
    return float(np.clip(1.0 / (1.0 + float(distance)), 0.0, 1.0))


class ROBINAdapter(WatermarkAdapter):
    """
    Adapter for ROBIN watermarking.

    Uses the project's own code from ROBIN/:
      - inverse_stable_diffusion.InversableStableDiffusionPipeline
      - optim_utils.get_watermarking_mask / inject_watermark
    """

    def __init__(
        self,
        *,
        robin_root: str | None = None,
        wm_path: str | None = None,
        model_id: str | None = None,
        prompt: str | None = None,
        device: str | None = None,
    ):
        self._root = _resolve_robin_root(robin_root)
        if str(self._root) not in sys.path:
            sys.path.insert(0, str(self._root))

        from diffusers import DPMSolverMultistepScheduler
        from inverse_stable_diffusion import InversableStableDiffusionPipeline
        from optim_utils import get_watermarking_mask

        self._device = (device or os.environ.get("WMBENCH_ROBIN_DEVICE", "")).strip() or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._model_id = (
            model_id
            or os.environ.get("WMBENCH_ROBIN_MODEL_ID")
            or "sd2-community/stable-diffusion-2-1-base"
        )
        self._prompt = str(prompt or os.environ.get("WMBENCH_ROBIN_PROMPT", "a photo"))
        self._base_seed = int(os.environ.get("WMBENCH_ROBIN_GEN_SEED", "0"))
        self._guidance_scale = float(os.environ.get("WMBENCH_ROBIN_GUIDANCE_SCALE", "7.5"))
        self._num_inference_steps = int(os.environ.get("WMBENCH_ROBIN_INFERENCE_STEPS", "50"))
        self._watermarking_steps = int(os.environ.get("WMBENCH_ROBIN_WATERMARK_STEPS", "35"))
        self._detect_steps = int(os.environ.get("WMBENCH_ROBIN_DETECT_STEPS", str(self._num_inference_steps)))
        self._image_length = int(os.environ.get("WMBENCH_ROBIN_IMAGE_LENGTH", "512"))

        self._args = SimpleNamespace(
            w_seed=int(os.environ.get("WMBENCH_ROBIN_W_SEED", "999999")),
            w_channel=int(os.environ.get("WMBENCH_ROBIN_W_CHANNEL", "3")),
            w_pattern=str(os.environ.get("WMBENCH_ROBIN_W_PATTERN", "ring")),
            w_mask_shape=str(os.environ.get("WMBENCH_ROBIN_W_MASK_SHAPE", "circle")),
            w_up_radius=int(os.environ.get("WMBENCH_ROBIN_W_UP_RADIUS", "30")),
            w_low_radius=int(os.environ.get("WMBENCH_ROBIN_W_LOW_RADIUS", "5")),
            w_measurement=str(os.environ.get("WMBENCH_ROBIN_W_MEASUREMENT", "l1_complex")),
            w_injection=str(os.environ.get("WMBENCH_ROBIN_W_INJECTION", "complex")),
            w_pattern_const=float(os.environ.get("WMBENCH_ROBIN_W_PATTERN_CONST", "0.0")),
        )

        self._wm_path = _resolve_robin_wm_path(self._root, wm_path)
        ckpt = torch.load(self._wm_path, map_location="cpu")
        if "opt_wm" not in ckpt:
            raise KeyError(
                f"ROBIN checkpoint missing key 'opt_wm': {self._wm_path}. "
                "Expected optimized ROBIN checkpoint."
            )
        self._opt_wm = ckpt["opt_wm"].to(self._device)

        scheduler = DPMSolverMultistepScheduler.from_pretrained(self._model_id, subfolder="scheduler")
        self._pipe = InversableStableDiffusionPipeline.from_pretrained(
            self._model_id,
            scheduler=scheduler,
            torch_dtype=torch.float32,
        ).to(self._device)

        self._text_embeddings = self._pipe.get_text_embedding("")
        self._opt_acond = ckpt.get("opt_acond")
        if self._opt_acond is not None:
            self._opt_acond = self._opt_acond.to(self._text_embeddings.dtype).to(self._device)

        init_latents = self._pipe.get_random_latents(
            height=self._image_length,
            width=self._image_length,
        )
        self._wm_mask = get_watermarking_mask(init_latents, self._args, self._device)

    @property
    def name(self) -> str:
        return "robin"

    def payload_for_meta(self) -> dict | None:
        return {
            "model_id": self._model_id,
            "wm_path": str(self._wm_path),
            "w_channel": int(self._args.w_channel),
            "w_up_radius": int(self._args.w_up_radius),
            "w_low_radius": int(self._args.w_low_radius),
            "w_mask_shape": str(self._args.w_mask_shape),
        }

    def embed(self, image: Image.Image) -> Image.Image:
        from optim_utils import set_random_seed

        seed = _seed_from_image(image, base_seed=self._base_seed)
        set_random_seed(seed)
        generator = torch.Generator(device=self._device).manual_seed(seed)
        init_latents = self._pipe.get_random_latents(
            height=self._image_length,
            width=self._image_length,
            generator=generator,
        )
        outputs = self._pipe(
            self._prompt,
            num_images_per_prompt=1,
            guidance_scale=self._guidance_scale,
            num_inference_steps=self._num_inference_steps,
            height=self._image_length,
            width=self._image_length,
            latents=init_latents,
            watermarking_mask=self._wm_mask,
            watermarking_steps=self._watermarking_steps,
            args=self._args,
            gt_patch=self._opt_wm,
            lguidance=self._guidance_scale,
            opt_acond=self._opt_acond,
        )
        return outputs.images[0].convert("RGB")

    def detect(
        self,
        image: Image.Image,
        original: Image.Image | None = None,
        *,
        meta: dict | None = None,
        blind: bool = False,
    ) -> float:
        del original, meta, blind
        from optim_utils import transform_img

        img = transform_img(image, target_size=self._image_length).unsqueeze(0).to(self._text_embeddings.dtype).to(
            self._device
        )
        image_latents = self._pipe.get_image_latents(img, sample=False)
        reversed_latents = self._pipe.forward_diffusion(
            latents=image_latents,
            text_embeddings=self._text_embeddings,
            guidance_scale=1.0,
            num_inference_steps=self._detect_steps,
        )
        rev_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents), dim=(-1, -2))
        target_patch = self._opt_wm.to(rev_fft.dtype)
        distance = torch.abs(rev_fft[self._wm_mask] - target_patch[self._wm_mask]).mean().item()
        return _score_from_distance(distance)

