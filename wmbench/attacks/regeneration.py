from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from typing import Callable

from .base import Attack

try:
    from compressai.zoo import (
        bmshj2018_factorized,
        bmshj2018_hyperprior,
        mbt2018_mean,
        mbt2018,
        cheng2020_anchor,
    )
except ImportError:  # pragma: no cover - optional dependency
    bmshj2018_factorized = None  # type: ignore[misc, assignment]

_RESD_PIPELINE_IMPORT_ERROR: ImportError | None = None
try:
    from wmbench.attacks.resd_pipeline import ReSDPipeline
except ImportError as _resd_import_err:  # pragma: no cover - optional dependency
    ReSDPipeline = None  # type: ignore[misc, assignment]
    _RESD_PIPELINE_IMPORT_ERROR = _resd_import_err


def _resd_pipeline_cls():
    """Return ReSDPipeline class or raise with the original import error chained."""
    if ReSDPipeline is None:
        raise ImportError(
            "Diffusion regeneration requires the `diffusers` package. "
            "In your activated venv run:  pip install diffusers"
        ) from _RESD_PIPELINE_IMPORT_ERROR
    return ReSDPipeline


class VAEWMAttacker:
    """WAVES VAE regeneration (waves/regeneration/regen.py VAEWMAttacker)."""

    def __init__(self, model_name: str, strength: int = 1, device: str | torch.device = "cpu"):
        if bmshj2018_factorized is None:
            raise ImportError("compressai is required for Regen-VAE")
        self.device = torch.device(device) if isinstance(device, str) else device
        st = int(strength)
        if model_name == "bmshj2018-factorized":
            self.model = bmshj2018_factorized(quality=st, pretrained=True).eval().to(self.device)
        elif model_name == "bmshj2018-hyperprior":
            self.model = bmshj2018_hyperprior(quality=st, pretrained=True).eval().to(self.device)
        elif model_name == "mbt2018-mean":
            self.model = mbt2018_mean(quality=st, pretrained=True).eval().to(self.device)
        elif model_name == "mbt2018":
            self.model = mbt2018(quality=st, pretrained=True).eval().to(self.device)
        elif model_name == "cheng2020-anchor":
            self.model = cheng2020_anchor(quality=st, pretrained=True).eval().to(self.device)
        else:
            raise ValueError(f"Unsupported VAE model name {model_name!r}")
        self.model_name = model_name

    def attack(self, image: Image.Image) -> Image.Image:
        img = image.convert("RGB")
        img = img.resize((512, 512))
        img_t = transforms.ToTensor()(img).unsqueeze(0).to(self.device)
        out = self.model(img_t)
        out["x_hat"].clamp_(0, 1)
        return transforms.ToPILImage()(out["x_hat"].squeeze().cpu())


class DiffWMAttacker:
    """WAVES diffusion regeneration (waves/regeneration/regen.py DiffWMAttacker)."""

    def __init__(self, pipe, noise_step: int = 60, captions: dict | None = None):
        self.pipe = pipe
        self.device = pipe.device
        self.noise_step = int(noise_step)
        self.captions = captions or {}

    def attack(self, image: Image.Image) -> Image.Image:
        out = self.attack_batch([image])
        return out[0]

    def _single_noised_latent(self, image: Image.Image, vae_dtype: torch.dtype, timestep: torch.Tensor) -> torch.Tensor:
        # Keep WAVES-like deterministic seed per image. This intentionally reseeds each sample.
        generator = torch.Generator(device=self.device).manual_seed(1024)
        img = np.asarray(image) / 255.0
        img = (img - 0.5) * 2.0
        img_t = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        img_vae = img_t.to(device=self.device, dtype=vae_dtype)
        latents = self.pipe.vae.encode(img_vae).latent_dist
        latents = latents.sample(generator) * self.pipe.vae.config.scaling_factor
        noise = torch.randn(
            [1, 4, img_t.shape[-2] // 8, img_t.shape[-1] // 8],
            device=self.device,
            dtype=vae_dtype,
            generator=generator,
        )
        latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        return latents.type(vae_dtype)

    def attack_batch(self, images: list[Image.Image]) -> list[Image.Image]:
        if not images:
            return []
        by_size: dict[tuple[int, int], list[tuple[int, Image.Image]]] = {}
        for i, im in enumerate(images):
            by_size.setdefault(im.size, []).append((i, im))
        if len(by_size) > 1:
            out: list[Image.Image | None] = [None] * len(images)
            for group in by_size.values():
                idxs = [it[0] for it in group]
                ims = [it[1] for it in group]
                vals = self.attack_batch(ims)
                for idx, val in zip(idxs, vals):
                    out[idx] = val
            return [im for im in out if im is not None]

        with torch.no_grad():
            vae_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            timestep = torch.tensor([self.noise_step], dtype=torch.long, device=self.device)
            latents = torch.cat(
                [self._single_noised_latent(im, vae_dtype, timestep) for im in images],
                dim=0,
            )
            gens = [torch.Generator(device=self.device).manual_seed(1024) for _ in images]
            out = self.pipe(
                [""] * len(images),
                head_start_latents=latents,
                head_start_step=50 - max(self.noise_step // 20, 1),
                guidance_scale=7.5,
                generator=gens,
            )
            ims = out[0]
            return [im for im in ims]


def remove_watermark(
    attack_method: str,
    image: Image.Image,
    strength: float | int,
    model: str,
    device: torch.device,
    *,
    vae_attacker: VAEWMAttacker | None = None,
    diffusion_pipe: object | None = None,
    diffusion_factory: object | None = None,
) -> Image.Image:
    """Core WAVES dispatch (regen.py remove_watermark), with fixed VAE kwargs."""
    if attack_method == "regen_vae":
        if vae_attacker is None:
            attacker = VAEWMAttacker(model, strength=int(strength), device=device)
        else:
            attacker = vae_attacker
        return attacker.attack(image)

    if attack_method == "regen_diffusion":
        pipe = diffusion_pipe
        if pipe is None:
            _RP = _resd_pipeline_cls()
            if diffusion_factory is not None:
                pipe = diffusion_factory()
            else:
                pipe = _RP.from_pretrained(
                    model,
                    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                    revision="fp16" if device.type == "cuda" else None,
                )
                pipe.set_progress_bar_config(disable=True)
                pipe.to(device)
        attacker = DiffWMAttacker(pipe, noise_step=int(strength), captions={})
        return attacker.attack(image)

    raise ValueError(f"Unknown regeneration attack method: {attack_method!r}")


def remove_watermark_batch(
    attack_method: str,
    images: list[Image.Image],
    strength: float | int,
    model: str,
    device: torch.device,
    *,
    vae_attacker: VAEWMAttacker | None = None,
    diffusion_pipe: object | None = None,
    diffusion_factory: object | None = None,
) -> list[Image.Image]:
    if not images:
        return []
    if attack_method == "regen_vae":
        if vae_attacker is None:
            attacker = VAEWMAttacker(model, strength=int(strength), device=device)
        else:
            attacker = vae_attacker
        return [attacker.attack(im) for im in images]

    if attack_method == "regen_diffusion":
        pipe = diffusion_pipe
        if pipe is None:
            _RP = _resd_pipeline_cls()
            if diffusion_factory is not None:
                pipe = diffusion_factory()
            else:
                pipe = _RP.from_pretrained(
                    model,
                    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                    revision="fp16" if device.type == "cuda" else None,
                )
                pipe.set_progress_bar_config(disable=True)
                pipe.to(device)
        attacker = DiffWMAttacker(pipe, noise_step=int(strength), captions={})
        return attacker.attack_batch(images)

    raise ValueError(f"Unknown regeneration attack method: {attack_method!r}")


class DiffusionRegenAttack(Attack):
    def __init__(
        self,
        strengths: list[int],
        diffusion_model_id: str,
        device: torch.device | None = None,
        *,
        pipe_provider: Callable[[], object] | None = None,
    ):
        self.name = "Regen-Diff"
        self.strengths = strengths
        self.model_id = diffusion_model_id
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self._pipe = None
        self._pipe_provider = pipe_provider

    def _get_pipe(self):
        if self._pipe_provider is not None:
            return self._pipe_provider()
        if self._pipe is None:
            _RP = _resd_pipeline_cls()
            self._pipe = _RP.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                revision="fp16" if self.device.type == "cuda" else None,
            )
            self._pipe.set_progress_bar_config(disable=True)
            self._pipe.to(self.device)
        return self._pipe

    def apply(self, image: Image.Image, strength: float | int) -> Image.Image:
        return remove_watermark(
            "regen_diffusion",
            image,
            strength,
            self.model_id,
            self.device,
            diffusion_pipe=self._get_pipe(),
        )

    def apply_batch(self, images: list[Image.Image], strength: float | int) -> list[Image.Image]:
        return remove_watermark_batch(
            "regen_diffusion",
            images,
            strength,
            self.model_id,
            self.device,
            diffusion_pipe=self._get_pipe(),
        )


class VAERegenAttack(Attack):
    def __init__(
        self,
        strengths: list[int],
        vae_model_name: str = "bmshj2018-factorized",
        device: torch.device | None = None,
    ):
        self.name = "Regen-VAE"
        self.strengths = strengths
        self.vae_model_name = vae_model_name
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self._attackers: dict[int, VAEWMAttacker] = {}

    def _get_attacker(self, strength: int) -> VAEWMAttacker:
        # WAVES loads per strength; quality level selects different compression bottleneck
        cached = self._attackers.get(strength)
        if cached is not None:
            return cached
        attacker = VAEWMAttacker(self.vae_model_name, strength=strength, device=self.device)
        self._attackers[strength] = attacker
        return attacker

    def apply(self, image: Image.Image, strength: float | int) -> Image.Image:
        st = int(strength)
        attacker = self._get_attacker(st)
        return attacker.attack(image)

    def apply_batch(self, images: list[Image.Image], strength: float | int) -> list[Image.Image]:
        st = int(strength)
        attacker = self._get_attacker(st)
        return [attacker.attack(im) for im in images]


class Rinse2xDiffAttack(Attack):
    def __init__(
        self,
        strengths: list[int],
        diffusion_model_id: str,
        device: torch.device | None = None,
        *,
        pipe_provider: Callable[[], object] | None = None,
    ):
        self.name = "Rinse-2xDiff"
        self.strengths = strengths
        self.model_id = diffusion_model_id
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self._pipe = None
        self._pipe_provider = pipe_provider

    def _get_pipe(self):
        if self._pipe_provider is not None:
            return self._pipe_provider()
        if self._pipe is None:
            _RP = _resd_pipeline_cls()
            self._pipe = _RP.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                revision="fp16" if self.device.type == "cuda" else None,
            )
            self._pipe.set_progress_bar_config(disable=True)
            self._pipe.to(self.device)
        return self._pipe

    def apply(self, image: Image.Image, strength: float | int) -> Image.Image:
        out = image
        pipe = self._get_pipe()
        for _ in range(2):
            out = remove_watermark(
                "regen_diffusion",
                out,
                strength,
                self.model_id,
                self.device,
                diffusion_pipe=pipe,
            )
        return out

    def apply_batch(self, images: list[Image.Image], strength: float | int) -> list[Image.Image]:
        out = images
        pipe = self._get_pipe()
        for _ in range(2):
            out = remove_watermark_batch(
                "regen_diffusion",
                out,
                strength,
                self.model_id,
                self.device,
                diffusion_pipe=pipe,
            )
        return out


class Rinse4xDiffAttack(Attack):
    def __init__(
        self,
        strengths: list[int],
        diffusion_model_id: str,
        device: torch.device | None = None,
        *,
        pipe_provider: Callable[[], object] | None = None,
    ):
        self.name = "Rinse-4xDiff"
        self.strengths = strengths
        self.model_id = diffusion_model_id
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self._pipe = None
        self._pipe_provider = pipe_provider

    def _get_pipe(self):
        if self._pipe_provider is not None:
            return self._pipe_provider()
        if self._pipe is None:
            _RP = _resd_pipeline_cls()
            self._pipe = _RP.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                revision="fp16" if self.device.type == "cuda" else None,
            )
            self._pipe.set_progress_bar_config(disable=True)
            self._pipe.to(self.device)
        return self._pipe

    def apply(self, image: Image.Image, strength: float | int) -> Image.Image:
        out = image
        pipe = self._get_pipe()
        for _ in range(4):
            out = remove_watermark(
                "regen_diffusion",
                out,
                strength,
                self.model_id,
                self.device,
                diffusion_pipe=pipe,
            )
        return out

    def apply_batch(self, images: list[Image.Image], strength: float | int) -> list[Image.Image]:
        out = images
        pipe = self._get_pipe()
        for _ in range(4):
            out = remove_watermark_batch(
                "regen_diffusion",
                out,
                strength,
                self.model_id,
                self.device,
                diffusion_pipe=pipe,
            )
        return out
