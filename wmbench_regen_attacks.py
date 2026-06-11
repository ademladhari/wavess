"""
WMBench regeneration attacks — standalone export.

Source (do not edit here; sync from wmbench when upstream changes):
  - wmbench/attacks/base.py
  - wmbench/attacks/resd_pipeline.py
  - wmbench/attacks/regeneration.py
  - wmbench/attacks/registry.py (strength defaults only)

Implemented attacks (WAVES paper):
  - Regen-Diff     — single SD head-start denoise pass
  - Regen-VAE      — CompressAI neural codec round-trip
  - Rinse-2xDiff   — Regen-Diff applied twice
  - Rinse-4xDiff   — Regen-Diff applied four times

Not implemented in wmbench (names only in WAVES constants):
  - Regen-DiffP, Regen-KLVAE

Dependencies:
  pip install torch torchvision pillow diffusers compressai

Default model: CompVis/stable-diffusion-v1-4
Default VAE codec: bmshj2018-factorized (CompressAI quality 1–7)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
# Strength grids (WAVES paper appendix / wmbench registry defaults)
# ---------------------------------------------------------------------------

DEFAULT_REGEN_DIFFUSION_STRENGTHS: list[int] = [40, 80, 120, 160, 200]  # noise timesteps
DEFAULT_RINSE_2X_DIFFUSION_STRENGTHS: list[int] = [20, 40, 60, 80, 100]
DEFAULT_RINSE_4X_DIFFUSION_STRENGTHS: list[int] = [10, 20, 30, 40, 50]
DEFAULT_REGEN_VAE_STRENGTHS: list[int] = [1, 2, 4, 5, 7]  # CompressAI quality levels

REGEN_ATTACK_NAMES: tuple[str, ...] = (
    "Regen-Diff",
    "Regen-VAE",
    "Rinse-2xDiff",
    "Rinse-4xDiff",
)

# ---------------------------------------------------------------------------
# Attack base (wmbench/attacks/base.py)
# ---------------------------------------------------------------------------


class Attack(ABC):
    """Single attack operator (one named benchmark attack)."""

    name: str
    strengths: Sequence[float | int]

    @abstractmethod
    def apply(self, image: Image.Image, strength: float | int) -> Image.Image:
        raise NotImplementedError

    def apply_batch(self, images: list[Image.Image], strength: float | int) -> list[Image.Image]:
        return [self.apply(im, strength) for im in images]


# ---------------------------------------------------------------------------
# ReSDPipeline (wmbench/attacks/resd_pipeline.py)
# Head-start latent denoising for diffusion regeneration.
# Adapted from WatermarkAttacker regen_pipe.py / WAVES waves/regeneration/regen.py
# ---------------------------------------------------------------------------

from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.utils import logging as diffusers_logging

logger = diffusers_logging.get_logger(__name__)


class ReSDPipeline(StableDiffusionPipeline):
    """Stable Diffusion pipeline with ``head_start_latents`` / ``head_start_step`` (WAVES regen)."""

    @torch.no_grad()
    def __call__(  # noqa: PLR0913
        self,
        prompt: Union[str, List[str]],
        prompt1_steps: Optional[int] = None,
        prompt2: Optional[str] = None,
        head_start_latents: Optional[Union[torch.FloatTensor, list]] = None,
        head_start_step: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
    ):
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=None,
        )

        if isinstance(prompt, str):
            batch_size = 1
        else:
            batch_size = len(prompt)
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        text_embeddings = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            lora_scale=None,
            clip_skip=None,
        )

        text_embeddings2 = None
        if prompt2 is not None:
            text_embeddings2 = self._encode_prompt(
                prompt2,
                device,
                num_images_per_prompt,
                do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                lora_scale=None,
                clip_skip=None,
            )

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        if head_start_latents is None:
            num_channels_latents = getattr(self.unet, "in_channels", self.unet.config.in_channels)
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                text_embeddings.dtype,
                device,
                generator,
                latents,
            )
        else:
            if isinstance(head_start_latents, list):
                latents = head_start_latents[-1]
                solver_order = getattr(self.scheduler.config, "solver_order", None)
                if solver_order is not None:
                    assert len(head_start_latents) == solver_order
            else:
                latents = head_start_latents

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if head_start_step is None or i >= head_start_step:
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    if prompt1_steps is None or i < prompt1_steps:
                        noise_pred = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=text_embeddings,
                            return_dict=False,
                        )[0]
                    else:
                        noise_pred = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=text_embeddings2,
                            return_dict=False,
                        )[0]

                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                    if (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0:
                        progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        image = self.decode_latents(latents)
        has_nsfw_concept = False

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


# ---------------------------------------------------------------------------
# Core attackers (wmbench/attacks/regeneration.py)
# ---------------------------------------------------------------------------

try:
    from compressai.zoo import (
        bmshj2018_factorized,
        bmshj2018_hyperprior,
        mbt2018_mean,
        mbt2018,
        cheng2020_anchor,
    )
except ImportError:  # pragma: no cover
    bmshj2018_factorized = None  # type: ignore[misc, assignment]


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
        return self.attack_batch([image])[0]

    def _single_noised_latent(self, image: Image.Image, vae_dtype: torch.dtype, timestep: torch.Tensor) -> torch.Tensor:
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
            return [im for im in out[0]]


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
    """Core WAVES dispatch (regen.py remove_watermark)."""
    if attack_method == "regen_vae":
        attacker = vae_attacker or VAEWMAttacker(model, strength=int(strength), device=device)
        return attacker.attack(image)

    if attack_method == "regen_diffusion":
        pipe = diffusion_pipe
        if pipe is None:
            if diffusion_factory is not None:
                pipe = diffusion_factory()
            else:
                pipe = ReSDPipeline.from_pretrained(
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
        attacker = vae_attacker or VAEWMAttacker(model, strength=int(strength), device=device)
        return [attacker.attack(im) for im in images]

    if attack_method == "regen_diffusion":
        pipe = diffusion_pipe
        if pipe is None:
            if diffusion_factory is not None:
                pipe = diffusion_factory()
            else:
                pipe = ReSDPipeline.from_pretrained(
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
        strengths: list[int] | None = None,
        diffusion_model_id: str = "CompVis/stable-diffusion-v1-4",
        device: torch.device | None = None,
        *,
        pipe_provider: Callable[[], object] | None = None,
    ):
        self.name = "Regen-Diff"
        self.strengths = strengths or list(DEFAULT_REGEN_DIFFUSION_STRENGTHS)
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
            self._pipe = ReSDPipeline.from_pretrained(
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
        strengths: list[int] | None = None,
        vae_model_name: str = "bmshj2018-factorized",
        device: torch.device | None = None,
    ):
        self.name = "Regen-VAE"
        self.strengths = strengths or list(DEFAULT_REGEN_VAE_STRENGTHS)
        self.vae_model_name = vae_model_name
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self._attackers: dict[int, VAEWMAttacker] = {}

    def _get_attacker(self, strength: int) -> VAEWMAttacker:
        cached = self._attackers.get(strength)
        if cached is not None:
            return cached
        attacker = VAEWMAttacker(self.vae_model_name, strength=strength, device=self.device)
        self._attackers[strength] = attacker
        return attacker

    def apply(self, image: Image.Image, strength: float | int) -> Image.Image:
        return self._get_attacker(int(strength)).attack(image)

    def apply_batch(self, images: list[Image.Image], strength: float | int) -> list[Image.Image]:
        attacker = self._get_attacker(int(strength))
        return [attacker.attack(im) for im in images]


class Rinse2xDiffAttack(Attack):
    def __init__(
        self,
        strengths: list[int] | None = None,
        diffusion_model_id: str = "CompVis/stable-diffusion-v1-4",
        device: torch.device | None = None,
        *,
        pipe_provider: Callable[[], object] | None = None,
    ):
        self.name = "Rinse-2xDiff"
        self.strengths = strengths or list(DEFAULT_RINSE_2X_DIFFUSION_STRENGTHS)
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
            self._pipe = ReSDPipeline.from_pretrained(
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
                "regen_diffusion", out, strength, self.model_id, self.device, diffusion_pipe=pipe
            )
        return out

    def apply_batch(self, images: list[Image.Image], strength: float | int) -> list[Image.Image]:
        out = images
        pipe = self._get_pipe()
        for _ in range(2):
            out = remove_watermark_batch(
                "regen_diffusion", out, strength, self.model_id, self.device, diffusion_pipe=pipe
            )
        return out


class Rinse4xDiffAttack(Attack):
    def __init__(
        self,
        strengths: list[int] | None = None,
        diffusion_model_id: str = "CompVis/stable-diffusion-v1-4",
        device: torch.device | None = None,
        *,
        pipe_provider: Callable[[], object] | None = None,
    ):
        self.name = "Rinse-4xDiff"
        self.strengths = strengths or list(DEFAULT_RINSE_4X_DIFFUSION_STRENGTHS)
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
            self._pipe = ReSDPipeline.from_pretrained(
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
                "regen_diffusion", out, strength, self.model_id, self.device, diffusion_pipe=pipe
            )
        return out

    def apply_batch(self, images: list[Image.Image], strength: float | int) -> list[Image.Image]:
        out = images
        pipe = self._get_pipe()
        for _ in range(4):
            out = remove_watermark_batch(
                "regen_diffusion", out, strength, self.model_id, self.device, diffusion_pipe=pipe
            )
        return out


# ---------------------------------------------------------------------------
# Registry helper
# ---------------------------------------------------------------------------

_ATTACK_CLS: dict[str, type[Attack]] = {
    "Regen-Diff": DiffusionRegenAttack,
    "Regen-VAE": VAERegenAttack,
    "Rinse-2xDiff": Rinse2xDiffAttack,
    "Rinse-4xDiff": Rinse4xDiffAttack,
}


def build_regen_attack(
    name: str,
    *,
    diffusion_model_id: str = "CompVis/stable-diffusion-v1-4",
    vae_model_name: str = "bmshj2018-factorized",
    device: torch.device | None = None,
    shared_diff_pipe: object | None = None,
) -> Attack:
    """Instantiate one regeneration attack by display name."""
    dev = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    if name not in _ATTACK_CLS:
        raise ValueError(f"Unknown regen attack {name!r}; choose from {list(_ATTACK_CLS)}")

    if name == "Regen-VAE":
        return VAERegenAttack(device=dev, vae_model_name=vae_model_name)

    pipe_provider = (lambda: shared_diff_pipe) if shared_diff_pipe is not None else None
    return _ATTACK_CLS[name](diffusion_model_id=diffusion_model_id, device=dev, pipe_provider=pipe_provider)


def build_all_regen_attacks(
    *,
    diffusion_model_id: str = "CompVis/stable-diffusion-v1-4",
    vae_model_name: str = "bmshj2018-factorized",
    device: torch.device | None = None,
) -> dict[str, Attack]:
    """All four regen attacks sharing one SD pipeline (saves VRAM)."""
    dev = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    shared_pipe: object | None = None

    def get_shared_pipe():
        nonlocal shared_pipe
        if shared_pipe is None:
            shared_pipe = ReSDPipeline.from_pretrained(
                diffusion_model_id,
                torch_dtype=torch.float16 if dev.type == "cuda" else torch.float32,
                revision="fp16" if dev.type == "cuda" else None,
            )
            shared_pipe.set_progress_bar_config(disable=True)
            shared_pipe.to(dev)
        return shared_pipe

    return {
        name: build_regen_attack(
            name,
            diffusion_model_id=diffusion_model_id,
            vae_model_name=vae_model_name,
            device=dev,
            shared_diff_pipe=get_shared_pipe() if name != "Regen-VAE" else None,
        )
        for name in REGEN_ATTACK_NAMES
    }


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Apply a wmbench regeneration attack to one image.")
    parser.add_argument("image", help="Input image path")
    parser.add_argument("output", help="Output image path")
    parser.add_argument(
        "--attack",
        default="Regen-VAE",
        choices=list(REGEN_ATTACK_NAMES),
        help="Attack name (Regen-VAE is fastest; diffusion attacks need GPU + diffusers)",
    )
    parser.add_argument("--strength", type=int, default=None, help="Strength (attack-specific; see module docstring)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    atk = build_regen_attack(args.attack, device=torch.device(args.device))
    strength = args.strength if args.strength is not None else atk.strengths[len(atk.strengths) // 2]
    with Image.open(args.image) as im:
        out = atk.apply(im.convert("RGB"), strength)
    out.save(args.output)
    print(f"{args.attack} strength={strength} -> {args.output}")
