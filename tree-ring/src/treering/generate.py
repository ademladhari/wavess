from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline

from .embed import apply_key_to_latents
from .keygen import KeyMaterial


DTYPE_LOOKUP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


@dataclass(slots=True)
class GenerationResult:
    images: list[Any]
    prompts: list[str]
    latents: torch.Tensor
    metadata: dict[str, Any]


class TreeRingGenerator:
    """Wrapper around Diffusers generation with optional Tree-Ring latent injection."""

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        dtype: str = "float16",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
    ) -> None:
        torch_dtype = DTYPE_LOOKUP[dtype]
        self.pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
        self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)
        self.pipeline = self.pipeline.to(device)
        self.device = device
        self.dtype = torch_dtype
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale

    @property
    def latent_shape(self) -> tuple[int, int]:
        sample_size = self.pipeline.unet.config.sample_size
        return (int(sample_size), int(sample_size))

    @property
    def latent_channels(self) -> int:
        return int(self.pipeline.unet.config.in_channels)

    def _initial_latents(
        self,
        batch_size: int,
        seed: int | None,
        key: KeyMaterial | None,
    ) -> torch.Tensor:
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        height, width = self.latent_shape
        latents = torch.randn(
            (batch_size, self.latent_channels, height, width),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
        if key is not None:
            latents = apply_key_to_latents(latents, key)
        return latents

    def generate(
        self,
        prompts: Sequence[str],
        key: KeyMaterial | None = None,
        seed: int | None = None,
    ) -> GenerationResult:
        prompts_list = list(prompts)
        latents = self._initial_latents(batch_size=len(prompts_list), seed=seed, key=key)
        output = self.pipeline(
            prompt=prompts_list,
            latents=latents,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            output_type="pil",
        )
        metadata = {
            "model_id": self.pipeline.config._name_or_path,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "seed": seed,
            "key_variant": key.variant if key is not None else None,
        }
        return GenerationResult(
            images=list(output.images),
            prompts=prompts_list,
            latents=latents.detach().cpu(),
            metadata=metadata,
        )
