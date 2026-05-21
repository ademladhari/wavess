from __future__ import annotations

from typing import Sequence

import torch
from diffusers import DDIMInverseScheduler, DDIMScheduler, StableDiffusionPipeline
from PIL import Image
from torchvision.transforms import functional as TF


class DDIMInverter:
    """Invert generated images back to an estimated initial latent/noise."""

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        pipeline: StableDiffusionPipeline | None = None,
    ) -> None:
        if pipeline is None:
            self.pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)
        else:
            self.pipeline = pipeline.to(device)
        self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)
        self.inverse_scheduler = DDIMInverseScheduler.from_config(self.pipeline.scheduler.config)
        self.device = device
        self.dtype = dtype

    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        tensor = TF.to_tensor(image.convert("RGB")).unsqueeze(0).to(device=self.device, dtype=self.dtype)
        tensor = tensor * 2.0 - 1.0
        with torch.no_grad():
            latent_dist = self.pipeline.vae.encode(tensor).latent_dist
            # Deterministic encoding is crucial for stable inversion-based detection.
            latents = latent_dist.mode() * self.pipeline.vae.config.scaling_factor
        return latents

    def _encode_prompt(self, prompt: str, batch_size: int) -> torch.Tensor:
        text_inputs = self.pipeline.tokenizer(
            [prompt] * batch_size,
            padding="max_length",
            max_length=self.pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        with torch.no_grad():
            return self.pipeline.text_encoder(**text_inputs)[0]

    def invert(
        self,
        images: Sequence[Image.Image],
        invert_prompt: str = "",
        num_inversion_steps: int = 50,
    ) -> torch.Tensor:
        latents = torch.cat([self._encode_image(image) for image in images], dim=0)
        text_embeds = self._encode_prompt(invert_prompt, batch_size=latents.shape[0])

        self.inverse_scheduler.set_timesteps(num_inversion_steps, device=self.device)
        for timestep in self.inverse_scheduler.timesteps:
            with torch.no_grad():
                noise_pred = self.pipeline.unet(
                    latents,
                    timestep,
                    encoder_hidden_states=text_embeds,
                ).sample
                step_output = self.inverse_scheduler.step(noise_pred, timestep, latents)
            if hasattr(step_output, "prev_sample"):
                latents = step_output.prev_sample
            elif hasattr(step_output, "next_sample"):
                latents = step_output.next_sample
            else:
                raise RuntimeError("Unsupported DDIM inverse scheduler output format.")
        return latents
