"""Frozen Stable Diffusion 2.1 wrapper (Sec. IV-A.1).

Accepts a pre-computed initial latent ``zT`` (produced by the watermark
encoder E) and a text prompt, runs the DPMSolver-Multistep scheduler for 25
steps at guidance scale 7.5, and returns both the generated image tensor and
the denoised intermediate latent ``z_o`` (Fig. 2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline


@dataclass
class SDMOutput:
    image: torch.Tensor
    latents: torch.Tensor


class FrozenSDM:
    def __init__(
        self,
        pretrained_id: str = "stabilityai/stable-diffusion-2-1",
        dtype: torch.dtype = torch.float16,
        device: str | torch.device = "cuda",
        enable_attention_slicing: bool = True,
        enable_vae_slicing: bool = True,
        enable_xformers: bool = False,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype

        pipe = StableDiffusionPipeline.from_pretrained(
            pretrained_id,
            torch_dtype=dtype,
            cache_dir=cache_dir,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(self.device)
        if enable_attention_slicing:
            pipe.enable_attention_slicing()
        if enable_vae_slicing:
            pipe.enable_vae_slicing()
        if enable_xformers:
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception as e:
                print(f"[sdm] xformers unavailable ({e}); continuing without")

        for p in pipe.unet.parameters():
            p.requires_grad_(False)
        for p in pipe.vae.parameters():
            p.requires_grad_(False)
        for p in pipe.text_encoder.parameters():
            p.requires_grad_(False)
        pipe.unet.eval()
        pipe.vae.eval()
        pipe.text_encoder.eval()

        self.pipe = pipe

    @torch.no_grad()
    def _encode_prompt(self, prompt: str | list[str], batch_size: int) -> torch.Tensor:
        if isinstance(prompt, str):
            prompts = [prompt] * batch_size
        else:
            prompts = list(prompt)
            assert len(prompts) == batch_size
        tok = self.pipe.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        tok_ids = tok.input_ids.to(self.device)
        text_embeds = self.pipe.text_encoder(tok_ids)[0]

        uncond = self.pipe.tokenizer(
            [""] * batch_size,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        uncond_ids = uncond.input_ids.to(self.device)
        uncond_embeds = self.pipe.text_encoder(uncond_ids)[0]

        return torch.cat([uncond_embeds, text_embeds], dim=0)

    @torch.no_grad()
    def generate(
        self,
        z_init: torch.Tensor,
        prompt: str | list[str],
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
    ) -> SDMOutput:
        """Run DPMSolver denoising from ``z_init`` and decode to image.

        ``z_init`` is treated as the starting noise at the first scheduler
        timestep (Gaussian-like), matching Sec. III-D where zT produced by the
        watermark encoder is used directly as SDM's initial latent.
        """

        b = z_init.shape[0]
        scheduler = self.pipe.scheduler
        scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = scheduler.timesteps

        # Match scheduler's expected initial noise scaling.
        latents = z_init.to(device=self.device, dtype=self.dtype)
        latents = latents * scheduler.init_noise_sigma

        text_embeds = self._encode_prompt(prompt, batch_size=b)

        do_cfg = guidance_scale > 1.0
        for t in timesteps:
            latent_input = torch.cat([latents, latents], dim=0) if do_cfg else latents
            latent_input = scheduler.scale_model_input(latent_input, t)
            noise_pred = self.pipe.unet(
                latent_input,
                t,
                encoder_hidden_states=text_embeds,
            ).sample
            if do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # Decode to image space in [0, 1].
        decoded = self.pipe.vae.decode(
            latents / self.pipe.vae.config.scaling_factor
        ).sample
        image = (decoded / 2 + 0.5).clamp(0, 1)

        return SDMOutput(image=image, latents=latents)
