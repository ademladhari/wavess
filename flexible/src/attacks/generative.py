"""Generative attacks (Sec. IV-D / Fig. 6).

Three regeneration-based attacks are applied:
  * Bmshj18 [46] - CompressAI factorized-prior VAE codec
  * Cheng20 [47] - CompressAI Cheng2020-anchor codec
  * Zhao23 [48] - diffusion-based regeneration (SD 2.1 image-to-image)

All functions take / return a 4-D tensor (B, 3, H, W) in [0, 1].
"""

from __future__ import annotations

from typing import Optional

import torch


# --- CompressAI VAE-based attacks -------------------------------------------


_CAI_MODELS: dict[tuple[str, int], "torch.nn.Module"] = {}


def _get_compressai_model(arch: str, quality: int, device: torch.device) -> "torch.nn.Module":
    key = (arch, quality)
    if key in _CAI_MODELS:
        return _CAI_MODELS[key].to(device)
    from compressai.zoo import bmshj2018_factorized, cheng2020_anchor

    if arch == "bmshj18":
        m = bmshj2018_factorized(quality=quality, pretrained=True)
    elif arch == "cheng20":
        m = cheng2020_anchor(quality=quality, pretrained=True)
    else:
        raise KeyError(arch)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    _CAI_MODELS[key] = m
    return m.to(device)


@torch.no_grad()
def bmshj18_attack(img: torch.Tensor, quality: int = 3) -> torch.Tensor:
    """CompressAI Ballé/Minnen/Hwang/Johnston 2018 factorized-prior codec."""

    device = img.device
    model = _get_compressai_model("bmshj18", int(quality), device)
    out = model(img.to(device).clamp(0.0, 1.0))
    return out["x_hat"].clamp(0.0, 1.0)


@torch.no_grad()
def cheng20_attack(img: torch.Tensor, quality: int = 3) -> torch.Tensor:
    """CompressAI Cheng2020-anchor codec."""

    device = img.device
    model = _get_compressai_model("cheng20", int(quality), device)
    out = model(img.to(device).clamp(0.0, 1.0))
    return out["x_hat"].clamp(0.0, 1.0)


# --- Zhao23: diffusion-based regeneration attack ----------------------------


_ZHAO_PIPE = None


def _get_zhao_pipe(
    pretrained_id: str = "stabilityai/stable-diffusion-2-1",
    dtype: torch.dtype = torch.float16,
    device: Optional[torch.device] = None,
    cache_dir: Optional[str] = None,
):
    global _ZHAO_PIPE
    if _ZHAO_PIPE is not None:
        return _ZHAO_PIPE
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        pretrained_id,
        torch_dtype=dtype,
        cache_dir=cache_dir,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    if device is not None:
        pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    for p in pipe.unet.parameters():
        p.requires_grad_(False)
    for p in pipe.vae.parameters():
        p.requires_grad_(False)
    for p in pipe.text_encoder.parameters():
        p.requires_grad_(False)
    _ZHAO_PIPE = pipe
    return pipe


@torch.no_grad()
def zhao23_attack(
    img: torch.Tensor,
    num_denoise_steps: int = 60,
    strength: float = 0.2,
    pretrained_id: str = "stabilityai/stable-diffusion-2-1",
    cache_dir: Optional[str] = None,
) -> torch.Tensor:
    """Diffusion-based regeneration attack.

    Implements Zhao et al. 2023's idea of adding noise to a clean image and
    running a diffusion model to reconstruct it, which tends to remove
    invisible watermarks. ``num_denoise_steps`` follows the paper's reporting
    (x-axis of Fig. 6c).
    """

    from PIL import Image
    import numpy as np

    device = img.device
    pipe = _get_zhao_pipe(pretrained_id=pretrained_id, device=device, cache_dir=cache_dir)

    # Img2Img pipeline expects PIL list; convert.
    out = []
    for x in img.clamp(0.0, 1.0):
        arr = (x.detach().cpu().numpy() * 255.0).round().astype(np.uint8)
        pil = Image.fromarray(np.transpose(arr, (1, 2, 0)))
        result = pipe(
            prompt="",
            image=pil,
            strength=float(strength),
            num_inference_steps=int(num_denoise_steps),
            guidance_scale=1.0,
            output_type="np",
        ).images[0]
        out.append(torch.from_numpy(result.transpose(2, 0, 1)).float().clamp(0.0, 1.0))
    return torch.stack(out, dim=0).to(device)
