from __future__ import annotations

import torch
from transformers import CLIPModel, CLIPProcessor

from .aesthetics_scorer import load_model, preprocess

# Tune this to your GPU VRAM. 8 is safe for 8GB, 16 for 24GB.
_CHUNK_SIZE = 8


def load_aesthetics_and_artifacts_models(device: torch.device | None = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Windows CUDA can crash in safetensors load path for this model.
    # Prefer PyTorch bin weights here for stability while staying on GPU.
    model = CLIPModel.from_pretrained(
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        use_safetensors=False,
    )
    vision_model = model.vision_model
    # FP16 vision on CUDA only — CPU stays FP32 (correctness + avoids unsupported slow paths).
    if device.type == "cuda":
        vision_model.to(device=device, dtype=torch.float16)
    else:
        vision_model.to(device=device)
    vision_model.eval()
    del model

    clip_processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

    # .eval() makes dropout and batchnorm deterministic
    rating_model = load_model("aesthetics_scorer_rating_openclip_vit_h_14").to(device).eval()
    artifacts_model = load_model("aesthetics_scorer_artifacts_openclip_vit_h_14").to(device).eval()

    return vision_model, clip_processor, rating_model, artifacts_model


def compute_aesthetics_and_artifacts_scores(
    images,
    models,
    device: torch.device | None = None,
    chunk_size: int = _CHUNK_SIZE,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vision_model, clip_processor, rating_model, artifacts_model = models

    if len(images) == 0:
        return [], []

    use_fp16_clip = device.type == "cuda" and next(vision_model.parameters()).dtype == torch.float16

    # --- chunked CLIP encoding ---
    # avoids one giant forward pass that can OOM or thrash on large image sets
    pooled_chunks: list[torch.Tensor] = []

    with torch.inference_mode():
        for i in range(0, len(images), chunk_size):
            chunk = images[i : i + chunk_size]
            inputs = clip_processor(
                images=chunk, return_tensors="pt"
            ).to(device)
            if use_fp16_clip:
                inputs["pixel_values"] = inputs["pixel_values"].half()
            vision_output = vision_model(**inputs)
            pooled_chunks.append(vision_output.pooler_output)

        # --- scorer ---
        pooled = torch.cat(pooled_chunks, dim=0)  # (N, embed_dim)

        # scorers are fp32 MLPs — cast back from fp16 CLIP output
        embedding = preprocess(pooled.float())

        rating   = rating_model(embedding).squeeze(-1)
        artifact = artifacts_model(embedding).squeeze(-1)

    # inference_mode tensors have no grad, .detach() not needed
    return rating.cpu().tolist(), artifact.cpu().tolist()