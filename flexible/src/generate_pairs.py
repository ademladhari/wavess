"""Pre-generate (w, zT, Iw, prompt) tuples for Stage 2 training.

Sec. III-D / Alg. 1:
    For each training step the paper draws a random watermark w, a prompt t,
    computes zT = E(w), runs SDM G to get Iw, and updates Dext / D.

On an 8 GB RTX 5060 we cannot fit the full SD 2.1 pipeline AND the extractor
gradients simultaneously with batch size 2. So we precompute a large pool of
tuples once (SDM forward-only, fp16 + attention slicing) and train Dext / D
offline without SDM loaded. All paper hyperparameters remain unchanged; only
the data-loading pattern differs from "generate on the fly".
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from src.data.prompts import load_prompts
from src.models import WatermarkEncoder
from src.models.sdm import FrozenSDM
from src.utils import load_config, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--bits", type=int, default=None)
    p.add_argument("--split", type=str, default="train", choices=["train", "val"])
    p.add_argument("--n", type=int, default=None, help="override number of pairs to generate")
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--start", type=int, default=0, help="resume from this index")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def _image_to_uint8(image_01: torch.Tensor) -> torch.Tensor:
    x = (image_01.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu()
    return x


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.generate_pairs.seed))

    n_bits = int(args.bits if args.bits is not None else cfg.watermark.primary_bits)
    split = args.split
    n_default = int(cfg.generate_pairs.n_train_pairs if split == "train" else cfg.generate_pairs.n_val_pairs)
    n_pairs = int(args.n if args.n is not None else n_default)

    out_root = Path(args.out_dir or cfg.generate_pairs.out_dir)
    out_dir = out_root / f"b{n_bits}" / split
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    enc_ckpt = Path(cfg.paths.checkpoints) / f"encoder_b{n_bits}.pt"
    if not enc_ckpt.exists():
        raise FileNotFoundError(f"Missing pretrained encoder at {enc_ckpt}. Run pretrain_ed first.")

    enc = WatermarkEncoder(
        n_bits=n_bits,
        latent_channels=int(cfg.sdm.latent_channels),
        latent_size=int(cfg.sdm.latent_size),
        base_channels=int(cfg.arch.encoder.base_channels),
        conv_blocks=int(cfg.arch.encoder.conv_blocks),
        feature_hw=int(cfg.arch.encoder.latent_feature_hw),
        kernel_size=int(cfg.arch.encoder.kernel_size),
    ).to(device)
    enc.load_state_dict(torch.load(enc_ckpt, map_location=device))
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)

    sdm = FrozenSDM(
        pretrained_id=str(cfg.sdm.pretrained_id),
        dtype=torch.float16 if str(cfg.sdm.dtype) == "float16" else torch.float32,
        device=device,
        enable_attention_slicing=bool(cfg.sdm.attention_slicing),
        enable_vae_slicing=bool(cfg.sdm.vae_slicing),
        enable_xformers=bool(cfg.sdm.enable_xformers),
        cache_dir=str(cfg.paths.hf_cache),
    )

    prompt_seed = int(cfg.generate_pairs.seed) + (0 if split == "train" else 10_000)
    prompts = load_prompts(
        name=str(cfg.generate_pairs.dataset_name),
        split=str(cfg.generate_pairs.split),
        n=n_pairs,
        seed=prompt_seed,
        cache_dir=str(cfg.paths.hf_cache),
    )
    if len(prompts) < n_pairs:
        # Repeat prompts cyclically if dataset is smaller than requested.
        reps = (n_pairs + len(prompts) - 1) // len(prompts)
        prompts = (prompts * reps)[:n_pairs]

    steps = int(cfg.sdm.num_inference_steps)
    cfg_scale = float(cfg.sdm.guidance_scale)

    gen = torch.Generator(device="cpu").manual_seed(int(cfg.generate_pairs.seed))

    pbar = tqdm(range(args.start, n_pairs), desc=f"gen-pairs {split} b{n_bits}", ncols=100)
    for i in pbar:
        out_path = out_dir / f"pair_{i:06d}.pt"
        if out_path.exists():
            continue

        # Bit sequence is reproducible per-index, independent of GPU state.
        w = torch.randint(0, 2, (1, n_bits), generator=gen).to(device=device, dtype=torch.float32)

        with torch.no_grad():
            z_img, _, _ = enc(w)
            z_img = z_img.to(sdm.dtype)
            out = sdm.generate(
                z_init=z_img,
                prompt=prompts[i],
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
            )
            image = out.image[0].float().cpu()

        torch.save(
            {
                "prompt": prompts[i],
                "w": w[0].to(torch.uint8).cpu(),
                "zT": z_img[0].to(torch.float16).cpu(),
                "image_uint8": _image_to_uint8(image),
                "n_bits": n_bits,
            },
            out_path,
        )

    print(f"[generate_pairs] wrote {n_pairs - args.start} pairs to {out_dir}")


if __name__ == "__main__":
    main()
