"""Reproduce Table I: fidelity + extraction performance without attacks.

For each watermark capacity n in {16, 24, 48}:
  * Generate N=1000 watermarked and N=1000 non-watermarked images with
    matched prompts and SD seeds.
  * Compute FID (watermarked vs non-watermarked), NIQE, PIQE, CLIP-score.
  * Extract watermarks via Dext -> Ddec; compute BitAcc and TPR@0.01FPR.

SDM V2.1 + DPMSolver 25 steps + CFG 7.5 (paper Sec. IV-A / IV-B).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from src.data.prompts import load_prompts
from src.data.watermark import sample_watermarks
from src.eval.extraction import bit_accuracy, tpr_at_fpr
from src.eval.fidelity import (
    _save_images_to_dir,
    compute_clip_score,
    compute_fid,
    compute_niqe,
    compute_piqe,
)
from src.models import WatermarkDecoder, WatermarkEncoder, WatermarkExtractor
from src.models.sdm import FrozenSDM
from src.utils import load_config, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--bits", type=int, nargs="+", default=None, help="list of watermark capacities")
    p.add_argument("--n-prompts", type=int, default=None)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--keep-images", action="store_true", help="keep on-disk image dumps after FID")
    return p.parse_args()


def _make_models(cfg, n_bits: int, device: torch.device):
    enc = WatermarkEncoder(
        n_bits=n_bits,
        latent_channels=int(cfg.sdm.latent_channels),
        latent_size=int(cfg.sdm.latent_size),
        base_channels=int(cfg.arch.encoder.base_channels),
        conv_blocks=int(cfg.arch.encoder.conv_blocks),
        feature_hw=int(cfg.arch.encoder.latent_feature_hw),
        kernel_size=int(cfg.arch.encoder.kernel_size),
    ).to(device)
    enc.load_state_dict(
        torch.load(Path(cfg.paths.checkpoints) / f"encoder_b{n_bits}.pt", map_location=device)
    )
    enc.eval()

    ext = WatermarkExtractor(
        image_size=int(cfg.sdm.image_size),
        in_channels=3,
        latent_channels=int(cfg.sdm.latent_channels),
        latent_size=int(cfg.sdm.latent_size),
        in_downsample=int(cfg.arch.extractor.in_downsample),
        token_dim=int(cfg.arch.extractor.token_dim),
        transformer_layers=int(cfg.arch.extractor.transformer_layers),
        transformer_heads=int(cfg.arch.extractor.transformer_heads),
        mlp_hidden=int(cfg.arch.extractor.mlp_hidden),
        ffn_dim=int(cfg.arch.extractor.ffn_dim),
    ).to(device)
    ext.load_state_dict(
        torch.load(Path(cfg.paths.checkpoints) / f"extractor_b{n_bits}.pt", map_location=device)
    )
    ext.eval()

    dec = WatermarkDecoder(
        n_bits=n_bits,
        latent_channels=int(cfg.sdm.latent_channels),
        latent_size=int(cfg.sdm.latent_size),
        base_channels=int(cfg.arch.decoder.base_channels),
        tconv_blocks=int(cfg.arch.decoder.tconv_blocks),
        feature_hw=int(cfg.arch.encoder.latent_feature_hw),
        kernel_size=int(cfg.arch.decoder.kernel_size),
    ).to(device)
    ft_ckpt = Path(cfg.paths.checkpoints) / f"decoder_ft_b{n_bits}.pt"
    if not ft_ckpt.exists():
        ft_ckpt = Path(cfg.paths.checkpoints) / f"decoder_b{n_bits}.pt"
    dec.load_state_dict(torch.load(ft_ckpt, map_location=device))
    dec.eval()
    return enc, ext, dec


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.eval.eval_seed))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    bits_list = list(args.bits) if args.bits else list(cfg.watermark.bit_lengths)
    n_prompts = int(args.n_prompts if args.n_prompts is not None else cfg.eval.n_prompts)

    out_dir = Path(args.out_dir or cfg.eval.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sdm = FrozenSDM(
        pretrained_id=str(cfg.sdm.pretrained_id),
        dtype=torch.float16 if str(cfg.sdm.dtype) == "float16" else torch.float32,
        device=device,
        enable_attention_slicing=bool(cfg.sdm.attention_slicing),
        enable_vae_slicing=bool(cfg.sdm.vae_slicing),
        enable_xformers=bool(cfg.sdm.enable_xformers),
        cache_dir=str(cfg.paths.hf_cache),
    )

    prompts = load_prompts(
        name=str(cfg.eval.dataset_name),
        split=str(cfg.eval.split),
        n=n_prompts,
        seed=int(cfg.eval.eval_seed),
        cache_dir=str(cfg.paths.hf_cache),
    )

    latent_shape = (
        int(cfg.sdm.latent_channels),
        int(cfg.sdm.latent_size),
        int(cfg.sdm.latent_size),
    )

    all_results: dict = {}
    for n_bits in bits_list:
        enc, ext, dec = _make_models(cfg, n_bits, device)

        wm_dir = out_dir / f"images_wm_b{n_bits}"
        nowm_dir = out_dir / f"images_nowm_b{n_bits}"
        wm_dir.mkdir(parents=True, exist_ok=True)
        nowm_dir.mkdir(parents=True, exist_ok=True)

        wd_list: list[torch.Tensor] = []
        w_list: list[torch.Tensor] = []
        piqe_images: list[torch.Tensor] = []
        niqe_images: list[torch.Tensor] = []
        clip_images: list[torch.Tensor] = []
        clip_prompts: list[str] = []

        gen = torch.Generator(device="cpu").manual_seed(int(cfg.eval.eval_seed) + n_bits)
        pbar = tqdm(range(n_prompts), desc=f"eval b{n_bits}", ncols=100)
        for i in pbar:
            w = torch.randint(0, 2, (1, n_bits), generator=gen).to(device=device, dtype=torch.float32)

            with torch.no_grad():
                z_img, _, _ = enc(w)

                # Non-watermarked: fresh Gaussian noise (same seed).
                z_nowm = torch.randn((1, *latent_shape), generator=gen).to(device)

                out_wm = sdm.generate(
                    z_init=z_img.to(sdm.dtype),
                    prompt=prompts[i],
                    num_inference_steps=int(cfg.sdm.num_inference_steps),
                    guidance_scale=float(cfg.sdm.guidance_scale),
                )
                out_nowm = sdm.generate(
                    z_init=z_nowm.to(sdm.dtype),
                    prompt=prompts[i],
                    num_inference_steps=int(cfg.sdm.num_inference_steps),
                    guidance_scale=float(cfg.sdm.guidance_scale),
                )
                img_wm = out_wm.image[0].float().cpu()
                img_nowm = out_nowm.image[0].float().cpu()

                z_rec = ext(img_wm.unsqueeze(0).to(device))
                wd = dec(z_rec).cpu()

            _save_images_to_dir([img_wm], wm_dir / f"tmp_{i:06d}")
            _save_images_to_dir([img_nowm], nowm_dir / f"tmp_{i:06d}")

            wd_list.append(wd)
            w_list.append(w.cpu())
            piqe_images.append(img_wm)
            niqe_images.append(img_wm)
            clip_images.append(img_wm)
            clip_prompts.append(prompts[i])

        # Flatten the per-call directories into a single directory (clean-fid
        # expects a flat directory).
        _flatten_dir(wm_dir)
        _flatten_dir(nowm_dir)

        fid = compute_fid(nowm_dir, wm_dir, device=device)
        niqe = compute_niqe(niqe_images, device=device)
        piqe = compute_piqe(piqe_images, device=device)
        clip_score = compute_clip_score(clip_images, clip_prompts, device=device)

        wd_all = torch.cat(wd_list, dim=0)
        w_all = torch.cat(w_list, dim=0)
        bit_acc = bit_accuracy(wd_all, w_all)
        tpr, tau = tpr_at_fpr(wd_all, w_all, fpr=0.01)

        result = {
            "bits": n_bits,
            "n_prompts": n_prompts,
            "FID": fid,
            "NIQE": niqe,
            "PIQE": piqe,
            "CLIP": clip_score,
            "BitAcc": bit_acc,
            "TPR@0.01FPR": tpr,
            "tau": tau,
        }
        print(json.dumps(result, indent=2))
        all_results[n_bits] = result

        if not args.keep_images:
            _rm_tree(wm_dir)
            _rm_tree(nowm_dir)

    out_path = out_dir / "table1_fidelity.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"[eval] wrote {out_path}")


def _flatten_dir(root: Path) -> None:
    """Move all PNGs from subdirectories up into ``root`` with unique names."""

    idx = 0
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        for f in sorted(sub.glob("*.png")):
            f.rename(root / f"{idx:06d}.png")
            idx += 1
        sub.rmdir()


def _rm_tree(root: Path) -> None:
    if not root.exists():
        return
    for p in sorted(root.rglob("*"), reverse=True):
        if p.is_file():
            p.unlink()
        elif p.is_dir():
            p.rmdir()
    root.rmdir()


if __name__ == "__main__":
    main()
