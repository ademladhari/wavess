"""Reproduce Table III (fixed-strength attacks), Fig. 5 (strength sweep),
and Fig. 6 (generative attacks).

Uses the pre-generated val pool (generate_pairs.py --split val), applies an
attack to each watermarked image, runs Dext -> Ddec, and reports BitAcc +
TPR@0.01FPR.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from src.attacks import generative as gen_atk
from src.attacks import image as img_atk
from src.data.pairs import PairsDataset
from src.eval.extraction import bit_accuracy, tpr_at_fpr
from src.models import WatermarkDecoder, WatermarkExtractor
from src.utils import load_config, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--bits", type=int, default=None)
    p.add_argument(
        "--mode",
        type=str,
        default="fixed",
        choices=["fixed", "swept", "generative"],
    )
    p.add_argument("--n", type=int, default=None, help="number of val samples to use")
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def _load_models(cfg, n_bits: int, device: torch.device):
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
    return ext, dec


@torch.no_grad()
def _run_extraction(ext, dec, images: torch.Tensor) -> torch.Tensor:
    z = ext(images)
    wd = dec(z)
    return wd


@torch.no_grad()
def _eval_on_attack(ext, dec, ds: PairsDataset, apply_fn, n: int, device: torch.device):
    wd_list, w_list = [], []
    pbar = tqdm(range(n), ncols=100)
    for i in pbar:
        item = ds[i]
        img = item["image"].unsqueeze(0).to(device)
        w = item["w"].unsqueeze(0)
        attacked = apply_fn(img)
        wd = _run_extraction(ext, dec, attacked.to(device)).cpu()
        wd_list.append(wd)
        w_list.append(w)
    wd_all = torch.cat(wd_list, dim=0)
    w_all = torch.cat(w_list, dim=0)
    acc = bit_accuracy(wd_all, w_all)
    tpr, tau = tpr_at_fpr(wd_all, w_all, fpr=0.01)
    return {"BitAcc": acc, "TPR@0.01FPR": tpr, "tau": tau}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.eval.eval_seed))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    n_bits = int(args.bits if args.bits is not None else cfg.watermark.primary_bits)

    val_root = Path(cfg.generate_pairs.out_dir) / f"b{n_bits}" / "val"
    ds = PairsDataset(val_root)
    n = int(args.n) if args.n else len(ds)

    ext, dec = _load_models(cfg, n_bits, device)

    out_dir = Path(args.out_dir or cfg.eval.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {"mode": args.mode, "bits": n_bits, "n": n}

    if args.mode == "fixed":
        # Clean baseline
        results["clean"] = _eval_on_attack(ext, dec, ds, lambda x: x, n, device)
        for name in img_atk.FIXED_ATTACK_REGISTRY:
            print(f"[attack/fixed] {name}")
            fn = img_atk.FIXED_ATTACK_REGISTRY[name]
            results[name] = _eval_on_attack(ext, dec, ds, fn, n, device)
        (out_dir / f"table3_fixed_b{n_bits}.json").write_text(json.dumps(results, indent=2))

    elif args.mode == "swept":
        swept = dict(cfg.attacks.swept)
        for name, values in swept.items():
            results[name] = {}
            for strength in values:
                print(f"[attack/swept] {name}={strength}")
                fn = (lambda x, n_=name, s_=strength: img_atk.apply_swept(x, n_, s_))
                results[name][str(strength)] = _eval_on_attack(ext, dec, ds, fn, n, device)
        (out_dir / f"fig5_swept_b{n_bits}.json").write_text(json.dumps(results, indent=2))

    elif args.mode == "generative":
        # bmshj18
        results["bmshj18"] = {}
        for q in cfg.attacks.generative.bmshj18_quality:
            print(f"[attack/gen] bmshj18 q={q}")
            fn = (lambda x, q_=q: gen_atk.bmshj18_attack(x, quality=int(q_)))
            results["bmshj18"][str(int(q))] = _eval_on_attack(ext, dec, ds, fn, n, device)
        # cheng20
        results["cheng20"] = {}
        for q in cfg.attacks.generative.cheng20_quality:
            print(f"[attack/gen] cheng20 q={q}")
            fn = (lambda x, q_=q: gen_atk.cheng20_attack(x, quality=int(q_)))
            results["cheng20"][str(int(q))] = _eval_on_attack(ext, dec, ds, fn, n, device)
        # zhao23
        results["zhao23"] = {}
        for steps in cfg.attacks.generative.zhao23_steps:
            print(f"[attack/gen] zhao23 steps={steps}")
            fn = (
                lambda x, s_=steps: gen_atk.zhao23_attack(
                    x,
                    num_denoise_steps=int(s_),
                    pretrained_id=str(cfg.sdm.pretrained_id),
                    cache_dir=str(cfg.paths.hf_cache),
                )
            )
            results["zhao23"][str(int(steps))] = _eval_on_attack(ext, dec, ds, fn, n, device)
        (out_dir / f"fig6_generative_b{n_bits}.json").write_text(json.dumps(results, indent=2))

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
