"""Reproduce the Table V subset: extractor param count + extraction time
+ BitAcc for Transformer+MLP (ours) vs ResNet18.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from src.data.pairs import PairsDataset
from src.eval.extraction import bit_accuracy, tpr_at_fpr
from src.models import WatermarkDecoder, WatermarkExtractor
from src.models.extractor_resnet18 import ResNet18Extractor
from src.utils import load_config, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--bits", type=int, default=None)
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def _param_count(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.eval.eval_seed))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    n_bits = int(args.bits if args.bits is not None else cfg.watermark.primary_bits)

    val_root = Path(cfg.generate_pairs.out_dir) / f"b{n_bits}" / "val"
    ds = PairsDataset(val_root)
    n = int(args.n) if args.n else len(ds)

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

    ours = WatermarkExtractor(
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
    ours.load_state_dict(
        torch.load(Path(cfg.paths.checkpoints) / f"extractor_b{n_bits}.pt", map_location=device)
    )
    ours.eval()

    resnet = ResNet18Extractor(
        latent_channels=int(cfg.sdm.latent_channels),
        latent_size=int(cfg.sdm.latent_size),
    ).to(device)
    r_ckpt = Path(cfg.paths.checkpoints) / f"extractor_resnet18_b{n_bits}.pt"
    if r_ckpt.exists():
        resnet.load_state_dict(torch.load(r_ckpt, map_location=device))
    resnet.eval()

    results = {
        "bits": n_bits,
        "n": n,
        "ours_params": _param_count(ours),
        "resnet18_params": _param_count(resnet),
    }

    @torch.no_grad()
    def _time_and_score(extractor, name: str) -> dict:
        wd_list, w_list = [], []
        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.time()
        for i in tqdm(range(n), ncols=100, desc=name):
            item = ds[i]
            img = item["image"].unsqueeze(0).to(device)
            z = extractor(img)
            wd = dec(z).cpu()
            wd_list.append(wd)
            w_list.append(item["w"].unsqueeze(0))
        torch.cuda.synchronize() if device.type == "cuda" else None
        t1 = time.time()
        wd_all = torch.cat(wd_list, dim=0)
        w_all = torch.cat(w_list, dim=0)
        return {
            "extraction_time_s": (t1 - t0),
            "per_image_ms": 1000.0 * (t1 - t0) / max(n, 1),
            "BitAcc": bit_accuracy(wd_all, w_all),
            "TPR@0.01FPR": tpr_at_fpr(wd_all, w_all, 0.01)[0],
        }

    results["ours"] = _time_and_score(ours, "ours")
    results["resnet18"] = _time_and_score(resnet, "resnet18")

    out_dir = Path(cfg.eval.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"table5_extractor_compare_b{n_bits}.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
