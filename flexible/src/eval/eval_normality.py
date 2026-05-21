"""Reproduce Table II: latent-distribution preservation (Sec. IV-C).

Generate 5000 initial latent vectors zT embedded with a fixed 48-bit
watermark and run normality diagnostics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.eval.normality import evaluate_normality
from src.models import WatermarkEncoder
from src.utils import load_config, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--bits", type=int, default=None)
    p.add_argument("--n-samples", type=int, default=None)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.seed))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    n_bits = int(args.bits if args.bits is not None else cfg.normality.bits)
    n_samples = int(args.n_samples if args.n_samples is not None else cfg.normality.n_samples)

    enc = WatermarkEncoder(
        n_bits=n_bits,
        latent_channels=int(cfg.sdm.latent_channels),
        latent_size=int(cfg.sdm.latent_size),
        base_channels=int(cfg.arch.encoder.base_channels),
        conv_blocks=int(cfg.arch.encoder.conv_blocks),
        feature_hw=int(cfg.arch.encoder.latent_feature_hw),
        kernel_size=int(cfg.arch.encoder.kernel_size),
    ).to(device)
    ckpt = Path(cfg.paths.checkpoints) / f"encoder_b{n_bits}.pt"
    enc.load_state_dict(torch.load(ckpt, map_location=device))
    enc.eval()

    # Fixed watermark per Sec. IV-C.
    gen = torch.Generator(device="cpu").manual_seed(int(cfg.seed))
    w_fixed = torch.randint(0, 2, (1, n_bits), generator=gen).to(device=device, dtype=torch.float32)

    all_z = []
    batch = 32
    pbar = tqdm(range(0, n_samples, batch), desc=f"gen zT b{n_bits}", ncols=100)
    for i in pbar:
        b = min(batch, n_samples - i)
        w = w_fixed.expand(b, -1)
        with torch.no_grad():
            z, _, _ = enc(w)
        all_z.append(z.cpu().numpy())
    z_np = np.concatenate(all_z, axis=0)

    res = evaluate_normality(z_np)

    out_dir = Path(args.out_dir or cfg.eval.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "bits": n_bits,
        "n_samples": res.n_samples,
        "global_mean": res.global_mean,
        "global_std": res.global_std,
        "dagostino_pass_rate": res.dagostino_pass_rate,
        "ks_pass_rate": res.ks_pass_rate,
        "jarque_bera_pass_rate": res.jarque_bera_pass_rate,
        "mean_wasserstein": res.mean_wasserstein,
    }
    (out_dir / f"table2_normality_b{n_bits}.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
