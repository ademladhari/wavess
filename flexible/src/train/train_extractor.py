"""Stage 2 step 1: train the watermark extractor Dext.

Per Sec. III-D / Alg. 1 (step 1):

    L_ext = MSE(z'T, zT)                          # eq. (5)

Hyperparameters (Sec. III-D / Sec. IV-A.4):
    Adam, lr=1e-5, batch_size=2.

The pretrained encoder E is frozen; SDM G is frozen. For memory reasons the
inputs (watermarked images Iw and their latent targets zT) are loaded from
the pre-generated pool produced by generate_pairs.py.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.pairs import PairsDataset, collate_pairs
from src.models import WatermarkExtractor
from src.utils import load_config, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--bits", type=int, default=None)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.seed))

    n_bits = int(args.bits if args.bits is not None else cfg.watermark.primary_bits)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    pool_root = Path(cfg.generate_pairs.out_dir) / f"b{n_bits}" / "train"
    ds = PairsDataset(pool_root)
    loader = DataLoader(
        ds,
        batch_size=int(cfg.train_extractor.batch_size),
        shuffle=True,
        num_workers=2,
        collate_fn=collate_pairs,
        pin_memory=True,
        drop_last=True,
    )
    cycle = itertools.cycle(loader)

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

    opt = torch.optim.Adam(ext.parameters(), lr=float(cfg.train_extractor.lr))
    mse = nn.MSELoss()

    out_dir = Path(args.out_dir or cfg.paths.checkpoints)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg.paths.outputs) / f"train_extractor_b{n_bits}"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    max_steps = int(cfg.train_extractor.max_steps)
    log_every = int(cfg.train_extractor.log_every)
    save_every = int(cfg.train_extractor.save_every)

    ext.train()
    pbar = tqdm(range(max_steps), desc=f"train Dext b{n_bits}", ncols=100)
    for step in pbar:
        batch = next(cycle)
        image = batch["image"].to(device, non_blocking=True)
        target = batch["zT"].to(device, non_blocking=True)

        pred = ext(image)
        L_ext = mse(pred, target)

        opt.zero_grad(set_to_none=True)
        L_ext.backward()
        opt.step()

        if step % log_every == 0:
            writer.add_scalar("loss/L_ext", L_ext.item(), step)
            pbar.set_postfix(L_ext=f"{L_ext.item():.4g}")

        if (step + 1) % save_every == 0:
            torch.save(ext.state_dict(), out_dir / f"extractor_b{n_bits}.pt")

    torch.save(ext.state_dict(), out_dir / f"extractor_b{n_bits}.pt")
    writer.close()
    print(f"[train_extractor] saved extractor_b{n_bits}.pt to {out_dir}")


if __name__ == "__main__":
    main()
