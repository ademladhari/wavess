"""Stage 2 step 2: fine-tune the pretrained watermark Decoder D with Dext frozen.

Per Sec. III-D / Alg. 1 (step 2):

    z*_T = Dext(Iw)          # Dext frozen
    wd   = D(z*_T)           # D trainable
    L_dec = MSE(wd, w)       # eq. (6)

Hyperparameters (Sec. IV-A.4):
    Adam, lr=1e-5, batch_size=2.
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
from src.models import WatermarkDecoder, WatermarkExtractor
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
        batch_size=int(cfg.finetune_decoder.batch_size),
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
    ext_ckpt = Path(cfg.paths.checkpoints) / f"extractor_b{n_bits}.pt"
    ext.load_state_dict(torch.load(ext_ckpt, map_location=device))
    ext.eval()
    for p in ext.parameters():
        p.requires_grad_(False)

    dec = WatermarkDecoder(
        n_bits=n_bits,
        latent_channels=int(cfg.sdm.latent_channels),
        latent_size=int(cfg.sdm.latent_size),
        base_channels=int(cfg.arch.decoder.base_channels),
        tconv_blocks=int(cfg.arch.decoder.tconv_blocks),
        feature_hw=int(cfg.arch.encoder.latent_feature_hw),
        kernel_size=int(cfg.arch.decoder.kernel_size),
    ).to(device)
    dec_ckpt = Path(cfg.paths.checkpoints) / f"decoder_b{n_bits}.pt"
    dec.load_state_dict(torch.load(dec_ckpt, map_location=device))
    dec.train()

    opt = torch.optim.Adam(dec.parameters(), lr=float(cfg.finetune_decoder.lr))
    mse = nn.MSELoss()

    out_dir = Path(args.out_dir or cfg.paths.checkpoints)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg.paths.outputs) / f"finetune_decoder_b{n_bits}"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    max_steps = int(cfg.finetune_decoder.max_steps)
    log_every = int(cfg.finetune_decoder.log_every)
    save_every = int(cfg.finetune_decoder.save_every)

    pbar = tqdm(range(max_steps), desc=f"finetune D b{n_bits}", ncols=100)
    for step in pbar:
        batch = next(cycle)
        image = batch["image"].to(device, non_blocking=True)
        w = batch["w"].to(device, non_blocking=True)

        with torch.no_grad():
            z_star = ext(image)
        wd = dec(z_star)
        L_dec = mse(wd, w)

        opt.zero_grad(set_to_none=True)
        L_dec.backward()
        opt.step()

        if step % log_every == 0:
            with torch.no_grad():
                bit_acc = ((wd > 0.5).float() == w).float().mean().item()
            writer.add_scalar("loss/L_dec", L_dec.item(), step)
            writer.add_scalar("train/bit_acc", bit_acc, step)
            pbar.set_postfix(L_dec=f"{L_dec.item():.4g}", acc=f"{bit_acc:.3f}")

        if (step + 1) % save_every == 0:
            torch.save(dec.state_dict(), out_dir / f"decoder_ft_b{n_bits}.pt")

    torch.save(dec.state_dict(), out_dir / f"decoder_ft_b{n_bits}.pt")
    writer.close()
    print(f"[finetune_decoder] saved decoder_ft_b{n_bits}.pt to {out_dir}")


if __name__ == "__main__":
    main()
