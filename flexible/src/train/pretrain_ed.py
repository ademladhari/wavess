"""Stage 1: joint pretraining of watermark Encoder E and Decoder D.

Per Sec. III-C / Alg. 1 of Luo et al.:

    L_w = lambda1 * L_d + lambda2 * L_r
    L_d = KL( N(mu, sigma^2) || N(0, 1) )        # eq. (2)
    L_r = MSE(wo, w)                              # eq. (3)

Hyperparameters (Sec. IV-A.4):
    Adam, lr=1e-5, batch_size=2, lambda1=lambda2=1.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.watermark import sample_watermarks
from src.models import WatermarkDecoder, WatermarkEncoder
from src.utils import load_config, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--bits", type=int, default=None, help="Override watermark.primary_bits")
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.seed))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    n_bits = int(args.bits if args.bits is not None else cfg.watermark.primary_bits)

    out_dir = Path(args.out_dir or cfg.paths.checkpoints)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg.paths.outputs) / f"pretrain_ed_b{n_bits}"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    enc = WatermarkEncoder(
        n_bits=n_bits,
        latent_channels=int(cfg.sdm.latent_channels),
        latent_size=int(cfg.sdm.latent_size),
        base_channels=int(cfg.arch.encoder.base_channels),
        conv_blocks=int(cfg.arch.encoder.conv_blocks),
        feature_hw=int(cfg.arch.encoder.latent_feature_hw),
        kernel_size=int(cfg.arch.encoder.kernel_size),
    ).to(device)

    dec = WatermarkDecoder(
        n_bits=n_bits,
        latent_channels=int(cfg.sdm.latent_channels),
        latent_size=int(cfg.sdm.latent_size),
        base_channels=int(cfg.arch.decoder.base_channels),
        tconv_blocks=int(cfg.arch.decoder.tconv_blocks),
        feature_hw=int(cfg.arch.encoder.latent_feature_hw),
        kernel_size=int(cfg.arch.decoder.kernel_size),
    ).to(device)

    params = list(enc.parameters()) + list(dec.parameters())
    opt = torch.optim.Adam(params, lr=float(cfg.pretrain.lr))

    mse = nn.MSELoss()
    lambda1 = float(cfg.pretrain.lambda1)
    lambda2 = float(cfg.pretrain.lambda2)
    batch_size = int(cfg.pretrain.batch_size)
    max_steps = int(cfg.pretrain.max_steps)
    log_every = int(cfg.pretrain.log_every)
    save_every = int(cfg.pretrain.save_every)
    target_lr = float(cfg.pretrain.target_L_r)
    target_ld = float(cfg.pretrain.target_L_d)

    enc.train()
    dec.train()

    running_lr = running_ld = 0.0
    pbar = tqdm(range(max_steps), desc=f"pretrain E+D ({n_bits}-bit)", ncols=100)
    for step in pbar:
        w = sample_watermarks(batch_size, n_bits, device=device)
        z, mu, logvar = enc(w)
        wo = dec(z)

        L_d = WatermarkEncoder.kl_loss(mu, logvar)
        L_r = mse(wo, w)
        L_w = lambda1 * L_d + lambda2 * L_r

        opt.zero_grad(set_to_none=True)
        L_w.backward()
        opt.step()

        running_lr = 0.95 * running_lr + 0.05 * L_r.item() if step > 0 else L_r.item()
        running_ld = 0.95 * running_ld + 0.05 * L_d.item() if step > 0 else L_d.item()

        if step % log_every == 0:
            writer.add_scalar("loss/L_r", L_r.item(), step)
            writer.add_scalar("loss/L_d", L_d.item(), step)
            writer.add_scalar("loss/L_w", L_w.item(), step)
            writer.add_scalar("loss/L_r_ema", running_lr, step)
            writer.add_scalar("loss/L_d_ema", running_ld, step)
            with torch.no_grad():
                bit_acc = ((wo > 0.5).float() == w).float().mean().item()
            writer.add_scalar("train/bit_acc", bit_acc, step)
            pbar.set_postfix(Lr=f"{running_lr:.4g}", Ld=f"{running_ld:.4g}", acc=f"{bit_acc:.3f}")

        if (step + 1) % save_every == 0:
            torch.save(enc.state_dict(), out_dir / f"encoder_b{n_bits}.pt")
            torch.save(dec.state_dict(), out_dir / f"decoder_b{n_bits}.pt")

        if running_lr < target_lr and running_ld < target_ld and step > 5000:
            break

    torch.save(enc.state_dict(), out_dir / f"encoder_b{n_bits}.pt")
    torch.save(dec.state_dict(), out_dir / f"decoder_b{n_bits}.pt")
    writer.close()
    print(f"[pretrain] saved encoder_b{n_bits}.pt / decoder_b{n_bits}.pt to {out_dir}")


if __name__ == "__main__":
    main()
