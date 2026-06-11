#!/usr/bin/env python3
"""
Benchmark HiDDeN (ep177) and MoE R0 (ep20, soft router) using the shared
WAVES attack suite (benchmark_attacks.py).

Metrics per attack:
  - PSNR   (host RGB vs watermarked+attacked, median over images)
  - Bit accuracy  (30-bit payload)
  - AUROC   (detection; negatives = clean images decoded without embed)
  - TPR @ 1% FPR

Usage (from D:\\waves):
  D:\waves\.venv\Scripts\python.exe run_benchmark_hidden_moe.py \
      --images "D:\\new method\\data\\coco100k\\val" \
      --n-images 100 \
      --output outputs_benchmark

Both models are evaluated in a single pass; results are written to
outputs_benchmark/results_hidden.csv  and  outputs_benchmark/results_moe.csv
plus a combined outputs_benchmark/results_combined.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn import metrics as sk_metrics
from tqdm.auto import tqdm

# ── repo roots ────────────────────────────────────────────────────────────────
WAVES_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(r"D:\new method")

sys.path.insert(0, str(WAVES_ROOT))                      # benchmark_attacks
sys.path.insert(0, str(REPO_ROOT / "hidden"))            # HiDDeN utils
sys.path.insert(0, str(REPO_ROOT / "hidden_moe_unfrozen"))  # MoE utils
sys.path.insert(0, str(REPO_ROOT / "scripts"))           # compare_hidden_vs_moe_watermark

from benchmark_attacks import ATTACKS, apply_attack_rgb  # noqa: E402

# ── default paths ──────────────────────────────────────────────────────────
DEFAULT_HIDDEN_RUN = (
    REPO_ROOT / "hidden" / "runs" / "my_hidden_experiment 2026.05.17--21-47-43"
)
DEFAULT_HIDDEN_CKPT = (
    REPO_ROOT / "hiddenepcoh177" / "my_hidden_experiment--epoch-177.pyt"
)
DEFAULT_MOE_RUN = (
    REPO_ROOT
    / "results/experiments/unfrozen_moe/coco100k/4exp_k1/batch128"
    / "sym_t14_unfrozen_bal004warm10_ep20_2026-06-01"
)
DEFAULT_MOE_CKPT = DEFAULT_MOE_RUN / "checkpoints" / "moe_unfrozen_sym_t14_v1--epoch-20.pyt"

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
PATCH = 128  # HiDDeN / MoE fixed input size


# ── image helpers ─────────────────────────────────────────────────────────
def list_images(root: Path, n: int) -> list[Path]:
    paths = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in _IMG_EXTS)
    if len(paths) < n:
        raise FileNotFoundError(f"Need {n} images in {root}, found {len(paths)}")
    return paths[:n]


def load_rgb_patch(path: Path, size: int = PATCH) -> Image.Image:
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        s = min(w, h)
        l, t = (w - s) // 2, (h - s) // 2
        im = im.crop((l, t, l + s, t + s)).resize((size, size), Image.Resampling.LANCZOS)
        return im.copy()


def pil_to_tensor(im: Image.Image, device: torch.device) -> torch.Tensor:
    arr = np.asarray(im, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return t * 2.0 - 1.0   # normalise to [-1, 1]


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = ((t.squeeze(0).clamp(-1, 1).detach().cpu().permute(1, 2, 0).numpy() + 1.0) * 127.5).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def psnr_pil(orig: Image.Image, attacked: Image.Image) -> float:
    a = np.asarray(orig, dtype=np.float32)
    b = np.asarray(attacked.resize(orig.size, Image.Resampling.LANCZOS), dtype=np.float32)
    mse = float(np.mean((a - b) ** 2))
    if mse < 1e-10:
        return 100.0
    return float(10.0 * np.log10(255.0 ** 2 / mse))


# ── model loading (thin wrappers around compare_hidden_vs_moe_watermark) ──
def load_hidden(ckpt: Path, run: Path, device: torch.device):
    import compare_hidden_vs_moe_watermark as cmp
    model, cfg, _ = cmp.load_hidden_baseline(ckpt, run / "options-and-config.pickle", device)
    return model, cfg


def load_moe(ckpt: Path, run: Path, device: torch.device):
    import compare_hidden_vs_moe_watermark as cmp
    model, cfg = cmp.load_moe_model(run, ckpt, device, soft_router=True)
    return model, cfg


# ── encode / decode one patch ─────────────────────────────────────────────
@torch.no_grad()
def encode_hidden(model, img_t: torch.Tensor, msg: torch.Tensor) -> torch.Tensor:
    return model.encoder_decoder.encoder(img_t, msg)


@torch.no_grad()
def decode_hidden(model, wm_t: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(model.encoder_decoder.decoder(wm_t))


@torch.no_grad()
def encode_moe(model, img_t: torch.Tensor, msg: torch.Tensor) -> torch.Tensor:
    return model.encoder_decoder.encoder(img_t, msg)


@torch.no_grad()
def decode_moe(model, wm_t: torch.Tensor) -> torch.Tensor:
    logits, _, _, _ = model.encoder_decoder.decoder(wm_t)
    return torch.sigmoid(logits)


def bits_from_prob(prob: torch.Tensor) -> np.ndarray:
    return prob.detach().cpu().numpy().flatten().round().clip(0, 1).astype(np.uint8)


def bit_acc(ref: np.ndarray, pred: np.ndarray) -> float:
    n = min(ref.size, pred.size)
    return float(np.mean(ref[:n] == pred[:n]))


def detection_score(prob: torch.Tensor, ref_bits: np.ndarray) -> float:
    """Normalised correlation in [0, 1]; higher = more likely watermarked."""
    pred = prob.detach().cpu().numpy().flatten()
    ref = ref_bits[:pred.size].astype(np.float32)
    rho = float(np.mean((pred - 0.5) * (ref - 0.5)) * 4.0)   # scale to ~[-1, 1]
    return float(np.clip((rho + 1.0) * 0.5, 0.0, 1.0))


def auroc_tpr(pos: np.ndarray, neg: np.ndarray, fpr_target: float = 0.01):
    y_true = np.concatenate([np.zeros(neg.size, dtype=np.int32), np.ones(pos.size, dtype=np.int32)])
    y_score = np.concatenate([neg, pos])
    auroc = float(sk_metrics.roc_auc_score(y_true, y_score))
    fpr, tpr, _ = sk_metrics.roc_curve(y_true, y_score, pos_label=1)
    below = np.where(fpr <= fpr_target)[0]
    tpr1 = float(tpr[below[-1]]) if below.size else float(tpr[0])
    return auroc, tpr1


# ── per-method benchmark ──────────────────────────────────────────────────
def benchmark_model(
    name: str,
    encode_fn,
    decode_fn,
    model,
    images: list[Path],
    device: torch.device,
    msg_len: int,
) -> list[dict]:
    rng = np.random.default_rng(42)
    ref_bits = rng.integers(0, 2, msg_len, dtype=np.uint8)
    msg_t = torch.from_numpy(ref_bits.astype(np.float32)).unsqueeze(0).to(device)

    # --- embed once, keep host + watermarked PIL ---
    records: list[dict] = []
    print(f"\n[{name}] Embedding {len(images)} images …", flush=True)
    for path in tqdm(images, desc=f"{name}/embed", unit="img"):
        host_pil = load_rgb_patch(path)
        host_t = pil_to_tensor(host_pil, device)
        wm_t = encode_fn(model, host_t, msg_t)
        wm_pil = tensor_to_pil(wm_t)
        records.append({"path": path, "host_pil": host_pil, "wm_pil": wm_pil, "host_t": host_t})

    # --- negative detection scores (decode clean host, no embed) ---
    print(f"[{name}] Computing negative scores …", flush=True)
    neg_scores = np.asarray(
        [detection_score(decode_fn(model, r["host_t"]), ref_bits) for r in tqdm(records, desc=f"{name}/neg")],
        dtype=np.float64,
    )

    # --- per-attack metrics ---
    rows: list[dict] = []
    for spec in ATTACKS:
        psnr_vals, bit_vals, pos_scores = [], [], []
        for i, rec in enumerate(tqdm(records, desc=f"{name}/{spec.name}", unit="img", leave=False)):
            attacked_pil = apply_attack_rgb(spec.name, rec["wm_pil"], seed=i)
            psnr_vals.append(psnr_pil(rec["host_pil"], attacked_pil))
            # normalise to [-1,1] for decoder
            arr = np.asarray(attacked_pil, dtype=np.float32) / 255.0
            atk_t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
            atk_t = atk_t * 2.0 - 1.0
            prob = decode_fn(model, atk_t)
            bit_vals.append(bit_acc(ref_bits, bits_from_prob(prob)))
            pos_scores.append(detection_score(prob, ref_bits))

        pos_arr = np.asarray(pos_scores, dtype=np.float64)
        auroc, tpr1 = auroc_tpr(pos_arr, neg_scores)

        row = {
            "method": name,
            "attack": spec.name,
            "description": spec.description,
            "n_images": len(records),
            "PSNR_mean": float(np.mean(psnr_vals)),
            "PSNR_median": float(np.median(psnr_vals)),
            "bit_accuracy": float(np.mean(bit_vals)),
            "AUROC": auroc,
            "TPR_at_1pct_FPR": tpr1,
        }
        rows.append(row)
        print(
            f"  {spec.name:20s}: PSNR={row['PSNR_median']:.1f} "
            f"BitAcc={row['bit_accuracy']:.3f} "
            f"AUROC={row['AUROC']:.3f} "
            f"TPR@1%={row['TPR_at_1pct_FPR']:.3f}",
            flush=True,
        )
    return rows


# ── main ──────────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser(description="HiDDeN + MoE WAVES benchmark")
    p.add_argument("--images", type=Path, default=Path(r"D:\new method\data\coco100k\val"))
    p.add_argument("--n-images", type=int, default=100)
    p.add_argument("--output", type=Path, default=Path("outputs_benchmark"))
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--hidden-run", type=Path, default=DEFAULT_HIDDEN_RUN)
    p.add_argument("--hidden-ckpt", type=Path, default=DEFAULT_HIDDEN_CKPT)
    p.add_argument("--moe-run", type=Path, default=DEFAULT_MOE_RUN)
    p.add_argument("--moe-ckpt", type=Path, default=DEFAULT_MOE_CKPT)
    p.add_argument("--skip-hidden", action="store_true")
    p.add_argument("--skip-moe", action="store_true")
    args = p.parse_args()

    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    images = list_images(args.images, args.n_images)
    print(f"Device: {device}  |  Images: {len(images)}  |  Output: {args.output}")

    all_rows: list[dict] = []

    # ── HiDDeN ───────────────────────────────────────────────────────────
    if not args.skip_hidden:
        print("\n=== Loading HiDDeN ep177 ===")
        hidden_model, hidden_cfg = load_hidden(args.hidden_ckpt, args.hidden_run, device)
        msg_len = hidden_cfg.message_length
        rows_h = benchmark_model("HiDDeN-ep177", encode_hidden, decode_hidden, hidden_model, images, device, msg_len)
        all_rows.extend(rows_h)
        _write_csv(args.output / "results_hidden.csv", rows_h)
        del hidden_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── MoE ──────────────────────────────────────────────────────────────
    if not args.skip_moe:
        print("\n=== Loading MoE R0 ep20 (soft router) ===")
        moe_model, moe_cfg = load_moe(args.moe_ckpt, args.moe_run, device)
        msg_len = moe_cfg.message_length
        rows_m = benchmark_model("MoE-R0-soft", encode_moe, decode_moe, moe_model, images, device, msg_len)
        all_rows.extend(rows_m)
        _write_csv(args.output / "results_moe.csv", rows_m)
        del moe_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── combined CSV ─────────────────────────────────────────────────────
    if all_rows:
        _write_csv(args.output / "results_combined.csv", all_rows)

    # ── print summary table ───────────────────────────────────────────────
    key_attacks = ["identity", "jpeg_q50", "crop", "gaussian", "combined"]
    print(f"\n{'Attack':<22} {'Method':<18} {'PSNR':>6} {'BitAcc':>7} {'AUROC':>7} {'TPR@1%':>7}")
    print("-" * 72)
    for row in all_rows:
        if row["attack"] in key_attacks:
            print(
                f"{row['attack']:<22} {row['method']:<18} "
                f"{row['PSNR_median']:>6.1f} {row['bit_accuracy']:>7.3f} "
                f"{row['AUROC']:>7.3f} {row['TPR_at_1pct_FPR']:>7.3f}"
            )

    print(f"\nWrote results to {args.output}/")
    return 0


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  → {path}")


if __name__ == "__main__":
    raise SystemExit(main())
