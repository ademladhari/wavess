#!/usr/bin/env python3
"""
Standalone flexible (SDM) benchmark using only flexible/src/*.

Generation-based: embeds via WatermarkEncoder + FrozenSDM (or loads pre-generated
pairs from generate_pairs.py). Detection is blind: WatermarkExtractor + WatermarkDecoder.

Metrics per attack:
  - PSNR, SSIM (watermarked vs attacked)
  - Bit accuracy
  - AUROC, TPR @ 1% FPR (score = mean bit accuracy vs payload)
  - TPR @ 0.01 FPR (paper bit-count threshold from src.eval.extraction)

Attacks: see benchmark_attacks.py (rotation, blur, brightness, JPEG, crop, etc.)

Example (generate on the fly — slow, ~seconds per image):
  python run_benchmark.py --n-images 10 --output outputs_benchmark

Example (fast, if pairs already exist):
  python run_benchmark.py --pairs-dir data/pairs/b48/val --n-images 100 --output outputs_benchmark
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sklearn import metrics
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent
_WAVES_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(_WAVES_ROOT) not in sys.path:
    sys.path.insert(0, str(_WAVES_ROOT))

from benchmark_attacks import ATTACKS, apply_attack_tensor  # noqa: E402
from src.data.pairs import PairsDataset  # noqa: E402
from src.data.prompts import load_prompts  # noqa: E402
from src.eval.extraction import bit_accuracy, tpr_at_fpr  # noqa: E402
from src.models import WatermarkDecoder, WatermarkEncoder, WatermarkExtractor  # noqa: E402
from src.models.sdm import FrozenSDM  # noqa: E402
from src.utils import load_config  # noqa: E402

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(root: Path, limit: int) -> list[Path]:
    paths = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in _IMG_EXTS)
    if len(paths) < limit:
        raise FileNotFoundError(f"Need {limit} images under {root}, found {len(paths)}")
    return paths[:limit]


def tensor_to_pil(img: torch.Tensor) -> Image.Image:
    arr = (img.clamp(0.0, 1.0).detach().cpu().numpy() * 255.0).round().astype(np.uint8)
    arr = np.transpose(arr, (1, 2, 0))
    return Image.fromarray(arr, mode="RGB")


def pil_to_tensor(pil: Image.Image) -> torch.Tensor:
    arr = np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(np.transpose(arr, (2, 0, 1)))


def load_clean_tensor(path: Path, size: int) -> torch.Tensor:
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        side = min(w, h)
        left, top = (w - side) // 2, (h - side) // 2
        im = im.crop((left, top, left + side, top + side))
        if im.size != (size, size):
            im = im.resize((size, size), Image.Resampling.LANCZOS)
        return pil_to_tensor(im)


def psnr_ssim_tensors(ref: torch.Tensor, cand: torch.Tensor) -> tuple[float, float]:
    a = tensor_to_pil(ref)
    b = tensor_to_pil(cand)
    if b.size != a.size:
        b = b.resize(a.size, Image.Resampling.BICUBIC)
    a_np = np.asarray(a, dtype=np.float64)
    b_np = np.asarray(b, dtype=np.float64)
    psnr = float(peak_signal_noise_ratio(a_np, b_np, data_range=255.0))
    ssim = float(structural_similarity(a_np, b_np, channel_axis=2, data_range=255.0))
    return psnr, ssim


def detection_auroc_and_tpr(
    pos: np.ndarray, neg: np.ndarray, fpr_target: float = 0.01
) -> tuple[float, float]:
    y_true = np.concatenate(
        [np.zeros(neg.size, dtype=np.int32), np.ones(pos.size, dtype=np.int32)]
    )
    y_score = np.concatenate([neg, pos])
    auroc = float(metrics.roc_auc_score(y_true, y_score))
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score, pos_label=1)
    below = np.where(fpr < fpr_target)[0]
    tpr_at = float(tpr[below[-1]]) if below.size else float(tpr[0])
    return auroc, tpr_at


def _load_extractor_decoder(cfg, n_bits: int, device: torch.device):
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
    ckpt_dir = ROOT / str(cfg.paths.checkpoints)
    ext.load_state_dict(torch.load(ckpt_dir / f"extractor_b{n_bits}.pt", map_location=device))
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
    dec_ft = ckpt_dir / f"decoder_ft_b{n_bits}.pt"
    dec_path = dec_ft if dec_ft.is_file() else ckpt_dir / f"decoder_b{n_bits}.pt"
    dec.load_state_dict(torch.load(dec_path, map_location=device))
    dec.eval()
    return ext, dec


@torch.no_grad()
def decode_watermark(ext, dec, image: torch.Tensor, device: torch.device) -> torch.Tensor:
    x = image.unsqueeze(0).to(device)
    z = ext(x)
    wd = dec(z)
    return wd[0].cpu()


def _score(ext, dec, image: torch.Tensor, w: torch.Tensor, device: torch.device) -> float:
    wd = decode_watermark(ext, dec, image, device).unsqueeze(0)
    w_batch = w.reshape(1, -1).float()
    return float(bit_accuracy(wd, w_batch))


def _load_records_from_pairs(pairs_dir: Path, n: int) -> list[dict]:
    ds = PairsDataset(pairs_dir)
    n = min(n, len(ds))
    records = []
    for i in range(n):
        item = ds[i]
        records.append(
            {
                "prompt": item["prompt"],
                "w": item["w"].float(),
                "image": item["image"].float(),
            }
        )
    return records


def _generate_records(
    cfg,
    *,
    n_images: int,
    n_bits: int,
    seed: int,
    device: torch.device,
    shared_payload: bool,
) -> list[dict]:
    ckpt_dir = ROOT / str(cfg.paths.checkpoints)
    enc_ckpt = ckpt_dir / f"encoder_b{n_bits}.pt"
    if not enc_ckpt.is_file():
        raise FileNotFoundError(f"Missing encoder checkpoint: {enc_ckpt}")

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

    hf_cache = str((ROOT / str(cfg.paths.hf_cache)).resolve())
    sdm_dtype = torch.float16 if str(cfg.sdm.dtype) == "float16" else torch.float32
    sdm = FrozenSDM(
        pretrained_id=str(cfg.sdm.pretrained_id),
        dtype=sdm_dtype,
        device=device,
        enable_attention_slicing=bool(cfg.sdm.attention_slicing),
        enable_vae_slicing=bool(cfg.sdm.vae_slicing),
        enable_xformers=bool(cfg.sdm.enable_xformers),
        cache_dir=hf_cache,
    )

    prompts = load_prompts(
        name=str(cfg.generate_pairs.dataset_name),
        split=str(cfg.generate_pairs.split),
        n=n_images,
        seed=int(cfg.generate_pairs.seed),
        cache_dir=hf_cache,
    )
    if len(prompts) < n_images:
        reps = (n_images + len(prompts) - 1) // len(prompts)
        prompts = (prompts * reps)[:n_images]

    gen = torch.Generator(device="cpu").manual_seed(seed)
    shared_w = (
        torch.randint(0, 2, (n_bits,), generator=gen).float()
        if shared_payload
        else None
    )

    steps = int(cfg.sdm.num_inference_steps)
    guidance = float(cfg.sdm.guidance_scale)
    records: list[dict] = []

    print(
        f"Generating {n_images} watermarked images via SDM "
        f"(n_bits={n_bits}, shared_payload={shared_payload})…",
        flush=True,
    )
    for i in tqdm(range(n_images), desc="embed/sdm", unit="img"):
        if shared_w is not None:
            w = shared_w.clone()
        else:
            w = torch.randint(0, 2, (n_bits,), generator=gen).float()

        with torch.no_grad():
            w_batch = w.unsqueeze(0).to(device)
            z_img, _, _ = enc(w_batch)
            z_img = z_img.to(sdm.dtype)
            out = sdm.generate(
                z_init=z_img,
                prompt=prompts[i],
                num_inference_steps=steps,
                guidance_scale=guidance,
            )
            image = out.image[0].float().cpu().clamp(0.0, 1.0)

        records.append({"prompt": prompts[i], "w": w, "image": image})

    del sdm
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return records


def run(
    output_dir: Path,
    *,
    n_images: int,
    n_bits: int,
    seed: int,
    device: torch.device,
    pairs_dir: Path | None,
    negatives_dir: Path | None,
    n_negatives: int,
    shared_payload: bool,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "configs" / "default.yaml")

    if pairs_dir is not None:
        records = _load_records_from_pairs(pairs_dir, n_images)
        print(f"Loaded {len(records)} pairs from {pairs_dir}", flush=True)
    else:
        records = _generate_records(
            cfg,
            n_images=n_images,
            n_bits=n_bits,
            seed=seed,
            device=device,
            shared_payload=shared_payload,
        )

    ext, dec = _load_extractor_decoder(cfg, n_bits, device)

    neg_scores: list[float] = []
    if negatives_dir is not None and negatives_dir.is_dir():
        neg_paths = list_images(negatives_dir, n_negatives)
        ref_w = records[0]["w"]
        for path in tqdm(neg_paths, desc="detect/negatives", unit="img"):
            clean = load_clean_tensor(path, int(cfg.sdm.image_size))
            neg_scores.append(_score(ext, dec, clean, ref_w, device))
    else:
        print("No --images dir: negatives = random noise tensors", flush=True)
        ref_w = records[0]["w"]
        rng = np.random.default_rng(seed + 1)
        for _ in tqdm(range(n_negatives), desc="detect/negatives", unit="img"):
            noise = torch.from_numpy(rng.random((3, int(cfg.sdm.image_size), int(cfg.sdm.image_size)), dtype=np.float32))
            neg_scores.append(_score(ext, dec, noise, ref_w, device))
    neg_arr = np.asarray(neg_scores, dtype=np.float64)

    rows_out: list[dict] = []
    for spec in ATTACKS:
        psnr_vals: list[float] = []
        ssim_vals: list[float] = []
        wd_list: list[torch.Tensor] = []
        w_list: list[torch.Tensor] = []
        pos_scores: list[float] = []

        for i, rec in enumerate(tqdm(records, desc=f"flexible/{spec.name}", unit="img")):
            wm = rec["image"]
            w = rec["w"]
            attacked = apply_attack_tensor(spec.name, wm, seed=i)
            p, s = psnr_ssim_tensors(wm, attacked)
            psnr_vals.append(p)
            ssim_vals.append(s)

            wd = decode_watermark(ext, dec, attacked, device)
            wd_list.append(wd.unsqueeze(0))
            w_list.append(w.reshape(1, -1))
            pos_scores.append(float(bit_accuracy(wd.unsqueeze(0), w.reshape(1, -1))))

        wd_all = torch.cat(wd_list, dim=0)
        w_all = torch.cat(w_list, dim=0)
        bit_acc = float(bit_accuracy(wd_all, w_all))
        tpr_paper, tau = tpr_at_fpr(wd_all, w_all, fpr=0.01)
        pos_arr = np.asarray(pos_scores, dtype=np.float64)
        auroc, tpr1 = detection_auroc_and_tpr(pos_arr, neg_arr)

        row = {
            "method": "flexible",
            "detector": "blind_extractor_decoder",
            "attack": spec.name,
            "description": spec.description,
            "n_images": len(records),
            "n_bits": n_bits,
            "PSNR": float(np.mean(psnr_vals)),
            "SSIM": float(np.mean(ssim_vals)),
            "bit_accuracy": bit_acc,
            "AUROC": auroc,
            "TPR_at_1pct_FPR": tpr1,
            "TPR_paper_0.01FPR": float(tpr_paper),
            "paper_tau_bits": int(tau),
        }
        rows_out.append(row)
        print(
            f"  {spec.name}: PSNR={row['PSNR']:.2f} SSIM={row['SSIM']:.3f} "
            f"BitAcc={row['bit_accuracy']:.3f} AUROC={row['AUROC']:.3f} "
            f"TPR@1%FPR={row['TPR_at_1pct_FPR']:.3f} "
            f"TPR_paper={row['TPR_paper_0.01FPR']:.3f}",
            flush=True,
        )

    csv_path = output_dir / "results.csv"
    fields = list(rows_out[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)

    summary = {
        "method": "flexible",
        "implementation": "flexible/src",
        "detector": "blind WatermarkExtractor + WatermarkDecoder",
        "n_images": len(records),
        "n_bits": n_bits,
        "seed": seed,
        "shared_payload": shared_payload,
        "pairs_dir": str(pairs_dir) if pairs_dir else None,
        "negatives_dir": str(negatives_dir) if negatives_dir else None,
        "attacks": rows_out,
    }
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote {csv_path}", flush=True)
    return rows_out


def main() -> int:
    p = argparse.ArgumentParser(description="Flexible SDM standalone benchmark")
    p.add_argument(
        "--images",
        type=Path,
        default=Path(r"D:\new method\data\coco100k\val"),
        help="clean images for detection negatives (unwatermarked)",
    )
    p.add_argument("--output", type=Path, default=Path("outputs_benchmark"))
    p.add_argument("--n-images", type=int, default=100)
    p.add_argument("--n-negatives", type=int, default=100)
    p.add_argument("--bits", type=int, default=None, help="payload bits (default from config)")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument(
        "--pairs-dir",
        type=Path,
        default=None,
        help="use pre-generated pairs (generate_pairs.py) instead of SDM embed",
    )
    p.add_argument(
        "--per-image-payload",
        action="store_true",
        help="random payload per image (default: one shared payload for all)",
    )
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    cfg = load_config(ROOT / "configs" / "default.yaml")
    n_bits = int(args.bits if args.bits is not None else cfg.watermark.primary_bits)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    run(
        args.output,
        n_images=args.n_images,
        n_bits=n_bits,
        seed=args.seed,
        device=device,
        pairs_dir=args.pairs_dir,
        negatives_dir=args.images if args.images.is_dir() else None,
        n_negatives=args.n_negatives,
        shared_payload=not args.per_image_payload,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
