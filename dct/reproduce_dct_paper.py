from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter
from scipy.fftpack import dct, idct
from scipy.ndimage import gaussian_filter


def dct2(x: np.ndarray) -> np.ndarray:
    return dct(dct(x.T, norm="ortho").T, norm="ortho")


def idct2(x: np.ndarray) -> np.ndarray:
    return idct(idct(x.T, norm="ortho").T, norm="ortho")


def to_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def load_grayscale_image(path: Path | None, dataset_dir: Path) -> tuple[np.ndarray, str]:
    if path is not None:
        img = Image.open(path).convert("L")
        return np.array(img, dtype=np.float64), str(path)

    candidates: list[Path] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
        candidates.extend(sorted(dataset_dir.glob(ext)))
    if candidates:
        img = Image.open(candidates[0]).convert("L")
        return np.array(img, dtype=np.float64), str(candidates[0])

    # Fallback synthetic image so pipeline always runs end-to-end.
    h = w = 512
    yy, xx = np.mgrid[0:h, 0:w]
    synthetic = (
        100
        + 60 * np.sin(2 * np.pi * xx / 28.0)
        + 40 * np.cos(2 * np.pi * yy / 42.0)
        + 0.15 * (xx + yy)
    )
    return np.clip(synthetic, 0, 255).astype(np.float64), "synthetic_fallback"


def top_magnitude_indices(coeffs: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    flat = np.abs(coeffs).ravel()
    dc_idx = 0
    flat[dc_idx] = -np.inf
    n = min(n, flat.size - 1)
    idx = np.argpartition(flat, -n)[-n:]
    idx = idx[np.argsort(flat[idx])[::-1]]
    rows, cols = np.unravel_index(idx, coeffs.shape)
    return rows, cols


def gaussian_watermark(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(loc=0.0, scale=1.0, size=n)


@dataclass
class WatermarkedImage:
    image: np.ndarray
    watermark: np.ndarray
    rows: np.ndarray
    cols: np.ndarray


def embed_watermark(
    image: np.ndarray,
    n: int,
    alpha: float,
    rng: np.random.Generator,
    watermark: np.ndarray | None = None,
) -> WatermarkedImage:
    coeffs = dct2(image)
    rows, cols = top_magnitude_indices(coeffs, n)
    mark = gaussian_watermark(len(rows), rng) if watermark is None else watermark

    coeffs_marked = coeffs.copy()
    coeffs_marked[rows, cols] = coeffs[rows, cols] * (1.0 + alpha * mark)
    wm_image = to_uint8(idct2(coeffs_marked)).astype(np.float64)

    return WatermarkedImage(image=wm_image, watermark=mark, rows=rows, cols=cols)


def extract_watermark(
    original_image: np.ndarray,
    candidate_image: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    alpha: float,
    eps: float = 1e-10,
) -> np.ndarray:
    c0 = dct2(original_image)[rows, cols]
    c1 = dct2(candidate_image)[rows, cols]
    denom = np.where(np.abs(c0) < eps, eps, c0)
    return (c1 / denom - 1.0) / alpha


def similarity(original_mark: np.ndarray, extracted_mark: np.ndarray) -> float:
    denom = np.linalg.norm(extracted_mark)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(original_mark, extracted_mark) / denom)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def psnr(a: np.ndarray, b: np.ndarray, max_val: float = 255.0) -> float:
    err = mse(a, b)
    if err <= 1e-12:
        return float("inf")
    return float(20.0 * np.log10(max_val) - 10.0 * np.log10(err))


def ssim(a: np.ndarray, b: np.ndarray, max_val: float = 255.0) -> float:
    # Standard SSIM map with Gaussian local statistics.
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    mu_a = gaussian_filter(a, sigma=1.5)
    mu_b = gaussian_filter(b, sigma=1.5)

    mu_a_sq = mu_a * mu_a
    mu_b_sq = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sigma_a_sq = gaussian_filter(a * a, sigma=1.5) - mu_a_sq
    sigma_b_sq = gaussian_filter(b * b, sigma=1.5) - mu_b_sq
    sigma_ab = gaussian_filter(a * b, sigma=1.5) - mu_ab

    numerator = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a_sq + mu_b_sq + c1) * (sigma_a_sq + sigma_b_sq + c2)
    ssim_map = numerator / (denominator + 1e-12)
    return float(np.mean(ssim_map))


def image_quality_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    return {
        "mse": mse(reference, candidate),
        "mae": mae(reference, candidate),
        "psnr_db": psnr(reference, candidate),
        "ssim": ssim(reference, candidate),
    }


def remove_mean(mark: np.ndarray) -> np.ndarray:
    return mark - np.mean(mark)


def sign_only(mark: np.ndarray) -> np.ndarray:
    out = np.sign(mark)
    out[out == 0] = 1
    return out


def attack_resize_half_restore(img: np.ndarray) -> np.ndarray:
    pil = Image.fromarray(to_uint8(img))
    w, h = pil.size
    half = pil.resize((max(1, w // 2), max(1, h // 2)), Image.Resampling.BICUBIC)
    restored = half.resize((w, h), Image.Resampling.BICUBIC)
    return np.array(restored, dtype=np.float64)


def attack_jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    pil = Image.fromarray(to_uint8(img))
    buf = BytesIO()
    pil.save(buf, format="JPEG", quality=quality, optimize=True)
    buf.seek(0)
    out = Image.open(buf).convert("L")
    return np.array(out, dtype=np.float64)


def attack_dither(img: np.ndarray) -> np.ndarray:
    pil = Image.fromarray(to_uint8(img)).convert("1")
    out = pil.convert("L")
    return np.array(out, dtype=np.float64)


def attack_crop_restore_center_quarter(wm_img: np.ndarray, original_img: np.ndarray) -> np.ndarray:
    h, w = wm_img.shape
    y0, y1 = h // 4, (3 * h) // 4
    x0, x1 = w // 4, (3 * w) // 4
    restored = original_img.copy()
    restored[y0:y1, x0:x1] = wm_img[y0:y1, x0:x1]
    return restored


def attack_print_xerox_scan_proxy(img: np.ndarray) -> np.ndarray:
    # Approximation of analog copy chain via blur, resample and additive noise.
    pil = Image.fromarray(to_uint8(img))
    pil = pil.filter(ImageFilter.GaussianBlur(radius=0.8))
    w, h = pil.size
    down = pil.resize((max(1, int(w * 0.7)), max(1, int(h * 0.7))), Image.Resampling.BILINEAR)
    up = down.resize((w, h), Image.Resampling.BILINEAR)
    arr = np.array(up, dtype=np.float64)
    noise = np.random.default_rng(123).normal(0, 5.0, arr.shape)
    return np.clip(arr + noise, 0, 255)


def save_image(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8(arr)).save(path)


def plot_uniqueness(scores: np.ndarray, true_index: int, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 3))
    plt.plot(scores, lw=1)
    plt.axvline(true_index, color="red", linestyle="--", lw=1, label="True watermark")
    plt.title("Detector response to random watermarks")
    plt.xlabel("Watermark candidate index")
    plt.ylabel("Similarity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def evaluate_candidates(
    original_img: np.ndarray,
    attacked_img: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    alpha: float,
    candidates: Iterable[np.ndarray],
) -> np.ndarray:
    extracted = extract_watermark(original_img, attacked_img, rows, cols, alpha)
    return np.array([similarity(c, extracted) for c in candidates], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce Cox et al. (1997) DCT watermarking experiments.")
    parser.add_argument("--image", type=str, default=None, help="Path to grayscale host image.")
    parser.add_argument("--dataset-dir", type=str, default="dataset", help="Directory to search for input image.")
    parser.add_argument("--out-dir", type=str, default="outputs", help="Directory for figures and report.")
    parser.add_argument("--watermark-length", type=int, default=1000, help="Watermark length n.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Global scale factor.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num-random-tests", type=int, default=1000, help="Random candidate count for uniqueness.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(args.dataset_dir)

    rng = np.random.default_rng(args.seed)
    original_img, image_source = load_grayscale_image(Path(args.image) if args.image else None, dataset_dir)
    save_image(out_dir / "original.png", original_img)

    wm = embed_watermark(original_img, n=args.watermark_length, alpha=args.alpha, rng=rng)
    save_image(out_dir / "watermarked.png", wm.image)

    results: dict[str, float] = {}
    quality: dict[str, dict[str, float]] = {}
    extracted_clean = extract_watermark(original_img, wm.image, wm.rows, wm.cols, args.alpha)
    results["clean"] = similarity(wm.watermark, extracted_clean)
    quality["watermarked_vs_original"] = image_quality_metrics(original_img, wm.image)

    # Paper experiment 1: uniqueness against random marks.
    random_candidates = [gaussian_watermark(len(wm.watermark), rng) for _ in range(args.num_random_tests)]
    true_index = int(rng.integers(0, args.num_random_tests))
    random_candidates[true_index] = wm.watermark.copy()
    uniqueness_scores = evaluate_candidates(
        original_img=original_img,
        attacked_img=wm.image,
        rows=wm.rows,
        cols=wm.cols,
        alpha=args.alpha,
        candidates=random_candidates,
    )
    plot_uniqueness(uniqueness_scores, true_index, out_dir / "exp1_uniqueness.png")
    results["exp1_true_score"] = float(uniqueness_scores[true_index])
    results["exp1_max_false_score"] = float(
        np.max(np.delete(uniqueness_scores, true_index)) if args.num_random_tests > 1 else 0.0
    )

    # Paper experiment 2: scale down + restore.
    scaled = attack_resize_half_restore(wm.image)
    save_image(out_dir / "exp2_scaled_restored.png", scaled)
    results["exp2_scaled_restored"] = similarity(
        wm.watermark,
        extract_watermark(original_img, scaled, wm.rows, wm.cols, args.alpha),
    )
    quality["exp2_scaled_restored_vs_original"] = image_quality_metrics(original_img, scaled)

    # Paper experiment 3: aggressive JPEG.
    jpeg10 = attack_jpeg(wm.image, quality=10)
    save_image(out_dir / "exp3_jpeg_q10.png", jpeg10)
    results["exp3_jpeg_q10"] = similarity(
        wm.watermark,
        extract_watermark(original_img, jpeg10, wm.rows, wm.cols, args.alpha),
    )
    quality["exp3_jpeg_q10_vs_original"] = image_quality_metrics(original_img, jpeg10)
    jpeg5 = attack_jpeg(wm.image, quality=5)
    save_image(out_dir / "exp3_jpeg_q5.png", jpeg5)
    results["exp3_jpeg_q5"] = similarity(
        wm.watermark,
        extract_watermark(original_img, jpeg5, wm.rows, wm.cols, args.alpha),
    )
    quality["exp3_jpeg_q5_vs_original"] = image_quality_metrics(original_img, jpeg5)

    # Paper experiment 4: dithering.
    dithered = attack_dither(wm.image)
    save_image(out_dir / "exp4_dithered.png", dithered)
    extracted_dithered = extract_watermark(original_img, dithered, wm.rows, wm.cols, args.alpha)
    results["exp4_dithered_raw"] = similarity(wm.watermark, extracted_dithered)
    results["exp4_dithered_mean_removed"] = similarity(wm.watermark, remove_mean(extracted_dithered))
    quality["exp4_dithered_vs_original"] = image_quality_metrics(original_img, dithered)

    # Paper experiment 5: cropping + restoration with original.
    cropped_restored = attack_crop_restore_center_quarter(wm.image, original_img)
    save_image(out_dir / "exp5_cropped_restored.png", cropped_restored)
    results["exp5_cropped_restored"] = similarity(
        wm.watermark,
        extract_watermark(original_img, cropped_restored, wm.rows, wm.cols, args.alpha),
    )
    quality["exp5_cropped_restored_vs_original"] = image_quality_metrics(original_img, cropped_restored)
    cropped_jpeg = attack_crop_restore_center_quarter(jpeg10, original_img)
    save_image(out_dir / "exp5_cropped_jpeg_restored.png", cropped_jpeg)
    results["exp5_cropped_jpeg_restored"] = similarity(
        wm.watermark,
        extract_watermark(original_img, cropped_jpeg, wm.rows, wm.cols, args.alpha),
    )
    quality["exp5_cropped_jpeg_restored_vs_original"] = image_quality_metrics(original_img, cropped_jpeg)

    # Paper experiment 6 proxy: print/xerox/scan.
    pxs = attack_print_xerox_scan_proxy(wm.image)
    save_image(out_dir / "exp6_print_xerox_scan_proxy.png", pxs)
    extracted_pxs = extract_watermark(original_img, pxs, wm.rows, wm.cols, args.alpha)
    results["exp6_proxy_raw"] = similarity(wm.watermark, extracted_pxs)
    results["exp6_proxy_mean_removed_sign"] = similarity(wm.watermark, sign_only(remove_mean(extracted_pxs)))
    quality["exp6_proxy_vs_original"] = image_quality_metrics(original_img, pxs)

    # Paper experiment 7: repeated watermarking.
    stacked = original_img.copy()
    multi_marks: list[np.ndarray] = []
    for _ in range(5):
        mark = gaussian_watermark(args.watermark_length, rng)
        wm_step = embed_watermark(stacked, n=args.watermark_length, alpha=args.alpha, rng=rng, watermark=mark)
        stacked = wm_step.image
        multi_marks.append(mark)
    save_image(out_dir / "exp7_five_successive_watermarks.png", stacked)
    random_pool_exp7 = [gaussian_watermark(args.watermark_length, rng) for _ in range(args.num_random_tests - 5)]
    candidates_exp7 = multi_marks + random_pool_exp7
    scores_exp7 = evaluate_candidates(original_img, stacked, wm.rows, wm.cols, args.alpha, candidates_exp7)
    plt.figure(figsize=(10, 3))
    plt.plot(scores_exp7, lw=1)
    for i in range(5):
        plt.axvline(i, color="red", linestyle="--", lw=0.8)
    plt.title("Experiment 7: candidate responses (first five are true marks)")
    plt.xlabel("Candidate index")
    plt.ylabel("Similarity")
    plt.tight_layout()
    plt.savefig(out_dir / "exp7_candidate_response.png", dpi=160)
    plt.close()
    results["exp7_top5_mean_score"] = float(np.mean(np.sort(scores_exp7)[-5:]))
    quality["exp7_five_successive_vs_original"] = image_quality_metrics(original_img, stacked)

    # Paper experiment 8: collusion by averaging 5 independently watermarked copies.
    collusion_marks: list[np.ndarray] = []
    collusion_imgs: list[np.ndarray] = []
    for _ in range(5):
        mark = gaussian_watermark(args.watermark_length, rng)
        wm_i = embed_watermark(original_img, n=args.watermark_length, alpha=args.alpha, rng=rng, watermark=mark)
        collusion_marks.append(mark)
        collusion_imgs.append(wm_i.image)
    colluded = np.mean(np.stack(collusion_imgs, axis=0), axis=0)
    save_image(out_dir / "exp8_collusion_average.png", colluded)
    random_pool_exp8 = [gaussian_watermark(args.watermark_length, rng) for _ in range(args.num_random_tests - 5)]
    candidates_exp8 = collusion_marks + random_pool_exp8
    scores_exp8 = evaluate_candidates(original_img, colluded, wm.rows, wm.cols, args.alpha, candidates_exp8)
    plt.figure(figsize=(10, 3))
    plt.plot(scores_exp8, lw=1)
    for i in range(5):
        plt.axvline(i, color="red", linestyle="--", lw=0.8)
    plt.title("Experiment 8: collusion response (first five are true marks)")
    plt.xlabel("Candidate index")
    plt.ylabel("Similarity")
    plt.tight_layout()
    plt.savefig(out_dir / "exp8_candidate_response.png", dpi=160)
    plt.close()
    results["exp8_top5_mean_score"] = float(np.mean(np.sort(scores_exp8)[-5:]))
    quality["exp8_collusion_average_vs_original"] = image_quality_metrics(original_img, colluded)

    paper_detector_scores = {
        "clean": 32.0,
        "exp2_scaled_restored": 13.4,
        "exp3_jpeg_q10": 22.8,
        "exp3_jpeg_q5": 13.9,
        "exp4_dithered_raw": 5.2,
        "exp4_dithered_mean_removed": 10.5,
        "exp5_cropped_restored": 14.6,
        "exp5_cropped_jpeg_restored": 10.6,
        "exp6_proxy_raw": 4.0,
        "exp6_proxy_mean_removed_sign": 7.0,
    }
    detector_comparison: dict[str, dict[str, float]] = {}
    for key, paper_value in paper_detector_scores.items():
        if key in results:
            ours = float(results[key])
            detector_comparison[key] = {
                "paper": paper_value,
                "reproduction": ours,
                "absolute_delta": abs(ours - paper_value),
                "ratio_reproduction_to_paper": ours / paper_value if abs(paper_value) > 1e-12 else float("nan"),
            }

    report = {
        "paper": "Cox, Kilian, Leighton, Shamoon (1997) Secure Spread Spectrum Watermarking for Multimedia",
        "image_source": image_source,
        "parameters": {
            "watermark_length": args.watermark_length,
            "alpha": args.alpha,
            "seed": args.seed,
            "num_random_tests": args.num_random_tests,
        },
        "metrics": results,
        "quality_metrics": quality,
        "paper_comparison": {
            "detector_scores": detector_comparison,
            "note": "Direct numeric matching is not expected unless image, attack operators, registration, and acquisition pipeline match the paper.",
        },
        "notes": [
            "Implements insertion formula x_i* = x_i * (1 + alpha * w_i) in top-magnitude DCT coefficients (excluding DC).",
            "Similarity follows sim(w, w_hat) = (w · w_hat) / ||w_hat||.",
            "Experiment 6 is an analog-chain proxy, not literal print-xerox-scan hardware reproduction.",
            "Best fidelity to the original paper requires using the same host image and registration assumptions.",
        ],
    }
    with (out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Done. Report: {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
