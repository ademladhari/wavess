#!/usr/bin/env python3
"""
SCRATCHPAD — provenance (read before changing benchmark semantics)

DCT embedding / decoding
- Embed (WAVES bit-sign DCT): `wmbench.utils.dct_utils` — embed_bits(image ndarray, bits, alpha);
  top DCT magnitudes excluding DC; α default 0.1.
- Detect: same module extract_bits(original, candidate, length, alpha) → bool bits vs ground truth;
  wmbench `watermarks/dct.DCTAdapter` maps bit accuracy to float score in [0,1].
- WAVES reference: `waves/utils/dct_utils.py`, params `waves/dev/constants.py` (len 1000, seed 42, α 0.1),
  decode flow `waves/scripts/decode.py` (resize attacked to original, non-blind).

DWT embedding / decoding
- Embed / detect rewritten from `dwt/dwt_watermark_xia1998.ipynb`: multilevel Haar DWT, embed on largest
  detail coeffs (Eq. 4), hierarchical normalized cross-correlation decoder with peak ratio.
- `watermarks/dwt.DWTAdapter`; per-image payload saved as `.wmbench_meta.pkl` sidecar for resume/detect.

WAVES distortion attacks (12)
- Singles + relative strengths: `waves/distortions/distortions.py` (`relative_strength_to_absolute`,
  `apply_single_distortion`).
- Combo ordering: `waves/scripts/apply_distortion_attacks_flat.py` `_chain_specs`.

WAVES regeneration (6 implemented here)
- `waves/regeneration/regen.py`: VAEWMAttacker, DiffWMAttacker (ReSDPipeline), regen_diff, regen_vae,
  rinse_2xDiff, rinse_4xDiff. VAE call uses `strength=` for CompressAI quality (fixes broken `quality=` kw).

Missing from upstream (logged to results/missing_components.txt)
- `Regen-DiffP` (regen_diffusion_prompt), `Regen-KLVAE` — names only in `waves/dev/constants.py`, no regen.py impl.

WAVES quality metrics
- PSNR/SSIM/NMI: `waves/metrics/image.py`; LPIPS: `waves/metrics/perceptual.py`;
  FID/CLIP-FID: `waves/metrics/distributional.py`; aesthetics/artifacts: `waves/metrics/aesthetics.py`.

Normalization / Q
- p10/p90 anchors and Q aggregation are **wmbench** (`wmbench.metrics.aggregate`), not in WAVES code.

Strength grids
- WAVES discovers strengths from existing result trees (`waves/dev/leaderboard_table.py`); no central grid file.
- Defaults (WAVES paper, 5 strengths each): relative ∈ {0, 0.25, 0.5, 0.75, 1};
  Regen-Diff timesteps 40–200; Rinse-2x 20–100; Rinse-4x 10–50; Regen-VAE quality 1–7 (evenly spaced).
"""

from __future__ import annotations

import argparse
import faulthandler
import os
import sys
import time
import traceback

# Running `py D:\...\wmbench\run_benchmark.py` puts only `...\wmbench` on sys.path; imports need the parent
# of that folder (the directory that contains the `wmbench` package).
_pkg_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_parent not in sys.path:
    sys.path.insert(0, _pkg_parent)

import torch

from wmbench.attacks.registry import resolve_attacks
from wmbench.output.plotter import render_all_plots
from wmbench.pipeline.aggregate import run_aggregate_stage
from wmbench.pipeline.attack import run_attack_stage
from wmbench.pipeline.detect import run_detect_stage
from wmbench.pipeline.embed import list_image_paths, run_embed
from wmbench.pipeline.evaluate import run_evaluate_stage
from wmbench.watermarks import get_adapter


def _resolve_device(device_arg: str) -> torch.device:
    """Select compute device. Use PyTorch built with CUDA (e.g. cu128 wheels) for GPU."""
    if device_arg == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dev = torch.device(device_arg)
    if dev.type == "cuda" and not torch.cuda.is_available():
        print(
            "CUDA was requested (--device cuda) but torch.cuda.is_available() is False.\n"
            "Install a GPU build of PyTorch, e.g. (see https://pytorch.org/get-started/locally/):\n"
            "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return dev


def _print_stage_profile(stage: str, dt_s: float, device: torch.device) -> None:
    if device.type != "cuda":
        print(f"profile/{stage}: {dt_s:.1f}s", flush=True)
        return
    try:
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        print(f"profile/{stage}: {dt_s:.1f}s, peak_cuda_alloc={peak_mb:.1f}MB", flush=True)
    except Exception:
        print(f"profile/{stage}: {dt_s:.1f}s", flush=True)


def main(argv: list[str] | None = None) -> int:
    # Dump Python stack on SIGUSR2/Linux; helps diagnose silent native crashes after CUDA loads.
    try:
        faulthandler.enable(all_threads=True)
    except Exception:
        pass

    p = argparse.ArgumentParser(description="wmbench: watermark robustness and quality benchmark")
    p.add_argument(
        "--methods",
        nargs="+",
        required=True,
        help="dct | dwt | svd | dct-dwt | dwt-dct-svd | flexible",
    )
    p.add_argument("--images", required=True, help="Directory of originals (clean host images)")
    p.add_argument("--negatives", required=True, help="Directory of negative (unwatermarked) images for detection calibration")
    p.add_argument("--output", required=True, help="Results root (work/, csv, plots, anchors)")
    p.add_argument("--attacks", nargs="*", default=None, help="Subset of attack display names (default: all implemented)")
    p.add_argument(
        "--skiprince4xdiff",
        "--skip-rinse4xdiff",
        dest="skip_rinse4xdiff",
        action="store_true",
        help="Skip the Rinse-4xDiff attack to reduce runtime.",
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--blind-detect",
        action="store_true",
        help="Blind detection: use embed sidecar only (no --images original at detect). Re-embed if sidecars lack dct_embed.",
    )
    p.add_argument("--strength-config", default=None, help="JSON mapping attack name -> strengths list")
    p.add_argument(
        "--skip-aesthetics-metrics",
        action="store_true",
        help="Skip aesthetics/artifacts metrics to speed up evaluation.",
    )
    p.add_argument("--diffusion-model", default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--vae-model", default="bmshj2018-factorized")
    p.add_argument(
        "--diffusion-attack-batch-size",
        type=int,
        default=1,
        help="Images per attack forward pass (for diffusion/Rinse this enables VRAM-heavy batching).",
    )
    p.add_argument(
        "--lpips-batch-size",
        type=int,
        default=1,
        help="LPIPS pair minibatch size during evaluate phase.",
    )
    p.add_argument(
        "--profile-stages",
        action="store_true",
        help="Print per-stage wall-clock and peak CUDA allocation for tuning.",
    )
    p.add_argument(
        "--device",
        default="auto",
        help="auto | cuda | cuda:0 | cpu. For RTX 50xx install PyTorch cu128 so 'cuda' works.",
    )
    args = p.parse_args(argv)

    out = os.path.abspath(args.output)
    work_root = os.path.join(out, "work")
    os.makedirs(out, exist_ok=True)

    image_paths = list_image_paths(args.images)
    if not image_paths:
        print("No images found under --images", file=sys.stderr)
        return 1

    device = _resolve_device(args.device.strip().lower() if args.device else "auto")
    print(f"wmbench device: {device} (torch {torch.__version__}, cuda available={torch.cuda.is_available()})")
    if args.blind_detect:
        print("Detection mode: blind (embed sidecar metadata; originals not used for detect)")
    else:
        print("Detection mode: non-blind (detect uses --images as original reference)")

    attacks = resolve_attacks(
        out,
        args.attacks if args.attacks else None,
        diffusion_model_id=args.diffusion_model,
        vae_model_name=args.vae_model,
        strength_config_path=args.strength_config,
        device=device,
    )
    if args.skip_rinse4xdiff and "Rinse-4xDiff" in attacks:
        attacks = {k: v for k, v in attacks.items() if k != "Rinse-4xDiff"}
        print("Skipping attack Rinse-4xDiff (--skiprince4xdiff).")
    if not attacks:
        print("No attacks selected after filtering; nothing to run.", file=sys.stderr)
        return 1
    strength_map = {name: list(atk.strengths) for name, atk in attacks.items()}
    attack_list = list(attacks.keys())

    for method in args.methods:
        m = method.strip().lower()
        adapter = get_adapter(m)
        work = os.path.join(work_root, m)
        wm_dir = os.path.join(work, "watermarked")
        if args.profile_stages and device.type == "cuda":
            try:
                torch.cuda.reset_peak_memory_stats(device)
            except Exception:
                pass
        t0 = time.perf_counter()
        run_embed(adapter, image_paths, wm_dir, resume=args.resume)
        if args.profile_stages:
            _print_stage_profile(f"{m}/embed", time.perf_counter() - t0, device)

        if args.profile_stages and device.type == "cuda":
            try:
                torch.cuda.reset_peak_memory_stats(device)
            except Exception:
                pass
        t0 = time.perf_counter()
        run_attack_stage(
            attacks,
            wm_dir,
            os.path.join(work, "attacked"),
            resume=args.resume,
            diffusion_attack_batch_size=args.diffusion_attack_batch_size,
        )
        if args.profile_stages:
            _print_stage_profile(f"{m}/attack", time.perf_counter() - t0, device)

        if args.profile_stages and device.type == "cuda":
            try:
                torch.cuda.reset_peak_memory_stats(device)
            except Exception:
                pass
        t0 = time.perf_counter()
        run_detect_stage(
            adapter,
            work,
            os.path.abspath(args.images),
            os.path.abspath(args.negatives),
            attack_list,
            strength_map,
            resume=args.resume,
            blind_detect=args.blind_detect,
        )
        if args.profile_stages:
            _print_stage_profile(f"{m}/detect", time.perf_counter() - t0, device)
        try:
            if args.profile_stages and device.type == "cuda":
                try:
                    torch.cuda.reset_peak_memory_stats(device)
                except Exception:
                    pass
            t0 = time.perf_counter()
            run_evaluate_stage(
                work,
                os.path.abspath(args.images),
                attack_list,
                strength_map,
                output_dir=out,
                resume=args.resume,
                device=device,
                skip_aesthetics_metrics=args.skip_aesthetics_metrics,
                lpips_batch_size=args.lpips_batch_size,
                profile_metrics=args.profile_stages,
            )
            if args.profile_stages:
                _print_stage_profile(f"{m}/evaluate", time.perf_counter() - t0, device)
        except BaseException:
            crash_path = os.path.join(out, "evaluate_crash.txt")
            tb = traceback.format_exc()
            try:
                with open(crash_path, "w", encoding="utf-8") as cf:
                    cf.write(tb)
            except OSError:
                pass
            print(
                f"wmbench: evaluate stage crashed (Python exception). Traceback saved to:\n  {crash_path}",
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exc()
            return 1

    run_aggregate_stage(work_root, out, [m.strip().lower() for m in args.methods])
    render_all_plots(work_root, out, [m.strip().lower() for m in args.methods], attack_list)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
