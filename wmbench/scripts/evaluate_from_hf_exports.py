#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath

import torch

# Running this script directly should still import `wmbench.*`.
_pkg_parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _pkg_parent not in sys.path:
    sys.path.insert(0, _pkg_parent)

from wmbench.output.plotter import render_all_plots
from wmbench.pipeline.aggregate import run_aggregate_stage
from wmbench.pipeline.evaluate import run_evaluate_stage


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dev = torch.device(device_arg)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return dev


def _coerce_strength(tag: str) -> float | int | str:
    try:
        f = float(tag)
    except ValueError:
        return tag
    return int(f) if f.is_integer() else f


def _safe_extract_member(
    zf: zipfile.ZipFile,
    member: zipfile.ZipInfo,
    dst_root: Path,
    *,
    rel_parts: tuple[str, ...] | None = None,
) -> Path | None:
    if member.is_dir():
        return None
    parts = PurePosixPath(member.filename).parts
    if not parts:
        return None
    if ".." in parts:
        raise ValueError(f"Unsafe zip member path: {member.filename!r}")
    use_parts = rel_parts if rel_parts is not None else parts
    if ".." in use_parts:
        raise ValueError(f"Unsafe relative path: {use_parts!r}")
    out_path = dst_root.joinpath(*use_parts)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(member) as src, out_path.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    return out_path


def _extract_hf_exports(
    *,
    method: str,
    exports_dir: Path,
    work_dir: Path,
) -> dict[str, list[float | int | str]]:
    """Extract method__<attack>.zip into work/<method>/{attacked,scores}/..."""
    prefix = f"{method}__"
    zip_paths = sorted(p for p in exports_dir.glob(f"{prefix}*.zip") if not p.name.endswith("__FULL_RESULTS.zip"))
    if not zip_paths:
        raise FileNotFoundError(f"No zip files matching {prefix}*.zip under {exports_dir}")

    attacked_root = work_dir / "attacked"
    scores_root = work_dir / "scores"
    attacked_root.mkdir(parents=True, exist_ok=True)
    scores_root.mkdir(parents=True, exist_ok=True)

    strength_map: dict[str, set[str]] = {}
    for zpath in zip_paths:
        attack = zpath.stem[len(prefix) :]
        if not attack:
            continue
        strength_map.setdefault(attack, set())
        with zipfile.ZipFile(zpath, "r") as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                parts = PurePosixPath(member.filename).parts
                if not parts:
                    continue

                # Layout written by kaggle_wmbench_common.py:
                # - attacked images: <strength>/<image_name>
                # - scores: scores/<strength>/scores.json
                if parts[0] == "scores":
                    if len(parts) < 3:
                        continue
                    strength = parts[1]
                    dst = scores_root / attack / strength
                    _safe_extract_member(zf, member, dst, rel_parts=parts[2:])
                    strength_map[attack].add(strength)
                else:
                    strength = parts[0]
                    dst = attacked_root / attack
                    _safe_extract_member(zf, member, dst)
                    strength_map[attack].add(strength)

    out: dict[str, list[float | int | str]] = {}
    for attack, strengths in strength_map.items():
        ordered = sorted(
            strengths,
            key=lambda s: float(s) if s.replace(".", "", 1).isdigit() else s,
        )
        out[attack] = [_coerce_strength(s) for s in ordered]
    return out


def _resolve_exports_dir(path: Path) -> Path:
    if (path / "wmbench_exports").is_dir():
        return path / "wmbench_exports"
    return path


def _seed_watermarked_basenames_from_attacked(work_dir: Path, attack_names: list[str]) -> int:
    """
    Create/refresh work/<method>/watermarked basenames from attacked outputs.
    This enables evaluate FID reference mirror path (avoids in-memory temp-dir FID path).
    """
    attacked_root = work_dir / "attacked"
    watermarked_dir = work_dir / "watermarked"
    watermarked_dir.mkdir(parents=True, exist_ok=True)
    seeded = 0
    for attack in attack_names:
        atk_dir = attacked_root / attack
        if not atk_dir.is_dir():
            continue
        strength_dirs = sorted(p for p in atk_dir.iterdir() if p.is_dir())
        for sdir in strength_dirs:
            imgs = sorted(
                p
                for p in sdir.iterdir()
                if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
            )
            if not imgs:
                continue
            for src in imgs:
                dst = watermarked_dir / src.name
                if dst.exists():
                    continue
                try:
                    # Hardlink is O(1) and keeps valid image content without duplication.
                    os.link(src, dst)
                except OSError:
                    shutil.copy2(src, dst)
                seeded += 1
            if seeded:
                return seeded
    return seeded


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Rebuild wmbench metrics/leaderboard from HF per-attack export zip files."
    )
    p.add_argument("--method", required=True, help="wmbench method id, e.g. dct, dwt, dct-dwt")
    p.add_argument("--exports", required=True, help="Directory containing method__*.zip (or parent with wmbench_exports/)")
    p.add_argument("--output", required=True, help="Output root for rebuilt results")
    p.add_argument("--originals", required=True, help="Directory of original clean images used for benchmark")
    p.add_argument("--device", default="auto", help="auto | cuda | cuda:0 | cpu")
    p.add_argument("--lpips-batch-size", type=int, default=16)
    p.add_argument("--skip-aesthetics-metrics", action="store_true")
    p.add_argument("--resume", action="store_true", help="Resume evaluate stage where metrics/.done exists")
    p.add_argument("--clean-work", action="store_true", help="Delete output/work/<method> before extraction")
    args = p.parse_args(argv)

    method = args.method.strip().lower()
    output_dir = Path(args.output).resolve()
    exports_dir = _resolve_exports_dir(Path(args.exports).resolve())
    originals_dir = Path(args.originals).resolve()
    if not originals_dir.is_dir():
        raise FileNotFoundError(f"--originals not found: {originals_dir}")
    if not exports_dir.is_dir():
        raise FileNotFoundError(f"--exports not found: {exports_dir}")

    work_dir = output_dir / "work" / method
    if args.clean_work and work_dir.exists():
        shutil.rmtree(work_dir)

    strength_map = _extract_hf_exports(method=method, exports_dir=exports_dir, work_dir=work_dir)
    attack_names = sorted(strength_map.keys())
    if not attack_names:
        raise RuntimeError(f"No attacks extracted for method {method!r} from {exports_dir}")

    seeded = _seed_watermarked_basenames_from_attacked(work_dir, attack_names)
    print(f"evaluate_from_hf_exports: seeded watermarked basenames={seeded}")

    dev = _resolve_device(args.device.strip().lower() if args.device else "auto")
    print(f"evaluate_from_hf_exports: method={method}, device={dev}")
    print(f"evaluate_from_hf_exports: attacks={attack_names}")

    run_evaluate_stage(
        work_dir=str(work_dir),
        originals_dir=str(originals_dir),
        attack_names=attack_names,
        strength_values=strength_map,
        output_dir=str(output_dir),
        resume=args.resume,
        device=dev,
        skip_aesthetics_metrics=args.skip_aesthetics_metrics,
        lpips_batch_size=max(1, int(args.lpips_batch_size)),
    )
    run_aggregate_stage(str(output_dir / "work"), str(output_dir), [method])
    render_all_plots(str(output_dir / "work"), str(output_dir), [method], attack_names)

    print("\nDone.")
    print(f"Leaderboard: {output_dir / 'results_leaderboard.csv'}")
    print(f"Averaged:    {output_dir / 'results_averaged.csv'}")
    print(f"Raw:         {output_dir / 'results_raw.csv'}")
    print(f"Plots:       {output_dir / 'plots'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
