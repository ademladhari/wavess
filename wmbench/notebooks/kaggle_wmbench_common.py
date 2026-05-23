"""
Shared helpers for Kaggle wmbench notebooks.
Zip per-attack folders under work/{method}/attacked/ and publish for download.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

# All implemented attacks (order: GPU/heavy first, then distortions)
GPU_ATTACKS: tuple[str, ...] = (
    "Regen-Diff",
    "Regen-VAE",
    "Rinse-2xDiff",
)
DIST_ATTACKS: tuple[str, ...] = (
    "Dist-Rotation",
    "Dist-RCrop",
    "Dist-Erase",
    "Dist-Bright",
    "Dist-Contrast",
    "Dist-Blur",
    "Dist-Noise",
    "Dist-JPEG",
    "DistCom-Geo",
    "DistCom-Photo",
    "DistCom-Deg",
    "DistCom-All",
)
ALL_ATTACKS: tuple[str, ...] = GPU_ATTACKS + DIST_ATTACKS


def slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s.strip())


def find_waves_root(start: Path | None = None) -> Path:
    """Locate repo root (directory that contains wmbench/)."""
    here = (start or Path.cwd()).resolve()
    for p in [here, *here.parents]:
        if (p / "wmbench" / "run_benchmark.py").is_file():
            return p
    raise FileNotFoundError(
        "Could not find waves repo (wmbench/run_benchmark.py). "
        "Clone into /kaggle/working/waves or set WAVES_ROOT."
    )


def run_benchmark(
    *,
    waves_root: Path,
    method: str,
    output_dir: Path,
    images: str | None,
    negatives: str | None,
    attacks: list[str],
    generate_based: bool = False,
    blind_detect: bool = True,
    device: str = "cuda",
    diff_batch: int = 4,
    lpips_batch: int = 16,
    skip_rinse4x: bool = True,
    resume: bool = True,
    skip_aesthetics: bool = False,
    extra_argv: list[str] | None = None,
) -> int:
    cmd = [
        sys.executable,
        str(waves_root / "wmbench" / "run_benchmark.py"),
        "--methods",
        method,
        "--output",
        str(output_dir),
        "--device",
        device,
        "--attacks",
        *attacks,
        "--diffusion-attack-batch-size",
        str(diff_batch),
        "--lpips-batch-size",
        str(lpips_batch),
        "--blind-detect",
    ]
    if images:
        cmd += ["--images", images]
    if negatives:
        cmd += ["--negatives", negatives]
    if generate_based:
        cmd.append("--generate-based")
    if skip_rinse4x:
        cmd.append("--skip-rinse4xdiff")
    if resume:
        cmd.append("--resume")
    if skip_aesthetics:
        cmd.append("--skip-aesthetics-metrics")
    if extra_argv:
        cmd.extend(extra_argv)
    print("Running:", " ".join(cmd), flush=True)

    # Prefer in-process run so Jupyter shows tqdm/logs immediately (subprocess buffers on Kaggle).
    argv = cmd[2:]  # drop sys.executable and script path
    old_cwd = os.getcwd()
    old_path0 = sys.path[0] if sys.path else None
    waves_s = str(waves_root.resolve())
    try:
        os.chdir(waves_s)
        if waves_s not in sys.path:
            sys.path.insert(0, waves_s)
        os.environ.setdefault("PYTHONUNBUFFERED", "1")
        from wmbench.run_benchmark import main as benchmark_main

        return int(benchmark_main(argv))
    except Exception:
        # Fallback if import path differs on Kaggle
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        return subprocess.call(
            [sys.executable, "-u", *cmd[1:]],
            cwd=waves_s,
            env=env,
            stdout=None,
            stderr=None,
        )
    finally:
        os.chdir(old_cwd)
        if old_path0 is not None:
            try:
                sys.path.remove(waves_s)
            except ValueError:
                pass


def zip_attack_folder(
    attacked_root: Path,
    attack_name: str,
    export_dir: Path,
    *,
    method: str,
    include_scores: bool = True,
    work_root: Path | None = None,
) -> Path:
    """Zip work/{method}/attacked/{attack}/ (+ optional detect scores)."""
    src = attacked_root / attack_name
    if not src.is_dir():
        raise FileNotFoundError(f"Attack folder missing: {src}")

    export_dir.mkdir(parents=True, exist_ok=True)
    zip_name = f"{slug(method)}__{slug(attack_name)}.zip"
    zip_path = export_dir / zip_name

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as zf:
        for f in sorted(src.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(src).as_posix())
        if include_scores and work_root is not None:
            scores = work_root / "scores" / attack_name
            if scores.is_dir():
                for f in sorted(scores.rglob("*")):
                    if f.is_file():
                        zf.write(f, ("scores/" + f.relative_to(scores).as_posix()))

    return zip_path


def append_manifest(export_dir: Path, entry: dict) -> Path:
    manifest = export_dir / "export_manifest.jsonl"
    entry = {**entry, "ts": datetime.now(timezone.utc).isoformat()}
    with manifest.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return manifest


def upload_to_huggingface(zip_path: Path, repo_id: str, token: str | None = None) -> str:
    from huggingface_hub import HfApi, create_repo

    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("Set HF_TOKEN in Kaggle Secrets (or pass token=).")

    api = HfApi(token=token)
    create_repo(repo_id, repo_type="dataset", exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(zip_path),
        path_in_repo=f"wmbench_exports/{zip_path.name}",
        repo_id=repo_id,
        repo_type="dataset",
    )
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/wmbench_exports/{zip_path.name}"
    return url


def run_per_attack_with_zips(
    *,
    method: str,
    waves_root: Path,
    output_dir: Path,
    attack_names: list[str],
    export_dir: Path,
    hf_repo_id: str | None = None,
    upload_each: bool = True,
    images: str | None = None,
    negatives: str | None = None,
    generate_based: bool = False,
    **bench_kwargs,
) -> list[dict]:
    """
    Run one attack at a time (--resume), zip attacked/{attack}/ after each, optional HF upload.
    Returns list of manifest entries.
    """
    output_dir = output_dir.resolve()
    work_root = output_dir / "work" / method
    attacked_root = work_root / "attacked"
    entries: list[dict] = []

    for attack in attack_names:
        print(f"\n{'=' * 60}\nMETHOD={method}  ATTACK={attack}\n{'=' * 60}\n", flush=True)
        rc = run_benchmark(
            waves_root=waves_root,
            method=method,
            output_dir=output_dir,
            images=images,
            negatives=negatives,
            attacks=[attack],
            generate_based=generate_based,
            **bench_kwargs,
        )
        if rc != 0:
            raise RuntimeError(f"run_benchmark failed for attack {attack!r} (exit {rc})")

        if not (attacked_root / attack).is_dir():
            print(f"Warning: no attacked folder for {attack}, skipping zip.")
            continue

        zip_path = zip_attack_folder(
            attacked_root,
            attack,
            export_dir,
            method=method,
            work_root=work_root,
        )
        size_gb = zip_path.stat().st_size / 1e9
        entry = {
            "method": method,
            "attack": attack,
            "zip": str(zip_path),
            "size_bytes": zip_path.stat().st_size,
        }
        print(f"Zipped: {zip_path} ({size_gb:.3f} GB)", flush=True)

        if upload_each and hf_repo_id:
            try:
                url = upload_to_huggingface(zip_path, hf_repo_id)
                entry["hf_url"] = url
                print(f"HF download: {url}", flush=True)
            except Exception as e:
                entry["hf_error"] = repr(e)
                print(f"HF upload failed: {e!r}", flush=True)

        append_manifest(export_dir, entry)
        entries.append(entry)

    return entries


def zip_full_results(output_dir: Path, export_dir: Path, method: str) -> Path:
    """Zip CSVs, plots, and remaining work/ for final download."""
    output_dir = output_dir.resolve()
    export_dir.mkdir(parents=True, exist_ok=True)
    zip_path = export_dir / f"{slug(method)}__FULL_RESULTS.zip"
    include_roots = [
        output_dir / "work" / method,
        output_dir / "plots",
    ]
    include_files = [
        output_dir / "results_raw.csv",
        output_dir / "results_averaged.csv",
        output_dir / "results_leaderboard.csv",
        output_dir / "normalization_anchors.json",
        output_dir / "missing_components.txt",
        output_dir / "export_manifest.jsonl",
    ]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as zf:
        for fp in include_files:
            if fp.is_file():
                zf.write(fp, fp.name)
        for root in include_roots:
            if not root.exists():
                continue
            for f in sorted(root.rglob("*")):
                if f.is_file():
                    zf.write(f, f.relative_to(output_dir).as_posix())
    return zip_path
