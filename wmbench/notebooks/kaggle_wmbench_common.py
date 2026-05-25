"""
Shared helpers for Kaggle wmbench notebooks.
Zip per-attack folders under work/{method}/attacked/ and publish for download.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

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
    skip_aesthetics: bool = True,
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
    include_metrics: bool = True,
    include_result_snapshots: bool = True,
    work_root: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Zip attacked/{attack} plus optional scores/metrics and current result files."""
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
        if include_metrics and work_root is not None:
            metrics = work_root / "metrics" / attack_name
            if metrics.is_dir():
                for f in sorted(metrics.rglob("*")):
                    if f.is_file():
                        zf.write(f, ("metrics/" + f.relative_to(metrics).as_posix()))
        if include_result_snapshots and output_dir is not None:
            # Keep lightweight snapshots of tabular outputs so HF per-attack zips can be
            # post-processed into leaderboard summaries without re-running evaluate.
            snapshot_files = (
                "results_raw.csv",
                "results_averaged.csv",
                "results_leaderboard.csv",
                "normalization_anchors.json",
                "missing_components.txt",
            )
            for name in snapshot_files:
                fp = output_dir / name
                if fp.is_file():
                    zf.write(fp, f"results/{name}")

    return zip_path


def _safe_extract_zip_member(
    zf: zipfile.ZipFile,
    member: zipfile.ZipInfo,
    dst_root: Path,
    *,
    rel_parts: tuple[str, ...] | None = None,
) -> None:
    if member.is_dir():
        return
    parts = PurePosixPath(member.filename).parts
    if not parts or ".." in parts:
        raise ValueError(f"Unsafe zip member path: {member.filename!r}")
    use_parts = rel_parts if rel_parts is not None else parts
    if ".." in use_parts:
        raise ValueError(f"Unsafe relative path: {use_parts!r}")
    out_path = dst_root.joinpath(*use_parts)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(member) as src, out_path.open("wb") as dst:
        shutil.copyfileobj(src, dst)


def _attack_from_export_zip(method: str, zip_path: Path) -> str | None:
    prefix = f"{slug(method)}__"
    if not zip_path.name.startswith(prefix) or not zip_path.name.endswith(".zip"):
        return None
    if zip_path.name.endswith("__FULL_RESULTS.zip"):
        return None
    return zip_path.stem[len(prefix) :]


def attack_is_complete(work_root: Path, attack: str) -> bool:
    """True if attacked/{attack} and scores for every strength are present."""
    attacked = work_root / "attacked" / attack
    if not attacked.is_dir():
        return False
    strengths = [p for p in attacked.iterdir() if p.is_dir()]
    if not strengths:
        return False
    for sdir in strengths:
        if not (work_root / "scores" / attack / sdir.name / "scores.json").is_file():
            return False
    return True


def restore_attack_zip_to_work(
    zip_path: Path,
    *,
    work_root: Path,
    output_dir: Path,
    method: str,
) -> str | None:
    """Extract one per-attack HF export zip into work/{method}/."""
    attack = _attack_from_export_zip(method, zip_path)
    if not attack:
        return None
    attacked_root = work_root / "attacked" / attack
    scores_root = work_root / "scores" / attack
    metrics_root = work_root / "metrics" / attack
    attacked_root.mkdir(parents=True, exist_ok=True)
    scores_root.mkdir(parents=True, exist_ok=True)
    metrics_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            parts = PurePosixPath(member.filename).parts
            if not parts:
                continue
            if parts[0] == "scores" and len(parts) >= 3:
                strength = parts[1]
                _safe_extract_zip_member(
                    zf, member, scores_root / strength, rel_parts=parts[2:]
                )
            elif parts[0] == "metrics" and len(parts) >= 3:
                strength = parts[1]
                _safe_extract_zip_member(
                    zf, member, metrics_root / strength, rel_parts=parts[2:]
                )
            elif parts[0] == "results":
                _safe_extract_zip_member(zf, member, output_dir, rel_parts=parts[1:])
            else:
                _safe_extract_zip_member(zf, member, attacked_root)

    for stren_dir in scores_root.iterdir() if scores_root.is_dir() else []:
        if (stren_dir / "scores.json").is_file():
            done_flag = stren_dir / ".done"
            done_flag.parent.mkdir(parents=True, exist_ok=True)
            done_flag.write_text("ok\n", encoding="utf-8")
    return attack


def restore_neg_cache_from_scores(
    work_root: Path,
    method: str,
    *,
    negatives_dir: str | None,
    blind_detect: bool = True,
) -> bool:
    """Rebuild neg calibration cache from restored scores.json (skips rerunning neg/)."""
    if not negatives_dir or not os.path.isdir(negatives_dir):
        return False
    scores_root = work_root / "scores"
    if not scores_root.is_dir():
        return False
    neg_scores: list[float] | None = None
    for scores_json in sorted(scores_root.glob("*/*/scores.json")):
        try:
            data = json.loads(scores_json.read_text(encoding="utf-8"))
            neg = data.get("negative")
            if isinstance(neg, list) and len(neg) > 0:
                neg_scores = [float(x) for x in neg]
                break
        except Exception:
            continue
    if not neg_scores:
        return False

    from wmbench.pipeline.detect import _file_state_key, _list_negative_image_paths

    neg_paths = _list_negative_image_paths(negatives_dir)
    neg_cache_key = {
        "adapter": method,
        "blind_detect": bool(blind_detect),
        "negatives_dir": os.path.abspath(negatives_dir),
        "neg_paths_state": _file_state_key(neg_paths),
    }
    neg_cache_sig = hashlib.sha1(
        json.dumps(neg_cache_key, sort_keys=True).encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()
    mode = "blind" if blind_detect else "nonblind"
    safe_method = method.replace(os.sep, "_")
    cache_path = scores_root / f"_neg_cache_{safe_method}_{mode}.json"
    cache_path.write_text(
        json.dumps({"sig": neg_cache_sig, "scores": neg_scores}),
        encoding="utf-8",
    )
    print(f"Restored neg cache ({len(neg_scores)} scores) -> {cache_path}", flush=True)
    return True


def download_hf_export_zips(
    repo_id: str,
    method: str,
    export_dir: Path,
    *,
    attacks: list[str] | None = None,
    token: str | None = None,
) -> list[Path]:
    """Download method__<attack>.zip files from HF dataset repo into export_dir."""
    from huggingface_hub import hf_hub_download

    token = resolve_hf_token(token)
    export_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"wmbench_exports/{slug(method)}__"
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    remote_files = api.list_repo_files(repo_id, repo_type="dataset")
    want = sorted(
        f
        for f in remote_files
        if f.startswith(prefix)
        and f.endswith(".zip")
        and not f.endswith("__FULL_RESULTS.zip")
    )
    downloaded: list[Path] = []
    for remote in want:
        attack = remote[len(prefix) : -4]
        if attacks is not None and attack not in attacks:
            continue
        local_name = Path(remote).name
        dest = export_dir / local_name
        if dest.is_file() and dest.stat().st_size > 0:
            downloaded.append(dest)
            continue
        path = hf_hub_download(
            repo_id=repo_id,
            filename=remote,
            repo_type="dataset",
            local_dir=str(export_dir),
            token=token,
        )
        downloaded.append(Path(path))
        print(f"Downloaded HF export: {local_name}", flush=True)
    return downloaded


def restore_from_hf_exports(
    *,
    method: str,
    output_dir: Path,
    export_dir: Path,
    hf_repo_id: str,
    attacks: list[str] | None = None,
    negatives_dir: str | None = None,
    blind_detect: bool = True,
    download: bool = True,
) -> list[str]:
    """
    Download per-attack zips from HF (if needed), extract into work/{method}/,
    restore CSV snapshots, and mark detect/evaluate done flags where possible.
    Returns list of attack names restored.
    """
    output_dir = output_dir.resolve()
    export_dir = export_dir.resolve()
    work_root = output_dir / "work" / method
    work_root.mkdir(parents=True, exist_ok=True)

    zip_paths: list[Path] = []
    if download:
        zip_paths = download_hf_export_zips(
            hf_repo_id, method, export_dir, attacks=attacks
        )
    prefix = f"{slug(method)}__"
    zip_paths.extend(
        sorted(
            p
            for p in export_dir.glob(f"{prefix}*.zip")
            if not p.name.endswith("__FULL_RESULTS.zip")
        )
    )
    seen: set[str] = set()
    restored: list[str] = []
    for zpath in zip_paths:
        if not zpath.is_file():
            continue
        key = str(zpath.resolve())
        if key in seen:
            continue
        seen.add(key)
        attack = restore_attack_zip_to_work(
            zpath, work_root=work_root, output_dir=output_dir, method=method
        )
        if attack and attack not in restored:
            restored.append(attack)
            print(f"Restored attack from HF/local zip: {attack}", flush=True)

    restore_neg_cache_from_scores(
        work_root, method, negatives_dir=negatives_dir, blind_detect=blind_detect
    )
    return restored


def append_manifest(export_dir: Path, entry: dict) -> Path:
    manifest = export_dir / "export_manifest.jsonl"
    entry = {**entry, "ts": datetime.now(timezone.utc).isoformat()}
    with manifest.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return manifest


def resolve_hf_token(token: str | None = None) -> str:
    """HF token from arg, env, or Kaggle Secrets (label must be exactly HF_TOKEN)."""
    if token:
        return token.strip()
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        val = os.environ.get(key)
        if val and val.strip():
            return val.strip()
    try:
        from kaggle_secrets import UserSecretsClient

        val = UserSecretsClient().get_secret("HF_TOKEN")
        if val and str(val).strip():
            os.environ["HF_TOKEN"] = str(val).strip()
            return os.environ["HF_TOKEN"]
    except Exception:
        pass
    raise RuntimeError(
        "HF token missing. In Kaggle: Add-ons → Secrets → label HF_TOKEN, check the box for "
        "this notebook, restart session, run the install cell, then verify with "
        "bool(os.environ.get('HF_TOKEN'))."
    )


def upload_to_huggingface(zip_path: Path, repo_id: str, token: str | None = None) -> str:
    from huggingface_hub import HfApi, create_repo

    token = resolve_hf_token(token)

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
    resume_from_hf: bool = False,
    images: str | None = None,
    negatives: str | None = None,
    generate_based: bool = False,
    **bench_kwargs,
) -> list[dict]:
    """
    Run one attack at a time (--resume), zip attacked/{attack}/ after each, optional HF upload.
    If resume_from_hf=True, download/extract completed attacks from HF first and skip them.
    Returns list of manifest entries.
    """
    output_dir = output_dir.resolve()
    work_root = output_dir / "work" / method
    attacked_root = work_root / "attacked"
    entries: list[dict] = []

    if resume_from_hf and hf_repo_id:
        restored = restore_from_hf_exports(
            method=method,
            output_dir=output_dir,
            export_dir=export_dir,
            hf_repo_id=hf_repo_id,
            attacks=attack_names,
            negatives_dir=negatives,
        )
        if restored:
            print(f"HF restore complete for attacks: {restored}", flush=True)

    for attack in attack_names:
        zip_path = export_dir / f"{slug(method)}__{slug(attack)}.zip"
        if attack_is_complete(work_root, attack):
            print(
                f"\nSkipping {attack}: already complete (restored from HF or prior run).",
                flush=True,
            )
            entry = {
                "method": method,
                "attack": attack,
                "zip": str(zip_path) if zip_path.is_file() else "",
                "skipped": True,
            }
            append_manifest(export_dir, entry)
            entries.append(entry)
            continue

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
            output_dir=output_dir,
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
