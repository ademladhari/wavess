#!/usr/bin/env python3
"""Metric computation runner for wmbench.

Computes image quality metrics (PSNR, SSIM, LPIPS, FID, CLIP-FID, aesthetics, etc.)
on result directories without importing from waves/, dct/, or dwt/.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import click
import torch
import warnings
from PIL import Image
from tqdm.auto import tqdm

import dotenv

from wmbench.dev import (
    LIMIT,
    SUBSET_LIMIT,
    QUALITY_METRICS,
    get_all_image_dir_paths,
    check_file_existence,
    existence_operation,
    existence_to_indices,
    parse_image_dir_path,
    save_json,
    load_json,
)
from wmbench.metrics import (
    compute_fid,
    compute_image_distance_repeated,
    load_perceptual_models,
    compute_perceptual_metric_repeated,
    load_aesthetics_and_artifacts_models,
    compute_aesthetics_and_artifacts_scores,
    load_open_clip_model_preprocess_and_tokenizer,
    compute_clip_score,
)

dotenv.load_dotenv(override=False)
warnings.filterwarnings("ignore")

DELTA_METRICS = ["aesthetics", "artifacts", "clip_score"]


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "1" if default else "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _result_stem(path: str) -> tuple[str, str]:
    dataset_name, attack_name, attack_strength, source_name = parse_image_dir_path(path, quiet=True)
    if attack_name is None or attack_strength is None:
        return dataset_name, source_name
    return dataset_name, f"{attack_name}-{attack_strength}-{source_name}"


def get_indices(
    mode: str,
    path: str,
    clean_path: str,
    attacked_path: Optional[str],
    quiet: bool,
    subset: bool,
    limit: int,
    subset_limit: int,
) -> list[int]:
    """Determine which indices need metric computation."""
    dataset_name, stem = _result_stem(path)
    json_path = os.path.join(
        os.environ.get("RESULT_DIR", "."),
        dataset_name,
        f"{stem}-metric.json",
    )

    # Check if metrics already computed
    if os.path.exists(json_path) and (data := load_json(json_path)) is not None:
        if mode != "aesthetics_and_artifacts":
            measured_existences = [data.get(str(i), {}).get(mode) is not None for i in range(limit)]
        else:
            measured_existences = [
                (data.get(str(i), {}).get("aesthetics") is not None)
                and (data.get(str(i), {}).get("artifacts") is not None)
                for i in range(limit)
            ]
        if (not subset and sum(measured_existences) == limit) or (
            subset and sum(measured_existences[:subset_limit]) == subset_limit
        ):
            return []

    clean_image_existences = check_file_existence(
        clean_path, name_pattern="{}.png", limit=limit
    )
    attacked_image_existences = (
        check_file_existence(attacked_path, name_pattern="{}.png", limit=limit)
        if attacked_path is not None
        else [True] * limit
    )

    if mode.endswith("_fid") and sum(clean_image_existences) != limit:
        raise ValueError(f"Cannot compute FID if not all {limit} clean images exist")

    if not quiet:
        if attacked_path is None:
            print(f"Found {sum(clean_image_existences)} not-attacked images")
        else:
            print(
                f"Found {sum(attacked_image_existences)} attacked images and "
                f"{sum(clean_image_existences)} corresponding not-attacked images"
            )

    existences = existence_operation(
        clean_image_existences, attacked_image_existences, op="union"
    )
    if os.path.exists(json_path):
        existences = existence_operation(
            existences, measured_existences, op="difference"
        )

    indices = existence_to_indices(
        existences,
        limit=limit if not subset else subset_limit,
    )
    return indices


def process_single(
    mode: str,
    indices: list[int],
    path: str,
    clean_path: str,
    attacked_path: Optional[str],
    quiet: bool,
    limit: int,
) -> dict[int, float | tuple[float, float]]:
    """Compute metric for a single mode on given indices."""

    if mode.endswith("_fid"):
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            if not quiet:
                print("No GPUs available; attempting CPU device for FID")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
            if not quiet:
                print(f"Using {num_gpus} GPUs for FID")

        metric = float(
            compute_fid(
                clean_path,
                attacked_path,
                mode=mode.split("_")[0],
                device=device,
                batch_size=64,
                num_workers=1,
                verbose=not quiet,
            )
        )
        return {
            idx: metric
            for idx in existence_to_indices(
                check_file_existence(attacked_path, name_pattern="{}.png", limit=limit),
                limit=limit,
            )
        }

    elif mode in ["psnr", "ssim", "nmi"]:
        if not quiet:
            print(f"Using {os.cpu_count()} CPUs for {mode}")
        clean_images = [
            Image.open(os.path.join(clean_path, f"{idx}.png")).convert("RGB") for idx in indices
        ]
        attacked_images = [
            Image.open(os.path.join(attacked_path, f"{idx}.png")).convert("RGB") for idx in indices
        ]
        metrics = compute_image_distance_repeated(
            clean_images,
            attacked_images,
            metric_name=mode,
            num_workers=8,
            verbose=not quiet,
        )
        results = {idx: metric for idx, metric in zip(indices, metrics)}
        for img in clean_images + attacked_images:
            img.close()
        return results

    elif mode in ["lpips", "watson"]:
        if not quiet:
            num_gpus = torch.cuda.device_count()
            print(f"Using {num_gpus} GPU(s) and {os.cpu_count()} CPUs for {mode}")

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        clean_images = []
        attacked_images = []
        for idx in indices:
            clean_image = Image.open(os.path.join(clean_path, f"{idx}.png")).convert("RGB")
            attacked_image = Image.open(os.path.join(attacked_path, f"{idx}.png")).convert("RGB")
            if attacked_image.size != clean_image.size:
                attacked_image = attacked_image.resize(clean_image.size, Image.Resampling.BICUBIC)
            clean_images.append(clean_image)
            attacked_images.append(attacked_image)

        model = load_perceptual_models(
            mode,
            mode="alex" if mode == "lpips" else "dft",
            device=device,
        )

        batch_size = 1
        pbar = tqdm(
            total=len(indices),
            desc=f"Computing {mode} metrics on images",
            unit="image",
            disable=quiet,
        )
        metrics = []
        for it in range(0, len(indices), batch_size):
            metrics.extend(
                compute_perceptual_metric_repeated(
                    clean_images[it : min(it + batch_size, len(indices))],
                    attacked_images[it : min(it + batch_size, len(indices))],
                    metric_name=mode,
                    mode="alex" if mode == "lpips" else "dft",
                    model=model,
                    device=device,
                )
            )
            pbar.update(min(batch_size, len(indices) - it))
        pbar.close()

        results = {idx: metric for idx, metric in zip(indices, metrics)}
        for img in clean_images + attacked_images:
            img.close()
        return results

    elif mode == "aesthetics_and_artifacts":
        if not quiet:
            num_gpus = torch.cuda.device_count()
            print(f"Using {num_gpus} GPU(s) and {os.cpu_count()} CPUs for {mode}")

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = load_aesthetics_and_artifacts_models(device=device)
        images = [Image.open(os.path.join(path, f"{idx}.png")) for idx in indices]

        batch_size = 8
        pbar = tqdm(
            total=len(indices),
            desc=f"Computing {mode} metrics on images",
            unit="image",
            disable=quiet,
        )
        metrics = []
        for it in range(0, len(indices), batch_size):
            aesthetics, artifacts = compute_aesthetics_and_artifacts_scores(
                images[it : min(it + batch_size, len(indices))],
                model,
                device=device,
            )
            metrics.extend(list(zip(aesthetics, artifacts)))
            pbar.update(min(batch_size, len(indices) - it))
        pbar.close()

        results = {idx: metric for idx, metric in zip(indices, metrics)}
        for img in images:
            img.close()
        return results

    elif mode == "clip_score":
        if not quiet:
            num_gpus = torch.cuda.device_count()
            print(f"Using {num_gpus} GPU(s) and {os.cpu_count()} CPUs for {mode}")

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = load_open_clip_model_preprocess_and_tokenizer(device=device)
        images = [Image.open(os.path.join(path, f"{idx}.png")) for idx in indices]

        dataset_name, _ = _result_stem(path)
        prompts_path = os.path.join(
            os.environ.get("RESULT_DIR", "."),
            dataset_name,
            "prompts.json",
        )
        if not os.path.exists(prompts_path):
            raise FileNotFoundError(f"Prompts file not found: {prompts_path}")

        prompts = list(load_json(prompts_path).values())
        prompts = [prompts[idx] for idx in indices]

        batch_size = 8
        pbar = tqdm(
            total=len(indices),
            desc=f"Computing {mode} metrics on images",
            unit="image",
            disable=quiet,
        )
        metrics = []
        for it in range(0, len(indices), batch_size):
            metrics.extend(
                compute_clip_score(
                    images[it : min(it + batch_size, len(indices))],
                    prompts[it : min(it + batch_size, len(indices))],
                    model,
                    device=device,
                )
            )
            pbar.update(min(batch_size, len(indices) - it))
        pbar.close()

        results = {idx: metric for idx, metric in zip(indices, metrics)}
        for img in images:
            img.close()
        return results

    else:
        raise ValueError(f"Unknown metric mode: {mode}")


def save_results(
    mode: str,
    results: dict[int, float | tuple[float, float]],
    path: str,
    delta_clean_path: Optional[str] = None,
) -> None:
    """Save metric results to JSON."""
    dataset_name, stem = _result_stem(path)
    json_path = os.path.join(
        os.environ.get("RESULT_DIR", "."),
        dataset_name,
        f"{stem}-metric.json",
    )

    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    # Load existing data if present
    if os.path.exists(json_path) and (data := load_json(json_path)) is not None:
        pass
    else:
        data = {}

    # Parse results
    for idx, value in results.items():
        idx_key = str(idx)
        if idx_key not in data:
            data[idx_key] = {}

        if mode in DELTA_METRICS and delta_clean_path is not None:
            # Compute delta (attacked - clean)
            clean_image = Image.open(os.path.join(delta_clean_path, f"{idx}.png")).convert("RGB")
            attacked_path = os.path.dirname(os.path.join(path, "dummy.png"))
            attacked_image = Image.open(os.path.join(attacked_path, f"{idx}.png")).convert("RGB")

            if mode == "clip_score":
                data[idx_key]["clip_score"] = value
            elif mode == "aesthetics_and_artifacts":
                data[idx_key]["aesthetics"] = value[0]
                data[idx_key]["artifacts"] = value[1]
            else:
                data[idx_key][mode] = value
            attacked_image.close()
            clean_image.close()
        else:
            if mode == "aesthetics_and_artifacts":
                data[idx_key]["aesthetics"] = value[0]
                data[idx_key]["artifacts"] = value[1]
            else:
                data[idx_key][mode] = value

    save_json(data, json_path)


@click.command()
@click.argument("mode", type=str)
@click.option("--result-dir", default=None, help="Result directory override")
@click.option("--data-dir", default=None, help="Dataset directory override")
@click.option("--limit", type=int, default=None, help="Image count limit")
@click.option("--subset", is_flag=True, help="Compute only first subset_limit images")
@click.option("--quiet", is_flag=True, help="Suppress progress output")
@click.option("--single-progress", is_flag=True, help="Show one overall progress bar")
@click.option("--include-clean", is_flag=True, help="Also compute metrics for clean/source dirs")
def main(
    mode: str,
    result_dir: Optional[str],
    data_dir: Optional[str],
    limit: Optional[int],
    subset: bool,
    quiet: bool,
    single_progress: bool,
    include_clean: bool,
):
    """Compute metrics on result directories.

    Mode options: psnr, ssim, nmi, lpips, watson, aesthetics_and_artifacts, clip_score,
                  legacy_fid, clean_fid, clip_fid
    """
    if result_dir is not None:
        os.environ["RESULT_DIR"] = result_dir
    if data_dir is not None:
        os.environ["DATA_DIR"] = data_dir

    result_dir = os.environ.get("RESULT_DIR", ".")
    limit = limit or LIMIT
    subset_limit = SUBSET_LIMIT

    if mode == "aesthetics_and_artifacts":
        pass
    elif mode not in QUALITY_METRICS and not mode.endswith("_fid"):
        print(f"Unknown mode: {mode}", file=sys.stderr)
        sys.exit(1)

    paths = get_all_image_dir_paths()
    if not paths:
        print("No image directories found", file=sys.stderr)
        sys.exit(1)

    progress = None
    if single_progress and not quiet:
        total_images = 0
        for (_, attack_name, _, _), path in paths.items():
            if attack_name is None:
                continue
            clean_path = os.path.join(os.path.dirname(path), "images")
            if not os.path.exists(clean_path):
                continue
            count = subset_limit if subset else limit
            total_images += count
        progress = tqdm(total=total_images, desc=f"{mode} (all attacks)", unit="image")

    for (dataset_name, attack_name, attack_strength, source_name), path in paths.items():
        try:
            if attack_name is None and not include_clean:
                # This is clean/reference data; skip unless requested
                continue

            clean_path = os.path.join(os.path.dirname(path), "images")
            if not os.path.exists(clean_path):
                if not quiet:
                    print(f"Warning: No clean images at {clean_path}, skipping")
                continue

            attacked_path = path
            indices = get_indices(
                mode,
                path,
                clean_path,
                attacked_path,
                quiet,
                subset,
                limit,
                subset_limit,
            )

            if not indices:
                if not quiet:
                    print(f"All metrics already computed for {path}")
                continue

            results = process_single(
                mode,
                indices,
                    path,
                    clean_path,
                    attacked_path,
                quiet,
                limit,
            )
            if progress is not None:
                progress.update(len(indices))
            save_results(mode, results, path, clean_path)

            if not quiet:
                print(f"Saved metrics to {_result_stem(path)}")

        except Exception as e:
            print(f"Error processing {path}: {e}", file=sys.stderr)
            if not quiet:
                import traceback
                traceback.print_exc()
            continue

    if progress is not None:
        progress.close()


if __name__ == "__main__":
    main()
