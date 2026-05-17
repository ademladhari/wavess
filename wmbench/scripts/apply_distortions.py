#!/usr/bin/env python3
"""Apply distortion attacks to a flat source image directory without importing from waves/.

Creates result directories named attack-strength-<source>, ready for decode/metric scripts.
Regenerative attacks (regen_* and *x_regen*) are skipped—provide those from full WAVES pipeline.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import click
import dotenv
from PIL import Image
from tqdm.auto import tqdm

import wmbench.dev
import wmbench.distortions

dotenv.load_dotenv(override=False)

# Import after dotenv
from wmbench.dev import LIMIT
from wmbench.dev.constants import waves_non_adv_attack_keys
from wmbench.distortions.distortions import apply_single_distortion, relative_strength_to_absolute

_SINGLE_INNER = {
    "rotation": "rotation",
    "resizedcrop": "resizedcrop",
    "erasing": "erasing",
    "brightness": "brightness",
    "contrast": "contrast",
    "blurring": "blurring",
    "noise": "noise",
    "jpeg": "compression",
}


def _suffix_for_attack(attack_key: str) -> str:
    """Map distortion_single_* attack key to inner distortion type."""
    if not attack_key.startswith("distortion_single_"):
        raise ValueError(f"Not a distortion_single attack: {attack_key}")
    inner = attack_key.removeprefix("distortion_single_")
    if inner not in _SINGLE_INNER:
        raise ValueError(f"Unknown distortion_single subtype {attack_key}")
    return _SINGLE_INNER[inner]


def _chain_specs(attack_key: str, relative_strength: float, image_seed: int) -> list[tuple]:
    """Decompose attack_key into sequence of (kind, dtype, abs_strength, seed) tuples."""
    specs: list[tuple] = []
    rel = relative_strength

    def one(dtype: str) -> None:
        abs_strength = relative_strength_to_absolute(rel, dtype)
        specs.append(("single", dtype, abs_strength, image_seed))

    def combo(steps: list[str]) -> None:
        sid = image_seed
        for dtype in steps:
            abs_strength = relative_strength_to_absolute(rel, dtype)
            specs.append(("single", dtype, abs_strength, sid))
            sid += 1

    if attack_key.startswith("distortion_single_"):
        one(_suffix_for_attack(attack_key))
    elif attack_key == "distortion_combo_geometric":
        combo(["rotation", "resizedcrop"])
    elif attack_key == "distortion_combo_photometric":
        combo(["brightness", "contrast"])
    elif attack_key == "distortion_combo_degradation":
        combo(["blurring", "compression"])
    elif attack_key == "distortion_combo_all":
        combo(["rotation", "brightness", "noise", "blurring", "compression"])
    else:
        raise ValueError(f"Unsupported attack key for built-in distortions: {attack_key}")

    return specs


def apply_specs_to_image(
    src_pil: Image.Image,
    attack_key: str,
    relative_strength: float,
    image_seed: int,
) -> Image.Image:
    """Apply attack spec chain to an image."""
    out = src_pil.convert("RGB")
    for kind, dtype, strength, sid in _chain_specs(attack_key, relative_strength, image_seed):
        if kind != "single":
            raise AssertionError(f"Unexpected chain kind: {kind}")
        out = apply_single_distortion(
            out,
            dtype,
            strength=strength,
            distortion_seed=sid,
        )
        if out.mode != "RGB":
            out = out.convert("RGB")
    return out


@click.command()
@click.option(
    "--data-dir",
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help="Data directory containing source images subdirectory.",
)
@click.option(
    "--source-subdir",
    type=str,
    default="dct",
    help="Subdirectory under DATA_DIR with indexed PNG inputs (e.g. dct or real).",
)
@click.option(
    "--output-source-name",
    type=str,
    default=None,
    help=(
        "Source token for attacked folder names (default: same as --source-subdir). "
        "Example: use 'real_dct' with --source-subdir real for combined evaluations."
    ),
)
@click.option(
    "--attack-suite",
    type=str,
    default="17",
    help="Attack suite: '17' / '18' / 'all_non_adv'; only distortion_* keys are applied.",
)
@click.option("--relative-strength", type=float, default=0.5, show_default=True)
@click.option("--limit", type=int, default=None, help="Max image index (exclusive); defaults to WAVES_LIMIT.")
@click.option("--overwrite", is_flag=True, default=False, help="Overwrite existing output directories.")
def main(
    data_dir: str,
    source_subdir: str,
    output_source_name: str | None,
    attack_suite: str,
    relative_strength: float,
    limit: int | None,
    overwrite: bool,
):
    """Apply distortion attacks to flat source directory."""
    resolved_limit = limit if limit is not None else LIMIT
    attacks = waves_non_adv_attack_keys(attack_suite)
    distort_attacks = [a for a in attacks if a.startswith("distortion_")]
    skipped = sorted(set(attacks).difference(distort_attacks))

    click.echo(f"Applying {len(distort_attacks)} distortion attacks (limit={resolved_limit}).")
    if skipped:
        click.echo(f"Skipped (provide separately): {', '.join(skipped)}")

    src_root = os.path.join(data_dir, source_subdir)
    if not os.path.isdir(src_root):
        raise click.ClickException(f"Missing source folder: {src_root}")

    tag = output_source_name or source_subdir
    st_tag = relative_strength
    stem = float(f"{st_tag:.8g}")

    for attack_key in distort_attacks:
        out_dir = os.path.join(data_dir, f"{attack_key}-{stem}-{tag}")
        if os.path.isdir(out_dir) and not overwrite:
            click.echo(f"Skip existing {out_dir} (pass --overwrite)")
            continue

        os.makedirs(out_dir, exist_ok=True)
        iterator = tqdm(range(resolved_limit), desc=attack_key, unit="img")
        for idx in iterator:
            p = os.path.join(src_root, f"{idx}.png")
            if not os.path.isfile(p):
                continue
            pil = Image.open(p)
            attacked = apply_specs_to_image(pil, attack_key, relative_strength, idx).convert("RGB")
            attacked.save(os.path.join(out_dir, f"{idx}.png"))

    click.echo("Done.")


if __name__ == "__main__":
    main()
