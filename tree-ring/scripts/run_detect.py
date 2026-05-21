from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from treering.attacks import apply_attack
from treering.config import load_config
from treering.detect import detect_watermark
from treering.fourier import circular_mask
from treering.invert import DDIMInverter
from treering.keygen import generate_key_material


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Tree-Ring detection on generated images.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--manifest", required=True, help="Path to generation manifest.json")
    parser.add_argument("--output", default="outputs/detection_results.json")
    parser.add_argument("--attack", default="none")
    parser.add_argument("--attack-args", default="{}", help="JSON object with attack kwargs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if not isinstance(manifest, list):
        raise ValueError("Manifest must contain a list of image records.")

    mask = circular_mask(
        cfg.watermark.height,
        cfg.watermark.width,
        cfg.watermark.radius,
        device=cfg.model.device,
    )
    key = generate_key_material(
        channels=cfg.watermark.channels,
        height=cfg.watermark.height,
        width=cfg.watermark.width,
        mask=mask,
        variant=cfg.watermark.key_variant,  # type: ignore[arg-type]
        seed=cfg.watermark.seed,
        device=cfg.model.device,
    )

    inverter = DDIMInverter(
        model_id=cfg.model.model_id,
        device=cfg.model.device,
        dtype=torch.float32,
    )
    attack_args = json.loads(args.attack_args)

    records = []
    for row in manifest:
        image_path = row["image_path"]
        image = Image.open(image_path).convert("RGB")
        image = apply_attack(image, args.attack, **attack_args)
        inverted = inverter.invert(
            images=[image],
            invert_prompt=cfg.detection.invert_prompt,
            num_inversion_steps=cfg.detection.num_inversion_steps,
        )
        detection = detect_watermark(
            inverted_latents=inverted,
            key=key,
            threshold=cfg.detection.threshold,
            alpha=cfg.detection.alpha,
            variance_estimation=cfg.detection.variance_estimation,
            pvalue_tail=cfg.detection.pvalue_tail,
        )[0]
        records.append(
            {
                "image_path": image_path,
                "watermarked_label": bool(row.get("watermarked", False)),
                "attack": args.attack,
                "distance": detection.distance,
                "p_value": detection.p_value,
                "detected_by_distance": detection.detected_by_distance,
                "detected_by_pvalue": detection.detected_by_pvalue,
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Wrote detection results for {len(records)} images to {output_path}")


if __name__ == "__main__":
    main()
