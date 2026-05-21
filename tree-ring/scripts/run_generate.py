from __future__ import annotations

import argparse
import json
from pathlib import Path

from treering.config import load_config
from treering.fourier import circular_mask
from treering.generate import TreeRingGenerator
from treering.keygen import generate_key_material


def _load_prompts(path: str | None, fallback_count: int) -> list[str]:
    if path is None:
        return [f"A photorealistic landscape {idx}" for idx in range(fallback_count)]
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    prompts = [line.strip() for line in lines if line.strip()]
    if not prompts:
        raise ValueError("Prompt file has no non-empty lines.")
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate (watermarked or clean) images.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--prompts", default=None, help="Optional text file with one prompt per line.")
    parser.add_argument("--output-dir", default="outputs/generated")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Generation micro-batch size (use 1-2 on 8GB GPUs to avoid OOM).",
    )
    parser.add_argument("--watermarked", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be >= 1")

    cfg = load_config(args.config)
    prompts = _load_prompts(args.prompts, fallback_count=args.num_samples)[: args.num_samples]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = TreeRingGenerator(
        model_id=cfg.model.model_id,
        device=cfg.model.device,
        dtype=cfg.model.dtype,
        num_inference_steps=cfg.model.num_inference_steps,
        guidance_scale=cfg.model.guidance_scale,
    )

    key = None
    if args.watermarked:
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

    records = []
    for start in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[start : start + args.batch_size]
        # Keep deterministic outputs across chunks.
        batch_seed = int(cfg.watermark.seed) + start
        result = generator.generate(prompts=batch_prompts, key=key, seed=batch_seed)
        for i, (prompt, image) in enumerate(zip(result.prompts, result.images)):
            idx = start + i
            name = f"{'wm' if args.watermarked else 'clean'}_{idx:04d}.png"
            image_path = output_dir / name
            image.save(image_path)
            records.append(
                {
                    "index": idx,
                    "prompt": prompt,
                    "image_path": str(image_path),
                    "watermarked": args.watermarked,
                    "metadata": result.metadata,
                }
            )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Saved {len(records)} images to {output_dir}")


if __name__ == "__main__":
    main()
