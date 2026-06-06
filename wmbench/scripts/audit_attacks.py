from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from PIL import Image

from wmbench.attacks.registry import build_default_registry
from wmbench.attacks.distortion import _chain_steps
from wmbench.distortions.distortions import relative_strength_to_absolute, apply_single_distortion


def audit_image(image_path: Path, out_dir: Path, attacks: dict[str, object]):
    img = Image.open(image_path).convert("RGB")
    report: dict[str, dict] = {}
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, atk in attacks.items():
        entry: dict = {"type": getattr(atk, "_inner", "unknown"), "strengths": list(getattr(atk, "strengths", [])), "results": []}
        # Only audit distortion-based attacks that expose "_inner"
        if not hasattr(atk, "_inner"):
            entry["note"] = "Non-distortion attack; skipped"
            report[name] = entry
            continue

        chain = _chain_steps(atk._inner)
        # For each relative strength, compute absolute per-step and apply
        for sidx, strength in enumerate(atk.strengths):
            rel = float(strength)
            abs_params = []
            out_img = img.copy()
            for sid, (_, dtype) in enumerate(chain):
                abs_s = relative_strength_to_absolute(rel, dtype)
                abs_params.append({"step": sid, "distortion": dtype, "absolute": abs_s})
                out_img = apply_single_distortion(out_img, dtype, abs_s, distortion_seed=0 + sid)

            out_path = out_dir / f"{name.replace(' ', '_')}_s{sidx}.png"
            out_img.save(out_path)
            entry["results"].append({"relative": rel, "absolute_steps": abs_params, "output": str(out_path)})

        report[name] = entry

    # Save report
    report_path = out_dir / "audit_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Audit WMbench attack application for single-distortion attacks.")
    parser.add_argument("image", type=str, help="Sample image path to apply attacks to")
    parser.add_argument("--out-dir", type=str, default="attack_audit_outputs", help="Directory to write outputs and report")
    parser.add_argument("--attacks", type=str, nargs="*", default=None, help="Optional list of attack names to audit (defaults to all)")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    out_dir = Path(args.out_dir)

    registry = build_default_registry()
    if args.attacks:
        missing = [a for a in args.attacks if a not in registry]
        if missing:
            raise SystemExit(f"Unknown attack names: {missing}")
        selected = {k: registry[k] for k in args.attacks}
    else:
        selected = registry

    report_path = audit_image(image_path, out_dir, selected)
    print(f"Audit report written to: {report_path}")


if __name__ == "__main__":
    main()
