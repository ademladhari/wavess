from __future__ import annotations

import json
import os

from wmbench.attacks.base import Attack
from wmbench.attacks.distortion import build_combo_attack, build_single_distortion_attack
from wmbench.attacks.regeneration import DiffusionRegenAttack, Rinse2xDiffAttack, Rinse4xDiffAttack, VAERegenAttack

# WAVES encodes distortion severity as relative strength in [0, 1] passed through
# relative_strength_to_absolute (waves/distortions/distortions.py). No fixed grid file exists
# in-repo; default sweep matches WAVES provenance in run_benchmark.py (5 points, includes 0).
DEFAULT_RELATIVE_STRENGTHS: list[float] = [0.0, 0.25, 0.5, 0.75, 1.0]

# WAVES paper appendix (regeneration attacks): five evenly spaced strengths between min and max.
# umd-huang-lab/WAVES regeneration/regen.py — noise_step / CompressAI quality.
DEFAULT_REGEN_DIFFUSION_STRENGTHS: list[int] = [40, 80, 120, 160, 200]  # Regen-Diff: timesteps 40–200
DEFAULT_RINSE_2X_DIFFUSION_STRENGTHS: list[int] = [20, 40, 60, 80, 100]  # Rinse-2x: 20–100 per pass
DEFAULT_RINSE_4X_DIFFUSION_STRENGTHS: list[int] = [10, 20, 30, 40, 50]  # Rinse-4x: 10–50 per pass
DEFAULT_REGEN_VAE_STRENGTHS: list[int] = [1, 2, 4, 5, 7]  # Regen-VAE: quality 1–7 (5 evenly spaced)

MISSING_ATTACKS: tuple[str, ...] = (
    "Regen-DiffP",  # regen_diffusion_prompt: no implementation in waves/regeneration/regen.py
    "Regen-KLVAE",  # kl_vae regen: only name in waves/dev/constants.py
)


def ensure_missing_logged(output_dir: str, lines: list[str]) -> None:
    path = os.path.join(output_dir, "missing_components.txt")
    existing: set[str] = set()
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            existing = {ln.strip() for ln in f if ln.strip()}
    new_lines: list[str] = []
    for ln in lines:
        if ln not in existing:
            new_lines.append(ln)
            existing.add(ln)
    if new_lines:
        os.makedirs(output_dir, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            for ln in new_lines:
                f.write(ln + "\n")


def load_strength_overrides(path: str | None) -> dict[str, list[float | int]]:
    if not path:
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("strength config JSON must be an object: attack_name -> list of strengths")
    out: dict[str, list[float | int]] = {}
    for k, v in data.items():
        if not isinstance(v, list):
            raise ValueError(f"Strength list for {k!r} must be a JSON array")
        out[str(k)] = [float(x) if isinstance(x, (int, float)) else x for x in v]
    return out


def build_default_registry(
    *,
    diffusion_model_id: str = "CompVis/stable-diffusion-v1-4",
    vae_model_name: str = "bmshj2018-factorized",
    device=None,
) -> dict[str, Attack]:
    """Instantiated Attack objects for all implemented benchmark attacks."""
    import torch

    dev = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    rel = DEFAULT_RELATIVE_STRENGTHS
    regen_diff_steps = DEFAULT_REGEN_DIFFUSION_STRENGTHS
    rinse_2x_steps = DEFAULT_RINSE_2X_DIFFUSION_STRENGTHS
    rinse_4x_steps = DEFAULT_RINSE_4X_DIFFUSION_STRENGTHS
    vq = DEFAULT_REGEN_VAE_STRENGTHS
    shared_diff_pipe: object | None = None

    def get_shared_diff_pipe():
        nonlocal shared_diff_pipe
        if shared_diff_pipe is None:
            from wmbench.attacks.regeneration import _resd_pipeline_cls

            _RP = _resd_pipeline_cls()
            shared_diff_pipe = _RP.from_pretrained(
                diffusion_model_id,
                torch_dtype=torch.float16 if dev.type == "cuda" else torch.float32,
                revision="fp16" if dev.type == "cuda" else None,
            )
            shared_diff_pipe.set_progress_bar_config(disable=True)
            shared_diff_pipe.to(dev)
        return shared_diff_pipe

    attacks: dict[str, Attack] = {
        "Dist-Rotation": build_single_distortion_attack("Dist-Rotation", "rotation", rel),
        "Dist-RCrop": build_single_distortion_attack("Dist-RCrop", "resizedcrop", rel),
        "Dist-Erase": build_single_distortion_attack("Dist-Erase", "erasing", rel),
        "Dist-Bright": build_single_distortion_attack("Dist-Bright", "brightness", rel),
        "Dist-Contrast": build_single_distortion_attack("Dist-Contrast", "contrast", rel),
        "Dist-Blur": build_single_distortion_attack("Dist-Blur", "blurring", rel),
        "Dist-Noise": build_single_distortion_attack("Dist-Noise", "noise", rel),
        "Dist-JPEG": build_single_distortion_attack("Dist-JPEG", "jpeg", rel),
        "DistCom-Geo": build_combo_attack("DistCom-Geo", "combo_geometric", rel),
        "DistCom-Photo": build_combo_attack("DistCom-Photo", "combo_photometric", rel),
        "DistCom-Deg": build_combo_attack("DistCom-Deg", "combo_degradation", rel),
        "DistCom-All": build_combo_attack("DistCom-All", "combo_all", rel),
        "Regen-Diff": DiffusionRegenAttack(
            regen_diff_steps, diffusion_model_id, device=dev, pipe_provider=get_shared_diff_pipe
        ),
        "Regen-VAE": VAERegenAttack(vq, vae_model_name=vae_model_name, device=dev),
        "Rinse-2xDiff": Rinse2xDiffAttack(
            rinse_2x_steps, diffusion_model_id, device=dev, pipe_provider=get_shared_diff_pipe
        ),
        "Rinse-4xDiff": Rinse4xDiffAttack(
            rinse_4x_steps, diffusion_model_id, device=dev, pipe_provider=get_shared_diff_pipe
        ),
    }
    return attacks


def resolve_attacks(
    output_dir: str,
    attack_names: list[str] | None,
    *,
    diffusion_model_id: str = "CompVis/stable-diffusion-v1-4",
    vae_model_name: str = "bmshj2018-factorized",
    strength_config_path: str | None = None,
    device=None,
) -> dict[str, Attack]:
    registry = build_default_registry(
        diffusion_model_id=diffusion_model_id,
        vae_model_name=vae_model_name,
        device=device,
    )
    overrides = load_strength_overrides(strength_config_path)
    for name, strengths in overrides.items():
        if name not in registry:
            continue
        atk = registry[name]
        atk.strengths = strengths  # type: ignore[misc]

    missing_msgs: list[str] = []
    for m in MISSING_ATTACKS:
        missing_msgs.append(f"attack missing upstream implementation: {m}")

    if attack_names is not None:
        name_set = set(attack_names)
        for m in MISSING_ATTACKS:
            if m in name_set:
                ensure_missing_logged(output_dir, [f"attack missing upstream implementation: {m}"])
        filtered = {k: v for k, v in registry.items() if k in name_set}
        unknown = name_set - set(registry.keys()) - set(MISSING_ATTACKS)
        if unknown:
            raise ValueError(f"Unknown attack names (and not known-missing): {sorted(unknown)}")
        return filtered

    ensure_missing_logged(output_dir, missing_msgs)
    return registry
