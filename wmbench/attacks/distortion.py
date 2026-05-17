from __future__ import annotations

from PIL import Image

from wmbench.distortions.distortions import apply_single_distortion, relative_strength_to_absolute

from .base import Attack

# Combo chain matches waves/scripts/apply_distortion_attacks_flat.py::_chain_specs


def _chain_steps(attack_inner: str) -> list[tuple[str, str]]:
    """Return list of (kind, distortion_type) steps; kind is always 'single' inner dtype key."""
    mapping: dict[str, list[str]] = {
        "rotation": ["rotation"],
        "resizedcrop": ["resizedcrop"],
        "erasing": ["erasing"],
        "brightness": ["brightness"],
        "contrast": ["contrast"],
        "blurring": ["blurring"],
        "noise": ["noise"],
        "jpeg": ["compression"],
        "combo_geometric": ["rotation", "resizedcrop"],
        "combo_photometric": ["brightness", "contrast"],
        "combo_degradation": ["blurring", "compression"],
        "combo_all": ["rotation", "brightness", "noise", "blurring", "compression"],
    }
    keys = mapping[attack_inner]
    return [("single", k) for k in keys]


class _DistortionAttack(Attack):
    def __init__(self, name: str, inner: str, strengths: list[float]):
        self.name = name
        self._inner = inner
        self.strengths = strengths

    def apply(self, image: Image.Image, strength: float | int) -> Image.Image:
        rel = float(strength)
        out = image.convert("RGB")
        base_seed = 0
        for sid, (_, dtype) in enumerate(_chain_steps(self._inner)):
            abs_s = relative_strength_to_absolute(rel, dtype)
            out = apply_single_distortion(out, dtype, abs_s, distortion_seed=base_seed + sid)
        return out


def build_single_distortion_attack(display_name: str, inner: str, strengths: list[float]) -> Attack:
    return _DistortionAttack(display_name, inner, strengths)


def build_combo_attack(display_name: str, combo_key: str, strengths: list[float]) -> Attack:
    return _DistortionAttack(display_name, combo_key, strengths)
