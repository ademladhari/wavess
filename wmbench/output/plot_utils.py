"""Shared helpers for wmbench result plots (numeric strength ordering)."""

from __future__ import annotations

import os
from typing import Iterable


def strength_key(tag: str) -> float:
    """Sort folder/CSV strength tags numerically (40 < 80 < 100 < 120)."""
    try:
        return float(str(tag).strip())
    except ValueError:
        return float("inf")


def sort_by_strength(items: Iterable[dict], strength_field: str = "strength") -> list[dict]:
    return sorted(items, key=lambda r: strength_key(str(r.get(strength_field, ""))))


def sort_strength_tags(tags: Iterable[str]) -> list[str]:
    return sorted({str(t) for t in tags}, key=strength_key)


def safe_attack_filename(attack: str) -> str:
    return attack.replace(os.sep, "_").replace("/", "_")
