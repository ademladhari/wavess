from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from omegaconf import DictConfig, OmegaConf


def load_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> DictConfig:
    cfg = OmegaConf.load(str(path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(dict(overrides)))
    return cfg  # type: ignore[return-value]
