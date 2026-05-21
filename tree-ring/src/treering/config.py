from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ModelConfig:
    model_id: str
    device: str = "cuda"
    dtype: str = "float16"
    num_inference_steps: int = 50
    guidance_scale: float = 7.5


@dataclass(slots=True)
class WatermarkConfig:
    radius: int = 10
    key_variant: str = "rand"
    seed: int = 0
    channels: int = 4
    height: int = 64
    width: int = 64


@dataclass(slots=True)
class DetectionConfig:
    num_inversion_steps: int = 50
    threshold: float = 8.0
    alpha: float = 0.01
    invert_prompt: str = ""
    variance_estimation: str = "outside_mask"
    pvalue_tail: str = "lower"


@dataclass(slots=True)
class EvaluationConfig:
    output_dir: str = "outputs"
    metrics_json: str = "metrics.json"
    report_md: str = "report.md"
    roc_fpr_target: float = 0.01


@dataclass(slots=True)
class ExperimentConfig:
    model: ModelConfig
    watermark: WatermarkConfig
    detection: DetectionConfig
    evaluation: EvaluationConfig


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    section = data.get(name, {})
    if not isinstance(section, dict):
        raise ValueError(f"Config section '{name}' must be a mapping.")
    return section


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping.")

    model = ModelConfig(**_section(raw, "model"))
    watermark = WatermarkConfig(**_section(raw, "watermark"))
    detection = DetectionConfig(**_section(raw, "detection"))
    evaluation = EvaluationConfig(**_section(raw, "evaluation"))
    return ExperimentConfig(
        model=model,
        watermark=watermark,
        detection=detection,
        evaluation=evaluation,
    )
