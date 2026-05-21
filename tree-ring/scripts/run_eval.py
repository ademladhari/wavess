from __future__ import annotations

import argparse
import json
from pathlib import Path

from treering.config import load_config
from treering.evaluate import compute_roc_metrics


def _read_results(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list in detection result file: {path}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Tree-Ring detection quality.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--watermarked-results", required=True)
    parser.add_argument("--clean-results", required=True)
    parser.add_argument("--output-dir", default="outputs/eval")
    args = parser.parse_args()

    cfg = load_config(args.config)
    wm_rows = _read_results(args.watermarked_results)
    clean_rows = _read_results(args.clean_results)

    wm_distance_scores = [-float(row["distance"]) for row in wm_rows]
    clean_distance_scores = [-float(row["distance"]) for row in clean_rows]
    distance_metrics = compute_roc_metrics(
        positive_scores=wm_distance_scores,
        negative_scores=clean_distance_scores,
        target_fpr=cfg.evaluation.roc_fpr_target,
    )

    wm_p_scores = [-float(row["p_value"]) for row in wm_rows]
    clean_p_scores = [-float(row["p_value"]) for row in clean_rows]
    pvalue_metrics = compute_roc_metrics(
        positive_scores=wm_p_scores,
        negative_scores=clean_p_scores,
        target_fpr=cfg.evaluation.roc_fpr_target,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_payload = {
        "distance": {
            "auc": distance_metrics.auc,
            "tpr_at_target_fpr": distance_metrics.tpr_at_target_fpr,
            "target_fpr": distance_metrics.target_fpr,
        },
        "p_value": {
            "auc": pvalue_metrics.auc,
            "tpr_at_target_fpr": pvalue_metrics.tpr_at_target_fpr,
            "target_fpr": pvalue_metrics.target_fpr,
        },
        "num_watermarked": len(wm_rows),
        "num_clean": len(clean_rows),
    }

    metrics_path = output_dir / cfg.evaluation.metrics_json
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    report = (
        "# Tree-Ring Evaluation Report\n\n"
        f"- Watermarked samples: {len(wm_rows)}\n"
        f"- Clean samples: {len(clean_rows)}\n"
        f"- Distance AUC: {distance_metrics.auc:.4f}\n"
        f"- Distance TPR@{distance_metrics.target_fpr:.2%}FPR: "
        f"{distance_metrics.tpr_at_target_fpr:.4f}\n"
        f"- P-value AUC: {pvalue_metrics.auc:.4f}\n"
        f"- P-value TPR@{pvalue_metrics.target_fpr:.2%}FPR: "
        f"{pvalue_metrics.tpr_at_target_fpr:.4f}\n"
    )
    report_path = output_dir / cfg.evaluation.report_md
    report_path.write_text(report, encoding="utf-8")
    print(f"Wrote metrics to {metrics_path} and report to {report_path}")


if __name__ == "__main__":
    main()
