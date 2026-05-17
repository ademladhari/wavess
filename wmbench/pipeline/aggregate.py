from __future__ import annotations

import csv
import glob
import json
import os
from collections import defaultdict

from wmbench.metrics import aggregate as agg


def _write_results_raw(output_dir: str, raw_rows: list[dict]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "results_raw.csv")
    if not raw_rows:
        return
    fieldnames = [
        "method",
        "attack",
        "strength",
        "P",
        "Q",
        "PSNR",
        "SSIM",
        "NMI",
        "LPIPS",
        "FID",
        "CLIP_FID",
        "aesthetics_delta",
        "artifacts",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in raw_rows:
            w.writerow(r)


def _write_results_leaderboard(output_dir: str, rows: list[dict]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "results_leaderboard.csv")
    fieldnames = ["method", "attack", "rank", "Q@0.7P", "Q@0.4P", "Avg_P", "Avg_Q"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _build_leaderboard(
    by_cell: dict[tuple[str, str, str], dict],
    anchors: dict[str, tuple[float, float]],
    methods: list[str],
) -> list[dict]:
    grouped: dict[tuple[str, str], list[tuple[float, float, float]]] = defaultdict(list)
    for (method, attack, stren_s), cell in by_cell.items():
        if method not in methods:
            continue
        raw = {k: cell[k] for k in agg.METRIC_KEYS if k in cell}
        q = agg.q_from_raw_row(raw, anchors)
        p = float(cell.get("P", float("nan")))
        try:
            s = float(stren_s)
        except ValueError:
            s = 0.0
        grouped[(method, attack)].append((s, p, q))

    rows: list[dict] = []
    for (method, attack), triples in grouped.items():
        triples.sort(key=lambda t: t[0])
        sts = [t[0] for t in triples]
        ps = [t[1] for t in triples]
        qs = [t[2] for t in triples]
        avg_p = sum(ps) / len(ps) if ps else float("nan")
        avg_q = sum(qs) / len(qs) if qs else float("nan")
        q70 = agg.interp_q_at_p(sts, ps, qs, 0.7)
        q40 = agg.interp_q_at_p(sts, ps, qs, 0.4)
        rows.append(
            {
                "method": method,
                "attack": attack,
                "rank": 0,
                "Q@0.7P": q70,
                "Q@0.4P": q40,
                "Avg_P": avg_p,
                "Avg_Q": avg_q,
            }
        )

    neg_inf = float("-inf")

    def norm(x: float) -> float:
        if x != x:
            return neg_inf
        return x

    def sort_key(r: dict) -> tuple:
        return (norm(float(r["Q@0.7P"])), norm(float(r["Q@0.4P"])), norm(float(r["Avg_P"])), norm(float(r["Avg_Q"])))

    rows.sort(key=sort_key, reverse=True)
    for i, r in enumerate(rows, start=1):
        r["rank"] = i
    return rows


def run_aggregate_stage(work_parent: str, output_dir: str, methods: list[str]) -> None:
    by_cell: dict[tuple[str, str, str], dict] = {}
    for m in methods:
        base = os.path.join(work_parent, m, "metrics")
        for path in glob.glob(os.path.join(base, "*", "*", "metrics.json")):
            rel = os.path.relpath(path, os.path.join(work_parent, m))
            parts = rel.split(os.sep)
            if len(parts) >= 4 and parts[0] == "metrics":
                attack, stren = parts[1], parts[2]
                with open(path, encoding="utf-8") as f:
                    by_cell[(m, attack, stren)] = json.load(f)

    pool_lists: dict[str, list[float]] = defaultdict(list)
    for cell in by_cell.values():
        for k in agg.METRIC_KEYS:
            if k in cell:
                pool_lists[k].append(float(cell[k]))
    anchor_path = os.path.join(output_dir, "normalization_anchors.json")
    anchors = agg.load_or_compute_anchors(anchor_path, dict(pool_lists))

    raw_rows: list[dict] = []
    for (method, attack, stren_s), cell in sorted(by_cell.items()):
        raw_metric_cells = {k: cell[k] for k in agg.METRIC_KEYS if k in cell}
        q = agg.q_from_raw_row(raw_metric_cells, anchors)
        p = float(cell.get("P", float("nan")))
        row = {
            "method": method,
            "attack": attack,
            "strength": stren_s,
            "P": p,
            "Q": q,
            "PSNR": float("nan"),
            "SSIM": float("nan"),
            "NMI": float("nan"),
            "LPIPS": float("nan"),
            "FID": float("nan"),
            "CLIP_FID": float("nan"),
            "aesthetics_delta": float("nan"),
            "artifacts": float("nan"),
        }
        for k in agg.METRIC_KEYS:
            if k in cell:
                row[k] = float(cell[k])
        raw_rows.append(row)

    _write_results_raw(output_dir, raw_rows)
    lb = _build_leaderboard(by_cell, anchors, methods)
    _write_results_leaderboard(output_dir, lb)
