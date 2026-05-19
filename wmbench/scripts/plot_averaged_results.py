#!/usr/bin/env python3
"""Plot results_averaged.csv and results_raw.csv → PNG charts (numeric strength sort)."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

_pkg_parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _pkg_parent not in sys.path:
    sys.path.insert(0, _pkg_parent)

from wmbench.output.plot_utils import safe_attack_filename, sort_by_strength, strength_key


def _read_csv(path: str) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _f(row: dict, key: str) -> float:
    try:
        v = float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")
    if math.isnan(v) or math.isinf(v):
        return float("nan")
    return v


def _group_raw_by_attack(raw_rows: list[dict], method: str) -> dict[str, list[dict]]:
    by_attack: dict[str, list[dict]] = {}
    for r in raw_rows:
        if r.get("method") != method:
            continue
        by_attack.setdefault(r["attack"], []).append(r)
    for atk in by_attack:
        by_attack[atk] = sort_by_strength(by_attack[atk])
    return by_attack


def _sort_by_metric(rows: list[dict], col: str, ascending: bool = True) -> list[dict]:
    return sorted(
        rows,
        key=lambda r: _f(r, col)
        if not math.isnan(_f(r, col))
        else (float("inf") if ascending else float("-inf")),
        reverse=not ascending,
    )


def _hbar(ax, labels: list[str], values: list[float], title: str, xlabel: str, color: str = "#2563eb") -> None:
    y = np.arange(len(labels))
    ax.barh(y, values, color=color, height=0.72)
    ax.set_yticks(y, labels, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)


def plot_averaged_bars(rows: list[dict], out_dir: str, method: str) -> None:
    metrics = [
        ("avg_P", "Avg detection P (TPR@0.1% FPR)", "Higher = watermark still detected", "#059669", True),
        ("avg_PSNR", "Avg PSNR (dB)", "Higher = closer to original", "#2563eb", False),
        ("avg_SSIM", "Avg SSIM", "Higher = closer to original", "#7c3aed", False),
        ("avg_LPIPS", "Avg LPIPS", "Higher = more perceptual change", "#dc2626", False),
        ("avg_FID", "Avg FID", "Higher = more distributional change", "#ea580c", False),
        ("avg_CLIP_FID", "Avg CLIP-FID", "Higher = more semantic drift", "#ca8a04", False),
        ("avg_Q", "Avg combined quality Q", "Higher = better normalized quality", "#0891b2", False),
    ]
    n_str = rows[0].get("n_strengths", "?") if rows else "?"
    for col, title, subtitle, color, asc_low in metrics:
        if col == "avg_P":
            sorted_rows = _sort_by_metric(rows, col, ascending=True)
        elif col in ("avg_LPIPS", "avg_FID", "avg_CLIP_FID"):
            sorted_rows = _sort_by_metric(rows, col, ascending=False)
        else:
            sorted_rows = _sort_by_metric(rows, col, ascending=False)
        labels = [r["attack"] for r in sorted_rows]
        vals = [_f(r, col) for r in sorted_rows]
        fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(labels))))
        _hbar(ax, labels, vals, f"{method} — {title}", subtitle, color=color)
        fig.text(
            0.5,
            0.01,
            f"Source: results_averaged.csv · mean over {n_str} strengths",
            ha="center",
            fontsize=7,
            color="#666",
        )
        fig.tight_layout(rect=[0, 0.03, 1, 1])
        safe = col.replace("avg_", "")
        fig.savefig(os.path.join(out_dir, f"bar_{safe}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_p_vs_psnr(rows: list[dict], out_dir: str, method: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    for r in rows:
        x, y = _f(r, "avg_PSNR"), _f(r, "avg_P")
        if math.isnan(x) or math.isnan(y):
            continue
        ax.scatter(x, y, s=80, alpha=0.85)
        ax.annotate(r["attack"], (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xlabel("Avg PSNR (dB) — image fidelity")
    ax.set_ylabel("Avg P — detection (TPR@0.1% FPR)")
    ax.set_title(f"{method}: detection vs fidelity (per attack, averaged over strengths)")
    ax.grid(alpha=0.3)
    ax.axhline(0.7, color="#888", linestyle=":", alpha=0.6, label="P = 0.7")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "scatter_P_vs_PSNR.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_p_vs_q(rows: list[dict], out_dir: str, method: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    for r in rows:
        x, y = _f(r, "avg_Q"), _f(r, "avg_P")
        if math.isnan(x) or math.isnan(y):
            continue
        ax.scatter(x, y, s=80, alpha=0.85, c="#2563eb")
        ax.annotate(r["attack"], (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xlabel("Avg Q — normalized combined quality")
    ax.set_ylabel("Avg P — detection (TPR@0.1% FPR)")
    ax.set_title(f"{method}: detection vs combined quality Q (averaged)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "scatter_P_vs_Q.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(rows: list[dict], out_dir: str, method: str) -> None:
    cols = ["avg_P", "avg_PSNR", "avg_SSIM", "avg_LPIPS", "avg_FID", "avg_CLIP_FID", "avg_Q"]
    labels = [r["attack"] for r in rows]
    mat = np.array([[_f(r, c) for c in cols] for r in rows], dtype=float)
    norm = np.zeros_like(mat)
    for j in range(mat.shape[1]):
        col = mat[:, j]
        ok = ~np.isnan(col)
        if not ok.any():
            continue
        lo, hi = np.nanmin(col), np.nanmax(col)
        if hi - lo < 1e-12:
            norm[ok, j] = 0.5
        else:
            norm[ok, j] = (col[ok] - lo) / (hi - lo)
    fig, ax = plt.subplots(figsize=(9, max(5, 0.35 * len(labels))))
    im = ax.imshow(norm, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)), [c.replace("avg_", "") for c in cols], rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_title(f"{method}: metrics heatmap (column-wise min–max, averaged)")
    plt.colorbar(im, ax=ax, label="normalized within column")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "heatmap_metrics.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_overview_combined(rows: list[dict], out_dir: str, method: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    attacks = [r["attack"] for r in rows]
    x = np.arange(len(attacks))
    axes[0, 0].bar(x, [_f(r, "avg_P") for r in rows], color="#059669")
    axes[0, 0].set_xticks(x, attacks, rotation=55, ha="right", fontsize=7)
    axes[0, 0].set_ylabel("Avg P")
    axes[0, 0].set_title("Detection (higher = more robust)")
    axes[0, 0].grid(axis="y", alpha=0.25)
    axes[0, 1].bar(x, [_f(r, "avg_PSNR") for r in rows], color="#2563eb")
    axes[0, 1].set_xticks(x, attacks, rotation=55, ha="right", fontsize=7)
    axes[0, 1].set_ylabel("Avg PSNR (dB)")
    axes[0, 1].set_title("Fidelity (higher = less damage)")
    axes[0, 1].grid(axis="y", alpha=0.25)
    axes[1, 0].bar(x, [_f(r, "avg_SSIM") for r in rows], color="#7c3aed")
    axes[1, 0].set_xticks(x, attacks, rotation=55, ha="right", fontsize=7)
    axes[1, 0].set_ylabel("Avg SSIM")
    axes[1, 0].set_title("Structural similarity")
    axes[1, 0].grid(axis="y", alpha=0.25)
    axes[1, 1].bar(x, [_f(r, "avg_LPIPS") for r in rows], color="#dc2626")
    axes[1, 1].set_xticks(x, attacks, rotation=55, ha="right", fontsize=7)
    axes[1, 1].set_ylabel("Avg LPIPS")
    axes[1, 1].set_title("Perceptual distance")
    axes[1, 1].grid(axis="y", alpha=0.25)
    fig.suptitle(f"{method} benchmark — averaged over all attack strengths", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "overview_4panel.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_raw_strength_overviews(by_attack: dict[str, list[dict]], out_dir: str, method: str) -> None:
    """All attacks: strength vs Q and strength vs P (numeric sort)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for attack, items in sorted(by_attack.items()):
        xs = [strength_key(str(x["strength"])) for x in items]
        qs = [_f(x, "Q") for x in items]
        ps = [_f(x, "P") for x in items]
        axes[0].plot(xs, qs, "o-", label=attack, alpha=0.85, markersize=4)
        axes[1].plot(xs, ps, "o-", label=attack, alpha=0.85, markersize=4)
    axes[0].set_xlabel("Attack strength")
    axes[0].set_ylabel("Q")
    axes[0].set_title(f"{method}: quality vs strength (raw)")
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("Attack strength")
    axes[1].set_ylabel("P (TPR@0.1% FPR)")
    axes[1].axhline(0.7, color="#888", linestyle=":", alpha=0.5)
    axes[1].set_title(f"{method}: detection vs strength (raw)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=6)
    fig.suptitle("Points connected in increasing numeric strength", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "raw_overview_strength_vs_P_Q.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_raw_p_vs_q_overview(by_attack: dict[str, list[dict]], out_dir: str, method: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for attack, items in sorted(by_attack.items()):
        ps = [_f(x, "P") for x in items]
        qs = [_f(x, "Q") for x in items]
        if not ps:
            continue
        ax.plot(ps, qs, "o-", label=attack, alpha=0.85, markersize=4)
    ax.axvline(0.7, color="k", linestyle=":", alpha=0.4)
    ax.axvline(0.95, color="k", linestyle="--", alpha=0.4)
    ax.set_xlabel("P (TPR@0.1% FPR)")
    ax.set_ylabel("Q")
    ax.set_title(f"{method}: P vs Q (raw, strength-sorted polylines)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "raw_overview_P_vs_Q.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_raw_heatmaps(by_attack: dict[str, list[dict]], out_dir: str, method: str) -> None:
    attacks = sorted(by_attack)
    all_strengths = sorted(
        {strength_key(str(r["strength"])) for items in by_attack.values() for r in items},
        key=strength_key,
    )
    if not attacks or not all_strengths:
        return
    st_labels = [str(s) for s in all_strengths]

    def _matrix(metric: str) -> np.ndarray:
        mat = np.full((len(attacks), len(all_strengths)), np.nan)
        for i, atk in enumerate(attacks):
            lookup = {strength_key(str(r["strength"])): _f(r, metric) for r in by_attack[atk]}
            for j, s in enumerate(all_strengths):
                if s in lookup:
                    mat[i, j] = lookup[s]
        return mat

    for metric, title, cmap in (
        ("P", "Detection P", "RdYlGn"),
        ("Q", "Combined quality Q", "viridis"),
        ("PSNR", "PSNR (dB)", "plasma"),
    ):
        mat = _matrix(metric)
        fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(st_labels)), max(5, 0.35 * len(attacks))))
        im = ax.imshow(mat, aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(st_labels)), st_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(attacks)), attacks, fontsize=8)
        ax.set_title(f"{method}: {title} by attack × strength")
        plt.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"heatmap_raw_{metric}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_raw_curves(by_attack: dict[str, list[dict]], out_dir: str, method: str) -> None:
    curves_dir = os.path.join(out_dir, "curves_by_strength")
    os.makedirs(curves_dir, exist_ok=True)
    for attack, items in sorted(by_attack.items()):
        xs = [strength_key(str(x["strength"])) for x in items]
        ps = [_f(x, "P") for x in items]
        qs = [_f(x, "Q") for x in items]
        psnrs = [_f(x, "PSNR") for x in items]
        lp = [_f(x, "LPIPS") for x in items]
        tags = [str(x["strength"]) for x in items]

        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        axes[0, 0].plot(xs, ps, "o-", color="#059669")
        axes[0, 0].axhline(0.7, color="#888", linestyle=":", alpha=0.5)
        axes[0, 0].set_xlabel("Strength")
        axes[0, 0].set_ylabel("P")
        axes[0, 0].set_title("Detection")
        axes[0, 0].grid(alpha=0.3)

        axes[0, 1].plot(xs, qs, "o-", color="#0891b2")
        axes[0, 1].set_xlabel("Strength")
        axes[0, 1].set_ylabel("Q")
        axes[0, 1].set_title("Combined quality")
        axes[0, 1].grid(alpha=0.3)

        axes[1, 0].plot(xs, psnrs, "o-", color="#2563eb")
        axes[1, 0].set_xlabel("Strength")
        axes[1, 0].set_ylabel("PSNR (dB)")
        axes[1, 0].set_title("Fidelity")
        axes[1, 0].grid(alpha=0.3)

        axes[1, 1].plot(ps, qs, "o-", color="#7c3aed")
        for p, q, t in zip(ps, qs, tags):
            axes[1, 1].annotate(t, (p, q), textcoords="offset points", xytext=(3, 3), fontsize=7)
        axes[1, 1].axvline(0.7, color="#888", linestyle=":", alpha=0.4)
        axes[1, 1].set_xlabel("P")
        axes[1, 1].set_ylabel("Q")
        axes[1, 1].set_title("P vs Q (strength order)")
        axes[1, 1].grid(alpha=0.3)

        fig.suptitle(f"{method} / {attack}", fontsize=11, fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(curves_dir, f"{safe_attack_filename(attack)}.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(xs, lp, "o-", color="#dc2626")
        ax2.set_xlabel("Strength")
        ax2.set_ylabel("LPIPS")
        ax2.set_title(f"{attack} — LPIPS vs strength")
        ax2.grid(alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(os.path.join(curves_dir, f"{safe_attack_filename(attack)}_LPIPS.png"), dpi=120, bbox_inches="tight")
        plt.close(fig2)


def plot_leaderboard_bars(results_dir: str, out_dir: str, method: str) -> None:
    lb_path = os.path.join(results_dir, "results_leaderboard.csv")
    if not os.path.isfile(lb_path):
        return
    rows = [r for r in _read_csv(lb_path) if r.get("method") == method]
    if not rows:
        return
    rows.sort(key=lambda r: int(r.get("rank", 999)))
    attacks = [r["attack"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, 0.35 * len(rows))))
    y = np.arange(len(rows))

    def _plot_col(ax, col: str, title: str, color: str) -> None:
        vals = []
        for r in rows:
            v = _f(r, col)
            vals.append(0.0 if math.isnan(v) else (1.05 if math.isinf(v) and v > 0 else (-0.05 if math.isinf(v) else v)))
        ax.barh(y, vals, color=color, height=0.7)
        ax.set_yticks(y, attacks, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(title)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)

    qcol = "Q@0.7P" if "Q@0.7P" in rows[0] else "Q@0.95P"
    _plot_col(axes[0], qcol, f"{qcol} (capped display for inf)", "#2563eb")
    _plot_col(axes[1], "Avg_P", "Avg P", "#059669")
    fig.suptitle(f"{method} — leaderboard summary", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "bar_leaderboard.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_all_for_method(
    mrows: list[dict],
    raw_rows: list[dict] | None,
    results_dir: str,
    method: str,
) -> str:
    out_dir = os.path.join(results_dir, "plots", "averaged", method)
    os.makedirs(out_dir, exist_ok=True)

    plot_averaged_bars(mrows, out_dir, method)
    plot_p_vs_psnr(mrows, out_dir, method)
    plot_p_vs_q(mrows, out_dir, method)
    plot_heatmap(mrows, out_dir, method)
    plot_overview_combined(mrows, out_dir, method)
    plot_leaderboard_bars(results_dir, out_dir, method)

    if raw_rows:
        by_attack = _group_raw_by_attack(raw_rows, method)
        plot_raw_strength_overviews(by_attack, out_dir, method)
        plot_raw_p_vs_q_overview(by_attack, out_dir, method)
        plot_raw_heatmaps(by_attack, out_dir, method)
        plot_raw_curves(by_attack, out_dir, method)

    return out_dir


def main() -> int:
    p = argparse.ArgumentParser(description="Plot wmbench averaged/raw CSV results.")
    p.add_argument(
        "--results-dir",
        "-d",
        required=True,
        help="Directory containing results_averaged.csv (and optionally results_raw.csv)",
    )
    p.add_argument("--method", default=None, help="Filter method (default: all in csv)")
    p.add_argument(
        "--render-work-plots",
        action="store_true",
        help="Also render plots from work/{method}/metrics (numeric strength sort)",
    )
    args = p.parse_args()
    results_dir = os.path.abspath(args.results_dir)
    avg_path = os.path.join(results_dir, "results_averaged.csv")
    raw_path = os.path.join(results_dir, "results_raw.csv")
    if not os.path.isfile(avg_path):
        print(f"Missing {avg_path}", file=sys.stderr)
        return 1

    rows = _read_csv(avg_path)
    if args.method:
        rows = [r for r in rows if r.get("method") == args.method]
    if not rows:
        print("No rows to plot.", file=sys.stderr)
        return 1

    raw_rows: list[dict] | None = _read_csv(raw_path) if os.path.isfile(raw_path) else None

    methods = sorted({r["method"] for r in rows})
    for method in methods:
        mrows = [r for r in rows if r["method"] == method]
        out_dir = plot_all_for_method(mrows, raw_rows, results_dir, method)
        print(f"Wrote plots to: {out_dir}")

    if args.render_work_plots:
        from wmbench.output.plotter import render_all_plots

        work_root = os.path.join(results_dir, "work")
        if os.path.isdir(work_root):
            attacks = sorted({r["attack"] for r in rows})
            render_all_plots(work_root, results_dir, methods, attacks)
            print(f"Wrote work/metrics plots under: {os.path.join(results_dir, 'plots')}")
        else:
            print(f"No work/ directory at {work_root}; skipped work plots.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
