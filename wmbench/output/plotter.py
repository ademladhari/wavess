from __future__ import annotations

import glob
import json
import os

import matplotlib.pyplot as plt

from wmbench.metrics import aggregate as agg
from wmbench.output.plot_utils import safe_attack_filename, strength_key


def _q_from_metrics_file(path: str, anchors: dict[str, tuple[float, float]]) -> float:
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    raw = {k: float(m[k]) for k in agg.METRIC_KEYS if k in m}
    return agg.q_from_raw_row(raw, anchors)


def _load_anchors(output_dir: str) -> dict[str, tuple[float, float]] | None:
    p = os.path.join(output_dir, "normalization_anchors.json")
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    return {k: (float(v[0]), float(v[1])) for k, v in data.items()}


def _collect_attack_series(
    base: str, anchors: dict[str, tuple[float, float]] | None
) -> tuple[list[float], list[float], list[float], list[str]]:
    """Return strengths, P, Q, tags sorted by numeric strength."""
    by_strength: dict[float, tuple[float, float, str]] = {}
    for st_dir in glob.glob(os.path.join(base, "*")):
        if not os.path.isdir(st_dir):
            continue
        tag = os.path.basename(st_dir)
        mj = os.path.join(st_dir, "metrics.json")
        if not os.path.isfile(mj):
            continue
        with open(mj, encoding="utf-8") as f:
            m = json.load(f)
        sk = strength_key(tag)
        p = float(m.get("P", float("nan")))
        if anchors:
            q = _q_from_metrics_file(mj, anchors)
        else:
            q = float(m.get("Q", float("nan")))
        by_strength[sk] = (p, q, tag)
    if not by_strength:
        return [], [], [], []
    keys = sorted(by_strength)
    ps = [by_strength[k][0] for k in keys]
    qs = [by_strength[k][1] for k in keys]
    tags = [by_strength[k][2] for k in keys]
    return keys, ps, qs, tags


def plot_method_overview(work_dir: str, method: str, output_dir: str, attacks: list[str]) -> None:
    anchors = _load_anchors(output_dir)
    fig, ax = plt.subplots(figsize=(10, 6))
    for atk in attacks:
        base = os.path.join(work_dir, "metrics", atk)
        if not os.path.isdir(base):
            continue
        _, ps, qs, _ = _collect_attack_series(base, anchors)
        if ps:
            ax.plot(ps, qs, "o-", label=atk, alpha=0.8, markersize=4)
    ax.axvline(0.95, color="k", linestyle="--", alpha=0.4)
    ax.axvline(0.7, color="k", linestyle=":", alpha=0.4)
    ax.set_xlabel("P (TPR@0.1%FPR)")
    ax.set_ylabel("Q")
    ax.set_title(f"P vs Q overview — {method}\n(points connected in increasing strength order)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
    fig.tight_layout()
    out = os.path.join(output_dir, "plots", method)
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, "overview_P_vs_Q.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_method_strength_vs_q(work_dir: str, method: str, output_dir: str, attacks: list[str]) -> None:
    anchors = _load_anchors(output_dir)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for atk in attacks:
        base = os.path.join(work_dir, "metrics", atk)
        if not os.path.isdir(base):
            continue
        sts, ps, qs, _ = _collect_attack_series(base, anchors)
        if not sts:
            continue
        axes[0].plot(sts, qs, "o-", label=atk, alpha=0.8, markersize=4)
        axes[1].plot(sts, ps, "o-", label=atk, alpha=0.8, markersize=4)
    axes[0].set_xlabel("Attack strength")
    axes[0].set_ylabel("Q")
    axes[0].set_title(f"{method}: quality vs strength")
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("Attack strength")
    axes[1].set_ylabel("P (TPR@0.1% FPR)")
    axes[1].set_title(f"{method}: detection vs strength")
    axes[1].axhline(0.7, color="#888", linestyle=":", alpha=0.5)
    axes[1].grid(alpha=0.3)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
    fig.suptitle("Curves sorted by numeric strength", fontsize=11)
    fig.tight_layout()
    out = os.path.join(output_dir, "plots", method)
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, "overview_strength_vs_P_Q.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_single_attack(work_dir: str, method: str, attack: str, output_dir: str) -> None:
    anchors = _load_anchors(output_dir)
    base = os.path.join(work_dir, "metrics", attack)
    if not os.path.isdir(base):
        return
    sts, ps, qs, tags = _collect_attack_series(base, anchors)
    if not tags:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].plot(sts, qs, "o-", color="#2563eb")
    for s, q, t in zip(sts, qs, tags):
        axes[0].annotate(str(t), (s, q), textcoords="offset points", xytext=(3, 3), fontsize=7)
    axes[0].set_xlabel("Strength")
    axes[0].set_ylabel("Q")
    axes[0].set_title("Quality vs strength")
    axes[0].grid(alpha=0.3)

    axes[1].plot(sts, ps, "o-", color="#059669")
    for s, p, t in zip(sts, ps, tags):
        axes[1].annotate(str(t), (s, p), textcoords="offset points", xytext=(3, 3), fontsize=7)
    axes[1].axhline(0.7, color="#888", linestyle=":", alpha=0.5)
    axes[1].set_xlabel("Strength")
    axes[1].set_ylabel("P")
    axes[1].set_title("Detection vs strength")
    axes[1].grid(alpha=0.3)

    axes[2].plot(ps, qs, "o-", color="#7c3aed")
    for p, q, t in zip(ps, qs, tags):
        axes[2].annotate(str(t), (p, q), textcoords="offset points", xytext=(3, 3), fontsize=7)
    axes[2].axvline(0.7, color="#888", linestyle=":", alpha=0.4)
    axes[2].set_xlabel("P")
    axes[2].set_ylabel("Q")
    axes[2].set_title("P vs Q (strength order)")
    axes[2].grid(alpha=0.3)

    fig.suptitle(f"{method} / {attack}", fontsize=11, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(output_dir, "plots", method, "by_attack")
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, f"{safe_attack_filename(attack)}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_all_plots(work_parent: str, output_dir: str, methods: list[str], attacks: list[str]) -> None:
    for m in methods:
        wd = os.path.join(work_parent, m)
        plot_method_overview(wd, m, output_dir, attacks)
        plot_method_strength_vs_q(wd, m, output_dir, attacks)
        for a in attacks:
            plot_single_attack(wd, m, a, output_dir)
