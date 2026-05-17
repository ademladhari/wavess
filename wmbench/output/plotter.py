from __future__ import annotations

import glob
import json
import os

import matplotlib.pyplot as plt

from wmbench.metrics import aggregate as agg


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


def plot_method_overview(work_dir: str, method: str, output_dir: str, attacks: list[str]) -> None:
    anchors = _load_anchors(output_dir)
    plt.figure(figsize=(10, 6))
    for atk in attacks:
        base = os.path.join(work_dir, "metrics", atk)
        if not os.path.isdir(base):
            continue
        xs: list[float] = []
        ys: list[float] = []
        for st_dir in sorted(glob.glob(os.path.join(base, "*"))):
            if not os.path.isdir(st_dir):
                continue
            mj = os.path.join(st_dir, "metrics.json")
            if not os.path.isfile(mj):
                continue
            with open(mj, encoding="utf-8") as f:
                m = json.load(f)
            xs.append(float(m.get("P", float("nan"))))
            if anchors:
                ys.append(_q_from_metrics_file(mj, anchors))
            else:
                ys.append(float(m.get("Q", float("nan"))))
        if xs:
            plt.plot(xs, ys, "o-", label=atk, alpha=0.8)
    plt.axvline(0.95, color="k", linestyle="--", alpha=0.4)
    plt.axvline(0.7, color="k", linestyle=":", alpha=0.4)
    plt.xlabel("P (TPR@0.1%FPR)")
    plt.ylabel("Q")
    plt.title(f"P vs Q overview — {method}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
    plt.tight_layout()
    out = os.path.join(output_dir, "plots", method)
    os.makedirs(out, exist_ok=True)
    plt.savefig(os.path.join(out, "overview.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_single_attack(work_dir: str, method: str, attack: str, output_dir: str) -> None:
    anchors = _load_anchors(output_dir)
    base = os.path.join(work_dir, "metrics", attack)
    if not os.path.isdir(base):
        return
    st_dirs = sorted(d for d in glob.glob(os.path.join(base, "*")) if os.path.isdir(d))
    tags: list[str] = []
    ps: list[float] = []
    qs: list[float] = []
    for d in st_dirs:
        mj = os.path.join(d, "metrics.json")
        if not os.path.isfile(mj):
            continue
        tags.append(os.path.basename(d))
        with open(mj, encoding="utf-8") as f:
            m = json.load(f)
        ps.append(float(m.get("P", float("nan"))))
        if anchors:
            qs.append(_q_from_metrics_file(mj, anchors))
        else:
            qs.append(float(m.get("Q", float("nan"))))

    if not tags:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(ps, qs, "bo-")
    for i, t in enumerate(tags):
        plt.annotate(str(t), (ps[i], qs[i]), textcoords="offset points", xytext=(4, 4), fontsize=8)
    plt.axvline(0.95, color="k", linestyle="--", alpha=0.4)
    plt.axvline(0.7, color="k", linestyle=":", alpha=0.4)
    plt.xlabel("P")
    plt.ylabel("Q")
    plt.title(f"{method} / {attack}")
    plt.tight_layout()
    out = os.path.join(output_dir, "plots", method)
    os.makedirs(out, exist_ok=True)
    safe = attack.replace(os.sep, "_").replace("/", "_")
    plt.savefig(os.path.join(out, f"{safe}.png"), dpi=150)
    plt.close()


def render_all_plots(work_parent: str, output_dir: str, methods: list[str], attacks: list[str]) -> None:
    for m in methods:
        wd = os.path.join(work_parent, m)
        plot_method_overview(wd, m, output_dir, attacks)
        for a in attacks:
            plot_single_attack(wd, m, a, output_dir)
