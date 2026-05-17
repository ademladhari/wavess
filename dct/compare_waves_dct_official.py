from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


OFFICIAL_DETECTION = [
    ("Dist-Rotation", 8, "-inf", 0.434, 0.131, 12, 0.613, 0.642, 0.400, 4, 0.454, 0.500, 0.288),
    ("Dist-RCrop", 11, "-inf", 0.592, 0.094, 24, "inf", "inf", 0.972, 6, 0.602, 0.602, 0.494),
    ("Dist-Erase", 26, "inf", "inf", 0.986, 25, "inf", "inf", 0.988, 25, "inf", "inf", 1.000),
    ("Dist-Bright", 22, "inf", "inf", 0.913, 23, "inf", "inf", 0.982, 22, "inf", "inf", 0.995),
    ("Dist-Contrast", 23, "inf", "inf", 0.949, 20, "inf", "inf", 0.979, 17, "inf", "inf", 0.994),
    ("Dist-Blur", 21, 1.105, 1.437, 0.551, 5, "-inf", "-inf", 0.000, 9, 0.897, 0.970, 0.280),
    ("Dist-Noise", 16, 0.427, "inf", 0.728, 8, 0.415, 0.480, 0.633, 24, "inf", "inf", 1.000),
    ("Dist-JPEG", 17, 0.499, 0.499, 0.700, 9, 0.485, 0.485, 0.540, 21, "inf", "inf", 0.995),
    ("DistCom-Geo", 9, "-inf", 0.559, 0.105, 13, 0.788, 0.835, 0.519, 7, 0.676, 0.717, 0.359),
    ("DistCom-Photo", 23, "inf", "inf", 0.947, 20, "inf", "inf", 0.981, 17, "inf", "inf", 0.994),
    ("DistCom-Deg", 18, 0.556, 0.864, 0.570, 7, 0.216, 0.281, 0.183, 8, 0.870, 0.957, 0.737),
    ("DistCom-All", 10, "-inf", 0.575, 0.123, 11, 0.550, 0.623, 0.176, 10, 0.995, 1.096, 0.682),
    ("Regen-Diff", 6, "-inf", 0.307, 0.258, 1, "-inf", "-inf", 0.000, 2, 0.333, "inf", 0.766),
    ("Regen-DiffP", 6, "-inf", 0.308, 0.256, 1, "-inf", "-inf", 0.000, 1, 0.336, 0.356, 0.763),
    ("Regen-VAE", 19, 0.578, 0.578, 0.701, 10, 0.545, 0.545, 0.340, 23, "inf", "inf", 1.000),
    ("Regen-KLVAE", 14, 0.257, "inf", 0.810, 6, "-inf", "-inf", 0.047, 17, "inf", "inf", 0.999),
    ("Rinse-2xDiff", 5, "-inf", 0.270, 0.220, 3, "-inf", "-inf", 0.000, 3, 0.390, 0.402, 0.778),
    ("Rinse-4xDiff", 1, "-inf", "-inf", 0.110, 4, "-inf", "-inf", 0.000, 5, 0.488, 0.676, 0.687),
    ("AdvEmbG-KLVAE8", 4, "-inf", 0.168, 0.259, 20, "inf", "inf", 0.985, 17, "inf", "inf", 1.000),
    ("AdvEmbB-RN18", 15, 0.288, "inf", 0.811, 17, "inf", "inf", 0.990, 14, "inf", "inf", 1.000),
    ("AdvEmbB-CLIP", 20, 0.697, "inf", 0.798, 26, "inf", "inf", 0.991, 25, "inf", "inf", 1.000),
    ("AdvEmbB-KLVAE16", 12, 0.158, 0.309, 0.540, 19, "inf", "inf", 0.983, 14, "inf", "inf", 1.000),
    ("AdvEmbB-SdxlVAE", 13, 0.214, "inf", 0.692, 17, "inf", "inf", 0.986, 14, "inf", "inf", 1.000),
    ("AdvCls-UnWM&WM", 2, "-inf", 0.123, 0.352, 14, "inf", "inf", 0.991, 11, "inf", "inf", 1.000),
    ("AdvCls-Real&WM", 25, "inf", "inf", 0.986, 14, "inf", "inf", 0.990, 11, "inf", "inf", 1.000),
    ("AdvCls-WM1&WM2", 2, "-inf", 0.118, 0.343, 14, "inf", "inf", 0.991, 13, "inf", "inf", 1.000),
]


def normalize_attack_name(name: str) -> str:
    aliases = {
        "Dist-Com-Geo": "DistCom-Geo",
        "Dist-Com-Photo": "DistCom-Photo",
        "Dist-Com-Deg": "DistCom-Deg",
        "Dist-Com-All": "DistCom-All",
        "Regen-Diffusion": "Regen-Diff",
        "Regen-Diffusion&P": "Regen-DiffP",
        "Regen-2xDiffusion": "Rinse-2xDiff",
        "Regen-4xDiffusion": "Rinse-4xDiff",
        "Regen-4xVAE": "Rinse-4xDiff",  # mapped closest family for WAVES naming
        "AdvEmb-RN18": "AdvEmbB-RN18",
        "AdvEmb-CLIP": "AdvEmbB-CLIP",
        "AdvEmb-KLVAE8": "AdvEmbG-KLVAE8",
        "AdvEmb-KLVAE16": "AdvEmbB-KLVAE16",
        "AdvEmb-SdxlVAE": "AdvEmbB-SdxlVAE",
        "AdvCls-UnWM-WM": "AdvCls-UnWM&WM",
        "AdvCls-Real-WM": "AdvCls-Real&WM",
        "AdvCls-WM1-WM2": "AdvCls-WM1&WM2",
    }
    return aliases.get(name, name)


def main() -> None:
    root = Path(".")
    dct_path = root / "outputs_waves_dct" / "waves_dct_leaderboard.csv"
    out_path = root / "outputs_waves_dct" / "waves_dct_vs_official_detection.csv"
    avgp_only_path = root / "outputs_waves_dct" / "waves_dct_vs_official_avgp_only.csv"
    report_path = root / "outputs_waves_dct" / "waves_dct_vs_official_detection.json"

    dct_df = pd.read_csv(dct_path)
    dct_df["attack_norm"] = dct_df["Attack"].map(normalize_attack_name)

    rows = []
    for (
        attack,
        tr_rank,
        tr_q07,
        tr_q04,
        tr_avg_p,
        ss_rank,
        ss_q07,
        ss_q04,
        ss_avg_p,
        st_rank,
        st_q07,
        st_q04,
        st_avg_p,
    ) in OFFICIAL_DETECTION:
        dct_match = dct_df[dct_df["attack_norm"] == attack]
        if dct_match.empty:
            continue
        d = dct_match.iloc[0]
        dct_avg_p = float(d["Avg P"])
        tr_delta = dct_avg_p - float(tr_avg_p)
        ss_delta = dct_avg_p - float(ss_avg_p)
        st_delta = dct_avg_p - float(st_avg_p)
        compare_all = f"TR:{tr_delta:+.3f} | SS:{ss_delta:+.3f} | ST:{st_delta:+.3f}"
        rows.append(
            {
                "Attack": attack,
                "DCT_Rank": int(d["Rank"]),
                "DCT_Q@0.7P": d["Q@0.7P"],
                "DCT_Q@0.4P": d["Q@0.4P"],
                "DCT_AvgP": dct_avg_p,
                "TreeRing_Rank": tr_rank,
                "TreeRing_Q@0.7P": tr_q07,
                "TreeRing_Q@0.4P": tr_q04,
                "TreeRing_AvgP": tr_avg_p,
                "StableSig_Rank": ss_rank,
                "StableSig_Q@0.7P": ss_q07,
                "StableSig_Q@0.4P": ss_q04,
                "StableSig_AvgP": ss_avg_p,
                "StegaStamp_Rank": st_rank,
                "StegaStamp_Q@0.7P": st_q07,
                "StegaStamp_Q@0.4P": st_q04,
                "StegaStamp_AvgP": st_avg_p,
                "AvgP_Compare_All_(TR|SS|ST)": compare_all,
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    out_df[
        ["Attack", "DCT_AvgP", "TreeRing_AvgP", "StableSig_AvgP", "StegaStamp_AvgP"]
    ].to_csv(avgp_only_path, index=False)

    # Simple aggregate summary for quick interpretation.
    summary = {
        "rows_compared": int(len(out_df)),
        "mean_avgp": {
            "dct": float(pd.to_numeric(out_df["DCT_AvgP"], errors="coerce").mean()),
            "tree_ring": float(pd.to_numeric(out_df["TreeRing_AvgP"], errors="coerce").mean()),
            "stable_sig": float(pd.to_numeric(out_df["StableSig_AvgP"], errors="coerce").mean()),
            "stegastamp": float(pd.to_numeric(out_df["StegaStamp_AvgP"], errors="coerce").mean()),
        },
    }
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote: {out_path}")
    print(f"Wrote: {avgp_only_path}")
    print(f"Wrote: {report_path}")


if __name__ == "__main__":
    main()
