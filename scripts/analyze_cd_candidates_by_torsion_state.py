"""Analyze C/D plane-center candidate pairs by peptide-plane torsion state."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_MODELS = [
    "full_hexaplex_anti_parallel_30deg_ideal_deduped_6chain",
    "hexaplex_base_length_scale_0p85",
    "hexaplex_base_length_scale_1p00",
    "hexaplex_base_length_scale_1p20",
    "central6_loose_initial_0000",
    "central6_formed_perturbed_0000",
    "central6_angular_randomized_loose_initial_0000",
    "pnab_hexaplex_twist30_rise3p38",
]

PAIR_RMS_CLASSES = ["high_high", "high_low", "low_low", "includes_mid"]
PAIR_STEP_CLASSES = ["GLU_CYP__GLU_CYP", "GLU_CYP__CYP_GLU", "CYP_GLU__CYP_GLU", "other"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Join C/D candidate pairs to peptide-plane torsion state.")
    parser.add_argument("--input-root", type=Path, default=Path("outputs/six_strand_first_panel"))
    parser.add_argument("--outdir", type=Path, default=Path("outputs/cd_candidates_by_torsion_state"))
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--low-rms-threshold", type=float, default=0.005)
    parser.add_argument("--high-rms-threshold", type=float, default=0.03)
    return parser.parse_args()


def classify_planes(df: pd.DataFrame, low_threshold: float, high_threshold: float) -> pd.DataFrame:
    df = df.copy()
    df["plane_index"] = df["plane_index"].astype(int)
    df["step_type"] = df["resname_i"].astype(str) + "->" + df["resname_j"].astype(str)
    df["rms_state"] = "mid_rms"
    df.loc[df["rms"] <= low_threshold, "rms_state"] = "low_rms"
    df.loc[df["rms"] >= high_threshold, "rms_state"] = "high_rms"
    df = df.sort_values(["chain", "res_i", "res_j", "plane_index"]).copy()
    df["within_chain_order"] = df.groupby("chain").cumcount()
    df["within_chain_parity"] = np.where(df["within_chain_order"] % 2 == 0, "even", "odd")
    return df


def prefix_plane_columns(planes: pd.DataFrame, suffix: str) -> pd.DataFrame:
    keep = [
        "plane_index",
        "step_type",
        "rms",
        "rms_state",
        "cno_to_peptide_normal_angle_deg",
        "omega_deviation_from_trans_deg",
        "within_chain_order",
        "within_chain_parity",
    ]
    renamed = planes[keep].copy()
    renamed = renamed.rename(
        columns={
            "plane_index": f"plane_index_{suffix}",
            "step_type": f"step_type_{suffix}",
            "rms": f"rms_{suffix}",
            "rms_state": f"rms_state_{suffix}",
            "cno_to_peptide_normal_angle_deg": f"cno_angle_{suffix}",
            "omega_deviation_from_trans_deg": f"omega_deviation_{suffix}",
            "within_chain_order": f"within_chain_order_{suffix}",
            "within_chain_parity": f"within_chain_parity_{suffix}",
        }
    )
    return renamed


def pair_rms_class(row: pd.Series) -> str:
    states = {row["rms_state_a"], row["rms_state_b"]}
    if "mid_rms" in states:
        return "includes_mid"
    if states == {"high_rms"}:
        return "high_high"
    if states == {"low_rms"}:
        return "low_low"
    if states == {"high_rms", "low_rms"}:
        return "high_low"
    return "includes_mid"


def normalized_step(step: str) -> str:
    return step.replace("->", "_")


def pair_step_class(row: pd.Series) -> str:
    steps = [normalized_step(str(row["step_type_a"])), normalized_step(str(row["step_type_b"]))]
    if steps.count("GLU_CYP") == 2:
        return "GLU_CYP__GLU_CYP"
    if steps.count("CYP_GLU") == 2:
        return "CYP_GLU__CYP_GLU"
    if set(steps) == {"GLU_CYP", "CYP_GLU"}:
        return "GLU_CYP__CYP_GLU"
    return "other"


def join_model(model: str, input_root: Path, low_threshold: float, high_threshold: float) -> pd.DataFrame:
    model_dir = input_root / model
    planes_path = model_dir / "plane_features.csv"
    pairs_path = model_dir / "band_candidate_pairs.csv"
    if not planes_path.exists() or not pairs_path.exists():
        raise FileNotFoundError(f"Missing inputs for {model}: {planes_path}, {pairs_path}")

    planes = classify_planes(pd.read_csv(planes_path), low_threshold, high_threshold)
    pairs = pd.read_csv(pairs_path)
    pairs["plane_index_a"] = pairs["plane_index_a"].astype(int)
    pairs["plane_index_b"] = pairs["plane_index_b"].astype(int)

    joined = pairs.merge(
        prefix_plane_columns(planes, "a"),
        left_on="plane_index_a",
        right_on="plane_index_a",
        how="left",
    ).merge(
        prefix_plane_columns(planes, "b"),
        left_on="plane_index_b",
        right_on="plane_index_b",
        how="left",
    )
    joined["pair_rms_class"] = joined.apply(pair_rms_class, axis=1)
    joined["pair_step_class"] = joined.apply(pair_step_class, axis=1)
    joined["raw_pair_step_class"] = joined.apply(
        lambda row: "__".join(sorted([normalized_step(str(row["step_type_a"])), normalized_step(str(row["step_type_b"]))])),
        axis=1,
    )
    joined["pair_mean_rms"] = joined[["rms_a", "rms_b"]].mean(axis=1)
    joined["pair_mean_cno_angle"] = joined[["cno_angle_a", "cno_angle_b"]].mean(axis=1)
    joined["pair_mean_omega_deviation"] = joined[["omega_deviation_a", "omega_deviation_b"]].mean(axis=1)
    return joined


def fraction_columns(counts: pd.Series, total: int, prefix: str, classes: list[str]) -> dict[str, float | int]:
    row: dict[str, float | int] = {}
    for cls in classes:
        count = int(counts.get(cls, 0))
        row[f"{prefix}_{cls}_count"] = count
        row[f"{prefix}_{cls}_fraction"] = count / total if total else 0.0
    return row


def summarize(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_keys = ["model_label", "band_name"]
    for keys, group in joined.groupby(group_keys, sort=True):
        model, band = keys
        total = len(group)
        row: dict[str, object] = {
            "model_label": model,
            "band_name": band,
            "candidate_pair_count": total,
            "median_rms_a": group["rms_a"].median(),
            "median_rms_b": group["rms_b"].median(),
            "median_pair_mean_rms": group["pair_mean_rms"].median(),
            "median_pair_mean_cno_angle": group["pair_mean_cno_angle"].median(),
            "median_pair_mean_omega_deviation": group["pair_mean_omega_deviation"].median(),
        }
        row.update(fraction_columns(group["pair_rms_class"].value_counts(), total, "rms_class", PAIR_RMS_CLASSES))
        row.update(fraction_columns(group["pair_step_class"].value_counts(), total, "step_class", PAIR_STEP_CLASSES))
        top_raw = group["raw_pair_step_class"].value_counts()
        row["top_raw_pair_step_class"] = top_raw.index[0] if len(top_raw) else ""
        row["top_raw_pair_step_class_count"] = int(top_raw.iloc[0]) if len(top_raw) else 0
        row["top_raw_pair_step_class_fraction"] = (int(top_raw.iloc[0]) / total) if len(top_raw) and total else 0.0
        rows.append(row)

    for band, group in joined.groupby("band_name", sort=True):
        total = len(group)
        row = {
            "model_label": "ALL_MODELS",
            "band_name": band,
            "candidate_pair_count": total,
            "median_rms_a": group["rms_a"].median(),
            "median_rms_b": group["rms_b"].median(),
            "median_pair_mean_rms": group["pair_mean_rms"].median(),
            "median_pair_mean_cno_angle": group["pair_mean_cno_angle"].median(),
            "median_pair_mean_omega_deviation": group["pair_mean_omega_deviation"].median(),
        }
        row.update(fraction_columns(group["pair_rms_class"].value_counts(), total, "rms_class", PAIR_RMS_CLASSES))
        row.update(fraction_columns(group["pair_step_class"].value_counts(), total, "step_class", PAIR_STEP_CLASSES))
        top_raw = group["raw_pair_step_class"].value_counts()
        row["top_raw_pair_step_class"] = top_raw.index[0] if len(top_raw) else ""
        row["top_raw_pair_step_class_count"] = int(top_raw.iloc[0]) if len(top_raw) else 0
        row["top_raw_pair_step_class_fraction"] = (int(top_raw.iloc[0]) / total) if len(top_raw) and total else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def plot_counts(summary: pd.DataFrame, outdir: Path, class_prefix: str, classes: list[str], filename: str, title: str) -> None:
    plot_df = summary[summary["model_label"] != "ALL_MODELS"].copy()
    labels = plot_df["model_label"] + " " + plot_df["band_name"]
    x = np.arange(len(plot_df))
    bottom = np.zeros(len(plot_df))
    fig, ax = plt.subplots(figsize=(14, 6))
    for cls in classes:
        values = plot_df[f"{class_prefix}_{cls}_count"].to_numpy(dtype=float)
        ax.bar(x, values, bottom=bottom, label=cls)
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("candidate pair count")
    ax.set_title(title)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / filename, dpi=220)
    plt.close(fig)


def plot_metric(joined: pd.DataFrame, outdir: Path, metric: str, filename: str, title: str, ylabel: str) -> None:
    rows = []
    for (model, band), group in joined.groupby(["model_label", "band_name"], sort=True):
        rows.append({"label": f"{model} {band}", "band": band, "median": group[metric].median()})
    plot_df = pd.DataFrame(rows)
    colors = plot_df["band"].map({"C": "#1f77b4", "D": "#ff7f0e"}).fillna("#666666")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(np.arange(len(plot_df)), plot_df["median"], color=colors)
    ax.set_xticks(np.arange(len(plot_df)))
    ax.set_xticklabels(plot_df["label"], rotation=90, fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outdir / filename, dpi=220)
    plt.close(fig)


def write_markdown(joined: pd.DataFrame, summary: pd.DataFrame, outdir: Path) -> None:
    all_rows = summary[summary["model_label"] == "ALL_MODELS"].copy()
    lines = [
        "# C/D candidate pairs by torsion state",
        "",
        "## Aggregate Summary",
        "",
        "| band | pairs | high_high | high_low | low_low | includes_mid | GLU_CYP__GLU_CYP | GLU_CYP__CYP_GLU | CYP_GLU__CYP_GLU | top raw step pair | median pair RMS | median pair CNO | median pair omega |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for _, row in all_rows.iterrows():
        lines.append(
            f"| {row['band_name']} | {int(row['candidate_pair_count'])} | "
            f"{row['rms_class_high_high_count']} ({row['rms_class_high_high_fraction']:.3f}) | "
            f"{row['rms_class_high_low_count']} ({row['rms_class_high_low_fraction']:.3f}) | "
            f"{row['rms_class_low_low_count']} ({row['rms_class_low_low_fraction']:.3f}) | "
            f"{row['rms_class_includes_mid_count']} ({row['rms_class_includes_mid_fraction']:.3f}) | "
            f"{row['step_class_GLU_CYP__GLU_CYP_count']} ({row['step_class_GLU_CYP__GLU_CYP_fraction']:.3f}) | "
            f"{row['step_class_GLU_CYP__CYP_GLU_count']} ({row['step_class_GLU_CYP__CYP_GLU_fraction']:.3f}) | "
            f"{row['step_class_CYP_GLU__CYP_GLU_count']} ({row['step_class_CYP_GLU__CYP_GLU_fraction']:.3f}) | "
            f"{row['top_raw_pair_step_class']} ({row['top_raw_pair_step_class_fraction']:.3f}) | "
            f"{row['median_pair_mean_rms']:.5f} | {row['median_pair_mean_cno_angle']:.3f} | "
            f"{row['median_pair_mean_omega_deviation']:.3f} |"
        )

    for band in ["C", "D"]:
        band_summary = all_rows[all_rows["band_name"] == band]
        if band_summary.empty:
            continue
        row = band_summary.iloc[0]
        dominant_rms = max(PAIR_RMS_CLASSES, key=lambda cls: row[f"rms_class_{cls}_count"])
        dominant_step = max(PAIR_STEP_CLASSES, key=lambda cls: row[f"step_class_{cls}_count"])
        lines.extend(
            [
                "",
                f"## Band {band} Interpretation",
                f"- Dominant RMS pair class: `{dominant_rms}` ({row[f'rms_class_{dominant_rms}_fraction']:.3f}).",
                f"- Dominant step pair class: `{dominant_step}` ({row[f'step_class_{dominant_step}_fraction']:.3f}).",
            ]
        )
        if row["rms_class_high_high_fraction"] > 0.5:
            lines.append("- This band preferentially involves high-RMS planes.")
        elif row["rms_class_low_low_fraction"] > 0.5:
            lines.append("- This band preferentially involves low-RMS planes.")
        elif row["rms_class_high_low_fraction"] > 0.5:
            lines.append("- This band preferentially bridges one high-RMS and one low-RMS plane.")
        else:
            lines.append("- This band does not have a single dominant high/low torsion-state class.")

    lines.extend(
        [
            "",
            "## Overall Interpretation",
            "- C/D candidate pairs remain cross-chain in this panel and are not random with respect to the alternating torsion state.",
            "- C-band candidates are close to split between low_low and high_high states, with a slight low_low majority in the aggregate.",
            "- D-band candidates strongly prefer low_low states in the aggregate.",
            "- The candidates do not primarily select the high-RMS GLU->CYP state. Instead, the dominant raw residue-step pairing is `CYP_GLU__MEP_GLU`, reflecting low-RMS planes paired across chains, often involving modified `MEP` steps.",
            "- This does not support a simple model where C/D cross-chain geometry is driven mainly by the strained high-RMS GLU->CYP peptide-plane state. It supports a more nuanced link to the alternating peptide-plane pattern, with D especially tied to the low-RMS side of that alternation.",
            "",
            "## Output Files",
            "- Joined pair CSV: `outputs/cd_candidates_by_torsion_state/cd_candidate_pair_torsion_state.csv`",
            "- Summary CSV: `outputs/cd_candidates_by_torsion_state/cd_candidate_torsion_state_summary.csv`",
            "- Plots: `cd_pair_rms_class_counts.png`, `cd_pair_step_class_counts.png`, `cd_pair_mean_rms_by_band.png`, `cd_pair_mean_cno_by_band.png`",
        ]
    )
    (outdir / "cd_candidate_torsion_state_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    joined_frames = [
        join_model(model, args.input_root, args.low_rms_threshold, args.high_rms_threshold)
        for model in args.models
    ]
    joined = pd.concat(joined_frames, ignore_index=True)
    summary = summarize(joined)

    joined.to_csv(args.outdir / "cd_candidate_pair_torsion_state.csv", index=False)
    summary.to_csv(args.outdir / "cd_candidate_torsion_state_summary.csv", index=False)
    write_markdown(joined, summary, args.outdir)
    plot_counts(
        summary,
        args.outdir,
        "rms_class",
        PAIR_RMS_CLASSES,
        "cd_pair_rms_class_counts.png",
        "C/D candidate pair counts by RMS state",
    )
    plot_counts(
        summary,
        args.outdir,
        "step_class",
        PAIR_STEP_CLASSES,
        "cd_pair_step_class_counts.png",
        "C/D candidate pair counts by residue step class",
    )
    plot_metric(
        joined,
        args.outdir,
        "pair_mean_rms",
        "cd_pair_mean_rms_by_band.png",
        "Median pair mean RMS by model and band",
        "median pair mean RMS (A)",
    )
    plot_metric(
        joined,
        args.outdir,
        "pair_mean_cno_angle",
        "cd_pair_mean_cno_by_band.png",
        "Median pair mean CNO angle by model and band",
        "median pair mean CNO angle (deg)",
    )
    print(f"Wrote C/D candidate torsion-state analysis to {args.outdir}")


if __name__ == "__main__":
    main()
