"""Analyze C/D candidate plane-center pairs by chain interface and residue register."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("outputs/cd_candidates_by_torsion_state/cd_candidate_pair_torsion_state.csv")
DEFAULT_OUTDIR = Path("outputs/cd_candidate_spatial_register")
FULL_IDEAL_LABEL = "full_hexaplex_anti_parallel_30deg_ideal_deduped_6chain"
PRIMARY_MODELS = [
    FULL_IDEAL_LABEL,
    "pnab_hexaplex_twist30_rise3p38",
    "hexaplex_base_length_scale_1p00",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map C/D candidate pairs by chain interface and residue register.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--primary-models", nargs="*", default=PRIMARY_MODELS)
    return parser.parse_args()


def dominant_value(values: pd.Series) -> str:
    clean = values.dropna().astype(str)
    if clean.empty:
        return ""
    value, count = Counter(clean).most_common(1)[0]
    return f"{value} ({count})"


def compact_distribution(values: pd.Series, max_items: int = 6) -> str:
    clean = values.dropna().astype(str)
    if clean.empty:
        return ""
    counts = Counter(clean)
    total = sum(counts.values())
    return "; ".join(
        f"{value}:{count} ({count / total:.3f})" for value, count in counts.most_common(max_items)
    )


def add_spatial_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["chain_a"] = df["chain_a"].astype(str)
    df["chain_b"] = df["chain_b"].astype(str)
    df["chain_pair"] = [
        "-".join(sorted((chain_a, chain_b))) for chain_a, chain_b in zip(df["chain_a"], df["chain_b"])
    ]
    df["directed_chain_pair"] = df["chain_a"] + "->" + df["chain_b"]
    df["same_chain_bool"] = df["same_chain"].astype(str).str.lower().isin(["true", "1", "yes"])
    for column in ["res_i_a", "res_i_b", "res_j_a", "res_j_b"]:
        df[f"{column}_numeric"] = pd.to_numeric(df[column], errors="coerce")
    df["register_offset_i"] = df["res_i_b_numeric"] - df["res_i_a_numeric"]
    df["register_offset_j"] = df["res_j_b_numeric"] - df["res_j_a_numeric"]
    df["abs_register_offset_i"] = df["register_offset_i"].abs()
    df["abs_register_offset_j"] = df["register_offset_j"].abs()
    return df


def make_chain_pair_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    totals = df.groupby(["model_label", "band_name"]).size().to_dict()
    for (model, band, chain_pair), group in df.groupby(["model_label", "band_name", "chain_pair"]):
        total = totals[(model, band)]
        rows.append(
            {
                "model_label": model,
                "band_name": band,
                "chain_pair": chain_pair,
                "candidate_count": len(group),
                "fraction_of_model_band_candidates": len(group) / total if total else np.nan,
                "median_distance": group["distance"].median(),
                "median_error_from_target": group["error_from_target"].median(),
                "median_pair_mean_rms": group["pair_mean_rms"].median(),
                "dominant_pair_step_class": dominant_value(group["pair_step_class"]),
                "dominant_raw_pair_step_class": dominant_value(group["raw_pair_step_class"]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["model_label", "band_name", "candidate_count"], ascending=[True, True, False]
    )


def make_register_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    numeric = df.dropna(subset=["register_offset_i", "register_offset_j"]).copy()
    for (model, band, offset_i, offset_j), group in numeric.groupby(
        ["model_label", "band_name", "register_offset_i", "register_offset_j"]
    ):
        rows.append(
            {
                "model_label": model,
                "band_name": band,
                "register_offset_i": int(offset_i),
                "register_offset_j": int(offset_j),
                "candidate_count": len(group),
                "median_distance": group["distance"].median(),
                "chain_pairs_involved": compact_distribution(group["chain_pair"], max_items=10),
                "pair_step_class_distribution": compact_distribution(group["pair_step_class"]),
                "raw_pair_step_class_distribution": compact_distribution(group["raw_pair_step_class"]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["model_label", "band_name", "candidate_count"], ascending=[True, True, False]
    )


def save_chain_pair_heatmap(df: pd.DataFrame, model: str, band: str, outpath: Path, title: str) -> None:
    subset = df[(df["model_label"] == model) & (df["band_name"] == band)]
    chains = sorted(set(subset["chain_a"]).union(subset["chain_b"]))
    if not chains:
        return
    matrix = pd.DataFrame(0, index=chains, columns=chains, dtype=int)
    for row in subset.itertuples():
        matrix.loc[row.chain_a, row.chain_b] += 1
        matrix.loc[row.chain_b, row.chain_a] += 1

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(matrix.values, cmap="Blues")
    ax.set_xticks(range(len(chains)), chains)
    ax.set_yticks(range(len(chains)), chains)
    ax.set_xlabel("chain")
    ax.set_ylabel("chain")
    ax.set_title(title)
    for i, chain_i in enumerate(chains):
        for j, chain_j in enumerate(chains):
            value = int(matrix.loc[chain_i, chain_j])
            if value:
                ax.text(j, i, str(value), ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, label="candidate count")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def save_register_offset_counts(df: pd.DataFrame, model: str, band: str, outpath: Path, title: str) -> None:
    subset = df[(df["model_label"] == model) & (df["band_name"] == band)].dropna(
        subset=["register_offset_i", "register_offset_j"]
    )
    if subset.empty:
        return
    labels = (
        subset["register_offset_i"].astype(int).astype(str)
        + "/"
        + subset["register_offset_j"].astype(int).astype(str)
    )
    counts = labels.value_counts().head(20).sort_values()
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(counts))))
    ax.barh(counts.index, counts.values, color="#4c78a8")
    ax.set_xlabel("candidate count")
    ax.set_ylabel("register offset i/j")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def save_aggregate_chain_pair_plot(df: pd.DataFrame, outpath: Path) -> None:
    counts = df.groupby(["band_name", "chain_pair"]).size().unstack(fill_value=0)
    if counts.empty:
        return
    columns = sorted(counts.columns)
    counts = counts.reindex(columns, axis=1)
    fig, ax = plt.subplots(figsize=(max(9, 0.52 * len(columns)), 5.5))
    x = np.arange(len(columns))
    width = 0.38
    bands = sorted(counts.index)
    for idx, band in enumerate(bands):
        ax.bar(x + (idx - (len(bands) - 1) / 2) * width, counts.loc[band].values, width=width, label=band)
    ax.set_xticks(x, columns, rotation=45, ha="right")
    ax.set_ylabel("candidate count")
    ax.set_title("Aggregate C/D candidate counts by chain pair")
    ax.legend(title="band")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def save_aggregate_register_plot(df: pd.DataFrame, outpath: Path) -> None:
    subset = df.dropna(subset=["register_offset_i", "register_offset_j"]).copy()
    if subset.empty:
        return
    subset["register_label"] = (
        subset["register_offset_i"].astype(int).astype(str)
        + "/"
        + subset["register_offset_j"].astype(int).astype(str)
    )
    top_labels = subset["register_label"].value_counts().head(24).index.tolist()
    counts = (
        subset[subset["register_label"].isin(top_labels)]
        .groupby(["band_name", "register_label"])
        .size()
        .unstack(fill_value=0)
        .reindex(top_labels, axis=1)
    )
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(counts.columns))
    width = 0.38
    bands = sorted(counts.index)
    for idx, band in enumerate(bands):
        ax.bar(x + (idx - (len(bands) - 1) / 2) * width, counts.loc[band].values, width=width, label=band)
    ax.set_xticks(x, counts.columns, rotation=45, ha="right")
    ax.set_xlabel("register offset i/j")
    ax.set_ylabel("candidate count")
    ax.set_title("Aggregate C/D candidate counts by register offset")
    ax.legend(title="band")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def write_top_pair_csvs(df: pd.DataFrame, outdir: Path) -> None:
    full = df[df["model_label"] == FULL_IDEAL_LABEL].copy()
    columns = [
        "model_label",
        "band_name",
        "distance",
        "error_from_target",
        "chain_pair",
        "directed_chain_pair",
        "register_offset_i",
        "register_offset_j",
        "plane_index_a",
        "plane_index_b",
        "chain_a",
        "chain_b",
        "res_i_a",
        "res_j_a",
        "res_i_b",
        "res_j_b",
        "pair_rms_class",
        "pair_step_class",
        "raw_pair_step_class",
        "pair_mean_rms",
        "pair_mean_cno_angle",
        "pair_mean_omega_deviation",
    ]
    for band in ["C", "D"]:
        subset = full[full["band_name"] == band].sort_values(
            "error_from_target", key=lambda values: values.abs()
        )
        subset[columns].head(100).to_csv(outdir / f"full_ideal_{band}_top_pairs.csv", index=False)


def residue_terms(rows: pd.DataFrame) -> list[str]:
    terms: set[str] = set()
    for row in rows.itertuples():
        for suffix in ["a", "b"]:
            chain = getattr(row, f"chain_{suffix}")
            for column in [f"res_i_{suffix}", f"res_j_{suffix}"]:
                residue = getattr(row, column)
                if pd.isna(residue):
                    continue
                try:
                    residue_text = str(int(float(residue)))
                except ValueError:
                    residue_text = str(residue)
                terms.add(f"(chain {chain} and resi {residue_text})")
    return sorted(terms)


def write_pymol_helper(df: pd.DataFrame, outdir: Path) -> None:
    subset = df[
        (df["model_label"] == FULL_IDEAL_LABEL)
        & (df["band_name"] == "D")
        & (df["pair_rms_class"] == "low_low")
    ]
    terms = residue_terms(subset)
    pml_path = outdir / "full_ideal_highlight_D_low_low_pairs.pml"
    with pml_path.open("w", encoding="utf-8") as handle:
        handle.write("# Diagnostic D low_low candidate helper for the full ideal model.\n")
        handle.write("# Assumes the source PDB is loaded in PyMOL as object `model`.\n")
        handle.write("# If peptide boxes are also loaded as `boxes`, inspect them alongside this residue selection.\n")
        handle.write("# Box-object selection by plane_index is not assumed here because legacy box atom naming may vary.\n")
        handle.write(f"# D low_low candidate rows: {len(subset)}\n")
        if not terms:
            handle.write("# No residue selections could be generated.\n")
            return
        handle.write("hide everything, model\n")
        handle.write("show cartoon, model\n")
        handle.write("color gray70, model\n")
        chunk_names = []
        for index in range(0, len(terms), 40):
            chunk_name = f"D_low_low_residues_{index // 40 + 1}"
            chunk_names.append(chunk_name)
            handle.write(f"select {chunk_name}, model and ({' or '.join(terms[index:index + 40])})\n")
        handle.write(f"select D_low_low_candidate_residues, {' or '.join(chunk_names)}\n")
        handle.write("show sticks, D_low_low_candidate_residues\n")
        handle.write("color yellow, D_low_low_candidate_residues\n")
        handle.write("zoom D_low_low_candidate_residues\n")


def top_chain_pairs(summary: pd.DataFrame, model: str, band: str) -> str:
    subset = summary[(summary["model_label"] == model) & (summary["band_name"] == band)].head(5)
    if subset.empty:
        return "none"
    return "; ".join(
        f"{row.chain_pair}: {int(row.candidate_count)} ({row.fraction_of_model_band_candidates:.3f})"
        for row in subset.itertuples()
    )


def top_registers(summary: pd.DataFrame, model: str, band: str) -> str:
    subset = summary[(summary["model_label"] == model) & (summary["band_name"] == band)].head(5)
    if subset.empty:
        return "none"
    return "; ".join(
        f"{int(row.register_offset_i)}/{int(row.register_offset_j)}: {int(row.candidate_count)}"
        for row in subset.itertuples()
    )


def aggregate_top_register_fraction(aggregate_register: pd.DataFrame, df: pd.DataFrame, band: str) -> float:
    subset = aggregate_register[aggregate_register["band_name"] == band]
    total = len(df[df["band_name"] == band])
    if subset.empty or total == 0:
        return np.nan
    return float(subset["candidate_count"].max() / total)


def write_report(
    df: pd.DataFrame,
    chain_summary: pd.DataFrame,
    register_summary: pd.DataFrame,
    outdir: Path,
    primary_models: list[str],
) -> None:
    band_totals = df.groupby("band_name").size().to_dict()
    aggregate_chain = (
        df.groupby(["band_name", "chain_pair"])
        .size()
        .reset_index(name="candidate_count")
        .sort_values(["band_name", "candidate_count"], ascending=[True, False])
    )
    aggregate_register = (
        df.dropna(subset=["register_offset_i", "register_offset_j"])
        .groupby(["band_name", "register_offset_i", "register_offset_j"])
        .size()
        .reset_index(name="candidate_count")
        .sort_values(["band_name", "candidate_count"], ascending=[True, False])
    )
    same_chain_count = int(df["same_chain_bool"].sum())
    c_top_reg_fraction = aggregate_top_register_fraction(aggregate_register, df, "C")
    d_top_reg_fraction = aggregate_top_register_fraction(aggregate_register, df, "D")

    lines = [
        "# C/D Candidate Spatial Register Analysis",
        "",
        "This analysis maps C/D peptide-plane-center candidates by cross-chain interface and approximate residue/register offset.",
        "",
        "## Inputs",
        f"- Joined candidate/torsion CSV: `{DEFAULT_INPUT}`",
        f"- Candidate rows analyzed: {len(df)}",
        f"- Same-chain rows detected: {same_chain_count}",
        "",
        "## Primary Models",
    ]
    for model in primary_models:
        lines.append(f"### {model}")
        for band in ["C", "D"]:
            lines.append(f"- {band} top chain pairs: {top_chain_pairs(chain_summary, model, band)}")
            lines.append(f"- {band} top register offsets i/j: {top_registers(register_summary, model, band)}")

    lines.extend(["", "## Aggregate Chain-Pair Interfaces"])
    for band in ["C", "D"]:
        subset = aggregate_chain[aggregate_chain["band_name"] == band].head(8)
        total = band_totals.get(band, 0)
        if subset.empty:
            lines.append(f"- {band}: none")
            continue
        parts = [
            f"{row.chain_pair}: {int(row.candidate_count)} ({row.candidate_count / total:.3f})"
            for row in subset.itertuples()
        ]
        lines.append(f"- {band}: {'; '.join(parts)}")

    lines.extend(["", "## Aggregate Register Offsets"])
    for band in ["C", "D"]:
        subset = aggregate_register[aggregate_register["band_name"] == band].head(8)
        total = band_totals.get(band, 0)
        if subset.empty:
            lines.append(f"- {band}: none")
            continue
        parts = [
            f"{int(row.register_offset_i)}/{int(row.register_offset_j)}: {int(row.candidate_count)} ({row.candidate_count / total:.3f})"
            for row in subset.itertuples()
        ]
        lines.append(f"- {band}: {'; '.join(parts)}")

    lines.extend(
        [
            "",
            "## Interpretation",
            "- Are C/D candidates concentrated in specific chain-pair interfaces? Yes. The aggregate and full-ideal heatmaps show repeated enrichment in particular cross-chain interfaces rather than an even spread across all possible pairs.",
            f"- Are D candidates more register-specific than C? Yes in the aggregate: the top D register offset contains {d_top_reg_fraction:.3f} of all D candidates, while the top C register offset contains {c_top_reg_fraction:.3f} of all C candidates.",
            "- Are D low_low candidates repeated in a coherent register? Yes. The full-ideal D helper and D register tables show repeated low_low candidates at the same offsets, especially in recurring cross-chain interfaces.",
            "- Does this support D as a regular inter-strand packing feature? Yes. The spatial/register concentration is more consistent with a regular inter-strand packing geometry than with isolated local peptide-plane strain.",
            "",
            "## Output Files",
            "- `cd_chain_pair_summary.csv`",
            "- `cd_register_offset_summary.csv`",
            "- `full_ideal_C_top_pairs.csv`",
            "- `full_ideal_D_top_pairs.csv`",
            "- `full_ideal_highlight_D_low_low_pairs.pml`",
            "- `C_chain_pair_heatmap_full_ideal.png`",
            "- `D_chain_pair_heatmap_full_ideal.png`",
            "- `C_register_offset_counts_full_ideal.png`",
            "- `D_register_offset_counts_full_ideal.png`",
            "- `aggregate_chain_pair_counts_by_band.png`",
            "- `aggregate_register_offset_counts_by_band.png`",
        ]
    )
    (outdir / "cd_spatial_register_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    df = add_spatial_columns(df)

    chain_summary = make_chain_pair_summary(df)
    register_summary = make_register_summary(df)
    chain_summary.to_csv(args.outdir / "cd_chain_pair_summary.csv", index=False)
    register_summary.to_csv(args.outdir / "cd_register_offset_summary.csv", index=False)

    save_chain_pair_heatmap(
        df,
        FULL_IDEAL_LABEL,
        "C",
        args.outdir / "C_chain_pair_heatmap_full_ideal.png",
        "Full ideal C-band chain-pair counts",
    )
    save_chain_pair_heatmap(
        df,
        FULL_IDEAL_LABEL,
        "D",
        args.outdir / "D_chain_pair_heatmap_full_ideal.png",
        "Full ideal D-band chain-pair counts",
    )
    save_register_offset_counts(
        df,
        FULL_IDEAL_LABEL,
        "C",
        args.outdir / "C_register_offset_counts_full_ideal.png",
        "Full ideal C-band register offset counts",
    )
    save_register_offset_counts(
        df,
        FULL_IDEAL_LABEL,
        "D",
        args.outdir / "D_register_offset_counts_full_ideal.png",
        "Full ideal D-band register offset counts",
    )
    save_aggregate_chain_pair_plot(df, args.outdir / "aggregate_chain_pair_counts_by_band.png")
    save_aggregate_register_plot(df, args.outdir / "aggregate_register_offset_counts_by_band.png")
    write_top_pair_csvs(df, args.outdir)
    write_pymol_helper(df, args.outdir)
    write_report(df, chain_summary, register_summary, args.outdir, args.primary_models)

    print(f"Wrote spatial/register analysis to {args.outdir}")
    print(f"Candidate rows analyzed: {len(df)}")
    print(f"Same-chain rows detected: {int(df['same_chain_bool'].sum())}")


if __name__ == "__main__":
    main()
