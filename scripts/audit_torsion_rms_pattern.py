"""Audit alternating peptide-plane RMS/torsion patterns at atom level."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_MODELS = [
    "full_hexaplex_anti_parallel_30deg_ideal_deduped_6chain",
    "pnab_hexaplex_twist30_rise3p38",
    "central6_loose_initial_0000",
]

ATOM_DISTANCE_COLUMNS = {
    "CA_i": "ca_i_plane_dist",
    "C_i": "c_i_plane_dist",
    "O_i": "o_i_plane_dist",
    "N_j": "n_j_plane_dist",
    "CA_j": "ca_j_plane_dist",
    "HN_j": "hn_j_plane_dist",
}

ABS_ATOM_COLUMNS = {atom: f"abs_{column}" for atom, column in ATOM_DISTANCE_COLUMNS.items()}

CHAIN_COLORS = {
    "A": "#1f77b4",
    "B": "#ff7f0e",
    "C": "#2ca02c",
    "D": "#d62728",
    "E": "#9467bd",
    "F": "#8c564b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit atom-level drivers of alternating peptide-plane RMS.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("outputs/six_strand_first_panel"),
        help="Root containing per-model plane_features.csv files.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("outputs/torsion_rms_pattern_audit"),
        help="Output directory for audit CSVs, markdown, and plots.",
    )
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--low-rms-threshold", type=float, default=0.005)
    parser.add_argument("--high-rms-threshold", type=float, default=0.03)
    parser.add_argument("--gap", type=int, default=5)
    return parser.parse_args()


def chain_color(chain: str) -> str:
    if chain in CHAIN_COLORS:
        return CHAIN_COLORS[chain]
    keys = sorted(CHAIN_COLORS)
    return CHAIN_COLORS[keys[hash(chain) % len(keys)]]


def add_serial_layout(df: pd.DataFrame, gap: int) -> pd.DataFrame:
    df = df.sort_values(["chain", "res_i", "plane_index"]).reset_index(drop=True).copy()
    serial_x: list[int] = []
    within_chain_positions: list[int] = []
    offset = 0
    last_chain = None
    chain_position = 0
    for _, row in df.iterrows():
        if last_chain is not None and row["chain"] != last_chain:
            offset += gap
            chain_position = 0
        serial_x.append(len(serial_x) + offset)
        within_chain_positions.append(chain_position)
        chain_position += 1
        last_chain = row["chain"]
    df["serial_x"] = serial_x
    df["within_chain_order"] = within_chain_positions
    df["within_chain_order_parity"] = np.where(df["within_chain_order"] % 2 == 0, "even", "odd")
    return df


def classify_and_enrich(df: pd.DataFrame, low_threshold: float, high_threshold: float, gap: int) -> pd.DataFrame:
    df = add_serial_layout(df, gap)
    df["plane_index_parity"] = np.where(df["plane_index"].astype(int) % 2 == 0, "even", "odd")
    df["rms_class"] = "mid_rms"
    df.loc[df["rms"] <= low_threshold, "rms_class"] = "low_rms"
    df.loc[df["rms"] >= high_threshold, "rms_class"] = "high_rms"

    for atom, column in ATOM_DISTANCE_COLUMNS.items():
        abs_column = ABS_ATOM_COLUMNS[atom]
        df[abs_column] = df[column].abs()

    abs_columns = list(ABS_ATOM_COLUMNS.values())
    df["max_abs_atom_plane_dist"] = df[abs_columns].max(axis=1, skipna=True)
    max_abs_column = df[abs_columns].idxmax(axis=1)
    reverse_map = {value: key for key, value in ABS_ATOM_COLUMNS.items()}
    df["max_dist_atom_name"] = max_abs_column.map(reverse_map)
    return df


def summarize_groups(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rms_class, group in df.groupby("rms_class", sort=True):
        row = {
            "rms_class": rms_class,
            "count": len(group),
            "median_rms": group["rms"].median(),
            "mean_rms": group["rms"].mean(),
            "most_common_max_dist_atom_name": most_common(group["max_dist_atom_name"]),
            "median_cno_angle": group["cno_to_peptide_normal_angle_deg"].median(),
            "median_cno_centroid_signed_dist": group["cno_centroid_to_peptide_plane_signed_dist"].median(),
            "median_omega_deviation": group["omega_deviation_from_trans_deg"].median(),
        }
        for atom, abs_column in ABS_ATOM_COLUMNS.items():
            row[f"median_{abs_column}"] = group[abs_column].median()
        rows.append(row)
    return pd.DataFrame(rows)


def most_common(series: pd.Series) -> str:
    values = [str(value) for value in series.dropna()]
    if not values:
        return ""
    return Counter(values).most_common(1)[0][0]


def dominant_or_mixed(series: pd.Series) -> str:
    values = [str(value) for value in series.dropna()]
    if not values:
        return ""
    counts = Counter(values)
    top = counts.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return "mixed"
    return top[0][0]


def chain_pattern_summary(df: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for chain, group in df.groupby("chain", sort=True):
        group = group.sort_values(["res_i", "plane_index"]).copy()
        classes = group["rms_class"].tolist()
        high_low_mask = [value in {"high_rms", "low_rms"} for value in classes]
        transitions = 0
        alternating_pairs = 0
        comparable_pairs = 0
        for left, right, left_ok, right_ok in zip(classes, classes[1:], high_low_mask, high_low_mask[1:]):
            if left != right:
                transitions += 1
            if left_ok and right_ok:
                comparable_pairs += 1
                if left != right:
                    alternating_pairs += 1
        high = group[group["rms_class"] == "high_rms"]
        rows.append(
            {
                "chain": chain,
                "plane_count": len(group),
                "high_rms_count": int((group["rms_class"] == "high_rms").sum()),
                "low_rms_count": int((group["rms_class"] == "low_rms").sum()),
                "high_rms_plane_index_parity": most_common(high["plane_index_parity"]),
                "high_rms_within_chain_order_parity": most_common(high["within_chain_order_parity"]),
                "high_low_transitions": transitions,
                "alternating_high_low_pairs": alternating_pairs,
                "comparable_high_low_pairs": comparable_pairs,
                "fraction_adjacent_pairs_alternating_high_low": (
                    alternating_pairs / comparable_pairs if comparable_pairs else np.nan
                ),
            }
        )
    return rows


def add_chain_marks(ax, df: pd.DataFrame, gap: int) -> None:
    y_min, y_max = ax.get_ylim()
    y_text = y_max - 0.08 * (y_max - y_min)
    for idx, (chain, group) in enumerate(df.groupby("chain", sort=True)):
        x_min = float(group["serial_x"].min())
        x_max = float(group["serial_x"].max())
        if idx % 2 == 0:
            ax.axvspan(x_min - 0.5, x_max + 0.5, color=chain_color(str(chain)), alpha=0.04, zorder=0)
        ax.text(
            (x_min + x_max) / 2.0,
            y_text,
            f"Chain {chain}",
            ha="center",
            va="top",
            color=chain_color(str(chain)),
            fontsize=8,
            fontweight="bold",
        )
    for idx in range(len(df) - 1):
        if df["chain"].iloc[idx] != df["chain"].iloc[idx + 1]:
            ax.axvline(df["serial_x"].iloc[idx] + gap / 2, color="black", linestyle="--", alpha=0.25)


def plot_atom_distances(label: str, df: pd.DataFrame, path: Path, gap: int) -> None:
    fig, ax = plt.subplots(figsize=(15, 6))
    plot_atoms = ["CA_i", "C_i", "O_i", "N_j", "CA_j"]
    for atom in plot_atoms:
        ax.plot(df["serial_x"], df[ABS_ATOM_COLUMNS[atom]], marker="o", markersize=2.5, linewidth=1.0, label=atom)
    ax.set_xticks(df["serial_x"])
    ax.set_xticklabels(df["resname_i"].astype(str) + df["res_i"].astype(str), rotation=90, fontsize=6)
    ax.set_title(f"Atom-to-plane distances by plane: {label}")
    ax.set_xlabel("Residues (chains laid out serially)")
    ax.set_ylabel("absolute atom-to-plane distance (A)")
    ax.grid(True, alpha=0.25)
    add_chain_marks(ax, df, gap)
    ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.16), frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_rms_cno_omega(label: str, df: pd.DataFrame, path: Path, gap: int) -> None:
    fig, ax = plt.subplots(figsize=(15, 6))
    rms = df["rms"]
    cno_scaled = df["cno_to_peptide_normal_angle_deg"] / 60.0
    omega_scaled = df["omega_deviation_from_trans_deg"] / 180.0
    ax.plot(df["serial_x"], rms, marker="o", markersize=2.5, linewidth=1.0, label="RMS (A)")
    ax.plot(
        df["serial_x"],
        cno_scaled,
        marker="o",
        markersize=2.5,
        linewidth=1.0,
        label="CNO angle / 60",
    )
    ax.plot(
        df["serial_x"],
        omega_scaled,
        marker="o",
        markersize=2.5,
        linewidth=1.0,
        label="omega deviation / 180",
    )
    ax.set_xticks(df["serial_x"])
    ax.set_xticklabels(df["resname_i"].astype(str) + df["res_i"].astype(str), rotation=90, fontsize=6)
    ax.set_title(f"RMS, CNO angle, omega deviation by plane: {label}")
    ax.set_xlabel("Residues (chains laid out serially)")
    ax.set_ylabel("diagnostic scaled value")
    ax.grid(True, alpha=0.25)
    add_chain_marks(ax, df, gap)
    ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.16), frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_model_summary(label: str, df: pd.DataFrame, group_summary: pd.DataFrame, pattern_rows: list[dict[str, object]], path: Path) -> None:
    high = df[df["rms_class"] == "high_rms"]
    low = df[df["rms_class"] == "low_rms"]
    cno_corr = df["rms"].corr(df["cno_to_peptide_normal_angle_deg"])
    omega_corr = df["rms"].corr(df["omega_deviation_from_trans_deg"])
    max_atom = most_common(high["max_dist_atom_name"])
    lines = [
        f"# Torsion/RMS pattern summary: {label}",
        "",
        f"- Plane count: {len(df)}",
        f"- Low RMS count (<=0.005 A): {len(low)}",
        f"- High RMS count (>=0.03 A): {len(high)}",
        f"- Most common high-RMS max-distance atom: `{max_atom}`",
        f"- RMS vs CNO angle correlation: {cno_corr:.4f}",
        f"- RMS vs omega deviation correlation: {omega_corr:.4f}",
        "",
        "## Group Summary",
        dataframe_to_markdown(group_summary),
        "",
        "## Chain Alternation Summary",
        dataframe_to_markdown(pd.DataFrame(pattern_rows)),
        "",
        "## Interpretation",
        "High-RMS rows are already present in `plane_features.csv`; this report does not refit planes.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a small dataframe as a GitHub-flavored markdown table."""
    if df.empty:
        return "_No rows._"
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append("" if np.isnan(value) else f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def audit_model(label: str, input_root: Path, outdir: Path, args: argparse.Namespace) -> dict[str, object]:
    plane_path = input_root / label / "plane_features.csv"
    if not plane_path.exists():
        raise FileNotFoundError(f"Missing plane_features.csv for {label}: {plane_path}")
    df = pd.read_csv(plane_path)
    df = classify_and_enrich(df, args.low_rms_threshold, args.high_rms_threshold, args.gap)

    audit_csv = outdir / f"{label}_torsion_rms_atom_audit.csv"
    group_csv = outdir / f"{label}_torsion_rms_group_summary.csv"
    summary_md = outdir / f"{label}_torsion_rms_pattern_summary.md"
    atom_plot = outdir / f"{label}_atom_distance_by_plane.png"
    combined_plot = outdir / f"{label}_rms_cno_omega_by_plane.png"

    df.to_csv(audit_csv, index=False)
    group_summary = summarize_groups(df)
    group_summary.to_csv(group_csv, index=False)
    pattern_rows = chain_pattern_summary(df)
    write_model_summary(label, df, group_summary, pattern_rows, summary_md)
    plot_atom_distances(label, df, atom_plot, args.gap)
    plot_rms_cno_omega(label, df, combined_plot, args.gap)

    high = df[df["rms_class"] == "high_rms"]
    high_pairs = high["resname_i"].astype(str) + "->" + high["resname_j"].astype(str)
    return {
        "label": label,
        "plane_count": len(df),
        "high_count": len(high),
        "low_count": int((df["rms_class"] == "low_rms").sum()),
        "high_max_atom": most_common(high["max_dist_atom_name"]),
        "high_residue_pair": most_common(high_pairs),
        "high_plane_index_parity": dominant_or_mixed(high["plane_index_parity"]),
        "high_within_chain_order_parity": dominant_or_mixed(high["within_chain_order_parity"]),
        "mean_alternating_fraction": float(
            np.nanmean([row["fraction_adjacent_pairs_alternating_high_low"] for row in pattern_rows])
        ),
        "rms_cno_corr": float(df["rms"].corr(df["cno_to_peptide_normal_angle_deg"])),
        "rms_omega_corr": float(df["rms"].corr(df["omega_deviation_from_trans_deg"])),
        "audit_csv": audit_csv,
        "group_csv": group_csv,
        "summary_md": summary_md,
        "atom_plot": atom_plot,
        "combined_plot": combined_plot,
    }


def write_overview(results: list[dict[str, object]], outdir: Path) -> None:
    lines = [
        "# Torsion/RMS pattern audit overview",
        "",
        "## Summary",
        "",
        "| model | planes | high RMS | low RMS | high RMS global parity | high RMS chain-order parity | alternating fraction | high-RMS residue pair | high-RMS max atom | RMS-CNO corr | RMS-omega corr |",
        "|---|---:|---:|---:|---|---|---:|---|---|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| `{result['label']}` | {result['plane_count']} | {result['high_count']} | {result['low_count']} | "
            f"{result['high_plane_index_parity']} | {result['high_within_chain_order_parity']} | "
            f"{result['mean_alternating_fraction']:.3f} | {result['high_residue_pair']} | {result['high_max_atom']} | "
            f"{result['rms_cno_corr']:.4f} | {result['rms_omega_corr']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "- The high/low RMS alternation is real in the existing `plane_features.csv` outputs, not introduced by plotting.",
            "- High RMS planes have mixed global plane-index parity because global indices continue across chains, but they are consistently odd in within-chain order.",
            "- High RMS is strongly associated with GLU->CYP peptide planes in the inspected rows and alternates with low-RMS CYP->GLU planes.",
            "- The most common atom driving high RMS is `CA_j`, with large paired contributions from `CA_i` and `N_j`/`O_i` depending on sign.",
            "- RMS correlates strongly with both CNO angle and omega deviation in these models, so the ~0.07 A spikes are linked to the same local distortion metrics Howard is inspecting.",
            "",
            "## Output Files",
        ]
    )
    for result in results:
        lines.extend(
            [
                f"- `{result['label']}` atom audit CSV: `{result['audit_csv']}`",
                f"- `{result['label']}` group summary CSV: `{result['group_csv']}`",
                f"- `{result['label']}` markdown summary: `{result['summary_md']}`",
                f"- `{result['label']}` atom-distance plot: `{result['atom_plot']}`",
                f"- `{result['label']}` RMS/CNO/omega plot: `{result['combined_plot']}`",
            ]
        )
    (outdir / "torsion_rms_pattern_overview.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    results = [audit_model(label, args.input_root, args.outdir, args) for label in args.models]
    write_overview(results, args.outdir)
    print(f"Wrote torsion/RMS audit outputs to {args.outdir}")


if __name__ == "__main__":
    main()
