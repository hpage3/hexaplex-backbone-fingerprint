"""Build and plot Howard-style peptide-plane theta/RMS fingerprints.

The legacy visual convention lays chains out serially, separated by gaps, and
plots one peptide-plane angle value per adjacent plane pair. This script creates
auditable ``*_fingerprint.csv`` compatibility files from either the legacy
adjacent-angle CSVs or the theta-sign audit CSVs, then emits angle-only and
torsion-only plots. The old combined theta/RMS y-error plot is still available
with ``--make-combined``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_MODELS = [
    "full_hexaplex_anti_parallel_30deg_ideal_deduped_6chain",
    "central6_loose_initial_0000",
    "pnab_hexaplex_twist30_rise3p38",
]

THETA_SOURCE_COLUMNS = {
    "signed": "angle_signed_deg",
    "unsigned": "angle_unsigned_deg",
    "dihedral": "dihedral_deg",
    "continuity_backbone_signed": "continuity_backbone_signed",
    "continuity_sign_only_preserve_magnitude": "continuity_sign_only_preserve_magnitude",
}

AUDIT_THETA_SOURCES = {"continuity_backbone_signed", "continuity_sign_only_preserve_magnitude"}
CHAIN_COLORS = {
    "A": "#1f77b4",
    "B": "#ff7f0e",
    "C": "#2ca02c",
    "D": "#d62728",
    "E": "#9467bd",
    "F": "#8c564b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create old-style peptide-plane theta/RMS fingerprint CSVs and plots."
    )
    parser.add_argument(
        "--numeric-root",
        type=Path,
        default=Path("outputs/six_strand_first_panel"),
        help="Directory containing per-model plane_features.csv files.",
    )
    parser.add_argument(
        "--visual-root",
        type=Path,
        default=Path("outputs/six_strand_first_panel_visual_boxes"),
        help="Directory containing legacy peptide-box adjacent-angle CSVs.",
    )
    parser.add_argument(
        "--audit-root",
        type=Path,
        default=Path("outputs/theta_sign_audit"),
        help="Directory containing *_theta_sign_audit.csv files.",
    )
    parser.add_argument(
        "--fingerprint-dir",
        type=Path,
        default=Path("outputs/six_strand_first_panel_oldstyle_fingerprints_signed/fingerprints"),
        help="Output directory for *_fingerprint.csv files.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("outputs/six_strand_first_panel_oldstyle_fingerprints_signed/plots"),
        help="Output directory for PNG plots.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/six_strand_first_panel_oldstyle_fingerprints_signed/oldstyle_fingerprint_report.md"),
        help="Markdown report path.",
    )
    parser.add_argument(
        "--theta-source",
        choices=sorted(THETA_SOURCE_COLUMNS),
        default="signed",
        help="Theta-like adjacent-plane column to plot. Defaults to signed theta-pp.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=DEFAULT_MODELS,
        help="Model labels to process.",
    )
    parser.add_argument("--gap", type=int, default=5, help="x-gap between chains.")
    parser.add_argument(
        "--rms-as-yerr",
        default="auto",
        help="Scale factor k for combined plot yerr = k * box_rms, or 'auto'.",
    )
    parser.add_argument(
        "--make-combined",
        action="store_true",
        help="Also create the legacy combined theta/RMS y-error plot.",
    )
    return parser.parse_args()


def read_columns(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def load_theta_table(
    model_label: str,
    visual_root: Path,
    audit_root: Path,
    theta_source: str,
) -> tuple[pd.DataFrame, Path, str, list[str]]:
    if theta_source in AUDIT_THETA_SOURCES:
        path = audit_root / f"{model_label}_theta_sign_audit.csv"
        source_kind = "theta audit"
        per_chain_columns: list[str] = []
    else:
        path = visual_root / model_label / f"{model_label}_boxes_adjacent_angles.csv"
        source_kind = "legacy adjacent angles"
        per_chain_paths = sorted(
            (visual_root / model_label).glob(f"{model_label}_boxes_chain_*_adjacent_angles.csv")
        )
        per_chain_columns = read_columns(per_chain_paths[0]) if per_chain_paths else []

    if not path.exists():
        raise FileNotFoundError(f"Missing theta input table: {path}")
    return pd.read_csv(path), path, source_kind, per_chain_columns


def build_fingerprint_csv(
    model_label: str,
    numeric_root: Path,
    visual_root: Path,
    audit_root: Path,
    fingerprint_dir: Path,
    theta_source: str,
) -> dict[str, object]:
    plane_path = numeric_root / model_label / "plane_features.csv"
    if not plane_path.exists():
        raise FileNotFoundError(f"Missing plane_features.csv: {plane_path}")

    planes = pd.read_csv(plane_path)
    theta_table, theta_path, source_kind, per_chain_columns = load_theta_table(
        model_label, visual_root, audit_root, theta_source
    )
    theta_column = THETA_SOURCE_COLUMNS[theta_source]
    if theta_column not in theta_table.columns:
        raise ValueError(f"Missing theta source column {theta_column!r} in {theta_path}")

    planes["plane_index"] = planes["plane_index"].astype(int)
    theta_table["plane_index_A"] = theta_table["plane_index_A"].astype(int)

    merged = theta_table.merge(
        planes,
        left_on=["plane_index_A", "chain"],
        right_on=["plane_index", "chain"],
        how="left",
        indicator=True,
    )
    dropped_rows = int((merged["_merge"] != "both").sum())
    if dropped_rows:
        merged = merged[merged["_merge"] == "both"].copy()

    fingerprint_data = {
        "chain": merged["chain"],
        "res_i": merged["res_i_A"].astype(int),
        "aa_i": merged["resname_i"].astype(str) + merged["res_i_A"].astype(int).astype(str),
        "theta_pp_deg": merged[theta_column],
        "theta_source": theta_source,
        "box_rms": merged["rms"],
        "plane_index": merged["plane_index_A"].astype(int),
        "res_j": merged["res_j_A"].astype(int),
        "cno_to_peptide_normal_angle_deg": merged["cno_to_peptide_normal_angle_deg"],
        "omega_deviation_from_trans_deg": merged["omega_deviation_from_trans_deg"],
        "source_model": model_label,
    }
    for column in [
        "angle_unsigned_deg",
        "angle_signed_deg",
        "dihedral_deg",
        "current_signed",
        "unsigned",
        "continuity_signed",
        "backbone_axis_signed",
        "continuity_backbone_signed",
        "continuity_sign_only_preserve_magnitude",
    ]:
        if column in merged.columns:
            fingerprint_data[column] = merged[column]

    fingerprint = pd.DataFrame(fingerprint_data)
    fingerprint = fingerprint.sort_values(["chain", "res_i"]).reset_index(drop=True)

    fingerprint_dir.mkdir(parents=True, exist_ok=True)
    out_csv = fingerprint_dir / f"{model_label}_fingerprint.csv"
    fingerprint.to_csv(out_csv, index=False)

    return {
        "model_label": model_label,
        "plane_path": plane_path,
        "theta_path": theta_path,
        "fingerprint_csv": out_csv,
        "plane_columns": read_columns(plane_path),
        "theta_columns": read_columns(theta_path),
        "per_chain_columns": per_chain_columns,
        "input_theta_rows": len(theta_table),
        "output_rows": len(fingerprint),
        "dropped_rows": dropped_rows,
        "theta_source": theta_source,
        "theta_column": theta_column,
        "source_kind": source_kind,
        "join_method": "theta table plane_index_A + chain -> plane_features plane_index + chain",
    }


def add_serial_x(df: pd.DataFrame, gap: int) -> pd.DataFrame:
    df = df.sort_values(["chain", "res_i"]).reset_index(drop=True).copy()
    serial_x: list[int] = []
    offset = 0
    last_chain = None
    for _, row in df.iterrows():
        if last_chain is not None and row["chain"] != last_chain:
            offset += gap
        serial_x.append(len(serial_x) + offset)
        last_chain = row["chain"]
    df["serial_x"] = serial_x
    return df


def add_chain_breaks(ax, df: pd.DataFrame, gap: int) -> None:
    for idx in range(len(df) - 1):
        if df["chain"].iloc[idx] != df["chain"].iloc[idx + 1]:
            ax.axvline(
                df["serial_x"].iloc[idx] + gap / 2,
                color="red",
                linestyle="--",
                alpha=0.5,
            )


def chain_color(chain: str) -> str:
    if chain in CHAIN_COLORS:
        return CHAIN_COLORS[chain]
    keys = sorted(CHAIN_COLORS)
    return CHAIN_COLORS[keys[hash(chain) % len(keys)]]


def plot_chain_segments(ax, df: pd.DataFrame, y_column: str) -> None:
    for chain, chain_df in df.groupby("chain", sort=True):
        ax.plot(
            chain_df["serial_x"],
            chain_df[y_column],
            "-o",
            linewidth=1.1,
            markersize=3,
            color=chain_color(str(chain)),
            label=f"Chain {chain}",
        )


def add_chain_labels(ax, df: pd.DataFrame) -> None:
    y_min, y_max = ax.get_ylim()
    y_text = y_max - 0.08 * (y_max - y_min)
    for idx, (chain, chain_df) in enumerate(df.groupby("chain", sort=True)):
        x_min = float(chain_df["serial_x"].min())
        x_max = float(chain_df["serial_x"].max())
        color = chain_color(str(chain))
        if idx % 2 == 0:
            ax.axvspan(x_min - 0.5, x_max + 0.5, color=color, alpha=0.05, zorder=0)
        ax.text(
            (x_min + x_max) / 2.0,
            y_text,
            f"Chain {chain}",
            ha="center",
            va="top",
            fontsize=8,
            fontweight="bold",
            color=color,
        )


def plot_peptide_angle(csv_path: Path, outdir: Path, gap: int) -> dict[str, object]:
    model_label = csv_path.name.replace("_fingerprint.csv", "")
    df = add_serial_x(pd.read_csv(csv_path), gap)
    theta_source = str(df["theta_source"].iloc[0]) if len(df) else "unknown"
    diagnostic = theta_source in AUDIT_THETA_SOURCES

    fig, ax = plt.subplots(figsize=(14, 5))
    plot_chain_segments(ax, df, "theta_pp_deg")
    ax.set_xticks(df["serial_x"])
    ax.set_xticklabels(df["aa_i"], rotation=90, fontsize=6)
    add_chain_breaks(ax, df, gap)
    ax.axhline(0, color="black", linewidth=0.7, alpha=0.4)
    if theta_source == "continuity_backbone_signed":
        title = f"Peptide angle plot for {model_label} (diagnostic continuity-backbone signed theta-pp)"
        ylabel = "theta-pp diagnostic (deg)"
    elif theta_source == "continuity_sign_only_preserve_magnitude":
        title = f"Peptide angle plot for {model_label} (diagnostic preserve-magnitude signed theta-pp)"
        ylabel = "theta-pp diagnostic (deg)"
    else:
        title = f"Peptide angle plot for {model_label} ({theta_source})"
        ylabel = "theta-pp (deg)"
    ax.set_title(title)
    ax.set_xlabel("Residues (chains laid out serially)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    add_chain_labels(ax, df)
    ax.legend(ncol=6, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=False)
    fig.tight_layout()

    outdir.mkdir(parents=True, exist_ok=True)
    out_png = outdir / f"{model_label}_peptide_angle_plot.png"
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    return {"model_label": model_label, "angle_plot_path": out_png}


def plot_torsion(csv_path: Path, outdir: Path, gap: int) -> dict[str, object]:
    model_label = csv_path.name.replace("_fingerprint.csv", "")
    df = add_serial_x(pd.read_csv(csv_path), gap)

    fig, ax = plt.subplots(figsize=(14, 5))
    plot_chain_segments(ax, df, "box_rms")
    ax.set_xticks(df["serial_x"])
    ax.set_xticklabels(df["aa_i"], rotation=90, fontsize=6)
    add_chain_breaks(ax, df, gap)
    ax.set_title(f"Torsion plot for {model_label}")
    ax.set_xlabel("Residues (chains laid out serially)")
    ax.set_ylabel("torsion / backbone strain (RMS)")
    ax.grid(True, alpha=0.25)
    add_chain_labels(ax, df)
    ax.legend(ncol=6, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=False)
    fig.tight_layout()

    outdir.mkdir(parents=True, exist_ok=True)
    out_png = outdir / f"{model_label}_torsion_plot.png"
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    return {"model_label": model_label, "torsion_plot_path": out_png}


def auto_rms_scale(df: pd.DataFrame) -> float:
    rms_vals = df["box_rms"].fillna(0.0).to_numpy()
    max_rms = float(np.nanmax(rms_vals))
    theta_range = float(df["theta_pp_deg"].max() - df["theta_pp_deg"].min())
    if max_rms > 0:
        rms_scale = 0.15 * theta_range / max_rms
    else:
        rms_scale = 1000.0
    return max(500.0, min(rms_scale, 10000.0))


def plot_combined_fingerprint(csv_path: Path, outdir: Path, gap: int, rms_as_yerr: str) -> dict[str, object]:
    model_label = csv_path.name.replace("_fingerprint.csv", "")
    df = add_serial_x(pd.read_csv(csv_path), gap)
    theta_source = str(df["theta_source"].iloc[0]) if len(df) else "unknown"
    auto_scale = rms_as_yerr is None or str(rms_as_yerr).lower() == "auto"
    rms_scale = auto_rms_scale(df) if auto_scale else float(rms_as_yerr)

    fig, ax = plt.subplots(figsize=(14, 5))
    yerr = (df["box_rms"].fillna(0.0) * rms_scale).values
    ax.errorbar(
        df["serial_x"],
        df["theta_pp_deg"],
        yerr=yerr,
        ecolor="red",
        fmt="-o",
        linewidth=1.0,
        markersize=3,
    )
    ax.set_xticks(df["serial_x"])
    ax.set_xticklabels(df["aa_i"], rotation=90, fontsize=6)
    add_chain_breaks(ax, df, gap)
    source_note = (
        "diagnostic continuity-backbone signed theta-pp"
        if theta_source == "continuity_backbone_signed"
        else theta_source
    )
    ax.set_title(f"Theta/RMS fingerprint for {model_label} ({source_note})")
    ax.set_xlabel("Residues (chains laid out serially)")
    ax.set_ylabel(f"theta-pp (deg), source={theta_source}")
    if auto_scale:
        ax.text(
            0.01,
            0.95,
            f"Auto RMS x {rms_scale:.1f}",
            transform=ax.transAxes,
            fontsize=8,
            color="red",
            va="top",
        )
    fig.tight_layout()

    outdir.mkdir(parents=True, exist_ok=True)
    out_png = outdir / f"{model_label}_fingerprint.png"
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    return {"model_label": model_label, "combined_plot_path": out_png, "rms_scale": rms_scale}


def count_sign_changes(values: pd.Series) -> int:
    count = 0
    previous = 0
    for value in values:
        sign = 1 if value > 0 else -1 if value < 0 else 0
        if previous and sign and sign != previous:
            count += 1
        if sign:
            previous = sign
    return count


def count_jumps(values: pd.Series, threshold: float = 150.0) -> int:
    return int((values.diff().abs() > threshold).sum())


def diagnostic_stats(csv_path: Path) -> dict[str, object]:
    df = pd.read_csv(csv_path).sort_values(["chain", "res_i"]).reset_index(drop=True)
    theta = df["theta_pp_deg"]
    before_jumps = 0
    after_jumps = 0
    before_sign_changes = 0
    after_sign_changes = 0
    for _, chain_df in df.groupby("chain", sort=True):
        if "current_signed" in chain_df.columns:
            before_jumps += count_jumps(chain_df["current_signed"])
            before_sign_changes += count_sign_changes(chain_df["current_signed"])
        after_jumps += count_jumps(chain_df["theta_pp_deg"])
        after_sign_changes += count_sign_changes(chain_df["theta_pp_deg"])

    return {
        "model_label": csv_path.name.replace("_fingerprint.csv", ""),
        "rows": len(df),
        "min_theta": float(theta.min()),
        "median_theta": float(theta.median()),
        "max_theta": float(theta.max()),
        "count_gt_90_abs": int((theta.abs() > 90.0).sum()),
        "count_90_130_abs": int(((theta.abs() >= 90.0) & (theta.abs() <= 130.0)).sum()),
        "before_jumps": before_jumps,
        "after_jumps": after_jumps,
        "before_sign_changes": before_sign_changes,
        "after_sign_changes": after_sign_changes,
    }


def write_report(
    report_path: Path,
    build_results: list[dict[str, object]],
    plot_results: list[dict[str, object]],
    theta_source: str,
) -> None:
    plot_by_label = {row["model_label"]: row for row in plot_results}
    diagnostic = theta_source in AUDIT_THETA_SOURCES
    title = (
        "# Diagnostic theta report"
        if diagnostic
        else "# Old-style peptide angle/torsion fingerprint report"
    )
    lines = [
        title,
        "",
        "- Theta source used: `" + theta_source + "`",
    ]
    if diagnostic:
        lines.extend(
            [
                "- This is a diagnostic corrected theta source pending controls and/or comparison to the original Howard/Loren implementation.",
                "- It should not yet be described as final manuscript theta-pp.",
            ]
        )
    if theta_source == "continuity_backbone_signed":
        lines.append(
            "- Note: this source flips adjacent normals to positive dot products before angle calculation; it removes large jumps but folds obtuse beta-like angles toward acute complements."
        )
    elif theta_source == "continuity_sign_only_preserve_magnitude":
        lines.append(
            "- Note: this source preserves the raw 0..180 degree inter-plane magnitude and assigns sign separately using the local backbone propagation axis."
        )
    lines.extend(
        [
            "",
            "## Schema Mapping",
            "- `chain`, `res_i`, `aa_i`, and `box_rms` come from `plane_features.csv` after joining.",
            "- `theta_pp_deg` comes from the selected theta source table.",
            "- Join method: theta table `plane_index_A + chain` to `plane_features.csv` `plane_index + chain`.",
            "",
        ]
    )

    if diagnostic:
        lines.extend(
            [
                "## Before/After Diagnostics",
                "",
                "| model | rows | current jumps >150 deg | corrected jumps >150 deg | current sign changes | corrected sign changes | min theta | median theta | max theta | |theta| > 90 | |theta| 90-130 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for result in build_results:
            stats = diagnostic_stats(Path(result["fingerprint_csv"]))
            lines.append(
                f"| `{stats['model_label']}` | {stats['rows']} | {stats['before_jumps']} | {stats['after_jumps']} | "
                f"{stats['before_sign_changes']} | {stats['after_sign_changes']} | "
                f"{stats['min_theta']:.3f} | {stats['median_theta']:.3f} | {stats['max_theta']:.3f} | "
                f"{stats['count_gt_90_abs']} | {stats['count_90_130_abs']} |"
            )
        if theta_source == "continuity_backbone_signed":
            lines.extend(
                [
                    "",
                    "The continuity-backbone diagnostic reduces artificial >150 degree jumps, but folds obtuse beta-like angle magnitudes into acute complements.",
                    "",
                ]
            )
        elif theta_source == "continuity_sign_only_preserve_magnitude":
            lines.extend(
                [
                    "",
                    "The preserve-magnitude diagnostic restores obtuse beta-like angle magnitudes while assigning sign separately. It remains diagnostic pending controls.",
                    "",
                ]
            )

    for result in build_results:
        label = result["model_label"]
        plot = plot_by_label[label]
        lines.extend(
            [
                f"## {label}",
                f"- Plane features: `{result['plane_path']}`",
                f"- Theta table: `{result['theta_path']}`",
                f"- Fingerprint CSV: `{result['fingerprint_csv']}`",
                f"- Peptide angle plot: `{plot['angle_plot_path']}`",
                f"- Torsion plot: `{plot['torsion_plot_path']}`",
                f"- Combined plot: `{plot.get('combined_plot_path', '')}`",
                f"- Source table kind: `{result['source_kind']}`",
                f"- Theta column: `{result['theta_column']}`",
                f"- Input theta rows: {result['input_theta_rows']}",
                f"- Output fingerprint rows: {result['output_rows']}",
                f"- Rows dropped during join: {result['dropped_rows']}",
                "",
            ]
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    build_results = [
        build_fingerprint_csv(
            model,
            args.numeric_root,
            args.visual_root,
            args.audit_root,
            args.fingerprint_dir,
            args.theta_source,
        )
        for model in args.models
    ]

    plot_results = []
    for result in build_results:
        csv_path = Path(result["fingerprint_csv"])
        angle_plot = plot_peptide_angle(csv_path, args.outdir, args.gap)
        torsion_plot = plot_torsion(csv_path, args.outdir, args.gap)
        plot_result = {
            "model_label": result["model_label"],
            "angle_plot_path": angle_plot["angle_plot_path"],
            "torsion_plot_path": torsion_plot["torsion_plot_path"],
        }
        if args.make_combined:
            plot_result.update(plot_combined_fingerprint(csv_path, args.outdir, args.gap, args.rms_as_yerr))
        plot_results.append(plot_result)

    write_report(args.report, build_results, plot_results, args.theta_source)
    print(f"Wrote {len(build_results)} fingerprint CSVs to {args.fingerprint_dir}")
    print(f"Wrote peptide angle and torsion plots to {args.outdir}")
    print(f"Wrote report to {args.report}")


if __name__ == "__main__":
    main()
