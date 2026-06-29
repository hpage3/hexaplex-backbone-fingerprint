"""Build and plot Howard-style theta/RMS peptide-plane fingerprints.

The legacy visual convention expects ``*_fingerprint.csv`` files with theta as
the main line and plane RMS overlaid as red y-error bars. This script creates
those compatibility CSVs from the current first-panel outputs, then plots them
using the same layout convention as the older ``plot_fingerprint.py`` helper.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_MODELS = [
    "full_hexaplex_anti_parallel_30deg_ideal_deduped_6chain",
    "central6_loose_initial_0000",
    "pnab_hexaplex_twist30_rise3p38",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create old-style theta/RMS fingerprint CSVs and plots."
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
        "--fingerprint-dir",
        type=Path,
        default=Path("outputs/six_strand_first_panel_oldstyle_fingerprints/fingerprints"),
        help="Output directory for *_fingerprint.csv files.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("outputs/six_strand_first_panel_oldstyle_fingerprints/plots"),
        help="Output directory for old-style fingerprint PNGs.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/six_strand_first_panel_oldstyle_fingerprints/oldstyle_fingerprint_report.md"),
        help="Markdown report path.",
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
        help="Scale factor k for yerr = k * box_rms, or 'auto'.",
    )
    return parser.parse_args()


def read_columns(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def build_fingerprint_csv(
    model_label: str,
    numeric_root: Path,
    visual_root: Path,
    fingerprint_dir: Path,
) -> dict[str, object]:
    plane_path = numeric_root / model_label / "plane_features.csv"
    angle_path = visual_root / model_label / f"{model_label}_boxes_adjacent_angles.csv"
    if not plane_path.exists():
        raise FileNotFoundError(f"Missing plane_features.csv: {plane_path}")
    if not angle_path.exists():
        raise FileNotFoundError(f"Missing adjacent-angle CSV: {angle_path}")

    planes = pd.read_csv(plane_path)
    angles = pd.read_csv(angle_path)

    planes["plane_index"] = planes["plane_index"].astype(int)
    angles["plane_index_A"] = angles["plane_index_A"].astype(int)

    merged = angles.merge(
        planes,
        left_on=["plane_index_A", "chain"],
        right_on=["plane_index", "chain"],
        how="left",
        indicator=True,
    )
    dropped_rows = int((merged["_merge"] != "both").sum())
    if dropped_rows:
        merged = merged[merged["_merge"] == "both"].copy()

    fingerprint = pd.DataFrame(
        {
            "chain": merged["chain"],
            "res_i": merged["res_i_A"].astype(int),
            "aa_i": merged["resname_i"].astype(str) + merged["res_i_A"].astype(int).astype(str),
            "theta_pp_deg": merged["angle_unsigned_deg"],
            "box_rms": merged["rms"],
            "plane_index": merged["plane_index_A"].astype(int),
            "res_j": merged["res_j_A"].astype(int),
            "cno_to_peptide_normal_angle_deg": merged["cno_to_peptide_normal_angle_deg"],
            "omega_deviation_from_trans_deg": merged["omega_deviation_from_trans_deg"],
            "theta_signed_deg": merged["angle_signed_deg"],
            "dihedral_deg": merged["dihedral_deg"],
            "source_model": model_label,
        }
    )
    fingerprint = fingerprint.sort_values(["chain", "res_i"]).reset_index(drop=True)

    fingerprint_dir.mkdir(parents=True, exist_ok=True)
    out_csv = fingerprint_dir / f"{model_label}_fingerprint.csv"
    fingerprint.to_csv(out_csv, index=False)

    per_chain_paths = sorted(
        (visual_root / model_label).glob(f"{model_label}_boxes_chain_*_adjacent_angles.csv")
    )
    per_chain_columns = read_columns(per_chain_paths[0]) if per_chain_paths else []

    return {
        "model_label": model_label,
        "plane_path": plane_path,
        "angle_path": angle_path,
        "fingerprint_csv": out_csv,
        "plane_columns": read_columns(plane_path),
        "angle_columns": read_columns(angle_path),
        "per_chain_columns": per_chain_columns,
        "input_angle_rows": len(angles),
        "output_rows": len(fingerprint),
        "dropped_rows": dropped_rows,
        "join_method": "aggregate adjacent plane_index_A + chain -> plane_features plane_index + chain",
    }


def auto_rms_scale(df: pd.DataFrame) -> float:
    rms_vals = df["box_rms"].fillna(0.0).to_numpy()
    max_rms = float(np.nanmax(rms_vals))
    theta_range = float(df["theta_pp_deg"].max() - df["theta_pp_deg"].min())
    if max_rms > 0:
        rms_scale = 0.15 * theta_range / max_rms
    else:
        rms_scale = 1000.0
    return max(500.0, min(rms_scale, 10000.0))


def plot_fingerprint(csv_path: Path, outdir: Path, gap: int, rms_as_yerr: str) -> dict[str, object]:
    model_label = csv_path.name.replace("_fingerprint.csv", "")
    df = pd.read_csv(csv_path)
    df = df.sort_values(["chain", "res_i"]).reset_index(drop=True)

    serial_x: list[int] = []
    offset = 0
    last_chain = None
    for _, row in df.iterrows():
        if last_chain is not None and row["chain"] != last_chain:
            offset += gap
        serial_x.append(len(serial_x) + offset)
        last_chain = row["chain"]
    df["serial_x"] = serial_x

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
    for idx in range(len(df) - 1):
        if df["chain"].iloc[idx] != df["chain"].iloc[idx + 1]:
            ax.axvline(
                df["serial_x"].iloc[idx] + gap / 2,
                color="red",
                linestyle="--",
                alpha=0.5,
            )

    ax.set_title(f"Theta/RMS fingerprint for {model_label}")
    ax.set_xlabel("Residues (chains laid out serially)")
    ax.set_ylabel("theta_pp (deg)")
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
    return {"model_label": model_label, "plot_path": out_png, "rms_scale": rms_scale}


def write_report(report_path: Path, build_results: list[dict[str, object]], plot_results: list[dict[str, object]]) -> None:
    plot_by_label = {row["model_label"]: row for row in plot_results}
    lines = [
        "# Old-style theta/RMS fingerprint report",
        "",
        "These files adapt the current peptide-plane outputs into Howard's older fingerprint plotting convention.",
        "",
        "## Schema Mapping",
        "- `chain`: from both source tables; join requires it to match.",
        "- `res_i`: `res_i_A` from the legacy aggregate adjacent-angle CSV.",
        "- `aa_i`: `resname_i + res_i_A` from `plane_features.csv` after joining.",
        "- `theta_pp_deg`: `angle_unsigned_deg` from the legacy aggregate adjacent-angle CSV.",
        "- `box_rms`: `rms` from `plane_features.csv`.",
        "- Optional columns: `plane_index`, `res_j`, CNO angle, omega deviation, signed theta, dihedral, and source model.",
        "",
        "## Join Method",
        "The aggregate `*_boxes_adjacent_angles.csv` files use zero-based `plane_index_A`, which matches `plane_features.csv:plane_index`. The per-chain adjacent-angle CSV files use one-based chain-local indices, so they were inspected but not used for row alignment.",
        "",
    ]

    for result in build_results:
        label = result["model_label"]
        plot = plot_by_label[label]
        lines.extend(
            [
                f"## {label}",
                f"- Plane features: `{result['plane_path']}`",
                f"- Adjacent angles: `{result['angle_path']}`",
                f"- Fingerprint CSV: `{result['fingerprint_csv']}`",
                f"- Fingerprint PNG: `{plot['plot_path']}`",
                f"- Plane feature columns: `{', '.join(result['plane_columns'])}`",
                f"- Adjacent-angle columns: `{', '.join(result['angle_columns'])}`",
                f"- Per-chain adjacent-angle columns inspected: `{', '.join(result['per_chain_columns'])}`",
                f"- Input adjacent rows: {result['input_angle_rows']}",
                f"- Output fingerprint rows: {result['output_rows']}",
                f"- Rows dropped during join: {result['dropped_rows']}",
                f"- RMS y-error scale: {float(plot['rms_scale']):.3f}",
                "- Warning: none; row alignment used direct zero-based plane index plus chain.",
                "",
            ]
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    build_results = [
        build_fingerprint_csv(model, args.numeric_root, args.visual_root, args.fingerprint_dir)
        for model in args.models
    ]
    plot_results = [
        plot_fingerprint(
            Path(result["fingerprint_csv"]),
            args.outdir,
            args.gap,
            args.rms_as_yerr,
        )
        for result in build_results
    ]
    write_report(args.report, build_results, plot_results)
    print(f"Wrote {len(build_results)} old-style fingerprint CSVs to {args.fingerprint_dir}")
    print(f"Wrote {len(plot_results)} old-style fingerprint plots to {args.outdir}")
    print(f"Wrote report to {args.report}")


if __name__ == "__main__":
    main()
