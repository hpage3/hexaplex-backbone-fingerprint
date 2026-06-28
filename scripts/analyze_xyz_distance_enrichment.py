"""Coordinate-only distance enrichment analysis for unlabeled XYZ files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hexaplex_backbone_fingerprint.xyz_distance_enrichment import (
    count_distance_band_pairs,
    infer_strand_count_from_filename,
)
from hexaplex_backbone_fingerprint.xyz_parser import parse_xyz


def main() -> int:
    args = parse_args()
    input_dir = Path(args.xyz_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    xyz_files = sorted(input_dir.glob("*.xyz"))
    if not xyz_files:
        raise SystemExit(f"No .xyz files found in {input_dir}")

    for xyz_file in xyz_files:
        atoms = parse_xyz(xyz_file)
        heavy_atom_count = sum(1 for atom in atoms if atom.element.upper() not in {"H", "D"})
        strand_count = infer_strand_count_from_filename(xyz_file.name)
        for band_name, target in [("C", args.c_target), ("D", args.d_target)]:
            stats = count_distance_band_pairs(atoms, target, args.tol, exclude_hydrogen=True)
            rows.append(
                {
                    "filename": xyz_file.name,
                    "inferred_strand_count": strand_count,
                    "atom_count": len(atoms),
                    "heavy_atom_count": heavy_atom_count,
                    "band_name": band_name,
                    "target_distance": target,
                    "tolerance": args.tol,
                    "candidate_pair_count": stats["candidate_pair_count"],
                    "total_possible_pairs": stats["total_possible_pairs"],
                    "normalized_candidate_fraction": stats["normalized_count"],
                    "min_distance": stats["min_distance"],
                    "median_distance": stats["median_distance"],
                    "mean_distance": stats["mean_distance"],
                    "max_distance": stats["max_distance"],
                    "median_abs_error": stats["median_abs_error"],
                }
            )

    df = pd.DataFrame(rows)
    csv_path = outdir / "xyz_distance_enrichment_summary.csv"
    md_path = outdir / "xyz_distance_enrichment_summary.md"
    plot_path = outdir / "xyz_distance_enrichment_plot.png"
    df.to_csv(csv_path, index=False)
    write_markdown_summary(df, md_path, input_dir, args.c_target, args.d_target, args.tol)
    write_enrichment_plot(df, plot_path)

    print("XYZ coordinate distance enrichment complete")
    print(f"Input directory: {input_dir}")
    print(f"XYZ files analyzed: {len(xyz_files)}")
    print(f"Output directory: {outdir}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xyz_dir", help="Directory containing standard XYZ files.")
    parser.add_argument(
        "--outdir",
        default="outputs/multistrand_xyz_distance_enrichment",
        help="Directory for generated outputs.",
    )
    parser.add_argument("--c-target", type=float, default=5.6, help="C band target distance in Angstrom.")
    parser.add_argument("--d-target", type=float, default=7.3, help="D band target distance in Angstrom.")
    parser.add_argument("--tol", type=float, default=0.25, help="Distance tolerance in Angstrom.")
    return parser.parse_args()


def write_markdown_summary(
    df: pd.DataFrame,
    path: Path,
    input_dir: Path,
    c_target: float,
    d_target: float,
    tolerance: float,
) -> None:
    lines = [
        "# XYZ Coordinate Distance Enrichment Summary",
        "",
        "This is coordinate-only analysis of unlabeled XYZ files.",
        "",
        "It cannot distinguish backbone/base/side-chain atoms or same-strand/cross-strand pairs. "
        "It is intended only as a preliminary enrichment check pending labeled PDB/topology files.",
        "",
        f"- Input directory: `{input_dir}`",
        f"- C target: {c_target:.3f} Angstrom",
        f"- D target: {d_target:.3f} Angstrom",
        f"- Tolerance: +/- {tolerance:.3f} Angstrom",
        "",
        "## Counts By File",
        "",
        "| Filename | Strand count | Atoms | Heavy atoms | Band | Candidate pairs | Total pairs | Normalized fraction | Median distance | Median abs error |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in df.sort_values(["inferred_strand_count", "filename", "band_name"], na_position="last").iterrows():
        lines.append(
            "| "
            f"`{row['filename']}` | "
            f"{_format_optional_int(row['inferred_strand_count'])} | "
            f"{int(row['atom_count'])} | "
            f"{int(row['heavy_atom_count'])} | "
            f"{row['band_name']} | "
            f"{int(row['candidate_pair_count'])} | "
            f"{int(row['total_possible_pairs'])} | "
            f"{row['normalized_candidate_fraction']:.6g} | "
            f"{_format_optional_float(row['median_distance'])} | "
            f"{_format_optional_float(row['median_abs_error'])} |"
        )

    lines.extend(
        [
            "",
            "## Normalized Fractions By Strand Count",
            "",
            "| Strand count | Band | Mean normalized fraction | Files |",
            "|---:|---|---:|---:|",
        ]
    )
    grouped = (
        df.dropna(subset=["inferred_strand_count"])
        .groupby(["inferred_strand_count", "band_name"], as_index=False)
        .agg(
            mean_normalized_fraction=("normalized_candidate_fraction", "mean"),
            files=("filename", "count"),
        )
        .sort_values(["inferred_strand_count", "band_name"])
    )
    for _, row in grouped.iterrows():
        lines.append(
            f"| {int(row['inferred_strand_count'])} | {row['band_name']} | "
            f"{row['mean_normalized_fraction']:.6g} | {int(row['files'])} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- These counts are all-heavy-atom coordinate pair counts only.",
            "- Enrichment near 5.6 or 7.3 Angstrom does not identify the atom type or structural origin.",
            "- A labeled PDB, topology, or XYZ index mapping is still required for peptide-plane or strand-aware analysis.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_enrichment_plot(df: pd.DataFrame, path: Path) -> None:
    plot_df = df.dropna(subset=["inferred_strand_count"]).copy()
    fig, ax = plt.subplots(figsize=(6, 4))
    for band_name in ["C", "D"]:
        band_df = plot_df[plot_df["band_name"] == band_name].sort_values("inferred_strand_count")
        if band_df.empty:
            continue
        ax.plot(
            band_df["inferred_strand_count"],
            band_df["normalized_candidate_fraction"],
            marker="o",
            label=f"{band_name} band",
        )
    ax.set_xlabel("Inferred strand count")
    ax.set_ylabel("Normalized candidate fraction")
    ax.set_title("Coordinate-Only XYZ Distance Enrichment")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _format_optional_float(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.3f}"


def _format_optional_int(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(int(value))


if __name__ == "__main__":
    raise SystemExit(main())
