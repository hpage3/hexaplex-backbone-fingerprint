"""Roll up rich-coordinate C/D pair-family diagnostics and Debye scores."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hexaplex_backbone_fingerprint.parametric_powder_scan import (
    debye_profile,
    make_q_grid,
    nearest_peak,
)


FOCUS_FAMILIES = [
    "same_strand_plusminus1_repeat",
    "adjacent_strand_same_register",
    "all_cross_strand",
    "all_same_strand",
    "alternating_interfaces_AB_CD_EF",
    "alternating_interfaces_BC_DE_FA",
]


def read_pdb_coordinates(path: Path, exclude_hydrogen: bool = False) -> np.ndarray:
    """Read coordinates from ATOM/HETATM records in a PDB file."""
    coords = []
    for line in path.read_text(encoding="ascii").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        atom_name = line[12:16].strip()
        element = (line[76:78].strip() or atom_name[:1]).upper()
        if exclude_hydrogen and (element == "H" or atom_name.upper().startswith("H")):
            continue
        coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    if not coords:
        raise ValueError(f"No coordinates found in {path}.")
    return np.asarray(coords, dtype=float)


def numeric_value(df: pd.DataFrame, family: str, column: str) -> float:
    """Return a numeric value from one family row, or NaN."""
    if column not in df.columns:
        return float("nan")
    subset = df[df["family"] == family]
    if subset.empty:
        return float("nan")
    value = pd.to_numeric(subset.iloc[0][column], errors="coerce")
    return float(value) if pd.notna(value) else float("nan")


def top_family(df: pd.DataFrame, column: str) -> tuple[str, float]:
    """Return the family with the maximum numeric column value."""
    if column not in df.columns or df.empty:
        return "", float("nan")
    values = pd.to_numeric(df[column], errors="coerce").fillna(float("-inf"))
    if values.max() == float("-inf"):
        return "", float("nan")
    idx = values.idxmax()
    return str(df.loc[idx, "family"]), float(values.loc[idx])


def score_pdb_profile(
    pdb_path: Path,
    target_c: float,
    target_d: float,
    tolerance: float,
    q_step: float,
    d_min: float,
    d_max: float,
) -> dict[str, object]:
    """Compute a direct point-scatterer Debye profile score for a PDB variant."""
    coords = read_pdb_coordinates(pdb_path, exclude_hydrogen=False)
    q_values = make_q_grid(d_min_A=d_min, d_max_A=d_max, q_step=q_step)
    profile = debye_profile(coords, q_values)
    c_hit = nearest_peak(profile, target_c, tolerance)
    d_hit = nearest_peak(profile, target_d, tolerance)
    return {
        "atom_count_scored": len(coords),
        "C_peak_d_A": c_hit.peak_d_A,
        "D_peak_d_A": d_hit.peak_d_A,
        "C_error_A": c_hit.error_A,
        "D_error_A": d_hit.error_A,
        "C_peak_intensity": c_hit.intensity,
        "D_peak_intensity": d_hit.intensity,
        "C_found_within_tolerance": c_hit.found_within_tolerance,
        "D_found_within_tolerance": d_hit.found_within_tolerance,
    }


def summarize_variant(
    manifest_row: pd.Series,
    metrics_dir: Path,
    target_c: float,
    target_d: float,
    tolerance: float,
    q_step: float,
    d_min: float,
    d_max: float,
) -> dict[str, object]:
    """Summarize one generated coordinate variant."""
    model_id = str(manifest_row["model_id"])
    summary_path = metrics_dir / f"{model_id}_pair_family_cd_summary.csv"
    row: dict[str, object] = {
        "variant": manifest_row.get("variant", ""),
        "model_id": model_id,
        "pdb_path": manifest_row.get("pdb_path", ""),
        "variant_atom_count": manifest_row.get("atom_count", ""),
        "variant_warnings": manifest_row.get("warnings", ""),
        "pair_family_summary_path": str(summary_path) if summary_path.exists() else "",
    }
    if not summary_path.exists():
        row["warnings"] = f"Missing pair-family summary: {summary_path}"
        return row

    df = pd.read_csv(summary_path)
    top_c, top_c_count = top_family(df, "C_pair_count")
    top_d, top_d_count = top_family(df, "D_pair_count")
    row.update(
        {
            "top_C_pair_family": top_c,
            "top_C_pair_count": top_c_count,
            "top_D_pair_family": top_d,
            "top_D_pair_count": top_d_count,
            "AB_CD_EF_C_pair_count": numeric_value(df, "alternating_interfaces_AB_CD_EF", "C_pair_count"),
            "AB_CD_EF_D_pair_count": numeric_value(df, "alternating_interfaces_AB_CD_EF", "D_pair_count"),
            "BC_DE_FA_C_pair_count": numeric_value(df, "alternating_interfaces_BC_DE_FA", "C_pair_count"),
            "BC_DE_FA_D_pair_count": numeric_value(df, "alternating_interfaces_BC_DE_FA", "D_pair_count"),
        }
    )
    for family in FOCUS_FAMILIES:
        row[f"{family}_C_pair_count"] = numeric_value(df, family, "C_pair_count")
        row[f"{family}_D_pair_count"] = numeric_value(df, family, "D_pair_count")

    pdb_path = Path(str(manifest_row.get("pdb_path", "")))
    if pdb_path.exists():
        row.update(score_pdb_profile(pdb_path, target_c, target_d, tolerance, q_step, d_min, d_max))
    else:
        row["warnings"] = f"Missing PDB variant: {pdb_path}"
    return row


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    """Render selected dataframe columns as a markdown table."""
    columns = [column for column in columns if column in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for record in df[columns].itertuples(index=False):
        values = []
        for value in record:
            if isinstance(value, float):
                values.append(f"{value:.4g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def save_c_vs_d_plot(rollup: pd.DataFrame, path_base: Path) -> None:
    """Save C/D peak comparison as PNG and SVG."""
    fig, ax = plt.subplots(figsize=(7, 5))
    valid = rollup.dropna(subset=["C_peak_d_A", "D_peak_d_A"])
    for row in valid.itertuples():
        ax.scatter(row.C_peak_d_A, row.D_peak_d_A, s=80)
        ax.annotate(str(row.variant), (row.C_peak_d_A, row.D_peak_d_A), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.axvline(5.6, color="#1f77b4", ls="--", lw=1, label="C target 5.6 A")
    ax.axhline(7.3, color="#ff7f0e", ls="--", lw=1, label="D target 7.3 A")
    ax.set_xlabel("nearest C-like peak d spacing (A)")
    ax.set_ylabel("nearest D-like peak d spacing (A)")
    ax.set_title("Rich-coordinate diagnostic C/D peak positions")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".png"), dpi=180)
    fig.savefig(path_base.with_suffix(".svg"))
    plt.close(fig)


def write_report(rollup: pd.DataFrame, path: Path) -> None:
    """Write concise markdown interpretation for rich-coordinate diagnostics."""
    if rollup.empty:
        path.write_text("# Rich-Coordinate C/D Rollup\n\nNo variants were summarized.\n", encoding="utf-8")
        return

    full = rollup[rollup["variant"] == "full"]
    full_c = full.iloc[0]["C_peak_d_A"] if not full.empty and "C_peak_d_A" in full else float("nan")
    best_c = rollup.loc[pd.to_numeric(rollup["C_error_A"], errors="coerce").abs().idxmin()] if "C_error_A" in rollup else None
    d_cross = rollup["top_D_pair_family"].astype(str).str.contains("cross|adjacent", case=False, regex=True).sum()
    c_cross = rollup["top_C_pair_family"].astype(str).str.contains("cross|adjacent", case=False, regex=True).sum()
    c_local = rollup["top_C_pair_family"].astype(str).str.startswith("same_strand").sum()

    summary_table = markdown_table(
        rollup,
        [
            "variant",
            "variant_atom_count",
            "C_peak_d_A",
            "C_error_A",
            "D_peak_d_A",
            "D_error_A",
            "top_C_pair_family",
            "top_D_pair_family",
            "AB_CD_EF_C_pair_count",
            "BC_DE_FA_C_pair_count",
            "AB_CD_EF_D_pair_count",
            "BC_DE_FA_D_pair_count",
        ],
    )
    best_text = (
        f"`{best_c['variant']}` (C peak {best_c['C_peak_d_A']:.3f} A; error {best_c['C_error_A']:.3f} A)"
        if best_c is not None and pd.notna(best_c.get("C_peak_d_A"))
        else "not available"
    )
    text = f"""# Rich-Coordinate C/D Diagnostic Rollup

This rollup uses the ideal/full Hexaflex/Hexaplex coordinate file as a controlled parent model for atom-selection variants. It is diagnostic only and should not be read as experimental structural truth.

## Variant Summary

{summary_table}

## Interpretation Questions

- Does the ideal full model recover C closer to 5.6 A than the simple peptide-plane model? The full variant C-like peak is `{full_c:.3f}` A when available. Compare this to the prior simple peptide-plane C-like feature near `~5.0` A.
- Does D remain cross-strand/register dominated in richer coordinate sets? `{d_cross}` of `{len(rollup)}` variants have a cross/adjacent-register-like top D pair family.
- Which selection first improves C? The closest C peak among generated variants is {best_text}.
- Does C remain same-strand/local-repeat dominated, or become cross-strand/composite? `{c_local}` variants have same-strand top C pair families; `{c_cross}` have cross/adjacent-like top C pair families.
- Are AB/CD/EF and BC/DE/FA still split more strongly for C than D? Inspect the alternating interface columns above; this report preserves those counts for direct comparison.
- Does adding side-chain/carboxylate atoms improve C while preserving D, or disrupt D? Compare `backbone_only`, `side_chain_only`, `carboxylate_only`, and `full` rows above.

## Outputs

- Rollup CSV: `outputs/metrics/rich_coordinate_cd_rollup.csv`
- C/D peak plot: `outputs/figures/rich_coordinate_cd_C_vs_D.png`
"""
    path.write_text(text, encoding="utf-8")


def rollup(
    variant_manifest: Path,
    metrics_dir: Path,
    out_metrics: Path,
    out_reports: Path,
    out_figures: Path,
    target_c: float = 5.6,
    target_d: float = 7.3,
    tolerance: float = 0.20,
    q_step: float = 0.01,
    d_min: float = 2.5,
    d_max: float = 12.0,
) -> pd.DataFrame:
    """Create rich-coordinate rollup outputs."""
    manifest = pd.read_csv(variant_manifest)
    rows = [
        summarize_variant(row, metrics_dir, target_c, target_d, tolerance, q_step, d_min, d_max)
        for _, row in manifest.iterrows()
        if bool(row.get("written", True))
    ]
    df = pd.DataFrame(rows)
    out_metrics.mkdir(parents=True, exist_ok=True)
    out_reports.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_metrics / "rich_coordinate_cd_rollup.csv", index=False)
    write_report(df, out_reports / "rich_coordinate_cd_rollup_report.md")
    if {"C_peak_d_A", "D_peak_d_A"}.issubset(df.columns):
        save_c_vs_d_plot(df, out_figures / "rich_coordinate_cd_C_vs_D")
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant-manifest",
        type=Path,
        default=Path("outputs/coordinates/ideal_hexaflex_variants/variant_manifest.csv"),
    )
    parser.add_argument("--metrics-dir", type=Path, default=Path("outputs/metrics"))
    parser.add_argument("--out-metrics", type=Path, default=Path("outputs/metrics"))
    parser.add_argument("--out-reports", type=Path, default=Path("outputs/reports"))
    parser.add_argument("--out-figures", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--target-c", type=float, default=5.6)
    parser.add_argument("--target-d", type=float, default=7.3)
    parser.add_argument("--tolerance", type=float, default=0.20)
    parser.add_argument("--q-step", type=float, default=0.01)
    parser.add_argument("--d-min", type=float, default=2.5)
    parser.add_argument("--d-max", type=float, default=12.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = rollup(
        args.variant_manifest,
        args.metrics_dir,
        args.out_metrics,
        args.out_reports,
        args.out_figures,
        target_c=args.target_c,
        target_d=args.target_d,
        tolerance=args.tolerance,
        q_step=args.q_step,
        d_min=args.d_min,
        d_max=args.d_max,
    )
    print(f"Summarized {len(df)} rich-coordinate variants")
    print(f"CSV: {args.out_metrics / 'rich_coordinate_cd_rollup.csv'}")
    print(f"Report: {args.out_reports / 'rich_coordinate_cd_rollup_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
