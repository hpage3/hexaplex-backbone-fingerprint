"""Score geometry-interpretable global deformation variants against C/D targets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rollup_rich_coordinate_cd_diagnostics import score_pdb_profile
from scripts.score_constrained_phi_psi_candidates_cd import TARGET_C, TARGET_D, TOLERANCE, combined_abs_error, parse_bool


DEFAULT_MANIFEST = Path("outputs/metrics/global_deformation_variant_manifest.csv")
DEFAULT_AUDIT_CSV = Path("outputs/metrics/global_deformation_variant_geometry_audit.csv")
DEFAULT_SCORE_CSV = Path("outputs/metrics/global_deformation_variant_cd_scores.csv")
DEFAULT_REPORT = Path("outputs/reports/global_deformation_variant_cd_scores.md")

MODE_BASELINES = {
    "radial_scale_xy": "radial_0",
    "axial_scale_z": "axial_0",
    "twist_about_z": "twist_0",
    "anisotropic_xy": "anis_xy_0",
}

MODE_SORT_COLUMNS = {
    "radial_scale_xy": "radial_scale_xy",
    "axial_scale_z": "axial_scale_z",
    "twist_about_z": "twist_total_deg",
    "anisotropic_xy": "x_scale",
}


def load_scoring_inputs(manifest_path: Path, audit_path: Path) -> pd.DataFrame:
    """Join global deformation manifest parameters to geometry audit rows."""
    manifest = pd.read_csv(manifest_path)
    audit = pd.read_csv(audit_path)
    parameter_cols = [
        "variant_id",
        "radial_scale_xy",
        "axial_scale_z",
        "twist_total_deg",
        "x_scale",
        "y_scale",
    ]
    return audit.merge(manifest[parameter_cols], on="variant_id", how="left")


def filter_geometry_interpretable(rows: pd.DataFrame) -> pd.DataFrame:
    """Return only variants marked geometry-interpretable."""
    if "geometry_interpretable" not in rows.columns:
        raise ValueError("Global deformation audit is missing geometry_interpretable.")
    return rows[rows["geometry_interpretable"].map(parse_bool)].copy()


def skipped_noninterpretable_count(rows: pd.DataFrame) -> int:
    """Count variants excluded by geometry interpretability gate."""
    return len(rows) - len(filter_geometry_interpretable(rows))


def mode_baseline_row(scores: pd.DataFrame, deformation_mode: str) -> pd.Series:
    """Return mode-specific zero/control baseline row."""
    baseline_id = MODE_BASELINES[deformation_mode]
    subset = scores[scores["variant_id"] == baseline_id]
    if subset.empty:
        raise ValueError(f"Missing baseline variant {baseline_id} for mode {deformation_mode}.")
    return subset.iloc[0]


def sort_mode_scores(scores: pd.DataFrame, deformation_mode: str) -> pd.DataFrame:
    """Sort one mode by its meaningful deformation parameter."""
    subset = scores[scores["deformation_mode"] == deformation_mode].copy()
    sort_col = MODE_SORT_COLUMNS[deformation_mode]
    subset[sort_col] = pd.to_numeric(subset[sort_col], errors="coerce")
    return subset.sort_values([sort_col, "variant_id"]).reset_index(drop=True)


def add_relative_scores_by_mode(scores: pd.DataFrame) -> pd.DataFrame:
    """Normalize C/D scores to each deformation mode's zero/control baseline."""
    if scores.empty:
        scores["relative_C_score_vs_mode_baseline"] = pd.NA
        scores["relative_D_score_vs_mode_baseline"] = pd.NA
        return scores
    frames = []
    for mode in sorted(scores["deformation_mode"].dropna().unique()):
        subset = sort_mode_scores(scores, mode)
        baseline = mode_baseline_row(subset, mode)
        c0 = float(baseline["C_score"])
        d0 = float(baseline["D_score"])
        subset["relative_C_score_vs_mode_baseline"] = pd.to_numeric(subset["C_score"], errors="coerce") / c0
        subset["relative_D_score_vs_mode_baseline"] = pd.to_numeric(subset["D_score"], errors="coerce") / d0
        frames.append(subset)
    return pd.concat(frames, ignore_index=True)


def best_variant(scores: pd.DataFrame) -> pd.Series:
    """Return row with the lowest combined C/D absolute error."""
    if scores.empty:
        raise ValueError("No global deformation scores available.")
    values = pd.to_numeric(scores["combined_abs_error_A"], errors="coerce")
    return scores.loc[values.idxmin()]


def worst_variant(scores: pd.DataFrame) -> pd.Series:
    """Return row with the highest combined C/D absolute error."""
    if scores.empty:
        raise ValueError("No global deformation scores available.")
    values = pd.to_numeric(scores["combined_abs_error_A"], errors="coerce")
    return scores.loc[values.idxmax()]


def classify_directional_trend(values: pd.Series, tolerance: float = 1e-6) -> str:
    """Classify a short ordered numeric series."""
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_list()
    if len(numeric) < 3:
        return "insufficient"
    diffs = [b - a for a, b in zip(numeric, numeric[1:])]
    if all(abs(delta) <= tolerance for delta in diffs):
        return "flat"
    if all(delta >= -tolerance for delta in diffs):
        return "monotonic increasing"
    if all(delta <= tolerance for delta in diffs):
        return "monotonic decreasing"
    return "nonmonotonic"


def score_variant_row(
    row: pd.Series,
    target_c: float,
    target_d: float,
    tolerance: float,
    q_step: float,
    d_min: float,
    d_max: float,
) -> dict[str, object]:
    """Score one geometry-interpretable global deformation variant."""
    input_pdb = Path(str(row["output_pdb"]))
    score = score_pdb_profile(input_pdb, target_c, target_d, tolerance, q_step, d_min, d_max)
    c_error = score["C_error_A"]
    d_error = score["D_error_A"]
    return {
        "variant_id": row.get("variant_id", ""),
        "deformation_mode": row.get("deformation_mode", ""),
        "radial_scale_xy": row.get("radial_scale_xy", ""),
        "axial_scale_z": row.get("axial_scale_z", ""),
        "twist_total_deg": row.get("twist_total_deg", ""),
        "x_scale": row.get("x_scale", ""),
        "y_scale": row.get("y_scale", ""),
        "geometry_interpretable": row.get("geometry_interpretable", ""),
        "C_peak_A": score["C_peak_d_A"],
        "D_peak_A": score["D_peak_d_A"],
        "C_error_A": c_error,
        "D_error_A": d_error,
        "combined_abs_error_A": combined_abs_error(c_error, d_error),
        "C_score": score["C_peak_intensity"],
        "D_score": score["D_peak_intensity"],
        "input_pdb": str(input_pdb),
    }


def score_interpretable_variants(
    rows: pd.DataFrame,
    target_c: float = TARGET_C,
    target_d: float = TARGET_D,
    tolerance: float = TOLERANCE,
    q_step: float = 0.01,
    d_min: float = 2.5,
    d_max: float = 12.0,
) -> pd.DataFrame:
    """Score all geometry-interpretable global deformation variants."""
    scored_rows = [
        score_variant_row(row, target_c, target_d, tolerance, q_step, d_min, d_max)
        for _, row in filter_geometry_interpretable(rows).iterrows()
    ]
    return add_relative_scores_by_mode(pd.DataFrame(scored_rows))


def range_text(scores: pd.DataFrame, column: str) -> str:
    """Return min-max text for a numeric column."""
    numeric = pd.to_numeric(scores[column], errors="coerce").dropna()
    if numeric.empty:
        return "unavailable"
    return f"{numeric.min():.4f}-{numeric.max():.4f}"


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    """Render selected dataframe columns as markdown."""
    if df.empty:
        return "_None._"
    columns = [column for column in columns if column in df.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for record in df[columns].itertuples(index=False):
        values = [f"{value:.6g}" if isinstance(value, float) else str(value) for value in record]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def mode_trend_summary(scores: pd.DataFrame, mode: str) -> dict[str, str]:
    """Return C/D peak and score trend text for one deformation mode."""
    subset = sort_mode_scores(scores, mode)
    return {
        "C_peak_trend": classify_directional_trend(subset["C_peak_A"]),
        "D_peak_trend": classify_directional_trend(subset["D_peak_A"]),
        "C_score_range": range_text(subset, "relative_C_score_vs_mode_baseline"),
        "D_score_range": range_text(subset, "relative_D_score_vs_mode_baseline"),
    }


def build_report_text(scores: pd.DataFrame, scored_count: int, skipped_count: int) -> str:
    """Build markdown report for global deformation C/D scoring."""
    if scores.empty:
        return "# Global Deformation C/D Scores\n\nNo geometry-interpretable variants were available for scoring.\n"
    best = best_variant(scores)
    worst = worst_variant(scores)
    table_cols = [
        "variant_id",
        "C_peak_A",
        "D_peak_A",
        "C_error_A",
        "D_error_A",
        "combined_abs_error_A",
        "relative_C_score_vs_mode_baseline",
        "relative_D_score_vs_mode_baseline",
    ]
    sections = []
    trend_lines = []
    for mode in MODE_BASELINES:
        subset = sort_mode_scores(scores, mode)
        trends = mode_trend_summary(scores, mode)
        sections.append(f"### {mode}\n\n{markdown_table(subset, table_cols)}")
        trend_lines.append(
            f"- {mode}: C peak {trends['C_peak_trend']}; D peak {trends['D_peak_trend']}; "
            f"C score range {trends['C_score_range']}; D score range {trends['D_score_range']}."
        )

    return f"""# Global Deformation Variant C/D Scores

## Purpose

This tests whether controlled global deformations move C/D after the local torsion basin showed robustness. Scoring is restricted to geometry-interpretable variants. These are controlled diagnostic perturbations, not minimized physical structures.

## Counts

- Variants scored: {scored_count}
- Variants skipped as non-interpretable: {skipped_count}
- C target: {TARGET_C:.3f} A
- D target: {TARGET_D:.3f} A

## Results By Deformation Mode

{chr(10).join(sections)}

## Best / Worst Combined C/D Error

- Best variant: `{best['variant_id']}` ({best['deformation_mode']}) with combined error {float(best['combined_abs_error_A']):.4f} A.
- Worst variant: `{worst['variant_id']}` ({worst['deformation_mode']}) with combined error {float(worst['combined_abs_error_A']):.4f} A.

## Directional And Intensity Trends

{chr(10).join(trend_lines)}

## Interpretation

If radial or axial deformations move C/D, use the direction and magnitude above to prioritize the next controlled diagnostic. If C/D remains flat for a mode, the tested small global deformation did not move picked C/D positions under the current Debye scoring convention. These rows should not be read as minimized physical structures or as a standalone structural mechanism.
"""


def run(
    manifest_path: Path,
    audit_csv: Path,
    score_csv: Path,
    report_path: Path,
    target_c: float,
    target_d: float,
    tolerance: float,
    q_step: float,
    d_min: float,
    d_max: float,
) -> pd.DataFrame:
    """Score global deformation variants and write outputs."""
    rows = load_scoring_inputs(manifest_path, audit_csv)
    scored_count = len(filter_geometry_interpretable(rows))
    skipped_count = skipped_noninterpretable_count(rows)
    scores = score_interpretable_variants(rows, target_c, target_d, tolerance, q_step, d_min, d_max)
    score_csv.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(score_csv, index=False)
    report_path.write_text(build_report_text(scores, scored_count, skipped_count), encoding="utf-8")
    return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--audit-csv", type=Path, default=DEFAULT_AUDIT_CSV)
    parser.add_argument("--score-csv", type=Path, default=DEFAULT_SCORE_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--target-c", type=float, default=TARGET_C)
    parser.add_argument("--target-d", type=float, default=TARGET_D)
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    parser.add_argument("--q-step", type=float, default=0.01)
    parser.add_argument("--d-min", type=float, default=2.5)
    parser.add_argument("--d-max", type=float, default=12.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scores = run(
        args.manifest,
        args.audit_csv,
        args.score_csv,
        args.report,
        args.target_c,
        args.target_d,
        args.tolerance,
        args.q_step,
        args.d_min,
        args.d_max,
    )
    print(f"Scored {len(scores)} geometry-interpretable global deformation variants")
    print(f"CSV: {args.score_csv}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
