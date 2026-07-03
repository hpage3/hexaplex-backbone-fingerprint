"""Score focused radial/axial refinement variants against C/D targets."""

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


DEFAULT_AUDIT_CSV = Path("outputs/metrics/radial_axial_refinement_variant_geometry_audit.csv")
DEFAULT_SCORE_CSV = Path("outputs/metrics/radial_axial_refinement_variant_cd_scores.csv")
DEFAULT_REPORT = Path("outputs/reports/radial_axial_refinement_variant_cd_scores.md")


def filter_geometry_interpretable(rows: pd.DataFrame) -> pd.DataFrame:
    """Return only geometry-interpretable radial/axial rows."""
    if "geometry_interpretable" not in rows.columns:
        raise ValueError("Radial/axial audit is missing geometry_interpretable.")
    return rows[rows["geometry_interpretable"].map(parse_bool)].copy()


def skipped_noninterpretable_count(rows: pd.DataFrame) -> int:
    """Count rows excluded by geometry gate."""
    return len(rows) - len(filter_geometry_interpretable(rows))


def baseline_control_row(scores: pd.DataFrame) -> pd.Series:
    """Return baseline/control row with radial=1.0000 and axial=1.0000."""
    radial = pd.to_numeric(scores["radial_scale_xy"], errors="coerce")
    axial = pd.to_numeric(scores["axial_scale_z"], errors="coerce")
    baseline = scores[(radial == 1.0) & (axial == 1.0)]
    if baseline.empty:
        raise ValueError("Missing radial/axial baseline row with scales 1.0000/1.0000.")
    return baseline.iloc[0]


def add_relative_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Add C/D scores normalized to 1.0000/1.0000 baseline."""
    out = sort_grid(scores)
    if out.empty:
        out["relative_C_score_vs_baseline"] = pd.NA
        out["relative_D_score_vs_baseline"] = pd.NA
        return out
    baseline = baseline_control_row(out)
    c0 = float(baseline["C_score"])
    d0 = float(baseline["D_score"])
    out["relative_C_score_vs_baseline"] = pd.to_numeric(out["C_score"], errors="coerce") / c0
    out["relative_D_score_vs_baseline"] = pd.to_numeric(out["D_score"], errors="coerce") / d0
    return out


def sort_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Sort rows by axial scale then radial scale."""
    out = df.copy()
    out["radial_scale_xy"] = pd.to_numeric(out["radial_scale_xy"], errors="coerce")
    out["axial_scale_z"] = pd.to_numeric(out["axial_scale_z"], errors="coerce")
    return out.sort_values(["axial_scale_z", "radial_scale_xy", "variant_id"]).reset_index(drop=True)


def best_variant(scores: pd.DataFrame) -> pd.Series:
    """Return row with lowest combined C/D absolute error."""
    if scores.empty:
        raise ValueError("No radial/axial scores available.")
    values = pd.to_numeric(scores["combined_abs_error_A"], errors="coerce")
    return scores.loc[values.idxmin()]


def pivot_table_data(scores: pd.DataFrame, value_column: str) -> pd.DataFrame:
    """Return pivot table with axial rows and radial columns."""
    return sort_grid(scores).pivot_table(
        index="axial_scale_z",
        columns="radial_scale_xy",
        values=value_column,
        aggfunc="first",
    ).sort_index().sort_index(axis=1)


def score_variant_row(
    row: pd.Series,
    target_c: float,
    target_d: float,
    tolerance: float,
    q_step: float,
    d_min: float,
    d_max: float,
) -> dict[str, object]:
    """Score one geometry-interpretable radial/axial variant."""
    input_pdb = Path(str(row["output_pdb"]))
    score = score_pdb_profile(input_pdb, target_c, target_d, tolerance, q_step, d_min, d_max)
    c_error = score["C_error_A"]
    d_error = score["D_error_A"]
    return {
        "variant_id": row.get("variant_id", ""),
        "radial_scale_xy": row.get("radial_scale_xy", ""),
        "axial_scale_z": row.get("axial_scale_z", ""),
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
    audit: pd.DataFrame,
    target_c: float = TARGET_C,
    target_d: float = TARGET_D,
    tolerance: float = TOLERANCE,
    q_step: float = 0.01,
    d_min: float = 2.5,
    d_max: float = 12.0,
) -> pd.DataFrame:
    """Score all geometry-interpretable radial/axial variants."""
    rows = [
        score_variant_row(row, target_c, target_d, tolerance, q_step, d_min, d_max)
        for _, row in filter_geometry_interpretable(audit).iterrows()
    ]
    return add_relative_scores(pd.DataFrame(rows))


def markdown_table(df: pd.DataFrame, columns: list[str] | None = None) -> str:
    """Render dataframe as markdown."""
    if df.empty:
        return "_None._"
    if columns is None:
        columns = list(df.columns)
    columns = [column for column in columns if column in df.columns]
    lines = ["| " + " | ".join(str(c) for c in columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for record in df[columns].itertuples(index=False):
        values = [f"{value:.6g}" if isinstance(value, float) else str(value) for value in record]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def markdown_pivot(pivot: pd.DataFrame) -> str:
    """Render pivot table with float row/column labels."""
    table = pivot.copy()
    table.index = [f"{value:.4f}" for value in table.index]
    table.columns = [f"{value:.4f}" for value in table.columns]
    table = table.reset_index().rename(columns={"index": "axial_scale_z"})
    return markdown_table(table)


def trend_summary(scores: pd.DataFrame) -> dict[str, object]:
    """Return concise C/D sensitivity descriptors."""
    c_by_axial = sort_grid(scores).groupby("axial_scale_z")["C_peak_A"].agg(lambda x: sorted(set(x)))
    d_by_radial = sort_grid(scores).groupby("radial_scale_xy")["D_peak_A"].agg(lambda x: sorted(set(x)))
    c_by_radial_unique = sort_grid(scores).groupby("radial_scale_xy")["C_peak_A"].agg(lambda x: len(set(x)))
    d_by_axial_unique = sort_grid(scores).groupby("axial_scale_z")["D_peak_A"].agg(lambda x: len(set(x)))
    return {
        "C_by_axial": c_by_axial.to_dict(),
        "D_by_radial": d_by_radial.to_dict(),
        "C_depends_on_radial": bool((c_by_radial_unique > 1).any()),
        "D_depends_on_axial": bool((d_by_axial_unique > 1).any()),
    }


def build_report_text(scores: pd.DataFrame, scored_count: int, skipped_count: int) -> str:
    """Build radial/axial C/D score report."""
    if scores.empty:
        return "# Radial/Axial Refinement C/D Scores\n\nNo geometry-interpretable variants were available for scoring.\n"
    sorted_by_error = scores.sort_values(["combined_abs_error_A", "axial_scale_z", "radial_scale_xy"]).reset_index(drop=True)
    best = best_variant(scores)
    trends = trend_summary(scores)
    score_cols = [
        "variant_id",
        "radial_scale_xy",
        "axial_scale_z",
        "C_peak_A",
        "D_peak_A",
        "C_error_A",
        "D_error_A",
        "combined_abs_error_A",
    ]
    return f"""# Radial/Axial Refinement Variant C/D Scores

## Purpose

This focused radial/axial refinement follows the global deformation result that suggested D is radial/inter-strand sensitive while C is axial/rise-like sensitive.

- Variants scored: {scored_count}
- Variants skipped as non-interpretable: {skipped_count}
- C target: {TARGET_C:.3f} A
- D target: {TARGET_D:.3f} A

These are controlled diagnostic perturbations, not minimized structures.

## Results Sorted By Combined Error

{markdown_table(sorted_by_error, score_cols)}

## Best Variant

- Best radial/axial pair: radial `{float(best['radial_scale_xy']):.4f}`, axial `{float(best['axial_scale_z']):.4f}`
- Variant ID: `{best['variant_id']}`
- C peak: {float(best['C_peak_A']):.4f} A
- D peak: {float(best['D_peak_A']):.4f} A
- C error: {float(best['C_error_A']):.4f} A
- D error: {float(best['D_error_A']):.4f} A
- Combined absolute error: {float(best['combined_abs_error_A']):.4f} A

## Trend Summary

- C by axial scale: {trends['C_by_axial']}
- D by radial scale: {trends['D_by_radial']}
- C depends on radial scale in this grid: {trends['C_depends_on_radial']}
- D depends on axial scale in this grid: {trends['D_depends_on_axial']}

## C Peak Pivot

{markdown_pivot(pivot_table_data(scores, 'C_peak_A'))}

## D Peak Pivot

{markdown_pivot(pivot_table_data(scores, 'D_peak_A'))}

## Combined Absolute Error Pivot

{markdown_pivot(pivot_table_data(scores, 'combined_abs_error_A'))}

## Interpretation

Use this only as a focused diagnostic. If one radial/axial pair improves C and D simultaneously, it identifies a useful follow-up region. These variants are controlled diagnostic perturbations, not minimized structures, and should not be over-interpreted as final physical models.
"""


def run(
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
    """Score radial/axial refinement variants and write outputs."""
    audit = pd.read_csv(audit_csv)
    scored_count = len(filter_geometry_interpretable(audit))
    skipped_count = skipped_noninterpretable_count(audit)
    scores = score_interpretable_variants(audit, target_c, target_d, tolerance, q_step, d_min, d_max)
    score_csv.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(score_csv, index=False)
    report_path.write_text(build_report_text(scores, scored_count, skipped_count), encoding="utf-8")
    return scores


def skipped_noninterpretable_count(audit: pd.DataFrame) -> int:
    """Count non-interpretable rows."""
    return len(audit) - len(filter_geometry_interpretable(audit))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    print(f"Scored {len(scores)} radial/axial refinement variants")
    print(f"CSV: {args.score_csv}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
