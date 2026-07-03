"""Score geometry-interpretable parameterized rise variants against C/D targets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rollup_rich_coordinate_cd_diagnostics import score_pdb_profile
from scripts.score_axial_only_extension_variant_cd import filter_geometry_interpretable
from scripts.score_constrained_phi_psi_candidates_cd import TARGET_C, TARGET_D, TOLERANCE, combined_abs_error
from scripts.score_radial_axial_refinement_variant_cd import markdown_table


DEFAULT_AUDIT_CSV = Path("outputs/metrics/parameterized_rise_variant_geometry_audit.csv")
DEFAULT_SCORE_CSV = Path("outputs/metrics/parameterized_rise_variant_cd_scores.csv")
DEFAULT_REPORT = Path("outputs/reports/parameterized_rise_variant_cd_scores.md")
RISE_LIKE_SCORE_CSV = Path("outputs/metrics/rise_like_variant_cd_scores.csv")
BASELINE_RISE_SCALE = 1.0000


def sort_by_rise(df: pd.DataFrame) -> pd.DataFrame:
    """Sort parameterized rise rows by rise scale."""
    out = df.copy()
    out["rise_scale"] = pd.to_numeric(out["rise_scale"], errors="coerce")
    return out.sort_values(["rise_scale", "variant_id"]).reset_index(drop=True)


def baseline_row(scores: pd.DataFrame, baseline_scale: float = BASELINE_RISE_SCALE) -> pd.Series:
    """Return parameterized_rise_1p0000 baseline/control row."""
    rise = pd.to_numeric(scores["rise_scale"], errors="coerce")
    subset = scores[abs(rise - baseline_scale) <= 1e-9]
    if subset.empty:
        raise ValueError(f"Missing parameterized rise baseline row at rise_scale={baseline_scale:.4f}.")
    return subset.iloc[0]


def add_relative_scores(scores: pd.DataFrame, baseline_scale: float = BASELINE_RISE_SCALE) -> pd.DataFrame:
    """Normalize C/D scores to baseline/control."""
    out = sort_by_rise(scores)
    if out.empty:
        out["relative_C_score_vs_baseline"] = pd.NA
        out["relative_D_score_vs_baseline"] = pd.NA
        return out
    base = baseline_row(out, baseline_scale)
    out["relative_C_score_vs_baseline"] = pd.to_numeric(out["C_score"], errors="coerce") / float(base["C_score"])
    out["relative_D_score_vs_baseline"] = pd.to_numeric(out["D_score"], errors="coerce") / float(base["D_score"])
    return out


def best_variant(scores: pd.DataFrame) -> pd.Series:
    """Return row with lowest combined C/D absolute error."""
    if scores.empty:
        raise ValueError("No parameterized rise scores available.")
    values = pd.to_numeric(scores["combined_abs_error_A"], errors="coerce")
    return scores.loc[values.idxmin()]


def score_variant_row(
    row: pd.Series,
    target_c: float,
    target_d: float,
    tolerance: float,
    q_step: float,
    d_min: float,
    d_max: float,
) -> dict[str, object]:
    """Score one parameterized rise variant."""
    input_pdb = Path(str(row["output_pdb"]))
    score = score_pdb_profile(input_pdb, target_c, target_d, tolerance, q_step, d_min, d_max)
    c_error = score["C_error_A"]
    d_error = score["D_error_A"]
    return {
        "variant_id": row.get("variant_id", ""),
        "rise_scale": row.get("rise_scale", ""),
        "estimated_percent_rise_compression": row.get("estimated_percent_rise_compression", ""),
        "layer_model": row.get("layer_model", ""),
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
    """Score all geometry-interpretable variants."""
    rows = [
        score_variant_row(row, target_c, target_d, tolerance, q_step, d_min, d_max)
        for _, row in filter_geometry_interpretable(audit).iterrows()
    ]
    return add_relative_scores(pd.DataFrame(rows))


def rise_like_comparison_text(rise_like_csv: Path) -> str:
    """Return comparison text against prior rise_like diagnostic if available."""
    if not rise_like_csv.exists():
        return "Prior rise_like diagnostic score CSV was not available for direct comparison."
    rise_like = pd.read_csv(rise_like_csv)
    best = rise_like.loc[pd.to_numeric(rise_like["combined_abs_error_A"], errors="coerce").idxmin()]
    return (
        f"Prior rise_like best was `{best['variant_id']}` with C {float(best['C_peak_A']):.4f} A, "
        f"D {float(best['D_peak_A']):.4f} A, combined error {float(best['combined_abs_error_A']):.4f} A."
    )


def build_report_text(scores: pd.DataFrame, scored_count: int, skipped_count: int, rise_like_csv: Path = RISE_LIKE_SCORE_CSV) -> str:
    """Build parameterized rise score report."""
    if scores.empty:
        return "# Parameterized Rise Variant C/D Scores\n\nNo geometry-interpretable variants were available for scoring.\n"
    sorted_scores = sort_by_rise(scores)
    best = best_variant(sorted_scores)
    cols = [
        "variant_id",
        "rise_scale",
        "estimated_percent_rise_compression",
        "C_peak_A",
        "D_peak_A",
        "C_error_A",
        "D_error_A",
        "combined_abs_error_A",
        "relative_C_score_vs_baseline",
        "relative_D_score_vs_baseline",
    ]
    return f"""# Parameterized Rise Variant C/D Scores

## Purpose

This branch tests layer/repeat-aware rise compression as a more interpretable structural parameter than continuous global z-scaling.

- Variants scored: {scored_count}
- Variants skipped as non-interpretable: {skipped_count}
- Baseline/control variant: `parameterized_rise_1p0000`
- C target: {TARGET_C:.3f} A
- D target: {TARGET_D:.3f} A

## Results By Rise Scale

{markdown_table(sorted_scores, cols)}

## Best Variant

- Best rise scale: {float(best['rise_scale']):.4f}
- Variant ID: `{best['variant_id']}`
- C peak: {float(best['C_peak_A']):.4f} A
- D peak: {float(best['D_peak_A']):.4f} A
- C error: {float(best['C_error_A']):.4f} A
- D error: {float(best['D_error_A']):.4f} A
- Combined absolute error: {float(best['combined_abs_error_A']):.4f} A

## Comparison To Earlier rise_like Diagnostic

{rise_like_comparison_text(rise_like_csv)}

## Interpretation

Effective rise compression remains the leading diagnostic handle on C if C moves toward 5.6 A as rise scale decreases. These variants are still not minimized physical structures.
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
    """Score parameterized rise variants and write outputs."""
    audit = pd.read_csv(audit_csv)
    scored_count = len(filter_geometry_interpretable(audit))
    skipped_count = len(audit) - scored_count
    scores = score_interpretable_variants(audit, target_c, target_d, tolerance, q_step, d_min, d_max)
    score_csv.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(score_csv, index=False)
    report_path.write_text(build_report_text(scores, scored_count, skipped_count), encoding="utf-8")
    return scores


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
    scores = run(args.audit_csv, args.score_csv, args.report, args.target_c, args.target_d, args.tolerance, args.q_step, args.d_min, args.d_max)
    print(f"Scored {len(scores)} parameterized rise variants")
    print(f"CSV: {args.score_csv}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
