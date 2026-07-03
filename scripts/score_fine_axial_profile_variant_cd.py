"""Score fine axial profile variants against C/D targets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rollup_rich_coordinate_cd_diagnostics import score_pdb_profile
from scripts.score_axial_only_extension_variant_cd import filter_geometry_interpretable, sort_by_axial
from scripts.score_constrained_phi_psi_candidates_cd import TARGET_C, TARGET_D, TOLERANCE, combined_abs_error
from scripts.score_radial_axial_refinement_variant_cd import markdown_table


DEFAULT_AUDIT_CSV = Path("outputs/metrics/fine_axial_profile_variant_geometry_audit.csv")
DEFAULT_SCORE_CSV = Path("outputs/metrics/fine_axial_profile_variant_cd_scores.csv")
DEFAULT_REPORT = Path("outputs/reports/fine_axial_profile_variant_cd_scores.md")
REFERENCE_AXIAL_SCALE = 0.9700


def reference_row(scores: pd.DataFrame, reference_axial: float = REFERENCE_AXIAL_SCALE) -> pd.Series:
    """Return local reference row by axial scale."""
    axial = pd.to_numeric(scores["axial_scale_z"], errors="coerce")
    subset = scores[abs(axial - reference_axial) <= 1e-9]
    if subset.empty:
        raise ValueError(f"Missing fine axial reference row at axial_scale_z={reference_axial:.4f}.")
    return subset.iloc[0]


def add_relative_scores(scores: pd.DataFrame, reference_axial: float = REFERENCE_AXIAL_SCALE) -> pd.DataFrame:
    """Normalize C/D scores to local reference axial scale."""
    out = sort_by_axial(scores)
    if out.empty:
        out["relative_C_score_vs_0p9700"] = pd.NA
        out["relative_D_score_vs_0p9700"] = pd.NA
        return out
    ref = reference_row(out, reference_axial)
    c0 = float(ref["C_score"])
    d0 = float(ref["D_score"])
    out["relative_C_score_vs_0p9700"] = pd.to_numeric(out["C_score"], errors="coerce") / c0
    out["relative_D_score_vs_0p9700"] = pd.to_numeric(out["D_score"], errors="coerce") / d0
    return out


def best_variant(scores: pd.DataFrame) -> pd.Series:
    """Return best combined C/D error row."""
    if scores.empty:
        raise ValueError("No fine axial scores available.")
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
    """Score one fine axial variant."""
    input_pdb = Path(str(row["output_pdb"]))
    score = score_pdb_profile(input_pdb, target_c, target_d, tolerance, q_step, d_min, d_max)
    c_error = score["C_error_A"]
    d_error = score["D_error_A"]
    return {
        "variant_id": row.get("variant_id", ""),
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
    """Score all geometry-interpretable fine axial variants."""
    rows = [
        score_variant_row(row, target_c, target_d, tolerance, q_step, d_min, d_max)
        for _, row in filter_geometry_interpretable(audit).iterrows()
    ]
    return add_relative_scores(pd.DataFrame(rows))


def build_report_text(scores: pd.DataFrame, scored_count: int, skipped_count: int) -> str:
    """Build fine axial score report."""
    if scores.empty:
        return "# Fine Axial Profile C/D Scores\n\nNo geometry-interpretable variants were available for scoring.\n"
    sorted_scores = sort_by_axial(scores)
    best = best_variant(scores)
    cols = [
        "variant_id",
        "axial_scale_z",
        "C_peak_A",
        "D_peak_A",
        "C_error_A",
        "D_error_A",
        "combined_abs_error_A",
        "relative_C_score_vs_0p9700",
        "relative_D_score_vs_0p9700",
    ]
    return f"""# Fine Axial Profile Variant C/D Scores

## Purpose

This fine axial profile diagnostic focuses on axial 0.9700 to 0.9850 after the axial-only extension showed a discretized C peak response.

- Variants scored: {scored_count}
- Variants skipped as non-interpretable: {skipped_count}
- Local reference variant: `fine_axial_0p9700`
- C target: {TARGET_C:.3f} A
- D target: {TARGET_D:.3f} A

These are controlled diagnostic perturbations, not minimized structures.

## Results By Axial Scale

{markdown_table(sorted_scores, cols)}

## Best Variant

- Best axial scale: {float(best['axial_scale_z']):.4f}
- Variant ID: `{best['variant_id']}`
- C peak: {float(best['C_peak_A']):.4f} A
- D peak: {float(best['D_peak_A']):.4f} A
- C error: {float(best['C_error_A']):.4f} A
- D error: {float(best['D_error_A']):.4f} A
- Combined absolute error: {float(best['combined_abs_error_A']):.4f} A

## Picked Peak Trend

- C picked peaks by axial scale: {sorted_scores[['axial_scale_z', 'C_peak_A']].to_dict('records')}
- D picked peaks by axial scale: {sorted_scores[['axial_scale_z', 'D_peak_A']].to_dict('records')}
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
    """Score fine axial variants and write outputs."""
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
    print(f"Scored {len(scores)} fine axial profile variants")
    print(f"CSV: {args.score_csv}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
