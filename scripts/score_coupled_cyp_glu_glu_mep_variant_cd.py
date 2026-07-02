"""Score geometry-safe coupled CYP->GLU + GLU->MEP variants against C/D targets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rollup_rich_coordinate_cd_diagnostics import score_pdb_profile
from scripts.score_constrained_phi_psi_candidates_cd import (
    TARGET_C,
    TARGET_D,
    TOLERANCE,
    combined_abs_error,
    parse_bool,
)


DEFAULT_AUDIT_CSV = Path("outputs/metrics/coupled_cyp_glu_glu_mep_variant_geometry_audit.csv")
DEFAULT_SCORE_CSV = Path("outputs/metrics/coupled_cyp_glu_glu_mep_variant_cd_scores.csv")
DEFAULT_REPORT = Path("outputs/reports/coupled_cyp_glu_glu_mep_variant_cd_scores.md")


def filter_geometry_safe_rows(audit: pd.DataFrame) -> pd.DataFrame:
    """Return coupled rows explicitly marked geometry-safe."""
    if "geometry_safe" not in audit.columns:
        raise ValueError("Coupled geometry audit is missing geometry_safe.")
    return audit[audit["geometry_safe"].map(parse_bool)].copy()


def skipped_unsafe_count(audit: pd.DataFrame) -> int:
    """Count coupled variants skipped because they were not geometry-safe."""
    return len(audit) - len(filter_geometry_safe_rows(audit))


def baseline_control_row(scores: pd.DataFrame) -> pd.Series:
    """Return the coupled baseline row: CYP->GLU delta 0 and GLU->MEP delta 0."""
    if scores.empty:
        raise ValueError("No coupled scores available.")
    cyp_delta = pd.to_numeric(scores["cyp_glu_delta_deg"], errors="coerce")
    glu_delta = pd.to_numeric(scores["glu_mep_delta_deg"], errors="coerce")
    baseline = scores[(cyp_delta == 0.0) & (glu_delta == 0.0)]
    if baseline.empty:
        raise ValueError("No coupled baseline/control row with deltas 0/0 was found.")
    return baseline.iloc[0]


def add_baseline_relative_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Normalize C/D peak scores to the coupled 0/0 baseline."""
    out = sort_by_coupled_deltas(scores)
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


def best_combined_error_row(scores: pd.DataFrame) -> pd.Series:
    """Return the row with the smallest combined absolute C/D error."""
    if scores.empty:
        raise ValueError("No coupled scores available.")
    values = pd.to_numeric(scores["combined_abs_error_A"], errors="coerce")
    return scores.loc[values.idxmin()]


def sort_by_coupled_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by GLU->MEP delta, then CYP->GLU delta, then variant ID."""
    out = df.copy()
    out["cyp_glu_delta_deg"] = pd.to_numeric(out["cyp_glu_delta_deg"], errors="coerce")
    out["glu_mep_delta_deg"] = pd.to_numeric(out["glu_mep_delta_deg"], errors="coerce")
    return out.sort_values(["glu_mep_delta_deg", "cyp_glu_delta_deg", "variant_id"]).reset_index(drop=True)


def classify_peak_shift(values: pd.Series, tolerance: float = 1e-6) -> str:
    """Classify a set of peak positions as flat or shifted."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return "insufficient"
    spread = float(numeric.max() - numeric.min())
    if spread <= tolerance:
        return "flat"
    return f"shifted; range {numeric.min():.4f}-{numeric.max():.4f} A"


def score_variant_row(
    row: pd.Series,
    target_c: float,
    target_d: float,
    tolerance: float,
    q_step: float,
    d_min: float,
    d_max: float,
) -> dict[str, object]:
    """Score one geometry-safe coupled variant PDB."""
    input_pdb = Path(str(row["output_pdb"]))
    score = score_pdb_profile(input_pdb, target_c, target_d, tolerance, q_step, d_min, d_max)
    c_error = score["C_error_A"]
    d_error = score["D_error_A"]
    return {
        "variant_id": row.get("variant_id", ""),
        "cyp_glu_delta_deg": row.get("cyp_glu_delta_deg", ""),
        "glu_mep_delta_deg": row.get("glu_mep_delta_deg", ""),
        "geometry_safe": row.get("geometry_safe", ""),
        "C_peak_A": score["C_peak_d_A"],
        "D_peak_A": score["D_peak_d_A"],
        "C_error_A": c_error,
        "D_error_A": d_error,
        "combined_abs_error_A": combined_abs_error(c_error, d_error),
        "C_score": score["C_peak_intensity"],
        "D_score": score["D_peak_intensity"],
        "input_pdb": str(input_pdb),
    }


def score_geometry_safe_variants(
    audit: pd.DataFrame,
    target_c: float = TARGET_C,
    target_d: float = TARGET_D,
    tolerance: float = TOLERANCE,
    q_step: float = 0.01,
    d_min: float = 2.5,
    d_max: float = 12.0,
) -> pd.DataFrame:
    """Score geometry-safe coupled variants and add baseline-relative scores."""
    rows = [
        score_variant_row(row, target_c, target_d, tolerance, q_step, d_min, d_max)
        for _, row in filter_geometry_safe_rows(audit).iterrows()
    ]
    return add_baseline_relative_scores(pd.DataFrame(rows))


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


def score_range_text(scores: pd.DataFrame, column: str) -> str:
    """Return min/max range text for a score column."""
    numeric = pd.to_numeric(scores[column], errors="coerce").dropna()
    if numeric.empty:
        return "unavailable"
    return f"{numeric.min():.4f}-{numeric.max():.4f}"


def build_report_text(scores: pd.DataFrame, scored_count: int, skipped_count: int) -> str:
    """Build markdown report for coupled C/D scoring."""
    if scores.empty:
        return (
            "# Coupled CYP->GLU + GLU->MEP C/D Scores\n\n"
            "No geometry-safe coupled variants were available for scoring. "
            "This is a coupled perturbation pilot; variants must be geometry-safe before scoring.\n"
        )
    sorted_scores = sort_by_coupled_deltas(scores)
    baseline = baseline_control_row(sorted_scores)
    best = best_combined_error_row(sorted_scores)
    c_shift = classify_peak_shift(sorted_scores["C_peak_A"])
    d_shift = classify_peak_shift(sorted_scores["D_peak_A"])
    columns = [
        "variant_id",
        "cyp_glu_delta_deg",
        "glu_mep_delta_deg",
        "C_peak_A",
        "D_peak_A",
        "C_error_A",
        "D_error_A",
        "combined_abs_error_A",
        "relative_C_score_vs_baseline",
        "relative_D_score_vs_baseline",
    ]
    return f"""# Coupled CYP->GLU + GLU->MEP C/D Scores

This is a coupled perturbation pilot. All scored variants were geometry-safe before scoring, and geometry-unsafe variants are excluded. Do not overclaim peak-position behavior if the peaks remain flat.

## Counts

- Variants scored: {scored_count}
- Variants skipped as unsafe: {skipped_count}
- Baseline/control variant ID: `{baseline['variant_id']}`
- C target: {TARGET_C:.3f} A
- D target: {TARGET_D:.3f} A

## Scored Coupled Variants

{markdown_table(sorted_scores, columns)}

## Peak Position Summary

- Best variant by combined absolute C/D error: `{best['variant_id']}` with combined error {float(best['combined_abs_error_A']):.4f} A.
- C peak positions: {c_shift}
- D peak positions: {d_shift}
- Relative C score range versus baseline: {score_range_text(sorted_scores, 'relative_C_score_vs_baseline')}
- Relative D score range versus baseline: {score_range_text(sorted_scores, 'relative_D_score_vs_baseline')}

## Interpretation

Use this table only as a small coupled-pilot diagnostic. If C/D peaks shift, the direction and best coupled delta pair above identify the leading follow-up region. If peaks remain flat, the coupled perturbations did not move the global C/D peak positions under the current Debye scoring convention, though score/intensity differences may still be useful as a secondary diagnostic.
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
    """Score geometry-safe coupled variants and write outputs."""
    audit = pd.read_csv(audit_csv)
    scored_count = len(filter_geometry_safe_rows(audit))
    skipped_count = skipped_unsafe_count(audit)
    scores = score_geometry_safe_variants(audit, target_c, target_d, tolerance, q_step, d_min, d_max)
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
    print(f"Scored {len(scores)} geometry-safe coupled variants")
    print(f"CSV: {args.score_csv}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
