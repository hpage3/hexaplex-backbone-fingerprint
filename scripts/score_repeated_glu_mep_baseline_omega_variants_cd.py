"""Score safe repeated GLU->MEP baseline-parent omega variants against C/D targets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rollup_rich_coordinate_cd_diagnostics import score_pdb_profile
from scripts.score_constrained_phi_psi_candidates_cd import TARGET_C, TARGET_D, TOLERANCE, combined_abs_error, parse_bool


DEFAULT_AUDIT_CSV = Path("outputs/metrics/repeated_glu_mep_baseline_omega_variant_geometry_audit.csv")
DEFAULT_SCORE_CSV = Path("outputs/metrics/repeated_glu_mep_baseline_omega_variant_cd_scores.csv")
DEFAULT_REPORT = Path("outputs/reports/repeated_glu_mep_baseline_omega_variant_cd_scores.md")
DEFAULT_FIGURE_BASE = Path("outputs/figures/repeated_glu_mep_baseline_omega_variant_cd_scores")


def filter_safe_baseline_parent_variants(audit: pd.DataFrame) -> pd.DataFrame:
    """Return safe baseline-parent GLU->MEP variants."""
    if "safe_for_diffraction_scoring" not in audit.columns:
        raise ValueError("Geometry audit is missing safe_for_diffraction_scoring.")
    return audit[
        audit["safe_for_diffraction_scoring"].map(parse_bool)
        & (audit["omega_mode"] == "baseline_parent")
    ].copy()


def skipped_unsafe_count(audit: pd.DataFrame) -> int:
    """Count rows excluded from scoring."""
    return len(audit) - len(filter_safe_baseline_parent_variants(audit))


def best_variant(scores: pd.DataFrame) -> pd.Series:
    """Return row with lowest combined C/D absolute error."""
    if scores.empty:
        raise ValueError("No variant scores available.")
    values = pd.to_numeric(scores["combined_abs_error_A"], errors="coerce")
    return scores.loc[values.idxmin()]


def sort_by_delta(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by fixed torsion delta."""
    out = df.copy()
    out["fixed_torsion_delta_deg"] = pd.to_numeric(out["fixed_torsion_delta_deg"], errors="coerce")
    return out.sort_values(["fixed_torsion_delta_deg", "variant_id"]).reset_index(drop=True)


def classify_two_point_movement(scores: pd.DataFrame, column: str, tolerance: float = 1e-6) -> str:
    """Classify whether a nonzero safe row changes a value relative to delta 0."""
    sorted_scores = sort_by_delta(scores)
    baseline = sorted_scores[pd.to_numeric(sorted_scores["fixed_torsion_delta_deg"], errors="coerce") == 0]
    nonzero = sorted_scores[pd.to_numeric(sorted_scores["fixed_torsion_delta_deg"], errors="coerce") != 0]
    if baseline.empty or nonzero.empty:
        return "insufficient"
    delta = float(nonzero.iloc[0][column]) - float(baseline.iloc[0][column])
    if abs(delta) <= tolerance:
        return "no change"
    return f"changed by {delta:+.6g}"


def add_baseline_relative_intensity(scores: pd.DataFrame) -> pd.DataFrame:
    """Add C/D relative intensity columns normalized to delta 0."""
    out = sort_by_delta(scores)
    baseline = out[pd.to_numeric(out["fixed_torsion_delta_deg"], errors="coerce") == 0]
    if baseline.empty:
        out["C_relative_to_baseline"] = pd.NA
        out["D_relative_to_baseline"] = pd.NA
        return out
    c0 = float(baseline.iloc[0]["C_peak_intensity_or_score"])
    d0 = float(baseline.iloc[0]["D_peak_intensity_or_score"])
    out["C_relative_to_baseline"] = pd.to_numeric(out["C_peak_intensity_or_score"], errors="coerce") / c0
    out["D_relative_to_baseline"] = pd.to_numeric(out["D_peak_intensity_or_score"], errors="coerce") / d0
    return out


def score_variant_row(row: pd.Series, target_c: float, target_d: float, tolerance: float, q_step: float, d_min: float, d_max: float) -> dict[str, object]:
    """Score one safe GLU->MEP baseline-parent repeated variant."""
    coordinate_path = Path(str(row["coordinate_path"]))
    score = score_pdb_profile(coordinate_path, target_c, target_d, tolerance, q_step, d_min, d_max)
    c_error = score["C_error_A"]
    d_error = score["D_error_A"]
    return {
        "variant_id": row.get("variant_id", ""),
        "fixed_torsion_delta_deg": row.get("fixed_torsion_delta_deg", ""),
        "solve_mode": row.get("solve_mode", ""),
        "omega_mode": row.get("omega_mode", ""),
        "attempted_window_count": row.get("attempted_window_count", ""),
        "applied_window_count": row.get("applied_window_count", ""),
        "skipped_window_count": row.get("skipped_window_count", ""),
        "max_ca_shift_A": row.get("max_ca_shift_A", ""),
        "max_backbone_bond_delta_A": row.get("max_backbone_bond_delta_A", ""),
        "max_backbone_angle_delta_deg": row.get("max_backbone_angle_delta_deg", ""),
        "max_omega_trans_deviation_deg": row.get("max_omega_trans_deviation_deg", ""),
        "coordinate_path": str(coordinate_path),
        "C_peak_A": score["C_peak_d_A"],
        "D_peak_A": score["D_peak_d_A"],
        "C_error_A": c_error,
        "D_error_A": d_error,
        "combined_abs_error_A": combined_abs_error(c_error, d_error),
        "C_peak_intensity_or_score": score["C_peak_intensity"],
        "D_peak_intensity_or_score": score["D_peak_intensity"],
        "notes": row.get("notes", ""),
    }


def score_safe_variants(audit: pd.DataFrame, target_c: float, target_d: float, tolerance: float, q_step: float, d_min: float, d_max: float) -> pd.DataFrame:
    """Score safe variants and add baseline-relative intensity columns."""
    rows = [
        score_variant_row(row, target_c, target_d, tolerance, q_step, d_min, d_max)
        for _, row in filter_safe_baseline_parent_variants(audit).iterrows()
    ]
    return add_baseline_relative_intensity(pd.DataFrame(rows))


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    """Render selected columns as markdown."""
    if df.empty:
        return "_None._"
    columns = [column for column in columns if column in df.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for record in df[columns].itertuples(index=False):
        values = [f"{value:.6g}" if isinstance(value, float) else str(value) for value in record]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_report_text(scores: pd.DataFrame, safe_count: int, skipped_count: int) -> str:
    """Build report text."""
    if scores.empty:
        return "# Repeated GLU->MEP Baseline-Omega C/D Scores\n\nNo safe variants were available for scoring.\n"
    best = best_variant(scores)
    c_position = classify_two_point_movement(scores, "C_peak_A")
    d_position = classify_two_point_movement(scores, "D_peak_A")
    c_intensity = classify_two_point_movement(scores, "C_relative_to_baseline", tolerance=0.005)
    d_intensity = classify_two_point_movement(scores, "D_relative_to_baseline", tolerance=0.005)
    columns = [
        "variant_id",
        "fixed_torsion_delta_deg",
        "C_peak_A",
        "D_peak_A",
        "C_error_A",
        "D_error_A",
        "combined_abs_error_A",
        "C_relative_to_baseline",
        "D_relative_to_baseline",
    ]
    return f"""# Repeated GLU->MEP Baseline-Omega C/D Scores

This is a two-point `baseline_parent` omega pilot, not a broad or unconstrained omega scan. Unsafe variants were excluded because they failed bond and/or omega-trans geometry thresholds.

## Counts

- Variants scored: {safe_count}
- Variants skipped as unsafe: {skipped_count}
- C target: {TARGET_C:.3f} A
- D target: {TARGET_D:.3f} A

## Per-Delta C/D Scores

{markdown_table(sort_by_delta(scores), columns)}

## Interpretation

- Best variant by combined C/D absolute error: `{best['variant_id']}` with combined error {float(best['combined_abs_error_A']):.4f} A.
- Delta -1 C peak position versus delta 0: {c_position}.
- Delta -1 D peak position versus delta 0: {d_position}.
- Delta -1 C intensity/score versus delta 0: {c_intensity}.
- Delta -1 D intensity/score versus delta 0: {d_intensity}.
- Intensity/score sensitivity is not the same as peak-position movement.
"""


def save_plot(scores: pd.DataFrame, figure_base: Path, target_c: float, target_d: float) -> None:
    """Save C/D peak-position plot."""
    if scores.empty:
        return
    scores = sort_by_delta(scores)
    x = pd.to_numeric(scores["fixed_torsion_delta_deg"], errors="coerce")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, scores["C_peak_A"], marker="o", label="C-like peak")
    ax.plot(x, scores["D_peak_A"], marker="o", label="D-like peak")
    ax.axhline(target_c, color="#1f77b4", ls="--", lw=1, label="C target")
    ax.axhline(target_d, color="#ff7f0e", ls="--", lw=1, label="D target")
    ax.axvline(0, color="0.5", ls=":", lw=1)
    ax.set_xlabel("fixed phi0 delta applied repeatedly (deg)")
    ax.set_ylabel("peak d-spacing (A)")
    ax.set_title("Repeated GLU->MEP baseline-omega C/D scores")
    ax.legend(fontsize=8)
    fig.tight_layout()
    figure_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_base.with_suffix(".png"), dpi=180)
    fig.savefig(figure_base.with_suffix(".svg"))
    plt.close(fig)


def run(audit_csv: Path, score_csv: Path, report_path: Path, figure_base: Path, target_c: float, target_d: float, tolerance: float, q_step: float, d_min: float, d_max: float) -> pd.DataFrame:
    """Score safe variants and write outputs."""
    audit = pd.read_csv(audit_csv)
    safe_count = len(filter_safe_baseline_parent_variants(audit))
    skipped_count = skipped_unsafe_count(audit)
    scores = score_safe_variants(audit, target_c, target_d, tolerance, q_step, d_min, d_max)
    score_csv.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(score_csv, index=False)
    report_path.write_text(build_report_text(scores, safe_count, skipped_count), encoding="utf-8")
    save_plot(scores, figure_base, target_c, target_d)
    return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-csv", type=Path, default=DEFAULT_AUDIT_CSV)
    parser.add_argument("--score-csv", type=Path, default=DEFAULT_SCORE_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--figure-base", type=Path, default=DEFAULT_FIGURE_BASE)
    parser.add_argument("--target-c", type=float, default=TARGET_C)
    parser.add_argument("--target-d", type=float, default=TARGET_D)
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    parser.add_argument("--q-step", type=float, default=0.01)
    parser.add_argument("--d-min", type=float, default=2.5)
    parser.add_argument("--d-max", type=float, default=12.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scores = run(args.audit_csv, args.score_csv, args.report, args.figure_base, args.target_c, args.target_d, args.tolerance, args.q_step, args.d_min, args.d_max)
    print(f"Scored {len(scores)} safe repeated GLU->MEP baseline-omega variants")
    print(f"CSV: {args.score_csv}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
