"""Score geometry-safe constrained phi/psi candidates against C/D powder targets."""

from __future__ import annotations

import argparse
import math
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


DEFAULT_AUDIT_CSV = Path("outputs/metrics/constrained_phi_psi_candidate_geometry_audit.csv")
DEFAULT_SCORE_CSV = Path("outputs/metrics/constrained_phi_psi_candidate_cd_scores.csv")
DEFAULT_REPORT = Path("outputs/reports/constrained_phi_psi_candidate_cd_scores.md")
DEFAULT_FIGURE_BASE = Path("outputs/figures/constrained_phi_psi_candidate_cd_scores")

TARGET_C = 5.6
TARGET_D = 7.3
TOLERANCE = 0.20


def parse_bool(value: object) -> bool:
    """Parse common CSV boolean representations."""
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def filter_safe_candidates(audit: pd.DataFrame) -> pd.DataFrame:
    """Return rows explicitly marked safe for diffraction scoring."""
    if "safe_for_diffraction_scoring" not in audit.columns:
        raise ValueError("Geometry audit is missing safe_for_diffraction_scoring.")
    mask = audit["safe_for_diffraction_scoring"].map(parse_bool)
    return audit[mask].copy()


def skipped_unsafe_count(audit: pd.DataFrame) -> int:
    """Count rows that are not safe for diffraction scoring."""
    return len(audit) - len(filter_safe_candidates(audit))


def combined_abs_error(c_error: object, d_error: object) -> float:
    """Return |C error| + |D error|."""
    return abs(float(c_error)) + abs(float(d_error))


def best_candidate(scores: pd.DataFrame) -> pd.Series:
    """Return the row with the smallest combined C/D absolute error."""
    if scores.empty:
        raise ValueError("No scored candidates available.")
    values = pd.to_numeric(scores["combined_abs_error_A"], errors="coerce")
    return scores.loc[values.idxmin()]


def sort_by_fixed_delta(df: pd.DataFrame) -> pd.DataFrame:
    """Sort rows by fixed torsion delta, preserving candidate ID as a tie-breaker."""
    out = df.copy()
    out["fixed_torsion_delta_deg"] = pd.to_numeric(out["fixed_torsion_delta_deg"], errors="coerce")
    return out.sort_values(["fixed_torsion_delta_deg", "candidate_id"]).reset_index(drop=True)


def monotonic_trend(values: pd.Series) -> str:
    """Classify a short numeric sequence as monotonic or nonmonotonic."""
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_list()
    if len(numeric) < 3:
        return "insufficient"
    diffs = [b - a for a, b in zip(numeric, numeric[1:])]
    nonnegative = all(delta >= -1e-12 for delta in diffs)
    nonpositive = all(delta <= 1e-12 for delta in diffs)
    if all(abs(delta) <= 1e-12 for delta in diffs):
        return "flat"
    if nonnegative:
        return "monotonic increasing"
    if nonpositive:
        return "monotonic decreasing"
    return "nonmonotonic"


def score_candidate_row(
    row: pd.Series,
    target_c: float,
    target_d: float,
    tolerance: float,
    q_step: float,
    d_min: float,
    d_max: float,
) -> dict[str, object]:
    """Score one geometry-safe candidate PDB."""
    coordinate_path = Path(str(row["coordinate_path"]))
    profile_score = score_pdb_profile(coordinate_path, target_c, target_d, tolerance, q_step, d_min, d_max)
    c_error = profile_score["C_error_A"]
    d_error = profile_score["D_error_A"]
    return {
        "candidate_id": row.get("candidate_id", ""),
        "source_chain": row.get("source_chain", ""),
        "repeat_type": row.get("repeat_type", ""),
        "solve_mode": row.get("solve_mode", ""),
        "fixed_torsion_name": row.get("fixed_torsion_name", ""),
        "fixed_torsion_delta_deg": row.get("fixed_torsion_delta_deg", ""),
        "solved_torsion_1_name": row.get("solved_torsion_1_name", ""),
        "solved_torsion_1_delta_deg": row.get("solved_torsion_1_delta_deg", ""),
        "solved_torsion_2_name": row.get("solved_torsion_2_name", ""),
        "solved_torsion_2_delta_deg": row.get("solved_torsion_2_delta_deg", ""),
        "omega_policy": row.get("omega_policy", ""),
        "endpoint_error_A": row.get("endpoint_error_A", ""),
        "max_ca_shift_A": row.get("max_ca_shift_A", ""),
        "max_backbone_bond_delta_A": row.get("max_backbone_bond_delta_A", ""),
        "max_backbone_angle_delta_deg": row.get("max_backbone_angle_delta_deg", ""),
        "max_omega_trans_deviation_deg": row.get("max_omega_trans_deviation_deg", ""),
        "coordinate_path": str(coordinate_path),
        "C_peak_A": profile_score["C_peak_d_A"],
        "D_peak_A": profile_score["D_peak_d_A"],
        "C_error_A": c_error,
        "D_error_A": d_error,
        "combined_abs_error_A": combined_abs_error(c_error, d_error),
        "C_peak_intensity_or_score": profile_score["C_peak_intensity"],
        "D_peak_intensity_or_score": profile_score["D_peak_intensity"],
        "notes": row.get("notes", ""),
    }


def score_safe_candidates(
    audit: pd.DataFrame,
    target_c: float = TARGET_C,
    target_d: float = TARGET_D,
    tolerance: float = TOLERANCE,
    q_step: float = 0.01,
    d_min: float = 2.5,
    d_max: float = 12.0,
) -> pd.DataFrame:
    """Score all safe candidates in the geometry audit."""
    safe = filter_safe_candidates(audit)
    rows = [
        score_candidate_row(row, target_c, target_d, tolerance, q_step, d_min, d_max)
        for _, row in safe.iterrows()
    ]
    return pd.DataFrame(rows)


def build_report_text(scores: pd.DataFrame, safe_count: int, skipped_count: int) -> str:
    """Build the markdown report body."""
    if scores.empty:
        return (
            "# Constrained Phi/Psi Candidate C/D Scores\n\n"
            "No geometry-safe candidates were available for scoring. This remains a tiny fixed-omega pilot.\n"
        )

    baseline = scores[scores["candidate_id"].astype(str).str.contains("cand_001|cand_002", regex=True)]
    cyp = scores[
        (scores["repeat_type"] == "CYP->GLU")
        & (scores["solve_mode"] == "one_torsion")
    ].copy()
    cyp = sort_by_fixed_delta(cyp)
    c_trend = monotonic_trend(cyp["C_peak_A"]) if not cyp.empty else "insufficient"
    d_trend = monotonic_trend(cyp["D_peak_A"]) if not cyp.empty else "insufficient"
    best = best_candidate(scores)

    columns = [
        "candidate_id",
        "repeat_type",
        "solve_mode",
        "fixed_torsion_delta_deg",
        "C_peak_A",
        "D_peak_A",
        "C_error_A",
        "D_error_A",
        "combined_abs_error_A",
    ]
    score_table = markdown_table(sort_by_fixed_delta(scores), columns)
    baseline_table = markdown_table(baseline, columns)

    return f"""# Constrained Phi/Psi Candidate C/D Scores

This is a tiny fixed-omega pilot for geometry-safe constrained phi/psi candidates only. Omega sensitivity is deferred, and the unsafe GLU->MEP two-torsion candidates are not scored because they failed the geometry audit.

## Counts

- Geometry-safe candidates scored: {safe_count}
- Unsafe candidates skipped: {skipped_count}
- C target: {TARGET_C:.3f} A
- D target: {TARGET_D:.3f} A

## Baseline Rows

{baseline_table}

## Score Table

{score_table}

## Trend Check

- CYP->GLU one-torsion C peak trend versus fixed phi delta: {c_trend}
- CYP->GLU one-torsion D peak trend versus fixed phi delta: {d_trend}
- Best candidate by combined C/D absolute error: `{best['candidate_id']}` with combined error {float(best['combined_abs_error_A']):.4f} A.

## Interpretation

Small safe CYP->GLU perturbations are scored here as a constrained local pilot, not as a full systematic scan. Use this table to decide whether the fixed-omega perturbation direction measurably moves C/D before generating a larger coordinate panel.
"""


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    """Render selected dataframe columns as markdown."""
    if df.empty:
        return "_None._"
    columns = [column for column in columns if column in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for record in df[columns].itertuples(index=False):
        values = []
        for value in record:
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def save_plot(scores: pd.DataFrame, figure_base: Path, target_c: float, target_d: float) -> None:
    """Plot C/D peak positions across safe CYP->GLU one-torsion perturbations."""
    cyp = scores[
        (scores["repeat_type"] == "CYP->GLU")
        & (scores["solve_mode"] == "one_torsion")
    ].copy()
    if cyp.empty:
        return
    cyp = sort_by_fixed_delta(cyp)
    x = pd.to_numeric(cyp["fixed_torsion_delta_deg"], errors="coerce")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, cyp["C_peak_A"], marker="o", label="C-like peak")
    ax.plot(x, cyp["D_peak_A"], marker="o", label="D-like peak")
    ax.axhline(target_c, color="#1f77b4", ls="--", lw=1, label="C target")
    ax.axhline(target_d, color="#ff7f0e", ls="--", lw=1, label="D target")
    ax.axvline(0, color="0.5", ls=":", lw=1)
    ax.set_xlabel("fixed phi0 delta (deg)")
    ax.set_ylabel("peak d-spacing (A)")
    ax.set_title("Geometry-safe constrained phi/psi pilot C/D scores")
    ax.legend(fontsize=8)
    fig.tight_layout()
    figure_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_base.with_suffix(".png"), dpi=180)
    fig.savefig(figure_base.with_suffix(".svg"))
    plt.close(fig)


def run(
    audit_csv: Path,
    score_csv: Path,
    report_path: Path,
    figure_base: Path,
    target_c: float,
    target_d: float,
    tolerance: float,
    q_step: float,
    d_min: float,
    d_max: float,
) -> pd.DataFrame:
    """Score safe candidates and write outputs."""
    audit = pd.read_csv(audit_csv)
    safe_count = len(filter_safe_candidates(audit))
    skipped_count = skipped_unsafe_count(audit)
    scores = score_safe_candidates(audit, target_c, target_d, tolerance, q_step, d_min, d_max)
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
    scores = run(
        args.audit_csv,
        args.score_csv,
        args.report,
        args.figure_base,
        args.target_c,
        args.target_d,
        args.tolerance,
        args.q_step,
        args.d_min,
        args.d_max,
    )
    print(f"Scored {len(scores)} geometry-safe candidates")
    print(f"CSV: {args.score_csv}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
