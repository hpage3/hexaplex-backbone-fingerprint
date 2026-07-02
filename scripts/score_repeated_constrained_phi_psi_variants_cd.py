"""Score geometry-safe repeated constrained phi/psi variants against C/D targets."""

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

from scripts.score_constrained_phi_psi_candidates_cd import (
    TARGET_C,
    TARGET_D,
    TOLERANCE,
    combined_abs_error,
    monotonic_trend,
    parse_bool,
)
from scripts.rollup_rich_coordinate_cd_diagnostics import score_pdb_profile


DEFAULT_AUDIT_CSV = Path("outputs/metrics/repeated_constrained_phi_psi_variant_geometry_audit.csv")
DEFAULT_SCORE_CSV = Path("outputs/metrics/repeated_constrained_phi_psi_variant_cd_scores.csv")
DEFAULT_REPORT = Path("outputs/reports/repeated_constrained_phi_psi_variant_cd_scores.md")
DEFAULT_FIGURE_BASE = Path("outputs/figures/repeated_constrained_phi_psi_variant_cd_scores")


def filter_safe_repeated_variants(audit: pd.DataFrame) -> pd.DataFrame:
    """Return repeated variants explicitly marked safe for diffraction scoring."""
    if "safe_for_diffraction_scoring" not in audit.columns:
        raise ValueError("Repeated variant audit is missing safe_for_diffraction_scoring.")
    return audit[audit["safe_for_diffraction_scoring"].map(parse_bool)].copy()


def skipped_unsafe_count(audit: pd.DataFrame) -> int:
    """Count repeated variants skipped by the geometry audit gate."""
    return len(audit) - len(filter_safe_repeated_variants(audit))


def best_variant(scores: pd.DataFrame) -> pd.Series:
    """Return row with lowest combined C/D absolute error."""
    if scores.empty:
        raise ValueError("No repeated variant scores available.")
    values = pd.to_numeric(scores["combined_abs_error_A"], errors="coerce")
    return scores.loc[values.idxmin()]


def sort_by_fixed_delta(df: pd.DataFrame) -> pd.DataFrame:
    """Sort rows by fixed torsion delta."""
    out = df.copy()
    out["fixed_torsion_delta_deg"] = pd.to_numeric(out["fixed_torsion_delta_deg"], errors="coerce")
    return out.sort_values(["fixed_torsion_delta_deg", "variant_id"]).reset_index(drop=True)


def score_variant_row(
    row: pd.Series,
    target_c: float,
    target_d: float,
    tolerance: float,
    q_step: float,
    d_min: float,
    d_max: float,
) -> dict[str, object]:
    """Score one safe repeated variant PDB."""
    coordinate_path = Path(str(row["coordinate_path"]))
    score = score_pdb_profile(coordinate_path, target_c, target_d, tolerance, q_step, d_min, d_max)
    c_error = score["C_error_A"]
    d_error = score["D_error_A"]
    return {
        "variant_id": row.get("variant_id", ""),
        "fixed_torsion_delta_deg": row.get("fixed_torsion_delta_deg", ""),
        "omega_policy": row.get("omega_policy", ""),
        "attempted_window_count": row.get("attempted_window_count", ""),
        "applied_window_count": row.get("applied_window_count", ""),
        "skipped_window_count": row.get("skipped_window_count", ""),
        "max_endpoint_error_A": row.get("max_endpoint_error_A", ""),
        "max_ca_anchor_shift_A": row.get("max_ca_shift_A", ""),
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


def score_safe_variants(
    audit: pd.DataFrame,
    target_c: float = TARGET_C,
    target_d: float = TARGET_D,
    tolerance: float = TOLERANCE,
    q_step: float = 0.01,
    d_min: float = 2.5,
    d_max: float = 12.0,
) -> pd.DataFrame:
    """Score all safe repeated variants."""
    rows = [
        score_variant_row(row, target_c, target_d, tolerance, q_step, d_min, d_max)
        for _, row in filter_safe_repeated_variants(audit).iterrows()
    ]
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    """Render selected dataframe columns as markdown."""
    if df.empty:
        return "_None._"
    columns = [column for column in columns if column in df.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for record in df[columns].itertuples(index=False):
        vals = []
        for value in record:
            if isinstance(value, float):
                vals.append(f"{value:.4f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def build_report_text(scores: pd.DataFrame, safe_count: int, skipped_count: int) -> str:
    """Build markdown report text for repeated variant C/D scoring."""
    if scores.empty:
        return (
            "# Repeated Constrained Phi/Psi Variant C/D Scores\n\n"
            "No geometry-safe repeated variants were available for scoring. "
            "This remains a tiny fixed-omega pilot; unsafe variants are excluded.\n"
        )

    sorted_scores = sort_by_fixed_delta(scores)
    best = best_variant(sorted_scores)
    c_trend = monotonic_trend(sorted_scores["C_peak_A"])
    d_trend = monotonic_trend(sorted_scores["D_peak_A"])
    columns = [
        "variant_id",
        "fixed_torsion_delta_deg",
        "C_peak_A",
        "D_peak_A",
        "C_error_A",
        "D_error_A",
        "combined_abs_error_A",
    ]
    return f"""# Repeated Constrained Phi/Psi Variant C/D Scores

This is a tiny fixed-omega pilot for geometry-safe repeated CYP->GLU variants only. Unsafe repeated variants were not scored because they failed bond/angle geometry thresholds. Omega sensitivity remains deferred.

## Counts

- Repeated variants scored: {safe_count}
- Repeated variants skipped as unsafe: {skipped_count}
- C target: {TARGET_C:.3f} A
- D target: {TARGET_D:.3f} A

## Per-Delta C/D Scores

{markdown_table(sorted_scores, columns)}

## Trend Check

- C peak trend across safe deltas: {c_trend}
- D peak trend across safe deltas: {d_trend}
- Best repeated variant by combined C/D absolute error: `{best['variant_id']}` with combined error {float(best['combined_abs_error_A']):.4f} A.

## Interpretation

Compare these rows with the local single-window pilot: if the repeated perturbation changes C/D, it means coherent application across equivalent CYP->GLU windows is large enough to move the global Debye peaks. This is still not the full systematic search.
"""


def save_plot(scores: pd.DataFrame, figure_base: Path, target_c: float, target_d: float) -> None:
    """Plot C/D peak positions by repeated fixed torsion delta."""
    if scores.empty:
        return
    scores = sort_by_fixed_delta(scores)
    x = pd.to_numeric(scores["fixed_torsion_delta_deg"], errors="coerce")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, scores["C_peak_A"], marker="o", label="C-like peak")
    ax.plot(x, scores["D_peak_A"], marker="o", label="D-like peak")
    ax.axhline(target_c, color="#1f77b4", ls="--", lw=1, label="C target")
    ax.axhline(target_d, color="#ff7f0e", ls="--", lw=1, label="D target")
    ax.axvline(0, color="0.5", ls=":", lw=1)
    ax.set_xlabel("fixed phi0 delta applied repeatedly (deg)")
    ax.set_ylabel("peak d-spacing (A)")
    ax.set_title("Repeated constrained phi/psi pilot C/D scores")
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
    """Score safe repeated variants and write outputs."""
    audit = pd.read_csv(audit_csv)
    safe_count = len(filter_safe_repeated_variants(audit))
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
    print(f"Scored {len(scores)} geometry-safe repeated variants")
    print(f"CSV: {args.score_csv}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
