"""Diagnose local C/D profiles for parameterized rise variants."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hexaplex_backbone_fingerprint.parametric_powder_scan import nearest_peak
from scripts.diagnose_coupled_cyp_glu_glu_mep_cd_profiles import (
    compute_profile,
    integrated_intensity,
    intensity_weighted_centroid,
    local_window,
    markdown_table,
    max_abs_shift,
    parabolic_peak_estimate,
    profile_difference_metrics,
    range_text,
)
from scripts.score_constrained_phi_psi_candidates_cd import TARGET_C, TARGET_D, TOLERANCE
from scripts.score_parameterized_rise_variant_cd import BASELINE_RISE_SCALE, baseline_row, sort_by_rise


DEFAULT_SCORE_CSV = Path("outputs/metrics/parameterized_rise_variant_cd_scores.csv")
DEFAULT_OUT_CSV = Path("outputs/metrics/parameterized_rise_cd_profile_diagnostics.csv")
DEFAULT_REPORT = Path("outputs/reports/parameterized_rise_cd_profile_diagnostics.md")
RISE_LIKE_DIAG_CSV = Path("outputs/metrics/rise_like_cd_profile_diagnostics.csv")


def diagnostic_band_row(
    score_row: pd.Series,
    profile: pd.DataFrame,
    baseline_profile: pd.DataFrame,
    band: str,
    target: float,
    tolerance: float,
) -> dict[str, object]:
    """Build diagnostics for one parameterized rise variant and one band."""
    window = local_window(profile, target, tolerance)
    base_window = local_window(baseline_profile, target, tolerance)
    d = window["d_A"].to_numpy(float)
    intensity = window["intensity"].to_numpy(float)
    base_d = base_window["d_A"].to_numpy(float)
    base_intensity = base_window["intensity"].to_numpy(float)
    picked = nearest_peak(profile, target, tolerance)
    base_picked = nearest_peak(baseline_profile, target, tolerance)
    centroid = intensity_weighted_centroid(d, intensity)
    base_centroid = intensity_weighted_centroid(base_d, base_intensity)
    parabola = parabolic_peak_estimate(d, intensity)
    base_parabola = parabolic_peak_estimate(base_d, base_intensity)
    diffs = profile_difference_metrics(d, intensity, base_d, base_intensity)
    integral = integrated_intensity(d, intensity)
    base_integral = integrated_intensity(base_d, base_intensity)
    return {
        "variant_id": score_row["variant_id"],
        "rise_scale": score_row["rise_scale"],
        "band": band,
        "picked_peak_A": picked.peak_d_A,
        "picked_peak_shift_vs_baseline_A": picked.peak_d_A - base_picked.peak_d_A,
        "picked_peak_score": picked.intensity,
        "local_centroid_A": centroid,
        "centroid_shift_vs_baseline_A": None if centroid is None or base_centroid is None else centroid - base_centroid,
        "local_parabolic_peak_A": parabola,
        "parabolic_shift_vs_baseline_A": None if parabola is None or base_parabola is None else parabola - base_parabola,
        "integrated_intensity": integral,
        "integrated_intensity_relative_vs_baseline": None
        if integral is None or base_integral in (None, 0)
        else integral / base_integral,
        "local_profile_l2_diff_vs_baseline": diffs["l2"],
        "local_profile_max_abs_diff_vs_baseline": diffs["max_abs"],
        "local_profile_corr_vs_baseline": diffs["corr"],
        "notes": "raw_debye_intensity_used",
    }


def run_diagnostics(
    score_csv: Path,
    out_csv: Path,
    report_path: Path,
    target_c: float,
    target_d: float,
    tolerance: float,
    q_step: float,
    d_min: float,
    d_max: float,
) -> pd.DataFrame:
    """Run parameterized rise local profile diagnostics and write outputs."""
    scores = sort_by_rise(pd.read_csv(score_csv))
    baseline = baseline_row(scores, BASELINE_RISE_SCALE)
    baseline_profile = compute_profile(Path(str(baseline["input_pdb"])), q_step, d_min, d_max)
    rows = []
    for _, row in scores.iterrows():
        profile = compute_profile(Path(str(row["input_pdb"])), q_step, d_min, d_max)
        rows.append(diagnostic_band_row(row, profile, baseline_profile, "C", target_c, tolerance))
        rows.append(diagnostic_band_row(row, profile, baseline_profile, "D", target_d, tolerance))
    diagnostics = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(out_csv, index=False)
    report_path.write_text(build_report_text(diagnostics, scores, baseline), encoding="utf-8")
    return diagnostics


def rise_like_profile_comparison_text(path: Path = RISE_LIKE_DIAG_CSV) -> str:
    """Return short comparison to rise_like profile diagnostics."""
    if not path.exists():
        return "Earlier rise_like profile diagnostics were not available for direct comparison."
    return f"Earlier rise_like profile diagnostics are available at `{path}` for direct side-by-side comparison."


def build_report_text(diagnostics: pd.DataFrame, scores: pd.DataFrame, baseline: pd.Series) -> str:
    """Build parameterized rise profile diagnostic report."""
    cols = [
        "variant_id",
        "rise_scale",
        "band",
        "picked_peak_A",
        "picked_peak_shift_vs_baseline_A",
        "local_centroid_A",
        "centroid_shift_vs_baseline_A",
        "local_parabolic_peak_A",
        "parabolic_shift_vs_baseline_A",
        "integrated_intensity_relative_vs_baseline",
    ]
    return f"""# Parameterized Rise Local C/D Profile Diagnostics

## Purpose

This parameterized rise diagnostic checks local C/D profile motion relative to `{baseline['variant_id']}`.

- Variants analyzed: {scores['variant_id'].nunique()}
- Band rows: {len(diagnostics)}

## Peak, Centroid, And Parabolic Diagnostics

{markdown_table(diagnostics, cols)}

- Maximum absolute C centroid shift: {max_abs_shift(diagnostics, 'C', 'centroid_shift_vs_baseline_A'):.6g} A
- Maximum absolute D centroid shift: {max_abs_shift(diagnostics, 'D', 'centroid_shift_vs_baseline_A'):.6g} A
- Maximum absolute C parabolic shift: {max_abs_shift(diagnostics, 'C', 'parabolic_shift_vs_baseline_A'):.6g} A
- Maximum absolute D parabolic shift: {max_abs_shift(diagnostics, 'D', 'parabolic_shift_vs_baseline_A'):.6g} A
- C integrated intensity relative range: {range_text(diagnostics, 'C', 'integrated_intensity_relative_vs_baseline')}
- D integrated intensity relative range: {range_text(diagnostics, 'D', 'integrated_intensity_relative_vs_baseline')}

## Comparison To rise_like Profile Diagnostics

{rise_like_profile_comparison_text()}

## Interpretation

C moves smoothly if centroid/parabolic estimates progress with parameterized rise compression even when picked peaks are discretized. D remains stable if picked peak and local profile estimates stay near baseline; otherwise D jumps under stronger compression.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-csv", type=Path, default=DEFAULT_SCORE_CSV)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
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
    diagnostics = run_diagnostics(args.score_csv, args.out_csv, args.report, args.target_c, args.target_d, args.tolerance, args.q_step, args.d_min, args.d_max)
    print(f"Generated {len(diagnostics)} parameterized rise local profile diagnostic rows")
    print(f"CSV: {args.out_csv}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
