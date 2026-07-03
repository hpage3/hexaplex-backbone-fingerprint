"""Diagnose local C/D profiles for fine axial variants."""

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
from scripts.score_fine_axial_profile_variant_cd import REFERENCE_AXIAL_SCALE, reference_row
from scripts.score_axial_only_extension_variant_cd import sort_by_axial


DEFAULT_SCORE_CSV = Path("outputs/metrics/fine_axial_profile_variant_cd_scores.csv")
DEFAULT_OUT_CSV = Path("outputs/metrics/fine_axial_profile_cd_profile_diagnostics.csv")
DEFAULT_REPORT = Path("outputs/reports/fine_axial_profile_cd_profile_diagnostics.md")


def diagnostic_band_row(
    score_row: pd.Series,
    profile: pd.DataFrame,
    reference_profile: pd.DataFrame,
    band: str,
    target: float,
    tolerance: float,
) -> dict[str, object]:
    """Build diagnostics for one fine axial variant and one band."""
    window = local_window(profile, target, tolerance)
    ref_window = local_window(reference_profile, target, tolerance)
    d = window["d_A"].to_numpy(float)
    intensity = window["intensity"].to_numpy(float)
    ref_d = ref_window["d_A"].to_numpy(float)
    ref_intensity = ref_window["intensity"].to_numpy(float)

    picked = nearest_peak(profile, target, tolerance)
    ref_picked = nearest_peak(reference_profile, target, tolerance)
    centroid = intensity_weighted_centroid(d, intensity)
    ref_centroid = intensity_weighted_centroid(ref_d, ref_intensity)
    parabola = parabolic_peak_estimate(d, intensity)
    ref_parabola = parabolic_peak_estimate(ref_d, ref_intensity)
    diffs = profile_difference_metrics(d, intensity, ref_d, ref_intensity)
    integral = integrated_intensity(d, intensity)
    ref_integral = integrated_intensity(ref_d, ref_intensity)
    return {
        "variant_id": score_row["variant_id"],
        "axial_scale_z": score_row["axial_scale_z"],
        "band": band,
        "picked_peak_A": picked.peak_d_A,
        "picked_peak_shift_vs_0p9700_A": picked.peak_d_A - ref_picked.peak_d_A,
        "picked_peak_score": picked.intensity,
        "local_centroid_A": centroid,
        "centroid_shift_vs_0p9700_A": None if centroid is None or ref_centroid is None else centroid - ref_centroid,
        "local_parabolic_peak_A": parabola,
        "parabolic_shift_vs_0p9700_A": None if parabola is None or ref_parabola is None else parabola - ref_parabola,
        "local_profile_l2_diff_vs_0p9700": diffs["l2"],
        "local_profile_max_abs_diff_vs_0p9700": diffs["max_abs"],
        "local_profile_corr_vs_0p9700": diffs["corr"],
        "integrated_intensity": integral,
        "integrated_intensity_relative_vs_0p9700": None
        if integral is None or ref_integral in (None, 0)
        else integral / ref_integral,
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
    """Run fine axial local profile diagnostics and write outputs."""
    scores = sort_by_axial(pd.read_csv(score_csv))
    reference = reference_row(scores, REFERENCE_AXIAL_SCALE)
    reference_profile = compute_profile(Path(str(reference["input_pdb"])), q_step, d_min, d_max)
    rows = []
    for _, row in scores.iterrows():
        profile = compute_profile(Path(str(row["input_pdb"])), q_step, d_min, d_max)
        rows.append(diagnostic_band_row(row, profile, reference_profile, "C", target_c, tolerance))
        rows.append(diagnostic_band_row(row, profile, reference_profile, "D", target_d, tolerance))
    diagnostics = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(out_csv, index=False)
    report_path.write_text(build_report_text(diagnostics, scores, reference), encoding="utf-8")
    return diagnostics


def build_report_text(diagnostics: pd.DataFrame, scores: pd.DataFrame, reference: pd.Series) -> str:
    """Build fine axial profile diagnostic report."""
    cols = [
        "variant_id",
        "axial_scale_z",
        "band",
        "picked_peak_A",
        "picked_peak_shift_vs_0p9700_A",
        "local_centroid_A",
        "centroid_shift_vs_0p9700_A",
        "local_parabolic_peak_A",
        "parabolic_shift_vs_0p9700_A",
        "integrated_intensity_relative_vs_0p9700",
    ]
    shape_cols = [
        "variant_id",
        "axial_scale_z",
        "band",
        "local_profile_l2_diff_vs_0p9700",
        "local_profile_max_abs_diff_vs_0p9700",
        "local_profile_corr_vs_0p9700",
    ]
    c_diag = diagnostics[diagnostics["band"] == "C"]
    d_diag = diagnostics[diagnostics["band"] == "D"]
    return f"""# Fine Axial Local C/D Profile Diagnostics

## Purpose

This diagnostic asks whether the local C profile moves smoothly underneath discretized picked peaks in the fine axial 0.9700 to 0.9850 window.

- Reference variant: `{reference['variant_id']}`
- Variants analyzed: {scores['variant_id'].nunique()}
- Band rows: {len(diagnostics)}

## Peak, Centroid, And Parabolic Diagnostics

{markdown_table(diagnostics, cols)}

- Maximum absolute C centroid shift: {max_abs_shift(diagnostics, 'C', 'centroid_shift_vs_0p9700_A'):.6g} A
- Maximum absolute D centroid shift: {max_abs_shift(diagnostics, 'D', 'centroid_shift_vs_0p9700_A'):.6g} A
- Maximum absolute C parabolic shift: {max_abs_shift(diagnostics, 'C', 'parabolic_shift_vs_0p9700_A'):.6g} A
- Maximum absolute D parabolic shift: {max_abs_shift(diagnostics, 'D', 'parabolic_shift_vs_0p9700_A'):.6g} A

## Profile Shape Diagnostics

{markdown_table(diagnostics, shape_cols)}

- C integrated intensity relative range: {range_text(diagnostics, 'C', 'integrated_intensity_relative_vs_0p9700')}
- D integrated intensity relative range: {range_text(diagnostics, 'D', 'integrated_intensity_relative_vs_0p9700')}

## Interpretation

If centroid/parabolic C shifts change monotonically while picked peaks stay binned, that supports smooth sub-bin motion underneath discretized picked peaks. If not, the diagnostic does not support smooth sub-bin motion. D should be read separately as a position-stability check. These variants are controlled diagnostic perturbations, not minimized structures.
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
    print(f"Generated {len(diagnostics)} fine axial local profile diagnostic rows")
    print(f"CSV: {args.out_csv}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
