"""Diagnose local C/D profile shapes for coupled CYP->GLU + GLU->MEP variants."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hexaplex_backbone_fingerprint.parametric_powder_scan import debye_profile, make_q_grid, nearest_peak
from scripts.rollup_rich_coordinate_cd_diagnostics import read_pdb_coordinates
from scripts.score_constrained_phi_psi_candidates_cd import TARGET_C, TARGET_D, TOLERANCE, parse_bool
from scripts.score_coupled_cyp_glu_glu_mep_variant_cd import baseline_control_row, sort_by_coupled_deltas


DEFAULT_SCORE_CSV = Path("outputs/metrics/coupled_cyp_glu_glu_mep_variant_cd_scores.csv")
DEFAULT_OUT_CSV = Path("outputs/metrics/coupled_cyp_glu_glu_mep_cd_profile_diagnostics.csv")
DEFAULT_REPORT = Path("outputs/reports/coupled_cyp_glu_glu_mep_cd_profile_diagnostics.md")


def intensity_weighted_centroid(d_values: np.ndarray, intensities: np.ndarray) -> float | None:
    """Return intensity-weighted centroid for a local profile window."""
    d_values = np.asarray(d_values, dtype=float)
    intensities = np.asarray(intensities, dtype=float)
    if len(d_values) == 0 or len(d_values) != len(intensities):
        return None
    total = float(np.sum(intensities))
    if not math.isfinite(total) or abs(total) <= 1e-12:
        return None
    return float(np.sum(d_values * intensities) / total)


def parabolic_peak_estimate(d_values: np.ndarray, intensities: np.ndarray) -> float | None:
    """Estimate sub-bin peak position with a 3-point parabola around the local maximum."""
    d_values = np.asarray(d_values, dtype=float)
    intensities = np.asarray(intensities, dtype=float)
    if len(d_values) < 3 or len(d_values) != len(intensities):
        return None
    peak_index = int(np.argmax(intensities))
    if peak_index == 0 or peak_index == len(intensities) - 1:
        return None
    x = d_values[peak_index - 1 : peak_index + 2]
    y = intensities[peak_index - 1 : peak_index + 2]
    if len(np.unique(x)) < 3:
        return None
    a, b, _ = np.polyfit(x, y, 2)
    if abs(a) <= 1e-12:
        return None
    vertex = -b / (2.0 * a)
    if vertex < min(x) or vertex > max(x):
        return None
    return float(vertex)


def profile_difference_metrics(
    d_values: np.ndarray,
    intensities: np.ndarray,
    baseline_d_values: np.ndarray,
    baseline_intensities: np.ndarray,
) -> dict[str, float]:
    """Return L2, max absolute difference, and correlation against baseline local profile."""
    d_values = np.asarray(d_values, dtype=float)
    intensities = np.asarray(intensities, dtype=float)
    baseline_d_values = np.asarray(baseline_d_values, dtype=float)
    baseline_intensities = np.asarray(baseline_intensities, dtype=float)
    if len(baseline_d_values) == 0:
        return {"l2": float("nan"), "max_abs": float("nan"), "corr": float("nan")}
    if len(d_values) != len(baseline_d_values) or not np.allclose(d_values, baseline_d_values):
        intensities = np.interp(baseline_d_values, d_values, intensities)
    diff = intensities - baseline_intensities
    l2 = float(np.linalg.norm(diff))
    max_abs = float(np.max(np.abs(diff))) if len(diff) else float("nan")
    if np.std(intensities) <= 1e-12 or np.std(baseline_intensities) <= 1e-12:
        corr = 1.0 if np.allclose(intensities, baseline_intensities) else float("nan")
    else:
        corr = float(np.corrcoef(intensities, baseline_intensities)[0, 1])
    return {"l2": l2, "max_abs": max_abs, "corr": corr}


def integrated_intensity(d_values: np.ndarray, intensities: np.ndarray) -> float | None:
    """Integrate local profile intensity over d-spacing."""
    d_values = np.asarray(d_values, dtype=float)
    intensities = np.asarray(intensities, dtype=float)
    if len(d_values) == 0 or len(d_values) != len(intensities):
        return None
    return float(np.trapezoid(intensities, d_values))


def classify_tiny_shifts(values: pd.Series, tolerance_A: float = 1e-4) -> str:
    """Classify local shifts as flat/tiny or nonzero using a documented tolerance."""
    numeric = pd.to_numeric(values, errors="coerce").dropna().abs()
    if numeric.empty:
        return "not feasible"
    maximum = float(numeric.max())
    if maximum <= tolerance_A:
        return f"flat/tiny (max <= {tolerance_A:g} A)"
    return f"nonzero (max {maximum:.6g} A)"


def local_window(profile: pd.DataFrame, target: float, tolerance: float) -> pd.DataFrame:
    """Return local target +/- tolerance profile window sorted by d-spacing."""
    return profile[profile["d_A"].between(target - tolerance, target + tolerance)].copy().reset_index(drop=True)


def compute_profile(pdb_path: Path, q_step: float, d_min: float, d_max: float) -> pd.DataFrame:
    """Compute Debye profile on the same grid used by existing scorers."""
    coords = read_pdb_coordinates(pdb_path, exclude_hydrogen=False)
    return debye_profile(coords, make_q_grid(d_min_A=d_min, d_max_A=d_max, q_step=q_step))


def band_row(
    score_row: pd.Series,
    profile: pd.DataFrame,
    baseline_profile: pd.DataFrame,
    baseline_score_row: pd.Series,
    band: str,
    target: float,
    tolerance: float,
) -> dict[str, object]:
    """Build diagnostics for one variant and one local C/D band window."""
    window = local_window(profile, target, tolerance)
    baseline_window = local_window(baseline_profile, target, tolerance)
    d = window["d_A"].to_numpy(float)
    intensity = window["intensity"].to_numpy(float)
    baseline_d = baseline_window["d_A"].to_numpy(float)
    baseline_intensity = baseline_window["intensity"].to_numpy(float)

    picked = nearest_peak(profile, target, tolerance)
    baseline_picked = nearest_peak(baseline_profile, target, tolerance)
    centroid = intensity_weighted_centroid(d, intensity)
    baseline_centroid = intensity_weighted_centroid(baseline_d, baseline_intensity)
    parabola = parabolic_peak_estimate(d, intensity)
    baseline_parabola = parabolic_peak_estimate(baseline_d, baseline_intensity)
    diffs = profile_difference_metrics(d, intensity, baseline_d, baseline_intensity)
    integral = integrated_intensity(d, intensity)
    baseline_integral = integrated_intensity(baseline_d, baseline_intensity)
    notes = "raw_debye_intensity_used"
    if len(d) != len(baseline_d) or not np.allclose(d, baseline_d):
        notes += "; interpolated_to_baseline_grid"

    return {
        "variant_id": score_row["variant_id"],
        "cyp_glu_delta_deg": score_row["cyp_glu_delta_deg"],
        "glu_mep_delta_deg": score_row["glu_mep_delta_deg"],
        "band": band,
        "picked_peak_A": picked.peak_d_A,
        "picked_peak_shift_vs_baseline_A": picked.peak_d_A - baseline_picked.peak_d_A,
        "picked_peak_score": picked.intensity,
        "local_centroid_A": centroid,
        "centroid_shift_vs_baseline_A": None if centroid is None or baseline_centroid is None else centroid - baseline_centroid,
        "local_parabolic_peak_A": parabola,
        "parabolic_shift_vs_baseline_A": None if parabola is None or baseline_parabola is None else parabola - baseline_parabola,
        "local_profile_l2_diff_vs_baseline": diffs["l2"],
        "local_profile_max_abs_diff_vs_baseline": diffs["max_abs"],
        "local_profile_corr_vs_baseline": diffs["corr"],
        "integrated_intensity": integral,
        "integrated_intensity_relative_vs_baseline": None
        if integral is None or baseline_integral in (None, 0)
        else integral / baseline_integral,
        "notes": notes,
    }


def score_rows_for_diagnostics(scores: pd.DataFrame) -> pd.DataFrame:
    """Return geometry-safe coupled score rows sorted by coupled deltas."""
    if "geometry_safe" not in scores.columns:
        raise ValueError("Coupled score CSV is missing geometry_safe.")
    safe = scores[scores["geometry_safe"].map(parse_bool)].copy()
    return sort_by_coupled_deltas(safe)


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
    """Run local C/D profile diagnostics and write CSV/report."""
    scores = score_rows_for_diagnostics(pd.read_csv(score_csv))
    baseline = baseline_control_row(scores)
    baseline_profile = compute_profile(Path(str(baseline["input_pdb"])), q_step, d_min, d_max)
    profiles: dict[str, pd.DataFrame] = {str(baseline["variant_id"]): baseline_profile}
    rows = []
    for _, row in scores.iterrows():
        variant_id = str(row["variant_id"])
        profile = profiles.get(variant_id)
        if profile is None:
            profile = compute_profile(Path(str(row["input_pdb"])), q_step, d_min, d_max)
            profiles[variant_id] = profile
        rows.append(band_row(row, profile, baseline_profile, baseline, "C", target_c, tolerance))
        rows.append(band_row(row, profile, baseline_profile, baseline, "D", target_d, tolerance))

    diagnostics = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(out_csv, index=False)
    report_path.write_text(build_report_text(diagnostics, scores, baseline), encoding="utf-8")
    return diagnostics


def max_abs_shift(df: pd.DataFrame, band: str, column: str) -> float:
    """Return maximum absolute shift for one band/column."""
    values = pd.to_numeric(df[df["band"] == band][column], errors="coerce").dropna().abs()
    return float(values.max()) if not values.empty else float("nan")


def range_text(df: pd.DataFrame, band: str, column: str) -> str:
    """Return min-max text for one band/column."""
    values = pd.to_numeric(df[df["band"] == band][column], errors="coerce").dropna()
    if values.empty:
        return "not available"
    return f"{values.min():.6g}-{values.max():.6g}"


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


def build_report_text(diagnostics: pd.DataFrame, scores: pd.DataFrame, baseline: pd.Series) -> str:
    """Build diagnostic markdown report."""
    picked_summary = diagnostics.pivot_table(
        index="band",
        values="picked_peak_shift_vs_baseline_A",
        aggfunc=lambda x: classify_tiny_shifts(pd.Series(x), tolerance_A=1e-8),
    )
    centroid_cols = [
        "variant_id",
        "cyp_glu_delta_deg",
        "glu_mep_delta_deg",
        "band",
        "centroid_shift_vs_baseline_A",
    ]
    parabolic_cols = [
        "variant_id",
        "cyp_glu_delta_deg",
        "glu_mep_delta_deg",
        "band",
        "parabolic_shift_vs_baseline_A",
    ]
    shape_cols = [
        "variant_id",
        "cyp_glu_delta_deg",
        "glu_mep_delta_deg",
        "band",
        "local_profile_l2_diff_vs_baseline",
        "local_profile_max_abs_diff_vs_baseline",
        "local_profile_corr_vs_baseline",
        "integrated_intensity_relative_vs_baseline",
    ]
    text = f"""# Coupled CYP->GLU + GLU->MEP Local C/D Profile Diagnostics

## Purpose

This diagnostic distinguishes truly flat C/D peak positions from possible binning/peak-picking artifacts by inspecting local Debye radial-profile shapes around C and D.

## Inputs

- Coupled score CSV: `outputs/metrics/coupled_cyp_glu_glu_mep_variant_cd_scores.csv`
- Geometry-audited coupled variants: {len(scores)}
- Baseline/control variant: `{baseline['variant_id']}` (CYP->GLU delta 0, GLU->MEP delta 0)

## Picked Peak Result

Picked peak shifts versus baseline:

{markdown_table(picked_summary.reset_index(), ['band', 'picked_peak_shift_vs_baseline_A'])}

## Centroid Diagnostics

{markdown_table(diagnostics, centroid_cols)}

- Maximum absolute C centroid shift: {max_abs_shift(diagnostics, 'C', 'centroid_shift_vs_baseline_A'):.6g} A
- Maximum absolute D centroid shift: {max_abs_shift(diagnostics, 'D', 'centroid_shift_vs_baseline_A'):.6g} A

## Parabolic Sub-Bin Diagnostics

{markdown_table(diagnostics, parabolic_cols)}

- Maximum absolute C parabolic shift: {max_abs_shift(diagnostics, 'C', 'parabolic_shift_vs_baseline_A'):.6g} A
- Maximum absolute D parabolic shift: {max_abs_shift(diagnostics, 'D', 'parabolic_shift_vs_baseline_A'):.6g} A

## Profile-Shape Diagnostics

{markdown_table(diagnostics, shape_cols)}

- C integrated intensity relative range: {range_text(diagnostics, 'C', 'integrated_intensity_relative_vs_baseline')}
- D integrated intensity relative range: {range_text(diagnostics, 'D', 'integrated_intensity_relative_vs_baseline')}
- Maximum C L2 difference: {max_abs_shift(diagnostics, 'C', 'local_profile_l2_diff_vs_baseline'):.6g}
- Maximum D L2 difference: {max_abs_shift(diagnostics, 'D', 'local_profile_l2_diff_vs_baseline'):.6g}

## Interpretation

If centroid and parabolic shifts are essentially zero or tiny, the local profile diagnostics support true robustness of C/D positions within this coupled safe basin. If small sub-bin shifts exist, interpret their magnitude and direction cautiously. This is not a new structural proof; it is a local profile diagnostic around the existing coupled perturbation pilot.
"""
    return text


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
    diagnostics = run_diagnostics(
        args.score_csv,
        args.out_csv,
        args.report,
        args.target_c,
        args.target_d,
        args.tolerance,
        args.q_step,
        args.d_min,
        args.d_max,
    )
    print(f"Generated {len(diagnostics)} coupled C/D local profile diagnostic rows")
    print(f"CSV: {args.out_csv}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
