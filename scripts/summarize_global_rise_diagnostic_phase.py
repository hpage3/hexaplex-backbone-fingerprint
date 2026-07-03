"""Summarize the global/rise C/D diagnostic phase from existing outputs."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.score_radial_axial_refinement_variant_cd import markdown_table


DEFAULT_OUT_CSV = Path("outputs/metrics/global_rise_diagnostic_phase_summary.csv")
DEFAULT_REPORT = Path("outputs/reports/global_rise_diagnostic_phase_summary.md")

INPUT_PATHS = [
    Path("outputs/reports/constrained_backbone_search_phase_summary.md"),
    Path("outputs/metrics/global_deformation_variant_cd_scores.csv"),
    Path("outputs/metrics/global_deformation_variant_geometry_audit.csv"),
    Path("outputs/reports/global_deformation_variant_cd_scores.md"),
    Path("outputs/metrics/radial_axial_refinement_variant_cd_scores.csv"),
    Path("outputs/metrics/radial_axial_refinement_variant_geometry_audit.csv"),
    Path("outputs/reports/radial_axial_refinement_variant_cd_scores.md"),
    Path("outputs/metrics/axial_only_extension_variant_cd_scores.csv"),
    Path("outputs/metrics/axial_only_extension_variant_geometry_audit.csv"),
    Path("outputs/reports/axial_only_extension_variant_cd_scores.md"),
    Path("outputs/metrics/fine_axial_profile_variant_cd_scores.csv"),
    Path("outputs/metrics/fine_axial_profile_cd_profile_diagnostics.csv"),
    Path("outputs/reports/fine_axial_profile_cd_profile_diagnostics.md"),
    Path("outputs/metrics/rise_like_variant_cd_scores.csv"),
    Path("outputs/metrics/rise_like_variant_geometry_audit.csv"),
    Path("outputs/metrics/rise_like_cd_profile_diagnostics.csv"),
    Path("outputs/reports/rise_like_variant_cd_scores.md"),
    Path("outputs/reports/rise_like_cd_profile_diagnostics.md"),
]


PHASE_INPUTS = {
    "global_deformation": {
        "scores": Path("outputs/metrics/global_deformation_variant_cd_scores.csv"),
        "geometry": Path("outputs/metrics/global_deformation_variant_geometry_audit.csv"),
        "branch": "global deformation pilot",
    },
    "radial_axial_refinement": {
        "scores": Path("outputs/metrics/radial_axial_refinement_variant_cd_scores.csv"),
        "geometry": Path("outputs/metrics/radial_axial_refinement_variant_geometry_audit.csv"),
        "branch": "focused radial/axial refinement",
    },
    "axial_only_extension": {
        "scores": Path("outputs/metrics/axial_only_extension_variant_cd_scores.csv"),
        "geometry": Path("outputs/metrics/axial_only_extension_variant_geometry_audit.csv"),
        "branch": "axial-only extension",
    },
    "fine_axial_profile_diagnostic": {
        "scores": Path("outputs/metrics/fine_axial_profile_variant_cd_scores.csv"),
        "geometry": Path("outputs/metrics/fine_axial_profile_variant_geometry_audit.csv"),
        "branch": "fine axial profile diagnostic",
    },
    "rise_like_diagnostic": {
        "scores": Path("outputs/metrics/rise_like_variant_cd_scores.csv"),
        "geometry": Path("outputs/metrics/rise_like_variant_geometry_audit.csv"),
        "branch": "rise-like diagnostic",
    },
}


def safe_float(value: object, default: float | None = math.nan) -> float | None:
    """Parse a float from a CSV value, returning default for blanks and invalid values."""
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_csv_if_present(path: Path) -> pd.DataFrame | None:
    """Read CSV if present."""
    if not path.exists():
        return None
    return pd.read_csv(path)


def bool_series(series: pd.Series) -> pd.Series:
    """Convert common CSV bool representations to boolean."""
    return series.map(lambda value: str(value).strip().lower() in {"true", "1", "yes"})


def summarize_geometry_counts(geometry: pd.DataFrame | None) -> tuple[object, object]:
    """Return generated/audited count and geometry-interpretable count."""
    if geometry is None:
        return "", ""
    generated = len(geometry)
    if "geometry_interpretable" in geometry.columns:
        interpretable = int(bool_series(geometry["geometry_interpretable"]).sum())
    else:
        interpretable = ""
    return generated, interpretable


def best_score_row(scores: pd.DataFrame) -> pd.Series:
    """Return the row with lowest combined C/D absolute error."""
    if scores.empty or "combined_abs_error_A" not in scores.columns:
        raise ValueError("Score table is empty or missing combined_abs_error_A.")
    values = pd.to_numeric(scores["combined_abs_error_A"], errors="coerce")
    if values.isna().all():
        raise ValueError("No numeric combined_abs_error_A values are available.")
    return scores.loc[values.idxmin()]


def trend_strings_for_phase(phase: str) -> tuple[str, str, str]:
    """Return C trend, D trend, and interpretation for a phase."""
    trends = {
        "constrained_backbone_context": (
            "local torsion perturbations did not move C peak positions meaningfully",
            "local torsion perturbations did not move D peak positions meaningfully",
            "local C-alpha anchored torsion basins were too local to move the larger C/D length scales",
        ),
        "global_deformation": (
            "C is axial-sensitive; radial/twist/anisotropic tests were mostly flat for C",
            "D is radial/inter-strand-distance sensitive and moved monotonically with radial scale",
            "global pilot separated C-like axial sensitivity from D-like radial sensitivity",
        ),
        "radial_axial_refinement": (
            "axial tuning mainly controls C but did not reach 5.6 A in this grid",
            "radial tuning mainly controls D; baseline radial scale preserved D best here",
            "focused grid confirmed axial/radial separation with some bin-threshold coupling",
        ),
        "axial_only_extension": (
            "stronger axial compression moved C toward 5.6 A",
            "D remained stable across 0.9700 to 1.0000 axial scale",
            "moderate axial compression improved combined C/D error without damaging D",
        ),
        "fine_axial_profile_diagnostic": (
            "C profile moves smoothly underneath discretized picked peaks",
            "D remains picked-position stable with very small local-profile shifts",
            "C response is real profile movement rather than random peak-picking noise",
        ),
        "rise_like_diagnostic": (
            "C continues toward 5.6 A under stronger rise-like compression",
            "D remains stable from 0.9700 to 1.0000 but drops at 0.9600/0.9650",
            "0.9700 is the best combined diagnostic compromise",
        ),
        "overall_best": (
            "best current diagnostic C improvement is at about 3% effective rise compression",
            "best current diagnostic D preservation remains near 7.2756 A",
            "moderate rise-like compression is the current best diagnostic result",
        ),
    }
    return trends.get(phase, ("", "", ""))


def summary_row_from_inputs(phase: str, branch: str, scores: pd.DataFrame | None, geometry: pd.DataFrame | None) -> dict[str, object]:
    """Construct one CSV summary row from score and geometry tables."""
    generated, interpretable = summarize_geometry_counts(geometry)
    c_trend, d_trend, interpretation = trend_strings_for_phase(phase)
    row: dict[str, object] = {
        "phase": phase,
        "branch": branch,
        "variants_generated": generated,
        "geometry_interpretable": interpretable,
        "variants_scored": "" if scores is None else len(scores),
        "best_variant": "",
        "best_C_peak_A": "",
        "best_D_peak_A": "",
        "best_C_error_A": "",
        "best_D_error_A": "",
        "best_combined_abs_error_A": "",
        "primary_C_trend": c_trend,
        "primary_D_trend": d_trend,
        "interpretation": interpretation,
    }
    if scores is not None and not scores.empty:
        best = best_score_row(scores)
        row.update(
            {
                "best_variant": best.get("variant_id", ""),
                "best_C_peak_A": safe_float(best.get("C_peak_A", "")),
                "best_D_peak_A": safe_float(best.get("D_peak_A", "")),
                "best_C_error_A": safe_float(best.get("C_error_A", "")),
                "best_D_error_A": safe_float(best.get("D_error_A", "")),
                "best_combined_abs_error_A": safe_float(best.get("combined_abs_error_A", "")),
            }
        )
    return row


def constrained_context_row() -> dict[str, object]:
    """Return summary row for the prior constrained-backbone context."""
    c_trend, d_trend, interpretation = trend_strings_for_phase("constrained_backbone_context")
    return {
        "phase": "constrained_backbone_context",
        "branch": "constrained local phi/psi torsion context",
        "variants_generated": "",
        "geometry_interpretable": "",
        "variants_scored": "",
        "best_variant": "",
        "best_C_peak_A": "",
        "best_D_peak_A": "",
        "best_C_error_A": "",
        "best_D_error_A": "",
        "best_combined_abs_error_A": "",
        "primary_C_trend": c_trend,
        "primary_D_trend": d_trend,
        "interpretation": interpretation,
    }


def overall_best_row(rows: Iterable[dict[str, object]]) -> dict[str, object]:
    """Return overall best row copied from the best phase row."""
    phase_priority = {
        "global_deformation": 0,
        "radial_axial_refinement": 1,
        "axial_only_extension": 2,
        "fine_axial_profile_diagnostic": 3,
        "rise_like_diagnostic": 4,
    }
    scored_rows = [
        row
        for row in rows
        if row["phase"] != "constrained_backbone_context" and safe_float(row["best_combined_abs_error_A"]) is not None
    ]
    scored_rows = [row for row in scored_rows if not math.isnan(float(row["best_combined_abs_error_A"]))]
    if not scored_rows:
        row = constrained_context_row()
        row["phase"] = "overall_best"
        row["branch"] = "overall best diagnostic"
        return row
    best = min(
        scored_rows,
        key=lambda row: (
            float(row["best_combined_abs_error_A"]),
            -phase_priority.get(str(row["phase"]), -1),
        ),
    )
    c_trend, d_trend, interpretation = trend_strings_for_phase("overall_best")
    return {
        **best,
        "phase": "overall_best",
        "branch": "overall best diagnostic",
        "primary_C_trend": c_trend,
        "primary_D_trend": d_trend,
        "interpretation": interpretation,
    }


def construct_summary_rows(root: Path = Path(".")) -> list[dict[str, object]]:
    """Build all summary rows from available output files."""
    rows = [constrained_context_row()]
    for phase, info in PHASE_INPUTS.items():
        scores = read_csv_if_present(root / info["scores"])
        geometry = read_csv_if_present(root / info["geometry"])
        rows.append(summary_row_from_inputs(phase, str(info["branch"]), scores, geometry))
    rows.append(overall_best_row(rows))
    return rows


def missing_inputs(root: Path = Path(".")) -> list[Path]:
    """Return expected input paths that are missing."""
    return [path for path in INPUT_PATHS if not (root / path).exists()]


def profile_shift_text(diag: pd.DataFrame | None, centroid_col: str, parabola_col: str) -> tuple[str, str]:
    """Return max absolute C/D centroid and parabolic shift text for a diagnostic table."""
    if diag is None or diag.empty:
        return "not available", "not available"
    pieces = []
    for band in ("C", "D"):
        subset = diag[diag["band"] == band]
        centroid = pd.to_numeric(subset.get(centroid_col, pd.Series(dtype=float)), errors="coerce").abs().max()
        parabola = pd.to_numeric(subset.get(parabola_col, pd.Series(dtype=float)), errors="coerce").abs().max()
        pieces.append(f"{band}: centroid {centroid:.6g} A, parabolic {parabola:.6g} A")
    return pieces[0], pieces[1]


def build_report_text(summary: pd.DataFrame, missing: list[Path], root: Path = Path(".")) -> str:
    """Build consolidated markdown report."""
    fine_diag = read_csv_if_present(root / "outputs/metrics/fine_axial_profile_cd_profile_diagnostics.csv")
    rise_diag = read_csv_if_present(root / "outputs/metrics/rise_like_cd_profile_diagnostics.csv")
    fine_c, fine_d = profile_shift_text(fine_diag, "centroid_shift_vs_0p9700_A", "parabolic_shift_vs_0p9700_A")
    rise_c, rise_d = profile_shift_text(rise_diag, "centroid_shift_vs_baseline_A", "parabolic_shift_vs_baseline_A")
    overall = summary[summary["phase"] == "overall_best"].iloc[0]
    missing_text = "\n".join(f"- `{path}`" for path in missing) if missing else "_None._"
    table_cols = [
        "phase",
        "variants_generated",
        "geometry_interpretable",
        "variants_scored",
        "best_variant",
        "best_C_peak_A",
        "best_D_peak_A",
        "best_combined_abs_error_A",
    ]
    return f"""# Global/Rise Diagnostic Phase Summary

## 1. Why This Phase Was Run

The constrained local torsion branch did not move C/D peak positions meaningfully. The next question was whether larger global geometric dimensions could move C and D in a controlled way. The goal here was diagnostic sensitivity, not finalized structural modeling.

## 2. Starting Point From Constrained-Backbone Phase

Local C-alpha anchored torsion variants were geometry-safe in narrow basins, but C/D peak positions were robust. CYP->GLU-only, GLU->MEP-only, and coupled CYP->GLU + GLU->MEP variants did not move C/D peak positions, and coupled profile diagnostics showed only tiny sub-bin movement. This motivated the global deformation tests.

## 3. Global Deformation Pilot

The global deformation pilot generated 12 variants and all 12 were geometry-interpretable. Radial mode moved D, axial mode moved C, and twist/anisotropic modes were flat. The best early global variant was `axial_m1`, with C near 5.6934 A, D near 7.2756 A, and combined error about 0.1178 A. Largest RMSD was about 0.1240 A and largest max displacement was about 0.1664 A.

## 4. Focused Radial/Axial Refinement

The focused grid generated 25 variants, with 25/25 geometry-interpretable and 25 scored. Axial tuning mainly controls C, radial tuning mainly controls D, and cross-coupling appears near peak-picking/bin thresholds. The best representative was `radial_1p0000__axial_0p9900`, with C near 5.6934 A, D near 7.2756 A, and combined error about 0.1178 A.

## 5. Axial-Only Extension

The axial-only branch compressed axial scale with radial scale fixed at 1.0000. The best variant was `axial_only_0p9700`: C 5.6422 A, D 7.2756 A, combined error 0.0667 A. D stayed stable across 0.9700 to 1.0000, while the picked C response was discretized by peak-picking/binning.

## 6. Fine Axial Profile Diagnostics

Fine axial diagnostics showed that the C profile moves smoothly underneath discretized picked peaks, while D remains picked-position stable. Max profile shifts versus 0.9700 were {fine_c}; {fine_d}. This supports real C-profile movement rather than random peak-picking noise.

## 7. Rise-Like Diagnostic Branch

The rise-like branch tested axial_rise_scale from 0.9600 to 1.0000. The best combined result was `rise_like_0p9700`: C 5.6422 A, D 7.2756 A, combined error 0.0667 A. C reaches about 5.5920 A at 0.9600/0.9650, but D drops to about 7.1923 A there. Thus 0.9700 is the best combined diagnostic compromise: it improves C while preserving D. Rise-like profile shifts versus baseline were {rise_c}; {rise_d}.

## 8. Overall Interpretation

C is mainly axial/rise-like sensitive. D is mainly radial/inter-strand-distance sensitive. Moderate rise-like compression improves C while preserving D. Stronger rise compression reaches the C target more closely but worsens D. Local torsion perturbations did not change the larger structural length scales enough to move C/D. The best diagnostic scale is around 0.9700, corresponding to about 3% effective rise compression.

## 9. What Not To Overclaim

- Treat these as diagnostic variants, not minimized physical structures.
- do not claim the final structure requires literal uniform 3% z-scaling.
- Do not claim backbone is irrelevant.
- Do not claim C/D sensitivity is fully solved.
- Do not treat loose global geometry gates as chemical validation.

## 10. Recommended Next Scientific Branches

- Option A: physically parameterized rise/rise-per-repeat model. Build or regenerate coordinates by changing helical rise/repeat spacing rather than globally scaling z.
- Option B: combined rise + radial compensation. Test whether mild radial adjustment can preserve D while stronger rise compression targets C.
- Option C: map diagnostic deformation back to backbone/hexad parameters. Ask what backbone/stack parameters produce an effective 3% rise compression.
- Option D: minimized/refined structural candidates. Use an external minimizer or physically constrained coordinate builder to relax the best diagnostic variants.
- Option E: prepare concise Nick/team update. Summarize the local torsion negative result plus the global/rise positive result.

## 11. Current Best Diagnostic Result

- Variant: `{overall['best_variant']}`
- C peak: {safe_float(overall['best_C_peak_A']):.4f} A
- D peak: {safe_float(overall['best_D_peak_A']):.4f} A
- Combined absolute error: {safe_float(overall['best_combined_abs_error_A']):.4f} A
- Diagnostic interpretation: moderate effective rise compression improves C while preserving D.

## Summary Table

{markdown_table(summary, table_cols)}

## Missing Optional Inputs

{missing_text}
"""


def run(out_csv: Path, report_path: Path, root: Path = Path(".")) -> pd.DataFrame:
    """Write global/rise phase summary CSV and markdown report."""
    rows = construct_summary_rows(root)
    summary = pd.DataFrame(rows)
    missing = missing_inputs(root)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)
    report_path.write_text(build_report_text(summary, missing, root), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run(args.out_csv, args.report)
    print(f"Summarized {len(summary)} global/rise diagnostic rows")
    print(f"CSV: {args.out_csv}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
