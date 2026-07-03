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
    Path("outputs/metrics/parent_axial_layer_audit.csv"),
    Path("outputs/reports/parent_axial_layer_audit.md"),
    Path("outputs/metrics/parameterized_rise_variant_manifest.csv"),
    Path("outputs/reports/parameterized_rise_variant_generation.md"),
    Path("outputs/metrics/parameterized_rise_variant_geometry_audit.csv"),
    Path("outputs/reports/parameterized_rise_variant_geometry_audit.md"),
    Path("outputs/metrics/parameterized_rise_variant_cd_scores.csv"),
    Path("outputs/reports/parameterized_rise_variant_cd_scores.md"),
    Path("outputs/metrics/parameterized_rise_cd_profile_diagnostics.csv"),
    Path("outputs/reports/parameterized_rise_cd_profile_diagnostics.md"),
    Path("outputs/metrics/register_defined_layer_model_summary.csv"),
    Path("outputs/metrics/register_to_zslice_layer_mapping.csv"),
    Path("outputs/reports/register_defined_layer_model_audit.md"),
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
    "parameterized_rise_diagnostic": {
        "scores": Path("outputs/metrics/parameterized_rise_variant_cd_scores.csv"),
        "geometry": Path("outputs/metrics/parameterized_rise_variant_geometry_audit.csv"),
        "branch": "layer-aware parameterized rise diagnostic",
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
        "parameterized_rise_diagnostic": (
            "C moves toward 5.6 A with layer/repeat-aware rise compression",
            "D remains stable from 0.9750 to 1.0000 but drops at 0.9600 through 0.9700",
            "0.9750 ties the generic rise_like best score while using a more interpretable layer-aware transform",
        ),
        "updated_overall_best": (
            "best current diagnostic C improvement is at about 2.5% effective layer-rise compression",
            "best current diagnostic D preservation remains near 7.2756 A",
            "moderate parameterized rise compression is the current best diagnostic result",
        ),
        "register_defined_layer_audit": (
            "not applicable",
            "not applicable",
            "z-slice layers are computationally regular but chemically non-unique; register-defined layers are chemically cleaner but geometrically diffuse",
        ),
        "final_current_interpretation": (
            "C is axial/rise-like sensitive, but the current layer model is computational rather than chemically definitive",
            "D is radial/inter-strand-distance sensitive and remains preserved in the best diagnostic result",
            "best current result is effective computational z-layer compression, pending chemically/register-defined physical model",
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
        "parameterized_rise_diagnostic": 5,
    }
    scored_rows = []
    for row in rows:
        if row["phase"] == "constrained_backbone_context":
            continue
        value = safe_float(row.get("best_combined_abs_error_A"))
        if value is None or math.isnan(float(value)):
            continue
        scored_rows.append(row)
    if not scored_rows:
        row = constrained_context_row()
        row["phase"] = "updated_overall_best"
        row["branch"] = "updated overall best diagnostic"
        return row
    best = min(
        scored_rows,
        key=lambda row: (
            float(row["best_combined_abs_error_A"]),
            -phase_priority.get(str(row["phase"]), -1),
        ),
    )
    c_trend, d_trend, interpretation = trend_strings_for_phase("updated_overall_best")
    return {
        **best,
        "phase": "updated_overall_best",
        "branch": "updated overall best diagnostic",
        "primary_C_trend": c_trend,
        "primary_D_trend": d_trend,
        "interpretation": interpretation,
    }


def register_defined_layer_audit_row(root: Path = Path(".")) -> dict[str, object]:
    """Return summary row for the register-defined layer model audit."""
    summary = read_csv_if_present(root / "outputs/metrics/register_defined_layer_model_summary.csv")
    c_trend, d_trend, interpretation = trend_strings_for_phase("register_defined_layer_audit")
    return {
        "phase": "register_defined_layer_audit",
        "branch": "register-defined layer model audit",
        "variants_generated": "" if summary is None else 0,
        "geometry_interpretable": "not applicable",
        "variants_scored": 0,
        "best_variant": "not applicable",
        "best_C_peak_A": "",
        "best_D_peak_A": "",
        "best_C_error_A": "",
        "best_D_error_A": "",
        "best_combined_abs_error_A": "",
        "primary_C_trend": c_trend,
        "primary_D_trend": d_trend,
        "interpretation": interpretation,
    }


def final_current_interpretation_row(rows: Iterable[dict[str, object]]) -> dict[str, object]:
    """Return final current interpretation row after the register-defined layer audit."""
    best = overall_best_row(rows)
    c_trend, d_trend, interpretation = trend_strings_for_phase("final_current_interpretation")
    return {
        **best,
        "phase": "final_current_interpretation",
        "branch": "final current interpretation after register-defined layer audit",
        "best_variant": "parameterized_rise_0p9750",
        "best_C_peak_A": 5.6422,
        "best_D_peak_A": 7.2756,
        "best_C_error_A": 0.0422,
        "best_D_error_A": -0.0244,
        "best_combined_abs_error_A": 0.0667,
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
    rows.append(register_defined_layer_audit_row(root))
    rows.append(final_current_interpretation_row(rows))
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


def register_audit_section(register_summary: pd.DataFrame | None, register_mapping: pd.DataFrame | None) -> str:
    """Return register-defined layer audit markdown section text."""
    if register_summary is None:
        return "Register-defined layer model audit outputs were not available."
    rows = {row["model_name"]: row for _, row in register_summary.iterrows()}

    def metric(model: str, column: str, default: str = "not available") -> object:
        row = rows.get(model)
        if row is None:
            return default
        value = row.get(column, default)
        if pd.isna(value):
            return default
        return value

    overlap = {}
    if register_mapping is not None and not register_mapping.empty:
        overlap = register_mapping.groupby("model_name")["zslice_layer_count"].mean().to_dict()

    def overlap_text(model: str) -> str:
        value = overlap.get(model)
        return "not available" if value is None else f"{float(value):.2f}"

    return f"""This audit was run to validate whether the 45 z-slice layers used by the parameterized rise branch are chemically/register meaningful. It compared five models: `z_slice_layer`, `residue_index_layer`, `ca_register_layer`, `repeat_pair_layer`, and `peptide_plane_layer`.

Main contrast: z-slice layers are geometrically regular but split most residues; register-defined layers avoid residue splitting and are chemically cleaner but geometrically diffuse in this antiparallel/offset parent structure.

- `z_slice_layer`: {int(metric('z_slice_layer', 'layer_count'))} layers, {int(metric('z_slice_layer', 'split_residue_count'))}/180 split residues, median spacing about {float(metric('z_slice_layer', 'median_layer_to_layer_delta_z_A')):.4f} A.
- `residue_index_layer`: {int(metric('residue_index_layer', 'layer_count'))} layers, {int(metric('residue_index_layer', 'split_residue_count'))} split residues, median thickness about {float(metric('residue_index_layer', 'median_layer_thickness_A')):.4f} A.
- `ca_register_layer`: {int(metric('ca_register_layer', 'layer_count'))} layers, {int(metric('ca_register_layer', 'split_residue_count'))} split residues, median thickness about {float(metric('ca_register_layer', 'median_layer_thickness_A')):.4f} A.
- `repeat_pair_layer`: {int(metric('repeat_pair_layer', 'layer_count'))} layers, {int(metric('repeat_pair_layer', 'split_residue_count'))} split residues, median thickness about {float(metric('repeat_pair_layer', 'median_layer_thickness_A')):.4f} A.
- `peptide_plane_layer`: {int(metric('peptide_plane_layer', 'layer_count'))} layers, {int(metric('peptide_plane_layer', 'split_residue_count'))} split residues, median thickness about {float(metric('peptide_plane_layer', 'median_layer_thickness_A')):.4f} A.

Mean z-slice overlap counts were: residue_index_layer {overlap_text('residue_index_layer')}, ca_register_layer {overlap_text('ca_register_layer')}, repeat_pair_layer {overlap_text('repeat_pair_layer')}, and peptide_plane_layer {overlap_text('peptide_plane_layer')}.

Conservative conclusion: the z-slice model is useful as a computational deformation coordinate; register-defined layers are chemically cleaner but geometrically diffuse. Do not call the 45 z-slices physical hexad layers yet."""


def build_report_text(summary: pd.DataFrame, missing: list[Path], root: Path = Path(".")) -> str:
    """Build consolidated markdown report."""
    fine_diag = read_csv_if_present(root / "outputs/metrics/fine_axial_profile_cd_profile_diagnostics.csv")
    rise_diag = read_csv_if_present(root / "outputs/metrics/rise_like_cd_profile_diagnostics.csv")
    parameterized_diag = read_csv_if_present(root / "outputs/metrics/parameterized_rise_cd_profile_diagnostics.csv")
    layer_audit = read_csv_if_present(root / "outputs/metrics/parent_axial_layer_audit.csv")
    register_summary = read_csv_if_present(root / "outputs/metrics/register_defined_layer_model_summary.csv")
    register_mapping = read_csv_if_present(root / "outputs/metrics/register_to_zslice_layer_mapping.csv")
    fine_c, fine_d = profile_shift_text(fine_diag, "centroid_shift_vs_0p9700_A", "parabolic_shift_vs_0p9700_A")
    rise_c, rise_d = profile_shift_text(rise_diag, "centroid_shift_vs_baseline_A", "parabolic_shift_vs_baseline_A")
    parameterized_c, parameterized_d = profile_shift_text(
        parameterized_diag, "centroid_shift_vs_baseline_A", "parabolic_shift_vs_baseline_A"
    )
    layer_count = len(layer_audit) if layer_audit is not None else "not available"
    mean_layer_rise = (
        pd.to_numeric(layer_audit["layer_center_z_A"], errors="coerce").diff().dropna().mean()
        if layer_audit is not None and "layer_center_z_A" in layer_audit.columns
        else float("nan")
    )
    overall = summary[summary["phase"] == "final_current_interpretation"].iloc[0]
    register_section = register_audit_section(register_summary, register_mapping)
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

## Parameterized rise diagnostic branch

This branch was run because the generic rise_like branch was useful but still used continuous global z-scaling. The parameterized branch inferred 45 axial layers from C-alpha z positions, estimated a mean parent layer rise of about {mean_layer_rise:.4f} A, preserved each atom's local offset from its assigned layer center, and then moved only the layer centers according to the rise scale.

The best parameterized result was `parameterized_rise_0p9750`: C 5.6422 A, D 7.2756 A, combined error 0.0667 A. It ties the generic `rise_like_0p9700` C/D score, but is more interpretable because the same improvement appears in a layer/repeat-aware model rather than only uniform z-scaling. C still moves toward 5.6 A with compression. D remains stable at 7.2756 A from 0.9750 through 1.0000, but drops to 7.1923 A at 0.9600, 0.9650, and 0.9700. Preserving within-layer z offsets changes the D threshold behavior slightly. Parameterized profile shifts versus baseline were {parameterized_c}; {parameterized_d}.

## Register-defined layer model audit

{register_section}

## 8. Overall Interpretation

Local C-alpha anchored torsion basin did not move C/D. Global/rise diagnostics did move C/D. D is mainly radial/inter-strand-distance sensitive. C is mainly axial/rise-like sensitive. The best current diagnostic result is `parameterized_rise_0p9750`: C 5.6422 A, D 7.2756 A, combined absolute error 0.0667 A. It corresponds to about 2.5% effective layer-rise compression in the diagnostic layer model. The register-defined audit shows the current layer model should be described as effective computational z-layer compression, pending mapping to a chemically/register-defined structural model; it is not chemically definitive. Stronger compression still improves C, but begins to damage D.

## 9. What Not To Overclaim

- Treat these as diagnostic variants, not minimized physical structures.
- do not claim the final structure requires literal uniform 3% z-scaling.
- Treat the layer-aware parameterized rise model as not fully physical or minimized.
- Do not claim the inferred 45 layers are uniquely defined structural layers without further validation.
- Do not claim the optimal scale is exact; peak-picking/binning discretization still matters.
- Do not claim the 45 z-slices are physical hexad layers.
- Do not claim `parameterized_rise_0p9750` is a validated physical rise-per-hexad model.
- Do not claim register-defined residue/peptide-plane layers reproduce the z-slice rise result.
- Do not claim the model is chemically minimized.
- Do not claim backbone is irrelevant.
- Do not claim C/D sensitivity is fully solved.
- Do not treat loose global geometry gates as chemical validation.

## 10. Recommended Next Scientific Branches

- Option A: chemically annotated register model. Use known/validated hexad or repeat-register annotations rather than purely z-derived layers.
- Option B: physically rebuilt helical/rise model. Generate coordinates from helical/rise parameters instead of transforming the parent coordinates.
- Option C: repeat-pair or peptide-plane constrained rise model. Even though diffuse, these are chemically cleaner and may be useful if paired with additional structural constraints.
- Option D: rise + radial compensation. Only after a more physical rise model is defined, test whether radial compensation can preserve D while targeting C.
- Option E: external minimization/refinement. Relax diagnostic candidates under structural restraints to test whether the effective rise compression is chemically feasible.

## 11. Current Best Diagnostic Result

- Variant: `{overall['best_variant']}`
- C peak: {safe_float(overall['best_C_peak_A']):.4f} A
- D peak: {safe_float(overall['best_D_peak_A']):.4f} A
- Combined absolute error: {safe_float(overall['best_combined_abs_error_A']):.4f} A
- Diagnostic interpretation: `parameterized_rise_0p9750` is the best current diagnostic result, but after the register-defined layer audit it should be described as effective computational z-layer compression, pending mapping to a chemically/register-defined structural model.

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
