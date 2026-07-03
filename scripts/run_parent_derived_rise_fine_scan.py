"""Run a fine parent-derived axial/rise scan.

This scan reuses the validated parent-derived layer-center transform from
``run_parent_derived_rise_bridge.py``. It preserves parent atom content,
residue identity, chain/register labels, carboxylates, and x/y coordinates.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_parent_axial_layers import infer_layers_from_ca_z, mean_layer_rise
from scripts.generate_global_deformation_variants import parse_pdb_atom_lines
from scripts.run_parent_derived_rise_bridge import (
    DEFAULT_PARENT_PDB,
    EXPECTED_PARENT_C_A,
    EXPECTED_PARENT_D_A,
    REFERENCE_TOLERANCE_A,
    TARGETS_A,
    ParentDerivedRiseSpec,
    geometry_summary_row,
    markdown_table,
    reference_reproduces_parent,
    required_score_columns as _bridge_required_score_columns,
    score_pdb_abcd,
    score_row,
    write_parent_derived_variant,
)


DEFAULT_OUTDIR = Path("outputs/coordinates/parent_derived_rise_fine_scan")
DEFAULT_SCORE_CSV = Path("outputs/metrics/parent_derived_rise_fine_scan_abcd_scores.csv")
DEFAULT_GEOMETRY_CSV = Path("outputs/metrics/parent_derived_rise_fine_scan_geometry_summary.csv")
DEFAULT_REPORT = Path("outputs/reports/parent_derived_rise_fine_scan_summary.md")

SCALE_VALUES = [1.0000, 0.9950, 0.9900, 0.9850, 0.9825, 0.9800, 0.9775, 0.9750, 0.9725, 0.9700]
NOMINAL_REFERENCE_RISE_A = 3.40
DIAGNOSTIC_BEST_C_A = 5.6422
DIAGNOSTIC_BEST_D_A = 7.2756


@dataclass(frozen=True)
class FineScanSpec:
    """One fine-scan axial scale variant."""

    variant_id: str
    axial_scale: float


def format_scale(scale: float) -> str:
    """Return stable filename text for a scale."""
    return f"{scale:.4f}".replace(".", "p").replace("-", "m")


def variant_id_for_scale(scale: float) -> str:
    """Return stable fine-scan variant ID."""
    return f"parent_derived_scale_{format_scale(scale)}"


def fine_scan_specs(scales: list[float] | None = None) -> list[FineScanSpec]:
    """Return exact requested scale set."""
    values = SCALE_VALUES if scales is None else scales
    return [FineScanSpec(variant_id_for_scale(scale), float(scale)) for scale in values]


def nominal_rise_equiv(scale: float, reference_rise_A: float = NOMINAL_REFERENCE_RISE_A) -> float:
    """Return nominal rise equivalent for an axial scale."""
    return float(scale) * float(reference_rise_A)


def output_path(outdir: Path, spec: FineScanSpec) -> Path:
    """Return PDB output path for one fine-scan variant."""
    return outdir / f"{spec.variant_id}.pdb"


def required_score_columns() -> list[str]:
    """Return required fine scan score columns."""
    base = _bridge_required_score_columns()
    columns = ["variant_id", "axial_scale", "nominal_rise_equiv_A"]
    columns.extend(column for column in base if column not in {"variant_id", "requested_nominal_rise_A", "axial_scale"})
    return columns


def fine_scan_score_row(
    spec: FineScanSpec,
    path: Path,
    parent_reference_rise_metric_A: float,
    reference_ok: bool,
) -> dict[str, object]:
    """Build one fine-scan score row."""
    bridge_spec = ParentDerivedRiseSpec(spec.variant_id, nominal_rise_equiv(spec.axial_scale), spec.axial_scale)
    row = score_row(bridge_spec, path, parent_reference_rise_metric_A, reference_ok)
    row["nominal_rise_equiv_A"] = nominal_rise_equiv(spec.axial_scale)
    row.pop("requested_nominal_rise_A", None)
    if not reference_ok:
        row["status"] = "blocked_reference_not_reproduced"
    return row


def fine_scan_recommendation(scores: pd.DataFrame) -> str:
    """Classify fine-scan result."""
    reference = scores[scores["variant_id"] == "parent_derived_scale_1p0000"]
    if reference.empty or not bool(reference.iloc[0]["reference_reproduces_parent"]):
        return "fine_scan_blocked_reference_not_reproduced"
    candidates = scores.copy()
    best = candidates.loc[pd.to_numeric(candidates["combined_CD_abs_error_A"], errors="coerce").idxmin()]
    c_close = abs(float(best["observed_C_d_A"]) - DIAGNOSTIC_BEST_C_A) <= 0.03
    d_close = abs(float(best["observed_D_d_A"]) - DIAGNOSTIC_BEST_D_A) <= 0.03
    if c_close and d_close:
        return "fine_scan_success"
    reference_error = float(reference.iloc[0]["combined_CD_abs_error_A"])
    if float(best["combined_CD_abs_error_A"]) < reference_error and abs(float(best["observed_D_d_A"]) - DIAGNOSTIC_BEST_D_A) <= 0.05:
        return "fine_scan_partial"
    return "fine_scan_failure"


def best_score_rows(scores: pd.DataFrame, tolerance: float = 1e-9) -> pd.DataFrame:
    """Return all rows tied for lowest combined C/D error, preserving scan order."""
    values = pd.to_numeric(scores["combined_CD_abs_error_A"], errors="coerce")
    best_value = values.min()
    return scores[values.sub(best_value).abs() <= tolerance].copy()


def best_score_row(scores: pd.DataFrame) -> pd.Series:
    """Return the first scan-order row among tied lowest combined C/D error rows."""
    return best_score_rows(scores).iloc[0]


def plateau_text(rows: pd.DataFrame) -> str:
    """Return compact text for a single-row or multi-row plateau."""
    if rows.empty:
        return "not observed"
    if len(rows) == 1:
        return str(rows.iloc[0]["variant_id"])
    return f"{rows.iloc[0]['variant_id']} through {rows.iloc[-1]['variant_id']}"


def build_report_text(scores: pd.DataFrame, geometry: pd.DataFrame, parent_pdb: Path, layer_count: int, parent_reference_rise_metric_A: float) -> str:
    """Build fine scan report."""
    recommendation = fine_scan_recommendation(scores)
    reference = scores[scores["variant_id"] == "parent_derived_scale_1p0000"].iloc[0]
    best_rows = best_score_rows(scores)
    best = best_score_row(scores)
    plateau = scores[
        (pd.to_numeric(scores["observed_C_d_A"], errors="coerce").sub(DIAGNOSTIC_BEST_C_A).abs() <= 0.001)
        & (pd.to_numeric(scores["observed_D_d_A"], errors="coerce").sub(DIAGNOSTIC_BEST_D_A).abs() <= 0.001)
    ]
    diagnostic_plateau_text = plateau_text(plateau)
    best_plateau_text = plateau_text(best_rows)
    overshoot = scores[scores["variant_id"] == "parent_derived_scale_0p9700"]
    overshoot_text = (
        f"0.9700 keeps C at {float(overshoot.iloc[0]['observed_C_d_A']):.4f} A but shifts D to "
        f"{float(overshoot.iloc[0]['observed_D_d_A']):.4f} A."
        if not overshoot.empty
        else "0.9700 was not included."
    )
    score_table = markdown_table(
        scores,
        [
            "variant_id",
            "axial_scale",
            "nominal_rise_equiv_A",
            "realized_rise_metric_A",
            "observed_C_d_A",
            "observed_D_d_A",
            "combined_CD_abs_error_A",
            "status",
        ],
    )
    geometry_table = markdown_table(
        geometry,
        [
            "variant_id",
            "z_span_A",
            "mean_ca_radius_A",
            "median_interstrand_nn_ca_distance_A",
            "median_ca_rise_A",
            "atom_count",
            "carboxylate_present",
        ],
    )
    return f"""# Parent-Derived Rise Fine Scan Summary

## Purpose

This fine scan tests whether a parent-derived axial/rise-like scale near `0.9750` can recover the prior diagnostic `parameterized_rise_0p9750` C/D behavior while preserving D.

## Parent And Transform

- Parent PDB: `{parent_pdb}`
- Transform: same parent-derived C-alpha z-gap layer-center transform used by `run_parent_derived_rise_bridge.py`
- Layer count: {layer_count}
- Parent reference rise metric: {parent_reference_rise_metric_A:.4f} A
- Nominal mapping: `nominal_rise_equiv_A = axial_scale * 3.40`
- Preserved: atom count, residue identity, chain IDs, residue numbering, atom names, carboxylates, and x/y coordinates.

## Reference Reproduction

- 1.0000 reference C/D: {float(reference['observed_C_d_A']):.4f} / {float(reference['observed_D_d_A']):.4f} A
- Expected parent C/D: about {EXPECTED_PARENT_C_A:.4f} / {EXPECTED_PARENT_D_A:.4f} A
- Reference reproduces parent: `{bool(reference['reference_reproduces_parent'])}`

## Result Summary

- The parent-derived fine scan reproduces parent C/D at scale 1.0000.
- The best combined-error plateau is `{best_plateau_text}`.
- The diagnostic C/D-matching plateau is `{diagnostic_plateau_text}`.
- The best C/D result matches the diagnostic `parameterized_rise_0p9750` result within the current peak-picking resolution.
- {overshoot_text}
- This is a success for the parent-derived fine-scan question, but it is not proof of exact original pNAB/YAML provenance.

## Model Scope / Asem Symmetry Caution

- This fine-scan success is limited to the existing six-fold-symmetric parent-derived coordinate family.
- It should not be interpreted as pNAB determining the physical twist/rise geometry.
- The pNAB-derived construction imposed a six-fold backbone-symmetry assumption because pNAB could not build two independent backbone types for two different strand classes.
- Asem flagged that the melamine/triamino and cyanuric/triketo backbone exit vectors may not be chemically equivalent. In the Builder construction, the melamine/triamino monomer required manual adjustment of an external torsion so cyanuric acid and melamine had the same exit vector; that torsion appears strained/cis-like.
- Combined with the peptide-planarity issue, this means the Builder-derived parent should not be emphasized as determining possible twist angles.
- A better next modeling hypothesis is separate three-fold backbone symmetry for the melamine/triamino and cyanuric/triketo strand classes.
- This motivates a new peptide-plane model track, not another one-dimensional parent-derived rise scan.

## Fine Scan Scores

{score_table}

## Geometry Summary

{geometry_table}

## Best Plateau

- Best combined-error plateau: `{best_plateau_text}`
- Representative tie-break member: `{best['variant_id']}`
- C/D: {float(best['observed_C_d_A']):.4f} / {float(best['observed_D_d_A']):.4f} A
- Combined C/D error: {float(best['combined_CD_abs_error_A']):.4f} A
- Tie-breaking note: when one row is required programmatically, the first plateau member in scan order is used. This does not imply a unique optimum.

## Comparison

- Parent-derived bridge 3.38/3.35 result: C about 5.6934 A, D about 7.2756 A.
- Prior diagnostic `parameterized_rise_0p9750`: C 5.6422 A, D 7.2756 A, combined C/D error 0.0667 A.

## Recommendation

`{recommendation}`

Preserve the distinction between diagnostic coordinate transforms, the failed pseudo reconstructed bridge, the validated parent-derived bridge, and this fine parent-derived refinement scan. These are still controlled coordinate transforms, not minimized physical structures, and they do not recover original model-generation provenance by themselves.
"""


def run_scan(
    parent_pdb: Path = DEFAULT_PARENT_PDB,
    outdir: Path = DEFAULT_OUTDIR,
    score_csv: Path = DEFAULT_SCORE_CSV,
    geometry_csv: Path = DEFAULT_GEOMETRY_CSV,
    report_path: Path = DEFAULT_REPORT,
    scales: list[float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate, score, and report fine parent-derived rise scan."""
    source_lines, atoms = parse_pdb_atom_lines(parent_pdb)
    layer_model = infer_layers_from_ca_z([atom.z for atom in atoms if atom.is_ca])
    global_center_z = float(np.mean(layer_model.layer_centers))
    parent_reference_rise_metric_A = mean_layer_rise(layer_model.layer_centers)
    specs = fine_scan_specs(scales)

    outdir.mkdir(parents=True, exist_ok=True)
    score_csv.parent.mkdir(parents=True, exist_ok=True)
    geometry_csv.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    for spec in specs:
        bridge_spec = ParentDerivedRiseSpec(spec.variant_id, nominal_rise_equiv(spec.axial_scale), spec.axial_scale)
        path = output_path(outdir, spec)
        write_parent_derived_variant(source_lines, atoms, bridge_spec, layer_model, global_center_z, path)
        paths[spec.variant_id] = path

    reference_scores = score_pdb_abcd(paths["parent_derived_scale_1p0000"])
    reference_ok = reference_reproduces_parent(reference_scores)
    score_rows = [fine_scan_score_row(spec, paths[spec.variant_id], parent_reference_rise_metric_A, reference_ok) for spec in specs]
    geometry_rows = [
        geometry_summary_row(ParentDerivedRiseSpec(spec.variant_id, nominal_rise_equiv(spec.axial_scale), spec.axial_scale), paths[spec.variant_id])
        for spec in specs
    ]
    scores = pd.DataFrame(score_rows).reindex(columns=required_score_columns())
    geometry = pd.DataFrame(geometry_rows)
    scores.to_csv(score_csv, index=False)
    geometry.to_csv(geometry_csv, index=False)
    report_path.write_text(build_report_text(scores, geometry, parent_pdb, len(layer_model.layer_centers), parent_reference_rise_metric_A), encoding="utf-8")
    return scores, geometry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-pdb", type=Path, default=DEFAULT_PARENT_PDB)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--score-csv", type=Path, default=DEFAULT_SCORE_CSV)
    parser.add_argument("--geometry-csv", type=Path, default=DEFAULT_GEOMETRY_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scores, _geometry = run_scan(args.parent_pdb, args.outdir, args.score_csv, args.geometry_csv, args.report)
    reference = scores[scores["variant_id"] == "parent_derived_scale_1p0000"].iloc[0]
    best_rows = best_score_rows(scores)
    best = best_score_row(scores)
    print(f"Generated and scored {len(scores)} parent-derived fine-scan variants")
    print(f"Reference reproduces parent: {bool(reference['reference_reproduces_parent'])}")
    print(f"Best plateau: {plateau_text(best_rows)} C={float(best['observed_C_d_A']):.4f} D={float(best['observed_D_d_A']):.4f}")
    print(f"Recommendation: {fine_scan_recommendation(scores)}")
    print(f"CSV: {args.score_csv}")
    print(f"Geometry CSV: {args.geometry_csv}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
