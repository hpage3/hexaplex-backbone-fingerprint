"""Audit parameterized rise variants with global and layer sanity gates."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_constrained_phi_psi_candidate_geometry import Atom, parse_pdb
from scripts.audit_global_deformation_variant_geometry import (
    MAX_BACKBONE_ANGLE_DELTA_DEG,
    MAX_BACKBONE_BOND_DELTA_A,
    backbone_delta_metrics,
    displacement_metrics,
    identity_matches,
    mean_ca_radius,
    z_span,
)
from scripts.audit_parent_axial_layers import assign_layer_index, infer_layers_from_ca_z, mean_layer_rise
from scripts.score_radial_axial_refinement_variant_cd import markdown_table


DEFAULT_MANIFEST = Path("outputs/metrics/parameterized_rise_variant_manifest.csv")
DEFAULT_SOURCE_PDB = Path("outputs/coordinates/ideal_hexaflex_variants/ideal_hexaflex_backbone_plus_carboxylate.pdb")
DEFAULT_OUT_CSV = Path("outputs/metrics/parameterized_rise_variant_geometry_audit.csv")
DEFAULT_REPORT = Path("outputs/reports/parameterized_rise_variant_geometry_audit.md")


def parent_layer_model(parent_atoms: list[Atom]) -> list[float]:
    """Infer parent layer centers from C-alpha atoms."""
    ca_z = [atom.z for atom in parent_atoms if atom.name == "CA"]
    return infer_layers_from_ca_z(ca_z).layer_centers


def assigned_layer_centers(parent_atoms: list[Atom], variant_atoms: list[Atom], parent_centers: list[float]) -> list[float]:
    """Return variant mean z per parent-assigned layer."""
    centers = []
    for layer_idx in range(len(parent_centers)):
        z_values = [
            variant.z
            for parent, variant in zip(parent_atoms, variant_atoms)
            if assign_layer_index(parent.z, parent_centers) == layer_idx
        ]
        centers.append(float(np.mean(z_values)) if z_values else float("nan"))
    return centers


def layer_order_preserved(layer_centers: list[float]) -> bool:
    """Return whether layer centers remain strictly ordered."""
    clean = [center for center in layer_centers if not math.isnan(center)]
    return all(b > a for a, b in zip(clean, clean[1:]))


def inter_layer_gaps(layer_centers: list[float]) -> list[float]:
    """Return adjacent layer gaps."""
    clean = [center for center in layer_centers if not math.isnan(center)]
    return [b - a for a, b in zip(clean, clean[1:])]


def geometry_interpretable(row: dict[str, object]) -> bool:
    """Classify parameterized rise geometry with diagnostic gates."""
    return (
        bool(row.get("atom_count_matches_parent"))
        and bool(row.get("identity_matches_parent"))
        and bool(row.get("layer_order_preserved"))
        and float(row.get("max_backbone_bond_delta_A", math.inf)) <= MAX_BACKBONE_BOND_DELTA_A
        and float(row.get("max_backbone_angle_delta_deg", math.inf)) <= MAX_BACKBONE_ANGLE_DELTA_DEG
    )


def failed_checks(row: dict[str, object]) -> list[str]:
    """Return stable failure reasons."""
    reasons = []
    if not row.get("atom_count_matches_parent"):
        reasons.append("atom_count_mismatch")
    if not row.get("identity_matches_parent"):
        reasons.append("identity_mismatch")
    if not row.get("layer_order_preserved"):
        reasons.append("layer_order_not_preserved")
    if float(row.get("max_backbone_bond_delta_A", math.inf)) > MAX_BACKBONE_BOND_DELTA_A:
        reasons.append("backbone_bond_delta_exceeds_global_threshold")
    if float(row.get("max_backbone_angle_delta_deg", math.inf)) > MAX_BACKBONE_ANGLE_DELTA_DEG:
        reasons.append("backbone_angle_delta_exceeds_global_threshold")
    return reasons


def audit_variant(parent_atoms: list[Atom], parent_centers: list[float], manifest_row: pd.Series) -> dict[str, object]:
    """Audit one parameterized rise variant."""
    output_pdb = Path(str(manifest_row["output_pdb"]))
    variant_atoms = parse_pdb(output_pdb) if output_pdb.exists() else []
    atom_count_matches = len(parent_atoms) == len(variant_atoms)
    identity_ok = identity_matches(parent_atoms, variant_atoms) if variant_atoms else False
    disp = displacement_metrics(parent_atoms, variant_atoms) if variant_atoms else {
        "max_displacement_A": math.inf,
        "rmsd_all_atoms_A": math.inf,
        "rmsd_ca_A": math.inf,
    }
    parent_radius = mean_ca_radius(parent_atoms)
    variant_radius = mean_ca_radius(variant_atoms) if variant_atoms else math.inf
    parent_span = z_span(parent_atoms)
    variant_span = z_span(variant_atoms) if variant_atoms else math.inf
    backbone = backbone_delta_metrics(parent_atoms, variant_atoms) if variant_atoms else {
        "max_backbone_bond_delta_A": math.inf,
        "max_backbone_angle_delta_deg": math.inf,
    }
    variant_centers = assigned_layer_centers(parent_atoms, variant_atoms, parent_centers) if variant_atoms else []
    gaps = inter_layer_gaps(variant_centers)
    row = {
        "variant_id": manifest_row["variant_id"],
        "rise_scale": manifest_row["rise_scale"],
        "estimated_percent_rise_compression": manifest_row.get("estimated_percent_rise_compression", ""),
        "layer_model": manifest_row.get("layer_model", ""),
        "atom_count_matches_parent": atom_count_matches,
        "identity_matches_parent": identity_ok,
        **disp,
        "parent_z_span_A": parent_span,
        "variant_z_span_A": variant_span,
        "z_span_delta_A": variant_span - parent_span,
        "parent_mean_ca_radius_A": parent_radius,
        "variant_mean_ca_radius_A": variant_radius,
        "mean_ca_radius_delta_A": variant_radius - parent_radius,
        **backbone,
        "parent_mean_layer_rise_A": mean_layer_rise(parent_centers),
        "variant_mean_layer_rise_A": mean_layer_rise(variant_centers) if variant_centers else math.inf,
        "mean_layer_rise_delta_A": (mean_layer_rise(variant_centers) - mean_layer_rise(parent_centers)) if variant_centers else math.inf,
        "layer_order_preserved": layer_order_preserved(variant_centers) if variant_centers else False,
        "min_inter_layer_gap_A": min(gaps) if gaps else math.inf,
        "max_inter_layer_gap_A": max(gaps) if gaps else math.inf,
        "output_pdb": manifest_row["output_pdb"],
    }
    row["geometry_interpretable"] = geometry_interpretable(row)
    row["failed_checks"] = ";".join(failed_checks(row))
    return row


def run_audit(manifest_path: Path, source_pdb: Path, out_csv: Path, report_path: Path) -> pd.DataFrame:
    """Run parameterized rise geometry audit and write outputs."""
    manifest = pd.read_csv(manifest_path)
    parent_atoms = parse_pdb(source_pdb)
    parent_centers = parent_layer_model(parent_atoms)
    results = pd.DataFrame([audit_variant(parent_atoms, parent_centers, row) for _, row in manifest.iterrows()])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_csv, index=False)
    report_path.write_text(build_report_text(results, source_pdb), encoding="utf-8")
    return results


def build_report_text(results: pd.DataFrame, source_pdb: Path) -> str:
    """Build geometry audit report."""
    total = len(results)
    safe = results[results["geometry_interpretable"].astype(bool)] if total else results
    failed = results[~results["geometry_interpretable"].astype(bool)] if total else results
    cols = [
        "variant_id",
        "rise_scale",
        "geometry_interpretable",
        "max_displacement_A",
        "rmsd_all_atoms_A",
        "max_backbone_bond_delta_A",
        "max_backbone_angle_delta_deg",
        "variant_mean_layer_rise_A",
        "min_inter_layer_gap_A",
        "max_inter_layer_gap_A",
        "failed_checks",
    ]
    return f"""# Parameterized Rise Variant Geometry Audit

## Purpose

This audit checks layer/repeat-aware rise variants before C/D scoring.

- Source PDB: `{source_pdb}`
- Variants audited: {total}
- Geometry-interpretable variants: {len(safe)}/{total}

## Gates

- Atom count must match parent.
- Chain/residue/atom identity must match parent.
- Layer order must be preserved.
- Max backbone bond-length delta <= {MAX_BACKBONE_BOND_DELTA_A:g} A.
- Max backbone angle delta <= {MAX_BACKBONE_ANGLE_DELTA_DEG:g} degrees.

These are diagnostic gates, not chemical validation.

## Layer/Rise Sanity Summary

{markdown_table(results, cols)}

## Failed Variants

{markdown_table(failed, ['variant_id', 'rise_scale', 'failed_checks'])}

## Caution

These are parameterized diagnostic structures, not minimized physical structures.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-pdb", type=Path, default=DEFAULT_SOURCE_PDB)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = run_audit(args.manifest, args.source_pdb, args.out_csv, args.report)
    safe_count = int(results["geometry_interpretable"].astype(bool).sum()) if not results.empty else 0
    print(f"Audited {len(results)} parameterized rise variants")
    print(f"Geometry-interpretable variants: {safe_count}/{len(results)}")
    print(f"CSV: {args.out_csv}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
