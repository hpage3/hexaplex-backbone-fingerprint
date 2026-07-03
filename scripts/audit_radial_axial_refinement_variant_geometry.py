"""Audit focused radial/axial refinement variants with global sanity gates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_constrained_phi_psi_candidate_geometry import parse_pdb
from scripts.audit_global_deformation_variant_geometry import (
    backbone_delta_metrics,
    displacement_metrics,
    failed_checks,
    geometry_interpretable,
    identity_matches,
    mean_ca_radius,
    z_span,
    MAX_BACKBONE_ANGLE_DELTA_DEG,
    MAX_BACKBONE_BOND_DELTA_A,
)


DEFAULT_MANIFEST = Path("outputs/metrics/radial_axial_refinement_variant_manifest.csv")
DEFAULT_SOURCE_PDB = Path("outputs/coordinates/ideal_hexaflex_variants/ideal_hexaflex_backbone_plus_carboxylate.pdb")
DEFAULT_OUT_CSV = Path("outputs/metrics/radial_axial_refinement_variant_geometry_audit.csv")
DEFAULT_REPORT = Path("outputs/reports/radial_axial_refinement_variant_geometry_audit.md")


def audit_variant(parent_atoms, manifest_row: pd.Series) -> dict[str, object]:
    """Audit one radial/axial refinement variant."""
    output_pdb = Path(str(manifest_row["output_pdb"]))
    variant_atoms = parse_pdb(output_pdb) if output_pdb.exists() else []
    atom_count_matches = len(parent_atoms) == len(variant_atoms)
    identity_ok = identity_matches(parent_atoms, variant_atoms) if variant_atoms else False
    disp = displacement_metrics(parent_atoms, variant_atoms) if variant_atoms else {
        "max_displacement_A": float("inf"),
        "rmsd_all_atoms_A": float("inf"),
        "rmsd_ca_A": float("inf"),
    }
    parent_radius = mean_ca_radius(parent_atoms)
    variant_radius = mean_ca_radius(variant_atoms) if variant_atoms else float("inf")
    parent_span = z_span(parent_atoms)
    variant_span = z_span(variant_atoms) if variant_atoms else float("inf")
    backbone = backbone_delta_metrics(parent_atoms, variant_atoms) if variant_atoms else {
        "max_backbone_bond_delta_A": float("inf"),
        "max_backbone_angle_delta_deg": float("inf"),
    }
    row = {
        "variant_id": manifest_row["variant_id"],
        "radial_scale_xy": manifest_row["radial_scale_xy"],
        "axial_scale_z": manifest_row["axial_scale_z"],
        "atom_count_matches_parent": atom_count_matches,
        "identity_matches_parent": identity_ok,
        **disp,
        "parent_mean_ca_radius_A": parent_radius,
        "variant_mean_ca_radius_A": variant_radius,
        "mean_ca_radius_delta_A": variant_radius - parent_radius,
        "parent_z_span_A": parent_span,
        "variant_z_span_A": variant_span,
        "z_span_delta_A": variant_span - parent_span,
        **backbone,
        "output_pdb": manifest_row["output_pdb"],
    }
    row["geometry_interpretable"] = geometry_interpretable(row)
    row["failed_checks"] = ";".join(failed_checks(row))
    return row


def run_audit(manifest_path: Path, source_pdb: Path, out_csv: Path, report_path: Path) -> pd.DataFrame:
    """Run radial/axial geometry audit and write outputs."""
    manifest = pd.read_csv(manifest_path)
    parent_atoms = parse_pdb(source_pdb)
    results = pd.DataFrame([audit_variant(parent_atoms, row) for _, row in manifest.iterrows()])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_csv, index=False)
    report_path.write_text(build_report_text(results, source_pdb), encoding="utf-8")
    return results


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


def build_report_text(results: pd.DataFrame, source_pdb: Path) -> str:
    """Build radial/axial geometry audit report."""
    total = len(results)
    safe = results[results["geometry_interpretable"].astype(bool)] if total else results
    failed = results[~results["geometry_interpretable"].astype(bool)] if total else results
    max_rmsd = pd.to_numeric(results.get("rmsd_all_atoms_A", pd.Series(dtype=float)), errors="coerce").max()
    max_disp = pd.to_numeric(results.get("max_displacement_A", pd.Series(dtype=float)), errors="coerce").max()
    summary_cols = [
        "variant_id",
        "radial_scale_xy",
        "axial_scale_z",
        "geometry_interpretable",
        "max_displacement_A",
        "rmsd_all_atoms_A",
        "mean_ca_radius_delta_A",
        "z_span_delta_A",
        "max_backbone_bond_delta_A",
        "max_backbone_angle_delta_deg",
        "failed_checks",
    ]
    return f"""# Radial/Axial Refinement Variant Geometry Audit

This audit uses the same global-deformation sanity gates as the previous global deformation pilot. These gates are diagnostic checks, not energy minimization.

- Source PDB: `{source_pdb}`
- Variants audited: {total}
- Geometry-interpretable variants: {len(safe)}/{total}
- Largest RMSD: {max_rmsd:.6g} A
- Largest max displacement: {max_disp:.6g} A

## Gates

- Atom count must match parent.
- Chain/residue/atom identity must match parent.
- Max backbone bond-length delta <= {MAX_BACKBONE_BOND_DELTA_A:g} A.
- Max backbone angle delta <= {MAX_BACKBONE_ANGLE_DELTA_DEG:g} degrees.

## Radius / Z-Span / Geometry Summary

{markdown_table(results, summary_cols)}

## Failed Variants

{markdown_table(failed, ['variant_id', 'radial_scale_xy', 'axial_scale_z', 'failed_checks'])}

## Caution

These are global diagnostic sanity gates for controlled perturbations, not energy minimization. Geometry-interpretable variants may be scored in a later C/D diagnostic pass.
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
    print(f"Audited {len(results)} radial/axial refinement variants")
    print(f"Geometry-interpretable variants: {safe_count}/{len(results)}")
    print(f"CSV: {args.out_csv}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
