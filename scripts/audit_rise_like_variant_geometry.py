"""Audit rise-like variants with global diagnostic geometry gates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_axial_only_extension_variant_geometry import audit_variant
from scripts.audit_constrained_phi_psi_candidate_geometry import parse_pdb
from scripts.audit_global_deformation_variant_geometry import MAX_BACKBONE_ANGLE_DELTA_DEG, MAX_BACKBONE_BOND_DELTA_A
from scripts.audit_radial_axial_refinement_variant_geometry import markdown_table


DEFAULT_MANIFEST = Path("outputs/metrics/rise_like_variant_manifest.csv")
DEFAULT_SOURCE_PDB = Path("outputs/coordinates/ideal_hexaflex_variants/ideal_hexaflex_backbone_plus_carboxylate.pdb")
DEFAULT_OUT_CSV = Path("outputs/metrics/rise_like_variant_geometry_audit.csv")
DEFAULT_REPORT = Path("outputs/reports/rise_like_variant_geometry_audit.md")


def run_audit(manifest_path: Path, source_pdb: Path, out_csv: Path, report_path: Path) -> pd.DataFrame:
    """Run rise-like geometry audit and write outputs."""
    manifest = pd.read_csv(manifest_path).rename(columns={"axial_rise_scale": "axial_scale_z"})
    manifest["radial_scale_xy"] = 1.0
    parent_atoms = parse_pdb(source_pdb)
    results = pd.DataFrame([audit_variant(parent_atoms, row) for _, row in manifest.iterrows()])
    results = results.rename(columns={"axial_scale_z": "axial_rise_scale"})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_csv, index=False)
    report_path.write_text(build_report_text(results, source_pdb), encoding="utf-8")
    return results


def build_report_text(results: pd.DataFrame, source_pdb: Path) -> str:
    """Build rise-like audit report."""
    total = len(results)
    safe = results[results["geometry_interpretable"].astype(bool)] if total else results
    failed = results[~results["geometry_interpretable"].astype(bool)] if total else results
    summary_cols = [
        "variant_id",
        "axial_rise_scale",
        "geometry_interpretable",
        "max_displacement_A",
        "rmsd_all_atoms_A",
        "z_span_delta_A",
        "max_backbone_bond_delta_A",
        "max_backbone_angle_delta_deg",
        "failed_checks",
    ]
    return f"""# Rise-Like Variant Geometry Audit

This audit checks diagnostic rise-like variants. These are controlled perturbations, not minimized structures.

- Source PDB: `{source_pdb}`
- Variants audited: {total}
- Geometry-interpretable variants: {len(safe)}/{total}

## Global Diagnostic Gates

- Atom count must match parent.
- Chain/residue/atom identity must match parent.
- Max backbone bond-length delta <= {MAX_BACKBONE_BOND_DELTA_A:g} A.
- Max backbone angle delta <= {MAX_BACKBONE_ANGLE_DELTA_DEG:g} degrees.

## Geometry Summary

{markdown_table(results, summary_cols)}

## Failed Variants

{markdown_table(failed, ['variant_id', 'axial_rise_scale', 'failed_checks'])}
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
    print(f"Audited {len(results)} rise-like variants")
    print(f"Geometry-interpretable variants: {safe_count}/{len(results)}")
    print(f"CSV: {args.out_csv}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
