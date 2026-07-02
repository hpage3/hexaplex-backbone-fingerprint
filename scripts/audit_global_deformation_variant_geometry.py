"""Audit global deformation variants with diagnostic geometry sanity gates."""

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

from scripts.audit_constrained_phi_psi_candidate_geometry import (
    Atom,
    atom_map,
    backbone_angles,
    backbone_bonds,
    distance,
    max_abs_delta,
    parse_pdb,
)


DEFAULT_MANIFEST = Path("outputs/metrics/global_deformation_variant_manifest.csv")
DEFAULT_SOURCE_PDB = Path("outputs/coordinates/ideal_hexaflex_variants/ideal_hexaflex_backbone_plus_carboxylate.pdb")
DEFAULT_OUT_CSV = Path("outputs/metrics/global_deformation_variant_geometry_audit.csv")
DEFAULT_REPORT = Path("outputs/reports/global_deformation_variant_geometry_audit.md")

MAX_BACKBONE_BOND_DELTA_A = 0.15
MAX_BACKBONE_ANGLE_DELTA_DEG = 10.0


def identity_matches(parent_atoms: list[Atom], variant_atoms: list[Atom]) -> bool:
    """Return whether parent and variant atom identities match in order."""
    if len(parent_atoms) != len(variant_atoms):
        return False
    return [atom.key for atom in parent_atoms] == [atom.key for atom in variant_atoms]


def coordinate_array(atoms: list[Atom]) -> np.ndarray:
    """Return atom coordinates as an array."""
    return np.array([atom.coord for atom in atoms], dtype=float)


def displacement_metrics(parent_atoms: list[Atom], variant_atoms: list[Atom]) -> dict[str, float]:
    """Return max displacement and RMSD metrics for matching atoms."""
    if len(parent_atoms) != len(variant_atoms):
        return {"max_displacement_A": math.inf, "rmsd_all_atoms_A": math.inf, "rmsd_ca_A": math.inf}
    parent = coordinate_array(parent_atoms)
    variant = coordinate_array(variant_atoms)
    shifts = np.linalg.norm(variant - parent, axis=1)
    ca_indices = [idx for idx, atom in enumerate(parent_atoms) if atom.name == "CA"]
    ca_rmsd = float(np.sqrt(np.mean(shifts[ca_indices] ** 2))) if ca_indices else float("nan")
    return {
        "max_displacement_A": float(np.max(shifts)) if len(shifts) else float("nan"),
        "rmsd_all_atoms_A": float(np.sqrt(np.mean(shifts**2))) if len(shifts) else float("nan"),
        "rmsd_ca_A": ca_rmsd,
    }


def mean_ca_radius(atoms: list[Atom]) -> float:
    """Return mean C-alpha radius around the C-alpha xy centroid."""
    ca = [atom for atom in atoms if atom.name == "CA"]
    chosen = ca if ca else atoms
    coords = coordinate_array(chosen)
    center_xy = coords[:, :2].mean(axis=0)
    radii = np.linalg.norm(coords[:, :2] - center_xy, axis=1)
    return float(np.mean(radii))


def z_span(atoms: list[Atom]) -> float:
    """Return z-coordinate span."""
    coords = coordinate_array(atoms)
    return float(coords[:, 2].max() - coords[:, 2].min())


def backbone_delta_metrics(parent_atoms: list[Atom], variant_atoms: list[Atom]) -> dict[str, float]:
    """Return max backbone bond and angle deviations relative to parent."""
    return {
        "max_backbone_bond_delta_A": max_abs_delta(backbone_bonds(parent_atoms), backbone_bonds(variant_atoms)),
        "max_backbone_angle_delta_deg": max_abs_delta(backbone_angles(parent_atoms), backbone_angles(variant_atoms)),
    }


def geometry_interpretable(row: dict[str, object]) -> bool:
    """Classify one global deformation variant with loose diagnostic gates."""
    return (
        bool(row.get("atom_count_matches_parent"))
        and bool(row.get("identity_matches_parent"))
        and float(row.get("max_backbone_bond_delta_A", math.inf)) <= MAX_BACKBONE_BOND_DELTA_A
        and float(row.get("max_backbone_angle_delta_deg", math.inf)) <= MAX_BACKBONE_ANGLE_DELTA_DEG
    )


def failed_checks(row: dict[str, object]) -> list[str]:
    """Return stable global-deformation failure reasons."""
    reasons = []
    if not row.get("atom_count_matches_parent"):
        reasons.append("atom_count_mismatch")
    if not row.get("identity_matches_parent"):
        reasons.append("identity_mismatch")
    if float(row.get("max_backbone_bond_delta_A", math.inf)) > MAX_BACKBONE_BOND_DELTA_A:
        reasons.append("backbone_bond_delta_exceeds_global_threshold")
    if float(row.get("max_backbone_angle_delta_deg", math.inf)) > MAX_BACKBONE_ANGLE_DELTA_DEG:
        reasons.append("backbone_angle_delta_exceeds_global_threshold")
    return reasons


def audit_variant(parent_atoms: list[Atom], manifest_row: pd.Series) -> dict[str, object]:
    """Audit one global deformation variant."""
    output_pdb = Path(str(manifest_row["output_pdb"]))
    variant_atoms = parse_pdb(output_pdb) if output_pdb.exists() else []
    atom_count_matches = len(parent_atoms) == len(variant_atoms)
    identity_ok = identity_matches(parent_atoms, variant_atoms) if output_pdb.exists() else False
    disp = displacement_metrics(parent_atoms, variant_atoms) if output_pdb.exists() else {
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
    row = {
        "variant_id": manifest_row["variant_id"],
        "deformation_mode": manifest_row["deformation_mode"],
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
    """Run global deformation geometry audit and write outputs."""
    manifest = pd.read_csv(manifest_path)
    parent_atoms = parse_pdb(source_pdb)
    rows = [audit_variant(parent_atoms, row) for _, row in manifest.iterrows()]
    results = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_csv, index=False)
    report_path.write_text(build_report_text(results, source_pdb), encoding="utf-8")
    return results


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    """Render selected columns as a markdown table."""
    if df.empty:
        return "_None._"
    columns = [column for column in columns if column in df.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for record in df[columns].itertuples(index=False):
        values = [f"{value:.6g}" if isinstance(value, float) else str(value) for value in record]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_report_text(results: pd.DataFrame, source_pdb: Path) -> str:
    """Build global deformation audit report."""
    total = len(results)
    safe = results[results["geometry_interpretable"].astype(bool)] if total else results
    failed = results[~results["geometry_interpretable"].astype(bool)] if total else results
    summary_cols = [
        "variant_id",
        "deformation_mode",
        "geometry_interpretable",
        "max_displacement_A",
        "rmsd_ca_A",
        "mean_ca_radius_delta_A",
        "z_span_delta_A",
        "max_backbone_bond_delta_A",
        "max_backbone_angle_delta_deg",
        "failed_checks",
    ]
    return f"""# Global Deformation Variant Geometry Audit

## Purpose

This audit checks controlled global deformation variants after the local C-alpha anchored torsion basin showed C/D robustness. The goal is to decide which generated deformations are interpretable enough for later C/D diffraction scoring.

## Inputs

- Source PDB: `{source_pdb}`
- Variants audited: {total}
- Geometry-interpretable variants: {len(safe)}/{total}

## Global-Deformation Sanity Gates

These are looser than local torsion gates because the deformation is global and diagnostic. They are not energy-minimized chemistry gates.

- Atom count must match parent.
- Chain/residue/atom identity must match parent.
- Max backbone bond-length delta <= {MAX_BACKBONE_BOND_DELTA_A:g} A.
- Max backbone angle delta <= {MAX_BACKBONE_ANGLE_DELTA_DEG:g} degrees.

## Deformation Modes Generated

{markdown_table(results, summary_cols)}

## Safe / Interpretable Variants

{markdown_table(safe, ['variant_id', 'deformation_mode', 'max_displacement_A', 'rmsd_ca_A', 'mean_ca_radius_delta_A', 'z_span_delta_A'])}

## Failed Variants

{markdown_table(failed, ['variant_id', 'deformation_mode', 'failed_checks'])}

## Caution

These are controlled diagnostic perturbations to test what kinds of global geometric changes move C/D in later scoring. They are not minimized structural models and should not be used to claim a structural mechanism by themselves.
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
    print(f"Audited {len(results)} global deformation variants")
    print(f"Geometry-interpretable variants: {safe_count}/{len(results)}")
    print(f"CSV: {args.out_csv}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
