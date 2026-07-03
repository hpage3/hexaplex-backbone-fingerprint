"""Generate a focused radial/axial global deformation refinement grid."""

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

from scripts.generate_global_deformation_variants import (
    center_from_atoms,
    format_pdb_coord_line,
    parse_pdb_atom_lines,
    z_bounds,
)


DEFAULT_SOURCE_PDB = Path("outputs/coordinates/ideal_hexaflex_variants/ideal_hexaflex_backbone_plus_carboxylate.pdb")
DEFAULT_OUTDIR = Path("outputs/coordinates/radial_axial_refinement_variants")
DEFAULT_MANIFEST = Path("outputs/metrics/radial_axial_refinement_variant_manifest.csv")
DEFAULT_REPORT = Path("outputs/reports/radial_axial_refinement_variant_generation.md")

RADIAL_SCALE_VALUES = [1.0000, 1.0025, 1.0050, 1.0075, 1.0100]
AXIAL_SCALE_VALUES = [0.9900, 0.9925, 0.9950, 0.9975, 1.0000]


@dataclass(frozen=True)
class RadialAxialSpec:
    """One radial/axial refinement variant."""

    variant_id: str
    radial_scale_xy: float
    axial_scale_z: float


def format_scale(value: float) -> str:
    """Format a scale value for stable filenames."""
    return f"{value:.4f}".replace(".", "p")


def variant_id(radial_scale_xy: float, axial_scale_z: float) -> str:
    """Return readable stable radial/axial variant ID."""
    return f"radial_{format_scale(radial_scale_xy)}__axial_{format_scale(axial_scale_z)}"


def refinement_grid() -> list[RadialAxialSpec]:
    """Return the focused 5x5 radial/axial refinement grid."""
    return [
        RadialAxialSpec(variant_id(radial, axial), radial, axial)
        for axial in AXIAL_SCALE_VALUES
        for radial in RADIAL_SCALE_VALUES
    ]


def output_path(outdir: Path, spec: RadialAxialSpec) -> Path:
    """Return output PDB path for one variant."""
    return outdir / f"{spec.variant_id}.pdb"


def apply_radial_axial_transform(coord: np.ndarray, radial_scale_xy: float, axial_scale_z: float, center: np.ndarray) -> np.ndarray:
    """Apply radial xy scaling and axial z scaling around center."""
    shifted = coord - center
    return np.array(
        [
            center[0] + radial_scale_xy * shifted[0],
            center[1] + radial_scale_xy * shifted[1],
            center[2] + axial_scale_z * shifted[2],
        ],
        dtype=float,
    )


def write_variant_pdb(source_lines: list[str], atoms, spec: RadialAxialSpec, center: np.ndarray, out_path: Path) -> None:
    """Write one transformed PDB, preserving non-coordinate fields."""
    out_lines = list(source_lines)
    for atom in atoms:
        coord = apply_radial_axial_transform(atom.coord, spec.radial_scale_xy, spec.axial_scale_z, center)
        out_lines[atom.index] = format_pdb_coord_line(atom.line, coord)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def manifest_row(
    spec: RadialAxialSpec,
    source_pdb: Path,
    out_path: Path,
    atom_count: int,
    center: np.ndarray,
    z_min: float,
    z_max: float,
    center_basis: str,
) -> dict[str, object]:
    """Build one manifest row."""
    return {
        "variant_id": spec.variant_id,
        "radial_scale_xy": spec.radial_scale_xy,
        "axial_scale_z": spec.axial_scale_z,
        "source_pdb": str(source_pdb),
        "output_pdb": str(out_path),
        "atom_count": atom_count,
        "center_x": center[0],
        "center_y": center[1],
        "center_z": center[2],
        "z_min": z_min,
        "z_max": z_max,
        "status": "ok",
        "notes": f"center_basis={center_basis}; focused_radial_axial_refinement; controlled_diagnostic_not_minimized",
    }


def generate_variants(source_pdb: Path, outdir: Path, manifest_path: Path, report_path: Path) -> pd.DataFrame:
    """Generate focused radial/axial variants and write manifest/report."""
    source_lines, atoms = parse_pdb_atom_lines(source_pdb)
    center, center_basis = center_from_atoms(atoms)
    z_min, z_max = z_bounds(atoms)
    rows = []
    for spec in refinement_grid():
        out_path = output_path(outdir, spec)
        write_variant_pdb(source_lines, atoms, spec, center, out_path)
        rows.append(manifest_row(spec, source_pdb, out_path, len(atoms), center, z_min, z_max, center_basis))
    manifest = pd.DataFrame(rows)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    report_path.write_text(build_report_text(manifest, source_pdb, center_basis), encoding="utf-8")
    return manifest


def build_report_text(manifest: pd.DataFrame, source_pdb: Path, center_basis: str) -> str:
    """Build generation report."""
    return f"""# Radial/Axial Refinement Variant Generation

## Purpose

This focused radial/axial refinement tests whether C and D can be tuned together after the first global deformation pilot separated axial sensitivity for C from radial sensitivity for D.

- Parent coordinate file: `{source_pdb}`
- Variants generated: {len(manifest)}
- Center basis: {center_basis}
- Radial scale values: {', '.join(f'{v:.4f}' for v in RADIAL_SCALE_VALUES)}
- Axial scale values: {', '.join(f'{v:.4f}' for v in AXIAL_SCALE_VALUES)}
- Coordinate directory: `outputs/coordinates/radial_axial_refinement_variants`

## Caution

These variants are controlled diagnostic perturbations, not minimized structures. They must be geometry-audited before any C/D scoring.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pdb", type=Path, default=DEFAULT_SOURCE_PDB)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = generate_variants(args.source_pdb, args.outdir, args.manifest, args.report)
    print(f"Generated {len(manifest)} radial/axial refinement variants")
    print(f"Coordinate directory: {args.outdir}")
    print(f"Manifest: {args.manifest}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
