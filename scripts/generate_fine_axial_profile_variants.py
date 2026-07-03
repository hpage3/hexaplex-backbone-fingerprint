"""Generate fine axial-only variants for local C/D profile diagnostics."""

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

from scripts.generate_axial_only_extension_variants import apply_axial_only_transform
from scripts.generate_global_deformation_variants import (
    center_from_atoms,
    format_pdb_coord_line,
    parse_pdb_atom_lines,
    z_bounds,
)
from scripts.generate_radial_axial_refinement_variants import format_scale


DEFAULT_SOURCE_PDB = Path("outputs/coordinates/ideal_hexaflex_variants/ideal_hexaflex_backbone_plus_carboxylate.pdb")
DEFAULT_OUTDIR = Path("outputs/coordinates/fine_axial_profile_variants")
DEFAULT_MANIFEST = Path("outputs/metrics/fine_axial_profile_variant_manifest.csv")
DEFAULT_REPORT = Path("outputs/reports/fine_axial_profile_variant_generation.md")

RADIAL_SCALE_XY = 1.0000
AXIAL_SCALE_VALUES = [0.9700, 0.9725, 0.9750, 0.9775, 0.9800, 0.9825, 0.9850]


@dataclass(frozen=True)
class FineAxialSpec:
    """One fine axial profile variant."""

    variant_id: str
    radial_scale_xy: float
    axial_scale_z: float


def variant_id(axial_scale_z: float) -> str:
    """Return stable fine axial variant ID."""
    return f"fine_axial_{format_scale(axial_scale_z)}"


def fine_axial_grid() -> list[FineAxialSpec]:
    """Return the fine axial diagnostic grid."""
    return [FineAxialSpec(variant_id(axial), RADIAL_SCALE_XY, axial) for axial in AXIAL_SCALE_VALUES]


def output_path(outdir: Path, spec: FineAxialSpec) -> Path:
    """Return output PDB path for one variant."""
    return outdir / f"{spec.variant_id}.pdb"


def write_variant_pdb(source_lines: list[str], atoms, spec: FineAxialSpec, center: np.ndarray, out_path: Path) -> None:
    """Write one fine axial transformed PDB."""
    out_lines = list(source_lines)
    for atom in atoms:
        coord = apply_axial_only_transform(atom.coord, spec.axial_scale_z, center)
        out_lines[atom.index] = format_pdb_coord_line(atom.line, coord)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def manifest_row(
    spec: FineAxialSpec,
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
        "notes": f"center_basis={center_basis}; fine_axial_profile_diagnostic; controlled_diagnostic_not_minimized",
    }


def generate_variants(source_pdb: Path, outdir: Path, manifest_path: Path, report_path: Path) -> pd.DataFrame:
    """Generate fine axial variants and write outputs."""
    source_lines, atoms = parse_pdb_atom_lines(source_pdb)
    center, center_basis = center_from_atoms(atoms)
    z_min, z_max = z_bounds(atoms)
    rows = []
    for spec in fine_axial_grid():
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
    return f"""# Fine Axial Profile Variant Generation

## Purpose

This fine axial profile diagnostic focuses on axial scale 0.9700 to 0.9850 after the axial-only extension showed a discretized C peak response.

- Parent coordinate file: `{source_pdb}`
- Variants generated: {len(manifest)}
- Center basis: {center_basis}
- Radial scale: {RADIAL_SCALE_XY:.4f}
- Axial scale values: {', '.join(f'{v:.4f}' for v in AXIAL_SCALE_VALUES)}
- Coordinate directory: `outputs/coordinates/fine_axial_profile_variants`

## Caution

These variants are controlled diagnostic perturbations, not minimized structures. They must be geometry-audited before C/D scoring or local profile diagnostics.
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
    print(f"Generated {len(manifest)} fine axial profile variants")
    print(f"Coordinate directory: {args.outdir}")
    print(f"Manifest: {args.manifest}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
