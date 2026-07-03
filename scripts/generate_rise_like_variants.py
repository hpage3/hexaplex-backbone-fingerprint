"""Generate controlled rise-like axial compression variants."""

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
from scripts.generate_global_deformation_variants import center_from_atoms, format_pdb_coord_line, parse_pdb_atom_lines, z_bounds
from scripts.generate_radial_axial_refinement_variants import format_scale


DEFAULT_SOURCE_PDB = Path("outputs/coordinates/ideal_hexaflex_variants/ideal_hexaflex_backbone_plus_carboxylate.pdb")
DEFAULT_OUTDIR = Path("outputs/coordinates/rise_like_variants")
DEFAULT_MANIFEST = Path("outputs/metrics/rise_like_variant_manifest.csv")
DEFAULT_REPORT = Path("outputs/reports/rise_like_variant_generation.md")

RISE_SCALE_VALUES = [0.9600, 0.9650, 0.9700, 0.9750, 0.9800, 0.9850, 0.9900, 0.9950, 1.0000]


@dataclass(frozen=True)
class RiseLikeSpec:
    """One rise-like diagnostic variant."""

    variant_id: str
    axial_rise_scale: float


def variant_id(axial_rise_scale: float) -> str:
    """Return stable rise-like variant ID."""
    return f"rise_like_{format_scale(axial_rise_scale)}"


def estimated_percent_compression(axial_rise_scale: float) -> float:
    """Return percent compression relative to the parent z-span."""
    return (1.0 - axial_rise_scale) * 100.0


def rise_like_grid() -> list[RiseLikeSpec]:
    """Return the rise-like diagnostic grid."""
    return [RiseLikeSpec(variant_id(scale), scale) for scale in RISE_SCALE_VALUES]


def output_path(outdir: Path, spec: RiseLikeSpec) -> Path:
    """Return output PDB path for one variant."""
    return outdir / f"{spec.variant_id}.pdb"


def write_variant_pdb(source_lines: list[str], atoms, spec: RiseLikeSpec, center: np.ndarray, out_path: Path) -> None:
    """Write one rise-like transformed PDB."""
    out_lines = list(source_lines)
    for atom in atoms:
        coord = apply_axial_only_transform(atom.coord, spec.axial_rise_scale, center)
        out_lines[atom.index] = format_pdb_coord_line(atom.line, coord)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def manifest_row(
    spec: RiseLikeSpec,
    source_pdb: Path,
    out_path: Path,
    atom_count: int,
    center: np.ndarray,
    z_min: float,
    z_max: float,
    center_basis: str,
) -> dict[str, object]:
    """Build one manifest row."""
    parent_span = z_max - z_min
    return {
        "variant_id": spec.variant_id,
        "axial_rise_scale": spec.axial_rise_scale,
        "estimated_percent_compression": estimated_percent_compression(spec.axial_rise_scale),
        "source_pdb": str(source_pdb),
        "output_pdb": str(out_path),
        "atom_count": atom_count,
        "center_x": center[0],
        "center_y": center[1],
        "center_z": center[2],
        "z_min": z_min,
        "z_max": z_max,
        "parent_z_span_A": parent_span,
        "variant_z_span_A": parent_span * spec.axial_rise_scale,
        "status": "ok",
        "notes": f"center_basis={center_basis}; rise_like_diagnostic_proxy; controlled_diagnostic_not_minimized",
    }


def generate_variants(source_pdb: Path, outdir: Path, manifest_path: Path, report_path: Path) -> pd.DataFrame:
    """Generate rise-like variants and write outputs."""
    source_lines, atoms = parse_pdb_atom_lines(source_pdb)
    center, center_basis = center_from_atoms(atoms)
    z_min, z_max = z_bounds(atoms)
    rows = []
    for spec in rise_like_grid():
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
    return f"""# Rise-Like Variant Generation

## Purpose

This branch translates generic axial-compression sensitivity into an explicit rise-like diagnostic proxy.

- Parent coordinate file: `{source_pdb}`
- Variants generated: {len(manifest)}
- Center basis: {center_basis}
- Rise-like scale values: {', '.join(f'{v:.4f}' for v in RISE_SCALE_VALUES)}
- Coordinate directory: `outputs/coordinates/rise_like_variants`

## Caution

These variants are controlled diagnostic perturbations, not minimized physical structures. They should be geometry-audited before C/D scoring.
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
    print(f"Generated {len(manifest)} rise-like variants")
    print(f"Coordinate directory: {args.outdir}")
    print(f"Manifest: {args.manifest}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
