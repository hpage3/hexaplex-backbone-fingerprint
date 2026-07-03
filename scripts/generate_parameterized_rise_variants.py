"""Generate layer/repeat-aware parameterized rise diagnostic variants."""

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

from scripts.audit_parent_axial_layers import LayerModel, assign_layer_index, infer_layers_from_ca_z, mean_layer_rise
from scripts.generate_global_deformation_variants import (
    PdbAtomLine,
    format_pdb_coord_line,
    parse_pdb_atom_lines,
    z_bounds,
)
from scripts.generate_radial_axial_refinement_variants import format_scale


DEFAULT_SOURCE_PDB = Path("outputs/coordinates/ideal_hexaflex_variants/ideal_hexaflex_backbone_plus_carboxylate.pdb")
DEFAULT_OUTDIR = Path("outputs/coordinates/parameterized_rise_variants")
DEFAULT_MANIFEST = Path("outputs/metrics/parameterized_rise_variant_manifest.csv")
DEFAULT_REPORT = Path("outputs/reports/parameterized_rise_variant_generation.md")
RISE_SCALE_VALUES = [0.9600, 0.9650, 0.9700, 0.9750, 0.9800, 0.9850, 0.9900, 0.9950, 1.0000]


@dataclass(frozen=True)
class ParameterizedRiseSpec:
    """One parameterized rise variant."""

    variant_id: str
    rise_scale: float


def variant_id(rise_scale: float) -> str:
    """Return stable parameterized-rise variant ID."""
    return f"parameterized_rise_{format_scale(rise_scale)}"


def estimated_percent_rise_compression(rise_scale: float) -> float:
    """Return percent effective rise compression."""
    return (1.0 - rise_scale) * 100.0


def rise_grid() -> list[ParameterizedRiseSpec]:
    """Return requested parameterized rise grid."""
    return [ParameterizedRiseSpec(variant_id(scale), scale) for scale in RISE_SCALE_VALUES]


def parameterized_rise_z(z: float, layer_center: float, global_center_z: float, rise_scale: float) -> float:
    """Move layer center by rise scale while preserving local z offset."""
    local_z_offset = z - layer_center
    new_layer_center = global_center_z + rise_scale * (layer_center - global_center_z)
    return new_layer_center + local_z_offset


def transform_atom(atom: PdbAtomLine, layer_model: LayerModel, global_center_z: float, rise_scale: float) -> np.ndarray:
    """Apply layer-aware parameterized rise transform to one atom."""
    layer_idx = assign_layer_index(atom.z, layer_model.layer_centers)
    coord = atom.coord.copy()
    coord[2] = parameterized_rise_z(atom.z, layer_model.layer_centers[layer_idx], global_center_z, rise_scale)
    return coord


def output_path(outdir: Path, spec: ParameterizedRiseSpec) -> Path:
    """Return PDB output path."""
    return outdir / f"{spec.variant_id}.pdb"


def write_variant_pdb(
    source_lines: list[str],
    atoms: list[PdbAtomLine],
    spec: ParameterizedRiseSpec,
    layer_model: LayerModel,
    global_center_z: float,
    out_path: Path,
) -> None:
    """Write one parameterized-rise PDB."""
    out_lines = list(source_lines)
    for atom in atoms:
        coord = transform_atom(atom, layer_model, global_center_z, spec.rise_scale)
        out_lines[atom.index] = format_pdb_coord_line(atom.line, coord)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def manifest_row(
    spec: ParameterizedRiseSpec,
    source_pdb: Path,
    out_path: Path,
    atoms: list[PdbAtomLine],
    layer_model: LayerModel,
    global_center_z: float,
) -> dict[str, object]:
    """Return one manifest row."""
    parent_z_min, parent_z_max = z_bounds(atoms)
    parent_span = parent_z_max - parent_z_min
    parent_mean_rise = mean_layer_rise(layer_model.layer_centers)
    transformed_z = [transform_atom(atom, layer_model, global_center_z, spec.rise_scale)[2] for atom in atoms]
    variant_span = max(transformed_z) - min(transformed_z)
    return {
        "variant_id": spec.variant_id,
        "rise_scale": spec.rise_scale,
        "estimated_percent_rise_compression": estimated_percent_rise_compression(spec.rise_scale),
        "layer_model": layer_model.layer_model,
        "layers_used": len(layer_model.layer_centers),
        "source_pdb": str(source_pdb),
        "output_pdb": str(out_path),
        "atom_count": len(atoms),
        "z_global_center": global_center_z,
        "parent_z_span_A": parent_span,
        "variant_z_span_A": variant_span,
        "parent_mean_layer_rise_A": parent_mean_rise,
        "variant_mean_layer_rise_A": parent_mean_rise * spec.rise_scale,
        "status": "ok",
        "notes": "layer_center_rise_transform; within_layer_z_offsets_preserved; diagnostic_not_minimized",
    }


def generate_variants(source_pdb: Path, outdir: Path, manifest_path: Path, report_path: Path) -> pd.DataFrame:
    """Generate parameterized rise variants and write manifest/report."""
    source_lines, atoms = parse_pdb_atom_lines(source_pdb)
    ca_atoms = [atom for atom in atoms if atom.is_ca]
    layer_model = infer_layers_from_ca_z([atom.z for atom in ca_atoms])
    global_center_z = float(np.mean(layer_model.layer_centers))
    rows = []
    for spec in rise_grid():
        out_path = output_path(outdir, spec)
        write_variant_pdb(source_lines, atoms, spec, layer_model, global_center_z, out_path)
        rows.append(manifest_row(spec, source_pdb, out_path, atoms, layer_model, global_center_z))
    manifest = pd.DataFrame(rows)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    report_path.write_text(build_report_text(manifest, source_pdb, layer_model), encoding="utf-8")
    return manifest


def build_report_text(manifest: pd.DataFrame, source_pdb: Path, layer_model: LayerModel) -> str:
    """Build generation report."""
    return f"""# Parameterized Rise Variant Generation

## Purpose

This branch tests layer/repeat-aware rise compression as a more interpretable structural parameter than continuous global z-scaling.

- Source PDB: `{source_pdb}`
- Layer model used: {layer_model.layer_model}
- Layer count: {len(layer_model.layer_centers)}
- Rise scale grid: {', '.join(f'{v:.4f}' for v in RISE_SCALE_VALUES)}
- Parent mean layer rise: {mean_layer_rise(layer_model.layer_centers):.4f} A
- Variant mean layer rise range: {manifest['variant_mean_layer_rise_A'].min():.4f} to {manifest['variant_mean_layer_rise_A'].max():.4f} A
- Variants generated: {len(manifest)}

## Caution

These are parameterized diagnostic structures, not minimized structures. They preserve within-layer z offsets while changing inter-layer spacing.
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
    print(f"Generated {len(manifest)} parameterized rise variants")
    print(f"Manifest: {args.manifest}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
