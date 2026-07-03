"""Audit parent axial layers for parameterized rise diagnostics."""

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

from scripts.generate_global_deformation_variants import PdbAtomLine, parse_pdb_atom_lines
from scripts.score_radial_axial_refinement_variant_cd import markdown_table


DEFAULT_SOURCE_PDB = Path("outputs/coordinates/ideal_hexaflex_variants/ideal_hexaflex_backbone_plus_carboxylate.pdb")
DEFAULT_OUT_CSV = Path("outputs/metrics/parent_axial_layer_audit.csv")
DEFAULT_REPORT = Path("outputs/reports/parent_axial_layer_audit.md")
DEFAULT_GAP_THRESHOLD_A = 0.2


@dataclass(frozen=True)
class LayerModel:
    """Deterministic axial layer model."""

    layer_centers: list[float]
    layer_model: str
    gap_threshold_A: float


def z_statistics(atoms: list[PdbAtomLine]) -> dict[str, float]:
    """Return basic z-coordinate statistics."""
    z_values = np.array([atom.z for atom in atoms], dtype=float)
    return {
        "z_min": float(z_values.min()),
        "z_max": float(z_values.max()),
        "z_span": float(z_values.max() - z_values.min()),
        "z_center": float(z_values.mean()),
    }


def infer_layers_from_ca_z(ca_z_values: list[float], gap_threshold_A: float = DEFAULT_GAP_THRESHOLD_A) -> LayerModel:
    """Infer axial layers from sorted C-alpha z gaps."""
    if not ca_z_values:
        raise ValueError("Cannot infer axial layers without C-alpha z values.")
    sorted_z = sorted(float(z) for z in ca_z_values)
    groups: list[list[float]] = [[sorted_z[0]]]
    for z in sorted_z[1:]:
        if z - groups[-1][-1] > gap_threshold_A:
            groups.append([])
        groups[-1].append(z)
    centers = [float(np.mean(group)) for group in groups]
    return LayerModel(centers, "ca_z_gap_layers", gap_threshold_A)


def fallback_even_layers(z_values: list[float], layer_count: int) -> LayerModel:
    """Build a deterministic fixed-number fallback layer model."""
    if layer_count <= 1:
        raise ValueError("layer_count must be greater than 1.")
    z_min = min(z_values)
    z_max = max(z_values)
    centers = np.linspace(z_min, z_max, layer_count).tolist()
    return LayerModel([float(value) for value in centers], "fallback_even_z_layers", float("nan"))


def assign_layer_index(z: float, layer_centers: list[float]) -> int:
    """Assign one z value to nearest layer center."""
    distances = [abs(float(z) - center) for center in layer_centers]
    return int(np.argmin(distances))


def mean_layer_rise(layer_centers: list[float]) -> float:
    """Return mean center-to-center rise for sorted layers."""
    if len(layer_centers) < 2:
        return float("nan")
    centers = sorted(layer_centers)
    return float(np.mean(np.diff(centers)))


def build_layer_table(atoms: list[PdbAtomLine], layer_model: LayerModel) -> pd.DataFrame:
    """Return per-layer atom and C-alpha summary table."""
    rows = []
    for idx, center in enumerate(layer_model.layer_centers):
        assigned = [atom for atom in atoms if assign_layer_index(atom.z, layer_model.layer_centers) == idx]
        ca = [atom for atom in assigned if atom.is_ca]
        z_values = [atom.z for atom in assigned]
        rows.append(
            {
                "layer_index": idx,
                "atom_count": len(assigned),
                "ca_count": len(ca),
                "mean_z_A": float(np.mean(z_values)) if z_values else float("nan"),
                "min_z_A": min(z_values) if z_values else float("nan"),
                "max_z_A": max(z_values) if z_values else float("nan"),
                "layer_center_z_A": center,
            }
        )
    return pd.DataFrame(rows)


def audit_parent_layers(source_pdb: Path, gap_threshold_A: float = DEFAULT_GAP_THRESHOLD_A) -> tuple[pd.DataFrame, dict[str, object]]:
    """Infer parent axial layers and return layer table plus summary."""
    _, atoms = parse_pdb_atom_lines(source_pdb)
    ca_atoms = [atom for atom in atoms if atom.is_ca]
    if ca_atoms:
        layer_model = infer_layers_from_ca_z([atom.z for atom in ca_atoms], gap_threshold_A)
        notes = "C-alpha z-gap layer model used."
    else:
        layer_model = fallback_even_layers([atom.z for atom in atoms], 1)
        notes = "No C-alpha atoms found; fallback layer model used."
    table = build_layer_table(atoms, layer_model)
    z_stats = z_statistics(atoms)
    summary = {
        "source_pdb": str(source_pdb),
        "atom_count": len(atoms),
        "ca_count": len(ca_atoms),
        **z_stats,
        "layer_model": layer_model.layer_model,
        "proposed_layer_count": len(layer_model.layer_centers),
        "mean_layer_rise_A": mean_layer_rise(layer_model.layer_centers),
        "gap_threshold_A": layer_model.gap_threshold_A,
        "notes": notes,
    }
    return table, summary


def build_report_text(layer_table: pd.DataFrame, summary: dict[str, object]) -> str:
    """Build parent axial layer audit report."""
    return f"""# Parent Axial Layer Audit

## Purpose

This audit checks whether the parent PDB has recognizable axial layers that can support a layer/repeat-aware rise parameterization.

- Source PDB: `{summary['source_pdb']}`
- Atom count: {summary['atom_count']}
- C-alpha count: {summary['ca_count']}
- z_min/z_max/z_span: {summary['z_min']:.4f} / {summary['z_max']:.4f} / {summary['z_span']:.4f} A
- z_center: {summary['z_center']:.4f} A
- Layer model: {summary['layer_model']}
- Proposed number of axial layers: {summary['proposed_layer_count']}
- Mean layer-to-layer rise estimate: {summary['mean_layer_rise_A']:.4f} A

## Layer Summary

{markdown_table(layer_table, ['layer_index', 'atom_count', 'ca_count', 'mean_z_A', 'min_z_A', 'max_z_A'])}

## Notes And Cautions

{summary['notes']} The inferred layers are deterministic and suitable for a diagnostic parameterized-rise branch. They are not chemical validation of a final structural model.
"""


def run(source_pdb: Path, out_csv: Path, report_path: Path, gap_threshold_A: float) -> tuple[pd.DataFrame, dict[str, object]]:
    """Run parent axial layer audit and write outputs."""
    layer_table, summary = audit_parent_layers(source_pdb, gap_threshold_A)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    layer_table.to_csv(out_csv, index=False)
    report_path.write_text(build_report_text(layer_table, summary), encoding="utf-8")
    return layer_table, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pdb", type=Path, default=DEFAULT_SOURCE_PDB)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--gap-threshold", type=float, default=DEFAULT_GAP_THRESHOLD_A)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    table, summary = run(args.source_pdb, args.out_csv, args.report, args.gap_threshold)
    print(f"Inferred {len(table)} axial layers")
    print(f"Mean parent layer rise: {summary['mean_layer_rise_A']:.4f} A")
    print(f"CSV: {args.out_csv}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
