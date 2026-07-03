"""Generate and score parent-derived rise bridge variants.

This workflow starts from the real parent PDB and preserves atom content,
residue identity, chain/register labels, carboxylates, and x/y coordinates.
Only axial layer-center spacing is changed. The no-change reference is scored
first and interpretation is blocked if it does not reproduce parent C/D.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hexaplex_backbone_fingerprint.parametric_powder_scan import debye_profile, make_q_grid, nearest_peak
from scripts.audit_parent_axial_layers import LayerModel, assign_layer_index, infer_layers_from_ca_z, mean_layer_rise
from scripts.generate_global_deformation_variants import PdbAtomLine, format_pdb_coord_line, parse_pdb_atom_lines


DEFAULT_PARENT_PDB = Path("outputs/coordinates/ideal_hexaflex_variants/ideal_hexaflex_backbone_plus_carboxylate.pdb")
DEFAULT_OUTDIR = Path("outputs/coordinates/parent_derived_rise_bridge")
DEFAULT_SCORE_CSV = Path("outputs/metrics/parent_derived_rise_bridge_abcd_scores.csv")
DEFAULT_GEOMETRY_CSV = Path("outputs/metrics/parent_derived_rise_bridge_geometry_summary.csv")
DEFAULT_REPORT = Path("outputs/reports/parent_derived_rise_bridge_summary.md")

NOMINAL_REFERENCE_RISE_A = 3.40
REQUESTED_NOMINAL_RISES_A = [3.40, 3.38, 3.35]
TARGETS_A = {"A": 7.9, "B": 6.5, "C": 5.6, "D": 7.3}
EXPECTED_PARENT_C_A = 5.745
EXPECTED_PARENT_D_A = 7.276
REFERENCE_TOLERANCE_A = 0.05


@dataclass(frozen=True)
class ParentDerivedRiseSpec:
    """One parent-derived rise bridge variant."""

    variant_id: str
    requested_nominal_rise_A: float | None
    axial_scale: float


def format_nominal_rise(value: float) -> str:
    """Return stable filename text for a nominal rise value."""
    return f"{value:.2f}".replace(".", "p").replace("-", "m")


def variant_id_for_nominal_rise(value: float) -> str:
    """Return stable parent-derived variant ID."""
    return f"parent_derived_rise_{format_nominal_rise(value)}_equiv"


def nominal_rise_to_scale(requested_rise_A: float, reference_rise_A: float = NOMINAL_REFERENCE_RISE_A) -> float:
    """Map nominal physical rise to parent-derived axial scale."""
    if reference_rise_A <= 0:
        raise ValueError("reference_rise_A must be positive.")
    return float(requested_rise_A) / float(reference_rise_A)


def bridge_specs() -> list[ParentDerivedRiseSpec]:
    """Return exact requested parent-derived bridge candidate set."""
    specs = [ParentDerivedRiseSpec("parent_derived_reference", None, 1.0)]
    specs.extend(
        ParentDerivedRiseSpec(variant_id_for_nominal_rise(value), value, nominal_rise_to_scale(value))
        for value in REQUESTED_NOMINAL_RISES_A
    )
    return specs


def output_path(outdir: Path, spec: ParentDerivedRiseSpec) -> Path:
    """Return output PDB path for a parent-derived variant."""
    return outdir / f"{spec.variant_id}.pdb"


def parameterized_rise_z(z: float, layer_center: float, global_center_z: float, axial_scale: float) -> float:
    """Move a layer center by axial scale while preserving local z offset."""
    local_offset = z - layer_center
    new_layer_center = global_center_z + axial_scale * (layer_center - global_center_z)
    return float(new_layer_center + local_offset)


def transformed_coord(atom: PdbAtomLine, layer_model: LayerModel, global_center_z: float, axial_scale: float) -> np.ndarray:
    """Return transformed coordinate for one atom."""
    layer_index = assign_layer_index(atom.z, layer_model.layer_centers)
    coord = atom.coord.copy()
    coord[2] = parameterized_rise_z(atom.z, layer_model.layer_centers[layer_index], global_center_z, axial_scale)
    return coord


def write_parent_derived_variant(
    source_lines: list[str],
    atoms: list[PdbAtomLine],
    spec: ParentDerivedRiseSpec,
    layer_model: LayerModel,
    global_center_z: float,
    path: Path,
) -> None:
    """Write a parent-derived PDB variant, preserving all non-coordinate fields."""
    out_lines = list(source_lines)
    for atom in atoms:
        coord = transformed_coord(atom, layer_model, global_center_z, spec.axial_scale)
        out_lines[atom.index] = format_pdb_coord_line(atom.line, coord)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def atom_identity_tuple(atom: PdbAtomLine) -> tuple[str, str, str, str]:
    """Return identity fields that must be preserved."""
    return (atom.chain, atom.resseq, atom.resname, atom.atom_name)


def identity_preserved(parent_atoms: list[PdbAtomLine], variant_atoms: list[PdbAtomLine]) -> bool:
    """Return whether atom identities are preserved in order."""
    return [atom_identity_tuple(atom) for atom in parent_atoms] == [atom_identity_tuple(atom) for atom in variant_atoms]


def pdb_coordinates(path: Path) -> np.ndarray:
    """Return all PDB atom coordinates."""
    _, atoms = parse_pdb_atom_lines(path)
    return np.array([atom.coord for atom in atoms], dtype=float)


def score_pdb_abcd(path: Path, q_step: float = 0.01, d_min: float = 2.5, d_max: float = 12.0) -> dict[str, object]:
    """Score PDB coordinates against A/B/C/D using the existing Debye helper."""
    coords = pdb_coordinates(path)
    profile = debye_profile(coords, make_q_grid(d_min_A=d_min, d_max_A=d_max, q_step=q_step))
    row: dict[str, object] = {}
    for band, target in TARGETS_A.items():
        hit = nearest_peak(profile, target, tolerance_A=0.20)
        row[f"observed_{band}_d_A"] = hit.peak_d_A
        row[f"{band}_error_A"] = hit.error_A
        row[f"{band}_score"] = hit.intensity
    return row


def residue_keys(atoms: list[PdbAtomLine]) -> list[tuple[str, str, str]]:
    """Return unique residue keys in order."""
    seen = set()
    keys = []
    for atom in atoms:
        key = (atom.chain, atom.resseq, atom.resname)
        if key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


def carboxylate_present(atoms: list[PdbAtomLine]) -> bool:
    """Return whether carboxylate-like atom names are present."""
    names = {atom.atom_name for atom in atoms}
    return bool(names & {"OE1", "OE2", "OD1", "OD2", "OXT"})


def xy_radius_values(atoms: list[PdbAtomLine]) -> np.ndarray:
    """Return C-alpha xy radii around the C-alpha xy centroid."""
    cas = [atom for atom in atoms if atom.is_ca]
    selected = cas if cas else atoms
    arr = np.array([atom.coord for atom in selected], dtype=float)
    center_xy = arr[:, :2].mean(axis=0)
    return np.linalg.norm(arr[:, :2] - center_xy, axis=1)


def ca_rise_values(atoms: list[PdbAtomLine]) -> np.ndarray:
    """Return absolute same-chain adjacent C-alpha z differences."""
    values = []
    for chain in sorted({atom.chain for atom in atoms}):
        cas = sorted([atom for atom in atoms if atom.chain == chain and atom.is_ca], key=lambda atom: atom.index)
        z_values = [atom.z for atom in cas]
        values.extend(abs(b - a) for a, b in zip(z_values, z_values[1:]))
    return np.array(values, dtype=float)


def interstrand_nn_ca_distances(atoms: list[PdbAtomLine]) -> np.ndarray:
    """Return nearest C-alpha distance to another chain for each C-alpha atom."""
    cas = [atom for atom in atoms if atom.is_ca]
    distances = []
    for atom in cas:
        others = [other for other in cas if other.chain != atom.chain]
        if others:
            distances.append(min(float(np.linalg.norm(atom.coord - other.coord)) for other in others))
    return np.array(distances, dtype=float)


def geometry_summary_row(spec: ParentDerivedRiseSpec, path: Path) -> dict[str, object]:
    """Return concise geometry summary for one variant."""
    _, atoms = parse_pdb_atom_lines(path)
    z_values = np.array([atom.z for atom in atoms], dtype=float)
    radii = xy_radius_values(atoms)
    nn = interstrand_nn_ca_distances(atoms)
    rises = ca_rise_values(atoms)
    return {
        "variant_id": spec.variant_id,
        "z_span_A": float(z_values.max() - z_values.min()),
        "mean_ca_radius_A": float(np.mean(radii)),
        "median_ca_radius_A": float(np.median(radii)),
        "median_interstrand_nn_ca_distance_A": float(np.median(nn)) if len(nn) else float("nan"),
        "median_ca_rise_A": float(np.median(rises)) if len(rises) else float("nan"),
        "atom_count": len(atoms),
        "residue_count": len(residue_keys(atoms)),
        "chain_count": len({atom.chain for atom in atoms}),
        "carboxylate_present": carboxylate_present(atoms),
    }


def reference_reproduces_parent(scores: dict[str, object]) -> bool:
    """Return whether reference score matches known parent C/D behavior."""
    c_ok = abs(float(scores["observed_C_d_A"]) - EXPECTED_PARENT_C_A) <= REFERENCE_TOLERANCE_A
    d_ok = abs(float(scores["observed_D_d_A"]) - EXPECTED_PARENT_D_A) <= REFERENCE_TOLERANCE_A
    return c_ok and d_ok


def score_row(
    spec: ParentDerivedRiseSpec,
    path: Path,
    parent_reference_rise_metric_A: float,
    reference_ok: bool,
) -> dict[str, object]:
    """Return one bridge score row."""
    _, atoms = parse_pdb_atom_lines(path)
    scores = score_pdb_abcd(path)
    errors = {band: float(scores[f"{band}_error_A"]) for band in TARGETS_A}
    requested = spec.requested_nominal_rise_A
    realized = parent_reference_rise_metric_A * spec.axial_scale
    row = {
        "variant_id": spec.variant_id,
        "requested_nominal_rise_A": requested if requested is not None else "",
        "axial_scale": spec.axial_scale,
        "realized_rise_metric_A": realized,
        "parent_reference_rise_metric_A": parent_reference_rise_metric_A,
        "coordinate_path": str(path),
        "atom_count": len(atoms),
        "residue_count": len(residue_keys(atoms)),
        "chain_count": len({atom.chain for atom in atoms}),
        "carboxylate_present": carboxylate_present(atoms),
        **scores,
        "combined_CD_abs_error_A": abs(errors["C"]) + abs(errors["D"]),
        "combined_ABCD_abs_error_A": sum(abs(value) for value in errors.values()),
        "reference_reproduces_parent": reference_ok,
        "status": "scored" if reference_ok else "blocked_reference_not_reproduced",
        "notes": (
            "parent-derived reference copy; no coordinate change"
            if spec.variant_id == "parent_derived_reference"
            else "parent-derived axial layer-center scaling; parent atom/residue content preserved"
        ),
    }
    return row


def required_score_columns() -> list[str]:
    """Return required output score columns."""
    return [
        "variant_id",
        "requested_nominal_rise_A",
        "axial_scale",
        "realized_rise_metric_A",
        "parent_reference_rise_metric_A",
        "coordinate_path",
        "atom_count",
        "residue_count",
        "chain_count",
        "carboxylate_present",
        "observed_A_d_A",
        "observed_B_d_A",
        "observed_C_d_A",
        "observed_D_d_A",
        "A_error_A",
        "B_error_A",
        "C_error_A",
        "D_error_A",
        "combined_CD_abs_error_A",
        "combined_ABCD_abs_error_A",
        "reference_reproduces_parent",
        "status",
        "notes",
    ]


def bridge_recommendation(scores: pd.DataFrame) -> str:
    """Classify parent-derived bridge result."""
    if scores.empty:
        return "bridge_blocked_reference_not_reproduced"
    reference = scores[scores["variant_id"] == "parent_derived_reference"]
    if reference.empty or not bool(reference.iloc[0]["reference_reproduces_parent"]):
        return "bridge_blocked_reference_not_reproduced"
    non_reference = scores[scores["variant_id"] != "parent_derived_reference"].copy()
    if non_reference.empty:
        return "bridge_failure"
    best = non_reference.loc[pd.to_numeric(non_reference["combined_CD_abs_error_A"], errors="coerce").idxmin()]
    best_c = abs(float(best["C_error_A"]))
    best_d = abs(float(best["D_error_A"]))
    if best_c <= 0.10 and best_d <= 0.05:
        return "bridge_success"
    if best_c <= 0.20 and best_d <= 0.10:
        return "bridge_partial"
    return "bridge_failure"


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    """Render selected columns as markdown."""
    if df.empty:
        return "_None._"
    columns = [column for column in columns if column in df.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for record in df[columns].itertuples(index=False):
        values = [f"{value:.5g}" if isinstance(value, float) else str(value) for value in record]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_report_text(scores: pd.DataFrame, geometry: pd.DataFrame, parent_pdb: Path, layer_model: LayerModel) -> str:
    """Build parent-derived bridge report."""
    recommendation = bridge_recommendation(scores)
    reference = scores[scores["variant_id"] == "parent_derived_reference"].iloc[0]
    table = markdown_table(
        scores,
        [
            "variant_id",
            "requested_nominal_rise_A",
            "axial_scale",
            "realized_rise_metric_A",
            "observed_A_d_A",
            "observed_B_d_A",
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
    return f"""# Parent-Derived Rise Bridge Summary

## Purpose

This workflow tests nominal 3.40/3.38/3.35 A rise-equivalent changes while starting from the real parent baseline PDB. It preserves parent atom content, residue names, chain IDs, residue numbering, carboxylates, x/y coordinates, and interstrand geometry as much as possible.

## Why The Pseudo-Generator Bridge Was Rejected

The pseudo-generator bridge was rejected because the prior pseudo parametric bridge did not reproduce the parent family: it lacked carboxylates, used pseudo PPI/PPJ residues, changed register/order, and had very different C-alpha rise and interstrand distances. Therefore this branch uses a parent-derived coordinate transform instead.

## Parent And Transform

- Parent PDB: `{parent_pdb}`
- Transform: C-alpha z-gap layer model from `generate_parameterized_rise_variants.py`
- Layer model: `{layer_model.layer_model}`
- Layer count: {len(layer_model.layer_centers)}
- Parent reference rise metric: {mean_layer_rise(layer_model.layer_centers):.4f} A
- Mapping: `axial_scale = requested_nominal_rise_A / 3.40`
- Within-layer atom z offsets are preserved; layer centers are scaled about the parent layer-center mean.

## Reference Reproduction Gate

- Reference C: {float(reference['observed_C_d_A']):.4f} A
- Reference D: {float(reference['observed_D_d_A']):.4f} A
- Expected parent C/D: about {EXPECTED_PARENT_C_A:.3f} / {EXPECTED_PARENT_D_A:.3f} A
- Reference reproduces parent: `{bool(reference['reference_reproduces_parent'])}`

## A/B/C/D Scores

{table}

## Geometry Summary

{geometry_table}

## Recommendation

`{recommendation}`

If the recommendation is not `bridge_success`, do not treat these parent-derived variants as solved physical models. They remain controlled diagnostic coordinate transforms, distinct from both the pseudo reconstructed bridge and the original parent structure.
"""


def run_bridge(
    parent_pdb: Path = DEFAULT_PARENT_PDB,
    outdir: Path = DEFAULT_OUTDIR,
    score_csv: Path = DEFAULT_SCORE_CSV,
    geometry_csv: Path = DEFAULT_GEOMETRY_CSV,
    report_path: Path = DEFAULT_REPORT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate and score parent-derived rise bridge variants."""
    source_lines, atoms = parse_pdb_atom_lines(parent_pdb)
    ca_z_values = [atom.z for atom in atoms if atom.is_ca]
    layer_model = infer_layers_from_ca_z(ca_z_values)
    global_center_z = float(np.mean(layer_model.layer_centers))
    parent_reference_rise_metric_A = mean_layer_rise(layer_model.layer_centers)

    outdir.mkdir(parents=True, exist_ok=True)
    score_csv.parent.mkdir(parents=True, exist_ok=True)
    geometry_csv.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    for spec in bridge_specs():
        path = output_path(outdir, spec)
        write_parent_derived_variant(source_lines, atoms, spec, layer_model, global_center_z, path)
        paths[spec.variant_id] = path

    reference_scores = score_pdb_abcd(paths["parent_derived_reference"])
    reference_ok = reference_reproduces_parent(reference_scores)
    score_rows = [score_row(spec, paths[spec.variant_id], parent_reference_rise_metric_A, reference_ok) for spec in bridge_specs()]
    geometry_rows = [geometry_summary_row(spec, paths[spec.variant_id]) for spec in bridge_specs()]
    scores = pd.DataFrame(score_rows).reindex(columns=required_score_columns())
    geometry = pd.DataFrame(geometry_rows)
    scores.to_csv(score_csv, index=False)
    geometry.to_csv(geometry_csv, index=False)
    report_path.write_text(build_report_text(scores, geometry, parent_pdb, layer_model), encoding="utf-8")
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
    scores, _geometry = run_bridge(args.parent_pdb, args.outdir, args.score_csv, args.geometry_csv, args.report)
    recommendation = bridge_recommendation(scores)
    reference = scores[scores["variant_id"] == "parent_derived_reference"].iloc[0]
    print(f"Generated and scored {len(scores)} parent-derived variants")
    print(f"Reference reproduces parent: {bool(reference['reference_reproduces_parent'])}")
    print(f"Recommendation: {recommendation}")
    print(f"CSV: {args.score_csv}")
    print(f"Geometry CSV: {args.geometry_csv}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
