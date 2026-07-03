"""Run a controlled two-class peptide-plane prototype transform.

This prototype tests whether independent three-fold peptide/backbone degrees
of freedom for A/C/E versus B/D/F are worth pursuing. It is not a final
atomistic reconstruction.
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

from scripts.analyze_class_separated_peptide_geometry import chain_geometry_rows, summary_table
from scripts.analyze_threefold_backbone_symmetry import parse_residues
from scripts.generate_global_deformation_variants import PdbAtomLine, format_pdb_coord_line, parse_pdb_atom_lines
from scripts.run_parent_derived_rise_bridge import (
    DEFAULT_PARENT_PDB,
    EXPECTED_PARENT_C_A,
    EXPECTED_PARENT_D_A,
    TARGETS_A,
    carboxylate_present,
    markdown_table,
    score_pdb_abcd,
)


DEFAULT_OUTDIR = Path("outputs/coordinates/two_class_peptide_plane_prototype")
DEFAULT_SCORE_CSV = Path("outputs/metrics/two_class_peptide_plane_prototype_scores.csv")
DEFAULT_GEOMETRY_CSV = Path("outputs/metrics/two_class_peptide_plane_prototype_geometry.csv")
DEFAULT_REPORT = Path("outputs/reports/two_class_peptide_plane_prototype_report.md")

TRIKETO_CHAINS = {"A", "C", "E"}
TRIAMINO_CHAINS = {"B", "D", "F"}
PERTURBATION_DEGREES = [-4.0, -2.0, 0.0, 2.0, 4.0]
PERTURBED_ATOM_NAMES = {"N", "C", "O"}
DIAGNOSTIC_C_A = 5.6422
DIAGNOSTIC_D_A = 7.2756


@dataclass(frozen=True)
class PrototypeSpec:
    """One class-separated peptide-plane perturbation."""

    variant_id: str
    triamino_tilt_deg: float
    triketo_tilt_deg: float


def class_for_chain(chain: str) -> str:
    """Return the fixed class assignment from the three-fold diagnostic."""
    if chain in TRIKETO_CHAINS:
        return "triketo_cyanuric_like"
    if chain in TRIAMINO_CHAINS:
        return "triamino_melamine_like"
    return "unclassified"


def angle_token(value: float) -> str:
    """Return stable filename token for an angle."""
    if abs(value) < 1e-12:
        return "0"
    prefix = "p" if value > 0 else "m"
    return f"{prefix}{abs(int(value))}"


def variant_id_for_angles(triamino_tilt_deg: float, triketo_tilt_deg: float) -> str:
    """Return stable prototype variant ID."""
    return f"two_class_tri{angle_token(triamino_tilt_deg)}_cy{angle_token(triketo_tilt_deg)}"


def prototype_specs(values: list[float] | None = None) -> list[PrototypeSpec]:
    """Return small two-class perturbation grid."""
    grid = PERTURBATION_DEGREES if values is None else values
    return [
        PrototypeSpec(variant_id_for_angles(tri, cy), float(tri), float(cy))
        for tri in grid
        for cy in grid
    ]


def rotation_matrix(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Return Rodrigues rotation matrix."""
    axis = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        raise ValueError("Cannot rotate around a zero-length axis.")
    x, y, z = axis / norm
    angle = math.radians(angle_deg)
    c = math.cos(angle)
    s = math.sin(angle)
    cc = 1.0 - c
    return np.array(
        [
            [c + x * x * cc, x * y * cc - z * s, x * z * cc + y * s],
            [y * x * cc + z * s, c + y * y * cc, y * z * cc - x * s],
            [z * x * cc - y * s, z * y * cc + x * s, c + z * z * cc],
        ],
        dtype=float,
    )


def residue_ca_map(atoms: list[PdbAtomLine]) -> dict[tuple[str, str], np.ndarray]:
    """Return residue C-alpha coordinates keyed by chain/resseq."""
    return {(atom.chain, atom.resseq): atom.coord for atom in atoms if atom.atom_name == "CA"}


def chain_radial_axes(atoms: list[PdbAtomLine]) -> dict[str, np.ndarray]:
    """Return per-chain radial XY axes from C-alpha centroids."""
    ca_atoms = [atom for atom in atoms if atom.atom_name == "CA"]
    if not ca_atoms:
        raise ValueError("No C-alpha atoms found for radial-axis construction.")
    center_xy = np.array([atom.coord for atom in ca_atoms], dtype=float)[:, :2].mean(axis=0)
    axes: dict[str, np.ndarray] = {}
    for chain in sorted({atom.chain for atom in ca_atoms}):
        coords = np.array([atom.coord for atom in ca_atoms if atom.chain == chain], dtype=float)
        radial_xy = coords[:, :2].mean(axis=0) - center_xy
        if np.linalg.norm(radial_xy) <= 1e-12:
            axes[chain] = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            axis = np.array([radial_xy[0], radial_xy[1], 0.0], dtype=float)
            axes[chain] = axis / np.linalg.norm(axis)
    return axes


def should_perturb_atom(atom: PdbAtomLine) -> bool:
    """Return whether this conservative prototype moves the atom."""
    return class_for_chain(atom.chain) != "unclassified" and atom.atom_name in PERTURBED_ATOM_NAMES


def transformed_atom_coord(
    atom: PdbAtomLine,
    spec: PrototypeSpec,
    ca_by_residue: dict[tuple[str, str], np.ndarray],
    axes_by_chain: dict[str, np.ndarray],
) -> np.ndarray:
    """Return transformed coordinate for one atom."""
    if not should_perturb_atom(atom):
        return atom.coord
    anchor = ca_by_residue.get((atom.chain, atom.resseq))
    if anchor is None:
        return atom.coord
    tilt = spec.triketo_tilt_deg if atom.chain in TRIKETO_CHAINS else spec.triamino_tilt_deg
    if abs(tilt) <= 1e-12:
        return atom.coord
    rot = rotation_matrix(axes_by_chain[atom.chain], tilt)
    return anchor + rot @ (atom.coord - anchor)


def write_prototype_variant(source_lines: list[str], atoms: list[PdbAtomLine], spec: PrototypeSpec, out_path: Path) -> None:
    """Write one prototype PDB variant preserving identity fields."""
    ca_by_residue = residue_ca_map(atoms)
    axes_by_chain = chain_radial_axes(atoms)
    out_lines = list(source_lines)
    for atom in atoms:
        coord = transformed_atom_coord(atom, spec, ca_by_residue, axes_by_chain)
        out_lines[atom.index] = format_pdb_coord_line(atom.line, coord)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def output_path(outdir: Path, spec: PrototypeSpec) -> Path:
    """Return coordinate output path."""
    return outdir / f"{spec.variant_id}.pdb"


def atom_identity(atom: PdbAtomLine) -> tuple[str, str, str, str]:
    """Return atom identity tuple preserved by this prototype."""
    return atom.chain, atom.resseq, atom.resname, atom.atom_name


def identity_preserved(parent_atoms: list[PdbAtomLine], variant_atoms: list[PdbAtomLine]) -> bool:
    """Return whether atom identity/order is preserved."""
    return [atom_identity(atom) for atom in parent_atoms] == [atom_identity(atom) for atom in variant_atoms]


def max_coordinate_delta(parent_atoms: list[PdbAtomLine], variant_atoms: list[PdbAtomLine]) -> float:
    """Return max atom displacement between two same-order atom lists."""
    if len(parent_atoms) != len(variant_atoms):
        return float("inf")
    return float(max(np.linalg.norm(a.coord - b.coord) for a, b in zip(parent_atoms, variant_atoms)))


def score_variant(spec: PrototypeSpec, path: Path, parent_atom_count: int) -> dict[str, object]:
    """Score one generated prototype variant."""
    _lines, atoms = parse_pdb_atom_lines(path)
    scores = score_pdb_abcd(path)
    errors = {band: float(scores[f"{band}_error_A"]) for band in TARGETS_A}
    return {
        "variant_id": spec.variant_id,
        "triamino_tilt_deg": spec.triamino_tilt_deg,
        "triketo_tilt_deg": spec.triketo_tilt_deg,
        "coordinate_path": str(path),
        "atom_count": len(atoms),
        "atom_count_changed": len(atoms) != parent_atom_count,
        "carboxylate_present": carboxylate_present(atoms),
        "C_peak_A": scores["observed_C_d_A"],
        "D_peak_A": scores["observed_D_d_A"],
        "C_error_A": scores["C_error_A"],
        "D_error_A": scores["D_error_A"],
        "C_score": scores["C_score"],
        "D_score": scores["D_score"],
        "combined_CD_abs_error_A": abs(errors["C"]) + abs(errors["D"]),
        "combined_ABCD_abs_error_A": sum(abs(value) for value in errors.values()),
        "notes": "no-change/reference prototype" if spec.triamino_tilt_deg == 0 and spec.triketo_tilt_deg == 0 else "controlled class-specific N/C/O rotation around local CA/radial axes",
    }


def geometry_rows_for_variant(spec: PrototypeSpec, path: Path) -> pd.DataFrame:
    """Return class-separated geometry rows for one variant."""
    summary = summary_table(spec.variant_id, chain_geometry_rows(parse_residues(path)))
    summary.insert(1, "variant_id", spec.variant_id)
    summary.insert(2, "triamino_tilt_deg", spec.triamino_tilt_deg)
    summary.insert(3, "triketo_tilt_deg", spec.triketo_tilt_deg)
    return summary


def best_score_rows(scores: pd.DataFrame, tolerance: float = 1e-9) -> pd.DataFrame:
    """Return all tied best rows by combined C/D error."""
    values = pd.to_numeric(scores["combined_CD_abs_error_A"], errors="coerce")
    best = values.min()
    return scores[values.sub(best).abs() <= tolerance].copy()


def plateau_text(rows: pd.DataFrame) -> str:
    """Return compact plateau text."""
    if rows.empty:
        return "not observed"
    if len(rows) == 1:
        return str(rows.iloc[0]["variant_id"])
    return f"{rows.iloc[0]['variant_id']} through {rows.iloc[-1]['variant_id']}"


def reference_reproduces_parent(scores: pd.DataFrame, tolerance_A: float = 0.01) -> bool:
    """Return whether no-change prototype reproduces parent C/D baseline."""
    ref = scores[scores["variant_id"] == variant_id_for_angles(0.0, 0.0)]
    if ref.empty:
        return False
    row = ref.iloc[0]
    return abs(float(row["C_peak_A"]) - EXPECTED_PARENT_C_A) <= tolerance_A and abs(float(row["D_peak_A"]) - EXPECTED_PARENT_D_A) <= tolerance_A


def build_report_text(scores: pd.DataFrame, geometry: pd.DataFrame, parent_pdb: Path) -> str:
    """Build markdown report."""
    ref_ok = reference_reproduces_parent(scores)
    ref = scores[scores["variant_id"] == variant_id_for_angles(0.0, 0.0)].iloc[0]
    best_rows = best_score_rows(scores)
    best = best_rows.iloc[0]
    diagnostic_like = scores[
        (pd.to_numeric(scores["C_peak_A"], errors="coerce").sub(DIAGNOSTIC_C_A).abs() <= 0.03)
        & (pd.to_numeric(scores["D_peak_A"], errors="coerce").sub(DIAGNOSTIC_D_A).abs() <= 0.03)
    ]
    score_table = markdown_table(
        scores.sort_values(["combined_CD_abs_error_A", "triamino_tilt_deg", "triketo_tilt_deg"]).head(12),
        [
            "variant_id",
            "triamino_tilt_deg",
            "triketo_tilt_deg",
            "C_peak_A",
            "D_peak_A",
            "combined_CD_abs_error_A",
            "C_score",
            "D_score",
        ],
    )
    best_geometry = geometry[
        (geometry["variant_id"] == best["variant_id"])
        & (geometry["row_type"].isin(["summary", "difference"]))
    ]
    geometry_table = markdown_table(
        best_geometry,
        [
            "group",
            "omega_median_deg",
            "omega_trans_deviation_median_deg",
            "theta_median_deg",
            "ca_rise_median_A",
            "exit_vector_angle_gap_rms_deg",
            "radial_angle_gap_rms_deg",
        ],
    )
    return f"""# Two-Class Peptide-Plane Prototype

## Scope

This is a controlled prototype transform, not a final atomistic reconstruction. The goal is to test whether two separate three-fold peptide/backbone degrees of freedom are worth pursuing for A/C/E versus B/D/F. Success means a plausible direction that improves or preserves diffraction score without worsening geometry; it is not proof of the physical structure.

- Parent/reference PDB: `{parent_pdb}`
- Triketo/cyanuric-like chains: A,C,E
- Triamino/melamine-like chains: B,D,F
- Conservative moved atom set: `N`, `C`, `O`
- Fixed atom set includes C-alpha anchors, carboxylates, residue names, residue order, atom names, and non-coordinate PDB fields.
- Transform: class-specific rotations around local residue C-alpha anchors and per-chain radial axes. A/C/E move together under one three-fold class; B/D/F move together under the other.

## Reference Reproduction

- No-change/reference variant: `{variant_id_for_angles(0.0, 0.0)}`
- Reference C/D: {float(ref['C_peak_A']):.4f} / {float(ref['D_peak_A']):.4f} A
- Expected parent C/D: about {EXPECTED_PARENT_C_A:.4f} / {EXPECTED_PARENT_D_A:.4f} A
- Reference reproduces parent baseline: `{ref_ok}`

## Top C/D Scores

{score_table}

## Best Plateau

- Best combined-error plateau: `{plateau_text(best_rows)}`
- Representative tie-break member: `{best['variant_id']}`
- C/D: {float(best['C_peak_A']):.4f} / {float(best['D_peak_A']):.4f} A
- Combined C/D error: {float(best['combined_CD_abs_error_A']):.4f} A
- Tie-breaking note: a representative row is used only for reporting; it does not imply a unique optimum.

## Geometry For Representative Best Variant

{geometry_table}

## Interpretation Questions

- Does the no-change/reference prototype reproduce the parent baseline? `{ref_ok}`.
- Do any class-specific peptide-plane perturbations preserve D while improving C? Inspect the top-score table for variants with D near {EXPECTED_PARENT_D_A:.4f} A and lower combined error than the reference.
- Do any prototypes approach the diagnostic/fine-scan C/D result? {len(diagnostic_like)} variants are within 0.03 A of C={DIAGNOSTIC_C_A:.4f} A and D={DIAGNOSTIC_D_A:.4f} A under this scoring path.
- Are improvements driven by one class, the other class, or symmetric opposing class perturbations? Compare `triamino_tilt_deg` and `triketo_tilt_deg` among the best plateau rows.
- Do the best-scoring variants worsen omega/theta/exit-vector geometry? Use the geometry table and `outputs/metrics/two_class_peptide_plane_prototype_geometry.csv`; this first prototype keeps C-alpha/register geometry fixed and perturbs only the conservative N/C/O set.
- Does this support continuing toward a two-class atomistic model? Continue only if score changes are meaningful and geometry remains plausible. Flat scores would mean this conservative perturbation is too weak, not that Asem's two-class hypothesis is falsified.
- What should the next model-building step be? If this direction is promising, build a peptide-plane model with explicit class-specific plane normals/exit vectors before attempting any new atomistic reconstruction.

## Output Files

- Scores: `outputs/metrics/two_class_peptide_plane_prototype_scores.csv`
- Geometry: `outputs/metrics/two_class_peptide_plane_prototype_geometry.csv`
- Coordinates: `outputs/coordinates/two_class_peptide_plane_prototype/`
"""


def run_prototype(
    parent_pdb: Path = DEFAULT_PARENT_PDB,
    outdir: Path = DEFAULT_OUTDIR,
    score_csv: Path = DEFAULT_SCORE_CSV,
    geometry_csv: Path = DEFAULT_GEOMETRY_CSV,
    report_path: Path = DEFAULT_REPORT,
    angle_values: list[float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate, score, and report two-class prototype variants."""
    source_lines, atoms = parse_pdb_atom_lines(parent_pdb)
    specs = prototype_specs(angle_values)
    outdir.mkdir(parents=True, exist_ok=True)
    score_csv.parent.mkdir(parents=True, exist_ok=True)
    geometry_csv.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    score_rows = []
    geometry_frames = []
    for spec in specs:
        path = output_path(outdir, spec)
        write_prototype_variant(source_lines, atoms, spec, path)
        score_rows.append(score_variant(spec, path, len(atoms)))
        geometry_frames.append(geometry_rows_for_variant(spec, path))

    scores = pd.DataFrame(score_rows)
    geometry = pd.concat(geometry_frames, ignore_index=True)
    scores.to_csv(score_csv, index=False)
    geometry.to_csv(geometry_csv, index=False)
    report_path.write_text(build_report_text(scores, geometry, parent_pdb), encoding="utf-8")
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
    scores, _geometry = run_prototype(args.parent_pdb, args.outdir, args.score_csv, args.geometry_csv, args.report)
    ref_ok = reference_reproduces_parent(scores)
    best = best_score_rows(scores).iloc[0]
    print(f"Generated and scored {len(scores)} two-class peptide-plane prototypes")
    print(f"Reference reproduces parent: {ref_ok}")
    print(f"Best plateau: {plateau_text(best_score_rows(scores))} C={float(best['C_peak_A']):.4f} D={float(best['D_peak_A']):.4f}")
    print(f"Scores: {args.score_csv}")
    print(f"Geometry: {args.geometry_csv}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
