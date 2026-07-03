"""Run a controlled two-class axial/theta-omega prototype scan.

This first version tests class-specific axial movement of backbone anchor/path
atoms for A/C/E and B/D/F. Theta-informed rotations and parent-derived global
compression are deferred until their local axes/coupling can be specified
unambiguously. This is not a final atomistic reconstruction.
"""

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


DEFAULT_OUTDIR = Path("outputs/coordinates/two_class_axial_theta_scan")
DEFAULT_SCORE_CSV = Path("outputs/metrics/two_class_axial_theta_scan_scores.csv")
DEFAULT_GEOMETRY_CSV = Path("outputs/metrics/two_class_axial_theta_scan_geometry.csv")
DEFAULT_REPORT = Path("outputs/reports/two_class_axial_theta_scan_report.md")

TRIKETO_CHAINS = {"A", "C", "E"}
TRIAMINO_CHAINS = {"B", "D", "F"}
AXIAL_OFFSETS_A = [-0.04, -0.02, 0.0, 0.02, 0.04]
BACKBONE_ANCHOR_ATOMS = {"N", "CA", "C", "O"}
DIAGNOSTIC_C_A = 5.6422
DIAGNOSTIC_D_A = 7.2756


@dataclass(frozen=True)
class AxialThetaScanSpec:
    """One two-class axial/theta prototype variant."""

    variant_id: str
    triamino_axial_offset_A: float
    triketo_axial_offset_A: float
    global_compression_mode: str = "none"
    theta_rotation_mode: str = "deferred"


def class_for_chain(chain: str) -> str:
    """Return fixed class assignment from the three-fold diagnostic."""
    if chain in TRIKETO_CHAINS:
        return "triketo_cyanuric_like"
    if chain in TRIAMINO_CHAINS:
        return "triamino_melamine_like"
    return "unclassified"


def offset_token(value: float) -> str:
    """Return stable filename token for an axial offset."""
    if abs(value) < 1e-12:
        return "0"
    prefix = "p" if value > 0 else "m"
    return f"{prefix}{int(round(abs(value) * 100)):02d}"


def variant_id_for_offsets(triamino_offset_A: float, triketo_offset_A: float) -> str:
    """Return stable variant ID."""
    return f"two_class_axial_tri{offset_token(triamino_offset_A)}_cy{offset_token(triketo_offset_A)}"


def axial_scan_specs(values: list[float] | None = None) -> list[AxialThetaScanSpec]:
    """Return the modest two-class axial scan grid."""
    grid = AXIAL_OFFSETS_A if values is None else values
    return [
        AxialThetaScanSpec(variant_id_for_offsets(tri, cy), float(tri), float(cy))
        for tri in grid
        for cy in grid
    ]


def should_move_atom(atom: PdbAtomLine) -> bool:
    """Return whether this conservative axial prototype moves the atom."""
    return class_for_chain(atom.chain) != "unclassified" and atom.atom_name in BACKBONE_ANCHOR_ATOMS


def offset_for_atom(atom: PdbAtomLine, spec: AxialThetaScanSpec) -> float:
    """Return axial offset for one atom based on class."""
    if atom.chain in TRIKETO_CHAINS:
        return spec.triketo_axial_offset_A
    if atom.chain in TRIAMINO_CHAINS:
        return spec.triamino_axial_offset_A
    return 0.0


def transformed_atom_coord(atom: PdbAtomLine, spec: AxialThetaScanSpec) -> np.ndarray:
    """Return transformed coordinate for one atom."""
    if not should_move_atom(atom):
        return atom.coord
    offset = offset_for_atom(atom, spec)
    if abs(offset) <= 1e-12:
        return atom.coord
    coord = atom.coord.copy()
    coord[2] += offset
    return coord


def write_axial_theta_variant(source_lines: list[str], atoms: list[PdbAtomLine], spec: AxialThetaScanSpec, out_path: Path) -> None:
    """Write one axial/theta scan variant preserving identity fields."""
    out_lines = list(source_lines)
    for atom in atoms:
        coord = transformed_atom_coord(atom, spec)
        out_lines[atom.index] = format_pdb_coord_line(atom.line, coord)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def output_path(outdir: Path, spec: AxialThetaScanSpec) -> Path:
    """Return coordinate output path."""
    return outdir / f"{spec.variant_id}.pdb"


def atom_identity(atom: PdbAtomLine) -> tuple[str, str, str, str]:
    """Return atom identity tuple preserved by this scan."""
    return atom.chain, atom.resseq, atom.resname, atom.atom_name


def identity_preserved(parent_atoms: list[PdbAtomLine], variant_atoms: list[PdbAtomLine]) -> bool:
    """Return whether atom identity/order is preserved."""
    return [atom_identity(atom) for atom in parent_atoms] == [atom_identity(atom) for atom in variant_atoms]


def max_coordinate_delta(parent_atoms: list[PdbAtomLine], variant_atoms: list[PdbAtomLine]) -> float:
    """Return max atom displacement for same-order atoms."""
    if len(parent_atoms) != len(variant_atoms):
        return float("inf")
    return float(max(np.linalg.norm(a.coord - b.coord) for a, b in zip(parent_atoms, variant_atoms)))


def score_variant(spec: AxialThetaScanSpec, path: Path, parent_atoms: list[PdbAtomLine]) -> dict[str, object]:
    """Score one generated axial/theta scan variant."""
    _lines, atoms = parse_pdb_atom_lines(path)
    scores = score_pdb_abcd(path)
    errors = {band: float(scores[f"{band}_error_A"]) for band in TARGETS_A}
    return {
        "variant_id": spec.variant_id,
        "triamino_axial_offset_A": spec.triamino_axial_offset_A,
        "triketo_axial_offset_A": spec.triketo_axial_offset_A,
        "global_compression_mode": spec.global_compression_mode,
        "theta_rotation_mode": spec.theta_rotation_mode,
        "coordinate_path": str(path),
        "atom_count": len(atoms),
        "atom_count_changed": len(atoms) != len(parent_atoms),
        "identity_preserved": identity_preserved(parent_atoms, atoms),
        "max_atom_displacement_A": max_coordinate_delta(parent_atoms, atoms),
        "carboxylate_present": carboxylate_present(atoms),
        "moved_atom_set": ",".join(sorted(BACKBONE_ANCHOR_ATOMS)),
        "C_peak_A": scores["observed_C_d_A"],
        "D_peak_A": scores["observed_D_d_A"],
        "C_error_A": scores["C_error_A"],
        "D_error_A": scores["D_error_A"],
        "C_score": scores["C_score"],
        "D_score": scores["D_score"],
        "combined_CD_abs_error_A": abs(errors["C"]) + abs(errors["D"]),
        "combined_ABCD_abs_error_A": sum(abs(value) for value in errors.values()),
        "notes": (
            "no-change/reference prototype"
            if spec.triamino_axial_offset_A == 0 and spec.triketo_axial_offset_A == 0
            else "controlled class-specific axial shift of N/CA/C/O backbone-anchor atoms"
        ),
    }


def geometry_rows_for_variant(spec: AxialThetaScanSpec, path: Path) -> pd.DataFrame:
    """Return class-separated geometry rows for one variant."""
    summary = summary_table(spec.variant_id, chain_geometry_rows(parse_residues(path)))
    summary.insert(1, "variant_id", spec.variant_id)
    summary.insert(2, "triamino_axial_offset_A", spec.triamino_axial_offset_A)
    summary.insert(3, "triketo_axial_offset_A", spec.triketo_axial_offset_A)
    summary.insert(4, "global_compression_mode", spec.global_compression_mode)
    summary.insert(5, "theta_rotation_mode", spec.theta_rotation_mode)
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
    ref = scores[scores["variant_id"] == variant_id_for_offsets(0.0, 0.0)]
    if ref.empty:
        return False
    row = ref.iloc[0]
    return abs(float(row["C_peak_A"]) - EXPECTED_PARENT_C_A) <= tolerance_A and abs(float(row["D_peak_A"]) - EXPECTED_PARENT_D_A) <= tolerance_A


def improvement_count(scores: pd.DataFrame) -> int:
    """Return count of variants improving combined C/D error relative to no-change."""
    ref = scores[scores["variant_id"] == variant_id_for_offsets(0.0, 0.0)]
    if ref.empty:
        return 0
    ref_error = float(ref.iloc[0]["combined_CD_abs_error_A"])
    return int((pd.to_numeric(scores["combined_CD_abs_error_A"], errors="coerce") < ref_error - 1e-9).sum())


def build_report_text(scores: pd.DataFrame, geometry: pd.DataFrame, parent_pdb: Path) -> str:
    """Build markdown report."""
    ref_ok = reference_reproduces_parent(scores)
    ref = scores[scores["variant_id"] == variant_id_for_offsets(0.0, 0.0)].iloc[0]
    best_rows = best_score_rows(scores)
    best = best_rows.iloc[0]
    improved = improvement_count(scores)
    diagnostic_like = scores[
        (pd.to_numeric(scores["C_peak_A"], errors="coerce").sub(DIAGNOSTIC_C_A).abs() <= 0.03)
        & (pd.to_numeric(scores["D_peak_A"], errors="coerce").sub(DIAGNOSTIC_D_A).abs() <= 0.03)
    ]
    score_table = markdown_table(
        scores.sort_values(["combined_CD_abs_error_A", "triamino_axial_offset_A", "triketo_axial_offset_A"]).head(12),
        [
            "variant_id",
            "triamino_axial_offset_A",
            "triketo_axial_offset_A",
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
    return f"""# Two-Class Axial/Theta-Omega Prototype Scan

## Scope

This is a controlled axial/theta-omega prototype, not a final atomistic reconstruction. It is motivated by Asem's pNAB symmetry caution and the earlier theta/omega analysis. Prior N/C/O-only and radial-anchor prototypes were too conservative, so this scan tests class-specific axial movement of the backbone path while preserving recognition-core identity as far as this conservative atom classification allows.

- Parent/reference PDB: `{parent_pdb}`
- Triketo/cyanuric-like chains: A,C,E
- Triamino/melamine-like chains: B,D,F
- Moved atom set: `N`, `CA`, `C`, `O`
- Fixed atom set includes carboxylates, residue names, residue order, atom names, and non-coordinate PDB fields.
- Transform: class-specific z-axis offsets for backbone-anchor atoms.
- Global parent-derived compression background: deferred in this first version.
- Theta-informed rotation: deferred in this first version because a robust local-axis definition should be specified before coordinate generation.

Success means finding a plausible direction for a two separate three-fold model, not proving the physical structure.

## Reference Reproduction

- No-change/reference variant: `{variant_id_for_offsets(0.0, 0.0)}`
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

- Does the no-change/reference reproduce parent baseline? `{ref_ok}`.
- Do class-specific axial offsets move C toward {DIAGNOSTIC_C_A:.4f} A? Inspect `C_peak_A` in the score table; {improved} variants improve combined C/D error relative to the no-change parent.
- Can D remain near {EXPECTED_PARENT_D_A:.4f} A? Inspect `D_peak_A` in the score table.
- Does adding or approximating the successful parent-derived rise compression background help? Not tested here; it is deferred so this first scan isolates class-specific axial offsets.
- Do any variants approach the fine-scan diagnostic plateau? {len(diagnostic_like)} variants are within 0.03 A of C={DIAGNOSTIC_C_A:.4f} A and D={DIAGNOSTIC_D_A:.4f} A.
- Are improvements driven by one class, the other class, or opposing class offsets? Compare `triamino_axial_offset_A` and `triketo_axial_offset_A` among the best plateau rows.
- Do improvements worsen theta/omega geometry? Use the geometry table and `outputs/metrics/two_class_axial_theta_scan_geometry.csv`.
- Does this support continuing toward a two-class atomistic model? Continue only if score changes are meaningful and geometry remains plausible. Flat scores would mean this conservative axial prototype is still too weak.

## Output Files

- Scores: `outputs/metrics/two_class_axial_theta_scan_scores.csv`
- Geometry: `outputs/metrics/two_class_axial_theta_scan_geometry.csv`
- Coordinates: `outputs/coordinates/two_class_axial_theta_scan/`
"""


def run_scan(
    parent_pdb: Path = DEFAULT_PARENT_PDB,
    outdir: Path = DEFAULT_OUTDIR,
    score_csv: Path = DEFAULT_SCORE_CSV,
    geometry_csv: Path = DEFAULT_GEOMETRY_CSV,
    report_path: Path = DEFAULT_REPORT,
    offset_values: list[float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate, score, and report two-class axial/theta variants."""
    source_lines, atoms = parse_pdb_atom_lines(parent_pdb)
    specs = axial_scan_specs(offset_values)
    outdir.mkdir(parents=True, exist_ok=True)
    score_csv.parent.mkdir(parents=True, exist_ok=True)
    geometry_csv.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    score_rows = []
    geometry_frames = []
    for spec in specs:
        path = output_path(outdir, spec)
        write_axial_theta_variant(source_lines, atoms, spec, path)
        score_rows.append(score_variant(spec, path, atoms))
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
    scores, _geometry = run_scan(args.parent_pdb, args.outdir, args.score_csv, args.geometry_csv, args.report)
    ref_ok = reference_reproduces_parent(scores)
    best = best_score_rows(scores).iloc[0]
    print(f"Generated and scored {len(scores)} two-class axial/theta scan variants")
    print(f"Reference reproduces parent: {ref_ok}")
    print(f"Best plateau: {plateau_text(best_score_rows(scores))} C={float(best['C_peak_A']):.4f} D={float(best['D_peak_A']):.4f}")
    print(f"Improving variants: {improvement_count(scores)}")
    print(f"Scores: {args.score_csv}")
    print(f"Geometry: {args.geometry_csv}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
