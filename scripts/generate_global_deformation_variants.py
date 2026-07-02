"""Generate small one-at-a-time global deformation variants of ideal Hexaflex."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_SOURCE_PDB = Path("outputs/coordinates/ideal_hexaflex_variants/ideal_hexaflex_backbone_plus_carboxylate.pdb")
DEFAULT_OUTDIR = Path("outputs/coordinates/global_deformation_variants")
DEFAULT_MANIFEST = Path("outputs/metrics/global_deformation_variant_manifest.csv")
DEFAULT_REPORT = Path("outputs/reports/global_deformation_variant_generation.md")


@dataclass(frozen=True)
class PdbAtomLine:
    """ATOM/HETATM line plus parsed coordinate and identity fields."""

    line: str
    index: int
    record: str
    atom_name: str
    resname: str
    chain: str
    resseq: str
    x: float
    y: float
    z: float

    @property
    def coord(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

    @property
    def is_ca(self) -> bool:
        return self.atom_name == "CA"


@dataclass(frozen=True)
class VariantSpec:
    """One global deformation variant specification."""

    variant_id: str
    deformation_mode: str
    radial_scale_xy: float = 1.0
    axial_scale_z: float = 1.0
    twist_total_deg: float = 0.0
    x_scale: float = 1.0
    y_scale: float = 1.0


def parse_pdb_atom_lines(path: Path) -> tuple[list[str], list[PdbAtomLine]]:
    """Parse ATOM/HETATM records while preserving original line order."""
    lines = path.read_text(encoding="utf-8").splitlines()
    atoms: list[PdbAtomLine] = []
    for index, line in enumerate(lines):
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        atoms.append(
            PdbAtomLine(
                line=line,
                index=index,
                record=line[0:6].strip(),
                atom_name=line[12:16].strip(),
                resname=line[17:20].strip(),
                chain=line[21:22].strip(),
                resseq=line[22:26].strip(),
                x=float(line[30:38]),
                y=float(line[38:46]),
                z=float(line[46:54]),
            )
        )
    if not atoms:
        raise ValueError(f"No ATOM/HETATM records found in {path}.")
    return lines, atoms


def format_pdb_coord_line(line: str, coord: np.ndarray) -> str:
    """Return PDB line with coordinate columns replaced."""
    padded = line.rstrip("\n")
    if len(padded) < 54:
        padded = padded.ljust(54)
    return f"{padded[:30]}{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}{padded[54:]}"


def center_from_atoms(atoms: list[PdbAtomLine]) -> tuple[np.ndarray, str]:
    """Return coordinate center, preferring C-alpha atoms when available."""
    ca_atoms = [atom for atom in atoms if atom.is_ca]
    chosen = ca_atoms if ca_atoms else atoms
    label = "C-alpha atoms" if ca_atoms else "all atoms"
    coords = np.array([atom.coord for atom in chosen], dtype=float)
    return coords.mean(axis=0), label


def z_bounds(atoms: list[PdbAtomLine]) -> tuple[float, float]:
    """Return min/max z coordinate across parsed atoms."""
    z_values = [atom.z for atom in atoms]
    return min(z_values), max(z_values)


def apply_deformation(coord: np.ndarray, spec: VariantSpec, center: np.ndarray, z_min: float, z_max: float) -> np.ndarray:
    """Apply one global deformation to one coordinate."""
    shifted = coord - center
    out = coord.astype(float).copy()
    if spec.deformation_mode == "radial_scale_xy":
        out[0] = center[0] + spec.radial_scale_xy * shifted[0]
        out[1] = center[1] + spec.radial_scale_xy * shifted[1]
    elif spec.deformation_mode == "axial_scale_z":
        out[2] = center[2] + spec.axial_scale_z * shifted[2]
    elif spec.deformation_mode == "twist_about_z":
        if abs(z_max - z_min) <= 1e-12:
            angle_deg = 0.0
        else:
            t = (coord[2] - z_min) / (z_max - z_min)
            angle_deg = spec.twist_total_deg * (t - 0.5)
        angle = math.radians(angle_deg)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        out[0] = center[0] + cos_a * shifted[0] - sin_a * shifted[1]
        out[1] = center[1] + sin_a * shifted[0] + cos_a * shifted[1]
    elif spec.deformation_mode == "anisotropic_xy":
        out[0] = center[0] + spec.x_scale * shifted[0]
        out[1] = center[1] + spec.y_scale * shifted[1]
    else:
        raise ValueError(f"Unknown deformation mode: {spec.deformation_mode}")
    return out


def variant_grid() -> list[VariantSpec]:
    """Return the small one-at-a-time global deformation pilot grid."""
    return [
        VariantSpec("radial_m1", "radial_scale_xy", radial_scale_xy=0.990),
        VariantSpec("radial_0", "radial_scale_xy", radial_scale_xy=1.000),
        VariantSpec("radial_p1", "radial_scale_xy", radial_scale_xy=1.010),
        VariantSpec("axial_m1", "axial_scale_z", axial_scale_z=0.995),
        VariantSpec("axial_0", "axial_scale_z", axial_scale_z=1.000),
        VariantSpec("axial_p1", "axial_scale_z", axial_scale_z=1.005),
        VariantSpec("twist_m05", "twist_about_z", twist_total_deg=-0.5),
        VariantSpec("twist_0", "twist_about_z", twist_total_deg=0.0),
        VariantSpec("twist_p05", "twist_about_z", twist_total_deg=0.5),
        VariantSpec("anis_xy_p", "anisotropic_xy", x_scale=1.005, y_scale=0.995),
        VariantSpec("anis_xy_0", "anisotropic_xy", x_scale=1.000, y_scale=1.000),
        VariantSpec("anis_xy_m", "anisotropic_xy", x_scale=0.995, y_scale=1.005),
    ]


def write_variant_pdb(
    source_lines: list[str],
    atoms: list[PdbAtomLine],
    spec: VariantSpec,
    center: np.ndarray,
    z_min: float,
    z_max: float,
    out_path: Path,
) -> None:
    """Write one deformed PDB, preserving non-coordinate fields."""
    out_lines = list(source_lines)
    for atom in atoms:
        coord = apply_deformation(atom.coord, spec, center, z_min, z_max)
        out_lines[atom.index] = format_pdb_coord_line(atom.line, coord)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def manifest_row(
    spec: VariantSpec,
    source_pdb: Path,
    output_pdb: Path,
    atom_count: int,
    center: np.ndarray,
    z_min: float,
    z_max: float,
    center_basis: str,
) -> dict[str, object]:
    """Build one manifest row."""
    return {
        "variant_id": spec.variant_id,
        "deformation_mode": spec.deformation_mode,
        "radial_scale_xy": spec.radial_scale_xy,
        "axial_scale_z": spec.axial_scale_z,
        "twist_total_deg": spec.twist_total_deg,
        "x_scale": spec.x_scale,
        "y_scale": spec.y_scale,
        "source_pdb": str(source_pdb),
        "output_pdb": str(output_pdb),
        "atom_count": atom_count,
        "center_x": center[0],
        "center_y": center[1],
        "center_z": center[2],
        "z_min": z_min,
        "z_max": z_max,
        "status": "ok",
        "notes": f"center_basis={center_basis}; one_at_a_time_global_deformation",
    }


def generate_variants(source_pdb: Path, outdir: Path, manifest_path: Path, report_path: Path) -> pd.DataFrame:
    """Generate global deformation variants and write manifest/report."""
    source_lines, atoms = parse_pdb_atom_lines(source_pdb)
    center, center_basis = center_from_atoms(atoms)
    z_min, z_max = z_bounds(atoms)
    rows = []
    for spec in variant_grid():
        out_path = outdir / f"{spec.variant_id}.pdb"
        write_variant_pdb(source_lines, atoms, spec, center, z_min, z_max, out_path)
        rows.append(manifest_row(spec, source_pdb, out_path, len(atoms), center, z_min, z_max, center_basis))

    manifest = pd.DataFrame(rows)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    report_path.write_text(build_report_text(manifest, source_pdb, center_basis), encoding="utf-8")
    return manifest


def build_report_text(manifest: pd.DataFrame, source_pdb: Path, center_basis: str) -> str:
    """Build generation report markdown."""
    mode_counts = manifest["deformation_mode"].value_counts().sort_index()
    mode_lines = "\n".join(f"- {mode}: {count}" for mode, count in mode_counts.items())
    return f"""# Global Deformation Variant Generation

This is a small controlled global deformation diagnostic after the local C-alpha anchored torsion basin showed robust C/D peak positions. These are not energy-minimized structural models.

- Source PDB: `{source_pdb}`
- Variants generated: {len(manifest)}
- Center basis: {center_basis}
- Coordinate output directory: `outputs/coordinates/global_deformation_variants`
- Manifest: `outputs/metrics/global_deformation_variant_manifest.csv`

## Deformation Modes

{mode_lines}

## Notes

- Modes are one-at-a-time only; this is not a combinatorial scan.
- Atom identity, residue identity, chain IDs, and non-coordinate PDB fields are preserved where practical.
- Every generated variant should be geometry-audited before any future C/D diffraction scoring.
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
    print(f"Generated {len(manifest)} global deformation variants")
    print(f"Coordinate directory: {args.outdir}")
    print(f"Manifest: {args.manifest}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
