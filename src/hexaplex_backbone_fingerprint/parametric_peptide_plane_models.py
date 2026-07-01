"""Parametric six-strand models built from idealized peptide-plane motifs.

Coordinate convention
---------------------
The canonical motif lies in the local XY plane with its plane normal along +Z.
It is intentionally minimal rather than fully chemical: residue ``PPI`` carries
``CA, C, O`` and the following residue ``PPJ`` carries ``N, CA, H``.  The C-N
distance is peptide-like so the existing backbone-plane parser can recover one
peptide plane per repeat.

For placement, the global helical axis is +Z.  Each repeat center lies on a
helix of radius ``helix_radius_A``.  The peptide-plane normal is defined by two
angles: ``plane_normal_to_axis_deg`` is the angle from +Z, and
``plane_azimuth_deg`` sets the direction of the normal projection in the local
radial/tangential plane.  ``in_plane_spin_deg`` rotates the motif within the
plane after the normal is chosen.  ``uniform_adjacent_z_offset_A`` adds an
axial strand-register stagger: strand ``s`` receives ``s * offset`` along +Z.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import angle_between_vectors, normalize


CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


@dataclass(frozen=True)
class MotifAtom:
    """One atom in the canonical local motif."""

    name: str
    residue_offset: int
    resname: str
    element: str
    coord: np.ndarray


@dataclass(frozen=True)
class PlacedAtom:
    """One atom after placement into the global assembly."""

    serial: int
    name: str
    resname: str
    chain: str
    resseq: int
    element: str
    coord: np.ndarray
    strand_index: int
    repeat_index: int


@dataclass(frozen=True)
class ModelParameters:
    """Parameters defining one parametric peptide-plane assembly."""

    n_strands: int = 6
    repeats_per_strand: int = 16
    helix_radius_A: float = 8.0
    twist_deg: float = 30.0
    rise_A: float = 3.38
    plane_normal_to_axis_deg: float = 60.0
    plane_azimuth_deg: float = 0.0
    in_plane_spin_deg: float = 0.0
    handedness: str = "right"
    uniform_adjacent_z_offset_A: float = 0.0
    z_offset_mode: str = "uniform_adjacent"
    strand_z_offset_A: float | None = None
    strand_phase_offset_deg: float = 0.0

    def __post_init__(self) -> None:
        """Keep the older strand_z_offset_A name as an alias for uniform z-offset."""
        if self.z_offset_mode not in {"uniform_adjacent", "alternating"}:
            raise ValueError("z_offset_mode must be 'uniform_adjacent' or 'alternating'.")
        if self.strand_z_offset_A is not None:
            object.__setattr__(self, "uniform_adjacent_z_offset_A", float(self.strand_z_offset_A))
        object.__setattr__(self, "strand_z_offset_A", float(self.uniform_adjacent_z_offset_A))

    @property
    def model_label(self) -> str:
        """Return a filename-safe label encoding the key parameters."""
        z_part = ""
        if self.uniform_adjacent_z_offset_A != 0.0 or self.z_offset_mode != "uniform_adjacent":
            z_part = f"_zoff{format_param(self.uniform_adjacent_z_offset_A)}"
            if self.z_offset_mode != "uniform_adjacent":
                z_part += f"_{self.z_offset_mode}"
        return (
            f"{self.n_strands}strand_tw{format_param(self.twist_deg)}"
            f"_rise{format_param(self.rise_A)}"
            f"_rad{format_param(self.helix_radius_A)}"
            f"_norm{format_param(self.plane_normal_to_axis_deg)}"
            f"_az{format_param(self.plane_azimuth_deg)}"
            f"_spin{format_param(self.in_plane_spin_deg)}"
            f"{z_part}"
            f"_rep{self.repeats_per_strand}_{self.handedness}"
        )


def format_param(value: float) -> str:
    """Format a numeric parameter for compact filenames."""
    return f"{value:.2f}".rstrip("0").rstrip(".").replace(".", "p").replace("-", "m")


def canonical_peptide_plane_motif(include_hn: bool = True) -> list[MotifAtom]:
    """Return the canonical peptide-plane-like motif centered at its centroid.

    The local coordinates are in Angstrom.  The motif is planar in z=0 and
    uses a C-N separation of about 1.33 A so the parser recognizes a peptide
    link between the two pseudo-residues.
    """
    atoms = [
        MotifAtom("CA", 0, "PPI", "C", np.array([-1.45, -0.55, 0.0])),
        MotifAtom("C", 0, "PPI", "C", np.array([0.00, 0.00, 0.0])),
        MotifAtom("O", 0, "PPI", "O", np.array([0.55, 1.10, 0.0])),
        MotifAtom("N", 1, "PPJ", "N", np.array([1.33, -0.05, 0.0])),
        MotifAtom("CA", 1, "PPJ", "C", np.array([2.45, -0.90, 0.0])),
    ]
    if include_hn:
        atoms.append(MotifAtom("H", 1, "PPJ", "H", np.array([1.55, 0.85, 0.0])))
    centroid = np.mean([atom.coord for atom in atoms], axis=0)
    return [
        MotifAtom(atom.name, atom.residue_offset, atom.resname, atom.element, atom.coord - centroid)
        for atom in atoms
    ]


def repeat_center(params: ModelParameters, strand_index: int, repeat_index: int) -> np.ndarray:
    """Return the global center coordinate for one repeat."""
    handedness_sign = 1.0 if params.handedness == "right" else -1.0
    strand_offset_deg = 360.0 * strand_index / params.n_strands
    azimuth_deg = (
        strand_offset_deg
        + params.strand_phase_offset_deg
        + handedness_sign * repeat_index * params.twist_deg
    )
    azimuth = np.radians(azimuth_deg)
    return np.array(
        [
            params.helix_radius_A * np.cos(azimuth),
            params.helix_radius_A * np.sin(azimuth),
            repeat_index * params.rise_A + strand_z_offset(params, strand_index),
        ],
        dtype=float,
    )


def strand_z_offset(params: ModelParameters, strand_index: int) -> float:
    """Return the axial offset assigned to one strand."""
    if params.z_offset_mode == "uniform_adjacent":
        return strand_index * params.uniform_adjacent_z_offset_A
    if params.z_offset_mode == "alternating":
        return (strand_index % 2) * params.uniform_adjacent_z_offset_A
    raise ValueError("z_offset_mode must be 'uniform_adjacent' or 'alternating'.")


def local_radial_tangential(params: ModelParameters, strand_index: int, repeat_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Return local radial and handed tangential unit vectors for a repeat."""
    center = repeat_center(params, strand_index, repeat_index)
    radial = normalize(np.array([center[0], center[1], 0.0], dtype=float))
    tangent_right = np.array([-radial[1], radial[0], 0.0], dtype=float)
    tangent = tangent_right if params.handedness == "right" else -tangent_right
    return radial, tangent


def orientation_frame(params: ModelParameters, strand_index: int, repeat_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return local motif axes ``(u, v, normal)`` in global coordinates."""
    radial, tangent = local_radial_tangential(params, strand_index, repeat_index)
    axis = np.array([0.0, 0.0, 1.0])
    normal_angle = np.radians(params.plane_normal_to_axis_deg)
    normal_azimuth = np.radians(params.plane_azimuth_deg)
    normal = normalize(
        np.cos(normal_angle) * axis
        + np.sin(normal_angle) * (np.cos(normal_azimuth) * radial + np.sin(normal_azimuth) * tangent)
    )

    projected_tangent = tangent - np.dot(tangent, normal) * normal
    if np.linalg.norm(projected_tangent) < 1e-8:
        projected_tangent = radial - np.dot(radial, normal) * normal
    u0 = normalize(projected_tangent)
    v0 = normalize(np.cross(normal, u0))

    spin = np.radians(params.in_plane_spin_deg)
    u = normalize(np.cos(spin) * u0 + np.sin(spin) * v0)
    v = normalize(-np.sin(spin) * u0 + np.cos(spin) * v0)
    return u, v, normal


def generate_model_atoms(params: ModelParameters, include_hn: bool = True) -> list[PlacedAtom]:
    """Generate placed atoms for a parametric model."""
    if params.n_strands > len(CHAIN_IDS):
        raise ValueError(f"n_strands cannot exceed {len(CHAIN_IDS)} with one-character PDB chain IDs.")
    if params.handedness not in {"right", "left"}:
        raise ValueError("handedness must be 'right' or 'left'.")
    motif = canonical_peptide_plane_motif(include_hn=include_hn)
    atoms: list[PlacedAtom] = []
    serial = 1
    for strand_index in range(params.n_strands):
        chain = CHAIN_IDS[strand_index]
        for repeat_index in range(params.repeats_per_strand):
            center = repeat_center(params, strand_index, repeat_index)
            u, v, _normal = orientation_frame(params, strand_index, repeat_index)
            for motif_atom in motif:
                local = motif_atom.coord
                coord = center + local[0] * u + local[1] * v
                atoms.append(
                    PlacedAtom(
                        serial=serial,
                        name=motif_atom.name,
                        resname=motif_atom.resname,
                        chain=chain,
                        resseq=2 * repeat_index + 1 + motif_atom.residue_offset,
                        element=motif_atom.element,
                        coord=coord,
                        strand_index=strand_index,
                        repeat_index=repeat_index,
                    )
                )
                serial += 1
    return atoms


def write_pdb(atoms: list[PlacedAtom], path: str | Path, params: ModelParameters) -> Path:
    """Write placed atoms to a minimal PDB file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "REMARK Parametric six-strand peptide-plane model",
        f"REMARK model_label {params.model_label}",
        f"REMARK n_strands {params.n_strands}",
        f"REMARK repeats_per_strand {params.repeats_per_strand}",
        f"REMARK helix_radius_A {params.helix_radius_A:.3f}",
        f"REMARK twist_deg {params.twist_deg:.3f}",
        f"REMARK rise_A {params.rise_A:.3f}",
        f"REMARK plane_normal_to_axis_deg {params.plane_normal_to_axis_deg:.3f}",
        f"REMARK plane_azimuth_deg {params.plane_azimuth_deg:.3f}",
        f"REMARK in_plane_spin_deg {params.in_plane_spin_deg:.3f}",
        f"REMARK uniform_adjacent_z_offset_A {params.uniform_adjacent_z_offset_A:.3f}",
        f"REMARK z_offset_mode {params.z_offset_mode}",
        f"REMARK handedness {params.handedness}",
    ]
    previous_chain = None
    previous_resseq = None
    for atom in atoms:
        if previous_chain is not None and atom.chain != previous_chain:
            lines.append(f"TER   {atom.serial:5d}      {previous_resseq or 1:>4}")
        lines.append(format_pdb_atom(atom))
        previous_chain = atom.chain
        previous_resseq = atom.resseq
    lines.append("END")
    output_path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return output_path


def format_pdb_atom(atom: PlacedAtom) -> str:
    """Format one ATOM record using fixed-width PDB columns."""
    x, y, z = atom.coord
    atom_name = f"{atom.name:<4}"
    return (
        f"ATOM  {atom.serial:5d} {atom_name} {atom.resname:>3} {atom.chain:1s}"
        f"{atom.resseq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
        f"  1.00  0.00          {atom.element:>2s}"
    )


def write_xyz(atoms: list[PlacedAtom], path: str | Path, comment: str = "") -> Path:
    """Write placed atoms to a simple XYZ file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(len(atoms)), comment]
    for atom in atoms:
        x, y, z = atom.coord
        lines.append(f"{atom.element} {x:.6f} {y:.6f} {z:.6f}")
    output_path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return output_path


def manifest_row(params: ModelParameters, pdb_path: Path, xyz_path: Path | None, atom_count: int) -> dict[str, object]:
    """Return one manifest row for a generated model."""
    return {
        "model_label": params.model_label,
        "pdb_path": str(pdb_path),
        "xyz_path": str(xyz_path) if xyz_path is not None else "",
        "n_strands": params.n_strands,
        "repeats_per_strand": params.repeats_per_strand,
        "helix_radius_A": params.helix_radius_A,
        "twist_deg": params.twist_deg,
        "rise_A": params.rise_A,
        "plane_normal_to_axis_deg": params.plane_normal_to_axis_deg,
        "plane_azimuth_deg": params.plane_azimuth_deg,
        "in_plane_spin_deg": params.in_plane_spin_deg,
        "uniform_adjacent_z_offset_A": params.uniform_adjacent_z_offset_A,
        "z_offset_mode": params.z_offset_mode,
        "handedness": params.handedness,
        "strand_z_offset_A": params.strand_z_offset_A,
        "strand_phase_offset_deg": params.strand_phase_offset_deg,
        "atom_count": atom_count,
    }


def measured_normal_to_axis_angle(params: ModelParameters) -> float:
    """Return the actual angle between a placed motif normal and the global z-axis."""
    _u, _v, normal = orientation_frame(params, 0, 0)
    return angle_between_vectors(normal, np.array([0.0, 0.0, 1.0]))
