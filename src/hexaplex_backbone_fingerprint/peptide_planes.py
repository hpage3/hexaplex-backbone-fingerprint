"""Build peptide-plane records from parsed backbone atoms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import fit_plane
from .pdb_parser import Residue, ResidueKey, ResidueMap, is_peptide_linked, iter_residue_pairs


@dataclass(frozen=True)
class PeptidePlane:
    """A best-fit plane spanning atoms around one peptide linkage."""

    chain: str
    res_i: int
    res_j: int
    resname_i: str
    resname_j: str
    center: np.ndarray
    normal: np.ndarray
    rms: float
    c_i: tuple[float, float, float] | None
    o_i: tuple[float, float, float] | None
    n_j: tuple[float, float, float] | None
    ca_i: tuple[float, float, float] | None
    ca_j: tuple[float, float, float] | None
    hn_j: tuple[float, float, float] | None


def build_peptide_planes(resmap: ResidueMap) -> list[PeptidePlane]:
    """Build peptide planes from adjacent linked residues in a residue map."""
    planes: list[PeptidePlane] = []
    for key_i, residue_i, key_j, residue_j in iter_residue_pairs(resmap):
        if key_i[0] != key_j[0] or not is_peptide_linked(residue_i, residue_j):
            continue
        plane = _build_plane_record(key_i, residue_i, key_j, residue_j)
        if plane is not None:
            planes.append(plane)
    return planes


def _build_plane_record(
    key_i: ResidueKey,
    residue_i: Residue,
    key_j: ResidueKey,
    residue_j: Residue,
) -> PeptidePlane | None:
    atom_names = ["CA", "C", "O"]
    atoms = [residue_i.get(name) for name in atom_names]
    atoms.extend([residue_j.get("N"), residue_j.get("CA")])
    hn_atom = residue_j.get("HN") or residue_j.get("H")
    if hn_atom is not None:
        atoms.append(hn_atom)

    required = [residue_i.get("CA"), residue_i.get("C"), residue_i.get("O"), residue_j.get("N"), residue_j.get("CA")]
    if any(atom is None for atom in required):
        return None

    points = np.array([atom.coord for atom in atoms if atom is not None], dtype=float)
    center, normal, rms = fit_plane(points)
    return PeptidePlane(
        chain=key_i[0],
        res_i=key_i[1],
        res_j=key_j[1],
        resname_i=next(iter(residue_i.values())).resname,
        resname_j=next(iter(residue_j.values())).resname,
        center=center,
        normal=normal,
        rms=rms,
        c_i=residue_i["C"].coord,
        o_i=residue_i["O"].coord,
        n_j=residue_j["N"].coord,
        ca_i=residue_i["CA"].coord,
        ca_j=residue_j["CA"].coord,
        hn_j=hn_atom.coord if hn_atom is not None else None,
    )
