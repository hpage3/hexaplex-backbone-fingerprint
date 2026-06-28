"""Minimal PDB parsing utilities for backbone fingerprint analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .geometry import distance


@dataclass(frozen=True)
class Atom:
    """A parsed PDB atom record."""

    serial: int
    name: str
    altloc: str
    resname: str
    chain: str
    resseq: int
    icode: str
    x: float
    y: float
    z: float
    element: str

    @property
    def coord(self) -> tuple[float, float, float]:
        """Return the atom coordinate as an ``(x, y, z)`` tuple."""
        return (self.x, self.y, self.z)


ResidueKey = tuple[str, int, str]
Residue = dict[str, Atom]
ResidueMap = dict[ResidueKey, Residue]


def parse_pdb(path: str | Path) -> ResidueMap:
    """Parse ATOM/HETATM records from a PDB file into a residue map."""
    resmap: ResidueMap = {}
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            atom = _parse_atom_line(line)
            if atom.altloc not in ("", "A", "1"):
                continue
            key = (atom.chain, atom.resseq, atom.icode)
            resmap.setdefault(key, {})[atom.name] = atom
    return resmap


def residues_by_chain(resmap: ResidueMap) -> dict[str, list[tuple[ResidueKey, Residue]]]:
    """Group residues by chain and sort them by residue number/insertion code."""
    chains: dict[str, list[tuple[ResidueKey, Residue]]] = {}
    for key, residue in resmap.items():
        chains.setdefault(key[0], []).append((key, residue))
    for residues in chains.values():
        residues.sort(key=lambda item: (item[0][1], item[0][2]))
    return chains


def peptide_link_distance(residue_i: Residue, residue_j: Residue) -> float | None:
    """Return the C(i)-N(i+1) peptide-link distance when both atoms exist."""
    c_atom = residue_i.get("C")
    n_atom = residue_j.get("N")
    if c_atom is None or n_atom is None:
        return None
    return distance(c_atom.coord, n_atom.coord)


def is_peptide_linked(
    residue_i: Residue,
    residue_j: Residue,
    max_distance: float = 1.7,
) -> bool:
    """Return true when residues have a plausible C-N peptide linkage."""
    link_distance = peptide_link_distance(residue_i, residue_j)
    return link_distance is not None and link_distance <= max_distance


def _parse_atom_line(line: str) -> Atom:
    """Parse one fixed-width PDB ATOM/HETATM line."""
    return Atom(
        serial=int(line[6:11]),
        name=line[12:16].strip(),
        altloc=line[16].strip(),
        resname=line[17:20].strip(),
        chain=line[21].strip() or "_",
        resseq=int(line[22:26]),
        icode=line[26].strip(),
        x=float(line[30:38]),
        y=float(line[38:46]),
        z=float(line[46:54]),
        element=line[76:78].strip() if len(line) >= 78 else "",
    )


def iter_residue_pairs(
    resmap: ResidueMap,
) -> Iterable[tuple[ResidueKey, Residue, ResidueKey, Residue]]:
    """Yield adjacent residue pairs within each chain."""
    for residues in residues_by_chain(resmap).values():
        for (key_i, residue_i), (key_j, residue_j) in zip(residues, residues[1:]):
            yield key_i, residue_i, key_j, residue_j
