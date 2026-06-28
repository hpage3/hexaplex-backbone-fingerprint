"""Parser for standard XYZ coordinate files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class XyzAtom:
    """One atom record from a standard XYZ file."""

    atom_index: int
    element: str
    x: float
    y: float
    z: float


def parse_xyz(path: str | Path) -> list[XyzAtom]:
    """Parse a standard XYZ file and validate its declared atom count."""
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()

    if len(lines) < 2:
        raise ValueError(f"Malformed XYZ file {input_path}: expected atom-count and comment lines.")

    try:
        declared_count = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError(f"Malformed XYZ file {input_path}: first line must be an integer atom count.") from exc

    atoms: list[XyzAtom] = []
    for line_number, line in enumerate(lines[2:], start=3):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(
                f"Malformed XYZ file {input_path}: line {line_number} must contain element x y z."
            )
        element, x_text, y_text, z_text = fields
        try:
            atoms.append(
                XyzAtom(
                    atom_index=len(atoms) + 1,
                    element=element,
                    x=float(x_text),
                    y=float(y_text),
                    z=float(z_text),
                )
            )
        except ValueError as exc:
            raise ValueError(
                f"Malformed XYZ file {input_path}: line {line_number} has non-numeric coordinates."
            ) from exc

    if len(atoms) != declared_count:
        raise ValueError(
            f"Malformed XYZ file {input_path}: declared {declared_count} atoms but parsed {len(atoms)}."
        )
    return atoms
