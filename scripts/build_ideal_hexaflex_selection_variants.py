"""Build conservative coordinate-selection variants from a labeled ideal Hexaflex PDB.

The variants are intended for diagnostic C/D pair-family comparisons. They
preserve original PDB atom/residue/chain labels and skip selections that cannot
be made from explicit atom names.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}
PEPTIDE_PLANE_ATOMS = {"N", "CA", "C", "O"}
CARBOXYLATE_ATOMS_BY_RESNAME = {
    "GLU": {"CD", "OE1", "OE2"},
    "ASP": {"CG", "OD1", "OD2"},
    "MEP": {"OC2", "OC4", "OC6"},
}
GENERIC_CARBOXYLATE_ATOMS = {"OE1", "OE2", "OD1", "OD2", "OXT", "OC2", "OC4", "OC6"}
VARIANTS = [
    "full",
    "no_h",
    "backbone_only",
    "peptide_plane_only",
    "no_side_chain",
    "side_chain_only",
    "carboxylate_only",
    "backbone_plus_carboxylate",
    "peptide_plane_plus_carboxylate",
]


@dataclass(frozen=True)
class PdbAtomRecord:
    """Parsed PDB atom record with its original line preserved."""

    line: str
    atom_name: str
    resname: str
    chain: str
    resseq: int
    element: str


def safe_label(text: str) -> str:
    """Return a filesystem-safe label."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def parse_pdb_atom_records(path: Path) -> list[PdbAtomRecord]:
    """Parse ATOM/HETATM records from a PDB file."""
    records: list[PdbAtomRecord] = []
    for line in path.read_text(encoding="ascii").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        atom_name = line[12:16].strip()
        resname = line[17:20].strip()
        chain = line[21].strip()
        resseq = int(line[22:26])
        element = (line[76:78].strip() or atom_name[:1]).upper()
        records.append(
            PdbAtomRecord(
                line=line,
                atom_name=atom_name,
                resname=resname,
                chain=chain,
                resseq=resseq,
                element=element,
            )
        )
    if not records:
        raise ValueError(f"No ATOM/HETATM records found in {path}.")
    return records


def is_hydrogen(atom: PdbAtomRecord) -> bool:
    """Return whether an atom is hydrogen by element or atom-name convention."""
    return atom.element.upper() == "H" or atom.atom_name.upper().startswith("H")


def is_backbone_atom(atom: PdbAtomRecord) -> bool:
    """Return whether an atom belongs to the conservative peptide backbone set."""
    return atom.atom_name.upper() in BACKBONE_ATOMS


def is_peptide_plane_atom(atom: PdbAtomRecord) -> bool:
    """Return whether an atom belongs to the conservative peptide-plane heavy set."""
    return atom.atom_name.upper() in PEPTIDE_PLANE_ATOMS


def is_carboxylate_atom(atom: PdbAtomRecord) -> bool:
    """Return whether an atom is an explicitly named carboxylate atom."""
    atom_name = atom.atom_name.upper()
    resname = atom.resname.upper()
    return atom_name in CARBOXYLATE_ATOMS_BY_RESNAME.get(resname, set()) or atom_name in GENERIC_CARBOXYLATE_ATOMS


def is_side_chain_atom(atom: PdbAtomRecord) -> bool:
    """Return whether an atom is a non-backbone heavy atom."""
    return (not is_hydrogen(atom)) and (not is_backbone_atom(atom))


def select_atoms(records: list[PdbAtomRecord], variant: str) -> tuple[list[PdbAtomRecord], list[str]]:
    """Select atoms for one named variant and return warnings."""
    warnings: list[str] = []
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}.")

    if variant == "full":
        selected = list(records)
    elif variant == "no_h":
        selected = [atom for atom in records if not is_hydrogen(atom)]
    elif variant == "backbone_only":
        selected = [atom for atom in records if (not is_hydrogen(atom)) and is_backbone_atom(atom)]
    elif variant == "peptide_plane_only":
        selected = [atom for atom in records if (not is_hydrogen(atom)) and is_peptide_plane_atom(atom)]
    elif variant == "no_side_chain":
        selected = [atom for atom in records if (not is_hydrogen(atom)) and is_backbone_atom(atom)]
        warnings.append("Conservative no_side_chain selection equals backbone_only for this labeled PDB.")
    elif variant == "side_chain_only":
        selected = [atom for atom in records if is_side_chain_atom(atom)]
    elif variant == "carboxylate_only":
        selected = [atom for atom in records if (not is_hydrogen(atom)) and is_carboxylate_atom(atom)]
    elif variant == "backbone_plus_carboxylate":
        selected = [
            atom
            for atom in records
            if (not is_hydrogen(atom)) and (is_backbone_atom(atom) or is_carboxylate_atom(atom))
        ]
    elif variant == "peptide_plane_plus_carboxylate":
        selected = [
            atom
            for atom in records
            if (not is_hydrogen(atom)) and (is_peptide_plane_atom(atom) or is_carboxylate_atom(atom))
        ]
    else:
        raise AssertionError("unreachable")

    if not selected:
        warnings.append(f"Skipped {variant}: no atoms matched conservative selection rules.")
    return selected, warnings


def infer_coordinate_type(records: list[PdbAtomRecord]) -> str:
    """Infer a coarse coordinate type from atom-name coverage."""
    heavy = [atom for atom in records if not is_hydrogen(atom)]
    heavy_names = {atom.atom_name.upper() for atom in heavy}
    has_side = any(is_side_chain_atom(atom) for atom in heavy)
    has_carboxylate = any(is_carboxylate_atom(atom) for atom in heavy)
    has_all_backbone = {"N", "CA", "C", "O"}.issubset(heavy_names)
    if has_side and has_carboxylate and has_all_backbone:
        return "full/carboxylate-containing"
    if has_side:
        return "side-chain-containing"
    if has_all_backbone:
        return "backbone-only or no-side-chain"
    return "partial/uncertain"


def inventory_row(path: Path) -> dict[str, object]:
    """Build one coordinate-inventory row for a PDB file."""
    records = parse_pdb_atom_records(path)
    chains = sorted({atom.chain for atom in records if atom.chain})
    residues = sorted({(atom.chain, atom.resseq) for atom in records if atom.chain})
    return {
        "file_path": str(path),
        "file_type": path.suffix.lower().lstrip("."),
        "atom_count": len(records),
        "hydrogens_present": any(is_hydrogen(atom) for atom in records),
        "chain_labels_present": bool(chains),
        "chain_ids": ",".join(chains),
        "residue_repeat_labels_present": bool(residues),
        "residue_count": len(residues),
        "apparent_coordinate_type": infer_coordinate_type(records),
    }


def write_pdb(path: Path, records: list[PdbAtomRecord], source_path: Path, variant: str) -> None:
    """Write selected PDB atom records without modifying their labels."""
    lines = [
        f"REMARK Diagnostic selection variant: {variant}",
        f"REMARK Source PDB: {source_path}",
    ]
    lines.extend(atom.line for atom in records)
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def build_variants(input_path: Path, out_dir: Path, parent_label: str = "ideal_hexaflex") -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Build PDB variants and return manifest plus inventory rows."""
    records = parse_pdb_atom_records(input_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []
    for variant in VARIANTS:
        selected, warnings = select_atoms(records, variant)
        model_id = f"{parent_label}_{variant}_pair_family_cd"
        pdb_path = out_dir / f"{parent_label}_{variant}.pdb"
        if selected:
            write_pdb(pdb_path, selected, input_path, variant)
        manifest_rows.append(
            {
                "parent_label": parent_label,
                "variant": variant,
                "model_id": model_id,
                "pdb_path": str(pdb_path) if selected else "",
                "atom_count": len(selected),
                "heavy_atom_count": sum(1 for atom in selected if not is_hydrogen(atom)),
                "hydrogen_count": sum(1 for atom in selected if is_hydrogen(atom)),
                "written": bool(selected),
                "warnings": "; ".join(warnings),
            }
        )

    manifest_path = out_dir / "variant_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    return manifest_rows, [inventory_row(input_path)]


def markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    """Render rows as a small markdown table."""
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def write_inventory_report(inventory_rows: list[dict[str, object]], manifest_rows: list[dict[str, object]], report_path: Path) -> None:
    """Write the rich-coordinate inventory and selection report."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_cols = [
        "file_path",
        "file_type",
        "atom_count",
        "hydrogens_present",
        "chain_labels_present",
        "chain_ids",
        "residue_repeat_labels_present",
        "residue_count",
        "apparent_coordinate_type",
    ]
    manifest_cols = ["variant", "atom_count", "heavy_atom_count", "hydrogen_count", "written", "warnings"]
    text = f"""# Coordinate Inventory for Rich-Model C/D Diagnostics

This report treats the ideal/full Hexaflex/Hexaplex coordinate file as a controlled parent model for selection, ablation, and pair-family diagnostics. It is not treated as experimental truth.

## Relevant Coordinate Files

{markdown_table(inventory_rows, inventory_cols)}

## Generated Selection Variants

{markdown_table(manifest_rows, manifest_cols)}

## Selection Rules

- `full`: all labeled PDB atoms from the parent.
- `no_h`: all non-hydrogen atoms.
- `backbone_only`: heavy atoms named `N`, `CA`, `C`, `O`, or `OXT`.
- `peptide_plane_only`: heavy atoms named `N`, `CA`, `C`, or `O`.
- `no_side_chain`: conservative backbone-only proxy, because side-chain removal is defined only by atom names here.
- `side_chain_only`: heavy non-backbone atoms.
- `carboxylate_only`: explicitly named carboxylate atoms/groups (`OE1/OE2`, `OD1/OD2`, `OXT`, and `OC2/OC4/OC6`; GLU `CD` and ASP `CG` when present).
- Combination variants preserve any atom selected by either component rule.

Variants with no matching atoms are skipped rather than guessed.
"""
    report_path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/coordinates/ideal_hexaflex_variants"))
    parser.add_argument("--parent-label", default="ideal_hexaflex")
    parser.add_argument(
        "--inventory-report",
        type=Path,
        default=Path("outputs/reports/coordinate_inventory_for_rich_model_cd.md"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_rows, inventory_rows = build_variants(args.input, args.out_dir, parent_label=safe_label(args.parent_label))
    write_inventory_report(inventory_rows, manifest_rows, args.inventory_report)
    print(f"Wrote {sum(1 for row in manifest_rows if row['written'])} variants to {args.out_dir}")
    print(f"Manifest: {args.out_dir / 'variant_manifest.csv'}")
    print(f"Inventory report: {args.inventory_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
