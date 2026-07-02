from pathlib import Path

from scripts.build_ideal_hexaflex_selection_variants import (
    PdbAtomRecord,
    build_variants,
    is_backbone_atom,
    is_carboxylate_atom,
    is_hydrogen,
    is_peptide_plane_atom,
    select_atoms,
)


def atom(atom_name: str, resname: str = "GLU", element: str | None = None) -> PdbAtomRecord:
    element = element or atom_name[0]
    return PdbAtomRecord(
        line=f"ATOM      1 {atom_name:>4s} {resname:>3s} A   1       0.000   0.000   0.000  1.00  0.00          {element:>2s}",
        atom_name=atom_name,
        resname=resname,
        chain="A",
        resseq=1,
        element=element,
    )


def test_atom_selection_classification():
    assert is_hydrogen(atom("H", element="H"))
    assert is_backbone_atom(atom("CA", element="C"))
    assert is_peptide_plane_atom(atom("N", element="N"))
    assert is_carboxylate_atom(atom("OE1", element="O"))
    assert is_carboxylate_atom(atom("OC4", resname="MEP", element="O"))


def test_select_atoms_conservative_groups():
    records = [
        atom("N", element="N"),
        atom("CA", element="C"),
        atom("C", element="C"),
        atom("O", element="O"),
        atom("CB", element="C"),
        atom("OE1", element="O"),
        atom("H", element="H"),
    ]
    backbone, _ = select_atoms(records, "backbone_only")
    side, _ = select_atoms(records, "side_chain_only")
    carboxylate, _ = select_atoms(records, "carboxylate_only")
    no_h, _ = select_atoms(records, "no_h")
    assert [a.atom_name for a in backbone] == ["N", "CA", "C", "O"]
    assert [a.atom_name for a in side] == ["CB", "OE1"]
    assert [a.atom_name for a in carboxylate] == ["OE1"]
    assert len(no_h) == 6


def test_build_variants_skips_empty_carboxylate_selection(tmp_path: Path):
    pdb = tmp_path / "toy.pdb"
    pdb.write_text(
        "\n".join(
            [
                "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N",
                "ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00           C",
                "ATOM      3  C   ALA A   1       2.000   0.000   0.000  1.00  0.00           C",
                "ATOM      4  O   ALA A   1       3.000   0.000   0.000  1.00  0.00           O",
                "END",
            ]
        )
        + "\n",
        encoding="ascii",
    )
    manifest, _ = build_variants(pdb, tmp_path / "variants", parent_label="toy")
    carboxylate = next(row for row in manifest if row["variant"] == "carboxylate_only")
    assert carboxylate["written"] is False
    assert "no atoms matched" in carboxylate["warnings"]
