from pathlib import Path

import numpy as np

from scripts.analyze_ideal_hexaflex_atom_class_cd import (
    RichAtom,
    atom_classes,
    canonical_pair_class,
    compute_atom_class_family_distances,
    parse_rich_pdb,
)


def rich_atom(
    atom_index: int,
    atom_name: str,
    resname: str,
    element: str,
    chain: str = "A",
    strand_index: int = 0,
    repeat_index: int = 0,
    coord: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> RichAtom:
    return RichAtom(
        atom_index=atom_index,
        atom_name=atom_name,
        resname=resname,
        element=element,
        chain=chain,
        strand_index=strand_index,
        repeat_index=repeat_index,
        coord=np.array(coord, dtype=float),
    )


def test_atom_classes_are_overlapping_and_conservative():
    peptide_o = rich_atom(1, "O", "GLU", "O")
    carboxylate = rich_atom(2, "OE1", "GLU", "O")
    hydrogen = rich_atom(3, "H", "GLU", "H")
    assert set(atom_classes(peptide_o)) == {"heavy_all", "backbone", "peptide_plane", "oxygen"}
    assert "carboxylate" in atom_classes(carboxylate)
    assert "side_chain" in atom_classes(carboxylate)
    assert atom_classes(hydrogen) == ("H",)


def test_pair_class_canonicalization():
    assert canonical_pair_class("peptide_plane", "carboxylate") == ("peptide_plane", "carboxylate")
    assert canonical_pair_class("carboxylate", "peptide_plane") == ("peptide_plane", "carboxylate")


def test_geometry_and_atom_class_aggregation_counts_wraparound():
    atom_a = rich_atom(1, "N", "GLU", "N", chain="A", strand_index=0, repeat_index=0)
    atom_f = rich_atom(2, "OE1", "GLU", "O", chain="F", strand_index=5, repeat_index=0, coord=(7.2, 0, 0))
    distances = compute_atom_class_family_distances([atom_a, atom_f], n_strands=6)
    key = ("peptide_plane", "carboxylate", "alternating_interfaces_BC_DE_FA")
    assert key in distances
    assert distances[key] == [7.2]
    assert ("heavy_all", "heavy_all", "adjacent_strand_same_register") in distances


def test_parse_rich_pdb_keeps_hydrogens(tmp_path: Path):
    pdb = tmp_path / "toy.pdb"
    pdb.write_text(
        "\n".join(
            [
                "ATOM      1  N   GLU A   1       0.000   0.000   0.000  1.00  0.00           N",
                "ATOM      2  H   GLU A   1       1.000   0.000   0.000  1.00  0.00           H",
                "END",
            ]
        )
        + "\n",
        encoding="ascii",
    )
    atoms = parse_rich_pdb(pdb)
    assert len(atoms) == 2
    assert atoms[1].element == "H"
