import numpy as np

from scripts.analyze_best_clean_model_register_interfaces import (
    aggregate_distances,
    alternating_interface_group,
    interface_label,
    register_offset_class,
)
from scripts.analyze_ideal_hexaflex_atom_class_cd import RichAtom


def atom(
    atom_index: int,
    atom_name: str,
    element: str,
    strand_index: int,
    repeat_index: int,
    coord=(0.0, 0.0, 0.0),
) -> RichAtom:
    chain = "ABCDEF"[strand_index]
    return RichAtom(
        atom_index=atom_index,
        atom_name=atom_name,
        resname="GLU",
        element=element,
        chain=chain,
        strand_index=strand_index,
        repeat_index=repeat_index,
        coord=np.array(coord, dtype=float),
    )


def test_interface_label_including_wraparound_fa():
    a = atom(1, "N", "N", 0, 0)
    b = atom(2, "CA", "C", 1, 0)
    f = atom(3, "C", "C", 5, 0)
    d = atom(4, "O", "O", 3, 0)
    assert interface_label(a, b) == "AB"
    assert interface_label(a, f) == "FA"
    assert interface_label(a, d) == "nonadjacent"
    assert interface_label(a, atom(5, "O", "O", 0, 1)) == "same_strand"


def test_alternating_interface_group():
    assert alternating_interface_group("AB") == "AB_CD_EF"
    assert alternating_interface_group("DE") == "BC_DE_FA"
    assert alternating_interface_group("FA") == "BC_DE_FA"
    assert alternating_interface_group("nonadjacent") == "nonadjacent"
    assert alternating_interface_group("same_strand") == "same_strand"


def test_register_offset_class():
    a = atom(1, "N", "N", 0, 4)
    assert register_offset_class(a, atom(2, "CA", "C", 1, 4)) == "same"
    assert register_offset_class(a, atom(3, "CA", "C", 1, 5)) == "plusminus1"
    assert register_offset_class(a, atom(4, "CA", "C", 1, 2)) == "plusminus2"
    assert register_offset_class(a, atom(5, "CA", "C", 1, 8)) == "plusminus3_or_more"
    assert register_offset_class(a, atom(6, "CA", "C", 0, 5)) == "same_strand"


def test_aggregation_fixture_contains_atom_interface_register_keys():
    a = atom(1, "N", "N", 0, 0)
    b = atom(2, "OE1", "O", 1, 1, coord=(5.6, 0.0, 0.0))
    distances = aggregate_distances([a, b])
    key = ("peptide_plane", "carboxylate", "AB", "AB_CD_EF", "plusminus1", "all_cross_strand")
    assert key in distances
    assert distances[key] == [5.6]
