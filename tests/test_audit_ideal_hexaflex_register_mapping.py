import numpy as np

from scripts.audit_ideal_hexaflex_register_mapping import (
    AuditAtom,
    alternating_interface_group,
    audit_pair_rows,
    build_audit_atoms,
    classify_delta,
    interface_label,
    register_classes,
    repeat_index_maps,
    unique_residues_in_coordinate_order,
)
from scripts.analyze_ideal_hexaflex_atom_class_cd import RichAtom


def test_unique_residues_in_coordinate_order():
    rows = [
        (1, "N", "GLU", "N", "A", 10, np.zeros(3)),
        (2, "CA", "GLU", "C", "A", 10, np.zeros(3)),
        (3, "N", "GLU", "N", "A", 4, np.zeros(3)),
        (4, "N", "GLU", "N", "B", 7, np.zeros(3)),
    ]
    assert unique_residues_in_coordinate_order(rows) == {"A": [10, 4], "B": [7]}


def test_repeat_and_reversed_repeat_index_assignment():
    maps = repeat_index_maps({"A": [1, 2, 3], "B": [1, 2, 3]})
    assert maps["A"][1] == (0, 2, 0)
    assert maps["A"][3] == (2, 0, 2)
    assert maps["B"][1] == (0, 2, 2)
    assert maps["B"][3] == (2, 0, 0)


def audit_atom(chain, strand_index, resseq, coord_index, rev_index, anti_index, atom_name="N", element="N"):
    return AuditAtom(
        atom_index=1,
        atom_name=atom_name,
        resname="GLU",
        element=element,
        chain=chain,
        strand_index=strand_index,
        resseq=resseq,
        coord=np.zeros(3),
        coordinate_repeat_index=coord_index,
        reversed_repeat_index=rev_index,
        antiparallel_repeat_index=anti_index,
    )


def test_register_offset_classification_and_scheme_comparison():
    a = audit_atom("A", 0, 1, 0, 4, 0)
    b = audit_atom("B", 1, 5, 0, 4, 4)
    regs = register_classes(a, b)
    assert regs["raw_register_offset_class"] == "plusminus3_or_more"
    assert regs["coordinate_order_register_offset_class"] == "same"
    assert regs["antiparallel_register_offset_class"] == "plusminus3_or_more"
    assert classify_delta(2) == "plusminus2"
    assert classify_delta(3) == "plusminus3_or_more"


def test_interface_label_wraparound_fa_and_alternating_group():
    a = RichAtom(1, "N", "GLU", "N", "A", 0, 0, np.zeros(3))
    f = RichAtom(2, "C", "GLU", "C", "F", 5, 0, np.zeros(3))
    assert interface_label(a, f) == "FA"
    assert alternating_interface_group("FA") == "BC_DE_FA"


def test_anti_parallel_fixture_would_misclassify_without_coordinate_normalization(tmp_path):
    pdb = tmp_path / "anti.pdb"
    pdb.write_text(
        "\n".join(
            [
                "ATOM      1  N   GLU A   1       0.000   0.000   0.000  1.00  0.00           N",
                "ATOM      2  CA  GLU A   1       0.100   0.000   0.000  1.00  0.00           C",
                "ATOM      3  N   GLU B   5       5.600   0.000   0.000  1.00  0.00           N",
                "ATOM      4  CA  GLU B   5       5.700   0.000   0.000  1.00  0.00           C",
                "END",
            ]
        )
        + "\n",
        encoding="ascii",
    )
    atoms = build_audit_atoms(pdb)
    _, aggregate = audit_pair_rows(atoms, (5.4, 5.8), (7.0, 7.5), sample_per_window=10)
    row = aggregate[aggregate["atom_class_pair"] == "backbone x backbone"].iloc[0]
    assert row["current_register_class"] == "plusminus2"
    assert row["coordinate_order_register_class"] == "same"
