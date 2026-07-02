from __future__ import annotations

import numpy as np
import pytest

from scripts.analyze_backbone_pair_family_cd import (
    LabeledAtom,
    classify_pair_families,
    interface_name,
    load_labeled_atoms,
    ring_separation,
)


def atom(chain: str, strand_index: int, repeat_index: int) -> LabeledAtom:
    return LabeledAtom(
        atom_index=1,
        atom_name="CA",
        element="C",
        chain=chain,
        strand_index=strand_index,
        repeat_index=repeat_index,
        coord=np.array([0.0, 0.0, 0.0]),
    )


def test_adjacent_strand_same_register_classification():
    families = classify_pair_families(atom("A", 0, 3), atom("B", 1, 3))

    assert "all_cross_strand" in families
    assert "all_adjacent_cross_strand" in families
    assert "adjacent_strand_same_register" in families
    assert "interface_AB" in families
    assert "alternating_interfaces_AB_CD_EF" in families


def test_wraparound_fa_interface_is_adjacent():
    atom_f = atom("F", 5, 2)
    atom_a = atom("A", 0, 2)

    assert ring_separation(atom_f.strand_index, atom_a.strand_index) == 1
    assert interface_name(atom_f, atom_a) == "interface_FA"
    families = classify_pair_families(atom_f, atom_a)
    assert "interface_FA" in families
    assert "alternating_interfaces_BC_DE_FA" in families


def test_adjacent_plusminus1_register_classification():
    families = classify_pair_families(atom("C", 2, 4), atom("D", 3, 5))

    assert "adjacent_strand_plusminus1_register" in families
    assert "adjacent_strand_same_register" not in families


def test_same_strand_plusminus1_repeat_classification():
    families = classify_pair_families(atom("E", 4, 10), atom("E", 4, 9))

    assert "all_same_strand" in families
    assert "same_strand_plusminus1_repeat" in families
    assert "all_cross_strand" not in families


def test_unlabeled_xyz_requires_mapping(tmp_path):
    xyz = tmp_path / "unlabeled.xyz"
    xyz.write_text("2\nmini\nC 0 0 0\nC 1 0 0\n", encoding="ascii")

    with pytest.raises(ValueError, match="strand/repeat labels"):
        load_labeled_atoms(xyz)
