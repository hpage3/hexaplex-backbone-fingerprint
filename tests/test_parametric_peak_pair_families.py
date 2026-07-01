from __future__ import annotations

import numpy as np

from scripts.diagnose_parametric_peak_pair_families import ParametricAtom, pair_family, ring_separation


def make_atom(chain: str, resseq: int, atom_name: str = "CA", resname: str = "PPI") -> ParametricAtom:
    return ParametricAtom(
        serial=1,
        atom_name=atom_name,
        resname=resname,
        chain=chain,
        resseq=resseq,
        element="C",
        coord=np.array([0.0, 0.0, 0.0]),
    )


def test_ring_separation_wraps_six_strand_ring():
    assert ring_separation(0, 1) == 1
    assert ring_separation(0, 5) == 1
    assert ring_separation(0, 3) == 3


def test_pair_family_classifies_cross_strand_repeat_offset():
    atom_a = make_atom("A", 1, atom_name="C", resname="PPI")
    atom_b = make_atom("F", 4, atom_name="N", resname="PPJ")

    family = pair_family(atom_a, atom_b)

    assert family["same_strand"] is False
    assert family["strand_pair"] == "A-F"
    assert family["ring_separation"] == 1
    assert family["repeat_offset"] == 1
    assert family["repeat_class"] == "neighboring_repeat"
    assert family["atom_pair_type"] == "PPI:C-PPJ:N"


def test_pair_family_classifies_same_repeat_same_strand():
    atom_a = make_atom("B", 5)
    atom_b = make_atom("B", 6, atom_name="N", resname="PPJ")

    family = pair_family(atom_a, atom_b)

    assert family["same_strand"] is True
    assert family["strand_pair"] == "B-B"
    assert family["ring_separation"] == 0
    assert family["abs_repeat_offset"] == 0
    assert family["repeat_class"] == "same_repeat"
