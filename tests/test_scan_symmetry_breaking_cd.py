from __future__ import annotations

import numpy as np

from hexaplex_backbone_fingerprint.parametric_peptide_plane_models import PlacedAtom
from scripts.scan_symmetry_breaking_cd import (
    perturb_atom,
    radial_delta_pattern,
    score_model,
    z_offset_pattern,
)


def test_radial_delta_three_class_pattern():
    values = [radial_delta_pattern("three_class_ACF", 0.2, index) for index in range(6)]

    assert values == [0.2, 0.0, -0.2, 0.2, 0.0, -0.2]


def test_z_offset_patterns():
    ace_bdf = [z_offset_pattern("ACE_vs_BDF", 0.5, index) for index in range(6)]
    paired = [z_offset_pattern("AB_CD_EF_biased", 0.5, index) for index in range(6)]

    assert ace_bdf == [0.0, 0.5, 0.0, 0.5, 0.0, 0.5]
    assert paired == [0.0, 0.0, 0.5, 0.5, 1.0, 1.0]


def test_perturb_atom_applies_radial_z_and_repeat_terms():
    atom = PlacedAtom(
        serial=1,
        name="CA",
        resname="PPI",
        chain="A",
        resseq=5,
        element="C",
        coord=np.array([8.0, 0.0, 10.0]),
        strand_index=0,
        repeat_index=2,
    )

    shifted = perturb_atom(
        atom,
        radial_pattern="three_class_ACF",
        radial_amplitude_A=0.25,
        z_pattern="AB_CD_EF_biased",
        z_offset_A=0.5,
        repeat_perturb_A=0.1,
    )

    assert np.allclose(shifted.coord, np.array([8.25, 0.0, 10.2]))


def test_score_model_rewards_expected_family_types():
    baseline = score_model(-0.5, 0.1, "all_cross_strand", "all_same_strand")
    rewarded = score_model(-0.5, 0.1, "same_strand_plusminus1_repeat", "all_cross_strand")

    assert rewarded > baseline
