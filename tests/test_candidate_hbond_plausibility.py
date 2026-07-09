from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analyze_candidate_hbond_plausibility import (
    AtomRecord,
    angle_degrees,
    build_report,
    classify_angle,
    classify_distance,
    hbond_pair_rows,
    is_likely_acceptor,
    is_likely_donor,
    missing_summary_row,
    pair_type,
    score_summary_row,
    should_exclude_pair,
)


def atom(
    serial: int,
    atom_name: str,
    element: str,
    coord: tuple[float, float, float],
    chain: str = "A",
    resseq: str = "1",
    resname: str = "GLU",
) -> AtomRecord:
    return AtomRecord(
        serial=serial,
        atom_name=atom_name,
        resname=resname,
        chain=chain,
        resseq=resseq,
        icode="",
        element=element,
        coord=np.array(coord, dtype=float),
    )


def test_donor_acceptor_atom_classification() -> None:
    donor_n = atom(1, "N", "N", (0, 0, 0))
    acceptor_o = atom(2, "OE1", "O", (3, 0, 0))
    hydrogen = atom(3, "H", "H", (1, 0, 0))

    assert is_likely_donor(donor_n)
    assert is_likely_acceptor(acceptor_o)
    assert not is_likely_donor(hydrogen)
    assert not is_likely_acceptor(hydrogen)


def test_heavy_atom_distance_classification_thresholds() -> None:
    assert classify_distance(2.3) == "rejected_too_close"
    assert classify_distance(2.8) == "strong_geometry"
    assert classify_distance(2.5) == "plausible_geometry"
    assert classify_distance(3.5) == "weak_geometry"
    assert classify_distance(3.8) == "rejected_too_far"


def test_explicit_hydrogen_angle_classification() -> None:
    donor = np.array([0.0, 0.0, 0.0])
    hydrogen = np.array([1.0, 0.0, 0.0])
    acceptor_linear = np.array([2.8, 0.0, 0.0])
    acceptor_bent = np.array([1.0, 1.0, 0.0])

    assert angle_degrees(donor, hydrogen, acceptor_linear) == 180.0
    assert classify_angle("plausible_geometry", angle_degrees(donor, hydrogen, acceptor_linear)) == "strong_geometry"
    assert classify_angle("plausible_geometry", angle_degrees(donor, hydrogen, acceptor_bent)) in {"weak_geometry", "rejected_too_far"}


def test_exclusion_of_obvious_same_residue_trivial_pairs() -> None:
    donor = atom(1, "N", "N", (0, 0, 0), chain="A", resseq="1")
    same_res_acceptor = atom(2, "O", "O", (2.8, 0, 0), chain="A", resseq="1")
    next_res_acceptor = atom(3, "O", "O", (2.8, 0, 0), chain="A", resseq="2")
    interchain_acceptor = atom(4, "O", "O", (2.8, 0, 0), chain="B", resseq="1")

    assert should_exclude_pair(donor, same_res_acceptor)
    assert should_exclude_pair(donor, next_res_acceptor)
    assert not should_exclude_pair(donor, interchain_acceptor)


def test_interchain_pair_classification() -> None:
    donor = atom(1, "N", "N", (0, 0, 0), chain="A", resseq="1")
    acceptor = atom(2, "O", "O", (2.8, 0, 0), chain="B", resseq="3")

    assert pair_type(donor, acceptor) == "interchain"


def test_hbond_pair_rows_without_hydrogen_use_proxy_caveat() -> None:
    atoms = [
        atom(1, "N", "N", (0, 0, 0), chain="A", resseq="1"),
        atom(2, "O", "O", (2.8, 0, 0), chain="B", resseq="3"),
    ]

    pairs = hbond_pair_rows("m", atoms)

    assert len(pairs) == 1
    assert pairs.iloc[0]["hbond_class"] == "missing_hydrogen_proxy"
    assert pairs.iloc[0]["geometry_basis"] == "hydrogen_missing_geometry_proxy"


def test_hbond_pair_rows_with_explicit_hydrogen_use_angle() -> None:
    atoms = [
        atom(1, "N", "N", (0, 0, 0), chain="A", resseq="1"),
        atom(2, "H", "H", (1, 0, 0), chain="A", resseq="1"),
        atom(3, "O", "O", (2.8, 0, 0), chain="B", resseq="3"),
    ]

    pairs = hbond_pair_rows("m", atoms)

    assert len(pairs) == 1
    assert pairs.iloc[0]["geometry_basis"] == "explicit_hydrogen_angle"
    assert pairs.iloc[0]["hbond_class"] == "strong_geometry"


def test_candidate_level_scoring_and_missing_hydrogen_caveat() -> None:
    atoms = [
        atom(1, "N", "N", (0, 0, 0), chain="A", resseq="1"),
        atom(2, "O", "O", (2.8, 0, 0), chain="B", resseq="3"),
        atom(3, "N", "N", (10, 0, 0), chain="A", resseq="5"),
        atom(4, "O", "O", (12.6, 0, 0), chain="B", resseq="6"),
    ]
    pairs = hbond_pair_rows("m", atoms)
    inventory_row = pd.Series({"status": "found", "path": "m.pdb", "inferred_family": "test", "inferred_scale": ""})

    summary = score_summary_row("m", atoms, pairs, inventory_row)

    assert summary["strong_hbond_count"] >= 1
    assert summary["hbond_plausibility_score"] > 0
    assert "hydrogen_missing_geometry_proxy" in summary["hbond_caveat"]


def test_missing_candidate_summary_row() -> None:
    row = missing_summary_row(pd.Series({"model_id": "missing", "inferred_family": "test", "path": ""}))

    assert row["hbond_network_classification"] == "missing_candidate_coordinates"


def test_report_wording_contains_required_cautions() -> None:
    summary = pd.DataFrame(
        [
            {
                "model_id": "m",
                "status": "found",
                "explicit_hydrogens_present": False,
                "total_candidate_pairs_considered": 1,
                "hbond_network_classification": "hbond_network_marginal",
                "hbond_plausibility_score": 2,
                "hbond_caveat": "hydrogen_missing_geometry_proxy; missing hydrogens limit angle-based interpretation",
                "hbond_plausibility_score_per_residue": 1.0,
                "strong_hbond_count": 0,
                "plausible_hbond_count": 1,
                "weak_hbond_count": 0,
                "interchain_hbond_count": 1,
                "carboxylate_contact_count": 0,
                "possible_bad_acceptor_acceptor_contact_count": 0,
                "possible_bad_donor_donor_contact_count": 0,
            }
        ]
    )
    text = build_report(summary, pd.DataFrame())

    for phrase in [
        "hydrogen-bond plausibility score",
        "not a true affinity",
        "not a free-energy calculation",
        "Missing hydrogens",
        "Protonation states",
        "comparative physical-sense filter",
        "Candidate elimination should not rely on this score alone",
    ]:
        assert phrase in text
