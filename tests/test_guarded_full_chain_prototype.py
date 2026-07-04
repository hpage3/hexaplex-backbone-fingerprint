from pathlib import Path

import pandas as pd

from scripts.build_guarded_full_chain_prototype import (
    attach_retained_parent_omega,
    chain_summary_after_terminal_completion,
    complete_terminal_edges,
    guard_conditions,
    build_report,
)
from scripts.analyze_global_chain_assembly_feasibility import drift_class, overlap_class
from scripts.run_internal_coordinate_endpoint_closure import detect_every_other_pattern, trans_deviation_deg
from scripts.run_parent_derived_rise_bridge import carboxylate_present
from scripts.generate_global_deformation_variants import parse_pdb_atom_lines


def edge_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "chain": "A",
                "class_label": "triketo_cyanuric_like",
                "segment_id": "A:1:CYP1->GLU2",
                "res_i": 1,
                "res_j": 2,
                "edge_order": 1,
                "expected_edge_count_in_chain": 2,
                "edge_status": "unresolved_or_retained_parent",
                "selected_omega_deg": None,
            },
            {
                "chain": "A",
                "class_label": "triketo_cyanuric_like",
                "segment_id": "A:2:GLU2->CYP3",
                "res_i": 2,
                "res_j": 3,
                "edge_order": 2,
                "expected_edge_count_in_chain": 2,
                "edge_status": "reconstructed",
                "selected_omega_deg": -172.0,
            },
        ]
    )


def prior_chain_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "chain": "A",
                "endpoint_drift_surrogate_A": 0.1,
                "drift_class": "good_drift",
                "selected_omega_every_other_detected": False,
            }
        ]
    )


def test_terminal_edge_completion_retains_parent_without_inventing_torsion() -> None:
    completed = complete_terminal_edges(edge_fixture())
    first = completed[completed["edge_order"] == 1].iloc[0]
    assert first["terminal_completion_method"] == "retained_parent_terminal_edge"
    assert first["assembly_edge_status"] == "terminal_retained_parent"
    assert bool(first["assembly_complete_edge"]) is True
    assert pd.isna(first["selected_omega_deg"])


def test_terminal_edge_uses_parent_omega_for_summary_not_new_torsion() -> None:
    completed = complete_terminal_edges(edge_fixture())
    selected = pd.DataFrame(
        [
            {
                "segment_id": "A:1:CYP1->GLU2",
                "parent_omega_deg": -167.0,
            }
        ]
    )
    enriched = attach_retained_parent_omega(completed, selected)
    first = enriched[enriched["edge_order"] == 1].iloc[0]
    assert first["terminal_completion_method"] == "retained_parent_terminal_edge"
    assert first["selected_omega_deg"] == -167.0


def test_chain_graph_becomes_complete_after_terminal_retention() -> None:
    completed = complete_terminal_edges(edge_fixture())
    summary = chain_summary_after_terminal_completion(completed, prior_chain_fixture()).iloc[0]
    assert summary["complete_edge_count"] == 2
    assert summary["terminal_retained_edge_count"] == 1
    assert bool(summary["complete_path_after_terminal_handling"]) is True


def test_guard_passes_only_when_all_conditions_pass() -> None:
    chains = pd.DataFrame(
        [
            {
                "complete_path_after_terminal_handling": True,
                "drift_class": "good_drift",
            }
        ]
    )
    overlaps = pd.DataFrame([{"overlap_class": "good_overlap"}])
    sterics = pd.DataFrame([{"steric_conflict_class": "no_conflict"}])
    assert guard_conditions(chains, overlaps, sterics, True, True, False) == (True, "all_guards_passed")
    assert guard_conditions(chains, overlaps, sterics, False, True, False)[1] == "atom_count_not_preserved"
    assert guard_conditions(chains, overlaps, sterics, True, False, False)[1] == "carboxylates_not_preserved"
    assert guard_conditions(chains, overlaps, sterics, True, True, True)[1] == "selected_or_retained_omega_every_other_detected"
    overlaps["overlap_class"] = "poor_overlap"
    assert guard_conditions(chains, overlaps, sterics, True, True, False)[1] == "poor_overlap_detected"


def test_overlap_and_drift_classification_reused() -> None:
    assert overlap_class(0.05) == "good_overlap"
    assert overlap_class(0.2) == "borderline_overlap"
    assert overlap_class(0.4) == "poor_overlap"
    assert drift_class(0.1) == "good_drift"
    assert drift_class(0.5) == "borderline_drift"
    assert drift_class(1.0) == "poor_drift"


def test_omega_every_other_detection_and_windows() -> None:
    deviations = [trans_deviation_deg(value) for value in [-180.0, -168.0, -180.0, -168.0]]
    assert deviations == [0.0, 12.0, 0.0, 12.0]
    assert detect_every_other_pattern(deviations)["every_other_detected"] is True


def test_carboxylate_preservation_check(tmp_path: Path) -> None:
    pdb = tmp_path / "carb.pdb"
    pdb.write_text(
        "\n".join(
            [
                "ATOM      1  N   GLU A   1       0.000   0.000   0.000  1.00  0.00           N",
                "ATOM      2  OE1 GLU A   1       1.000   0.000   0.000  1.00  0.00           O",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _lines, atoms = parse_pdb_atom_lines(pdb)
    assert carboxylate_present(atoms)


def test_report_wording_contains_required_terms() -> None:
    segments = complete_terminal_edges(edge_fixture())
    chains = chain_summary_after_terminal_completion(segments, prior_chain_fixture())
    geometry = pd.DataFrame(
        [
            {
                "full_pdb_written": True,
                "guard_status": "passed",
                "guard_blocker": "all_guards_passed",
                "terminal_edges_completed": 1,
                "terminal_completion_method": "retained_parent_terminal_edge",
                "reconstructed_segment_count": 1,
                "retained_parent_segment_count": 1,
                "atom_count_preserved": True,
                "source_atom_count": 2,
                "prototype_atom_count": 2,
                "source_carboxylate_present": True,
                "prototype_carboxylate_present": True,
                "carboxylates_preserved": True,
                "residue_register_preserved": True,
                "omega_count": 1,
                "omega_within_8deg_count": 1,
                "omega_within_10deg_count": 1,
                "omega_every_other_detected": False,
                "overlap_good_count": 1,
                "overlap_poor_count": 0,
                "drift_good_chain_count": 1,
                "steric_severe_conflict_count": 0,
                "steric_possible_conflict_count": 0,
                "rebuilt_backbone_atom_rmsd_to_parent_A": 0.1,
            }
        ]
    )
    abcd = pd.DataFrame([{"diffraction_status": "scored_preliminary", "notes": "preliminary"}])
    text = build_report(segments, chains, geometry, abcd, pd.DataFrame(), pd.DataFrame())
    assert "terminal-edge completion" in text
    assert "guarded full-chain assembly" in text
    assert "not a final structure" in text
    assert "not energy minimized" in text
    assert "pNAB" in text
    assert "every-other" in text
    assert "diffraction scoring" in text.lower()
