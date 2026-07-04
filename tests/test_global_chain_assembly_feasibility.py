import numpy as np
import pandas as pd

from scripts.analyze_global_chain_assembly_feasibility import (
    build_report,
    chain_summary,
    class_for_chain,
    drift_class,
    edge_summary,
    overlap_class,
    safe_to_write_full_pdb,
    segment_order_from_id,
    steric_conflict_class,
)
from scripts.run_internal_coordinate_endpoint_closure import detect_every_other_pattern, trans_deviation_deg


def selected_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "segment_id": "A:1:CYP1->GLU2",
                "chain": "A",
                "class_label": "triketo_cyanuric_like",
                "res_i": 1,
                "res_j": 2,
                "selection_reason": "retain_parent_unresolved",
                "selected_phi_deg": np.nan,
                "selected_psi_deg": np.nan,
                "selected_omega_deg": np.nan,
                "omega_window_class": "insufficient_data",
                "closure_residual_A": np.nan,
            },
            {
                "segment_id": "A:2:GLU2->CYP3",
                "chain": "A",
                "class_label": "triketo_cyanuric_like",
                "res_i": 2,
                "res_j": 3,
                "selection_reason": "good_within_8deg",
                "selected_phi_deg": -120.0,
                "selected_psi_deg": 100.0,
                "selected_omega_deg": -172.0,
                "omega_window_class": "within_8deg",
                "closure_residual_A": 0.05,
            },
        ]
    )


def edge_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "chain": "A",
                "class_label": "triketo_cyanuric_like",
                "edge_order": 1,
                "expected_edge_count_in_chain": 2,
                "residue_count_in_chain": 3,
                "edge_status": "unresolved_or_retained_parent",
                "closure_residual_A": np.nan,
                "selected_omega_deg": np.nan,
            },
            {
                "chain": "A",
                "class_label": "triketo_cyanuric_like",
                "edge_order": 2,
                "expected_edge_count_in_chain": 2,
                "residue_count_in_chain": 3,
                "edge_status": "reconstructed",
                "closure_residual_A": 0.05,
                "selected_omega_deg": -172.0,
            },
        ]
    )


def test_chain_graph_completeness_detects_missing_reconstructed_edge() -> None:
    summary = chain_summary(edge_fixture()).iloc[0]
    assert summary["expected_edge_count"] == 2
    assert summary["reconstructed_edge_count"] == 1
    assert summary["unresolved_or_retained_edge_count"] == 1
    assert bool(summary["has_continuous_reconstructed_path"]) is False
    assert summary["blocking_edges"] == "1"


def test_segment_order_uses_coordinate_order_not_raw_residue_id() -> None:
    assert segment_order_from_id("B:2:GLU32->MEP33", 32) == 2


def test_overlap_rmsd_classification_good_borderline_poor() -> None:
    assert overlap_class(0.05) == "good_overlap"
    assert overlap_class(0.2) == "borderline_overlap"
    assert overlap_class(0.4) == "poor_overlap"


def test_endpoint_drift_surrogate_classification_good_borderline_poor() -> None:
    assert drift_class(0.1) == "good_drift"
    assert drift_class(0.5) == "borderline_drift"
    assert drift_class(1.0) == "poor_drift"


def test_omega_window_preservation_and_every_other_detection() -> None:
    values = [-180.0, -168.0, -180.0, -168.0]
    deviations = [trans_deviation_deg(value) for value in values]
    assert deviations == [0.0, 12.0, 0.0, 12.0]
    assert detect_every_other_pattern(deviations)["every_other_detected"] is True


def test_class_assignment_for_two_families() -> None:
    assert class_for_chain("A") == "triketo_cyanuric_like"
    assert class_for_chain("E") == "triketo_cyanuric_like"
    assert class_for_chain("B") == "triamino_melamine_like"
    assert class_for_chain("F") == "triamino_melamine_like"


def test_steric_conflict_classification() -> None:
    assert steric_conflict_class(1.0) == "severe_conflict"
    assert steric_conflict_class(1.4) == "possible_conflict"
    assert steric_conflict_class(2.0) == "no_conflict"


def test_no_full_pdb_when_graph_incomplete_or_overlap_poor() -> None:
    chains = pd.DataFrame(
        [
            {
                "has_continuous_reconstructed_path": False,
                "drift_class": "good_drift",
            }
        ]
    )
    overlaps = pd.DataFrame([{"overlap_class": "good_overlap"}])
    sterics = pd.DataFrame([{"steric_conflict_class": "no_conflict"}])
    assert safe_to_write_full_pdb(chains, overlaps, sterics) == (False, "incomplete_reconstructed_chain_paths")

    chains["has_continuous_reconstructed_path"] = True
    overlaps["overlap_class"] = "poor_overlap"
    assert safe_to_write_full_pdb(chains, overlaps, sterics) == (False, "overlap_not_globally_good")


def test_report_wording_contains_required_scope_terms() -> None:
    edges = edge_fixture()
    chains = chain_summary(edges)
    overlaps = pd.DataFrame(
        [
            {
                "overlap_class": "poor_overlap",
                "overlap_rmsd_A": 0.5,
            }
        ]
    )
    sterics = pd.DataFrame(
        [
            {
                "steric_conflict_class": "no_conflict",
                "severe_conflict_count": 0,
                "possible_conflict_count": 0,
            }
        ]
    )
    text = build_report(edges, chains, overlaps, sterics, False, "incomplete_reconstructed_chain_paths")
    assert "global chain assembly feasibility" in text
    assert "not a final structure" in text
    assert "not energy minimized" in text
    assert "phi/psi/omega" in text
    assert "every-other" in text
    assert "diffraction scoring should not be performed" in text.lower()
