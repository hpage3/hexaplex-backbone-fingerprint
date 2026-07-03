from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_external_backbone_prototype import (
    atom_count_preserved,
    build_report,
    count_atom_records,
    select_torsions,
    selected_omega_summary,
    write_local_fragment_pdb,
)
from scripts.run_internal_coordinate_endpoint_closure import omega_window_class, trans_deviation_deg
from scripts.run_parent_derived_rise_bridge import carboxylate_present
from scripts.run_phi_psi_omega_closure_scan import PhiPsiOmegaSegment
from scripts.generate_global_deformation_variants import parse_pdb_atom_lines


def selection_fixture(rows: list[dict[str, object]]) -> pd.DataFrame:
    base = {
        "segment_id": "A:1:CYP1->GLU2",
        "chain": "A",
        "class_label": "triketo_cyanuric_like",
        "res_i": "1",
        "res_j": "2",
        "parent_phi_deg": -120.0,
        "parent_psi_deg": 150.0,
        "parent_omega_deg": -168.0,
        "scanned_phi_deg": -120.0,
        "scanned_psi_deg": 150.0,
        "scanned_omega_deg": -180.0,
        "omega_window_class": "within_8deg",
        "omega_trans_deviation_deg": 0.0,
        "phi_delta_from_parent_deg": 0.0,
        "psi_delta_from_parent_deg": 0.0,
        "endpoint_distance_parent_A": 3.8,
        "endpoint_distance_model_A": 3.8,
        "closure_residual_A": 0.01,
        "closure_class": "good_closure",
        "scan_status": "scored",
    }
    return pd.DataFrame([{**base, **row} for row in rows])


def test_selected_torsion_prefers_good_within_8() -> None:
    scan = selection_fixture(
        [
            {"segment_id": "A:1", "omega_window_class": "within_10deg", "closure_residual_A": 0.001},
            {"segment_id": "A:1", "omega_window_class": "within_8deg", "closure_residual_A": 0.05},
        ]
    )
    selected = select_torsions(scan).iloc[0]
    assert selected["selection_reason"] == "good_within_8deg"
    assert selected["closure_residual_A"] == 0.05


def test_selected_torsion_falls_back_to_good_within_10() -> None:
    scan = selection_fixture(
        [
            {"segment_id": "A:1", "omega_window_class": "within_10deg", "closure_residual_A": 0.04},
            {"segment_id": "A:1", "omega_window_class": "outside_10deg", "closure_residual_A": 0.001},
        ]
    )
    selected = select_torsions(scan).iloc[0]
    assert selected["selection_reason"] == "good_within_10deg"


def test_selected_torsion_falls_back_to_borderline_within_10() -> None:
    scan = selection_fixture(
        [
            {"segment_id": "A:1", "omega_window_class": "within_10deg", "closure_class": "borderline_closure", "closure_residual_A": 0.2},
            {"segment_id": "A:1", "omega_window_class": "outside_10deg", "closure_class": "good_closure", "closure_residual_A": 0.001},
        ]
    )
    selected = select_torsions(scan).iloc[0]
    assert selected["selection_reason"] == "borderline_within_10deg"


def test_selected_torsion_retains_parent_when_unresolved() -> None:
    scan = selection_fixture(
        [
            {"segment_id": "A:1", "omega_window_class": "outside_10deg", "closure_class": "poor_closure", "scan_status": "scored"},
        ]
    )
    selected = select_torsions(scan).iloc[0]
    assert selected["selection_reason"] == "retain_parent_unresolved"


def test_omega_wraparound_and_window_classification() -> None:
    assert trans_deviation_deg(-180.0) == 0.0
    assert trans_deviation_deg(172.0) == 8.0
    assert omega_window_class(-172.0) == "within_8deg"
    assert omega_window_class(-170.0) == "within_10deg"


def test_selected_omega_summary_detects_every_other() -> None:
    selected = pd.DataFrame(
        {
            "selected_omega_deg": [-180.0, -168.0, -180.0, -168.0],
            "selection_reason": ["good_within_8deg"] * 4,
        }
    )
    summary = selected_omega_summary(selected)
    assert summary["selected_omega_every_other_detected"] is True


def test_atom_count_preservation_and_carboxylate_check(tmp_path: Path) -> None:
    pdb = tmp_path / "same.pdb"
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
    assert count_atom_records(pdb) == 2
    assert atom_count_preserved(pdb, pdb)
    assert carboxylate_present(atoms)


def test_coordinate_output_contains_expected_identity_fields(tmp_path: Path) -> None:
    segment = PhiPsiOmegaSegment(
        chain="A",
        class_label="triketo_cyanuric_like",
        segment_index=1,
        segment_id="A:1:CYP1->GLU2",
        residue_pair="CYP1->GLU2",
        res_i="1",
        res_j="2",
        c_prev=np.array([-0.8, -0.4, 0.2]),
        n_i=np.array([0.0, 0.0, 0.0]),
        ca_i=np.array([1.0, 0.0, 0.0]),
        c_i=np.array([1.5, 1.0, 0.0]),
        n_j=np.array([2.0, 1.2, 0.0]),
        ca_j=np.array([2.5, 2.0, 0.0]),
        parent_phi_deg=-120.0,
        parent_psi_deg=150.0,
        parent_omega_deg=-180.0,
        ca_c_length_A=1.5,
        c_n_length_A=1.3,
        n_ca_length_A=1.4,
        n_ca_c_angle_deg=110.0,
        ca_c_n_angle_deg=115.0,
        c_n_ca_angle_deg=120.0,
    )
    selected = pd.DataFrame(
        [
            {
                "segment_id": "A:1:CYP1->GLU2",
                "selection_reason": "good_within_8deg",
                "selected_phi_deg": -120.0,
                "selected_psi_deg": 150.0,
                "selected_omega_deg": -180.0,
            }
        ]
    )
    out = tmp_path / "fragment.pdb"
    info = write_local_fragment_pdb(selected, {"A:1:CYP1->GLU2": segment}, out)
    text = out.read_text(encoding="utf-8")
    assert info["fragment_model_count"] == 1
    assert "MODEL" in text
    assert " CYP A   1" in text
    assert " GLU A   2" in text
    assert " CA " in text


def test_report_wording_contains_required_terms() -> None:
    selected = pd.DataFrame({"selection_reason": ["good_within_8deg"]})
    geometry = pd.DataFrame(
        [
            {
                "prototype_type": "local_fragment_multimodel",
                "full_prototype_pdb_produced": False,
                "fragment_prototype_pdb_produced": True,
                "source_atom_count": 10,
                "prototype_atom_count": 5,
                "atom_count_preserved": False,
                "source_carboxylate_present": True,
                "prototype_carboxylate_present": False,
                "selected_segment_count": 1,
                "reconstructed_segment_count": 1,
                "retained_parent_segment_count": 0,
                "selected_count": 1,
                "selected_omega_median_deg": -180.0,
                "selected_omega_within_8deg_count": 1,
                "selected_omega_within_10deg_count": 1,
                "selected_omega_every_other_detected": False,
                "median_closure_residual_A": 0.01,
                "max_closure_residual_A": 0.01,
                "diffraction_scoring_status": "not_applicable_for_fragment_prototype",
            }
        ]
    )
    abcd = pd.DataFrame([{"diffraction_scoring_status": "not_applicable_for_fragment_prototype"}])
    text = build_report(selected, geometry, abcd, Path("prototype.pdb"))
    assert "coordinate-producing external backbone prototype" in text
    assert "not a final structure" in text
    assert "not energy minimized" in text
    assert "pNAB" in text
    assert "every-other" in text
    assert "phi/psi/omega" in text
    assert "+/- 8" in text
    assert "+/- 10" in text
