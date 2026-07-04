from __future__ import annotations

import pandas as pd

from scripts.run_internal_coordinate_endpoint_closure import detect_every_other_pattern
from scripts.run_omega_clean_torsion_envelope_scan import (
    DELTAS_DEG,
    PLATEAU_SCALES,
    baseline_specs,
    build_report,
    cd_status,
    combined_status,
    generate_specs,
    geometry_status,
    plateau_scale_id,
    summarize,
)
from scripts.run_omega_clean_rise_compression_scan import classify_omega_value


def test_class_level_perturbation_grid_generation() -> None:
    specs = generate_specs(scales=[0.9825], deltas=[-2.0, 0.0, 2.0])

    assert len(specs) == 3 * 3 * 3
    assert {spec.torsion_family for spec in specs} == {"phi", "psi", "omega"}
    assert {spec.triketo_delta_deg for spec in specs} == {-2.0, 0.0, 2.0}
    assert {spec.triamino_delta_deg for spec in specs} == {-2.0, 0.0, 2.0}


def test_one_torsion_family_at_a_time_scan_generation() -> None:
    specs = generate_specs(scales=[0.9825], deltas=DELTAS_DEG)

    assert len([spec for spec in specs if spec.torsion_family == "phi"]) == 25
    assert len([spec for spec in specs if spec.torsion_family == "psi"]) == 25
    assert len([spec for spec in specs if spec.torsion_family == "omega"]) == 25


def test_plateau_scale_id_handling() -> None:
    assert [plateau_scale_id(scale) for scale in PLATEAU_SCALES] == ["0p9825", "0p9800", "0p9775", "0p9750", "0p9725"]


def test_no_perturbation_baseline_included() -> None:
    baselines = baseline_specs(scales=[0.9825])
    ids = {spec.variant_id for spec in baselines}

    assert len(baselines) == 3
    assert "omega_clean_0p9825_phi_tri0_mel0" in ids
    assert "omega_clean_0p9825_psi_tri0_mel0" in ids
    assert "omega_clean_0p9825_omega_tri0_mel0" in ids


def test_omega_wraparound_and_window_classification() -> None:
    assert classify_omega_value(-179.9) == "within_8deg"
    assert classify_omega_value(172.0) == "within_8deg"
    assert classify_omega_value(171.0) == "within_10deg"
    assert classify_omega_value(165.0) == "outside_10deg"


def test_every_other_detection_on_synthetic_ordered_chain_values() -> None:
    assert detect_every_other_pattern([1.0, 15.0, 2.0, 16.0, 1.0, 15.0])["every_other_detected"] is True
    assert detect_every_other_pattern([1.0, 2.0, 1.5, 2.5, 2.0, 1.0])["every_other_detected"] is False


def test_cd_status_classification() -> None:
    assert cd_status(5.6422, 7.2756) == "cd_plateau_preserved"
    assert cd_status(5.6422, 7.1923) == "c_preserved_d_degraded"
    assert cd_status(5.7454, 7.2756) == "parent_like"
    assert cd_status(5.90, 7.10) == "degraded_other"


def test_geometry_status_classification() -> None:
    clean = {
        "write_status": "written",
        "atom_count_preserved": True,
        "carboxylates_preserved": True,
        "coordinate_omega_every_other_detected": False,
        "omega_within_8_count": 10,
        "omega_within_10_count": 10,
        "omega_count": 10,
    }
    borderline = {**clean, "omega_within_8_count": 8, "omega_within_10_count": 10}
    outside_10 = {**clean, "omega_within_10_count": 9}
    implausible = {**clean, "coordinate_omega_every_other_detected": True}
    failed = {**clean, "write_status": "failed"}

    assert geometry_status(clean) == "geometry_clean"
    assert geometry_status(borderline) == "geometry_borderline"
    assert geometry_status(outside_10) == "geometry_borderline"
    assert geometry_status(implausible) == "geometry_implausible"
    assert geometry_status(failed) == "reconstruction_failed"


def test_combined_status_classification() -> None:
    assert combined_status("cd_plateau_preserved", "geometry_clean") == "viable_envelope_member"
    assert combined_status("cd_plateau_preserved", "geometry_borderline") == "viable_envelope_member"
    assert combined_status("cd_plateau_preserved", "geometry_implausible") == "diffraction_only_member"
    assert combined_status("degraded_other", "geometry_clean") == "geometry_only_member"
    assert combined_status("degraded_other", "geometry_implausible") == "rejected"


def test_report_wording_and_plateau_language() -> None:
    scores = pd.DataFrame(
        [
            {
                "variant_id": "omega_clean_0p9825_phi_tri0_mel0",
                "scale": 0.9825,
                "torsion_family": "phi",
                "triketo_delta_deg": 0.0,
                "triamino_delta_deg": 0.0,
                "combined_CD_abs_error_A": 0.0667,
                "combined_status": "viable_envelope_member",
                "scoreable": True,
                "cd_status": "cd_plateau_preserved",
            },
            {
                "variant_id": "omega_clean_0p9800_phi_tri0_mel0",
                "scale": 0.9800,
                "torsion_family": "phi",
                "triketo_delta_deg": 0.0,
                "triamino_delta_deg": 0.0,
                "combined_CD_abs_error_A": 0.0667,
                "combined_status": "viable_envelope_member",
                "scoreable": True,
                "cd_status": "cd_plateau_preserved",
            },
        ]
    )
    geometry = pd.DataFrame(
        [
            {"variant_id": "omega_clean_0p9825_phi_tri0_mel0", "geometry_status": "geometry_clean"},
            {"variant_id": "omega_clean_0p9800_phi_tri0_mel0", "geometry_status": "geometry_clean"},
        ]
    )
    summary = pd.DataFrame(
        [
            {
                "scale": 0.9825,
                "torsion_family": "phi",
                "attempted_variant_count": 1,
                "scoreable_variant_count": 1,
                "guard_failed_count": 0,
                "cd_plateau_preserved_count": 1,
                "geometry_clean_count": 1,
                "viable_envelope_member_count": 1,
                "max_abs_triketo_delta_viable_deg": 0.0,
                "max_abs_triamino_delta_viable_deg": 0.0,
            }
        ]
    )

    text = build_report(scores, geometry, summary)

    for phrase in [
        "omega-clean torsion-envelope scan",
        "local (phi/psi/omega) x 2",
        "not a final structure",
        "not energy minimized",
        "Nick",
        "+/-8",
        "+/-10",
    ]:
        assert phrase in text
    assert "Best score plateau" in text


def test_summarize_accepts_scores_with_geometry_status() -> None:
    scores = pd.DataFrame(
        [
            {
                "variant_id": "v1",
                "scale": 0.9825,
                "torsion_family": "phi",
                "scoreable": True,
                "cd_status": "cd_plateau_preserved",
                "combined_status": "viable_envelope_member",
                "geometry_status": "geometry_borderline",
                "triketo_delta_deg": -2.0,
                "triamino_delta_deg": 2.0,
            }
        ]
    )
    geometry = pd.DataFrame([{"variant_id": "v1", "geometry_status": "geometry_borderline"}])

    summary = summarize(scores, geometry)

    assert summary.iloc[0]["attempted_variant_count"] == 1
    assert summary.iloc[0]["scoreable_variant_count"] == 1
    assert summary.iloc[0]["viable_envelope_member_count"] == 1
