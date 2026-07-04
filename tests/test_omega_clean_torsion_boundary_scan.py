from __future__ import annotations

import pandas as pd

from scripts.run_internal_coordinate_endpoint_closure import detect_every_other_pattern
from scripts.run_omega_clean_rise_compression_scan import classify_omega_value
from scripts.run_omega_clean_torsion_boundary_scan import (
    BoundarySpec,
    baseline_specs,
    build_report,
    combined_specs,
    extended_delta_grid,
    first_failure_delta,
    generate_specs,
    largest_all_viable_square,
    max_abs_viable_delta,
    one_family_specs,
    scale_id,
    summarize_boundary,
)
from scripts.run_omega_clean_torsion_envelope_scan import cd_status, combined_status, geometry_status


def test_extended_delta_grid_generation() -> None:
    assert extended_delta_grid() == [-12.0, -10.0, -8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]


def test_one_torsion_family_at_a_time_variant_generation() -> None:
    specs = one_family_specs(scales=[0.9825], deltas=[-2.0, 0.0, 2.0])

    assert len(specs) == 3 * 3 * 3
    assert {spec.torsion_family for spec in specs} == {"phi", "psi", "omega"}
    assert all(spec.scan_family == "one_family" for spec in specs)


def test_limited_combined_perturbation_generation() -> None:
    specs = combined_specs(scales=[0.9825])
    families = {spec.scan_family for spec in specs}

    assert "combined_symmetric_same_sign" in families
    assert "combined_opposing_class" in families
    assert "combined_phi_psi_compensation" in families
    assert len(specs) == 8 + 18 + 6


def test_full_spec_generation_count_for_one_scale() -> None:
    specs = generate_specs(scales=[0.9825])

    assert len(specs) == 3 * 13 * 13 + 32


def test_scale_id_handling() -> None:
    assert [scale_id(scale) for scale in [0.9825, 0.9800, 0.9775, 0.9750, 0.9725]] == [
        "0p9825",
        "0p9800",
        "0p9775",
        "0p9750",
        "0p9725",
    ]


def test_no_perturbation_baseline_inclusion() -> None:
    baselines = baseline_specs(scales=[0.9825])

    assert len(baselines) == 3
    assert {spec.torsion_family for spec in baselines} == {"phi", "psi", "omega"}
    assert all(spec.triketo_phi_delta_deg == 0 for spec in baselines)
    assert all(spec.triamino_omega_delta_deg == 0 for spec in baselines)


def test_omega_wraparound_and_every_other_detection() -> None:
    assert classify_omega_value(180.0) == "within_8deg"
    assert classify_omega_value(-171.0) == "within_10deg"
    assert classify_omega_value(166.0) == "outside_10deg"
    assert detect_every_other_pattern([1.0, 15.0, 1.0, 15.0, 2.0, 16.0])["every_other_detected"] is True


def test_status_classifications() -> None:
    assert cd_status(5.6422, 7.2756) == "cd_plateau_preserved"
    assert cd_status(5.6422, 7.1923) == "c_preserved_d_degraded"
    assert cd_status(5.7454, 7.2756) == "parent_like"
    assert cd_status(5.8, 7.1) == "degraded_other"

    clean = {
        "write_status": "written",
        "atom_count_preserved": True,
        "carboxylates_preserved": True,
        "coordinate_omega_every_other_detected": False,
        "omega_within_8_count": 10,
        "omega_within_10_count": 10,
        "omega_count": 10,
    }
    assert geometry_status(clean) == "geometry_clean"
    assert geometry_status({**clean, "omega_within_8_count": 8}) == "geometry_borderline"
    assert geometry_status({**clean, "coordinate_omega_every_other_detected": True}) == "geometry_implausible"
    assert geometry_status({**clean, "write_status": "failed"}) == "reconstruction_failed"

    assert combined_status("cd_plateau_preserved", "geometry_clean") == "viable_envelope_member"
    assert combined_status("cd_plateau_preserved", "geometry_implausible") == "diffraction_only_member"
    assert combined_status("degraded_other", "geometry_borderline") == "geometry_only_member"
    assert combined_status("degraded_other", "geometry_implausible") == "rejected"


def boundary_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"triketo_delta_deg": -2.0, "triamino_delta_deg": -2.0, "combined_status": "viable_envelope_member", "cd_status": "cd_plateau_preserved", "geometry_status": "geometry_clean"},
            {"triketo_delta_deg": 0.0, "triamino_delta_deg": 0.0, "combined_status": "viable_envelope_member", "cd_status": "cd_plateau_preserved", "geometry_status": "geometry_clean"},
            {"triketo_delta_deg": 2.0, "triamino_delta_deg": 2.0, "combined_status": "viable_envelope_member", "cd_status": "cd_plateau_preserved", "geometry_status": "geometry_clean"},
            {"triketo_delta_deg": 4.0, "triamino_delta_deg": 0.0, "combined_status": "geometry_only_member", "cd_status": "degraded_other", "geometry_status": "geometry_clean"},
            {"triketo_delta_deg": 0.0, "triamino_delta_deg": 6.0, "combined_status": "rejected", "cd_status": "degraded_other", "geometry_status": "geometry_implausible"},
        ]
    )


def test_boundary_estimation_helpers() -> None:
    group = boundary_fixture()

    assert max_abs_viable_delta(group, "triketo_delta_deg") == 2.0
    assert first_failure_delta(group.assign(_fail=group["cd_status"] != "cd_plateau_preserved"), "_fail", deltas=[0.0, 2.0, 4.0, 6.0]) == 4.0
    assert first_failure_delta(group.assign(_fail=group["geometry_status"] == "geometry_implausible"), "_fail", deltas=[0.0, 2.0, 4.0, 6.0]) == 6.0
    assert largest_all_viable_square(group) == 2.0


def test_summarize_boundary_and_report_wording() -> None:
    scores = pd.DataFrame(
        [
            {
                "variant_id": "v1",
                "scale": 0.9825,
                "scan_family": "one_family",
                "torsion_family": "phi",
                "triketo_delta_deg": 0.0,
                "triamino_delta_deg": 0.0,
                "max_abs_delta_deg": 0.0,
                "scoreable": True,
                "cd_status": "cd_plateau_preserved",
                "geometry_status": "geometry_clean",
                "combined_status": "viable_envelope_member",
            },
            {
                "variant_id": "v2",
                "scale": 0.9825,
                "scan_family": "one_family",
                "torsion_family": "phi",
                "triketo_delta_deg": 12.0,
                "triamino_delta_deg": 0.0,
                "max_abs_delta_deg": 12.0,
                "scoreable": False,
                "cd_status": "degraded_other",
                "geometry_status": "geometry_implausible",
                "combined_status": "rejected",
            },
        ]
    )
    geometry = scores[["variant_id", "geometry_status"]].copy()
    summary = summarize_boundary(scores)
    report = build_report(scores, geometry, summary)

    assert summary.iloc[0]["attempted_variant_count"] == 2
    for phrase in [
        "omega-clean torsion-boundary scan",
        "beyond +/-4",
        "local (phi/psi/omega)x2",
        "not a final structure",
        "not energy minimized",
        "Nick",
        "narrow but finite",
        "scientific judgment for the PI",
    ]:
        assert phrase in report
    assert "does not claim a unique structure" in report
