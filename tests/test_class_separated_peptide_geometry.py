from __future__ import annotations

import numpy as np

from scripts.analyze_class_separated_peptide_geometry import (
    ChainGeometry,
    build_report_text,
    class_distinguishability,
    class_for_chain,
    summary_table,
)


def chain_geometry(
    chain: str,
    backbone_class: str,
    radial_angle: float,
    exit_angle: float,
    omega_dev: float = 1.0,
    theta: float = 74.0,
    rise: float = 1.0,
) -> ChainGeometry:
    return ChainGeometry(
        chain=chain,
        backbone_class=backbone_class,
        residue_names="CYP,GLU" if backbone_class == "triketo_cyanuric_like" else "GLU,MEP",
        ca_count=30,
        omega_median_deg=-180.0 + omega_dev,
        omega_trans_deviation_median_deg=omega_dev,
        theta_median_deg=theta,
        theta_std_deg=0.2,
        ca_rise_median_A=rise,
        ca_rise_std_A=0.05,
        exit_vector_xy_angle_deg=exit_angle,
        radial_angle_deg=radial_angle,
        radial_radius_A=8.0,
        interstrand_nn_ca_median_A=4.0,
    )


def two_class_rows(identical_geometry: bool = True) -> list[ChainGeometry]:
    triketo = [
        chain_geometry("A", "triketo_cyanuric_like", 0.0, 0.0),
        chain_geometry("C", "triketo_cyanuric_like", 120.0, 120.0),
        chain_geometry("E", "triketo_cyanuric_like", 240.0, 240.0),
    ]
    if identical_geometry:
        triamino = [
            chain_geometry("B", "triamino_melamine_like", 60.0, 60.0),
            chain_geometry("D", "triamino_melamine_like", 180.0, 180.0),
            chain_geometry("F", "triamino_melamine_like", 300.0, 300.0),
        ]
    else:
        triamino = [
            chain_geometry("B", "triamino_melamine_like", 60.0, 70.0, omega_dev=13.0, theta=82.0, rise=1.2),
            chain_geometry("D", "triamino_melamine_like", 180.0, 190.0, omega_dev=13.0, theta=82.0, rise=1.2),
            chain_geometry("F", "triamino_melamine_like", 300.0, 310.0, omega_dev=13.0, theta=82.0, rise=1.2),
        ]
    return triketo + triamino


def test_class_for_chain_uses_threefold_assignment() -> None:
    assert class_for_chain("A") == "triketo_cyanuric_like"
    assert class_for_chain("C") == "triketo_cyanuric_like"
    assert class_for_chain("E") == "triketo_cyanuric_like"
    assert class_for_chain("B") == "triamino_melamine_like"
    assert class_for_chain("D") == "triamino_melamine_like"
    assert class_for_chain("F") == "triamino_melamine_like"
    assert class_for_chain("Z") == "unclassified"


def test_identical_two_class_fixture_has_zero_class_differences() -> None:
    summary = summary_table("synthetic", two_class_rows(identical_geometry=True))
    diff = summary[summary["group"] == "triamino_minus_triketo"].iloc[0]

    assert np.isclose(float(diff["omega_trans_deviation_median_deg"]), 0.0)
    assert np.isclose(float(diff["theta_median_deg"]), 0.0)
    assert np.isclose(float(diff["ca_rise_median_A"]), 0.0)
    assert np.isclose(float(diff["interstrand_nn_ca_median_A"]), 0.0)


def test_deliberately_different_fixture_detects_class_differences() -> None:
    summary = summary_table("synthetic", two_class_rows(identical_geometry=False))
    diff = summary[summary["group"] == "triamino_minus_triketo"].iloc[0]
    distinguish = class_distinguishability(summary)

    assert float(diff["omega_trans_deviation_median_deg"]) == 12.0
    assert float(diff["theta_median_deg"]) == 8.0
    assert np.isclose(float(diff["ca_rise_median_A"]), 0.2)
    assert distinguish["has_difference"] is True
    assert distinguish["most_distinct_metric"] in {"omega_trans_deviation_median_deg", "theta_median_deg"}


def test_report_wording_marks_diagnostic_not_reconstruction(tmp_path) -> None:
    summary = summary_table("synthetic", two_class_rows(identical_geometry=False))

    text = build_report_text(tmp_path / "parent.pdb", summary)

    assert "diagnostic analysis, not a new atomistic reconstruction" in text
    assert "Triketo/cyanuric-like chains: A,C,E" in text
    assert "Triamino/melamine-like chains: B,D,F" in text
    assert "not final chemistry" in text
    assert "Start with class-specific exit-vector orientation" in text
    assert "Keep pNAB/YAML provenance separate" in text
