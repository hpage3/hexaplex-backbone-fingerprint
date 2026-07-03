from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.analyze_threefold_backbone_symmetry import (
    ChainFingerprint,
    build_report_text,
    classify_chain,
    interpretation,
    symmetry_gap_rms_deg,
    symmetry_summary,
)


def chain_row(chain: str, backbone_class: str, radial_angle: float, exit_angle: float | None = None) -> ChainFingerprint:
    angle = np.radians(radial_angle)
    return ChainFingerprint(
        chain=chain,
        backbone_class=backbone_class,
        classification_confidence="high",
        residue_names="CYP" if backbone_class == "triketo_cyanuric_like" else "MEP",
        ca_centroid=np.array([np.cos(angle), np.sin(angle), 0.0], dtype=float),
        radial_angle_deg=radial_angle,
        exit_vector=np.array([1.0, 0.0, 0.0], dtype=float),
        exit_vector_xy_angle_deg=radial_angle if exit_angle is None else exit_angle,
        peptide_normal=None,
        peptide_normal_xy_angle_deg=None,
        theta_median_deg=110.0 if backbone_class == "triketo_cyanuric_like" else 108.0,
        omega_median_deg=-179.0 if backbone_class == "triketo_cyanuric_like" else -167.0,
        omega_trans_deviation_median_deg=1.0 if backbone_class == "triketo_cyanuric_like" else 13.0,
    )


def test_classify_chain_is_conservative() -> None:
    assert classify_chain({"GLU", "MEP"}) == ("triamino_melamine_like", "high")
    assert classify_chain({"GLU", "CYP"}) == ("triketo_cyanuric_like", "high")
    assert classify_chain({"MEP", "CYP"}) == ("mixed_or_uncertain", "low")
    assert classify_chain({"GLU"}) == ("unclassified", "low")


def test_ideal_sixfold_synthetic_fixture_has_zero_sixfold_rms() -> None:
    chains = [
        chain_row("A", "triketo_cyanuric_like", 0.0),
        chain_row("B", "triamino_melamine_like", 60.0),
        chain_row("C", "triketo_cyanuric_like", 120.0),
        chain_row("D", "triamino_melamine_like", 180.0),
        chain_row("E", "triketo_cyanuric_like", 240.0),
        chain_row("F", "triamino_melamine_like", 300.0),
    ]
    summary = symmetry_summary("ideal_sixfold", chains)
    six = summary[summary["family"] == "forced_sixfold_all_chains"].iloc[0]

    assert six["chain_count"] == 6
    assert float(six["radial_angle_gap_rms_deg"]) == 0.0
    assert float(six["exit_vector_angle_gap_rms_deg"]) == 0.0


def test_two_class_threefold_fixture_beats_forced_sixfold_when_classes_are_offset() -> None:
    chains = [
        chain_row("A", "triketo_cyanuric_like", 0.0),
        chain_row("B", "triamino_melamine_like", 20.0),
        chain_row("C", "triketo_cyanuric_like", 120.0),
        chain_row("D", "triamino_melamine_like", 140.0),
        chain_row("E", "triketo_cyanuric_like", 240.0),
        chain_row("F", "triamino_melamine_like", 260.0),
    ]
    summary = symmetry_summary("two_class", chains)
    six = summary[summary["family"] == "forced_sixfold_all_chains"].iloc[0]
    three = summary[summary["family"].str.startswith("threefold_")]

    assert float(six["radial_angle_gap_rms_deg"]) > 0.0
    assert all(float(value) == 0.0 for value in three["radial_angle_gap_rms_deg"])
    assert interpretation(summary)["supports_threefold_scope_concern"] is True


def test_symmetry_gap_rms_reports_nonideal_spacing() -> None:
    assert symmetry_gap_rms_deg([0.0, 120.0, 240.0], 120.0) == 0.0
    assert symmetry_gap_rms_deg([0.0, 100.0, 240.0], 120.0) > 0.0


def test_report_wording_marks_scope_not_reconstruction(tmp_path) -> None:
    chains = [
        chain_row("A", "triketo_cyanuric_like", 0.0),
        chain_row("B", "triamino_melamine_like", 20.0),
        chain_row("C", "triketo_cyanuric_like", 120.0),
        chain_row("D", "triamino_melamine_like", 140.0),
        chain_row("E", "triketo_cyanuric_like", 240.0),
        chain_row("F", "triamino_melamine_like", 260.0),
    ]
    summary = symmetry_summary("two_class", chains)
    chain_df = pd.DataFrame(
        [
            {
                "chain": row.chain,
                "backbone_class": row.backbone_class,
                "classification_confidence": row.classification_confidence,
                "residue_names": row.residue_names,
                "radial_angle_deg": row.radial_angle_deg,
                "exit_vector_xy_angle_deg": row.exit_vector_xy_angle_deg,
                "theta_median_deg": row.theta_median_deg,
                "omega_median_deg": row.omega_median_deg,
                "omega_trans_deviation_median_deg": row.omega_trans_deviation_median_deg,
            }
            for row in chains
        ]
    )

    text = build_report_text(tmp_path / "parent.pdb", summary, chain_df)

    assert "model-scope/symmetry analysis, not a new atomistic reconstruction" in text
    assert "failed pseudo reconstructed bridge is not parent-equivalent" in text
    assert "fine parent-derived rise scan succeeded within the constrained six-fold parent-derived family" in text
    assert "two independent three-fold peptide-backbone symmetry classes" in text
    assert "Build a new peptide-plane model track" in text
