from __future__ import annotations

import pandas as pd

from scripts.audit_rise_like_variant_geometry import build_report_text


def test_report_text_uses_rise_like_language_and_global_gates() -> None:
    results = pd.DataFrame(
        {
            "variant_id": ["rise_like_1p0000"],
            "axial_rise_scale": [1.0],
            "geometry_interpretable": [True],
            "max_displacement_A": [0.0],
            "rmsd_all_atoms_A": [0.0],
            "z_span_delta_A": [0.0],
            "max_backbone_bond_delta_A": [0.0],
            "max_backbone_angle_delta_deg": [0.0],
            "failed_checks": [""],
        }
    )
    text = build_report_text(results, source_pdb="parent.pdb")  # type: ignore[arg-type]
    assert "Rise-Like Variant Geometry Audit" in text
    assert "controlled perturbations, not minimized structures" in text
    assert "Max backbone bond-length delta" in text


def test_report_lists_failed_rise_like_variants() -> None:
    results = pd.DataFrame(
        {
            "variant_id": ["rise_like_0p9600"],
            "axial_rise_scale": [0.96],
            "geometry_interpretable": [False],
            "max_displacement_A": [1.0],
            "rmsd_all_atoms_A": [0.1],
            "z_span_delta_A": [-1.0],
            "max_backbone_bond_delta_A": [0.2],
            "max_backbone_angle_delta_deg": [0.0],
            "failed_checks": ["backbone_bond_delta_exceeds_global_threshold"],
        }
    )
    text = build_report_text(results, source_pdb="parent.pdb")  # type: ignore[arg-type]
    assert "rise_like_0p9600" in text
    assert "backbone_bond_delta_exceeds_global_threshold" in text
