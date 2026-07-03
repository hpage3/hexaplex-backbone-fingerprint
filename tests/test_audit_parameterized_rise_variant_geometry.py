from __future__ import annotations

from scripts.audit_parameterized_rise_variant_geometry import (
    failed_checks,
    geometry_interpretable,
    inter_layer_gaps,
    layer_order_preserved,
)


def clean_row() -> dict[str, object]:
    return {
        "atom_count_matches_parent": True,
        "identity_matches_parent": True,
        "layer_order_preserved": True,
        "max_backbone_bond_delta_A": 0.01,
        "max_backbone_angle_delta_deg": 1.0,
    }


def test_layer_order_preserved_classifier() -> None:
    assert layer_order_preserved([0.0, 1.0, 2.0])
    assert not layer_order_preserved([0.0, 1.0, 0.9])


def test_inter_layer_gap_calculation() -> None:
    assert inter_layer_gaps([0.0, 1.0, 2.5]) == [1.0, 1.5]


def test_geometry_interpretable_classifier() -> None:
    assert geometry_interpretable(clean_row())
    row = clean_row()
    row["layer_order_preserved"] = False
    assert not geometry_interpretable(row)


def test_failed_checks_formatting() -> None:
    row = clean_row()
    row["identity_matches_parent"] = False
    row["max_backbone_bond_delta_A"] = 0.2
    assert failed_checks(row) == ["identity_mismatch", "backbone_bond_delta_exceeds_global_threshold"]
