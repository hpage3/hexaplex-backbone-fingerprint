from __future__ import annotations

import numpy as np

from scripts.audit_constrained_phi_psi_candidate_geometry import Atom
from scripts.audit_global_deformation_variant_geometry import (
    displacement_metrics,
    failed_checks,
    geometry_interpretable,
    identity_matches,
    mean_ca_radius,
    z_span,
)


def atom(name: str, x: float, y: float, z: float, serial: int = 1) -> Atom:
    return Atom(
        record="ATOM",
        serial=serial,
        name=name,
        altloc="",
        resname="GLU",
        chain="A",
        resseq=1,
        icode="",
        x=x,
        y=y,
        z=z,
    )


def clean_row() -> dict[str, object]:
    return {
        "atom_count_matches_parent": True,
        "identity_matches_parent": True,
        "max_backbone_bond_delta_A": 0.01,
        "max_backbone_angle_delta_deg": 1.0,
    }


def test_atom_identity_comparison_passes_identical_records() -> None:
    parent = [atom("N", 0, 0, 0, 1), atom("CA", 1, 0, 0, 2)]
    variant = [atom("N", 0.1, 0, 0, 1), atom("CA", 1.1, 0, 0, 2)]
    assert identity_matches(parent, variant)


def test_atom_identity_comparison_fails_changed_order_or_name() -> None:
    parent = [atom("N", 0, 0, 0, 1), atom("CA", 1, 0, 0, 2)]
    changed_order = [atom("CA", 1, 0, 0, 2), atom("N", 0, 0, 0, 1)]
    changed_name = [atom("N", 0, 0, 0, 1), atom("C", 1, 0, 0, 2)]
    assert not identity_matches(parent, changed_order)
    assert not identity_matches(parent, changed_name)


def test_rmsd_and_displacement_calculations() -> None:
    parent = [atom("N", 0, 0, 0, 1), atom("CA", 1, 0, 0, 2)]
    variant = [atom("N", 3, 4, 0, 1), atom("CA", 1, 0, 2, 2)]
    metrics = displacement_metrics(parent, variant)
    assert metrics["max_displacement_A"] == 5.0
    assert np.isclose(metrics["rmsd_all_atoms_A"], np.sqrt((25.0 + 4.0) / 2.0))
    assert metrics["rmsd_ca_A"] == 2.0


def test_mean_ca_radius_and_z_span() -> None:
    atoms = [atom("CA", 1, 0, -1, 1), atom("CA", -1, 0, 3, 2), atom("N", 10, 0, 99, 3)]
    assert mean_ca_radius(atoms) == 1.0
    assert z_span(atoms) == 100.0


def test_geometry_interpretable_classifier_passes_clean_row() -> None:
    assert geometry_interpretable(clean_row())


def test_geometry_interpretable_fails_large_bond_delta() -> None:
    row = clean_row()
    row["max_backbone_bond_delta_A"] = 0.151
    assert not geometry_interpretable(row)
    assert failed_checks(row) == ["backbone_bond_delta_exceeds_global_threshold"]


def test_geometry_interpretable_fails_large_angle_delta() -> None:
    row = clean_row()
    row["max_backbone_angle_delta_deg"] = 10.1
    assert not geometry_interpretable(row)
    assert failed_checks(row) == ["backbone_angle_delta_exceeds_global_threshold"]


def test_failed_checks_formatting_is_stable_for_multiple_failures() -> None:
    row = clean_row()
    row["atom_count_matches_parent"] = False
    row["identity_matches_parent"] = False
    row["max_backbone_bond_delta_A"] = 0.2
    assert failed_checks(row) == [
        "atom_count_mismatch",
        "identity_mismatch",
        "backbone_bond_delta_exceeds_global_threshold",
    ]
