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


def test_identity_comparison_passes_and_fails() -> None:
    parent = [atom("N", 0, 0, 0, 1), atom("CA", 1, 0, 0, 2)]
    same = [atom("N", 0.1, 0, 0, 1), atom("CA", 1.1, 0, 0, 2)]
    changed = [atom("CA", 1.1, 0, 0, 2), atom("N", 0.1, 0, 0, 1)]
    assert identity_matches(parent, same)
    assert not identity_matches(parent, changed)


def test_rmsd_displacement_mean_radius_and_z_span() -> None:
    parent = [atom("N", 0, 0, 0, 1), atom("CA", 1, 0, -1, 2), atom("CA", -1, 0, 3, 3)]
    variant = [atom("N", 0, 3, 4, 1), atom("CA", 1, 0, 1, 2), atom("CA", -1, 0, 5, 3)]
    metrics = displacement_metrics(parent, variant)
    assert metrics["max_displacement_A"] == 5.0
    assert np.isclose(metrics["rmsd_ca_A"], 2.0)
    assert mean_ca_radius(parent) == 1.0
    assert z_span(parent) == 4.0


def test_geometry_interpretable_classifier_and_failure_formatting() -> None:
    assert geometry_interpretable(clean_row())
    row = clean_row()
    row["max_backbone_bond_delta_A"] = 0.151
    assert not geometry_interpretable(row)
    assert failed_checks(row) == ["backbone_bond_delta_exceeds_global_threshold"]
    row["max_backbone_angle_delta_deg"] = 11.0
    assert failed_checks(row) == [
        "backbone_bond_delta_exceeds_global_threshold",
        "backbone_angle_delta_exceeds_global_threshold",
    ]
