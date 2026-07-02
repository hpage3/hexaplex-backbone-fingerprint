from __future__ import annotations

import pandas as pd

from scripts.audit_constrained_phi_psi_candidate_geometry import Atom
from scripts.audit_repeated_constrained_phi_psi_variant_geometry import (
    atom_label_sets_match,
    build_report_text,
    failure_reasons,
    join_manifest_audit,
    max_ca_shift,
    safe_for_diffraction,
)


def atom(name: str, x: float, serial: int = 1) -> Atom:
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
        y=0.0,
        z=0.0,
    )


def passing_audit() -> dict[str, object]:
    return {
        "candidate_file_exists": True,
        "atom_count_match": True,
        "labels_preserved": True,
        "max_ca_shift_A": 0.0,
        "max_backbone_bond_delta_A": 0.01,
        "max_backbone_angle_delta_deg": 1.0,
        "max_omega_trans_deviation_deg": 5.0,
    }


def test_atom_label_sets_match() -> None:
    parent = [atom("CA", 0.0, 1), atom("C", 1.0, 2)]
    same = [atom("CA", 0.1, 1), atom("C", 1.1, 2)]
    missing = [atom("CA", 0.1, 1)]
    assert atom_label_sets_match(parent, same)
    assert not atom_label_sets_match(parent, missing)


def test_max_ca_shift_calculation() -> None:
    parent = [atom("CA", 0.0, 1), atom("C", 1.0, 2)]
    variant = [atom("CA", 0.2, 1), atom("C", 4.0, 2)]
    assert max_ca_shift(parent, variant) == 0.2


def test_safe_for_diffraction_and_failure_reasons() -> None:
    audit = passing_audit()
    assert safe_for_diffraction(audit)
    audit["max_backbone_angle_delta_deg"] = 9.0
    assert not safe_for_diffraction(audit)
    assert failure_reasons(audit) == ["backbone_angle_delta_exceeds_threshold"]


def test_join_manifest_audit_preserves_metadata() -> None:
    manifest = pd.Series(
        {
            "variant_id": "rep",
            "fixed_torsion_delta_deg": 2.0,
            "omega_policy": "fixed_180",
            "attempted_window_count": 45,
            "applied_window_count": 45,
            "skipped_window_count": 0,
            "coordinate_path": "coord.pdb",
            "notes": "ok",
        }
    )
    audit = {
        **passing_audit(),
        "candidate_file_exists": True,
        "atom_count_parent": 10,
        "atom_count_candidate": 10,
        "missing_label_count": 0,
        "extra_label_count": 0,
        "max_atom_shift_A": 0.3,
        "max_non_anchor_atom_shift_A": 0.3,
        "omega_count": 4,
        "median_omega_trans_deviation_deg": 1.0,
    }
    row = join_manifest_audit(manifest, audit)
    assert row["variant_id"] == "rep"
    assert row["omega_policy"] == "fixed_180"
    assert row["safe_for_diffraction_scoring"] is True


def test_report_text_mentions_fixed_omega_and_deferred_sensitivity(tmp_path) -> None:
    results = pd.DataFrame(
        [
            {
                "variant_id": "rep",
                "fixed_torsion_delta_deg": 0.0,
                "max_ca_shift_A": 0.0,
                "max_backbone_bond_delta_A": 0.01,
                "max_backbone_angle_delta_deg": 1.0,
                "max_omega_trans_deviation_deg": 5.0,
                "safe_for_diffraction_scoring": True,
                "failure_reasons": "",
            }
        ]
    )
    text = build_report_text(results, tmp_path / "source.pdb")
    assert "fixed_180" in text
    assert "Omega sensitivity remains deferred" in text
