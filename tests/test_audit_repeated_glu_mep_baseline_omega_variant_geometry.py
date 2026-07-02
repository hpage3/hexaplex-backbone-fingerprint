from __future__ import annotations

import pandas as pd

from scripts.audit_constrained_phi_psi_candidate_geometry import Atom
from scripts.audit_repeated_glu_mep_baseline_omega_variant_geometry import (
    OMEGA_MODE,
    atom_label_sets_match,
    build_report_text,
    classify_audit,
    join_manifest_audit,
    max_ca_shift,
)


def atom(name: str, x: float, serial: int = 1) -> Atom:
    return Atom(
        record="ATOM",
        serial=serial,
        name=name,
        altloc="",
        resname="GLU",
        chain="B",
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
        "max_omega_trans_deviation_deg": 12.0,
    }


def test_atom_label_matching_and_ca_shift() -> None:
    parent = [atom("CA", 0.0, 1), atom("C", 1.0, 2)]
    variant = [atom("CA", 0.0, 1), atom("C", 2.0, 2)]
    assert atom_label_sets_match(parent, variant)
    assert max_ca_shift(parent, variant) == 0.0


def test_safe_fail_classification_and_failure_reason() -> None:
    safe, reason = classify_audit(passing_audit())
    assert safe
    assert reason == ""
    audit = passing_audit()
    audit["max_omega_trans_deviation_deg"] = 20.0
    safe, reason = classify_audit(audit)
    assert not safe
    assert reason == "omega_trans_deviation_exceeds_threshold"


def test_join_manifest_audit_includes_baseline_parent_mode() -> None:
    manifest = pd.Series(
        {
            "variant_id": "v",
            "fixed_torsion_delta_deg": 1.0,
            "solve_mode": "one_torsion",
            "omega_mode": OMEGA_MODE,
            "attempted_window_count": 42,
            "applied_window_count": 42,
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
        "max_atom_shift_A": 0.2,
        "max_non_anchor_atom_shift_A": 0.2,
        "omega_count": 4,
        "median_omega_trans_deviation_deg": 2.0,
    }
    row = join_manifest_audit(manifest, audit)
    assert row["omega_mode"] == OMEGA_MODE
    assert row["safe_for_diffraction_scoring"] is True


def test_report_mentions_baseline_parent_not_fixed_180_only(tmp_path) -> None:
    results = pd.DataFrame(
        [
            {
                "variant_id": "v",
                "fixed_torsion_delta_deg": 0.0,
                "solve_mode": "one_torsion",
                "omega_mode": OMEGA_MODE,
                "max_ca_shift_A": 0.0,
                "max_backbone_bond_delta_A": 0.01,
                "max_backbone_angle_delta_deg": 1.0,
                "max_omega_trans_deviation_deg": 12.0,
                "safe_for_diffraction_scoring": True,
                "failure_reason": "",
            }
        ]
    )
    text = build_report_text(results, tmp_path / "source.pdb")
    assert "baseline_parent" in text
    assert "not `fixed_180`" in text
    assert "does not require omega to be exactly 180 degrees" in text
