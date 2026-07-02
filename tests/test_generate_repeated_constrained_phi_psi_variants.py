from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.audit_backbone_torsion_repeat import Residue
from scripts.generate_repeated_constrained_phi_psi_variants import (
    OMEGA_POLICY,
    RepeatWindow,
    identify_cyp_glu_windows,
    manifest_row,
    safe_local_cyp_glu_rows,
    variant_id,
)


def residue(chain: str, resseq: int, resname: str) -> Residue:
    return Residue(
        chain=chain,
        resseq=resseq,
        resname=resname,
        atoms={"CA": np.array([float(resseq), 0.0, 0.0])},
        atom_names_in_order=("CA",),
    )


def test_identify_cyp_glu_windows_uses_coordinate_order_not_raw_residue_ids() -> None:
    residues = {
        "A": [
            residue("A", 20, "GLU"),
            residue("A", 10, "CYP"),
            residue("A", 30, "GLU"),
        ],
        "C": [
            residue("C", 500, "CYP"),
            residue("C", 100, "GLU"),
        ],
    }
    windows = identify_cyp_glu_windows(residues)
    assert [(w.chain, w.start_index, w.resseq_i, w.resseq_j) for w in windows] == [
        ("A", 1, 10, 30),
        ("C", 0, 500, 100),
    ]


def test_filter_safe_local_cyp_glu_one_torsion_rows() -> None:
    cd_scores = pd.DataFrame(
        {
            "candidate_id": ["safe_cyp", "unsafe_cyp", "safe_glu"],
            "repeat_type": ["CYP->GLU", "CYP->GLU", "GLU->MEP"],
            "solve_mode": ["one_torsion", "one_torsion", "one_torsion"],
            "fixed_torsion_name": ["phi0_deg", "phi0_deg", "phi0_deg"],
            "fixed_torsion_delta_deg": [1.0, 2.0, 0.0],
        }
    )
    audit = pd.DataFrame(
        {
            "candidate_id": ["safe_cyp", "unsafe_cyp", "safe_glu"],
            "safe_for_diffraction_scoring": ["True", "False", "True"],
        }
    )
    rows = safe_local_cyp_glu_rows(cd_scores, audit)
    assert rows["candidate_id"].tolist() == ["safe_cyp"]


def test_variant_id_generation() -> None:
    assert variant_id(-2.0) == "repeated_CYP_GLU_one_torsion_phi0_deg_m2"
    assert variant_id(3.0) == "repeated_CYP_GLU_one_torsion_phi0_deg_p3"


def test_manifest_row_bookkeeping_and_fixed_omega_policy(tmp_path) -> None:
    row = pd.Series(
        {
            "fixed_torsion_name": "phi0_deg",
            "fixed_torsion_delta_deg": 1.0,
            "endpoint_error_A": 0.02,
        }
    )
    out = manifest_row(
        row,
        "variant",
        tmp_path / "source.pdb",
        attempted=9,
        applied=8,
        skipped=["one skipped"],
        max_ca_shift=0.0,
        coordinate_path=tmp_path / "variant.pdb",
    )
    assert out["omega_policy"] == OMEGA_POLICY
    assert out["attempted_window_count"] == 9
    assert out["applied_window_count"] == 8
    assert out["skipped_window_count"] == 1
    assert out["max_ca_anchor_shift_A"] == 0.0
