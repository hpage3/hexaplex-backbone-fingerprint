from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.audit_backbone_torsion_repeat import Residue
from scripts.generate_repeated_glu_mep_baseline_omega_variants import (
    OMEGA_MODE,
    identify_glu_mep_windows,
    manifest_row,
    safe_baseline_parent_glu_mep_rows,
    select_representative_rows,
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


def test_identify_glu_mep_windows_uses_coordinate_order() -> None:
    residues = {
        "B": [residue("B", 9, "MEP"), residue("B", 3, "GLU"), residue("B", 1, "MEP")],
        "F": [residue("F", 20, "GLU"), residue("F", 10, "MEP")],
    }
    windows = identify_glu_mep_windows(residues)
    assert [(w.chain, w.start_index, w.resseq_i, w.resseq_j) for w in windows] == [
        ("B", 1, 3, 1),
        ("F", 0, 20, 10),
    ]


def test_filter_safe_baseline_parent_rows_excludes_fixed_180_and_unsafe() -> None:
    df = pd.DataFrame(
        {
            "attempt_id": ["safe", "fixed", "unsafe", "other_repeat"],
            "omega_mode": ["baseline_parent", "fixed_180", "baseline_parent", "baseline_parent"],
            "repeat_type": ["GLU->MEP", "GLU->MEP", "GLU->MEP", "CYP->GLU"],
            "geometry_safe": ["True", "True", "False", "True"],
            "fixed_torsion_delta_deg": [1.0, 1.0, 2.0, 0.0],
            "solve_mode": ["one_torsion"] * 4,
        }
    )
    rows = safe_baseline_parent_glu_mep_rows(df)
    assert rows["attempt_id"].tolist() == ["safe"]


def test_select_representative_rows_prefers_one_per_delta_and_one_torsion() -> None:
    rows = pd.DataFrame(
        {
            "attempt_id": ["two", "one", "base"],
            "fixed_torsion_delta_deg": [1.0, 1.0, 0.0],
            "solve_mode": ["two_torsion", "one_torsion", "one_torsion"],
            "endpoint_error_A": [0.01, 0.2, 0.0],
        }
    )
    selected = select_representative_rows(rows)
    assert selected["attempt_id"].tolist() == ["base", "one"]


def test_variant_id_includes_baseline_parent_metadata() -> None:
    row = pd.Series({"fixed_torsion_delta_deg": -2.0, "solve_mode": "one_torsion"})
    assert variant_id(row) == "repeated_GLU_MEP_baseline_parent_one_torsion_phi0_deg_m2"


def test_manifest_row_bookkeeping_and_omega_mode(tmp_path) -> None:
    row = pd.Series(
        {
            "fixed_torsion_name": "phi0_deg",
            "fixed_torsion_delta_deg": 1.0,
            "solve_mode": "one_torsion",
            "endpoint_error_A": 0.02,
        }
    )
    out = manifest_row(
        row,
        "variant",
        tmp_path / "source.pdb",
        attempted=42,
        applied=41,
        skipped=["skip"],
        max_ca_shift=0.0,
        coordinate_path=tmp_path / "variant.pdb",
    )
    assert out["omega_mode"] == OMEGA_MODE
    assert out["attempted_window_count"] == 42
    assert out["applied_window_count"] == 41
    assert out["skipped_window_count"] == 1
