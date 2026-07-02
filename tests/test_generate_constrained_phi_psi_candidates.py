from pathlib import Path

import pandas as pd

from scripts.generate_constrained_phi_psi_candidates import (
    OMEGA_POLICY,
    accepted_closure_rows,
    candidate_id,
    generate_candidates,
    manifest_row,
    output_paths,
)


def closure_row(success=True):
    return pd.Series(
        {
            "model_id": "toy",
            "chain_id": "A",
            "repeat_start_index": 1,
            "residue_names": "CYP->GLU",
            "solve_mode": "one_torsion",
            "fixed_torsion_name": "phi0_deg",
            "fixed_torsion_delta_deg": 0.0,
            "solved_torsion_1_name": "psi0_deg",
            "solved_torsion_1_delta_deg": 0.0,
            "solved_torsion_2_name": "",
            "solved_torsion_2_delta_deg": float("nan"),
            "omega_fixed_deg": 180.0,
            "endpoint_error_A": 0.0,
            "closure_success": success,
            "notes": "fixture",
        }
    )


def test_filtering_accepted_closure_rows():
    df = pd.DataFrame([closure_row(True), closure_row(False)])
    accepted = accepted_closure_rows(df)
    assert len(accepted) == 1
    assert bool(accepted.iloc[0]["closure_success"]) is True


def test_candidate_id_generation_and_output_paths(tmp_path: Path):
    cid = candidate_id(closure_row(True), 7)
    assert cid.startswith("cand_007_A_CYP_GLU_one_torsion_phi0_deg")
    pdb_path, xyz_path = output_paths(tmp_path, cid)
    assert pdb_path.name.endswith(".pdb")
    assert xyz_path.name.endswith(".xyz")


def test_manifest_row_includes_fixed_omega_policy():
    row = manifest_row(closure_row(True), "cand_001", "coord.pdb", "notes")
    assert row["omega_policy"] == OMEGA_POLICY
    assert row["candidate_id"] == "cand_001"


def write_toy_pdb(path: Path):
    lines = [
        "ATOM      1  N   PRE A   1      -1.000  -1.000   0.000  1.00  0.00           N",
        "ATOM      2  CA  PRE A   1      -0.500  -0.500   0.000  1.00  0.00           C",
        "ATOM      3  C   PRE A   1       0.000   0.000   0.000  1.00  0.00           C",
        "ATOM      4  O   PRE A   1       0.000   0.000   1.000  1.00  0.00           O",
        "ATOM      5  N   CYP A   2       1.000   0.000   0.000  1.00  0.00           N",
        "ATOM      6  CA  CYP A   2       1.000   1.000   0.000  1.00  0.00           C",
        "ATOM      7  C   CYP A   2       2.000   1.000   0.000  1.00  0.00           C",
        "ATOM      8  O   CYP A   2       2.000   1.000   1.000  1.00  0.00           O",
        "ATOM      9  N   GLU A   3       2.000   2.000   0.000  1.00  0.00           N",
        "ATOM     10  CA  GLU A   3       3.000   2.000   0.000  1.00  0.00           C",
        "ATOM     11  C   GLU A   3       3.000   3.000   0.000  1.00  0.00           C",
        "ATOM     12  O   GLU A   3       3.000   3.000   1.000  1.00  0.00           O",
        "ATOM     13  N   CYP A   4       4.000   3.000   0.000  1.00  0.00           N",
        "ATOM     14  CA  CYP A   4       4.000   4.000   0.000  1.00  0.00           C",
        "ATOM     15  C   CYP A   4       5.000   4.000   0.000  1.00  0.00           C",
        "ATOM     16  O   CYP A   4       5.000   4.000   1.000  1.00  0.00           O",
        "END",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def test_candidate_generation_preserves_ca_anchors_and_metadata(tmp_path: Path):
    source_pdb = tmp_path / "toy.pdb"
    write_toy_pdb(source_pdb)
    closure_csv = tmp_path / "closure.csv"
    pd.DataFrame([closure_row(True)]).to_csv(closure_csv, index=False)
    manifest = generate_candidates(
        source_pdb,
        closure_csv,
        tmp_path / "candidates",
        tmp_path / "manifest.csv",
        tmp_path / "report.md",
        max_candidates=10,
    )
    assert len(manifest) == 1
    assert manifest.iloc[0]["omega_policy"] == "fixed_180"
    assert manifest.iloc[0]["max_ca_anchor_shift_A"] == 0.0
    assert Path(manifest.iloc[0]["coordinate_path"]).exists()
    assert Path(manifest.iloc[0]["xyz_path"]).exists()
