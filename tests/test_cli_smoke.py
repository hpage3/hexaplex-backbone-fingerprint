import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_cli_writes_expected_outputs(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    fixture = repo_root / "tests" / "fixtures" / "mini_peptide.pdb"
    outdir = tmp_path / "mini_out"

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "analyze_backbone_fingerprint.py"),
            str(fixture),
            "--outdir",
            str(outdir),
            "--label",
            "mini_peptide_test",
            "--c-target",
            "3.3",
            "--d-target",
            "6.6",
            "--tol",
            "0.4",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (outdir / "plane_features.csv").exists()
    assert (outdir / "band_candidate_pairs.csv").exists()
    assert (outdir / "summary.md").exists()
    assert (outdir / "rms_vs_cno_angle.png").exists()
    assert (outdir / "rms_vs_omega_deviation.png").exists()
    assert (outdir / "cno_angle_vs_omega_deviation.png").exists()
    assert "Backbone fingerprint analysis complete" in result.stdout

    plane_features = pd.read_csv(outdir / "plane_features.csv")
    for column in [
        "ca_i_plane_dist",
        "c_i_plane_dist",
        "o_i_plane_dist",
        "n_j_plane_dist",
        "ca_j_plane_dist",
        "hn_j_plane_dist",
        "cno_to_peptide_normal_angle_deg",
        "cno_centroid_to_peptide_plane_signed_dist",
        "omega_like_deg",
        "omega_deviation_from_trans_deg",
    ]:
        assert column in plane_features.columns
