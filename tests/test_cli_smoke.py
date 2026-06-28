import subprocess
import sys
from pathlib import Path


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
    assert "Backbone fingerprint analysis complete" in result.stdout
