from pathlib import Path

import pandas as pd

from scripts.rollup_rich_coordinate_cd_diagnostics import (
    numeric_value,
    read_pdb_coordinates,
    rollup,
    top_family,
)


def test_top_family_and_numeric_value():
    df = pd.DataFrame(
        [
            {"family": "same_strand_plusminus1_repeat", "C_pair_count": 5, "D_pair_count": 1},
            {"family": "all_cross_strand", "C_pair_count": 2, "D_pair_count": 8},
        ]
    )
    assert top_family(df, "C_pair_count") == ("same_strand_plusminus1_repeat", 5.0)
    assert top_family(df, "D_pair_count") == ("all_cross_strand", 8.0)
    assert numeric_value(df, "all_cross_strand", "D_pair_count") == 8.0


def test_read_pdb_coordinates_filters_hydrogen(tmp_path: Path):
    pdb = tmp_path / "toy.pdb"
    pdb.write_text(
        "\n".join(
            [
                "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N",
                "ATOM      2  H   ALA A   1       1.000   0.000   0.000  1.00  0.00           H",
                "END",
            ]
        )
        + "\n",
        encoding="ascii",
    )
    assert read_pdb_coordinates(pdb, exclude_hydrogen=False).shape == (2, 3)
    assert read_pdb_coordinates(pdb, exclude_hydrogen=True).shape == (1, 3)


def test_rollup_with_temporary_pair_family_fixture(tmp_path: Path):
    metrics = tmp_path / "metrics"
    reports = tmp_path / "reports"
    figures = tmp_path / "figures"
    metrics.mkdir()
    pdb = tmp_path / "toy_full.pdb"
    pdb.write_text(
        "\n".join(
            [
                "ATOM      1  N   GLU A   1       0.000   0.000   0.000  1.00  0.00           N",
                "ATOM      2  CA  GLU A   1       1.000   0.000   0.000  1.00  0.00           C",
                "ATOM      3  C   GLU B   1       0.000   7.300   0.000  1.00  0.00           C",
                "END",
            ]
        )
        + "\n",
        encoding="ascii",
    )
    manifest = tmp_path / "variant_manifest.csv"
    pd.DataFrame(
        [
            {
                "variant": "full",
                "model_id": "toy_full_pair_family_cd",
                "pdb_path": str(pdb),
                "atom_count": 3,
                "warnings": "",
                "written": True,
            }
        ]
    ).to_csv(manifest, index=False)
    pd.DataFrame(
        [
            {
                "model_id": "toy_full_pair_family_cd",
                "family": "same_strand_plusminus1_repeat",
                "C_pair_count": 4,
                "D_pair_count": 1,
            },
            {
                "model_id": "toy_full_pair_family_cd",
                "family": "all_cross_strand",
                "C_pair_count": 1,
                "D_pair_count": 7,
            },
            {
                "model_id": "toy_full_pair_family_cd",
                "family": "alternating_interfaces_AB_CD_EF",
                "C_pair_count": 3,
                "D_pair_count": 2,
            },
            {
                "model_id": "toy_full_pair_family_cd",
                "family": "alternating_interfaces_BC_DE_FA",
                "C_pair_count": 0,
                "D_pair_count": 1,
            },
        ]
    ).to_csv(metrics / "toy_full_pair_family_cd_pair_family_cd_summary.csv", index=False)

    df = rollup(
        manifest,
        metrics,
        metrics,
        reports,
        figures,
        q_step=0.05,
    )
    assert df.iloc[0]["top_C_pair_family"] == "same_strand_plusminus1_repeat"
    assert df.iloc[0]["top_D_pair_family"] == "all_cross_strand"
    assert (metrics / "rich_coordinate_cd_rollup.csv").exists()
    assert (reports / "rich_coordinate_cd_rollup_report.md").exists()
