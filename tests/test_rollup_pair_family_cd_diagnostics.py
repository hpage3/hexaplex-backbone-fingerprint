from __future__ import annotations

import pandas as pd

from scripts.rollup_pair_family_cd_diagnostics import rollup, summarize_file


def write_summary(path, model_id: str, rows: list[dict]):
    pd.DataFrame(
        [
            {
                "model_id": model_id,
                "family": row["family"],
                "C_pair_count": row.get("C_pair_count", 0),
                "D_pair_count": row.get("D_pair_count", 0),
                "C_profile_max_intensity": row.get("C_profile_max_intensity", 0.0),
                "D_profile_max_intensity": row.get("D_profile_max_intensity", 0.0),
            }
            for row in rows
        ]
    ).to_csv(path, index=False)


def test_summarize_file_identifies_top_families(tmp_path):
    path = tmp_path / "toy_pair_family_cd_pair_family_cd_summary.csv"
    write_summary(
        path,
        "toy",
        [
            {"family": "same_strand_plusminus1_repeat", "C_pair_count": 10, "D_pair_count": 1},
            {"family": "all_cross_strand", "C_pair_count": 2, "D_pair_count": 12},
            {"family": "all_same_strand", "C_profile_max_intensity": 8.0, "D_profile_max_intensity": 1.0},
            {"family": "adjacent_strand_same_register", "C_profile_max_intensity": 1.0, "D_profile_max_intensity": 9.0},
        ],
    )

    row = summarize_file(path)

    assert row["model_id"] == "toy"
    assert row["top_C_pair_count_family"] == "same_strand_plusminus1_repeat"
    assert row["top_D_pair_count_family"] == "all_cross_strand"
    assert row["top_C_profile_family"] == "all_same_strand"
    assert row["top_D_profile_family"] == "adjacent_strand_same_register"


def test_rollup_writes_csv_and_report(tmp_path):
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    write_summary(
        metrics_dir / "m1_pair_family_cd_pair_family_cd_summary.csv",
        "m1",
        [
            {"family": "same_strand_plusminus1_repeat", "C_pair_count": 10},
            {"family": "all_cross_strand", "D_pair_count": 8},
            {"family": "alternating_interfaces_AB_CD_EF", "C_pair_count": 5, "D_pair_count": 4},
            {"family": "alternating_interfaces_BC_DE_FA", "C_pair_count": 2, "D_pair_count": 4},
        ],
    )
    write_summary(
        metrics_dir / "m2_pair_family_cd_pair_family_cd_summary.csv",
        "m2",
        [
            {"family": "all_same_strand", "C_pair_count": 6},
            {"family": "adjacent_strand_same_register", "D_pair_count": 7},
            {"family": "alternating_interfaces_AB_CD_EF", "C_pair_count": 1, "D_pair_count": 3},
            {"family": "alternating_interfaces_BC_DE_FA", "C_pair_count": 4, "D_pair_count": 1},
        ],
    )

    df, csv_path, report_path = rollup(metrics_dir, "roll")

    assert len(df) == 2
    assert csv_path.exists()
    assert report_path.exists()
    assert "alternating_AB_CD_EF_minus_BC_DE_FA_C_pair_count" in df.columns
    assert "Diagnostic files summarized: 2" in report_path.read_text(encoding="utf-8")
