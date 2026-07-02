from pathlib import Path

import pandas as pd

from scripts.rollup_ideal_hexaflex_atom_class_cd import pair_value, rollup, summarize_file


def write_summary(path: Path, model_id: str) -> None:
    pd.DataFrame(
        [
            {
                "model_id": model_id,
                "atom_class_1": "peptide_plane",
                "atom_class_2": "peptide_plane",
                "geometry_family": "all_cross_strand",
                "C_pair_count": 5,
                "D_pair_count": 2,
                "C_profile_max_intensity": 10.0,
                "C_profile_peak_d_A": 5.55,
                "D_profile_max_intensity": 3.0,
                "D_profile_peak_d_A": 7.2,
            },
            {
                "model_id": model_id,
                "atom_class_1": "backbone",
                "atom_class_2": "carboxylate",
                "geometry_family": "all_cross_strand",
                "C_pair_count": 1,
                "D_pair_count": 8,
                "C_profile_max_intensity": 2.0,
                "C_profile_peak_d_A": 5.6,
                "D_profile_max_intensity": 12.0,
                "D_profile_peak_d_A": 7.3,
            },
        ]
    ).to_csv(path, index=False)


def test_pair_value_is_order_insensitive(tmp_path: Path):
    summary = tmp_path / "toy_atom_class_cd_summary.csv"
    write_summary(summary, "toy")
    df = pd.read_csv(summary)
    assert pair_value(df, "carboxylate", "backbone", "all_cross_strand", "D_pair_count") == 8.0


def test_summarize_file_identifies_top_rows(tmp_path: Path):
    summary = tmp_path / "toy_atom_class_cd_summary.csv"
    write_summary(summary, "toy")
    row = summarize_file(summary, "toy_variant")
    assert row["top_C_pair_count_label"] == "peptide_plane x peptide_plane / all_cross_strand"
    assert row["top_D_pair_count_label"] == "backbone x carboxylate / all_cross_strand"


def test_rollup_writes_outputs_from_fixture(tmp_path: Path):
    metrics = tmp_path / "metrics"
    reports = tmp_path / "reports"
    figures = tmp_path / "figures"
    metrics.mkdir()
    manifest = tmp_path / "manifest.csv"
    model_id = "ideal_hexaflex_full_pair_family_cd"
    pd.DataFrame([{"variant": "full", "model_id": model_id, "written": True}]).to_csv(manifest, index=False)
    write_summary(metrics / f"{model_id}_atom_class_cd_summary.csv", model_id)
    df = rollup(metrics, manifest, metrics, reports, figures)
    assert len(df) == 1
    assert (metrics / "ideal_hexaflex_atom_class_cd_rollup.csv").exists()
    assert (reports / "ideal_hexaflex_atom_class_cd_rollup_report.md").exists()
    assert (figures / "ideal_hexaflex_atom_class_cd_C_D_heatmap.png").exists()
