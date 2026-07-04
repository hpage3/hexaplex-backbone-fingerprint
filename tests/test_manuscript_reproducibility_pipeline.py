from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.run_manuscript_reproducibility_pipeline import (
    MANUSCRIPT_FIGURES,
    MANUSCRIPT_METRICS,
    MANUSCRIPT_REPORTS,
    build_summary_report,
    claim_map_dataframe,
    copy_if_present,
    ensure_dirs,
    item_status_dataframe,
    manuscript_items,
    surviving_family_dataframe,
)


def test_item_claim_map_contains_items_1_through_8() -> None:
    items = manuscript_items()
    assert [item.item for item in items] == list(range(1, 9))

    claim_map = claim_map_dataframe()
    assert claim_map["item"].tolist() == list(range(1, 9))
    assert "pNAB" in claim_map.loc[claim_map["item"] == 1, "claim"].iloc[0]


def test_pnab_caveat_and_item6_surviving_family_are_preserved() -> None:
    claim_map = claim_map_dataframe()
    item1 = claim_map[claim_map["item"] == 1].iloc[0]
    item6 = claim_map[claim_map["item"] == 6].iloc[0]

    assert "insufficient_data" in item1["current_status"]
    assert "does not eliminate parallel" in item1["caveats"]
    assert "omega-clean rise-compressed plateau" in item6["current_status"]
    assert "pNAB" in item6["caveats"]


def test_item8_rise_compression_numbers_are_present() -> None:
    claim_map = claim_map_dataframe()
    item8 = claim_map[claim_map["item"] == 8].iloc[0]
    text = " ".join(str(item8[column]) for column in claim_map.columns)
    for phrase in ["5.7454", "7.2756", "0.1698", "5.6422", "0.0667", "7.1923"]:
        assert phrase in text


def test_pipeline_output_paths_are_created() -> None:
    ensure_dirs()
    assert MANUSCRIPT_METRICS.exists()
    assert MANUSCRIPT_REPORTS.exists()
    assert MANUSCRIPT_FIGURES.exists()


def test_missing_data_and_reused_existing_output_statuses(tmp_path: Path) -> None:
    missing = tmp_path / "missing.csv"
    present = tmp_path / "present.csv"
    present.write_text("x\n1\n", encoding="utf-8")
    dest = tmp_path / "dest"

    assert copy_if_present(missing, dest) == "missing_data"
    assert copy_if_present(present, dest) == "reused_existing_output"
    assert (dest / "present.csv").exists()


def test_item_status_supports_pipeline_statuses() -> None:
    step_rows = [
        {"pipeline_step": "A", "output_status": "reused_existing_output"},
        {"pipeline_step": "C", "output_status": "reused_existing_output"},
        {"pipeline_step": "D", "output_status": "missing_data"},
        {"pipeline_step": "E", "output_status": "generated_and_copied"},
    ]
    status = item_status_dataframe(step_rows)

    assert status.loc[status["item"] == 1, "pipeline_status"].iloc[0] == "reused_existing_output"
    assert status.loc[status["item"] == 6, "pipeline_status"].iloc[0] == "generated_and_copied"
    assert status.loc[status["item"] == 7, "pipeline_status"].iloc[0] == "missing_data"


def test_surviving_family_summary_fallback_and_existing_fixture(tmp_path: Path) -> None:
    missing = surviving_family_dataframe(tmp_path / "missing.csv")
    assert missing.iloc[0]["status"] == "missing_data"

    source = tmp_path / "survivor.csv"
    pd.DataFrame(
        [
            {
                "surviving_model_family": "omega-clean rise-compressed plateau",
                "status": "strongest_current_surviving_family_with_pnab_caveat",
            }
        ]
    ).to_csv(source, index=False)
    existing = surviving_family_dataframe(source)
    assert existing.iloc[0]["surviving_model_family"] == "omega-clean rise-compressed plateau"
    assert "pNAB caveat" in existing.iloc[0]["publication_track_note"]


def test_summary_report_wording_contains_required_phrases() -> None:
    item_status = claim_map_dataframe()
    item_status["publication_track_status"] = "documented"
    item_status["pipeline_status"] = "reused_existing_output"
    survivor = pd.DataFrame(
        [
            {
                "surviving_model_family": "omega-clean rise-compressed plateau",
                "status": "strongest_current_surviving_family_with_pnab_caveat",
            }
        ]
    )
    step_status = pd.DataFrame(
        [
            {
                "pipeline_step": "A",
                "pipeline_name": "pNAB parallel/anti-parallel audit",
                "script_status": "not_rerun_reused_existing_output",
                "output_status": "reused_existing_output",
                "copied_output_count": 5,
                "missing_output_count": 0,
            }
        ]
    )

    text = build_summary_report(item_status, survivor, step_status)
    for phrase in [
        "publication track",
        "pNAB",
        "insufficient_data",
        "omega-clean rise-compressed plateau",
        "C/D agreement is necessary but not sufficient",
        "PI-level interpretation",
        "No exploratory files were deleted or moved",
    ]:
        assert phrase in text
