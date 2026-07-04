from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.audit_manuscript_pipeline_coverage import (
    build_report,
    classify_report_status,
    infer_generating_script,
    known_report_mappings,
    known_script_mappings,
    report_row,
)


def test_known_report_mapping_logic() -> None:
    mappings = known_report_mappings()
    assert mappings["pnab_parallel_antiparallel_audit_report.md"].manuscript_items == "1"
    assert mappings["item6_candidate_filter_funnel_report.md"].manuscript_items == "4,6"
    assert mappings["omega_clean_torsion_boundary_report.md"].manuscript_items == "7"
    assert mappings["omega_clean_rise_compression_report.md"].manuscript_items == "3,5,8"


def test_known_script_mapping_logic() -> None:
    scripts = known_script_mappings()
    assert scripts["scripts/audit_pnab_parallel_antiparallel_models.py"] == "pnab_parallel_antiparallel_audit_report.md"
    assert scripts["scripts/report_item6_candidate_filter_funnel.py"] == "item6_candidate_filter_funnel_report.md"
    assert scripts["scripts/run_manuscript_reproducibility_pipeline.py"] == "manuscript_reproducibility_summary.md"


def test_statuses_include_publication_coverage_and_review() -> None:
    assert (
        classify_report_status(
            Path("outputs/reports/pnab_parallel_antiparallel_audit_report.md"),
            known_report_mappings()["pnab_parallel_antiparallel_audit_report.md"],
            "scripts/audit_pnab_parallel_antiparallel_models.py",
            True,
            True,
        )
        == "reproducible_in_publication_pipeline"
    )
    assert classify_report_status(Path("README.md"), None, "", False, False) == "orphan_report_no_script"
    assert classify_report_status(Path("some.md"), None, "scripts/some.py", False, False) == "needs_review"


def test_infer_generating_script_prefers_known_mapping() -> None:
    mapping = known_report_mappings()["item6_candidate_filter_funnel_report.md"]
    assert infer_generating_script(Path("outputs/reports/item6_candidate_filter_funnel_report.md"), mapping) == (
        "scripts/report_item6_candidate_filter_funnel.py"
    )


def test_report_row_maps_key_reports_to_items() -> None:
    row = report_row(Path("outputs/reports/pnab_parallel_antiparallel_audit_report.md"))
    assert row["inferred_manuscript_items"] == "1"
    assert row["included_in_publication_pipeline"] is True

    item6 = report_row(Path("outputs/reports/item6_candidate_filter_funnel_report.md"))
    assert item6["inferred_manuscript_items"] == "4,6"

    torsion = report_row(Path("outputs/reports/omega_clean_torsion_boundary_report.md"))
    assert torsion["inferred_manuscript_items"] == "7"

    rise = report_row(Path("outputs/reports/omega_clean_rise_compression_report.md"))
    assert rise["inferred_manuscript_items"] == "3,5,8"


def test_coverage_report_wording() -> None:
    df = pd.DataFrame(
        [
            {
                "report_name": "pnab_parallel_antiparallel_audit_report.md",
                "inferred_manuscript_items": "1",
                "inferred_generating_script": "scripts/audit_pnab_parallel_antiparallel_models.py",
                "status": "reproducible_in_publication_pipeline",
            },
            {
                "report_name": "some_unmapped_report.md",
                "inferred_manuscript_items": "",
                "inferred_generating_script": "",
                "status": "needs_review",
            },
        ]
    )
    text = build_report(df)
    for phrase in [
        "pipeline coverage",
        "manuscript-supporting analysis documents",
        "reproducible_in_publication_pipeline",
        "needs_review",
    ]:
        assert phrase in text
