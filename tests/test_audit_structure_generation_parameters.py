from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.audit_structure_generation_parameters import (
    build_report,
    build_scan_plan,
    likely_parameter_type,
    matched_terms,
    recommendation_option,
    run_audit,
    scan_repository,
)


def test_parameter_terms_are_detected_from_synthetic_files(tmp_path: Path) -> None:
    (tmp_path / "scripts").mkdir()
    source = tmp_path / "scripts" / "make_model.py"
    source.write_text(
        "\n".join(
            [
                "rise_A = 3.38",
                "twist_deg = 30.0",
                "helix_radius_A = 8.0",
                "strand_z_offset = 0.5",
                "orientation = 'anti_parallel'",
            ]
        ),
        encoding="utf-8",
    )

    hits, _plan = scan_repository(tmp_path)

    assert {"rise", "twist", "radius", "register", "orientation"}.issubset(
        set(hits["likely_parameter_type"])
    )
    assert (hits["confidence"] == "high").any()


def test_likely_parameter_type_mapping() -> None:
    assert likely_parameter_type("helical_rise") == "rise"
    assert likely_parameter_type("h_twist") == "twist"
    assert likely_parameter_type("helix_radius") == "radius"
    assert likely_parameter_type("anti_parallel") == "orientation"
    assert likely_parameter_type("register") == "register"


def test_matched_terms_detects_case_insensitive_pnab_and_register() -> None:
    terms = matched_terms("HelicalParameters from pNAB include register and sequence settings")

    assert "HelicalParameters" in terms
    assert "pNAB" in terms
    assert "register" in terms
    assert "sequence" in terms


def test_skipped_directories_are_not_scanned(tmp_path: Path) -> None:
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "generated.py").write_text("rise_A = 3.35\n", encoding="utf-8")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "source.py").write_text("twist_deg = 30.0\n", encoding="utf-8")

    hits, plan = scan_repository(tmp_path)

    assert "outputs/generated.py" not in set(hits["file_path"])
    assert "scripts/source.py" in set(hits["file_path"])
    assert any(path.name == "outputs" for path in plan.skipped_dirs)


def test_report_and_csv_outputs_are_written(tmp_path: Path) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "generator.py").write_text(
        "rise_A = 3.38\ntwist_deg = 30\nhelix_radius_A = 8\n",
        encoding="utf-8",
    )
    hits_csv = tmp_path / "out" / "hits.csv"
    summary_csv = tmp_path / "out" / "summary.csv"
    report = tmp_path / "out" / "report.md"

    hits, summary = run_audit(tmp_path, hits_csv, summary_csv, report)

    assert hits_csv.exists()
    assert summary_csv.exists()
    assert report.exists()
    assert len(hits) >= 3
    assert "recommendation" in summary
    report_text = report.read_text(encoding="utf-8")
    assert "Structure Generation Parameter Audit" in report_text
    assert "3.40/3.38/3.35 A rise family" in report_text


def test_recommendation_option_b_for_partial_parameterized_provenance() -> None:
    hits = pd.DataFrame(
        [
            {
                "likely_parameter_type": "rise",
                "confidence": "high",
                "context_snippet": "rise_A = 3.38",
            },
            {
                "likely_parameter_type": "twist",
                "confidence": "high",
                "context_snippet": "twist_deg = 30",
            },
            {
                "likely_parameter_type": "radius",
                "confidence": "high",
                "context_snippet": "helix_radius_A = 8",
            },
        ]
    )

    option, rationale = recommendation_option(hits)

    assert option == "Option B"
    assert "partial provenance" in rationale


def test_build_report_mentions_skipped_paths(tmp_path: Path) -> None:
    (tmp_path / "outputs").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "note.md").write_text("rise and register notes\n", encoding="utf-8")
    hits, plan = scan_repository(tmp_path)

    report = build_report(hits, plan, tmp_path)

    assert "Skipped generated/heavy directories" in report
    assert "outputs" in report
