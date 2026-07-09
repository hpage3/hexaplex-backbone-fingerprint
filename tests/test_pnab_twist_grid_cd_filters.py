from __future__ import annotations

import math

import pandas as pd
import pytest

from scripts.score_pnab_twist_grid_cd_filters import (
    build_report,
    build_scored_candidates,
    cd_agreement_status,
    conclusion_logic,
    d_guardrail_status,
    recover_score,
    rise_window_status,
    select_candidate_sets,
    summarize_by_twist,
)


def test_loading_and_selecting_candidate_sets() -> None:
    inventory = pd.DataFrame(
        [
            {
                "candidate_id": "raw18",
                "path": "inputs/raw18.pdb",
                "candidate_set": "pnab_3p38_twist_grid",
                "inferred_twist_deg": 18,
                "inferred_rise_A": 3.4,
                "twist_source": "dir",
                "rise_source": "path",
            },
            {
                "candidate_id": "unknown",
                "path": "inputs/unknown.pdb",
                "candidate_set": "unknown",
                "inferred_twist_deg": 30,
                "inferred_rise_A": 3.4,
                "twist_source": "name",
                "rise_source": "name",
            },
        ]
    )
    # Use monkeypatching would be overkill; keep this logic test focused by
    # marking paths as existing after selection via a temporary path-free check.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("scripts.score_pnab_twist_grid_cd_filters.ROOT", type("FakeRoot", (), {"__truediv__": lambda self, other: type("FakePath", (), {"exists": lambda self: True})()})())
        selected = select_candidate_sets(inventory)
    assert len(selected) == 1
    assert selected.iloc[0]["candidate_set"] == "pnab_3p38_twist_grid"


def test_recover_scores_by_path_and_name() -> None:
    existing = pd.DataFrame(
        [
            {
                "path_key": "outputs\\foo\\candidate_a.pdb",
                "name_key": "candidate_a",
                "observed_C_d_A": 5.6,
                "observed_D_d_A": 7.3,
            }
        ]
    )
    by_path = pd.Series({"pdb_path": "outputs/foo/candidate_a.pdb", "candidate_id": "different"})
    by_name = pd.Series({"pdb_path": "outputs/foo/missing.pdb", "candidate_id": "candidate_a"})
    assert recover_score(by_path, existing) is not None
    assert recover_score(by_name, existing) is not None


def test_missing_scores_and_scoring_failure_handling() -> None:
    candidate = pd.DataFrame(
        [
            {
                "candidate_id": "raw18",
                "pdb_path": "missing.pdb",
                "candidate_set": "pnab_3p38_twist_grid",
                "final_twist_deg": 18,
                "final_rise_A": 3.4,
                "inference_source": "dir",
                "provenance_confidence": "path",
            }
        ]
    )
    scored = build_scored_candidates(candidate, pd.DataFrame(), score_missing=False)
    assert scored.iloc[0]["score_status"] == "missing_score"
    assert scored.iloc[0]["cd_agreement_status"] == "unknown"


def test_filter_classifiers() -> None:
    assert rise_window_status(3.3) == "pass"
    assert rise_window_status(3.4) == "pass"
    assert rise_window_status(3.29).startswith("fail")
    assert rise_window_status(None) == "unknown"
    assert cd_agreement_status(0.05) == "pass"
    assert cd_agreement_status(0.12) == "borderline_reference_like"
    assert cd_agreement_status(0.3) == "fail"
    assert d_guardrail_status(7.1923).startswith("fail")
    assert d_guardrail_status(7.2756) == "pass"


def test_twist_grouping_summary() -> None:
    scored = pd.DataFrame(
        [
            {
                "candidate_name": "a",
                "final_twist_deg": 30.0,
                "score_status": "scored",
                "rise_window_status": "pass",
                "cd_agreement_status": "pass",
                "D_guardrail_status": "pass",
                "physical_filter_status": "unknown",
                "all_available_filters_status": "pass_available_filters",
                "combined_C_D_error": 0.05,
                "observed_C_d_A": 5.6,
                "observed_D_d_A": 7.3,
            },
            {
                "candidate_name": "b",
                "final_twist_deg": 30.0,
                "score_status": "scoring_failed",
                "rise_window_status": "pass",
                "cd_agreement_status": "unknown",
                "D_guardrail_status": "unknown",
                "physical_filter_status": "unknown",
                "all_available_filters_status": "unknown",
                "combined_C_D_error": math.nan,
                "observed_C_d_A": math.nan,
                "observed_D_d_A": math.nan,
            },
        ]
    )
    summary = summarize_by_twist(scored)
    assert summary.iloc[0]["candidate_count"] == 2
    assert summary.iloc[0]["scored_candidate_count"] == 1
    assert summary.iloc[0]["scoring_failed_count"] == 1
    assert summary.iloc[0]["cd_pass_count"] == 1


def test_conclusion_logic_variants() -> None:
    def row(twist: float, scored_count: int, pass_count: int) -> dict[str, object]:
        return {
            "twist_deg": twist,
            "scored_candidate_count": scored_count,
            "all_available_filters_pass_count": pass_count,
        }

    publication = pd.DataFrame(
        [
            {
                "candidate_set": "omega_clean_publication_track",
                "score_status": "scored",
                "combined_C_D_error": 0.06665,
            }
        ]
    )
    narrowed_30 = pd.DataFrame([row(t, 1, 1 if t == 30 else 0) for t in range(18, 33)])
    assert conclusion_logic(narrowed_30, publication) == "narrowed_to_30_degree_like"

    narrowed_family = pd.DataFrame([row(t, 1, 1 if t >= 28 else 0) for t in range(18, 33)])
    assert conclusion_logic(narrowed_family, publication) == "narrowed_to_28_32_degree_family"

    broad = pd.DataFrame([row(t, 1, 1) for t in range(18, 33)])
    assert conclusion_logic(broad, publication) == "broad_18_32_remains_plausible"

    insufficient = pd.DataFrame([row(t, 1, 1 if t == 30 else 0) for t in [28, 29, 30, 31, 32]])
    assert conclusion_logic(insufficient, publication) == "insufficient_scoring_data"

    bad_publication = pd.DataFrame(
        [{"candidate_set": "omega_clean_publication_track", "score_status": "scored", "combined_C_D_error": 0.5}]
    )
    assert conclusion_logic(broad, bad_publication) == "scoring_method_needs_validation"


def test_report_wording_includes_required_cautions() -> None:
    scored = pd.DataFrame(
        [
            {
                "candidate_name": "raw18",
                "candidate_set": "pnab_3p38_twist_grid",
                "score_status": "missing_score",
                "score_source": "not_scored",
                "physical_filter_status": "unknown",
            }
        ]
    )
    by_twist = pd.DataFrame(
        [
            {
                "twist_deg": 18,
                "candidate_count": 1,
                "scored_candidate_count": 0,
                "cd_pass_count": 0,
                "all_available_filters_pass_count": 0,
                "best_C_D_error": math.nan,
                "best_C": math.nan,
                "best_D": math.nan,
                "twist_status": "no_scored_candidates",
            }
        ]
    )
    report = build_report(scored, by_twist, "insufficient_scoring_data")
    for phrase in [
        "corrects the earlier twist-tightening scan",
        "raw pNAB twist-grid",
        "18-32",
        "28-32",
        "3.3-3.4 A",
        "C/D agreement is necessary but not sufficient",
        "heavy-atom plausibility proxy",
        "Candidate elimination should not rely on any single filter",
        "insufficient_scoring_data",
    ]:
        assert phrase in report
