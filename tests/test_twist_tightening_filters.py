from __future__ import annotations

import pandas as pd

from scripts.analyze_twist_tightening_filters import (
    build_report,
    combined_classification,
    conservative_twist_conclusion,
    rise_window_status,
    twist_status_from_counts,
    twist_summary,
)


def row(**kwargs) -> pd.Series:
    base = {
        "rise_window_status": "within_window",
        "twist_status": "inferred_30_degree_like_from_current_family",
        "cd_pass": True,
        "omega_geometry_pass": True,
        "torsion_envelope_pass": True,
        "hbond_pass": True,
    }
    base.update(kwargs)
    return pd.Series(base)


def test_rise_window_classification() -> None:
    assert rise_window_status(3.29) == "below_window"
    assert rise_window_status(3.35) == "within_window"
    assert rise_window_status(3.41) == "above_window"
    assert rise_window_status(None) == "unknown"


def test_combined_filter_classification() -> None:
    assert combined_classification(row()) == "twist_viable_current_filters"
    assert combined_classification(row(cd_pass=False)) == "twist_disfavored_cd"
    assert combined_classification(row(omega_geometry_pass=False)) == "twist_disfavored_geometry"
    assert combined_classification(row(torsion_envelope_pass=False)) == "twist_disfavored_geometry"
    assert combined_classification(row(hbond_pass=False)) == "twist_disfavored_hbond"
    assert combined_classification(row(rise_window_status="unknown")) == "twist_insufficient_data"
    assert combined_classification(row(twist_status="unknown")) == "twist_unknown_provenance"
    assert combined_classification(row(rise_window_status="below_window")) == "rejected_outside_rise_window"


def test_twist_grouping_and_summary_logic() -> None:
    candidates = pd.DataFrame(
        [
            {
                "candidate_name": "a",
                "twist_deg": 30.0,
                "rise_window_pass": True,
                "cd_pass": True,
                "omega_geometry_pass": True,
                "torsion_envelope_pass": True,
                "hbond_pass": True,
                "combined_filter_classification": "twist_viable_current_filters",
                "combined_CD_abs_error_A": 0.05,
            },
            {
                "candidate_name": "b",
                "twist_deg": 30.0,
                "rise_window_pass": True,
                "cd_pass": False,
                "omega_geometry_pass": True,
                "torsion_envelope_pass": True,
                "hbond_pass": True,
                "combined_filter_classification": "twist_disfavored_cd",
                "combined_CD_abs_error_A": 0.2,
            },
        ]
    )
    summary = twist_summary(candidates)

    assert len(summary) == 1
    assert int(summary.iloc[0]["candidate_count"]) == 2
    assert int(summary.iloc[0]["all_filters_pass_count"]) == 1
    assert summary.iloc[0]["best_candidate_name"] == "a"


def test_twist_status_from_counts() -> None:
    assert twist_status_from_counts(pd.Series({"all_filters_pass_count": 2, "candidate_count": 2})) == "strongly_supported_current_filters"
    assert twist_status_from_counts(pd.Series({"all_filters_pass_count": 1, "candidate_count": 2})) == "plausible_current_filters"
    assert twist_status_from_counts(pd.Series({"all_filters_pass_count": 0, "candidate_count": 2, "cd_pass_count": 0})) == "disfavored_current_filters"
    assert twist_status_from_counts(pd.Series({"all_filters_pass_count": 0, "candidate_count": 2, "cd_pass_count": 1})) == "insufficient_data"


def test_conservative_twist_conclusion_logic() -> None:
    only_30 = pd.DataFrame({"twist_deg": [30.0], "all_filters_pass_count": [3]})
    broad = pd.DataFrame({"twist_deg": [28.0, 30.0, 32.0], "all_filters_pass_count": [1, 1, 1]})
    missing = pd.DataFrame({"twist_deg": [30.0], "all_filters_pass_count": [0]})

    assert "30-degree-like" in conservative_twist_conclusion(only_30)
    assert "28-32" in conservative_twist_conclusion(broad)
    assert conservative_twist_conclusion(missing) == "insufficient_data"


def test_report_wording_includes_required_cautions() -> None:
    candidates = pd.DataFrame(
        [
                {
                    "candidate_name": "omega_clean_scale_0p9825",
                    "twist_deg": 30.0,
                    "twist_label": "30-degree-like",
                "rise_equivalent_A": 3.3405,
                "rise_window_status": "within_window",
                "combined_CD_abs_error_A": 0.0667,
                "cd_pass": True,
                "omega_geometry_pass": True,
                "torsion_envelope_pass": True,
                "hbond_pass": True,
                "hbond_plausibility_score": 90,
                "combined_filter_classification": "twist_viable_current_filters",
                "twist_status": "inferred_30_degree_like_from_current_family",
            }
        ]
    )
    by_twist = pd.DataFrame(
        [
            {
                "twist_deg": 30.0,
                "candidate_count": 1,
                "rise_window_pass_count": 1,
                "cd_pass_count": 1,
                "omega_geometry_pass_count": 1,
                "torsion_envelope_pass_count": 1,
                "hbond_pass_count": 1,
                "all_filters_pass_count": 1,
                "best_cd_error": 0.0667,
                "best_candidate_name": "omega_clean_scale_0p9825",
                "twist_status": "strongly_supported_current_filters",
            }
        ]
    )
    text = build_report(candidates, by_twist)

    for phrase in [
        "Band A",
        "3.3-3.4 A",
        "30-degree-like",
        "C/D agreement remains necessary but not sufficient",
        "heavy-atom plausibility proxy",
        "not affinity",
        "not free energy",
        "Candidate elimination should not rely on any single filter",
        "If nearby twists remain viable, say so",
    ]:
        assert phrase in text
