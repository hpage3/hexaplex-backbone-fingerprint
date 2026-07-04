from __future__ import annotations

import pandas as pd

from scripts.report_item6_candidate_filter_funnel import (
    build_report_text,
    classify_candidate_status,
    classify_cd_filter,
    classify_physical_filter,
    classify_pnab_filter,
)


def test_candidate_classification_logic() -> None:
    pnab = "partial_pnab_scaffold_filter_parallel_not_eliminated"
    physical = "passes_physical_chemical_sense"

    assert classify_candidate_status(pnab, "cd_plateau_preserved", physical) == "survives_current_filters_with_pnab_caveat"
    assert classify_candidate_status(pnab, "parent_like", physical) == "survives_current_filters_with_pnab_caveat"
    assert classify_candidate_status(pnab, "cd_plateau_preserved", "fails_physical_chemical_sense") == "cd_only_not_physical"
    assert classify_candidate_status(pnab, "cd_mismatch", physical) == "physical_not_cd"
    assert classify_candidate_status("pnab_insufficient", "cd_plateau_preserved", physical) == "pnab_insufficient"
    assert classify_candidate_status(pnab, "cd_mismatch", "fails_physical_chemical_sense") == "rejected"


def test_pnab_caveat_does_not_eliminate_parallel() -> None:
    status = classify_pnab_filter("insufficient_data", "plausible_candidate")
    assert status == "partial_pnab_scaffold_filter_parallel_not_eliminated"


def test_cd_filter_classification() -> None:
    assert classify_cd_filter(5.7454, 7.2756, 0.1698) == "parent_like"
    assert classify_cd_filter(5.6422, 7.2756, 0.0667) == "cd_plateau_preserved"
    assert classify_cd_filter(5.6422, 7.1923, 0.1499) == "over_compressed_d_degraded"
    assert classify_cd_filter(5.0, 8.0, 1.0) == "over_compressed_d_degraded"


def test_physical_sense_filter_requires_core_guards() -> None:
    assert (
        classify_physical_filter(
            atom_count_preserved=True,
            carboxylates_preserved=True,
            omega_within_8_count=174,
            omega_count=174,
            every_other_absent=True,
            unresolved_segments=0,
            guard_passed=True,
        )
        == "passes_physical_chemical_sense"
    )
    assert (
        classify_physical_filter(
            atom_count_preserved=True,
            carboxylates_preserved=True,
            omega_within_8_count=160,
            omega_count=174,
            every_other_absent=True,
            unresolved_segments=0,
            guard_passed=True,
        )
        == "physical_with_caveats"
    )
    assert (
        classify_physical_filter(
            atom_count_preserved=False,
            carboxylates_preserved=True,
            omega_within_8_count=174,
            omega_count=174,
            every_other_absent=True,
            unresolved_segments=0,
            guard_passed=True,
        )
        == "fails_physical_chemical_sense"
    )


def test_report_wording_contains_required_cautions() -> None:
    funnel = pd.DataFrame(
        [
            {
                "candidate_family": "omega_clean_guarded_full_chain_baseline",
                "variant_id": "guarded_full_chain_prototype",
                "filter_2_cd_band": "parent_like",
                "observed_C_d_A": 5.7454,
                "observed_D_d_A": 7.2756,
                "combined_CD_abs_error_A": 0.1698,
                "filter_3_physical_chemical_sense": "passes_physical_chemical_sense",
                "candidate_status": "survives_current_filters_with_pnab_caveat",
            },
            {
                "candidate_family": "omega_clean_rise_compressed",
                "variant_id": "omega_clean_scale_0p9825",
                "filter_2_cd_band": "cd_plateau_preserved",
                "observed_C_d_A": 5.6422,
                "observed_D_d_A": 7.2756,
                "combined_CD_abs_error_A": 0.0667,
                "filter_3_physical_chemical_sense": "passes_physical_chemical_sense",
                "candidate_status": "survives_current_filters_with_pnab_caveat",
            },
        ]
    )
    survivor = pd.DataFrame(
        [
            {
                "surviving_model_family": "omega-clean rise-compressed plateau",
                "status": "strongest_current_surviving_family_with_pnab_caveat",
                "variant_range": "omega_clean_scale_0p9825 through omega_clean_scale_0p9725",
                "variant_count": 5,
                "C_peak_A": 5.6422,
                "D_peak_A": 7.2756,
                "combined_CD_abs_error_A": 0.0667,
                "pnab_caveat": "parallel models not eliminated from current labeled pNAB data",
                "torsion_boundary_summary": "Measured compatible range.",
            }
        ]
    )
    pnab = {
        "parallel_elimination_status": "insufficient_data",
        "anti_parallel_30_status": "plausible_candidate",
        "pnab_filter": "partial_pnab_scaffold_filter_parallel_not_eliminated",
    }
    text = build_report_text(funnel, survivor, pnab, [], pd.DataFrame())

    for phrase in [
        "C/D agreement is necessary but not sufficient",
        "pNAB",
        "compatibility/scaffold filter",
        "not final structural proof",
        "insufficient to eliminate parallel",
        "plausible candidate",
        "omega-clean rise-compressed",
        "PI-level interpretation",
    ]:
        assert phrase in text
