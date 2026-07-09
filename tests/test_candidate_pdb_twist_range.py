from __future__ import annotations

import pandas as pd

from scripts.audit_candidate_pdb_twist_range import (
    build_report,
    classify_candidate_set,
    extract_rise_A,
    extract_twist_deg,
    has_18_to_32_grid,
    summarize_inventory,
)


def test_twist_extraction_filename_variants() -> None:
    cases = {
        "model_twist18.pdb": 18.0,
        "model_twist_18.pdb": 18.0,
        "candidate_18deg.pdb": 18.0,
        "h_twist_30_candidate.pdb": 30.0,
        "candidate_30_degree.pdb": 30.0,
        r"inputs\asem_original_3p4\GitHub_research_sidechains_tleap_structures\32\cand0\initial.pdb": 32.0,
    }
    for name, expected in cases.items():
        value, source = extract_twist_deg(name)
        assert value == expected
        assert source != "unknown"


def test_rise_extraction_filename_variants() -> None:
    cases = {
        "model_3p38_candidate.pdb": 3.38,
        "model_3.38_candidate.pdb": 3.38,
        "model_rise_3p380_candidate.pdb": 3.38,
        "h_rise_3.4_candidate.pdb": 3.4,
    }
    for name, expected in cases.items():
        value, source = extract_rise_A(name)
        assert value == expected
        assert source != "unknown"


def test_candidate_set_classification() -> None:
    assert classify_candidate_set(r"inputs\asem_original_3p4\GitHub_research_sidechains_tleap_structures\18\cand0\initial.pdb") == "pnab_3p38_twist_grid"
    assert classify_candidate_set(r"inputs\selected_345\candidate.pdb") == "selected_from_345"
    assert classify_candidate_set(r"outputs\parametric_six_strand_peptide_plane_models\models\foo.pdb") == "powder_scored_candidate"
    assert classify_candidate_set(r"outputs\coordinates\omega_clean_rise_compression_scan\foo.pdb") == "omega_clean_publication_track"
    assert classify_candidate_set(r"scratch\other.pdb") == "unknown"


def test_summary_min_max_unique_and_grid_detection() -> None:
    inventory = pd.DataFrame(
        [
            {
                "candidate_set": "pnab_3p38_twist_grid",
                "inferred_twist_deg": float(twist),
                "inferred_rise_A": 3.38,
            }
            for twist in range(18, 33)
        ]
    )
    crosswalk = pd.DataFrame(
        [
            {
                "candidate_set": "omega_clean_publication_track",
                "inferred_twist_deg": 30.0,
                "inferred_rise_A": 3.34,
                "scoring_available": True,
                "combined_CD_abs_error_A": 0.0667,
                "candidate_name": "omega_clean_scale_0p9825",
            }
        ]
    )

    summary = summarize_inventory(inventory, crosswalk)
    grid = summary[summary["candidate_set"] == "pnab_3p38_twist_grid"].iloc[0]

    assert grid["pdb_count"] == 15
    assert grid["twist_min_deg"] == 18.0
    assert grid["twist_max_deg"] == 32.0
    assert "18" in grid["twist_unique_values"]
    assert "32" in grid["twist_unique_values"]
    assert has_18_to_32_grid(summary)


def test_report_wording_includes_required_cautions() -> None:
    inventory = pd.DataFrame(
        [
            {
                "candidate_set": "pnab_3p38_twist_grid",
                "inferred_twist_deg": float(twist),
                "inferred_rise_A": 3.38,
            }
            for twist in range(18, 33)
        ]
    )
    crosswalk = pd.DataFrame(
        [
            {
                "candidate_set": "omega_clean_publication_track",
                "inferred_twist_deg": 30.0,
                "inferred_rise_A": 3.38,
                "scoring_available": True,
                "combined_CD_abs_error_A": 0.0667,
                "candidate_name": "omega_clean_scale_0p9825",
            }
        ]
    )
    summary = summarize_inventory(inventory, crosswalk)
    report = build_report(inventory, crosswalk, summary)

    for phrase in [
        "18",
        "32",
        "3.38",
        "345",
        "previous twist-tightening report only covered",
        "inventory/provenance audit",
        "not a new structural conclusion",
        "pNAB-derived twist-grid candidates must be included",
    ]:
        assert phrase in report
