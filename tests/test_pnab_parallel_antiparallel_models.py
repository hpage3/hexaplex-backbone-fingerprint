from __future__ import annotations

import pandas as pd

from scripts.audit_pnab_parallel_antiparallel_models import (
    build_report,
    classify_anti_parallel_30,
    classify_parallel_elimination,
    infer_orientation,
    infer_rise_A,
    infer_twist_deg,
    is_visual_box_file,
    should_score_coordinate,
)
from scripts.run_internal_coordinate_endpoint_closure import detect_every_other_pattern, omega_window_class, trans_deviation_deg


def test_orientation_inference() -> None:
    assert infer_orientation("model_parallel_twist30.pdb") == "parallel"
    assert infer_orientation("model_anti-parallel_30deg.pdb") == "anti_parallel"
    assert infer_orientation("model_antiparallel_twist30.pdb") == "anti_parallel"
    assert infer_orientation("model_anti_parallel_twist30.pdb") == "anti_parallel"
    assert infer_orientation("model_unknown.pdb") == "unknown"


def test_twist_extraction() -> None:
    assert infer_twist_deg("hexaplex_30deg.pdb") == 30
    assert infer_twist_deg("pnab_twist30_rise3p4.pdb") == 30
    assert infer_twist_deg("anti_30_degree_model.pdb") == 30
    assert infer_twist_deg("6strand_tw32_rise3p4.pdb") == 32


def test_rise_extraction() -> None:
    assert infer_rise_A("pnab_twist30_rise_3p40.pdb") == 3.40
    assert infer_rise_A("pnab_twist30_rise-3.4.pdb") == 3.4
    assert infer_rise_A("model_3p4_candidate.pdb") == 3.4


def test_parallel_elimination_classification() -> None:
    inventory = pd.DataFrame(
        [
            {"path": "parallel.pdb", "inferred_orientation": "parallel"},
            {"path": "anti.pdb", "inferred_orientation": "anti_parallel"},
        ]
    )
    summary = pd.DataFrame()
    scores_good_parallel = pd.DataFrame(
        [{"inferred_orientation": "parallel", "approaches_omega_clean_plateau": True}]
    )
    scores_bad_parallel = pd.DataFrame(
        [{"inferred_orientation": "parallel", "approaches_omega_clean_plateau": False}]
    )
    no_parallel_inventory = pd.DataFrame([{"path": "anti.pdb", "inferred_orientation": "anti_parallel"}])

    assert classify_parallel_elimination(inventory, summary, scores_good_parallel) == "not_eliminated"
    assert classify_parallel_elimination(inventory, summary, scores_bad_parallel) == "disfavored_not_eliminated"
    assert classify_parallel_elimination(no_parallel_inventory, summary, pd.DataFrame()) == "insufficient_data"


def test_anti_parallel_30_status_classification() -> None:
    inventory = pd.DataFrame(
        [
            {"path": "anti30.pdb", "inferred_orientation": "anti_parallel", "inferred_twist_deg": 30.0},
        ]
    )
    scores = pd.DataFrame(
        [
            {"inferred_orientation": "anti_parallel", "inferred_twist_deg": 30.0, "approaches_omega_clean_plateau": True},
        ]
    )
    assert classify_anti_parallel_30(inventory, scores) == "strongest_current_pnab_candidate"
    assert classify_anti_parallel_30(inventory, pd.DataFrame()) == "plausible_candidate"
    assert classify_anti_parallel_30(pd.DataFrame(), pd.DataFrame()) == "insufficient_data"


def test_omega_window_and_every_other_synthetic() -> None:
    deviations = [trans_deviation_deg(value) for value in [-180, -168, -180, -168, -180, -168]]
    assert omega_window_class(-180) == "within_8deg"
    assert omega_window_class(-171) == "within_10deg"
    assert detect_every_other_pattern(deviations)["every_other_detected"] is True


def test_report_wording() -> None:
    inventory = pd.DataFrame(
        [
            {
                "path": "anti_parallel_30deg.pdb",
                "file_type": "coordinate",
                "inferred_orientation": "anti_parallel",
                "inferred_twist_deg": 30.0,
                "inferred_rise_A": 3.4,
                "coordinate_file_exists": True,
                "metric_or_scoring_file_exists": False,
            }
        ]
    )
    summary = pd.DataFrame(
        [
            {
                "inferred_orientation": "anti_parallel",
                "inferred_twist_deg": 30.0,
                "file_count": 1,
                "coordinate_count": 1,
            }
        ]
    )
    geometry = pd.DataFrame()
    scores = pd.DataFrame()

    text = build_report(inventory, summary, geometry, scores)

    for phrase in [
        "pNAB",
        "parallel",
        "anti-parallel",
        "compatibility",
        "not final structural proof",
        "disfavored_not_eliminated",
        "insufficient_data",
    ]:
        assert phrase in text


def test_visual_box_pdb_is_not_scored_as_atomistic_candidate() -> None:
    from pathlib import Path

    box_path = Path("outputs/six_strand_first_panel_visual_boxes/foo/foo_boxes.pdb")
    actual_path = Path("outputs/foo/pnab_hexaplex_twist30_rise3p38.pdb")

    assert is_visual_box_file(box_path)
    assert not should_score_coordinate(box_path)
    assert should_score_coordinate(actual_path)
