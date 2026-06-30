from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "integrate_peptide_plane_with_abcd_scan.py"
    spec = importlib.util.spec_from_file_location("integrate_peptide_plane_with_abcd_scan", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_rms_alternation_fraction_counts_high_low_transitions():
    module = load_script_module()
    df = pd.DataFrame(
        {
            "chain": ["A", "A", "A", "A"],
            "res_i": [1, 2, 3, 4],
            "res_j": [2, 3, 4, 5],
            "plane_index": [0, 1, 2, 3],
            "rms_state": ["low_rms", "high_rms", "low_rms", "mid_rms"],
        }
    )

    assert module.rms_alternation_fraction(df) == 2 / 3


def test_entropy_fraction_is_normalized():
    module = load_script_module()

    assert module.entropy_fraction(pd.Series(["A-B", "A-B", "A-B"])) == 0.0
    assert round(module.entropy_fraction(pd.Series(["A-B", "C-D"])), 6) == 1.0


def test_cd_cell_group_score_direction_higher_is_better():
    module = load_script_module()
    cells = pd.DataFrame({"mean_CD_score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]})

    grouped = module.classify_cd_cell_groups(cells)

    assert grouped.loc[0, "cd_cell_group"] == "bottom"
    assert grouped.loc[5, "cd_cell_group"] == "top"
