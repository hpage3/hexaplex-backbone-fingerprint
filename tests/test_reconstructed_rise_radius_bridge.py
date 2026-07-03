from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.run_reconstructed_rise_radius_bridge import (
    TARGETS_A,
    bridge_coordinate_path,
    bridge_recommendation,
    bridge_summary_row,
    build_candidate,
    build_report_text,
    controllable_parameters,
    derive_repeats_per_strand,
    format_rise_variant_id,
    parse_parent_ca_coordinates,
    required_score_columns,
)


def write_parent_fixture(path: Path, residues_per_chain: int = 4) -> None:
    lines = []
    serial = 1
    for chain_index, chain in enumerate(["A", "B", "C", "D", "E", "F"]):
        for resseq in range(1, residues_per_chain + 1):
            x = 8.0 + chain_index
            y = float(chain_index)
            z = float(resseq)
            lines.append(
                f"ATOM  {serial:5d} CA   GLY {chain}{resseq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
            )
            serial += 1
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def test_requested_rise_values_normalize_to_stable_variant_ids() -> None:
    assert format_rise_variant_id(3.40) == "reconstructed_rise_3p40"
    assert format_rise_variant_id(3.38) == "reconstructed_rise_3p38"
    assert format_rise_variant_id(3.35) == "reconstructed_rise_3p35"


def test_output_paths_are_under_bridge_coordinate_directory() -> None:
    outdir = Path("outputs/coordinates/reconstructed_rise_radius_bridge")

    path = bridge_coordinate_path(outdir, "reconstructed_rise_3p35")

    assert path == outdir / "reconstructed_rise_3p35.pdb"


def test_bridge_summary_row_includes_required_columns(tmp_path: Path) -> None:
    candidate = build_candidate(3.38, tmp_path, radius_A=8.0, twist_deg=30.0, repeats_per_strand=2)
    manifest = {
        "rise_A": 3.38,
        "helix_radius_A": 8.0,
        "twist_deg": 30.0,
    }
    scores = {}
    for band, target in TARGETS_A.items():
        scores[f"observed_{band}_d_A"] = target
        scores[f"{band}_error_A"] = 0.0
        scores[f"{band}_score"] = 1.0
        scores[f"{band}_found_within_tolerance"] = True

    row = bridge_summary_row(candidate, manifest, scores)

    for column in required_score_columns():
        assert column in row
    assert row["variant_id"] == "reconstructed_rise_3p38"
    assert row["combined_CD_abs_error_A"] == 0.0
    assert row["combined_ABCD_abs_error_A"] == 0.0


def test_report_text_is_written_from_scores(tmp_path: Path) -> None:
    parent = tmp_path / "parent.pdb"
    write_parent_fixture(parent)
    scores = pd.DataFrame(
        [
            {
                "variant_id": "reconstructed_rise_3p38",
                "requested_rise_A": 3.38,
                "realized_rise_A": 3.38,
                "radius_parameter_A": 8.0,
                "twist_parameter_deg": 30.0,
                "observed_A_d_A": 7.9,
                "observed_B_d_A": 6.5,
                "observed_C_d_A": 5.6,
                "observed_D_d_A": 7.3,
                "A_error_A": 0.0,
                "B_error_A": 0.0,
                "C_error_A": 0.0,
                "D_error_A": 0.0,
                "combined_CD_abs_error_A": 0.0,
                "combined_ABCD_abs_error_A": 0.0,
            }
        ]
    )

    text = build_report_text(scores, parent, radius_A=8.0, repeats_per_strand=2)

    assert "Reconstructed Rise/Radius Bridge Summary" in text
    assert "Option B" in text
    assert "reconstructed bridge models" in text
    assert "reconstructed_rise_3p38" in text


def test_workflow_handles_unavailable_source_parameters_without_crashing(tmp_path: Path) -> None:
    parent = tmp_path / "empty_parent.pdb"
    parent.write_text("END\n", encoding="ascii")

    with pytest.raises(ValueError, match="No C-alpha atoms"):
        parse_parent_ca_coordinates(parent)

    assert derive_repeats_per_strand(parent, fallback=7) == 7


def test_controllable_parameter_summary_mentions_core_controls() -> None:
    controls = controllable_parameters()

    assert "rise" in controls
    assert "radius" in controls
    assert "twist" in controls
    assert "register" in controls
    assert "chain_count" in controls


def test_bridge_recommendation_classification() -> None:
    success = pd.DataFrame({"combined_CD_abs_error_A": [0.05], "C_error_A": [0.02], "D_error_A": [0.03]})
    partial = pd.DataFrame({"combined_CD_abs_error_A": [0.4], "C_error_A": [0.3], "D_error_A": [0.1]})
    failure = pd.DataFrame({"combined_CD_abs_error_A": [1.2], "C_error_A": [0.8], "D_error_A": [0.4]})

    assert bridge_recommendation(success) == "success"
    assert bridge_recommendation(partial) == "partial"
    assert bridge_recommendation(failure) == "failure"
