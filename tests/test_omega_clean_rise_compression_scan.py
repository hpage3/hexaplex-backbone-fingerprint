from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.generate_global_deformation_variants import parse_pdb_atom_lines
from scripts.run_internal_coordinate_endpoint_closure import detect_every_other_pattern
from scripts.run_omega_clean_rise_compression_scan import (
    atom_count_preserved,
    build_report_text,
    carboxylate_present,
    classify_omega_value,
    coordinate_rmsd,
    identities_preserved,
    omega_clean_output_path,
    omega_clean_variant_id,
    reference_reproduces_guarded,
)
from scripts.run_parent_derived_rise_fine_scan import best_score_row, best_score_rows, plateau_text


def atom_line(serial: int, name: str, resname: str, chain: str, resseq: int, x: float, y: float, z: float, element: str) -> str:
    return (
        f"ATOM  {serial:5d} {name:<4} {resname:>3} {chain}{resseq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2}"
    )


def write_fixture(path: Path, include_carboxylate: bool = True) -> None:
    lines = [
        atom_line(1, "N", "CYP", "A", 1, 0.0, 0.0, 0.0, "N"),
        atom_line(2, "CA", "CYP", "A", 1, 1.0, 0.0, 0.0, "C"),
        atom_line(3, "C", "CYP", "A", 1, 2.0, 0.0, 0.0, "C"),
        atom_line(4, "O", "CYP", "A", 1, 3.0, 0.0, 0.0, "O"),
        atom_line(5, "N", "GLU", "A", 2, 2.2, 1.0, 0.0, "N"),
        atom_line(6, "CA", "GLU", "A", 2, 3.2, 1.0, 0.0, "C"),
        atom_line(7, "C", "GLU", "A", 2, 4.2, 1.0, 0.0, "C"),
        atom_line(8, "O", "GLU", "A", 2, 5.2, 1.0, 0.0, "O"),
    ]
    if include_carboxylate:
        lines.append(atom_line(9, "OE1", "GLU", "A", 2, 4.2, 2.0, 0.0, "O"))
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def test_scale_id_formatting_and_output_path(tmp_path: Path) -> None:
    assert omega_clean_variant_id(1.0) == "omega_clean_scale_1p0000"
    assert omega_clean_variant_id(0.9825) == "omega_clean_scale_0p9825"
    assert omega_clean_output_path(tmp_path, 0.9825) == tmp_path / "omega_clean_scale_0p9825.pdb"


def test_no_change_preserves_coordinates_atom_count_and_identity(tmp_path: Path) -> None:
    pdb = tmp_path / "fixture.pdb"
    write_fixture(pdb)
    _lines, atoms = parse_pdb_atom_lines(pdb)

    assert atom_count_preserved(atoms, atoms)
    assert identities_preserved(atoms, atoms)
    assert coordinate_rmsd(atoms, atoms) <= 1e-12


def test_carboxylate_preservation_check(tmp_path: Path) -> None:
    with_carboxylate = tmp_path / "with.pdb"
    without_carboxylate = tmp_path / "without.pdb"
    write_fixture(with_carboxylate, include_carboxylate=True)
    write_fixture(without_carboxylate, include_carboxylate=False)

    _lines, atoms = parse_pdb_atom_lines(with_carboxylate)
    _lines2, atoms2 = parse_pdb_atom_lines(without_carboxylate)

    assert carboxylate_present(atoms)
    assert not carboxylate_present(atoms2)


def test_omega_wraparound_and_window_classification() -> None:
    assert classify_omega_value(180.0) == "within_8deg"
    assert classify_omega_value(-179.5) == "within_8deg"
    assert classify_omega_value(171.5) == "within_10deg"
    assert classify_omega_value(-167.0) == "outside_10deg"


def test_every_other_detection_on_synthetic_ordered_chain_values() -> None:
    pattern = detect_every_other_pattern([2.0, 15.0, 1.0, 16.0, 2.0, 15.0])
    assert pattern["every_other_detected"] is True
    flat = detect_every_other_pattern([2.0, 2.5, 1.5, 2.0, 2.5, 1.0])
    assert flat["every_other_detected"] is False


def test_best_score_plateau_without_unique_optimum() -> None:
    scores = pd.DataFrame(
        [
            {"variant_id": "omega_clean_scale_1p0000", "combined_CD_abs_error_A": 0.1698},
            {"variant_id": "omega_clean_scale_0p9825", "combined_CD_abs_error_A": 0.0667},
            {"variant_id": "omega_clean_scale_0p9800", "combined_CD_abs_error_A": 0.0667},
            {"variant_id": "omega_clean_scale_0p9775", "combined_CD_abs_error_A": 0.0667},
        ]
    )

    rows = best_score_rows(scores)
    assert best_score_row(scores)["variant_id"] == "omega_clean_scale_0p9825"
    assert rows["variant_id"].tolist() == [
        "omega_clean_scale_0p9825",
        "omega_clean_scale_0p9800",
        "omega_clean_scale_0p9775",
    ]
    assert plateau_text(rows) == "omega_clean_scale_0p9825 through omega_clean_scale_0p9775"


def test_reference_reproduction_status() -> None:
    assert reference_reproduces_guarded({"observed_C_d_A": 5.7454, "observed_D_d_A": 7.2756})
    assert not reference_reproduces_guarded({"observed_C_d_A": 5.50, "observed_D_d_A": 7.2756})


def test_report_wording_includes_required_scope_language(tmp_path: Path) -> None:
    scores = pd.DataFrame(
        [
            {
                "variant_id": "omega_clean_scale_1p0000",
                "axial_scale": 1.0,
                "observed_C_d_A": 5.7454,
                "observed_D_d_A": 7.2756,
                "combined_CD_abs_error_A": 0.1698,
                "reference_reproduces_guarded": True,
                "C_moves_toward_fine_scan_target": False,
                "D_near_guarded_baseline": True,
                "status": "scored",
            },
            {
                "variant_id": "omega_clean_scale_0p9825",
                "axial_scale": 0.9825,
                "observed_C_d_A": 5.6422,
                "observed_D_d_A": 7.2756,
                "combined_CD_abs_error_A": 0.0667,
                "reference_reproduces_guarded": True,
                "C_moves_toward_fine_scan_target": True,
                "D_near_guarded_baseline": True,
                "status": "scored",
            },
        ]
    )
    geometry = pd.DataFrame(
        [
            {
                "variant_id": row["variant_id"],
                "atom_count": 1131,
                "atom_count_preserved_vs_guarded": True,
                "carboxylates_preserved_vs_guarded": True,
                "overall_omega_count": 174,
                "overall_omega_within_8_count": 174,
                "overall_omega_within_10_count": 174,
                "overall_omega_outside_10_count": 0,
                "overall_omega_every_other_detected": False,
                "all_atom_rmsd_to_uncompressed_guarded_A": 0.0,
                "triketo_ca_rise_median_A": 3.0,
                "triamino_ca_rise_median_A": 3.0,
            }
            for _, row in scores.iterrows()
        ]
    )

    text = build_report_text(scores, geometry, tmp_path / "guarded.pdb", tmp_path / "parent.pdb")

    for phrase in [
        "omega-clean rise-compression scan",
        "not a final structure",
        "not energy minimized",
        "pNAB",
        "every-other",
        "parent-derived rise-compression",
        "+/-8",
        "+/-10",
    ]:
        assert phrase in text
    assert "does not imply a unique optimum" in text or "avoids claiming a unique optimum" in text
