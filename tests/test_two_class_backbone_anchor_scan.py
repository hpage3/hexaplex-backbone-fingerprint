from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.generate_global_deformation_variants import parse_pdb_atom_lines
from scripts.run_two_class_backbone_anchor_scan import (
    BACKBONE_ANCHOR_ATOMS,
    AnchorScanSpec,
    best_score_rows,
    build_report_text,
    class_for_chain,
    identity_preserved,
    max_coordinate_delta,
    plateau_text,
    should_move_atom,
    variant_id_for_adjustments,
    write_anchor_scan_variant,
)


def atom_line(serial: int, atom: str, resname: str, chain: str, resseq: int, x: float, y: float, z: float, element: str) -> str:
    return (
        f"ATOM  {serial:5d} {atom:<4} {resname:>3} {chain}{resseq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2}"
    )


def write_fixture(path: Path) -> None:
    lines = []
    serial = 1
    chain_resnames = {
        "A": "CYP",
        "B": "MEP",
        "C": "CYP",
        "D": "MEP",
        "E": "CYP",
        "F": "MEP",
    }
    for chain_index, (chain, resname) in enumerate(chain_resnames.items()):
        angle = np.radians(chain_index * 60.0)
        cx = 8.0 * np.cos(angle)
        cy = 8.0 * np.sin(angle)
        for resseq in [1, 2]:
            z = float(resseq)
            coords = [
                ("N", cx + 0.0, cy + 0.2, z, "N"),
                ("CA", cx + 0.5, cy + 0.0, z, "C"),
                ("C", cx + 1.0, cy - 0.2, z, "C"),
                ("O", cx + 1.2, cy - 0.4, z, "O"),
                ("OE1", cx + 1.4, cy + 0.4, z, "O"),
            ]
            for atom, x, y, atom_z, element in coords:
                lines.append(atom_line(serial, atom, resname, chain, resseq, x, y, atom_z, element))
                serial += 1
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def test_class_assignment_and_variant_ids() -> None:
    assert class_for_chain("A") == "triketo_cyanuric_like"
    assert class_for_chain("C") == "triketo_cyanuric_like"
    assert class_for_chain("E") == "triketo_cyanuric_like"
    assert class_for_chain("B") == "triamino_melamine_like"
    assert class_for_chain("D") == "triamino_melamine_like"
    assert class_for_chain("F") == "triamino_melamine_like"
    assert variant_id_for_adjustments(0.02, -0.04) == "two_class_anchor_trip02_cym04"


def test_no_change_reference_preserves_coordinates_and_identity(tmp_path: Path) -> None:
    parent = tmp_path / "parent.pdb"
    out = tmp_path / "reference.pdb"
    write_fixture(parent)
    lines, atoms = parse_pdb_atom_lines(parent)

    write_anchor_scan_variant(lines, atoms, AnchorScanSpec("ref", 0.0, 0.0), out)
    _out_lines, variant_atoms = parse_pdb_atom_lines(out)

    assert identity_preserved(atoms, variant_atoms)
    assert max_coordinate_delta(atoms, variant_atoms) == 0.0


def test_class_specific_transform_moves_intended_class_and_atom_set(tmp_path: Path) -> None:
    parent = tmp_path / "parent.pdb"
    out = tmp_path / "tri_only.pdb"
    write_fixture(parent)
    lines, atoms = parse_pdb_atom_lines(parent)

    write_anchor_scan_variant(lines, atoms, AnchorScanSpec("tri_only", 0.04, 0.0), out)
    _out_lines, variant_atoms = parse_pdb_atom_lines(out)

    moved = []
    fixed = []
    for before, after in zip(atoms, variant_atoms):
        delta = float(np.linalg.norm(before.coord - after.coord))
        if delta > 1e-6:
            moved.append(before)
        else:
            fixed.append(before)

    assert moved
    assert {atom.chain for atom in moved} <= {"B", "D", "F"}
    assert {atom.atom_name for atom in moved} <= BACKBONE_ANCHOR_ATOMS
    assert "CA" in {atom.atom_name for atom in moved}
    assert all(atom.atom_name != "OE1" for atom in moved)
    assert all(atom.atom_name == "OE1" or not should_move_atom(atom) or atom.chain in {"A", "C", "E"} for atom in fixed)


def test_recognition_core_register_and_carboxylates_are_preserved(tmp_path: Path) -> None:
    parent = tmp_path / "parent.pdb"
    out = tmp_path / "both.pdb"
    write_fixture(parent)
    lines, atoms = parse_pdb_atom_lines(parent)

    write_anchor_scan_variant(lines, atoms, AnchorScanSpec("both", 0.04, -0.04), out)
    _out_lines, variant_atoms = parse_pdb_atom_lines(out)

    assert identity_preserved(atoms, variant_atoms)
    parent_carboxylates = [atom for atom in atoms if atom.atom_name == "OE1"]
    variant_carboxylates = [atom for atom in variant_atoms if atom.atom_name == "OE1"]
    assert len(parent_carboxylates) == len(variant_carboxylates)
    assert all(np.allclose(before.coord, after.coord) for before, after in zip(parent_carboxylates, variant_carboxylates))


def test_best_score_logic_reports_plateau_without_unique_claim() -> None:
    scores = pd.DataFrame(
        {
            "variant_id": ["a", "b", "c"],
            "combined_CD_abs_error_A": [0.2, 0.1, 0.1],
        }
    )
    best = best_score_rows(scores)
    assert best["variant_id"].tolist() == ["b", "c"]
    assert plateau_text(best) == "b through c"


def test_report_wording_contains_required_cautions() -> None:
    scores = pd.DataFrame(
        {
            "variant_id": ["two_class_anchor_tri0_cy0", "two_class_anchor_trip02_cym02"],
            "triamino_radial_adjust_A": [0.0, 0.02],
            "triketo_radial_adjust_A": [0.0, -0.02],
            "C_peak_A": [5.745, 5.7],
            "D_peak_A": [7.276, 7.276],
            "C_error_A": [0.145, 0.1],
            "D_error_A": [-0.024, -0.024],
            "C_score": [1.0, 1.1],
            "D_score": [1.0, 1.0],
            "combined_CD_abs_error_A": [0.169, 0.124],
        }
    )
    geometry = pd.DataFrame(
        {
            "variant_id": ["two_class_anchor_trip02_cym02"],
            "row_type": ["summary"],
            "group": ["all_six_chains"],
            "omega_median_deg": [-167.0],
            "omega_trans_deviation_median_deg": [0.1],
            "theta_median_deg": [74.0],
            "ca_rise_median_A": [2.2],
            "exit_vector_angle_gap_rms_deg": [42.0],
            "radial_angle_gap_rms_deg": [28.0],
        }
    )

    text = build_report_text(scores, geometry, Path("parent.pdb"))

    assert "controlled backbone-anchor" in text
    assert "not a final atomistic reconstruction" in text
    assert "two separate three-fold" in text
    assert "theta/omega" in text
    assert "prior N/C/O-only peptide-plane prototype was too conservative" in text
    assert "does not imply a unique optimum" in text
