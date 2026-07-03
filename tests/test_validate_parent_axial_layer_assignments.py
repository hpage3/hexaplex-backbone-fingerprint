from __future__ import annotations

import pandas as pd

from scripts.validate_parent_axial_layer_assignments import (
    AtomRecord,
    balance_score,
    build_chain_coverage,
    build_layer_composition,
    build_report_text,
    build_residue_participation,
    detect_outlier_layers,
    parse_atom_record,
)


def atom(chain: str, resnum: int, name: str, z: float, serial: int = 1, resname: str = "GLU") -> AtomRecord:
    return AtomRecord(serial, name, resname, chain, resnum, name[0], 0.0, 0.0, z)


def assignment_fixture() -> pd.DataFrame:
    rows = [
        {"layer_index": 0, "atom_serial": 1, "atom_name": "N", "residue_name": "GLU", "chain_id": "A", "residue_number": 1, "element": "N", "x": 0, "y": 0, "z": 0.0},
        {"layer_index": 0, "atom_serial": 2, "atom_name": "CA", "residue_name": "GLU", "chain_id": "A", "residue_number": 1, "element": "C", "x": 0, "y": 0, "z": 0.1},
        {"layer_index": 1, "atom_serial": 3, "atom_name": "C", "residue_name": "GLU", "chain_id": "A", "residue_number": 1, "element": "C", "x": 0, "y": 0, "z": 1.0},
        {"layer_index": 1, "atom_serial": 4, "atom_name": "CA", "residue_name": "CYP", "chain_id": "B", "residue_number": 2, "element": "C", "x": 0, "y": 0, "z": 1.1},
    ]
    return pd.DataFrame(rows)


def test_parse_atom_record_from_minimal_pdb_line() -> None:
    line = "ATOM      7  CA  GLU A  12       1.000   2.000   3.000  1.00 20.00           C"
    rec = parse_atom_record(line)
    assert rec.serial == 7
    assert rec.atom_name == "CA"
    assert rec.residue_name == "GLU"
    assert rec.chain_id == "A"
    assert rec.residue_number == 12
    assert rec.element == "C"
    assert rec.z == 3.0


def test_balance_score_calculation() -> None:
    assert balance_score({"A", "B", "C"}, ["A", "B", "C", "D", "E", "F"]) == 0.5
    assert balance_score({"A", "B"}, ["A", "B"]) == 1.0


def test_layer_composition_aggregation() -> None:
    comp = build_layer_composition(assignment_fixture(), ["A", "B"])
    assert len(comp) == 2
    assert comp.loc[0, "atom_count"] == 2
    assert comp.loc[1, "ca_count"] == 1
    assert comp.loc[0, "balance_score"] == 0.5


def test_residue_participation_detects_split_residue() -> None:
    residues = build_residue_participation(assignment_fixture())
    split = residues[residues["chain_id"] == "A"].iloc[0]
    assert split["layer_count"] == 2
    assert split["notes"] == "split_across_layers"


def test_chain_coverage_aggregation() -> None:
    coverage = build_chain_coverage(assignment_fixture())
    a = coverage[coverage["chain_id"] == "A"].iloc[0]
    assert a["first_layer"] == 0
    assert a["last_layer"] == 1
    assert a["missing_layer_count_within_span"] == 0


def test_outlier_layer_detection_by_thickness() -> None:
    comp = pd.DataFrame({"layer_index": [0, 1, 2, 3], "z_thickness_A": [0.1, 0.1, 0.2, 2.0]})
    assert detect_outlier_layers(comp) == {3}


def test_report_text_contains_required_cautions() -> None:
    assignments = assignment_fixture()
    comp = build_layer_composition(assignments, ["A", "B"])
    residues = build_residue_participation(assignments)
    coverage = build_chain_coverage(assignments)
    atoms = [atom("A", 1, "CA", 0.0), atom("B", 2, "CA", 1.0)]
    text = build_report_text(
        source_pdb="parent.pdb",  # type: ignore[arg-type]
        atoms=atoms,
        composition=comp,
        residue_participation=residues,
        chain_coverage=coverage,
        interpretation="layers are useful computational slices but should not be interpreted as unique chemical hexad levels",
        plots=[],
    )
    assert "Parent Axial Layer Assignment Validation" in text
    assert "parameterized-rise" in text
    assert "not minimized" in text
    assert "should not be interpreted as unique chemical hexad levels" in text
