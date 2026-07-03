from __future__ import annotations

from pathlib import Path

import math

import pandas as pd
import pytest

from scripts.visualize_key_structure_variants import (
    StructureSpec,
    build_report_text,
    ca_atoms,
    ca_displacement_summary,
    key_structure_specs,
    locate_key_structures,
    mean_ca_radius,
    parse_structure,
    rmsd_to_parent,
    structure_summary_row,
    z_bounds_and_span,
)


def pdb_atom_line(
    serial: int,
    atom_name: str,
    residue_name: str,
    chain_id: str,
    residue_number: int,
    x: float,
    y: float,
    z: float,
    element: str,
) -> str:
    return (
        f"ATOM  {serial:5d} {atom_name:^4s} {residue_name:>3s} {chain_id:1s}"
        f"{residue_number:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}"
        f"{1.00:6.2f}{0.00:6.2f}          {element:>2s}\n"
    )


def write_tiny_pdb(path: Path, z_offset: float = 0.0, x_offset: float = 0.0) -> None:
    lines = [
        pdb_atom_line(1, "N", "GLY", "A", 1, 0.0 + x_offset, 0.0, 0.0 + z_offset, "N"),
        pdb_atom_line(2, "CA", "GLY", "A", 1, 1.0 + x_offset, 0.0, 1.0 + z_offset, "C"),
        pdb_atom_line(3, "C", "GLY", "A", 1, 2.0 + x_offset, 0.0, 2.0 + z_offset, "C"),
        pdb_atom_line(4, "N", "ALA", "B", 2, 0.0 + x_offset, 2.0, 0.5 + z_offset, "N"),
        pdb_atom_line(5, "CA", "ALA", "B", 2, 1.0 + x_offset, 2.0, 1.5 + z_offset, "C"),
        pdb_atom_line(6, "C", "ALA", "B", 2, 2.0 + x_offset, 2.0, 2.5 + z_offset, "C"),
        "END\n",
    ]
    path.write_text("".join(lines), encoding="utf-8")


def test_locate_key_structure_paths_with_fixture_files(tmp_path: Path) -> None:
    specs = []
    for idx in range(4):
        path = tmp_path / f"structure_{idx}.pdb"
        write_tiny_pdb(path)
        specs.append(StructureSpec(f"id_{idx}", f"label {idx}", path, "fixture"))

    found = locate_key_structures(specs)

    assert [spec.structure_id for spec in found] == ["id_0", "id_1", "id_2", "id_3"]


def test_locate_key_structure_paths_reports_missing(tmp_path: Path) -> None:
    specs = [StructureSpec("missing", "missing", tmp_path / "missing.pdb", "fixture")]

    with pytest.raises(FileNotFoundError, match="Missing key structure"):
        locate_key_structures(specs)


def test_default_key_structure_specs_include_parameterized_best() -> None:
    specs = key_structure_specs(Path("repo"))

    assert any(spec.structure_id == "parameterized_rise_0p9750" for spec in specs)


def test_parse_pdb_coordinates_and_extract_ca_atoms(tmp_path: Path) -> None:
    path = tmp_path / "tiny.pdb"
    write_tiny_pdb(path)

    atoms = parse_structure(path)

    assert len(atoms) == 6
    assert [atom.atom_name for atom in ca_atoms(atoms)] == ["CA", "CA"]


def test_z_span_and_mean_ca_radius(tmp_path: Path) -> None:
    path = tmp_path / "tiny.pdb"
    write_tiny_pdb(path)
    atoms = parse_structure(path)

    z_min, z_max, z_span = z_bounds_and_span(atoms)

    assert z_min == pytest.approx(0.0)
    assert z_max == pytest.approx(2.5)
    assert z_span == pytest.approx(2.5)
    assert mean_ca_radius(atoms) == pytest.approx(1.0)


def test_displacement_summary_and_rmsd_to_parent(tmp_path: Path) -> None:
    parent_path = tmp_path / "parent.pdb"
    variant_path = tmp_path / "variant.pdb"
    write_tiny_pdb(parent_path)
    write_tiny_pdb(variant_path, z_offset=1.0)
    parent = parse_structure(parent_path)
    variant = parse_structure(variant_path)

    disp = ca_displacement_summary(parent, variant)

    assert disp["max_ca_displacement_A"] == pytest.approx(1.0)
    assert disp["mean_ca_displacement_A"] == pytest.approx(1.0)
    assert rmsd_to_parent(parent, variant) == pytest.approx(1.0)


def test_structure_summary_row_generation(tmp_path: Path) -> None:
    parent_path = tmp_path / "parent.pdb"
    variant_path = tmp_path / "variant.pdb"
    write_tiny_pdb(parent_path)
    write_tiny_pdb(variant_path, z_offset=0.5)
    parent = parse_structure(parent_path)
    variant = parse_structure(variant_path)
    spec = StructureSpec("variant", "Variant", variant_path, "fixture note")

    row = structure_summary_row(spec, variant, parent)

    assert row["structure_id"] == "variant"
    assert row["ca_count"] == 2
    assert row["atom_count"] == 6
    assert row["rmsd_to_parent_A"] == pytest.approx(0.5)
    assert row["mean_ca_displacement_A"] == pytest.approx(0.5)
    assert row["notes"] == "fixture note"
    assert "mean_interlayer_rise_A" in row


def test_report_contains_required_cautions(tmp_path: Path) -> None:
    spec = StructureSpec(
        "parameterized_rise_0p9750",
        "Parameterized rise 0.9750",
        tmp_path / "parameterized_rise_0p9750.pdb",
        "fixture",
    )
    summary = pd.DataFrame(
        [
            {
                "structure_id": "parameterized_rise_0p9750",
                "z_span_A": 1.0,
                "mean_ca_radius_A": 2.0,
                "rmsd_to_parent_A": 0.1,
                "max_ca_displacement_A": 0.2,
                "mean_interlayer_rise_A": 0.3,
            }
        ]
    )

    text = build_report_text(summary, [spec])

    assert "Key Structure Variant Visualization" in text
    assert "parameterized_rise_0p9750" in text
    assert "diagnostic transformed structures" in text
    assert "not minimized physical structures" in text


