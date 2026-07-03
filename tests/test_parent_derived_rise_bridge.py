from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_parent_axial_layers import infer_layers_from_ca_z
from scripts.generate_global_deformation_variants import parse_pdb_atom_lines
from scripts.run_parent_derived_rise_bridge import (
    ParentDerivedRiseSpec,
    bridge_recommendation,
    bridge_specs,
    build_report_text,
    identity_preserved,
    nominal_rise_to_scale,
    output_path,
    parameterized_rise_z,
    run_bridge,
    transformed_coord,
    variant_id_for_nominal_rise,
    write_parent_derived_variant,
)


def atom_line(serial: int, name: str, resname: str, chain: str, resseq: int, x: float, y: float, z: float, element: str) -> str:
    return (
        f"ATOM  {serial:5d} {name:<4} {resname:>3} {chain}{resseq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2}"
    )


def write_parent_fixture(path: Path) -> None:
    lines = []
    serial = 1
    for chain_index, chain in enumerate(["A", "B"]):
        for resseq, resname in enumerate(["GLU", "CYP", "GLU"], start=1):
            z = float(resseq * 2 + chain_index * 0.1)
            x = float(chain_index * 5)
            for name, element, dx in [("N", "N", 0.0), ("CA", "C", 1.0), ("C", "C", 2.0), ("O", "O", 3.0), ("OE1", "O", 4.0)]:
                lines.append(atom_line(serial, name, resname, chain, resseq, x + dx, float(chain_index), z, element))
                serial += 1
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def test_variant_ids_and_scales_are_stable() -> None:
    assert variant_id_for_nominal_rise(3.40) == "parent_derived_rise_3p40_equiv"
    assert variant_id_for_nominal_rise(3.38) == "parent_derived_rise_3p38_equiv"
    assert nominal_rise_to_scale(3.35) == 3.35 / 3.40
    assert [spec.variant_id for spec in bridge_specs()] == [
        "parent_derived_reference",
        "parent_derived_rise_3p40_equiv",
        "parent_derived_rise_3p38_equiv",
        "parent_derived_rise_3p35_equiv",
    ]


def test_output_path_uses_parent_derived_directory(tmp_path: Path) -> None:
    spec = ParentDerivedRiseSpec("parent_derived_reference", None, 1.0)

    assert output_path(tmp_path, spec) == tmp_path / "parent_derived_reference.pdb"


def test_axial_scale_one_leaves_coordinates_unchanged(tmp_path: Path) -> None:
    parent = tmp_path / "parent.pdb"
    out = tmp_path / "reference.pdb"
    write_parent_fixture(parent)
    lines, atoms = parse_pdb_atom_lines(parent)
    layer_model = infer_layers_from_ca_z([atom.z for atom in atoms if atom.is_ca])
    center_z = float(np.mean(layer_model.layer_centers))
    spec = ParentDerivedRiseSpec("reference", None, 1.0)

    write_parent_derived_variant(lines, atoms, spec, layer_model, center_z, out)
    _, variant_atoms = parse_pdb_atom_lines(out)

    assert identity_preserved(atoms, variant_atoms)
    for parent_atom, variant_atom in zip(atoms, variant_atoms):
        assert np.allclose(parent_atom.coord, variant_atom.coord)


def test_axial_scale_less_than_one_compresses_z_around_center(tmp_path: Path) -> None:
    parent = tmp_path / "parent.pdb"
    write_parent_fixture(parent)
    _lines, atoms = parse_pdb_atom_lines(parent)
    layer_model = infer_layers_from_ca_z([atom.z for atom in atoms if atom.is_ca])
    center_z = float(np.mean(layer_model.layer_centers))
    low_atom = min(atoms, key=lambda atom: atom.z)
    high_atom = max(atoms, key=lambda atom: atom.z)

    low_coord = transformed_coord(low_atom, layer_model, center_z, 0.9)
    high_coord = transformed_coord(high_atom, layer_model, center_z, 0.9)

    assert low_coord[2] > low_atom.z
    assert high_coord[2] < high_atom.z
    assert low_coord[0] == low_atom.x
    assert high_coord[1] == high_atom.y


def test_parent_derived_transform_preserves_identity_fields(tmp_path: Path) -> None:
    parent = tmp_path / "parent.pdb"
    out = tmp_path / "scaled.pdb"
    write_parent_fixture(parent)
    lines, atoms = parse_pdb_atom_lines(parent)
    layer_model = infer_layers_from_ca_z([atom.z for atom in atoms if atom.is_ca])
    center_z = float(np.mean(layer_model.layer_centers))
    spec = ParentDerivedRiseSpec("scaled", 3.35, 3.35 / 3.40)

    write_parent_derived_variant(lines, atoms, spec, layer_model, center_z, out)
    _, variant_atoms = parse_pdb_atom_lines(out)

    assert len(variant_atoms) == len(atoms)
    assert identity_preserved(atoms, variant_atoms)


def test_report_and_csv_outputs_are_written_with_blocked_reference(monkeypatch, tmp_path: Path) -> None:
    parent = tmp_path / "parent.pdb"
    write_parent_fixture(parent)

    def fake_score(_path: Path):
        return {
            "observed_A_d_A": 1.0,
            "A_error_A": 0.0,
            "A_score": 1.0,
            "observed_B_d_A": 1.0,
            "B_error_A": 0.0,
            "B_score": 1.0,
            "observed_C_d_A": 1.0,
            "C_error_A": 0.0,
            "C_score": 1.0,
            "observed_D_d_A": 1.0,
            "D_error_A": 0.0,
            "D_score": 1.0,
        }

    monkeypatch.setattr("scripts.run_parent_derived_rise_bridge.score_pdb_abcd", fake_score)
    score_csv = tmp_path / "scores.csv"
    geometry_csv = tmp_path / "geometry.csv"
    report = tmp_path / "report.md"

    scores, geometry = run_bridge(parent, tmp_path / "coords", score_csv, geometry_csv, report)

    assert score_csv.exists()
    assert geometry_csv.exists()
    assert report.exists()
    assert len(scores) == 4
    assert len(geometry) == 4
    assert bridge_recommendation(scores) == "bridge_blocked_reference_not_reproduced"
    assert "Reference reproduces parent: `False`" in report.read_text(encoding="utf-8")


def test_report_text_contains_parent_derived_cautions(tmp_path: Path) -> None:
    parent = tmp_path / "parent.pdb"
    write_parent_fixture(parent)
    scores = pd.DataFrame(
        [
            {
                "variant_id": "parent_derived_reference",
                "requested_nominal_rise_A": "",
                "axial_scale": 1.0,
                "realized_rise_metric_A": 1.0,
                "observed_A_d_A": 7.0,
                "observed_B_d_A": 6.0,
                "observed_C_d_A": 5.745,
                "observed_D_d_A": 7.276,
                "combined_CD_abs_error_A": 0.0,
                "status": "scored",
                "reference_reproduces_parent": True,
            }
        ]
    )
    geometry = pd.DataFrame(
        [
            {
                "variant_id": "parent_derived_reference",
                "z_span_A": 1.0,
                "mean_ca_radius_A": 1.0,
                "median_interstrand_nn_ca_distance_A": 1.0,
                "median_ca_rise_A": 1.0,
                "atom_count": 1,
                "carboxylate_present": True,
            }
        ]
    )
    layer_model = infer_layers_from_ca_z([0.0, 1.0, 2.0])

    text = build_report_text(scores, geometry, parent, layer_model)

    assert "parent-derived coordinate transform" in text or "parent-derived" in text
    assert "pseudo-generator bridge was rejected" in text
    assert "controlled diagnostic coordinate transforms" in text
