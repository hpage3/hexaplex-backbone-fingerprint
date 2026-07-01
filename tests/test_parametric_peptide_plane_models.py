from __future__ import annotations

from hexaplex_backbone_fingerprint.parametric_peptide_plane_models import (
    ModelParameters,
    generate_model_atoms,
    manifest_row,
    measured_normal_to_axis_angle,
    repeat_center,
    write_pdb,
)
from hexaplex_backbone_fingerprint.pdb_parser import parse_pdb
from hexaplex_backbone_fingerprint.peptide_planes import build_peptide_planes


def test_parametric_model_atom_count_and_strands():
    params = ModelParameters(n_strands=6, repeats_per_strand=4)

    atoms = generate_model_atoms(params)

    assert len(atoms) == 6 * 4 * 6
    assert sorted({atom.chain for atom in atoms}) == ["A", "B", "C", "D", "E", "F"]
    assert max(atom.repeat_index for atom in atoms) == 3


def test_orientation_angle_parameter_is_respected():
    params = ModelParameters(plane_normal_to_axis_deg=60.0, plane_azimuth_deg=30.0)

    assert abs(measured_normal_to_axis_angle(params) - 60.0) < 1e-6


def test_manifest_row_records_core_parameters(tmp_path):
    params = ModelParameters(twist_deg=32.0, rise_A=3.4, plane_normal_to_axis_deg=90.0)

    row = manifest_row(params, tmp_path / "model.pdb", tmp_path / "model.xyz", atom_count=123)

    assert row["twist_deg"] == 32.0
    assert row["rise_A"] == 3.4
    assert row["plane_normal_to_axis_deg"] == 90.0
    assert row["atom_count"] == 123
    assert row["uniform_adjacent_z_offset_A"] == 0.0
    assert row["z_offset_mode"] == "uniform_adjacent"


def test_generated_pdb_is_peptide_plane_parser_compatible(tmp_path):
    params = ModelParameters(n_strands=2, repeats_per_strand=3)
    atoms = generate_model_atoms(params)
    pdb_path = write_pdb(atoms, tmp_path / "parametric.pdb", params)

    resmap = parse_pdb(pdb_path)
    planes = build_peptide_planes(resmap)

    assert len(planes) == 2 * 3


def test_uniform_adjacent_z_offset_staggers_strands():
    params = ModelParameters(n_strands=6, uniform_adjacent_z_offset_A=0.5)

    z0 = repeat_center(params, strand_index=0, repeat_index=0)[2]
    z1 = repeat_center(params, strand_index=1, repeat_index=0)[2]
    z5 = repeat_center(params, strand_index=5, repeat_index=0)[2]

    assert abs(z1 - z0 - 0.5) < 1e-9
    assert abs(z5 - z0 - 2.5) < 1e-9


def test_zero_z_offset_reproduces_synchronized_register():
    params = ModelParameters(n_strands=6, uniform_adjacent_z_offset_A=0.0)

    strand_z_values = [repeat_center(params, strand_index=i, repeat_index=0)[2] for i in range(6)]

    assert strand_z_values == [0.0] * 6


def test_legacy_strand_z_offset_alias_records_uniform_offset():
    params = ModelParameters(strand_z_offset_A=0.25)
    row = manifest_row(params, "model.pdb", "model.xyz", atom_count=1)

    assert params.uniform_adjacent_z_offset_A == 0.25
    assert row["uniform_adjacent_z_offset_A"] == 0.25
    assert row["strand_z_offset_A"] == 0.25
