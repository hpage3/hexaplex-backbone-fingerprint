from __future__ import annotations

import pandas as pd

from scripts.audit_register_defined_layer_models import (
    AtomRecord,
    build_register_layers,
    build_report_text,
    build_residues,
    layer_records_for_model,
    model_summary,
    parse_atom_line,
    peptide_plane_z_mean,
    register_to_zslice_mapping,
)


def atom(chain: str, resnum: int, name: str, z: float, serial: int = 1, resname: str = "GLU") -> AtomRecord:
    return AtomRecord(serial, name, resname, chain, resnum, "", name[0], 0.0, 0.0, z)


def synthetic_atoms() -> list[AtomRecord]:
    atoms = []
    serial = 1
    for chain in ["A", "B"]:
        for resnum in [10, 11, 12, 13]:
            for name, dz in [("N", 0.0), ("CA", 0.1), ("C", 0.2), ("O", 0.3)]:
                atoms.append(atom(chain, resnum, name, resnum + dz, serial, "GLU"))
                serial += 1
    return atoms


def test_parse_minimal_pdb_atoms_into_residues() -> None:
    line = "ATOM      7  CA  GLU A  12       1.000   2.000   3.000  1.00 20.00           C"
    rec = parse_atom_line(line)
    assert rec.atom_serial == 7
    assert rec.atom_name == "CA"
    residues = build_residues([rec])
    assert len(residues) == 1
    assert residues[0].residue_order_index == 0


def test_assign_residue_order_index_by_chain_order() -> None:
    residues = build_residues(synthetic_atoms())
    a = [r.residue_order_index for r in residues if r.chain_id == "A"]
    b = [r.residue_order_index for r in residues if r.chain_id == "B"]
    assert a == [0, 1, 2, 3]
    assert b == [0, 1, 2, 3]


def test_build_residue_index_and_repeat_pair_layers() -> None:
    residues = build_residues(synthetic_atoms())
    residue_layers = layer_records_for_model("residue_index_layer", residues)
    repeat_layers = layer_records_for_model("repeat_pair_layer", residues)
    assert len(residue_layers) == 4
    assert len(repeat_layers) == 2
    assert repeat_layers[0]["residue_order_indices"] == "0,1"


def test_build_ca_register_layer() -> None:
    residues = build_residues(synthetic_atoms())
    layers = layer_records_for_model("ca_register_layer", residues)
    assert len(layers) == 4
    assert all(row["atom_count"] == 2 for row in layers)


def test_peptide_plane_z_mean_requires_complete_backbone() -> None:
    complete = [atom("A", 1, name, i) for i, name in enumerate(["N", "CA", "C", "O"])]
    incomplete = complete[:-1]
    assert peptide_plane_z_mean(complete) == 1.5
    assert peptide_plane_z_mean(incomplete) is None


def test_compute_layer_spacing_metrics_and_split_residue_count() -> None:
    residues = build_residues(synthetic_atoms())
    composition = build_register_layers(residues)
    row = model_summary("residue_index_layer", composition, split_residue_count=0, total_residues=len(residues), expected_chains=["A", "B"])
    assert row["layer_count"] == 4
    assert row["split_residue_count"] == 0
    assert row["residues_split_fraction"] == 0


def test_mapping_from_register_layer_atoms_to_zslice_layers() -> None:
    residues = build_residues(synthetic_atoms())
    composition = build_register_layers(residues)
    rows = []
    for r in residues:
        for a in r.atoms:
            rows.append(
                {
                    "chain_id": a.chain_id,
                    "residue_number": a.residue_number,
                    "residue_name": a.residue_name,
                    "atom_name": a.atom_name,
                    "layer_index": r.residue_order_index,
                    "z": a.z,
                }
            )
    mapping = register_to_zslice_mapping(composition, pd.DataFrame(rows))
    residue_mapping = mapping[mapping["model_name"] == "residue_index_layer"]
    assert not residue_mapping.empty
    assert residue_mapping["primary_zslice_atom_fraction"].max() == 1.0


def test_report_text_contains_required_phrases() -> None:
    residues = build_residues(synthetic_atoms())
    composition = build_register_layers(residues)
    summary = pd.DataFrame(
        [
            model_summary("residue_index_layer", composition, 0, len(residues), ["A", "B"]),
            model_summary("repeat_pair_layer", composition, 0, len(residues), ["A", "B"]),
        ]
    )
    mapping = pd.DataFrame(
        {
            "model_name": ["residue_index_layer"],
            "zslice_layer_count": [1],
            "primary_zslice_atom_fraction": [1.0],
        }
    )
    text = build_report_text(
        source_pdb="parent.pdb",  # type: ignore[arg-type]
        atoms=synthetic_atoms(),
        residues=residues,
        summary=summary,
        mapping=mapping,
        recommendation="z-slice model is better as a computational deformation coordinate",
        plots=[],
    )
    assert "Register-Defined Layer Model Audit" in text
    assert "computational z-slices" in text
    assert "chemically/register-defined layers" in text
    assert "should not be interpreted as unique chemical hexad layers" in text
