from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_parent_axial_layers import infer_layers_from_ca_z
from scripts.generate_global_deformation_variants import parse_pdb_atom_lines
from scripts.run_parent_derived_rise_bridge import ParentDerivedRiseSpec, identity_preserved, write_parent_derived_variant
from scripts.run_parent_derived_rise_fine_scan import (
    best_score_row,
    best_score_rows,
    fine_scan_recommendation,
    fine_scan_specs,
    format_scale,
    nominal_rise_equiv,
    output_path,
    plateau_text,
    required_score_columns,
    run_scan,
    variant_id_for_scale,
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


def test_scale_values_normalize_to_stable_variant_ids() -> None:
    assert format_scale(1.0) == "1p0000"
    assert format_scale(0.975) == "0p9750"
    assert variant_id_for_scale(0.975) == "parent_derived_scale_0p9750"
    assert [spec.variant_id for spec in fine_scan_specs([1.0, 0.995])] == [
        "parent_derived_scale_1p0000",
        "parent_derived_scale_0p9950",
    ]
    assert nominal_rise_equiv(0.975) == 0.975 * 3.40


def test_scale_one_preserves_coordinates_and_identity(tmp_path: Path) -> None:
    parent = tmp_path / "parent.pdb"
    out = tmp_path / "scale1.pdb"
    write_parent_fixture(parent)
    lines, atoms = parse_pdb_atom_lines(parent)
    layer_model = infer_layers_from_ca_z([atom.z for atom in atoms if atom.is_ca])
    center_z = float(np.mean(layer_model.layer_centers))
    spec = ParentDerivedRiseSpec("scale1", 3.40, 1.0)

    write_parent_derived_variant(lines, atoms, spec, layer_model, center_z, out)
    _, variant_atoms = parse_pdb_atom_lines(out)

    assert identity_preserved(atoms, variant_atoms)
    for parent_atom, variant_atom in zip(atoms, variant_atoms):
        assert np.allclose(parent_atom.coord, variant_atom.coord)


def test_scale_less_than_one_compresses_z_around_parent_center(tmp_path: Path) -> None:
    parent = tmp_path / "parent.pdb"
    out = tmp_path / "scale0975.pdb"
    write_parent_fixture(parent)
    lines, atoms = parse_pdb_atom_lines(parent)
    layer_model = infer_layers_from_ca_z([atom.z for atom in atoms if atom.is_ca])
    center_z = float(np.mean(layer_model.layer_centers))
    spec = ParentDerivedRiseSpec("scale0975", nominal_rise_equiv(0.975), 0.975)

    write_parent_derived_variant(lines, atoms, spec, layer_model, center_z, out)
    _, variant_atoms = parse_pdb_atom_lines(out)

    assert identity_preserved(atoms, variant_atoms)
    assert min(atom.z for atom in variant_atoms) > min(atom.z for atom in atoms)
    assert max(atom.z for atom in variant_atoms) < max(atom.z for atom in atoms)


def test_output_paths_and_required_columns(tmp_path: Path) -> None:
    spec = fine_scan_specs([0.975])[0]

    assert output_path(tmp_path, spec) == tmp_path / "parent_derived_scale_0p9750.pdb"
    assert "variant_id" in required_score_columns()
    assert "nominal_rise_equiv_A" in required_score_columns()
    assert "observed_C_d_A" in required_score_columns()


def test_report_csv_outputs_written_and_blocked_when_reference_fails(monkeypatch, tmp_path: Path) -> None:
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

    monkeypatch.setattr("scripts.run_parent_derived_rise_fine_scan.score_pdb_abcd", fake_score)
    score_csv = tmp_path / "scores.csv"
    geometry_csv = tmp_path / "geometry.csv"
    report = tmp_path / "report.md"

    scores, geometry = run_scan(parent, tmp_path / "coords", score_csv, geometry_csv, report, scales=[1.0, 0.975])

    assert score_csv.exists()
    assert geometry_csv.exists()
    assert report.exists()
    assert len(scores) == 2
    assert len(geometry) == 2
    assert fine_scan_recommendation(scores) == "fine_scan_blocked_reference_not_reproduced"
    assert "Reference reproduces parent: `False`" in report.read_text(encoding="utf-8")


def test_best_score_and_recommendation_success() -> None:
    scores = pd.DataFrame(
        [
            {
                "variant_id": "parent_derived_scale_1p0000",
                "reference_reproduces_parent": True,
                "observed_C_d_A": 5.745,
                "observed_D_d_A": 7.276,
                "combined_CD_abs_error_A": 0.17,
            },
            {
                "variant_id": "parent_derived_scale_0p9825",
                "reference_reproduces_parent": True,
                "observed_C_d_A": 5.6422,
                "observed_D_d_A": 7.2756,
                "combined_CD_abs_error_A": 0.0667,
            },
            {
                "variant_id": "parent_derived_scale_0p9800",
                "reference_reproduces_parent": True,
                "observed_C_d_A": 5.6422,
                "observed_D_d_A": 7.2756,
                "combined_CD_abs_error_A": 0.0667,
            },
            {
                "variant_id": "parent_derived_scale_0p9775",
                "reference_reproduces_parent": True,
                "observed_C_d_A": 5.6422,
                "observed_D_d_A": 7.2756,
                "combined_CD_abs_error_A": 0.0667,
            },
            {
                "variant_id": "parent_derived_scale_0p9750",
                "reference_reproduces_parent": True,
                "observed_C_d_A": 5.6422,
                "observed_D_d_A": 7.2756,
                "combined_CD_abs_error_A": 0.0667,
            },
            {
                "variant_id": "parent_derived_scale_0p9725",
                "reference_reproduces_parent": True,
                "observed_C_d_A": 5.6422,
                "observed_D_d_A": 7.2756,
                "combined_CD_abs_error_A": 0.0667,
            },
        ]
    )

    best_rows = best_score_rows(scores)
    assert best_score_row(scores)["variant_id"] == "parent_derived_scale_0p9825"
    assert best_rows["variant_id"].tolist() == [
        "parent_derived_scale_0p9825",
        "parent_derived_scale_0p9800",
        "parent_derived_scale_0p9775",
        "parent_derived_scale_0p9750",
        "parent_derived_scale_0p9725",
    ]
    assert plateau_text(best_rows) == "parent_derived_scale_0p9825 through parent_derived_scale_0p9725"
    assert fine_scan_recommendation(scores) == "fine_scan_success"


def test_report_wording_distinguishes_provenance_and_plateau(tmp_path: Path) -> None:
    from scripts.run_parent_derived_rise_fine_scan import build_report_text

    parent = tmp_path / "parent.pdb"
    write_parent_fixture(parent)
    scores = pd.DataFrame(
        [
            {
                "variant_id": "parent_derived_scale_1p0000",
                "axial_scale": 1.0,
                "nominal_rise_equiv_A": 3.4,
                "realized_rise_metric_A": 1.18,
                "observed_C_d_A": 5.7454,
                "observed_D_d_A": 7.2756,
                "combined_CD_abs_error_A": 0.1698,
                "reference_reproduces_parent": True,
                "status": "scored",
            },
            {
                "variant_id": "parent_derived_scale_0p9825",
                "axial_scale": 0.9825,
                "nominal_rise_equiv_A": 3.3405,
                "realized_rise_metric_A": 1.16,
                "observed_C_d_A": 5.6422,
                "observed_D_d_A": 7.2756,
                "combined_CD_abs_error_A": 0.0667,
                "reference_reproduces_parent": True,
                "status": "scored",
            },
            {
                "variant_id": "parent_derived_scale_0p9750",
                "axial_scale": 0.975,
                "nominal_rise_equiv_A": 3.315,
                "realized_rise_metric_A": 1.15,
                "observed_C_d_A": 5.6422,
                "observed_D_d_A": 7.2756,
                "combined_CD_abs_error_A": 0.0667,
                "reference_reproduces_parent": True,
                "status": "scored",
            },
            {
                "variant_id": "parent_derived_scale_0p9700",
                "axial_scale": 0.97,
                "nominal_rise_equiv_A": 3.298,
                "realized_rise_metric_A": 1.14,
                "observed_C_d_A": 5.6422,
                "observed_D_d_A": 7.1923,
                "combined_CD_abs_error_A": 0.1499,
                "reference_reproduces_parent": True,
                "status": "scored",
            },
        ]
    )
    geometry = pd.DataFrame(
        [
            {
                "variant_id": row["variant_id"],
                "z_span_A": 1.0,
                "mean_ca_radius_A": 1.0,
                "median_interstrand_nn_ca_distance_A": 1.0,
                "median_ca_rise_A": 1.0,
                "atom_count": 1,
                "carboxylate_present": True,
            }
            for _, row in scores.iterrows()
        ]
    )

    text = build_report_text(scores, geometry, parent, layer_count=3, parent_reference_rise_metric_A=1.18)

    assert "best combined-error plateau" in text
    assert "Best combined-error plateau: `parent_derived_scale_0p9825 through parent_derived_scale_0p9750`" in text
    assert "does not imply a unique optimum" in text
    assert "0.9700 keeps C" in text
    assert "not proof of exact original pNAB/YAML provenance" in text
    assert "failed pseudo reconstructed bridge" in text
    assert "Model Scope / Asem Symmetry Caution" in text
    assert "six-fold-symmetric parent-derived coordinate family" in text
    assert "should not be interpreted as pNAB determining the physical twist/rise geometry" in text
    assert "pNAB-derived construction imposed a six-fold backbone-symmetry assumption" in text
    assert "melamine/triamino and cyanuric/triketo backbone exit vectors may not be chemically equivalent" in text
    assert "separate three-fold backbone symmetry" in text
    assert "new peptide-plane model track, not another one-dimensional parent-derived rise scan" in text
