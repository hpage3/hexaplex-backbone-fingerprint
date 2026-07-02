import numpy as np

from hexaplex_backbone_fingerprint.geometry import dihedral_degrees
from scripts.audit_backbone_torsion_repeat import (
    Residue,
    identify_repeat_windows,
    missing_backbone_atoms,
    omega_near_trans,
    residue_torsions,
)


def residue(chain, resseq, resname, atoms):
    return Residue(
        chain=chain,
        resseq=resseq,
        resname=resname,
        atoms={name: np.array(coord, dtype=float) for name, coord in atoms.items()},
        atom_names_in_order=tuple(atoms),
    )


def test_dihedral_angle_calculation_trans_like():
    angle = dihedral_degrees(
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 0.0]),
        np.array([2.0, 1.0, 0.0]),
    )
    assert abs(abs(angle) - 180.0) < 1e-6


def test_phi_psi_omega_extraction_on_synthetic_peptide():
    prev_res = residue("A", 1, "GLU", {"C": (0.0, 0.0, 0.0)})
    current = residue(
        "A",
        2,
        "CYP",
        {"N": (1.0, 0.0, 0.0), "CA": (1.0, 1.0, 0.0), "C": (2.0, 1.0, 0.0), "O": (2.5, 1.0, 0.0)},
    )
    next_res = residue("A", 3, "GLU", {"N": (2.0, 2.0, 0.0), "CA": (3.0, 2.0, 0.0)})
    torsions = residue_torsions(prev_res, current, next_res)
    assert set(torsions) == {"phi_deg", "psi_deg", "omega_deg"}
    assert np.isfinite(torsions["phi_deg"])
    assert np.isfinite(torsions["psi_deg"])
    assert np.isfinite(torsions["omega_deg"])


def test_missing_atom_detection():
    res = residue("A", 1, "GLU", {"N": (0, 0, 0), "CA": (1, 0, 0)})
    assert missing_backbone_atoms(res) == ["C", "O"]


def test_repeat_window_identification():
    residues = [
        residue("A", 1, "CYP", {"CA": (0, 0, 0)}),
        residue("A", 2, "GLU", {"CA": (3, 0, 0)}),
        residue("A", 3, "CYP", {"CA": (6, 0, 0)}),
    ]
    windows = identify_repeat_windows(residues)
    assert len(windows) == 2
    assert windows[0]["repeat_residue_names"] == "CYP->GLU"
    assert windows[0]["ca_to_ca_distance_A"] == 3.0


def test_omega_near_180_classification():
    assert omega_near_trans(179.0)
    assert omega_near_trans(-170.0)
    assert not omega_near_trans(90.0)
    assert not omega_near_trans(np.nan)
