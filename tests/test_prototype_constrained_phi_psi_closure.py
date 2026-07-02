import numpy as np

from hexaplex_backbone_fingerprint.geometry import dihedral_degrees
from scripts.audit_backbone_torsion_repeat import Residue
from scripts.prototype_constrained_phi_psi_closure import (
    build_closure_window,
    classify_solver_result,
    endpoint_closure_error,
    perturbation_values,
    reconstruct_endpoint,
    run_window_closure,
    solve_one_torsion,
)


def residue(chain, resseq, resname, atoms):
    return Residue(
        chain=chain,
        resseq=resseq,
        resname=resname,
        atoms={name: np.array(coord, dtype=float) for name, coord in atoms.items()},
        atom_names_in_order=tuple(atoms),
    )


def synthetic_residues():
    return [
        residue("A", 1, "PRE", {"N": (-1.0, -1.0, 0.0), "CA": (-0.5, -0.5, 0.0), "C": (0.0, 0.0, 0.0), "O": (0.0, 0.0, 1.0)}),
        residue("A", 2, "CYP", {"N": (1.0, 0.0, 0.0), "CA": (1.0, 1.0, 0.0), "C": (2.0, 1.0, 0.0), "O": (2.0, 1.0, 1.0)}),
        residue("A", 3, "GLU", {"N": (2.0, 2.0, 0.0), "CA": (3.0, 2.0, 0.0), "C": (3.0, 3.0, 0.0), "O": (3.0, 3.0, 1.0)}),
    ]


def test_dihedral_calculation_available():
    angle = dihedral_degrees(
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 0.0]),
        np.array([2.0, 1.0, 0.0]),
    )
    assert abs(abs(angle) - 180.0) < 1e-6


def test_endpoint_closure_error():
    assert endpoint_closure_error(np.array([0, 0, 0]), np.array([0, 3, 4])) == 5.0


def test_torsion_perturbation_table_construction():
    assert perturbation_values(-2, 2, 1) == [-2.0, -1.0, 0.0, 1.0, 2.0]


def test_solver_result_classification():
    assert classify_solver_result(0.049, 0.05)
    assert not classify_solver_result(0.051, 0.05)


def test_fallback_behavior_on_synthetic_chain():
    window = build_closure_window(synthetic_residues(), "A", 1)
    endpoint = reconstruct_endpoint(
        window,
        window.baseline_torsions["phi0_deg"],
        window.baseline_torsions["psi0_deg"],
        window.baseline_torsions["omega0_deg"],
    )
    assert endpoint_closure_error(endpoint, window.end_ca) < 1e-6
    delta, error, method = solve_one_torsion(
        window,
        "phi0_deg",
        0.0,
        "psi0_deg",
        window.baseline_torsions["omega0_deg"],
        force_fallback=True,
    )
    assert abs(delta) < 1e-9
    assert error < 1e-6
    assert method == "fallback_grid"
    rows = run_window_closure(window, "phi0_deg", [0.0], 0.05)
    assert rows[0]["closure_success"] is True
