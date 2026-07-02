import numpy as np

from hexaplex_backbone_fingerprint.geometry import dihedral_degrees
from scripts.audit_backbone_torsion_repeat import Residue
from scripts.prototype_constrained_phi_psi_closure import (
    build_closure_window,
    classify_solver_result,
    endpoint_closure_error,
    perturbation_values,
    reconstruct_endpoint,
    reconstruct_two_unit_endpoint,
    run_window_closure,
    solve_one_torsion,
    solve_two_torsions_grid,
    solved_torsion_names_for_two_mode,
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
        residue("A", 4, "CYP", {"N": (4.0, 3.0, 0.0), "CA": (4.0, 4.0, 0.0), "C": (5.0, 4.0, 0.0), "O": (5.0, 4.0, 1.0)}),
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
    assert rows[0]["solve_mode"] == "one_torsion"


def test_two_unit_reconstruction_baseline_endpoint():
    window = build_closure_window(synthetic_residues(), "A", 1)
    endpoint = reconstruct_two_unit_endpoint(
        window,
        window.baseline_torsions["phi0_deg"],
        window.baseline_torsions["psi0_deg"],
        window.baseline_torsions["omega0_deg"],
        window.baseline_torsions["phi1_deg"],
        window.baseline_torsions["psi1_deg"],
        window.baseline_torsions["omega1_deg"],
    )
    assert endpoint_closure_error(endpoint, window.two_unit_end_ca) < 1e-6


def test_two_dimensional_grid_search_bookkeeping_and_best_selection():
    window = build_closure_window(synthetic_residues(), "A", 1)
    solved_1, solved_2 = solved_torsion_names_for_two_mode("phi0_deg")
    assert (solved_1, solved_2) == ("psi0_deg", "phi1_deg")
    delta_1, delta_2, error = solve_two_torsions_grid(
        window,
        "phi0_deg",
        0.0,
        solved_1,
        solved_2,
        window.baseline_torsions["omega0_deg"],
        coarse_radius_deg=2.0,
        coarse_step_deg=1.0,
        refine_radius_deg=0.5,
        refine_step_deg=0.25,
    )
    assert abs(delta_1) < 1e-9
    assert abs(delta_2) < 1e-9
    assert error < 1e-6


def test_two_solve_mode_improves_or_equals_one_solve_on_synthetic_fixture():
    window = build_closure_window(synthetic_residues(), "A", 1)
    one = run_window_closure(window, "phi0_deg", [2.0], 0.05, solve_mode="one_torsion")[0]
    two = run_window_closure(window, "phi0_deg", [2.0], 0.05, solve_mode="two_torsion")[0]
    assert two["endpoint_error_A"] <= one["endpoint_error_A"] + 1e-9
    assert two["solve_mode"] == "two_torsion"
    assert two["solved_torsion_1_name"]
    assert two["solved_torsion_2_name"]
