"""Prototype constrained two-residue phi/psi closure for ideal Hexaflex backbones.

This prototype does not perform the full systematic search. It tests whether a
two-residue repeat can be rebuilt from fixed bond lengths/angles with fixed CA
anchors while perturbing one torsion and solving one remaining torsion on a
small deterministic grid. SciPy is used if available; otherwise the grid search
is reported as lower-confidence fallback.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hexaplex_backbone_fingerprint.geometry import angle_between_vectors, dihedral_degrees, distance, normalize
from scripts.audit_backbone_torsion_repeat import PARENT_LABEL, Residue, parse_residues

try:  # pragma: no cover - depends on optional local environment
    from scipy.optimize import minimize_scalar

    SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover - exercised by fallback tests instead
    minimize_scalar = None
    SCIPY_AVAILABLE = False


@dataclass(frozen=True)
class ClosureWindow:
    """Backbone geometry for one two-residue closure window."""

    model_id: str
    chain_id: str
    repeat_start_index: int
    residue_names: str
    prev_residue: Residue
    first_residue: Residue
    second_residue: Residue
    next_residue: Residue | None
    bond_lengths: dict[str, float]
    bond_angles: dict[str, float]
    baseline_torsions: dict[str, float]
    start_ca: np.ndarray
    end_ca: np.ndarray
    two_unit_end_ca: np.ndarray | None


def find_parent_pdb() -> Path:
    """Resolve ideal parent PDB from existing summary."""
    summary = pd.read_csv(ROOT / "outputs/six_strand_first_panel/six_strand_first_panel_summary.csv")
    row = summary[summary["label"] == PARENT_LABEL]
    if row.empty:
        raise FileNotFoundError("Could not locate ideal parent PDB.")
    return Path(str(row.iloc[0]["source_path"]))


def angle_degrees(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return angle ABC in degrees."""
    return angle_between_vectors(a - b, c - b)


def point_from_internal(a: np.ndarray, b: np.ndarray, c: np.ndarray, length: float, angle_deg: float, dihedral_deg: float) -> np.ndarray:
    """Construct point D from A-B-C plus |C-D|, angle B-C-D, and dihedral A-B-C-D."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    bc = normalize(c - b)
    n = normalize(np.cross(b - a, bc))
    m = np.cross(n, bc)
    theta = np.radians(angle_deg)
    phi = np.radians(dihedral_deg)
    direction = -np.cos(theta) * bc + np.sin(theta) * (np.cos(phi) * m + np.sin(phi) * n)
    return c + length * direction


def build_closure_window(residues: list[Residue], chain_id: str, start_index: int) -> ClosureWindow:
    """Build baseline geometry for residues start_index/start_index+1 with previous C."""
    if start_index <= 0 or start_index + 1 >= len(residues):
        raise ValueError("Need previous residue and two-residue window.")
    prev_res = residues[start_index - 1]
    first = residues[start_index]
    second = residues[start_index + 1]
    next_res = residues[start_index + 2] if start_index + 2 < len(residues) else None
    required = {"N", "CA", "C", "O"}
    for residue in [prev_res, first, second] + ([next_res] if next_res is not None else []):
        missing = required - set(residue.atoms)
        if missing:
            raise ValueError(f"Residue {chain_id}:{residue.resseq} missing atoms {sorted(missing)}")
    p_c_prev = prev_res.atoms["C"]
    p_n0 = first.atoms["N"]
    p_ca0 = first.atoms["CA"]
    p_c0 = first.atoms["C"]
    p_n1 = second.atoms["N"]
    p_ca1 = second.atoms["CA"]
    p_c1 = second.atoms["C"]
    bond_lengths = {
        "CA0_C0": distance(p_ca0, p_c0),
        "C0_N1": distance(p_c0, p_n1),
        "N1_CA1": distance(p_n1, p_ca1),
        "CA1_C1": distance(p_ca1, p_c1),
    }
    bond_angles = {
        "N0_CA0_C0": angle_degrees(p_n0, p_ca0, p_c0),
        "CA0_C0_N1": angle_degrees(p_ca0, p_c0, p_n1),
        "C0_N1_CA1": angle_degrees(p_c0, p_n1, p_ca1),
        "N1_CA1_C1": angle_degrees(p_n1, p_ca1, p_c1),
    }
    baseline_torsions = {
        "phi0_deg": dihedral_degrees(p_c_prev, p_n0, p_ca0, p_c0),
        "psi0_deg": dihedral_degrees(p_n0, p_ca0, p_c0, p_n1),
        "omega0_deg": dihedral_degrees(p_ca0, p_c0, p_n1, p_ca1),
    }
    two_unit_end_ca = None
    if next_res is not None:
        p_n2 = next_res.atoms["N"]
        p_ca2 = next_res.atoms["CA"]
        bond_lengths.update(
            {
                "C1_N2": distance(p_c1, p_n2),
                "N2_CA2": distance(p_n2, p_ca2),
            }
        )
        bond_angles.update(
            {
                "CA1_C1_N2": angle_degrees(p_ca1, p_c1, p_n2),
                "C1_N2_CA2": angle_degrees(p_c1, p_n2, p_ca2),
            }
        )
        baseline_torsions.update(
            {
                "phi1_deg": dihedral_degrees(p_c0, p_n1, p_ca1, p_c1),
                "psi1_deg": dihedral_degrees(p_n1, p_ca1, p_c1, p_n2),
                "omega1_deg": dihedral_degrees(p_ca1, p_c1, p_n2, p_ca2),
            }
        )
        two_unit_end_ca = p_ca2
    return ClosureWindow(
        model_id=PARENT_LABEL,
        chain_id=chain_id,
        repeat_start_index=start_index,
        residue_names=f"{first.resname}->{second.resname}",
        prev_residue=prev_res,
        first_residue=first,
        second_residue=second,
        next_residue=next_res,
        bond_lengths=bond_lengths,
        bond_angles=bond_angles,
        baseline_torsions=baseline_torsions,
        start_ca=p_ca0,
        end_ca=p_ca1,
        two_unit_end_ca=two_unit_end_ca,
    )


def reconstruct_endpoint(window: ClosureWindow, phi0_deg: float, psi0_deg: float, omega0_deg: float) -> np.ndarray:
    """Reconstruct second CA endpoint from baseline internal geometry."""
    p_c_prev = window.prev_residue.atoms["C"]
    p_n0 = window.first_residue.atoms["N"]
    p_ca0 = window.first_residue.atoms["CA"]
    p_c0 = point_from_internal(
        p_c_prev,
        p_n0,
        p_ca0,
        window.bond_lengths["CA0_C0"],
        window.bond_angles["N0_CA0_C0"],
        phi0_deg,
    )
    p_n1 = point_from_internal(
        p_n0,
        p_ca0,
        p_c0,
        window.bond_lengths["C0_N1"],
        window.bond_angles["CA0_C0_N1"],
        psi0_deg,
    )
    p_ca1 = point_from_internal(
        p_ca0,
        p_c0,
        p_n1,
        window.bond_lengths["N1_CA1"],
        window.bond_angles["C0_N1_CA1"],
        omega0_deg,
    )
    return p_ca1


def reconstruct_two_unit_endpoint(
    window: ClosureWindow,
    phi0_deg: float,
    psi0_deg: float,
    omega0_deg: float,
    phi1_deg: float,
    psi1_deg: float,
    omega1_deg: float,
) -> np.ndarray:
    """Reconstruct the C-alpha endpoint after two peptide units."""
    if window.next_residue is None or window.two_unit_end_ca is None:
        raise ValueError("Two-unit reconstruction requires a following residue.")
    p_c_prev = window.prev_residue.atoms["C"]
    p_n0 = window.first_residue.atoms["N"]
    p_ca0 = window.first_residue.atoms["CA"]
    p_c0 = point_from_internal(
        p_c_prev,
        p_n0,
        p_ca0,
        window.bond_lengths["CA0_C0"],
        window.bond_angles["N0_CA0_C0"],
        phi0_deg,
    )
    p_n1 = point_from_internal(
        p_n0,
        p_ca0,
        p_c0,
        window.bond_lengths["C0_N1"],
        window.bond_angles["CA0_C0_N1"],
        psi0_deg,
    )
    p_ca1 = point_from_internal(
        p_ca0,
        p_c0,
        p_n1,
        window.bond_lengths["N1_CA1"],
        window.bond_angles["C0_N1_CA1"],
        omega0_deg,
    )
    p_c1 = point_from_internal(
        p_c0,
        p_n1,
        p_ca1,
        window.bond_lengths["CA1_C1"],
        window.bond_angles["N1_CA1_C1"],
        phi1_deg,
    )
    p_n2 = point_from_internal(
        p_n1,
        p_ca1,
        p_c1,
        window.bond_lengths["C1_N2"],
        window.bond_angles["CA1_C1_N2"],
        psi1_deg,
    )
    return point_from_internal(
        p_ca1,
        p_c1,
        p_n2,
        window.bond_lengths["N2_CA2"],
        window.bond_angles["C1_N2_CA2"],
        omega1_deg,
    )


def endpoint_closure_error(endpoint: np.ndarray, target_endpoint: np.ndarray) -> float:
    """Return endpoint closure error in Angstrom."""
    return distance(endpoint, target_endpoint)


def perturbation_values(min_delta: int = -5, max_delta: int = 5, step: int = 1) -> list[float]:
    """Return inclusive perturbation grid values."""
    if step <= 0:
        raise ValueError("step must be positive")
    return [float(value) for value in range(min_delta, max_delta + 1, step)]


def solve_one_torsion_grid(
    window: ClosureWindow,
    fixed_torsion_name: str,
    fixed_delta_deg: float,
    solved_torsion_name: str,
    omega_fixed_deg: float,
    search_radius_deg: float = 30.0,
    step_deg: float = 0.25,
) -> tuple[float, float]:
    """Solve one torsion by deterministic grid search around its baseline value."""
    baseline = window.baseline_torsions
    fixed_value = baseline[fixed_torsion_name] + fixed_delta_deg
    solved_baseline = baseline[solved_torsion_name]
    best_delta = 0.0
    best_error = float("inf")
    grid = np.arange(-search_radius_deg, search_radius_deg + 0.5 * step_deg, step_deg)
    for solved_delta in grid:
        phi = fixed_value if fixed_torsion_name == "phi0_deg" else baseline["phi0_deg"]
        psi = fixed_value if fixed_torsion_name == "psi0_deg" else baseline["psi0_deg"]
        if solved_torsion_name == "phi0_deg":
            phi = solved_baseline + solved_delta
        if solved_torsion_name == "psi0_deg":
            psi = solved_baseline + solved_delta
        endpoint = reconstruct_endpoint(window, phi, psi, omega_fixed_deg)
        error = endpoint_closure_error(endpoint, window.end_ca)
        if error < best_error:
            best_error = error
            best_delta = float(solved_delta)
    return best_delta, best_error


def solve_one_torsion(
    window: ClosureWindow,
    fixed_torsion_name: str,
    fixed_delta_deg: float,
    solved_torsion_name: str,
    omega_fixed_deg: float,
    force_fallback: bool = False,
) -> tuple[float, float, str]:
    """Solve one remaining torsion using scipy when available, otherwise grid fallback."""
    if SCIPY_AVAILABLE and not force_fallback and minimize_scalar is not None:  # pragma: no cover
        baseline = window.baseline_torsions
        fixed_value = baseline[fixed_torsion_name] + fixed_delta_deg
        solved_baseline = baseline[solved_torsion_name]

        def objective(delta: float) -> float:
            phi = fixed_value if fixed_torsion_name == "phi0_deg" else baseline["phi0_deg"]
            psi = fixed_value if fixed_torsion_name == "psi0_deg" else baseline["psi0_deg"]
            if solved_torsion_name == "phi0_deg":
                phi = solved_baseline + delta
            if solved_torsion_name == "psi0_deg":
                psi = solved_baseline + delta
            return endpoint_closure_error(reconstruct_endpoint(window, phi, psi, omega_fixed_deg), window.end_ca)

        result = minimize_scalar(objective, bounds=(-30.0, 30.0), method="bounded")
        return float(result.x), float(result.fun), "scipy_minimize_scalar"
    delta, error = solve_one_torsion_grid(window, fixed_torsion_name, fixed_delta_deg, solved_torsion_name, omega_fixed_deg)
    return delta, error, "fallback_grid"


def two_unit_torsion_values(
    window: ClosureWindow,
    fixed_torsion_name: str,
    fixed_delta_deg: float,
    solved_torsion_1_name: str,
    solved_delta_1_deg: float,
    solved_torsion_2_name: str,
    solved_delta_2_deg: float,
    omega_fixed_deg: float,
) -> dict[str, float]:
    """Return torsion values for two-unit reconstruction."""
    values = {
        "phi0_deg": window.baseline_torsions["phi0_deg"],
        "psi0_deg": window.baseline_torsions["psi0_deg"],
        "omega0_deg": omega_fixed_deg,
        "phi1_deg": window.baseline_torsions["phi1_deg"],
        "psi1_deg": window.baseline_torsions["psi1_deg"],
        "omega1_deg": omega_fixed_deg,
    }
    values[fixed_torsion_name] = window.baseline_torsions[fixed_torsion_name] + fixed_delta_deg
    values[solved_torsion_1_name] = window.baseline_torsions[solved_torsion_1_name] + solved_delta_1_deg
    values[solved_torsion_2_name] = window.baseline_torsions[solved_torsion_2_name] + solved_delta_2_deg
    return values


def two_unit_endpoint_error(
    window: ClosureWindow,
    fixed_torsion_name: str,
    fixed_delta_deg: float,
    solved_torsion_1_name: str,
    solved_delta_1_deg: float,
    solved_torsion_2_name: str,
    solved_delta_2_deg: float,
    omega_fixed_deg: float,
) -> float:
    """Return two-unit endpoint closure error for candidate solved deltas."""
    if window.two_unit_end_ca is None:
        raise ValueError("Two-torsion solve requires a two-unit endpoint.")
    torsions = two_unit_torsion_values(
        window,
        fixed_torsion_name,
        fixed_delta_deg,
        solved_torsion_1_name,
        solved_delta_1_deg,
        solved_torsion_2_name,
        solved_delta_2_deg,
        omega_fixed_deg,
    )
    endpoint = reconstruct_two_unit_endpoint(
        window,
        torsions["phi0_deg"],
        torsions["psi0_deg"],
        torsions["omega0_deg"],
        torsions["phi1_deg"],
        torsions["psi1_deg"],
        torsions["omega1_deg"],
    )
    return endpoint_closure_error(endpoint, window.two_unit_end_ca)


def solve_two_torsions_grid(
    window: ClosureWindow,
    fixed_torsion_name: str,
    fixed_delta_deg: float,
    solved_torsion_1_name: str,
    solved_torsion_2_name: str,
    omega_fixed_deg: float,
    coarse_radius_deg: float = 30.0,
    coarse_step_deg: float = 2.0,
    refine_radius_deg: float = 2.0,
    refine_step_deg: float = 0.25,
) -> tuple[float, float, float]:
    """Solve two torsions by coarse grid followed by local refinement."""

    def evaluate_grid(center_1: float, center_2: float, radius: float, step: float) -> tuple[float, float, float]:
        best = (0.0, 0.0, float("inf"))
        values_1 = np.arange(center_1 - radius, center_1 + radius + 0.5 * step, step)
        values_2 = np.arange(center_2 - radius, center_2 + radius + 0.5 * step, step)
        for delta_1 in values_1:
            for delta_2 in values_2:
                error = two_unit_endpoint_error(
                    window,
                    fixed_torsion_name,
                    fixed_delta_deg,
                    solved_torsion_1_name,
                    float(delta_1),
                    solved_torsion_2_name,
                    float(delta_2),
                    omega_fixed_deg,
                )
                if error < best[2]:
                    best = (float(delta_1), float(delta_2), float(error))
        return best

    coarse_1, coarse_2, _ = evaluate_grid(0.0, 0.0, coarse_radius_deg, coarse_step_deg)
    return evaluate_grid(coarse_1, coarse_2, refine_radius_deg, refine_step_deg)


def solved_torsion_names_for_two_mode(fixed_torsion_name: str) -> tuple[str, str]:
    """Choose two phi/psi torsions to solve while one selected torsion is fixed."""
    choices = ["phi0_deg", "psi0_deg", "phi1_deg", "psi1_deg"]
    remaining = [name for name in choices if name != fixed_torsion_name]
    # Keep the final psi as baseline by default; solve the two earliest remaining torsions.
    preferred = [name for name in remaining if name != "psi1_deg"]
    if len(preferred) >= 2:
        return preferred[0], preferred[1]
    return remaining[0], remaining[1]


def classify_solver_result(endpoint_error_A: float, tolerance_A: float) -> bool:
    """Return closure success from endpoint error and tolerance."""
    return float(endpoint_error_A) <= float(tolerance_A)


def run_window_closure(
    window: ClosureWindow,
    fixed_torsion_name: str,
    perturbations: list[float],
    tolerance_A: float,
    omega_mode: str = "baseline",
    solve_mode: str = "one_torsion",
) -> list[dict[str, object]]:
    """Run closure attempts for one selected torsion in one window."""
    if fixed_torsion_name not in {"phi0_deg", "psi0_deg"}:
        raise ValueError("Prototype supports fixed torsion phi0_deg or psi0_deg.")
    solved_torsion_name = "psi0_deg" if fixed_torsion_name == "phi0_deg" else "phi0_deg"
    omega_fixed = 180.0 if omega_mode == "trans180" else window.baseline_torsions["omega0_deg"]
    rows = []
    for delta in perturbations:
        if solve_mode == "one_torsion":
            solved_delta, error, method = solve_one_torsion(
                window,
                fixed_torsion_name,
                delta,
                solved_torsion_name,
                omega_fixed,
            )
            solved_torsion_1_name = solved_torsion_name
            solved_torsion_1_delta = solved_delta
            solved_torsion_2_name = ""
            solved_torsion_2_delta = np.nan
        elif solve_mode == "two_torsion":
            solved_torsion_1_name, solved_torsion_2_name = solved_torsion_names_for_two_mode(fixed_torsion_name)
            solved_torsion_1_delta, solved_torsion_2_delta, error = solve_two_torsions_grid(
                window,
                fixed_torsion_name,
                delta,
                solved_torsion_1_name,
                solved_torsion_2_name,
                omega_fixed,
            )
            method = "fallback_2d_grid_coarse_refine"
        else:
            raise ValueError(f"Unknown solve_mode {solve_mode!r}.")
        rows.append(
            {
                "model_id": window.model_id,
                "chain_id": window.chain_id,
                "repeat_start_index": window.repeat_start_index,
                "residue_names": window.residue_names,
                "solve_mode": solve_mode,
                "fixed_torsion_name": fixed_torsion_name,
                "fixed_torsion_delta_deg": delta,
                "solved_torsion_1_name": solved_torsion_1_name,
                "solved_torsion_1_delta_deg": solved_torsion_1_delta,
                "solved_torsion_2_name": solved_torsion_2_name,
                "solved_torsion_2_delta_deg": solved_torsion_2_delta,
                "omega_fixed_deg": omega_fixed,
                "endpoint_error_A": error,
                "closure_success": classify_solver_result(error, tolerance_A),
                "notes": method,
            }
        )
    return rows


def find_representative_window(residues_by_chain: dict[str, list[Residue]], chain_id: str, residue_names: str) -> ClosureWindow:
    """Find first window matching residue_names with previous residue available."""
    residues = residues_by_chain[chain_id]
    target = residue_names.split("->")
    for idx in range(1, len(residues) - 1):
        if [residues[idx].resname, residues[idx + 1].resname] == target:
            return build_closure_window(residues, chain_id, idx)
    raise ValueError(f"No {residue_names} window with previous residue found on chain {chain_id}.")


def write_report(results: pd.DataFrame, path: Path) -> None:
    """Write prototype closure report."""
    summary = results.groupby(["solve_mode", "chain_id", "residue_names", "fixed_torsion_name", "notes"], as_index=False).agg(
        attempts=("closure_success", "size"),
        successes=("closure_success", "sum"),
        median_error_A=("endpoint_error_A", "median"),
        max_error_A=("endpoint_error_A", "max"),
    )
    mode_summary = results.groupby(["solve_mode"], as_index=False).agg(
        attempts=("closure_success", "size"),
        successes=("closure_success", "sum"),
        median_error_A=("endpoint_error_A", "median"),
        max_error_A=("endpoint_error_A", "max"),
    )
    total = len(results)
    successes = int(results["closure_success"].sum())
    method = ", ".join(sorted(results["notes"].unique()))
    text = f"""# Constrained Phi/Psi Closure Prototype

This is a small feasibility prototype, not the full systematic torsion search. It keeps CA anchors fixed, keeps omega fixed to the baseline trans-like value, perturbs one torsion, and compares one-solved-torsion versus two-solved-torsion closure using `{method}`.

## Summary

- Closure attempts: {total}
- Successful closures: {successes}
- Success fraction: {successes / total if total else 0:.3f}
- Closure tolerance: endpoint CA error <= values reported in the CSV setup
- SciPy available: {SCIPY_AVAILABLE}

By solve mode:

{markdown_table(mode_summary, ['solve_mode', 'attempts', 'successes', 'median_error_A', 'max_error_A'])}

By repeat:

{markdown_table(summary, ['solve_mode', 'chain_id', 'residue_names', 'fixed_torsion_name', 'notes', 'attempts', 'successes', 'median_error_A', 'max_error_A'])}

## Interpretation

- Does two-solved-torsion mode rescue GLU->MEP? Compare the `GLU->MEP` rows for `one_torsion` and `two_torsion`.
- Does CYP->GLU remain easy to close? Compare `CYP->GLU` successes across solve modes.
- What fraction of perturbations close under one-solve vs two-solve? See the solve-mode table above.
- Are endpoint errors low enough to justify a larger constrained torsion scan? Median and max errors are reported above.
- Which torsions are best treated as fixed versus solved? In the two-unit mode, the selected torsion is fixed, two remaining phi/psi torsions are solved, and the final psi remains at baseline.
- Are CYP->GLU and GLU->MEP repeats similarly feasible? Compare the per-window success rows above.
- Optimization availability: SciPy was {'available' if SCIPY_AVAILABLE else 'not available'}, so this run used {'SciPy bounded minimization' if SCIPY_AVAILABLE else 'deterministic fallback grid search'}.
- Next step: generate coordinates for accepted closures only after adding steric/geometry filters, anti-parallel chain direction metadata, and diffraction scoring hooks.
"""
    path.write_text(text, encoding="utf-8")


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    """Render a small markdown table."""
    if df.empty:
        return "_No rows._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in df[columns].itertuples(index=False):
        vals = [f"{v:.4g}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def run_prototype(
    parent_pdb: Path,
    metrics_dir: Path,
    reports_dir: Path,
    perturb_min: int,
    perturb_max: int,
    perturb_step: int,
    tolerance_A: float,
    fixed_torsion_name: str,
    omega_mode: str,
    solve_modes: list[str] | None = None,
) -> pd.DataFrame:
    """Run prototype closure on representative CYP->GLU and GLU->MEP windows."""
    residues_by_chain = parse_residues(parent_pdb)
    windows = [
        find_representative_window(residues_by_chain, "A", "CYP->GLU"),
        find_representative_window(residues_by_chain, "B", "GLU->MEP"),
    ]
    perturbations = perturbation_values(perturb_min, perturb_max, perturb_step)
    solve_modes = solve_modes or ["one_torsion", "two_torsion"]
    rows = []
    for window in windows:
        for solve_mode in solve_modes:
            rows.extend(
                run_window_closure(
                    window,
                    fixed_torsion_name,
                    perturbations,
                    tolerance_A,
                    omega_mode=omega_mode,
                    solve_mode=solve_mode,
                )
            )
    df = pd.DataFrame(rows)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(metrics_dir / "constrained_phi_psi_closure_prototype.csv", index=False)
    write_report(df, reports_dir / "constrained_phi_psi_closure_prototype.md")
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-pdb", type=Path, default=None)
    parser.add_argument("--metrics-dir", type=Path, default=Path("outputs/metrics"))
    parser.add_argument("--reports-dir", type=Path, default=Path("outputs/reports"))
    parser.add_argument("--perturb-min", type=int, default=-5)
    parser.add_argument("--perturb-max", type=int, default=5)
    parser.add_argument("--perturb-step", type=int, default=1)
    parser.add_argument("--closure-tolerance", type=float, default=0.05)
    parser.add_argument("--fixed-torsion", choices=["phi0_deg", "psi0_deg"], default="phi0_deg")
    parser.add_argument("--omega-mode", choices=["baseline", "trans180"], default="baseline")
    parser.add_argument("--solve-mode", choices=["one_torsion", "two_torsion", "both"], default="both")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    parent = args.parent_pdb or find_parent_pdb()
    df = run_prototype(
        parent,
        args.metrics_dir,
        args.reports_dir,
        args.perturb_min,
        args.perturb_max,
        args.perturb_step,
        args.closure_tolerance,
        args.fixed_torsion,
        args.omega_mode,
        solve_modes=["one_torsion", "two_torsion"] if args.solve_mode == "both" else [args.solve_mode],
    )
    print(f"Wrote {len(df)} closure attempts")
    print(f"Successes: {int(df['closure_success'].sum())}/{len(df)}")
    print(f"Metrics: {args.metrics_dir / 'constrained_phi_psi_closure_prototype.csv'}")
    print(f"Report: {args.reports_dir / 'constrained_phi_psi_closure_prototype.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
