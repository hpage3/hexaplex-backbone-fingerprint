"""Prototype fixed-omega GLU->MEP closure refinement under geometry gates."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_backbone_torsion_repeat import parse_residues
from scripts.audit_constrained_phi_psi_candidate_geometry import audit_candidate, parse_pdb
from scripts.audit_repeated_constrained_phi_psi_variant_geometry import (
    failure_reasons,
    safe_for_diffraction,
)
from scripts.generate_constrained_phi_psi_candidates import (
    atom_key_from_line,
    read_pdb_lines,
    reconstruct_candidate_points,
    update_pdb_coordinate_line,
)
from scripts.prototype_constrained_phi_psi_closure import (
    build_closure_window,
    solve_one_torsion,
    solve_two_torsions_grid,
)


DEFAULT_SOURCE_PDB = Path("outputs/coordinates/ideal_hexaflex_variants/ideal_hexaflex_backbone_plus_carboxylate.pdb")
DEFAULT_OUT_CSV = Path("outputs/metrics/glu_mep_fixed_omega_closure_refinement.csv")
DEFAULT_REPORT = Path("outputs/reports/glu_mep_fixed_omega_closure_refinement.md")
DEFAULT_FIGURE_BASE = Path("outputs/figures/glu_mep_fixed_omega_closure_refinement")
OMEGA_POLICY = "fixed_180"
FIXED_TORSION = "phi0_deg"
PERTURBATION_DELTAS = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
ENDPOINT_TOLERANCE_A = 0.05


def identify_glu_mep_windows(residues_by_chain: dict[str, list]) -> list[dict[str, object]]:
    """Identify GLU->MEP windows using per-chain coordinate order."""
    windows: list[dict[str, object]] = []
    for chain, residues in residues_by_chain.items():
        for index, (first, second) in enumerate(zip(residues, residues[1:])):
            if first.resname == "GLU" and second.resname == "MEP":
                windows.append(
                    {
                        "chain_id": chain,
                        "repeat_start_index": index,
                        "res_i": first.resseq,
                        "res_j": second.resseq,
                        "repeat_type": "GLU->MEP",
                    }
                )
    return windows


def classify_geometry(audit: dict[str, object]) -> tuple[bool, str]:
    """Return geometry-safe boolean and semicolon-delimited failure reason."""
    safe = safe_for_diffraction(audit)
    reasons = failure_reasons(audit)
    return safe, ";".join(reasons)


def deterministic_sort_attempts(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by delta then solve mode for deterministic reports."""
    mode_order = {"one_torsion": 0, "two_torsion": 1, "local_refine": 2}
    out = df.copy()
    out["fixed_torsion_delta_deg"] = pd.to_numeric(out["fixed_torsion_delta_deg"], errors="coerce")
    out["_mode_order"] = out["solve_mode"].map(mode_order).fillna(99)
    return out.sort_values(["fixed_torsion_delta_deg", "_mode_order", "attempt_id"]).drop(columns=["_mode_order"]).reset_index(drop=True)


def solved_torsions_for_mode(window, delta: float, solve_mode: str) -> tuple[str, float, str, float, float, str]:
    """Solve GLU->MEP torsions for one mode and delta."""
    if solve_mode == "one_torsion":
        solved_delta, endpoint_error, method = solve_one_torsion(
            window,
            FIXED_TORSION,
            delta,
            "psi0_deg",
            180.0,
            force_fallback=True,
        )
        return "psi0_deg", solved_delta, "", np.nan, endpoint_error, method
    if solve_mode == "two_torsion":
        solved_1, solved_2, endpoint_error = solve_two_torsions_grid(
            window,
            FIXED_TORSION,
            delta,
            "psi0_deg",
            "phi1_deg",
            180.0,
            coarse_radius_deg=20.0,
            coarse_step_deg=2.0,
            refine_radius_deg=2.0,
            refine_step_deg=0.25,
        )
        return "psi0_deg", solved_1, "phi1_deg", solved_2, endpoint_error, "fallback_2d_grid_coarse_refine"
    if solve_mode == "local_refine":
        solved_1, solved_2, endpoint_error = solve_two_torsions_grid(
            window,
            FIXED_TORSION,
            delta,
            "psi0_deg",
            "phi1_deg",
            180.0,
            coarse_radius_deg=12.0,
            coarse_step_deg=1.0,
            refine_radius_deg=1.0,
            refine_step_deg=0.1,
        )
        return "psi0_deg", solved_1, "phi1_deg", solved_2, endpoint_error, "fallback_local_refine_grid"
    raise ValueError(f"Unknown solve mode: {solve_mode}")


def candidate_output_lines(source_lines: list[str], updates: dict[tuple[str, int, str], np.ndarray]) -> list[str]:
    """Return source PDB lines with local coordinate updates applied."""
    output = []
    for line in source_lines:
        key = atom_key_from_line(line)
        if key in updates:
            output.append(update_pdb_coordinate_line(line, updates[key]))
        else:
            output.append(line)
    if not output or output[-1] != "END":
        output.append("END")
    return output


def audit_attempt_geometry(parent_atoms, source_lines: list[str], window, row: pd.Series) -> dict[str, object]:
    """Reconstruct one attempted solution and audit it with existing geometry gates."""
    updates = reconstruct_candidate_points(window, row)
    output_lines = candidate_output_lines(source_lines, updates)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "attempt.pdb"
        tmp_path.write_text("\n".join(output_lines) + "\n", encoding="ascii")
        return audit_candidate(parent_atoms, tmp_path)


def attempt_row(
    attempt_id: str,
    window,
    delta: float,
    solve_mode: str,
    solved_1_name: str,
    solved_1_delta: float,
    solved_2_name: str,
    solved_2_delta: float,
    endpoint_error: float,
    geometry_audit: dict[str, object],
    method: str,
) -> dict[str, object]:
    """Build one output row."""
    closure_success = endpoint_error <= ENDPOINT_TOLERANCE_A
    geometry_safe, failure_reason = classify_geometry(geometry_audit)
    if not closure_success:
        failure_reason = ";".join(filter(None, ["endpoint_closure_failed", failure_reason]))
    return {
        "attempt_id": attempt_id,
        "repeat_type": window.residue_names,
        "chain_id": window.chain_id,
        "repeat_start_index": window.repeat_start_index,
        "fixed_torsion_name": FIXED_TORSION,
        "fixed_torsion_delta_deg": delta,
        "solve_mode": solve_mode,
        "solved_torsion_1_name": solved_1_name,
        "solved_torsion_1_delta_deg": solved_1_delta,
        "solved_torsion_2_name": solved_2_name,
        "solved_torsion_2_delta_deg": solved_2_delta,
        "omega_policy": OMEGA_POLICY,
        "endpoint_error_A": endpoint_error,
        "max_ca_anchor_shift_A": geometry_audit["max_ca_shift_A"],
        "max_backbone_bond_delta_A": geometry_audit["max_backbone_bond_delta_A"],
        "max_backbone_angle_delta_deg": geometry_audit["max_backbone_angle_delta_deg"],
        "max_omega_trans_deviation_deg": geometry_audit["max_omega_trans_deviation_deg"],
        "closure_success": closure_success,
        "geometry_safe": geometry_safe,
        "failure_reason": failure_reason,
        "notes": method,
    }


def representative_glu_mep_window(residues_by_chain: dict[str, list], windows: list[dict[str, object]]):
    """Return the first GLU->MEP window with previous and next residues available."""
    for info in windows:
        chain = str(info["chain_id"])
        index = int(info["repeat_start_index"])
        if index > 0 and index + 2 < len(residues_by_chain[chain]):
            return build_closure_window(residues_by_chain[chain], chain, index)
    raise ValueError("No GLU->MEP window has previous and next residues for closure refinement.")


def run_refinement(source_pdb: Path, out_csv: Path, report_path: Path, figure_base: Path) -> pd.DataFrame:
    """Run the GLU->MEP fixed-omega closure refinement prototype."""
    residues_by_chain = parse_residues(source_pdb)
    windows = identify_glu_mep_windows(residues_by_chain)
    chains = sorted({str(window["chain_id"]) for window in windows})
    window = representative_glu_mep_window(residues_by_chain, windows)
    source_lines = read_pdb_lines(source_pdb)
    parent_atoms = parse_pdb(source_pdb)
    rows = []
    ordinal = 1
    for delta in PERTURBATION_DELTAS:
        for solve_mode in ["one_torsion", "two_torsion", "local_refine"]:
            s1_name, s1_delta, s2_name, s2_delta, endpoint_error, method = solved_torsions_for_mode(window, delta, solve_mode)
            series = pd.Series(
                {
                    "chain_id": window.chain_id,
                    "repeat_start_index": window.repeat_start_index,
                    "residue_names": window.residue_names,
                    "solve_mode": "two_torsion" if solve_mode in {"two_torsion", "local_refine"} else "one_torsion",
                    "fixed_torsion_name": FIXED_TORSION,
                    "fixed_torsion_delta_deg": delta,
                    "solved_torsion_1_name": s1_name,
                    "solved_torsion_1_delta_deg": s1_delta,
                    "solved_torsion_2_name": s2_name,
                    "solved_torsion_2_delta_deg": s2_delta,
                }
            )
            audit = audit_attempt_geometry(parent_atoms, source_lines, window, series)
            rows.append(
                attempt_row(
                    f"glu_mep_refine_{ordinal:03d}",
                    window,
                    delta,
                    solve_mode,
                    s1_name,
                    s1_delta,
                    s2_name,
                    s2_delta,
                    endpoint_error,
                    audit,
                    method,
                )
            )
            ordinal += 1
    results = deterministic_sort_attempts(pd.DataFrame(rows))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    figure_base.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_csv, index=False)
    write_report(results, windows, chains, report_path)
    save_plot(results, figure_base)
    return results


def write_report(results: pd.DataFrame, windows: list[dict[str, object]], chains: list[str], path: Path) -> None:
    """Write closure-refinement markdown report."""
    safe = results[results["geometry_safe"].astype(bool)]
    nonzero_safe = safe[pd.to_numeric(safe["fixed_torsion_delta_deg"], errors="coerce") != 0]
    mode_summary = results.groupby("solve_mode", as_index=False).agg(
        attempts=("attempt_id", "size"),
        endpoint_successes=("closure_success", "sum"),
        geometry_safe=("geometry_safe", "sum"),
        median_endpoint_error_A=("endpoint_error_A", "median"),
    )
    failures = results.assign(
        failure_reason=results["failure_reason"].replace("", "none")
    ).groupby(["solve_mode", "failure_reason"], as_index=False).size()
    decision = (
        "fixed-omega repeated GLU->MEP variant generation is possible for the safe nonzero deltas."
        if not nonzero_safe.empty
        else "no nonzero strict fixed-omega GLU->MEP solution passed; defer omega sensitivity or test baseline-omega mode next."
    )
    text = f"""# GLU->MEP Fixed-Omega Closure Refinement

This is a targeted prototype/audit only. It does not generate a large model set and does not score diffraction. Omega is held to Nick's current `fixed_180` policy throughout.

- GLU->MEP windows found: {len(windows)}
- Chains containing GLU->MEP windows: {', '.join(chains)}
- Representative window: chain `{results.iloc[0]['chain_id']}`, coordinate-order index `{results.iloc[0]['repeat_start_index']}`
- Solve modes attempted: one_torsion, two_torsion, local_refine
- Geometry-safe attempts: {int(results['geometry_safe'].sum())}/{len(results)}
- Nonzero geometry-safe attempts: {len(nonzero_safe)}

## Solve-Mode Summary

{markdown_table(mode_summary)}

## Geometry-Safe Attempts

{markdown_table(safe[['attempt_id', 'solve_mode', 'fixed_torsion_delta_deg', 'endpoint_error_A', 'max_backbone_bond_delta_A', 'max_backbone_angle_delta_deg', 'max_omega_trans_deviation_deg']])}

## Failure Reasons

{markdown_table(failures)}

## Interpretation

- Were any nonzero GLU->MEP perturbations geometry-safe under fixed omega? {'yes' if not nonzero_safe.empty else 'no'}.
- Endpoint closure alone is not enough; attempts must pass bond, angle, C-alpha, label, and omega gates.
- Failure drivers are summarized above by solve mode and threshold reason.
- Does GLU->MEP look feasible under strict fixed-omega constraints? {'Possibly, but only for the listed safe nonzero attempts.' if not nonzero_safe.empty else 'Not from this strict refinement prototype.'}
- Recommended next branch: {decision}

Omega sensitivity remains deferred until after fixed-omega options are exhausted.
"""
    path.write_text(text, encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> str:
    """Render dataframe as markdown."""
    if df.empty:
        return "_No rows._"
    columns = list(df.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for record in df.itertuples(index=False):
        values = [f"{value:.6g}" if isinstance(value, float) else str(value) for value in record]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def save_plot(results: pd.DataFrame, figure_base: Path) -> None:
    """Save small diagnostic endpoint/bond plot."""
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    markers = {"one_torsion": "o", "two_torsion": "s", "local_refine": "^"}
    for mode, group in results.groupby("solve_mode"):
        group = group.sort_values("fixed_torsion_delta_deg")
        ax.plot(group["fixed_torsion_delta_deg"], group["endpoint_error_A"], marker=markers.get(mode, "o"), label=f"{mode} endpoint")
    safe = results[results["geometry_safe"].astype(bool)]
    if not safe.empty:
        ax.scatter(safe["fixed_torsion_delta_deg"], safe["endpoint_error_A"], s=80, facecolors="none", edgecolors="green", linewidths=1.5, label="geometry safe")
    ax.axhline(ENDPOINT_TOLERANCE_A, color="0.5", ls="--", lw=1, label="endpoint tolerance")
    ax.set_xlabel("fixed phi0 delta (deg)")
    ax.set_ylabel("endpoint error (A)")
    ax.set_title("GLU->MEP fixed-omega closure refinement")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_base.with_suffix(".png"), dpi=180)
    fig.savefig(figure_base.with_suffix(".svg"))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pdb", type=Path, default=DEFAULT_SOURCE_PDB)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--figure-base", type=Path, default=DEFAULT_FIGURE_BASE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = run_refinement(args.source_pdb, args.out_csv, args.report, args.figure_base)
    print(f"Wrote {len(results)} GLU->MEP closure-refinement attempts")
    print(f"Geometry-safe attempts: {int(results['geometry_safe'].sum())}/{len(results)}")
    print(f"CSV: {args.out_csv}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
