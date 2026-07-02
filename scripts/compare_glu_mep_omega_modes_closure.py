"""Compare GLU->MEP closure under fixed-180 versus parent-baseline omega modes."""

from __future__ import annotations

import argparse
import sys
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
from scripts.prototype_constrained_phi_psi_closure import solve_one_torsion, solve_two_torsions_grid
from scripts.prototype_glu_mep_fixed_omega_closure_refinement import (
    ENDPOINT_TOLERANCE_A,
    FIXED_TORSION,
    PERTURBATION_DELTAS,
    audit_attempt_geometry,
    classify_geometry,
    deterministic_sort_attempts,
    identify_glu_mep_windows,
    representative_glu_mep_window,
)
from scripts.audit_constrained_phi_psi_candidate_geometry import parse_pdb, trans_deviation_deg
from scripts.generate_constrained_phi_psi_candidates import read_pdb_lines


DEFAULT_SOURCE_PDB = Path("outputs/coordinates/ideal_hexaflex_variants/ideal_hexaflex_backbone_plus_carboxylate.pdb")
DEFAULT_OUT_CSV = Path("outputs/metrics/glu_mep_omega_mode_closure_comparison.csv")
DEFAULT_REPORT = Path("outputs/reports/glu_mep_omega_mode_closure_comparison.md")
DEFAULT_FIGURE_BASE = Path("outputs/figures/glu_mep_omega_mode_closure_comparison")
OMEGA_MODES = ("fixed_180", "baseline_parent")


def baseline_omega_values(window) -> dict[str, float]:
    """Return parent omega values for the representative two-unit window."""
    values = {"omega0_deg": float(window.baseline_torsions["omega0_deg"])}
    if "omega1_deg" in window.baseline_torsions:
        values["omega1_deg"] = float(window.baseline_torsions["omega1_deg"])
    return values


def omega_values_for_mode(window, omega_mode: str) -> tuple[float, float]:
    """Return omega0/omega1 values for a comparison mode."""
    if omega_mode == "fixed_180":
        return 180.0, 180.0
    if omega_mode == "baseline_parent":
        omega0 = float(window.baseline_torsions["omega0_deg"])
        omega1 = float(window.baseline_torsions.get("omega1_deg", omega0))
        return omega0, omega1
    raise ValueError(f"Unknown omega mode: {omega_mode}")


def solve_for_mode(window, delta: float, solve_mode: str, omega_mode: str) -> tuple[str, float, str, float, float, str]:
    """Solve one GLU->MEP attempt under one omega mode."""
    omega0, omega1 = omega_values_for_mode(window, omega_mode)
    if solve_mode == "one_torsion":
        solved_delta, endpoint_error, method = solve_one_torsion(
            window,
            FIXED_TORSION,
            delta,
            "psi0_deg",
            omega0,
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
            omega0,
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
            omega0,
            coarse_radius_deg=12.0,
            coarse_step_deg=1.0,
            refine_radius_deg=1.0,
            refine_step_deg=0.1,
        )
        return "psi0_deg", solved_1, "phi1_deg", solved_2, endpoint_error, "fallback_local_refine_grid"
    raise ValueError(f"Unknown solve mode: {solve_mode}")


def attempt_series(window, delta: float, solve_mode: str, omega_mode: str, s1_name: str, s1_delta: float, s2_name: str, s2_delta: float) -> pd.Series:
    """Build reconstruction metadata for the existing coordinate updater."""
    return pd.Series(
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
            "omega_mode": omega_mode,
        }
    )


def comparison_attempt_row(
    attempt_id: str,
    omega_mode: str,
    window,
    delta: float,
    solve_mode: str,
    s1_name: str,
    s1_delta: float,
    s2_name: str,
    s2_delta: float,
    endpoint_error: float,
    geometry_audit: dict[str, object],
    method: str,
) -> dict[str, object]:
    """Build one output row for the omega-mode comparison."""
    closure_success = endpoint_error <= ENDPOINT_TOLERANCE_A
    geometry_safe, failure_reason = classify_geometry(geometry_audit)
    if not closure_success:
        failure_reason = ";".join(filter(None, ["endpoint_closure_failed", failure_reason]))
    omega0, omega1 = omega_values_for_mode(window, omega_mode)
    return {
        "attempt_id": attempt_id,
        "repeat_type": window.residue_names,
        "chain_id": window.chain_id,
        "repeat_start_index": window.repeat_start_index,
        "omega_mode": omega_mode,
        "omega0_used_deg": omega0,
        "omega1_used_deg": omega1,
        "omega0_trans_deviation_deg": trans_deviation_deg(omega0),
        "omega1_trans_deviation_deg": trans_deviation_deg(omega1),
        "fixed_torsion_name": FIXED_TORSION,
        "fixed_torsion_delta_deg": delta,
        "solve_mode": solve_mode,
        "solved_torsion_1_name": s1_name,
        "solved_torsion_1_delta_deg": s1_delta,
        "solved_torsion_2_name": s2_name,
        "solved_torsion_2_delta_deg": s2_delta,
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


def mode_comparison_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize endpoint and geometry success by omega mode."""
    return results.groupby("omega_mode", as_index=False).agg(
        attempts=("attempt_id", "size"),
        endpoint_successes=("closure_success", "sum"),
        geometry_safe=("geometry_safe", "sum"),
        nonzero_geometry_safe=(
            "fixed_torsion_delta_deg",
            lambda s: int(((pd.to_numeric(s, errors="coerce") != 0) & results.loc[s.index, "geometry_safe"].astype(bool)).sum()),
        ),
        median_endpoint_error_A=("endpoint_error_A", "median"),
    )


def run_comparison(source_pdb: Path, out_csv: Path, report_path: Path, figure_base: Path) -> pd.DataFrame:
    """Run GLU->MEP closure comparison for fixed_180 and baseline_parent omega modes."""
    residues_by_chain = parse_residues(source_pdb)
    windows = identify_glu_mep_windows(residues_by_chain)
    chains = sorted({str(window["chain_id"]) for window in windows})
    window = representative_glu_mep_window(residues_by_chain, windows)
    source_lines = read_pdb_lines(source_pdb)
    parent_atoms = parse_pdb(source_pdb)
    rows = []
    ordinal = 1
    for omega_mode in OMEGA_MODES:
        for delta in PERTURBATION_DELTAS:
            for solve_mode in ["one_torsion", "two_torsion", "local_refine"]:
                s1_name, s1_delta, s2_name, s2_delta, endpoint_error, method = solve_for_mode(window, delta, solve_mode, omega_mode)
                series = attempt_series(window, delta, solve_mode, omega_mode, s1_name, s1_delta, s2_name, s2_delta)
                audit = audit_attempt_geometry(parent_atoms, source_lines, window, series)
                rows.append(
                    comparison_attempt_row(
                        f"glu_mep_omega_mode_{ordinal:03d}",
                        omega_mode,
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
    results = results.sort_values(["omega_mode", "fixed_torsion_delta_deg", "solve_mode"]).reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    figure_base.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_csv, index=False)
    write_report(results, windows, chains, window, report_path)
    save_plot(results, figure_base)
    return results


def write_report(results: pd.DataFrame, windows: list[dict[str, object]], chains: list[str], window, path: Path) -> None:
    """Write omega-mode comparison report."""
    summary = mode_comparison_summary(results)
    baseline = baseline_omega_values(window)
    fixed_safe = int(results[(results["omega_mode"] == "fixed_180") & results["geometry_safe"].astype(bool)].shape[0])
    baseline_safe = int(results[(results["omega_mode"] == "baseline_parent") & results["geometry_safe"].astype(bool)].shape[0])
    baseline_nonzero_safe = results[
        (results["omega_mode"] == "baseline_parent")
        & results["geometry_safe"].astype(bool)
        & (pd.to_numeric(results["fixed_torsion_delta_deg"], errors="coerce") != 0)
    ]
    failures = results.assign(failure_reason=results["failure_reason"].replace("", "none")).groupby(
        ["omega_mode", "solve_mode", "failure_reason"], as_index=False
    ).size()
    decision = (
        "repeated GLU->MEP baseline-omega variant generation is justified for the safe nonzero attempts."
        if not baseline_nonzero_safe.empty
        else "GLU->MEP remains too constrained here; broaden search to another torsion/repeat family before a large scan."
    )
    omega_lines = "\n".join(
        f"- {name}: {value:.4f} deg; trans deviation {trans_deviation_deg(value):.4f} deg"
        for name, value in baseline.items()
    )
    text = f"""# GLU->MEP Omega-Mode Closure Comparison

This prototype compares GLU->MEP closure feasibility under `fixed_180` and `baseline_parent` omega modes. It does not generate a large model set and does not score diffraction.

- GLU->MEP windows found: {len(windows)}
- Chains containing GLU->MEP windows: {', '.join(chains)}
- Representative window: chain `{window.chain_id}`, coordinate-order index `{window.repeat_start_index}`

## Parent Baseline Omega

{omega_lines}

## Mode Summary

{markdown_table(summary)}

## Geometry-Safe Attempts

{markdown_table(results[results['geometry_safe'].astype(bool)][['omega_mode', 'solve_mode', 'fixed_torsion_delta_deg', 'endpoint_error_A', 'max_backbone_bond_delta_A', 'max_backbone_angle_delta_deg', 'max_omega_trans_deviation_deg']])}

## Failure Reasons

{markdown_table(failures)}

## Interpretation

- Geometry-safe attempts under fixed_180: {fixed_safe}
- Geometry-safe attempts under baseline_parent: {baseline_safe}
- Did baseline_parent rescue any nonzero GLU->MEP perturbations? {'yes' if not baseline_nonzero_safe.empty else 'no'}.
- If baseline_parent produces nonzero safe attempts where fixed_180 does not, fixed-180 may be too restrictive for this repeat.
- If baseline_parent also fails, GLU->MEP is constrained even when retaining parent omega geometry.
- Recommended next branch: {decision}
"""
    path.write_text(text, encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> str:
    """Render dataframe as markdown."""
    if df.empty:
        return "_No rows._"
    columns = list(df.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for record in df.itertuples(index=False):
        vals = [f"{value:.6g}" if isinstance(value, float) else str(value) for value in record]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def save_plot(results: pd.DataFrame, figure_base: Path) -> None:
    """Save comparison plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"fixed_180": "#1f77b4", "baseline_parent": "#ff7f0e"}
    for (omega_mode, solve_mode), group in results.groupby(["omega_mode", "solve_mode"]):
        group = group.sort_values("fixed_torsion_delta_deg")
        ax.plot(
            group["fixed_torsion_delta_deg"],
            group["max_backbone_angle_delta_deg"],
            marker="o",
            color=colors.get(omega_mode, "0.3"),
            alpha=0.55 if solve_mode != "one_torsion" else 1.0,
            label=f"{omega_mode} {solve_mode}",
        )
    safe = results[results["geometry_safe"].astype(bool)]
    if not safe.empty:
        ax.scatter(safe["fixed_torsion_delta_deg"], safe["max_backbone_angle_delta_deg"], s=90, facecolors="none", edgecolors="green", linewidths=1.6, label="geometry safe")
    ax.axhline(5.0, color="0.5", ls="--", lw=1, label="angle threshold")
    ax.set_xlabel("fixed phi0 delta (deg)")
    ax.set_ylabel("max backbone angle delta (deg)")
    ax.set_title("GLU->MEP omega-mode closure comparison")
    ax.legend(fontsize=7, ncol=2)
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
    results = run_comparison(args.source_pdb, args.out_csv, args.report, args.figure_base)
    summary = mode_comparison_summary(results)
    print(f"Wrote {len(results)} GLU->MEP omega-mode closure attempts")
    for row in summary.itertuples(index=False):
        print(f"{row.omega_mode}: geometry-safe {row.geometry_safe}/{row.attempts}")
    print(f"CSV: {args.out_csv}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
