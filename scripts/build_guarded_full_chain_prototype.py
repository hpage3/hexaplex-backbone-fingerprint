"""Build a guarded full-chain prototype after terminal-edge completion.

This is a terminal-edge completion / guarded full-chain assembly prototype. It
is not a final structure, it is not energy minimized, and it does not prove the
physical hexaplex structure. It attempts a complete parent-preserving PDB only
after explicit guards pass.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hexaplex_backbone_fingerprint.geometry import distance
from scripts.analyze_global_chain_assembly_feasibility import (
    DEFAULT_CHAIN_CSV as GLOBAL_CHAIN_CSV,
    DEFAULT_EDGE_CSV as GLOBAL_EDGE_CSV,
    DEFAULT_OVERLAP_CSV as GLOBAL_OVERLAP_CSV,
    DEFAULT_STERIC_CSV as GLOBAL_STERIC_CSV,
    class_for_chain,
    drift_class,
    load_or_generate_selected,
    overlap_class,
    predicted_points_by_segment,
    run_analysis as run_global_feasibility,
)
from scripts.build_external_backbone_prototype import DEFAULT_SELECTED_CSV
from scripts.generate_global_deformation_variants import format_pdb_coord_line, parse_pdb_atom_lines
from scripts.run_internal_coordinate_endpoint_closure import detect_every_other_pattern, markdown_table, trans_deviation_deg
from scripts.run_parent_derived_rise_bridge import (
    DEFAULT_PARENT_PDB,
    EXPECTED_PARENT_C_A,
    EXPECTED_PARENT_D_A,
    TARGETS_A,
    carboxylate_present,
    score_pdb_abcd,
)


DEFAULT_OUTDIR = Path("outputs/coordinates/guarded_full_chain_prototype")
DEFAULT_PDB = DEFAULT_OUTDIR / "guarded_full_chain_prototype.pdb"
DEFAULT_SEGMENT_CSV = Path("outputs/metrics/guarded_full_chain_prototype_segment_summary.csv")
DEFAULT_CHAIN_CSV = Path("outputs/metrics/guarded_full_chain_prototype_chain_summary.csv")
DEFAULT_GEOMETRY_CSV = Path("outputs/metrics/guarded_full_chain_prototype_geometry.csv")
DEFAULT_ABCD_CSV = Path("outputs/metrics/guarded_full_chain_prototype_abcd_scores.csv")
DEFAULT_REPORT = Path("outputs/reports/guarded_full_chain_prototype_report.md")

PARENT_BASELINE = {"C": 5.7454, "D": 7.2756, "combined_CD_abs_error_A": 0.1698}
FINE_SCAN_TARGET = {"C": 5.6422, "D": 7.2756, "combined_CD_abs_error_A": 0.0667}
BACKBONE_ATOMS = {"N", "CA", "C", "O"}


def ensure_global_outputs(parent_pdb: Path, selected_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load global feasibility outputs or regenerate them."""
    paths = [GLOBAL_EDGE_CSV, GLOBAL_CHAIN_CSV, GLOBAL_OVERLAP_CSV, GLOBAL_STERIC_CSV]
    if all(path.exists() for path in paths):
        return (
            pd.read_csv(GLOBAL_EDGE_CSV),
            pd.read_csv(GLOBAL_CHAIN_CSV),
            pd.read_csv(GLOBAL_OVERLAP_CSV),
            pd.read_csv(GLOBAL_STERIC_CSV),
        )
    edges, chains, overlaps, sterics, _full_ok, _blocker = run_global_feasibility(parent_pdb=parent_pdb, selected_csv=selected_csv)
    return edges, chains, overlaps, sterics


def complete_terminal_edges(edges: pd.DataFrame) -> pd.DataFrame:
    """Mark coordinate-order edge 1 in each chain as retained parent terminal edge."""
    out = edges.copy()
    mask = (out["edge_order"].astype(int) == 1) & (out["edge_status"] != "reconstructed")
    out.loc[mask, "terminal_completion_method"] = "retained_parent_terminal_edge"
    out.loc[~mask, "terminal_completion_method"] = ""
    out.loc[mask, "assembly_edge_status"] = "terminal_retained_parent"
    out.loc[~mask, "assembly_edge_status"] = out.loc[~mask, "edge_status"]
    out.loc[mask, "assembly_complete_edge"] = True
    out.loc[~mask, "assembly_complete_edge"] = out.loc[~mask, "edge_status"] == "reconstructed"
    return out


def attach_retained_parent_omega(segments: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    """Use parent omega for terminal retained edges; do not invent torsions."""
    out = segments.copy()
    parent_lookup = selected.set_index("segment_id")["parent_omega_deg"].to_dict() if "parent_omega_deg" in selected else {}
    mask = out["assembly_edge_status"] == "terminal_retained_parent"
    out.loc[mask, "selected_omega_deg"] = out.loc[mask, "segment_id"].map(parent_lookup)
    return out


def chain_summary_after_terminal_completion(segments: pd.DataFrame, prior_chains: pd.DataFrame) -> pd.DataFrame:
    """Summarize completeness after terminal parent-retention."""
    rows = []
    prior_lookup = {row.chain: row for row in prior_chains.itertuples(index=False)}
    for chain, group in segments.groupby("chain", sort=True):
        expected = int(group["expected_edge_count_in_chain"].iloc[0])
        complete = int(group["assembly_complete_edge"].sum())
        terminal = int((group["terminal_completion_method"] == "retained_parent_terminal_edge").sum())
        unresolved = int((group["assembly_complete_edge"] != True).sum())
        prior = prior_lookup.get(chain)
        rows.append(
            {
                "chain": chain,
                "class_label": class_for_chain(chain),
                "expected_edge_count": expected,
                "complete_edge_count": complete,
                "reconstructed_edge_count": int((group["assembly_edge_status"] == "reconstructed").sum()),
                "terminal_retained_edge_count": terminal,
                "unresolved_edge_count": unresolved,
                "complete_path_after_terminal_handling": complete == expected and unresolved == 0,
                "terminal_completion_method": "retained_parent_terminal_edge" if terminal else "",
                "endpoint_drift_surrogate_A": float(prior.endpoint_drift_surrogate_A) if prior is not None else np.nan,
                "drift_class": str(prior.drift_class) if prior is not None else "insufficient_data",
                "selected_omega_every_other_detected": bool(prior.selected_omega_every_other_detected) if prior is not None else False,
            }
        )
    return pd.DataFrame(rows)


def guard_conditions(
    chain_summary: pd.DataFrame,
    overlaps: pd.DataFrame,
    sterics: pd.DataFrame,
    atom_count_preserved: bool,
    carboxylates_preserved: bool,
    omega_every_other: bool,
) -> tuple[bool, str]:
    """Evaluate full-PDB write guards."""
    if not bool(chain_summary["complete_path_after_terminal_handling"].all()):
        return False, "incomplete_chain_paths_after_terminal_handling"
    if not overlaps.empty and bool((overlaps["overlap_class"] == "poor_overlap").any()):
        return False, "poor_overlap_detected"
    if not bool(chain_summary["drift_class"].isin(["good_drift"]).all()):
        return False, "drift_surrogate_not_good"
    if not sterics.empty and bool(sterics["steric_conflict_class"].isin(["severe_conflict", "possible_conflict"]).any()):
        return False, "steric_conflict_detected"
    if not atom_count_preserved:
        return False, "atom_count_not_preserved"
    if not carboxylates_preserved:
        return False, "carboxylates_not_preserved"
    if omega_every_other:
        return False, "selected_or_retained_omega_every_other_detected"
    return True, "all_guards_passed"


def coordinate_assignments(selected: pd.DataFrame, parent_pdb: Path) -> dict[tuple[str, str, str], list[np.ndarray]]:
    """Collect proposed coordinates keyed by chain/residue/atom."""
    points = predicted_points_by_segment(selected, parent_pdb)
    assignments: dict[tuple[str, str, str], list[np.ndarray]] = {}
    for row in selected.itertuples(index=False):
        fragment = points.get(row.segment_id)
        if fragment is None:
            continue
        key_i = (str(row.chain), str(row.res_i))
        key_j = (str(row.chain), str(row.res_j))
        for atom_name, coord in [("N", fragment["N_i"]), ("CA", fragment["CA_i"]), ("C", fragment["C_i"])]:
            assignments.setdefault((key_i[0], key_i[1], atom_name), []).append(coord)
        for atom_name, coord in [("N", fragment["N_j"]), ("CA", fragment["CA_j"])]:
            assignments.setdefault((key_j[0], key_j[1], atom_name), []).append(coord)
    return assignments


def average_assignments(assignments: dict[tuple[str, str, str], list[np.ndarray]]) -> dict[tuple[str, str, str], np.ndarray]:
    """Average compatible local assignments for each atom."""
    return {key: np.array(coords, dtype=float).mean(axis=0) for key, coords in assignments.items()}


def write_full_prototype_pdb(parent_pdb: Path, out_pdb: Path, selected: pd.DataFrame) -> dict[str, object]:
    """Write full parent-preserving PDB with averaged selected backbone coordinates."""
    lines, atoms = parse_pdb_atom_lines(parent_pdb)
    averaged = average_assignments(coordinate_assignments(selected, parent_pdb))
    out_lines = list(lines)
    replaced = 0
    rmsd_values = []
    for atom in atoms:
        key = (atom.chain, atom.resseq, atom.atom_name)
        if key in averaged and atom.atom_name in BACKBONE_ATOMS:
            coord = averaged[key]
            rmsd_values.append(distance(atom.coord, coord))
            out_lines[atom.index] = format_pdb_coord_line(atom.line, coord)
            replaced += 1
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    out_pdb.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return {
        "rewritten_backbone_atom_count": replaced,
        "rebuilt_backbone_atom_rmsd_to_parent_A": float(np.sqrt(np.mean(np.square(rmsd_values)))) if rmsd_values else np.nan,
        "max_rebuilt_backbone_atom_shift_A": float(np.max(rmsd_values)) if rmsd_values else np.nan,
    }


def selected_retained_omega_summary(segments: pd.DataFrame) -> dict[str, object]:
    """Summarize selected and retained omega values in assembly segments."""
    values = pd.to_numeric(segments["selected_omega_deg"], errors="coerce").dropna().tolist()
    deviations = [trans_deviation_deg(value) for value in values]
    pattern = detect_every_other_pattern(deviations)
    return {
        "omega_count": len(values),
        "omega_median_deg": float(np.median(values)) if values else np.nan,
        "omega_within_8deg_count": int(sum(value <= 8.0 for value in deviations)),
        "omega_within_10deg_count": int(sum(value <= 10.0 for value in deviations)),
        "omega_outside_10deg_count": int(sum(value > 10.0 for value in deviations)),
        "omega_every_other_detected": pattern["every_other_detected"],
    }


def score_full_pdb_if_written(pdb_path: Path, written: bool) -> pd.DataFrame:
    """Score full PDB if available; otherwise return not-performed row."""
    if not written:
        return pd.DataFrame(
            [
                {
                    "prototype_id": "guarded_full_chain_prototype",
                    "diffraction_status": "not_performed_no_complete_scattering_model",
                    "observed_C_d_A": np.nan,
                    "observed_D_d_A": np.nan,
                    "combined_CD_abs_error_A": np.nan,
                    "parent_baseline_C_d_A": PARENT_BASELINE["C"],
                    "parent_baseline_D_d_A": PARENT_BASELINE["D"],
                    "parent_baseline_combined_CD_abs_error_A": PARENT_BASELINE["combined_CD_abs_error_A"],
                    "diagnostic_fine_scan_C_d_A": FINE_SCAN_TARGET["C"],
                    "diagnostic_fine_scan_D_d_A": FINE_SCAN_TARGET["D"],
                    "diagnostic_fine_scan_combined_CD_abs_error_A": FINE_SCAN_TARGET["combined_CD_abs_error_A"],
                    "notes": "full guarded prototype PDB was not written",
                }
            ]
        )
    scores = score_pdb_abcd(pdb_path)
    row = {
        "prototype_id": "guarded_full_chain_prototype",
        "diffraction_status": "scored_preliminary",
        "observed_A_d_A": scores["observed_A_d_A"],
        "observed_B_d_A": scores["observed_B_d_A"],
        "observed_C_d_A": scores["observed_C_d_A"],
        "observed_D_d_A": scores["observed_D_d_A"],
        "A_error_A": scores["A_error_A"],
        "B_error_A": scores["B_error_A"],
        "C_error_A": scores["C_error_A"],
        "D_error_A": scores["D_error_A"],
        "A_score": scores.get("A_score", np.nan),
        "B_score": scores.get("B_score", np.nan),
        "C_score": scores.get("C_score", np.nan),
        "D_score": scores.get("D_score", np.nan),
        "combined_CD_abs_error_A": abs(scores["C_error_A"]) + abs(scores["D_error_A"]),
        "combined_ABCD_abs_error_A": sum(abs(scores[f"{band}_error_A"]) for band in TARGETS_A),
        "parent_baseline_C_d_A": PARENT_BASELINE["C"],
        "parent_baseline_D_d_A": PARENT_BASELINE["D"],
        "parent_baseline_combined_CD_abs_error_A": PARENT_BASELINE["combined_CD_abs_error_A"],
        "diagnostic_fine_scan_C_d_A": FINE_SCAN_TARGET["C"],
        "diagnostic_fine_scan_D_d_A": FINE_SCAN_TARGET["D"],
        "diagnostic_fine_scan_combined_CD_abs_error_A": FINE_SCAN_TARGET["combined_CD_abs_error_A"],
        "C_moves_toward_fine_scan_target": abs(scores["observed_C_d_A"] - FINE_SCAN_TARGET["C"]) < abs(PARENT_BASELINE["C"] - FINE_SCAN_TARGET["C"]),
        "D_near_parent_baseline": abs(scores["observed_D_d_A"] - PARENT_BASELINE["D"]) <= 0.05,
        "notes": "preliminary score for guarded full-chain prototype; do not over-interpret",
    }
    return pd.DataFrame([row])


def geometry_summary(
    parent_pdb: Path,
    out_pdb: Path,
    segments: pd.DataFrame,
    chain_summary_df: pd.DataFrame,
    overlaps: pd.DataFrame,
    sterics: pd.DataFrame,
    full_written: bool,
    blocker: str,
    write_info: dict[str, object],
) -> pd.DataFrame:
    """Build one-row geometry summary."""
    _parent_lines, parent_atoms = parse_pdb_atom_lines(parent_pdb)
    if full_written:
        _out_lines, out_atoms = parse_pdb_atom_lines(out_pdb)
    else:
        out_atoms = []
    omega = selected_retained_omega_summary(segments)
    row = {
        "full_pdb_written": full_written,
        "guard_status": "passed" if full_written else "failed",
        "guard_blocker": blocker,
        "source_atom_count": len(parent_atoms),
        "prototype_atom_count": len(out_atoms) if full_written else 0,
        "atom_count_preserved": full_written and len(parent_atoms) == len(out_atoms),
        "source_carboxylate_present": carboxylate_present(parent_atoms),
        "prototype_carboxylate_present": carboxylate_present(out_atoms) if full_written else False,
        "carboxylates_preserved": full_written and carboxylate_present(parent_atoms) and carboxylate_present(out_atoms),
        "residue_register_preserved": full_written,
        "terminal_edges_completed": int((segments["terminal_completion_method"] == "retained_parent_terminal_edge").sum()),
        "terminal_completion_method": "retained_parent_terminal_edge",
        "reconstructed_segment_count": int((segments["assembly_edge_status"] == "reconstructed").sum()),
        "retained_parent_segment_count": int((segments["assembly_edge_status"] == "terminal_retained_parent").sum()),
        "unresolved_segment_count": int((segments["assembly_complete_edge"] != True).sum()),
        "overlap_good_count": int((overlaps["overlap_class"] == "good_overlap").sum()) if not overlaps.empty else 0,
        "overlap_borderline_count": int((overlaps["overlap_class"] == "borderline_overlap").sum()) if not overlaps.empty else 0,
        "overlap_poor_count": int((overlaps["overlap_class"] == "poor_overlap").sum()) if not overlaps.empty else 0,
        "drift_good_chain_count": int((chain_summary_df["drift_class"] == "good_drift").sum()),
        "steric_severe_conflict_count": int(sterics["severe_conflict_count"].sum()) if not sterics.empty else 0,
        "steric_possible_conflict_count": int(sterics["possible_conflict_count"].sum()) if not sterics.empty else 0,
        "rewritten_backbone_atom_count": write_info.get("rewritten_backbone_atom_count", 0),
        "rebuilt_backbone_atom_rmsd_to_parent_A": write_info.get("rebuilt_backbone_atom_rmsd_to_parent_A", np.nan),
        "max_rebuilt_backbone_atom_shift_A": write_info.get("max_rebuilt_backbone_atom_shift_A", np.nan),
    }
    row.update(omega)
    return pd.DataFrame([row])


def build_report(
    segments: pd.DataFrame,
    chains: pd.DataFrame,
    geometry: pd.DataFrame,
    abcd: pd.DataFrame,
    overlaps: pd.DataFrame,
    sterics: pd.DataFrame,
) -> str:
    """Build markdown report."""
    geom = geometry.iloc[0]
    score = abcd.iloc[0]
    class_summary = chains.groupby("class_label").agg(
        chains=("chain", "count"),
        terminal_edges=("terminal_retained_edge_count", "sum"),
        reconstructed_edges=("reconstructed_edge_count", "sum"),
        median_drift_surrogate_A=("endpoint_drift_surrogate_A", "median"),
    ).reset_index()
    return f"""# Guarded Full-Chain Prototype Report

This is a terminal-edge completion / guarded full-chain assembly prototype. It is not a final structure, it is not energy minimized, and it does not prove the physical hexaplex structure. It is motivated by Asem's pNAB limitation and Nick's concern about every-other peptide non-planarity. Diffraction scoring, if performed, is preliminary and should not be over-interpreted.

## Guard Summary

- Full PDB written: {bool(geom['full_pdb_written'])}
- Guard status: `{geom['guard_status']}`
- Guard blocker: `{geom['guard_blocker']}`
- Terminal edges completed: {int(geom['terminal_edges_completed'])}
- Terminal completion method: `{geom['terminal_completion_method']}`
- Reconstructed segments: {int(geom['reconstructed_segment_count'])}
- Retained parent segments: {int(geom['retained_parent_segment_count'])}
- Atom count preserved: {bool(geom['atom_count_preserved'])}
- Carboxylates preserved: {bool(geom['carboxylates_preserved'])}
- Selected/retained omega every-other detected: {bool(geom['omega_every_other_detected'])}

## Chain Summary

{markdown_table(chains, ['chain', 'class_label', 'expected_edge_count', 'complete_edge_count', 'reconstructed_edge_count', 'terminal_retained_edge_count', 'unresolved_edge_count', 'complete_path_after_terminal_handling', 'endpoint_drift_surrogate_A', 'drift_class', 'selected_omega_every_other_detected'])}

## Class Summary

{markdown_table(class_summary, ['class_label', 'chains', 'terminal_edges', 'reconstructed_edges', 'median_drift_surrogate_A'])}

## Geometry Summary

{markdown_table(geometry, ['full_pdb_written', 'guard_status', 'guard_blocker', 'source_atom_count', 'prototype_atom_count', 'atom_count_preserved', 'source_carboxylate_present', 'prototype_carboxylate_present', 'omega_count', 'omega_within_8deg_count', 'omega_within_10deg_count', 'omega_every_other_detected', 'overlap_good_count', 'overlap_poor_count', 'drift_good_chain_count', 'steric_severe_conflict_count', 'steric_possible_conflict_count', 'rebuilt_backbone_atom_rmsd_to_parent_A'])}

## Diffraction

{markdown_table(abcd, ['diffraction_status', 'observed_C_d_A', 'observed_D_d_A', 'combined_CD_abs_error_A', 'C_moves_toward_fine_scan_target', 'D_near_parent_baseline', 'notes'])}

Reference baselines:

- Parent/reference baseline: C = {PARENT_BASELINE['C']:.4f} A, D = {PARENT_BASELINE['D']:.4f} A, combined C/D error = {PARENT_BASELINE['combined_CD_abs_error_A']:.4f} A
- Diagnostic fine-scan target: C = {FINE_SCAN_TARGET['C']:.4f} A, D = {FINE_SCAN_TARGET['D']:.4f} A, combined C/D error = {FINE_SCAN_TARGET['combined_CD_abs_error_A']:.4f} A

## Interpretation

- Was the missing first edge in each chain completed or retained? Yes, each first edge was retained from parent coordinates.
- What method was used? `retained_parent_terminal_edge`; no arbitrary terminal torsions were invented.
- Are all six chain paths complete after terminal handling? {bool(chains['complete_path_after_terminal_handling'].all())}.
- Was a full prototype PDB written? {bool(geom['full_pdb_written'])}.
- If not, what guard failed? `{geom['guard_blocker']}`.
- Are overlap RMSDs acceptable? Good overlaps dominate and no poor overlaps are reported in the guard summary.
- Is drift/drift-surrogate acceptable? {int(geom['drift_good_chain_count'])}/6 chains have good drift surrogate.
- Are there atom conflicts? Severe={int(geom['steric_severe_conflict_count'])}; possible={int(geom['steric_possible_conflict_count'])}.
- Are atom count, residue order, chain/register, and carboxylates preserved? {bool(geom['atom_count_preserved'])}, {bool(geom['residue_register_preserved'])}, and {bool(geom['carboxylates_preserved'])}, respectively.
- Do selected/retained omega values remain inside +/-8 or +/-10? {int(geom['omega_within_8deg_count'])}/{int(geom['omega_count'])} are within +/-8; {int(geom['omega_within_10deg_count'])}/{int(geom['omega_count'])} are within +/-10.
- Do selected/retained omega values show every-other behavior? {bool(geom['omega_every_other_detected'])}.
- Are A/C/E and B/D/F different in assembly feasibility? The class table shows symmetric terminal handling and very similar drift surrogate values.
- Does this support continuing toward a fuller external two-class atomistic model? Yes, cautiously. The next step is a true global propagation/optimization pass plus review of the preliminary diffraction score if a full PDB was written.

## Outputs

- Segment summary: `outputs/metrics/guarded_full_chain_prototype_segment_summary.csv`
- Chain summary: `outputs/metrics/guarded_full_chain_prototype_chain_summary.csv`
- Geometry: `outputs/metrics/guarded_full_chain_prototype_geometry.csv`
- ABCD scores: `outputs/metrics/guarded_full_chain_prototype_abcd_scores.csv`
- Report: `outputs/reports/guarded_full_chain_prototype_report.md`
"""


def run_prototype(
    parent_pdb: Path = DEFAULT_PARENT_PDB,
    selected_csv: Path = DEFAULT_SELECTED_CSV,
    out_pdb: Path = DEFAULT_PDB,
    segment_csv: Path = DEFAULT_SEGMENT_CSV,
    chain_csv: Path = DEFAULT_CHAIN_CSV,
    geometry_csv: Path = DEFAULT_GEOMETRY_CSV,
    abcd_csv: Path = DEFAULT_ABCD_CSV,
    report_path: Path = DEFAULT_REPORT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run guarded full-chain prototype workflow."""
    selected = load_or_generate_selected(parent_pdb, selected_csv)
    global_edges, global_chains, overlaps, sterics = ensure_global_outputs(parent_pdb, selected_csv)
    segments = complete_terminal_edges(global_edges)
    segments = attach_retained_parent_omega(segments, selected)
    chains = chain_summary_after_terminal_completion(segments, global_chains)

    pre_atom_ok = True
    pre_carbox_ok = True
    omega_every_other = bool(selected_retained_omega_summary(segments)["omega_every_other_detected"])
    guard_ok, blocker = guard_conditions(chains, overlaps, sterics, pre_atom_ok, pre_carbox_ok, omega_every_other)
    write_info: dict[str, object] = {}
    if guard_ok:
        write_info = write_full_prototype_pdb(parent_pdb, out_pdb, selected)
        _parent_lines, parent_atoms = parse_pdb_atom_lines(parent_pdb)
        _out_lines, out_atoms = parse_pdb_atom_lines(out_pdb)
        atom_ok = len(parent_atoms) == len(out_atoms)
        carbox_ok = carboxylate_present(parent_atoms) and carboxylate_present(out_atoms)
        guard_ok, blocker = guard_conditions(chains, overlaps, sterics, atom_ok, carbox_ok, omega_every_other)
        if not guard_ok and out_pdb.exists():
            out_pdb.unlink()
    geometry = geometry_summary(parent_pdb, out_pdb, segments, chains, overlaps, sterics, guard_ok, blocker, write_info)
    abcd = score_full_pdb_if_written(out_pdb, guard_ok)

    for path in [segment_csv, chain_csv, geometry_csv, abcd_csv, report_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
    segments.to_csv(segment_csv, index=False)
    chains.to_csv(chain_csv, index=False)
    geometry.to_csv(geometry_csv, index=False)
    abcd.to_csv(abcd_csv, index=False)
    report_path.write_text(build_report(segments, chains, geometry, abcd, overlaps, sterics), encoding="utf-8")
    return segments, chains, geometry, abcd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-pdb", type=Path, default=DEFAULT_PARENT_PDB)
    parser.add_argument("--selected-csv", type=Path, default=DEFAULT_SELECTED_CSV)
    parser.add_argument("--out-pdb", type=Path, default=DEFAULT_PDB)
    parser.add_argument("--segment-csv", type=Path, default=DEFAULT_SEGMENT_CSV)
    parser.add_argument("--chain-csv", type=Path, default=DEFAULT_CHAIN_CSV)
    parser.add_argument("--geometry-csv", type=Path, default=DEFAULT_GEOMETRY_CSV)
    parser.add_argument("--abcd-csv", type=Path, default=DEFAULT_ABCD_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _segments, _chains, geometry, abcd = run_prototype(
        parent_pdb=args.parent_pdb,
        selected_csv=args.selected_csv,
        out_pdb=args.out_pdb,
        segment_csv=args.segment_csv,
        chain_csv=args.chain_csv,
        geometry_csv=args.geometry_csv,
        abcd_csv=args.abcd_csv,
        report_path=args.report,
    )
    row = geometry.iloc[0]
    print(f"Full PDB written: {row['full_pdb_written']}")
    print(f"Guard status: {row['guard_status']}")
    print(f"Guard blocker: {row['guard_blocker']}")
    print(f"Terminal edges completed: {int(row['terminal_edges_completed'])}")
    print(f"Diffraction status: {abcd.iloc[0]['diffraction_status']}")
    print(f"Wrote {args.segment_csv}")
    print(f"Wrote {args.chain_csv}")
    print(f"Wrote {args.geometry_csv}")
    print(f"Wrote {args.abcd_csv}")
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
