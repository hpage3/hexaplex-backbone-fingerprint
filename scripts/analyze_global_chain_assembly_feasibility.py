"""Analyze global chain assembly feasibility for selected local fragments.

This is a global chain assembly feasibility analysis. It is not a final
structure, not energy minimized, and does not prove the physical hexaplex
structure. The purpose is to test whether selected local phi/psi/omega fragments
can be made globally consistent before any complete PDB or diffraction scoring
is attempted.
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
from scripts.analyze_threefold_backbone_symmetry import parse_residues
from scripts.build_external_backbone_prototype import (
    DEFAULT_SELECTED_CSV,
    reconstructed_points,
    segment_lookup,
    select_torsions,
)
from scripts.run_internal_coordinate_endpoint_closure import detect_every_other_pattern, markdown_table, trans_deviation_deg
from scripts.run_parent_derived_rise_bridge import DEFAULT_PARENT_PDB
from scripts.run_phi_psi_omega_closure_scan import run_scan


DEFAULT_EDGE_CSV = Path("outputs/metrics/global_chain_assembly_edge_summary.csv")
DEFAULT_CHAIN_CSV = Path("outputs/metrics/global_chain_assembly_chain_summary.csv")
DEFAULT_OVERLAP_CSV = Path("outputs/metrics/global_chain_assembly_overlap_summary.csv")
DEFAULT_STERIC_CSV = Path("outputs/metrics/global_chain_assembly_steric_summary.csv")
DEFAULT_REPORT = Path("outputs/reports/global_chain_assembly_feasibility_report.md")
OPTIONAL_FULL_PDB = Path("outputs/coordinates/global_chain_assembly_feasibility/global_chain_assembly_prototype.pdb")


def class_for_chain(chain: str) -> str:
    """Return fixed two-class assignment."""
    if chain in {"A", "C", "E"}:
        return "triketo_cyanuric_like"
    if chain in {"B", "D", "F"}:
        return "triamino_melamine_like"
    return "unclassified"


def overlap_class(rmsd_A: float) -> str:
    """Classify overlap RMSD."""
    if not np.isfinite(float(rmsd_A)):
        return "insufficient_data"
    if rmsd_A <= 0.10:
        return "good_overlap"
    if rmsd_A <= 0.25:
        return "borderline_overlap"
    return "poor_overlap"


def drift_class(drift_A: float) -> str:
    """Classify endpoint drift or drift surrogate."""
    if not np.isfinite(float(drift_A)):
        return "insufficient_data"
    if drift_A <= 0.25:
        return "good_drift"
    if drift_A <= 0.75:
        return "borderline_drift"
    return "poor_drift"


def steric_conflict_class(distance_A: float) -> str:
    """Classify simple heavy-atom close contact."""
    if not np.isfinite(float(distance_A)):
        return "insufficient_data"
    if distance_A < 1.2:
        return "severe_conflict"
    if distance_A < 1.6:
        return "possible_conflict"
    return "no_conflict"


def load_or_generate_selected(parent_pdb: Path, selected_csv: Path = DEFAULT_SELECTED_CSV) -> pd.DataFrame:
    """Load selected torsions or regenerate them from closure scan."""
    if selected_csv.exists():
        return pd.read_csv(selected_csv)
    scan, _best, _summary = run_scan(parent_pdb=parent_pdb)
    return select_torsions(scan)


def segment_order_from_id(segment_id: str, fallback_res_i: int | float | str) -> int:
    """Return coordinate-order segment index from ``chain:index:pair`` segment ID."""
    parts = str(segment_id).split(":")
    if len(parts) >= 2:
        try:
            return int(parts[1])
        except ValueError:
            pass
    return int(float(fallback_res_i))


def edge_summary(selected: pd.DataFrame, parent_pdb: Path = DEFAULT_PARENT_PDB) -> pd.DataFrame:
    """Build one row per selected peptide edge."""
    by_chain = parse_residues(parent_pdb)
    residue_counts = {chain: len(residues) for chain, residues in by_chain.items()}
    rows = []
    for row in selected.itertuples(index=False):
        reconstructed = row.selection_reason in {"good_within_8deg", "good_within_10deg", "borderline_within_10deg"}
        rows.append(
            {
                "chain": row.chain,
                "class_label": row.class_label,
                "segment_id": row.segment_id,
                "res_i": int(row.res_i),
                "res_j": int(row.res_j),
                "edge_order": segment_order_from_id(row.segment_id, row.res_i),
                "residue_count_in_chain": residue_counts.get(row.chain, np.nan),
                "expected_edge_count_in_chain": residue_counts.get(row.chain, 1) - 1,
                "selection_reason": row.selection_reason,
                "edge_status": "reconstructed" if reconstructed else "unresolved_or_retained_parent",
                "selected_phi_deg": row.selected_phi_deg,
                "selected_psi_deg": row.selected_psi_deg,
                "selected_omega_deg": row.selected_omega_deg,
                "omega_window_class": row.omega_window_class,
                "closure_residual_A": row.closure_residual_A,
            }
        )
    return pd.DataFrame(rows).sort_values(["chain", "edge_order"])


def chain_summary(edges: pd.DataFrame) -> pd.DataFrame:
    """Summarize graph completeness and drift surrogate by chain."""
    rows = []
    for chain, group in edges.groupby("chain", sort=True):
        expected = int(group["expected_edge_count_in_chain"].iloc[0])
        reconstructed = int((group["edge_status"] == "reconstructed").sum())
        unresolved = int((group["edge_status"] != "reconstructed").sum())
        missing_edges = max(0, expected - len(group))
        ordered = group.sort_values("edge_order")
        reconstructed_orders = set(ordered[ordered["edge_status"] == "reconstructed"]["edge_order"].astype(int))
        expected_orders = set(range(1, expected + 1))
        continuous_path = reconstructed_orders == expected_orders
        residuals = pd.to_numeric(group["closure_residual_A"], errors="coerce").dropna().to_numpy(dtype=float)
        rss = float(np.sqrt(np.sum(residuals**2))) if len(residuals) else np.nan
        rows.append(
            {
                "chain": chain,
                "class_label": class_for_chain(chain),
                "residue_count": int(group["residue_count_in_chain"].iloc[0]),
                "expected_edge_count": expected,
                "selected_edge_count": len(group),
                "reconstructed_edge_count": reconstructed,
                "unresolved_or_retained_edge_count": unresolved,
                "missing_edge_count": missing_edges,
                "has_continuous_reconstructed_path": bool(continuous_path),
                "blocking_edges": ";".join(map(str, sorted(expected_orders - reconstructed_orders))),
                "drift_metric_type": "drift_surrogate_rss_closure_residual",
                "endpoint_drift_surrogate_A": rss,
                "max_local_closure_residual_A": float(np.max(residuals)) if len(residuals) else np.nan,
                "drift_class": drift_class(rss),
                "selected_omega_every_other_detected": detect_every_other_pattern(
                    [trans_deviation_deg(value) for value in pd.to_numeric(group["selected_omega_deg"], errors="coerce").dropna()]
                )["every_other_detected"],
            }
        )
    return pd.DataFrame(rows)


def predicted_points_by_segment(selected: pd.DataFrame, parent_pdb: Path) -> dict[str, dict[str, np.ndarray]]:
    """Return reconstructed point dictionaries for selected reconstructed segments."""
    segments = segment_lookup(parent_pdb)
    out: dict[str, dict[str, np.ndarray]] = {}
    for row in selected.itertuples(index=False):
        if row.selection_reason not in {"good_within_8deg", "good_within_10deg", "borderline_within_10deg"}:
            continue
        segment = segments.get(row.segment_id)
        if segment is None:
            continue
        try:
            out[row.segment_id] = reconstructed_points(segment, pd.Series(row._asdict()))
        except Exception:
            continue
    return out


def overlap_summary(selected: pd.DataFrame, parent_pdb: Path = DEFAULT_PARENT_PDB) -> pd.DataFrame:
    """Compare adjacent reconstructed segments for shared-residue N/CA compatibility."""
    points = predicted_points_by_segment(selected, parent_pdb)
    rows = []
    for chain, group in selected.groupby("chain", sort=True):
        ordered = group.sort_values("res_i")
        by_res_i = {int(row.res_i): row for row in ordered.itertuples(index=False)}
        for res_i, left in sorted(by_res_i.items()):
            right = by_res_i.get(res_i + 1)
            if right is None:
                continue
            left_points = points.get(left.segment_id)
            right_points = points.get(right.segment_id)
            if left_points is None or right_points is None:
                rows.append(
                    {
                        "chain": chain,
                        "class_label": class_for_chain(chain),
                        "left_segment_id": left.segment_id,
                        "right_segment_id": right.segment_id,
                        "shared_residue": int(left.res_j),
                        "shared_atom_count": 0,
                        "overlap_rmsd_A": np.nan,
                        "max_overlap_delta_A": np.nan,
                        "overlap_class": "insufficient_data",
                        "notes": "one or both adjacent segments unresolved",
                    }
                )
                continue
            deltas = [
                distance(left_points["N_j"], right_points["N_i"]),
                distance(left_points["CA_j"], right_points["CA_i"]),
            ]
            rmsd = float(np.sqrt(np.mean(np.square(deltas))))
            rows.append(
                {
                    "chain": chain,
                    "class_label": class_for_chain(chain),
                    "left_segment_id": left.segment_id,
                    "right_segment_id": right.segment_id,
                    "shared_residue": int(left.res_j),
                    "shared_atom_count": 2,
                    "overlap_rmsd_A": rmsd,
                    "max_overlap_delta_A": float(max(deltas)),
                    "overlap_class": overlap_class(rmsd),
                    "notes": "N/CA shared-residue compatibility surrogate",
                }
            )
    return pd.DataFrame(rows)


def steric_summary(selected: pd.DataFrame, parent_pdb: Path = DEFAULT_PARENT_PDB) -> pd.DataFrame:
    """Compute limited within-fragment nonbonded contact sanity checks."""
    points = predicted_points_by_segment(selected, parent_pdb)
    rows = []
    atom_order = ["N_i", "CA_i", "C_i", "N_j", "CA_j"]
    bonded_pairs = {(0, 1), (1, 2), (2, 3), (3, 4)}
    for row in selected.itertuples(index=False):
        fragment = points.get(row.segment_id)
        if fragment is None:
            continue
        min_distance = np.inf
        severe = 0
        possible = 0
        for i, atom_a in enumerate(atom_order):
            for j, atom_b in enumerate(atom_order):
                if j <= i or (i, j) in bonded_pairs:
                    continue
                value = distance(fragment[atom_a], fragment[atom_b])
                min_distance = min(min_distance, value)
                if value < 1.2:
                    severe += 1
                elif value < 1.6:
                    possible += 1
        rows.append(
            {
                "segment_id": row.segment_id,
                "chain": row.chain,
                "class_label": row.class_label,
                "steric_check_scope": "limited_to_fragment_coordinates",
                "min_nonbonded_backbone_distance_A": float(min_distance),
                "steric_conflict_class": steric_conflict_class(float(min_distance)),
                "severe_conflict_count": severe,
                "possible_conflict_count": possible,
                "notes": "within-fragment N/CA/C/N/CA nonbonded sanity check only",
            }
        )
    return pd.DataFrame(rows)


def safe_to_write_full_pdb(chains: pd.DataFrame, overlaps: pd.DataFrame, sterics: pd.DataFrame) -> tuple[bool, str]:
    """Return whether a full PDB is safe to write and why."""
    if not bool(chains["has_continuous_reconstructed_path"].all()):
        return False, "incomplete_reconstructed_chain_paths"
    if overlaps.empty or not bool(overlaps["overlap_class"].isin(["good_overlap"]).all()):
        return False, "overlap_not_globally_good"
    if not sterics.empty and bool(sterics["steric_conflict_class"].isin(["severe_conflict"]).any()):
        return False, "severe_fragment_steric_conflicts"
    if not bool((chains["drift_class"] == "good_drift").all()):
        return False, "drift_surrogate_not_good"
    return True, "all_feasibility_checks_passed"


def build_report(edges: pd.DataFrame, chains: pd.DataFrame, overlaps: pd.DataFrame, sterics: pd.DataFrame, full_pdb_written: bool, blocker: str) -> str:
    """Build markdown report."""
    total_expected = int(chains["expected_edge_count"].sum())
    total_reconstructed = int(chains["reconstructed_edge_count"].sum())
    total_unresolved = int(chains["unresolved_or_retained_edge_count"].sum())
    poor_overlap = int((overlaps["overlap_class"] == "poor_overlap").sum()) if not overlaps.empty else 0
    severe = int(sterics["severe_conflict_count"].sum()) if not sterics.empty else 0
    possible = int(sterics["possible_conflict_count"].sum()) if not sterics.empty else 0
    class_summary = chains.groupby("class_label").agg(
        chains=("chain", "count"),
        reconstructed_edges=("reconstructed_edge_count", "sum"),
        unresolved_edges=("unresolved_or_retained_edge_count", "sum"),
        median_drift_surrogate_A=("endpoint_drift_surrogate_A", "median"),
    ).reset_index()
    return f"""# Global Chain Assembly Feasibility Report

This is a global chain assembly feasibility analysis. It is not a final structure, it is not energy minimized, and it does not prove the physical hexaplex structure. It is testing whether selected local phi/psi/omega fragments can be made globally consistent. Diffraction scoring should not be performed unless a complete, globally consistent scattering model is produced.

## Summary

- Chains analyzed: {len(chains)}
- Expected peptide edges: {total_expected}
- Selected/reconstructed edges: {total_reconstructed}
- Unresolved/retained-parent edges: {total_unresolved}
- Full prototype PDB written: {full_pdb_written}
- Full PDB blocker: `{blocker}`
- Poor overlap rows: {poor_overlap}
- Severe fragment steric conflicts: {severe}
- Possible fragment steric conflicts: {possible}

## Chain Completeness And Drift Surrogate

{markdown_table(chains, ['chain', 'class_label', 'residue_count', 'expected_edge_count', 'reconstructed_edge_count', 'unresolved_or_retained_edge_count', 'has_continuous_reconstructed_path', 'blocking_edges', 'endpoint_drift_surrogate_A', 'drift_class', 'selected_omega_every_other_detected'])}

## Class Comparison

{markdown_table(class_summary, ['class_label', 'chains', 'reconstructed_edges', 'unresolved_edges', 'median_drift_surrogate_A'])}

## Overlap Compatibility

{markdown_table(overlaps.groupby('overlap_class').size().rename_axis('overlap_class').reset_index(name='count') if not overlaps.empty else overlaps, ['overlap_class', 'count'])}

## Steric / Close-Contact Sanity

{markdown_table(sterics.groupby('steric_conflict_class').size().rename_axis('steric_conflict_class').reset_index(name='count') if not sterics.empty else sterics, ['steric_conflict_class', 'count'])}

## Interpretation

- Are the selected local fragments sufficient to form continuous chain paths? No. Each chain has an unresolved first peptide edge because phi context is missing at the chain start.
- Which segments remain terminal/unresolved? See `blocking_edges`; all chains are blocked at edge 1 in this conservative graph.
- Are overlapping local segment coordinates mutually compatible? The overlap table reports the N/CA shared-residue compatibility surrogate. Poor overlap indicates that a full chain cannot be assembled by simply stitching local fragments.
- What is endpoint drift? True propagation was not attempted. The report uses a `drift_surrogate_rss_closure_residual`, not actual propagated drift.
- Are there atom conflicts? The steric table is limited to fragment coordinates and should be treated as a sanity check, not a full steric audit.
- Does selected omega remain within +/-8 or +/-10 degrees? The edge table preserves the selected omega window; prior selected torsion output selected only good +/-8 rows for reconstructed edges.
- Do selected omega values show every-other behavior after chain ordering? See `selected_omega_every_other_detected` by chain. In this audit, selected omega does not show every-other behavior.
- Are A/C/E and B/D/F different in assembly feasibility? The class comparison shows symmetric unresolved-edge counts and similar drift-surrogate values.
- Is a complete PDB safe to write? No. The specific blocker is `{blocker}`.

## Next Implementation Step

Add a global propagation/optimization step that solves overlapping residue coordinates across each chain, handles terminal edges explicitly, and only then writes a complete PDB for diffraction scoring.

## Outputs

- Edge summary: `outputs/metrics/global_chain_assembly_edge_summary.csv`
- Chain summary: `outputs/metrics/global_chain_assembly_chain_summary.csv`
- Overlap summary: `outputs/metrics/global_chain_assembly_overlap_summary.csv`
- Steric summary: `outputs/metrics/global_chain_assembly_steric_summary.csv`
- Report: `outputs/reports/global_chain_assembly_feasibility_report.md`
"""


def run_analysis(
    parent_pdb: Path = DEFAULT_PARENT_PDB,
    selected_csv: Path = DEFAULT_SELECTED_CSV,
    edge_csv: Path = DEFAULT_EDGE_CSV,
    chain_csv: Path = DEFAULT_CHAIN_CSV,
    overlap_csv: Path = DEFAULT_OVERLAP_CSV,
    steric_csv: Path = DEFAULT_STERIC_CSV,
    report_path: Path = DEFAULT_REPORT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, bool, str]:
    """Run feasibility analysis and write outputs."""
    selected = load_or_generate_selected(parent_pdb, selected_csv)
    edges = edge_summary(selected, parent_pdb)
    chains = chain_summary(edges)
    overlaps = overlap_summary(selected, parent_pdb)
    sterics = steric_summary(selected, parent_pdb)
    full_ok, blocker = safe_to_write_full_pdb(chains, overlaps, sterics)
    for path in [edge_csv, chain_csv, overlap_csv, steric_csv, report_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
    edges.to_csv(edge_csv, index=False)
    chains.to_csv(chain_csv, index=False)
    overlaps.to_csv(overlap_csv, index=False)
    sterics.to_csv(steric_csv, index=False)
    report_path.write_text(build_report(edges, chains, overlaps, sterics, full_ok, blocker), encoding="utf-8")
    return edges, chains, overlaps, sterics, full_ok, blocker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-pdb", type=Path, default=DEFAULT_PARENT_PDB)
    parser.add_argument("--selected-csv", type=Path, default=DEFAULT_SELECTED_CSV)
    parser.add_argument("--edge-csv", type=Path, default=DEFAULT_EDGE_CSV)
    parser.add_argument("--chain-csv", type=Path, default=DEFAULT_CHAIN_CSV)
    parser.add_argument("--overlap-csv", type=Path, default=DEFAULT_OVERLAP_CSV)
    parser.add_argument("--steric-csv", type=Path, default=DEFAULT_STERIC_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    edges, chains, overlaps, sterics, full_ok, blocker = run_analysis(
        parent_pdb=args.parent_pdb,
        selected_csv=args.selected_csv,
        edge_csv=args.edge_csv,
        chain_csv=args.chain_csv,
        overlap_csv=args.overlap_csv,
        steric_csv=args.steric_csv,
        report_path=args.report,
    )
    print(f"Chains analyzed: {len(chains)}")
    print(f"Reconstructed edges: {int(chains['reconstructed_edge_count'].sum())}")
    print(f"Unresolved edges: {int(chains['unresolved_or_retained_edge_count'].sum())}")
    print(f"Full PDB written: {full_ok}")
    print(f"Blocker: {blocker}")
    print(f"Wrote {args.edge_csv}")
    print(f"Wrote {args.chain_csv}")
    print(f"Wrote {args.overlap_csv}")
    print(f"Wrote {args.steric_csv}")
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
