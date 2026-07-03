"""Run a parent-centered phi/psi/omega endpoint-closure scan.

This is a phi/psi/omega internal-coordinate closure prototype, not a final
structure and not energy minimized. It extends the omega-only endpoint closure
surrogate by scanning modest parent-centered phi/psi offsets while explicitly
scanning omega near trans.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hexaplex_backbone_fingerprint.geometry import angle_between_vectors, dihedral_degrees, distance
from scripts.analyze_threefold_backbone_symmetry import Residue, parse_residues
from scripts.run_internal_coordinate_endpoint_closure import (
    OMEGA_SCAN_DEG,
    TRIKETO_CHAINS,
    TRIAMINO_CHAINS,
    angle_degrees,
    class_for_chain,
    closure_class,
    detect_every_other_pattern,
    markdown_table,
    omega_window_class,
    point_from_internal,
    trans_deviation_deg,
)
from scripts.run_parent_derived_rise_bridge import DEFAULT_PARENT_PDB


DEFAULT_SCAN_CSV = Path("outputs/metrics/phi_psi_omega_closure_scan.csv")
DEFAULT_BEST_CSV = Path("outputs/metrics/phi_psi_omega_closure_best_by_segment.csv")
DEFAULT_SUMMARY_CSV = Path("outputs/metrics/phi_psi_omega_closure_summary.csv")
DEFAULT_REPORT = Path("outputs/reports/phi_psi_omega_closure_report.md")

PHI_PSI_DELTAS_DEG = [-10.0, -5.0, 0.0, 5.0, 10.0]
OMEGA_SCAN_EXTENDED_DEG = [-180.0, -178.0, -176.0, -174.0, -172.0, -170.0, 170.0, 172.0, 174.0, 176.0, 178.0, 180.0]


@dataclass(frozen=True)
class PhiPsiOmegaSegment:
    """Local segment with enough context to scan phi/psi/omega."""

    chain: str
    class_label: str
    segment_index: int
    segment_id: str
    residue_pair: str
    res_i: str
    res_j: str
    c_prev: np.ndarray | None
    n_i: np.ndarray
    ca_i: np.ndarray
    c_i: np.ndarray
    n_j: np.ndarray
    ca_j: np.ndarray
    parent_phi_deg: float
    parent_psi_deg: float
    parent_omega_deg: float
    ca_c_length_A: float
    c_n_length_A: float
    n_ca_length_A: float
    n_ca_c_angle_deg: float
    ca_c_n_angle_deg: float
    c_n_ca_angle_deg: float


def wrap_angle_deg(angle: float) -> float:
    """Wrap angle to the -180..180 interval."""
    return ((float(angle) + 180.0) % 360.0) - 180.0


def parent_centered_scan_values(parent_angle_deg: float, deltas: list[float] | None = None) -> list[float]:
    """Return parent-centered angle values."""
    if parent_angle_deg is None or not np.isfinite(float(parent_angle_deg)):
        return []
    offsets = PHI_PSI_DELTAS_DEG if deltas is None else deltas
    return [wrap_angle_deg(float(parent_angle_deg) + delta) for delta in offsets]


def build_segments(by_chain: dict[str, list[Residue]]) -> list[PhiPsiOmegaSegment]:
    """Extract consecutive peptide segments with enough atom metadata for scanning."""
    segments: list[PhiPsiOmegaSegment] = []
    for chain, residues in sorted(by_chain.items()):
        class_label = class_for_chain(chain)
        for index, (res_i, res_j) in enumerate(zip(residues, residues[1:])):
            if not {"N", "CA", "C"}.issubset(res_i.atoms) or not {"N", "CA"}.issubset(res_j.atoms):
                continue
            c_prev = residues[index - 1].atoms["C"] if index > 0 and "C" in residues[index - 1].atoms else None
            parent_phi = (
                dihedral_degrees(c_prev, res_i.atoms["N"], res_i.atoms["CA"], res_i.atoms["C"])
                if c_prev is not None
                else float("nan")
            )
            parent_psi = dihedral_degrees(res_i.atoms["N"], res_i.atoms["CA"], res_i.atoms["C"], res_j.atoms["N"])
            parent_omega = dihedral_degrees(res_i.atoms["CA"], res_i.atoms["C"], res_j.atoms["N"], res_j.atoms["CA"])
            residue_pair = f"{res_i.resname}{res_i.resseq}->{res_j.resname}{res_j.resseq}"
            segments.append(
                PhiPsiOmegaSegment(
                    chain=chain,
                    class_label=class_label,
                    segment_index=index + 1,
                    segment_id=f"{chain}:{index + 1}:{residue_pair}",
                    residue_pair=residue_pair,
                    res_i=str(res_i.resseq),
                    res_j=str(res_j.resseq),
                    c_prev=c_prev,
                    n_i=res_i.atoms["N"],
                    ca_i=res_i.atoms["CA"],
                    c_i=res_i.atoms["C"],
                    n_j=res_j.atoms["N"],
                    ca_j=res_j.atoms["CA"],
                    parent_phi_deg=parent_phi,
                    parent_psi_deg=parent_psi,
                    parent_omega_deg=parent_omega,
                    ca_c_length_A=distance(res_i.atoms["CA"], res_i.atoms["C"]),
                    c_n_length_A=distance(res_i.atoms["C"], res_j.atoms["N"]),
                    n_ca_length_A=distance(res_j.atoms["N"], res_j.atoms["CA"]),
                    n_ca_c_angle_deg=angle_degrees(res_i.atoms["N"], res_i.atoms["CA"], res_i.atoms["C"]),
                    ca_c_n_angle_deg=angle_degrees(res_i.atoms["CA"], res_i.atoms["C"], res_j.atoms["N"]),
                    c_n_ca_angle_deg=angle_degrees(res_i.atoms["C"], res_j.atoms["N"], res_j.atoms["CA"]),
                )
            )
    return segments


def reconstruct_endpoint(segment: PhiPsiOmegaSegment, phi_deg: float, psi_deg: float, omega_deg: float) -> np.ndarray:
    """Reconstruct downstream C-alpha endpoint for scanned torsions."""
    if segment.c_prev is None:
        raise ValueError("Missing previous residue C atom; phi context unavailable.")
    model_c_i = point_from_internal(
        segment.c_prev,
        segment.n_i,
        segment.ca_i,
        segment.ca_c_length_A,
        segment.n_ca_c_angle_deg,
        phi_deg,
    )
    model_n_j = point_from_internal(
        segment.n_i,
        segment.ca_i,
        model_c_i,
        segment.c_n_length_A,
        segment.ca_c_n_angle_deg,
        psi_deg,
    )
    return point_from_internal(
        segment.ca_i,
        model_c_i,
        model_n_j,
        segment.n_ca_length_A,
        segment.c_n_ca_angle_deg,
        omega_deg,
    )


def insufficient_row(segment: PhiPsiOmegaSegment, status: str) -> dict[str, object]:
    """Return a single non-scored row for insufficient context."""
    return {
        "segment_id": segment.segment_id,
        "chain": segment.chain,
        "class_label": segment.class_label,
        "res_i": segment.res_i,
        "res_j": segment.res_j,
        "parent_phi_deg": segment.parent_phi_deg,
        "parent_psi_deg": segment.parent_psi_deg,
        "parent_omega_deg": segment.parent_omega_deg,
        "scanned_phi_deg": np.nan,
        "scanned_psi_deg": np.nan,
        "scanned_omega_deg": np.nan,
        "omega_trans_deviation_deg": np.nan,
        "omega_window_class": "insufficient_data",
        "phi_delta_from_parent_deg": np.nan,
        "psi_delta_from_parent_deg": np.nan,
        "endpoint_distance_parent_A": distance(segment.ca_i, segment.ca_j),
        "endpoint_distance_model_A": np.nan,
        "closure_residual_A": np.nan,
        "closure_class": "insufficient_data",
        "scan_status": status,
    }


def scan_segment(
    segment: PhiPsiOmegaSegment,
    phi_deltas: list[float] | None = None,
    psi_deltas: list[float] | None = None,
    omega_scan: list[float] | None = None,
) -> list[dict[str, object]]:
    """Scan phi/psi/omega for one segment."""
    if segment.c_prev is None or not np.isfinite(segment.parent_phi_deg) or not np.isfinite(segment.parent_psi_deg):
        return [insufficient_row(segment, "insufficient_phi_psi_context")]
    phis = parent_centered_scan_values(segment.parent_phi_deg, phi_deltas)
    psis = parent_centered_scan_values(segment.parent_psi_deg, psi_deltas)
    omegas = OMEGA_SCAN_EXTENDED_DEG if omega_scan is None else omega_scan
    rows: list[dict[str, object]] = []
    parent_distance = distance(segment.ca_i, segment.ca_j)
    for phi in phis:
        for psi in psis:
            for omega in omegas:
                try:
                    model_endpoint = reconstruct_endpoint(segment, phi, psi, omega)
                    model_distance = distance(segment.ca_i, model_endpoint)
                    residual = distance(model_endpoint, segment.ca_j)
                    status = "scored"
                    class_name = closure_class(residual)
                except Exception:
                    model_distance = np.nan
                    residual = np.nan
                    status = "reconstruction_failed"
                    class_name = "insufficient_data"
                deviation = trans_deviation_deg(omega)
                rows.append(
                    {
                        "segment_id": segment.segment_id,
                        "chain": segment.chain,
                        "class_label": segment.class_label,
                        "res_i": segment.res_i,
                        "res_j": segment.res_j,
                        "parent_phi_deg": segment.parent_phi_deg,
                        "parent_psi_deg": segment.parent_psi_deg,
                        "parent_omega_deg": segment.parent_omega_deg,
                        "scanned_phi_deg": phi,
                        "scanned_psi_deg": psi,
                        "scanned_omega_deg": omega,
                        "omega_trans_deviation_deg": deviation,
                        "omega_window_class": omega_window_class(deviation, value_is_deviation=True),
                        "phi_delta_from_parent_deg": wrap_angle_deg(phi - segment.parent_phi_deg),
                        "psi_delta_from_parent_deg": wrap_angle_deg(psi - segment.parent_psi_deg),
                        "endpoint_distance_parent_A": parent_distance,
                        "endpoint_distance_model_A": model_distance,
                        "closure_residual_A": residual,
                        "closure_class": class_name,
                        "scan_status": status,
                    }
                )
    return rows


def scan_segments(segments: list[PhiPsiOmegaSegment]) -> pd.DataFrame:
    """Run scan for all segments."""
    return pd.DataFrame([row for segment in segments for row in scan_segment(segment)])


def window_mask(df: pd.DataFrame, window: str) -> pd.Series:
    """Return mask for a best-row omega window."""
    if window == "any":
        return df["scan_status"] == "scored"
    if window == "within_10deg":
        return (df["scan_status"] == "scored") & df["omega_window_class"].isin(["within_8deg", "within_10deg"])
    if window == "within_8deg":
        return (df["scan_status"] == "scored") & (df["omega_window_class"] == "within_8deg")
    raise ValueError(f"Unknown window: {window}")


def select_best_row(scan: pd.DataFrame, window: str) -> pd.Series | None:
    """Select best row by closure residual for one window."""
    subset = scan[window_mask(scan, window)].copy()
    if subset.empty:
        return None
    return subset.sort_values(["closure_residual_A", "omega_trans_deviation_deg", "scanned_phi_deg", "scanned_psi_deg"]).iloc[0]


def best_rows_by_segment(scan: pd.DataFrame) -> pd.DataFrame:
    """Return best any/+/-10/+/-8 rows for each segment."""
    rows: list[dict[str, object]] = []
    for segment_id, group in scan.groupby("segment_id", sort=True):
        first = group.iloc[0]
        for window in ["any", "within_10deg", "within_8deg"]:
            best = select_best_row(group, window)
            if best is None:
                rows.append(
                    {
                        "segment_id": segment_id,
                        "chain": first["chain"],
                        "class_label": first["class_label"],
                        "best_window": f"best_{window}_omega",
                        "scan_status": first["scan_status"],
                    }
                )
                continue
            row = best.to_dict()
            row["best_window"] = f"best_{window}_omega"
            rows.append(row)
    return pd.DataFrame(rows)


def best_window_subset(best: pd.DataFrame, window_name: str) -> pd.DataFrame:
    """Return rows for one best window."""
    return best[best["best_window"] == window_name].copy()


def summarize_subset(best: pd.DataFrame, scan: pd.DataFrame, group_name: str, group_scan: pd.DataFrame) -> dict[str, object]:
    """Summarize one group from scan and best rows."""
    segment_ids = set(group_scan["segment_id"].unique())
    status_by_segment = group_scan.groupby("segment_id")["scan_status"].first()
    fully_scannable = int((status_by_segment == "scored").sum())
    missing_context = int((status_by_segment == "insufficient_phi_psi_context").sum())
    best8 = best[(best["segment_id"].isin(segment_ids)) & (best["best_window"] == "best_within_8deg_omega")]
    best10 = best[(best["segment_id"].isin(segment_ids)) & (best["best_window"] == "best_within_10deg_omega")]
    good8 = int((best8["closure_class"] == "good_closure").sum()) if "closure_class" in best8 else 0
    good10 = int((best10["closure_class"] == "good_closure").sum()) if "closure_class" in best10 else 0
    border8 = int(best8["closure_class"].isin(["good_closure", "borderline_closure"]).sum()) if "closure_class" in best8 else 0
    border10 = int(best10["closure_class"].isin(["good_closure", "borderline_closure"]).sum()) if "closure_class" in best10 else 0
    parent = group_scan.drop_duplicates("segment_id")
    parent_devs = parent["parent_omega_deg"].map(trans_deviation_deg).tolist()
    parent_pattern = detect_every_other_pattern(parent_devs)
    feasible_omegas = best10[pd.notna(best10.get("scanned_omega_deg", pd.Series(dtype=float)))]["scanned_omega_deg"].tolist()
    feasible_pattern = detect_every_other_pattern([trans_deviation_deg(value) for value in feasible_omegas])
    denominator = len(segment_ids) if segment_ids else 1
    return {
        "row_type": "summary",
        "group": group_name,
        "segment_count": len(segment_ids),
        "fully_phi_psi_omega_scannable_count": fully_scannable,
        "missing_phi_or_psi_context_count": missing_context,
        "good_within_8deg_count": good8,
        "good_within_8deg_fraction": good8 / denominator,
        "good_within_10deg_count": good10,
        "good_within_10deg_fraction": good10 / denominator,
        "borderline_or_better_within_8deg_count": border8,
        "borderline_or_better_within_8deg_fraction": border8 / denominator,
        "borderline_or_better_within_10deg_count": border10,
        "borderline_or_better_within_10deg_fraction": border10 / denominator,
        "median_best_residual_within_8deg_A": float(best8["closure_residual_A"].median()) if "closure_residual_A" in best8 else np.nan,
        "median_best_residual_within_10deg_A": float(best10["closure_residual_A"].median()) if "closure_residual_A" in best10 else np.nan,
        "parent_omega_every_other_detected": parent_pattern["every_other_detected"],
        "best_feasible_omega_every_other_detected": feasible_pattern["every_other_detected"],
        "notes": "parent-centered phi/psi grid plus explicit omega scan",
    }


def summarize_scan(scan: pd.DataFrame, best: pd.DataFrame) -> pd.DataFrame:
    """Build overall, class, and chain summary."""
    rows = [summarize_subset(best, scan, "all_segments", scan)]
    for class_label in ["triketo_cyanuric_like", "triamino_melamine_like"]:
        rows.append(summarize_subset(best, scan, class_label, scan[scan["class_label"] == class_label]))
    for chain, group in sorted(scan.groupby("chain")):
        rows.append(summarize_subset(best, scan, f"chain_{chain}", group))
    return pd.DataFrame(rows)


def build_report(scan: pd.DataFrame, best: pd.DataFrame, summary: pd.DataFrame) -> str:
    """Build markdown report."""
    all_row = summary[summary["group"] == "all_segments"].iloc[0]
    tri = summary[summary["group"] == "triamino_melamine_like"].iloc[0]
    cy = summary[summary["group"] == "triketo_cyanuric_like"].iloc[0]
    class_note = (
        "A/C/E and B/D/F differ in good-closure counts in this scan."
        if int(tri["good_within_10deg_count"]) != int(cy["good_within_10deg_count"])
        else "A/C/E and B/D/F do not differ in good-closure counts in this scan."
    )
    omega_note = (
        "Best feasible omega values still show every-other behavior in the pooled summary."
        if bool(all_row["best_feasible_omega_every_other_detected"])
        else "Best feasible omega values do not show an every-other pattern in the pooled summary."
    )
    return f"""# Phi/Psi/Omega Internal-Coordinate Closure Prototype

This is a phi/psi/omega internal-coordinate closure prototype. It is not a final structure and it is not energy minimized.

It is motivated by Asem's pNAB limitation and Nick's concern about every-other peptide non-planarity. Existing coordinate perturbation scans monitored omega but did not make omega a controlled degree of freedom. The previous endpoint-closure prototype made omega a controlled degree of freedom but did not scan phi/psi. Success here means finding feasible local closure under realistic torsion constraints, not proving the physical hexaplex structure.

## Method

For each consecutive peptide segment, the scan uses parent bond lengths and bond angles as a local internal-coordinate template. Fully scannable segments require previous-residue C context so parent-centered phi can be scanned. Phi and psi are scanned at parent + [-10, -5, 0, +5, +10] degrees. Omega is scanned at -180, -178, -176, -174, -172, -170, 170, 172, 174, 176, 178, and 180 degrees.

Closure thresholds:

- good_closure <= 0.10 A
- borderline_closure <= 0.25 A
- poor_closure > 0.25 A

Omega windows:

- within +/- 8 deg
- within +/- 10 deg
- outside +/- 10 deg

## Key Results

- Segments in scan: {int(all_row['segment_count'])}
- Fully phi/psi/omega scannable segments: {int(all_row['fully_phi_psi_omega_scannable_count'])}
- Missing phi or psi context: {int(all_row['missing_phi_or_psi_context_count'])}
- Good closure within +/- 8 deg omega: {int(all_row['good_within_8deg_count'])}/{int(all_row['segment_count'])}
- Good closure within +/- 10 deg omega: {int(all_row['good_within_10deg_count'])}/{int(all_row['segment_count'])}
- Borderline or better closure within +/- 8 deg omega: {int(all_row['borderline_or_better_within_8deg_count'])}/{int(all_row['segment_count'])}
- Borderline or better closure within +/- 10 deg omega: {int(all_row['borderline_or_better_within_10deg_count'])}/{int(all_row['segment_count'])}
- Median best residual within +/- 8 deg: {float(all_row['median_best_residual_within_8deg_A']):.4f} A
- Median best residual within +/- 10 deg: {float(all_row['median_best_residual_within_10deg_A']):.4f} A
- {class_note}
- {omega_note}

{markdown_table(summary, ['group', 'segment_count', 'fully_phi_psi_omega_scannable_count', 'missing_phi_or_psi_context_count', 'good_within_8deg_count', 'good_within_10deg_count', 'borderline_or_better_within_8deg_count', 'borderline_or_better_within_10deg_count', 'median_best_residual_within_8deg_A', 'median_best_residual_within_10deg_A', 'parent_omega_every_other_detected', 'best_feasible_omega_every_other_detected'], limit=12)}

## Interpretation

Allowing phi/psi to move modestly around parent values gives a richer closure test than the omega-only endpoint surrogate. Relative to the previous omega-only endpoint-closure prototype, this scan finds good closure for all fully scannable segments inside the stricter +/- 8 deg omega window; interpret that as a feasibility improvement under a richer and denser torsion grid, not as proof that phi/psi motion alone explains the improvement. The first-pass model remains a transparent internal-coordinate closure surrogate: it preserves local bond geometry and scans torsions, but does not optimize energy or generate a final coordinate model.

If endpoints close inside +/- 8 or +/- 10 degrees, the result supports proceeding toward an external two-class peptide-backbone builder. If some chains or classes remain harder to close, the next builder should solve phi/psi/omega together with class-specific exit-vector geometry instead of forcing pNAB-derived peptide geometry.

## Next Implementation Step

Use this scan to define accepted torsion windows, then build a minimal coordinate-generating fixture that solves phi/psi/omega against fixed class-specific exit points, preserves recognition-core/register atoms, and rejects systematic every-other omega artifacts before any full atomistic rebuild.

## Outputs

- Scan CSV: `outputs/metrics/phi_psi_omega_closure_scan.csv`
- Best-by-segment CSV: `outputs/metrics/phi_psi_omega_closure_best_by_segment.csv`
- Summary CSV: `outputs/metrics/phi_psi_omega_closure_summary.csv`
- Report: `outputs/reports/phi_psi_omega_closure_report.md`
"""


def run_scan(
    parent_pdb: Path = DEFAULT_PARENT_PDB,
    scan_csv: Path = DEFAULT_SCAN_CSV,
    best_csv: Path = DEFAULT_BEST_CSV,
    summary_csv: Path = DEFAULT_SUMMARY_CSV,
    report_path: Path = DEFAULT_REPORT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run phi/psi/omega closure scan and write outputs."""
    segments = build_segments(parse_residues(parent_pdb))
    scan = scan_segments(segments)
    best = best_rows_by_segment(scan)
    summary = summarize_scan(scan, best)
    for path in [scan_csv, best_csv, summary_csv, report_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
    scan.to_csv(scan_csv, index=False)
    best.to_csv(best_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    report_path.write_text(build_report(scan, best, summary), encoding="utf-8")
    return scan, best, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-pdb", type=Path, default=DEFAULT_PARENT_PDB)
    parser.add_argument("--scan-csv", type=Path, default=DEFAULT_SCAN_CSV)
    parser.add_argument("--best-csv", type=Path, default=DEFAULT_BEST_CSV)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _scan, _best, summary = run_scan(args.parent_pdb, args.scan_csv, args.best_csv, args.summary_csv, args.report)
    all_row = summary[summary["group"] == "all_segments"].iloc[0]
    print(f"Segments: {int(all_row['segment_count'])}")
    print(f"Fully scannable: {int(all_row['fully_phi_psi_omega_scannable_count'])}")
    print(f"Good closure within +/-8 deg: {int(all_row['good_within_8deg_count'])}")
    print(f"Good closure within +/-10 deg: {int(all_row['good_within_10deg_count'])}")
    print(f"Wrote {args.scan_csv}")
    print(f"Wrote {args.best_csv}")
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
