"""Run a minimal internal-coordinate endpoint-closure omega scan.

This is a feasibility prototype, not a final structure. It tests whether a
local idealized peptide segment can connect fixed parent endpoint geometry while
omega is scanned near trans. The first pass uses parent bond lengths, bond
angles, and parent psi as a local template, then reconstructs the downstream
C-alpha endpoint for each scanned omega value.
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

from hexaplex_backbone_fingerprint.geometry import angle_between_vectors, dihedral_degrees, distance, normalize
from scripts.analyze_threefold_backbone_symmetry import Residue, parse_residues, peptide_plane_normal
from scripts.run_parent_derived_rise_bridge import DEFAULT_PARENT_PDB


DEFAULT_SCAN_CSV = Path("outputs/metrics/internal_coordinate_endpoint_closure_scan.csv")
DEFAULT_SUMMARY_CSV = Path("outputs/metrics/internal_coordinate_endpoint_closure_summary.csv")
DEFAULT_REPORT = Path("outputs/reports/internal_coordinate_endpoint_closure_report.md")

TRIKETO_CHAINS = {"A", "C", "E"}
TRIAMINO_CHAINS = {"B", "D", "F"}
OMEGA_SCAN_DEG = [-180.0, -175.0, -172.0, -170.0, -168.0, -165.0, -160.0]


@dataclass(frozen=True)
class PeptideSegment:
    """Local peptide endpoint geometry for one consecutive residue pair."""

    chain: str
    class_label: str
    segment_index: int
    residue_pair: str
    res_i_index: int
    res_j_index: int
    n_i: np.ndarray
    ca_i: np.ndarray
    c_i: np.ndarray
    n_j: np.ndarray
    ca_j: np.ndarray
    parent_phi_deg: float
    parent_psi_deg: float
    parent_omega_deg: float
    parent_theta_deg: float
    c_n_length_A: float
    n_ca_length_A: float
    ca_c_n_angle_deg: float
    c_n_ca_angle_deg: float


def class_for_chain(chain: str) -> str:
    """Return class assignment established by the three-fold diagnostic."""
    if chain in TRIKETO_CHAINS:
        return "triketo_cyanuric_like"
    if chain in TRIAMINO_CHAINS:
        return "triamino_melamine_like"
    return "unclassified"


def trans_deviation_deg(omega_deg: float) -> float:
    """Return deviation from trans, handling +/-180 degree wraparound."""
    if omega_deg is None or not np.isfinite(float(omega_deg)):
        return float("nan")
    return abs(abs(float(omega_deg)) - 180.0)


def omega_window_class(omega_or_deviation_deg: float, *, value_is_deviation: bool = False) -> str:
    """Classify omega as within +/-8, +/-10, or outside +/-10 degrees."""
    deviation = float(omega_or_deviation_deg) if value_is_deviation else trans_deviation_deg(float(omega_or_deviation_deg))
    if not np.isfinite(deviation):
        return "insufficient_data"
    if deviation <= 8.0:
        return "within_8deg"
    if deviation <= 10.0:
        return "within_10deg"
    return "outside_10deg"


def closure_class(residual_A: float) -> str:
    """Classify endpoint closure residual."""
    if not np.isfinite(float(residual_A)):
        return "insufficient_data"
    if residual_A <= 0.10:
        return "good_closure"
    if residual_A <= 0.25:
        return "borderline_closure"
    return "poor_closure"


def detect_every_other_pattern(values: list[float], threshold_deg: float = 10.0) -> dict[str, object]:
    """Detect a simple every-other high/low pattern in trans deviations."""
    finite = [float(value) for value in values if np.isfinite(float(value))]
    if len(finite) < 4:
        return {
            "every_other_detected": False,
            "alternating_fraction": float("nan"),
            "even_median_deg": float("nan"),
            "odd_median_deg": float("nan"),
        }
    states = [value > threshold_deg for value in finite]
    transitions = [left != right for left, right in zip(states, states[1:])]
    alternating_fraction = float(sum(transitions) / len(transitions)) if transitions else float("nan")
    even_median = float(np.median(finite[0::2]))
    odd_median = float(np.median(finite[1::2]))
    detected = alternating_fraction >= 0.8 and abs(even_median - odd_median) >= threshold_deg * 0.5
    return {
        "every_other_detected": bool(detected),
        "alternating_fraction": alternating_fraction,
        "even_median_deg": even_median,
        "odd_median_deg": odd_median,
    }


def angle_degrees(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return angle ABC in degrees."""
    return angle_between_vectors(np.asarray(a) - np.asarray(b), np.asarray(c) - np.asarray(b))


def point_from_internal(a: np.ndarray, b: np.ndarray, c: np.ndarray, length: float, angle_deg: float, dihedral_deg: float) -> np.ndarray:
    """Construct point D from A-B-C, |C-D|, angle B-C-D, and dihedral A-B-C-D."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    bc = normalize(c - b)
    n = normalize(np.cross(b - a, bc))
    m = np.cross(n, bc)
    theta = np.radians(angle_deg)
    phi = np.radians(dihedral_deg)
    direction = -np.cos(theta) * bc + np.sin(theta) * (np.cos(phi) * m + np.sin(phi) * n)
    return c + float(length) * direction


def reconstruct_downstream_ca(segment: PeptideSegment, scanned_omega_deg: float) -> np.ndarray:
    """Reconstruct downstream C-alpha for one segment at a scanned omega value."""
    model_n_j = point_from_internal(
        segment.n_i,
        segment.ca_i,
        segment.c_i,
        segment.c_n_length_A,
        segment.ca_c_n_angle_deg,
        segment.parent_psi_deg,
    )
    return point_from_internal(
        segment.ca_i,
        segment.c_i,
        model_n_j,
        segment.n_ca_length_A,
        segment.c_n_ca_angle_deg,
        scanned_omega_deg,
    )


def parent_theta(residues: list[Residue], index: int) -> float:
    """Return parent peptide-plane theta for adjacent plane normals if available."""
    if index + 2 >= len(residues):
        return float("nan")
    first = peptide_plane_normal(residues[index], residues[index + 1])
    second = peptide_plane_normal(residues[index + 1], residues[index + 2])
    if first is None or second is None:
        return float("nan")
    return angle_between_vectors(first, second)


def build_segments(by_chain: dict[str, list[Residue]]) -> list[PeptideSegment]:
    """Extract analyzable consecutive peptide segments from parsed residues."""
    segments: list[PeptideSegment] = []
    required = {"N", "CA", "C"}
    for chain, residues in sorted(by_chain.items()):
        class_label = class_for_chain(chain)
        for index, (res_i, res_j) in enumerate(zip(residues, residues[1:])):
            if not required.issubset(res_i.atoms) or not {"N", "CA"}.issubset(res_j.atoms):
                continue
            parent_phi = float("nan")
            if index > 0 and "C" in residues[index - 1].atoms:
                parent_phi = dihedral_degrees(residues[index - 1].atoms["C"], res_i.atoms["N"], res_i.atoms["CA"], res_i.atoms["C"])
            parent_psi = dihedral_degrees(res_i.atoms["N"], res_i.atoms["CA"], res_i.atoms["C"], res_j.atoms["N"])
            parent_omega = dihedral_degrees(res_i.atoms["CA"], res_i.atoms["C"], res_j.atoms["N"], res_j.atoms["CA"])
            segments.append(
                PeptideSegment(
                    chain=chain,
                    class_label=class_label,
                    segment_index=index + 1,
                    residue_pair=f"{res_i.resname}{res_i.resseq}->{res_j.resname}{res_j.resseq}",
                    res_i_index=res_i.resseq,
                    res_j_index=res_j.resseq,
                    n_i=res_i.atoms["N"],
                    ca_i=res_i.atoms["CA"],
                    c_i=res_i.atoms["C"],
                    n_j=res_j.atoms["N"],
                    ca_j=res_j.atoms["CA"],
                    parent_phi_deg=parent_phi,
                    parent_psi_deg=parent_psi,
                    parent_omega_deg=parent_omega,
                    parent_theta_deg=parent_theta(residues, index),
                    c_n_length_A=distance(res_i.atoms["C"], res_j.atoms["N"]),
                    n_ca_length_A=distance(res_j.atoms["N"], res_j.atoms["CA"]),
                    ca_c_n_angle_deg=angle_degrees(res_i.atoms["CA"], res_i.atoms["C"], res_j.atoms["N"]),
                    c_n_ca_angle_deg=angle_degrees(res_i.atoms["C"], res_j.atoms["N"], res_j.atoms["CA"]),
                )
            )
    return segments


def scan_segment(segment: PeptideSegment, omega_scan: list[float] | None = None) -> list[dict[str, object]]:
    """Run omega endpoint scan for one segment."""
    scan_values = OMEGA_SCAN_DEG if omega_scan is None else omega_scan
    rows: list[dict[str, object]] = []
    parent_distance = distance(segment.ca_i, segment.ca_j)
    for omega in scan_values:
        model_ca_j = reconstruct_downstream_ca(segment, omega)
        model_distance = distance(segment.ca_i, model_ca_j)
        residual = distance(model_ca_j, segment.ca_j)
        deviation = trans_deviation_deg(omega)
        rows.append(
            {
                "chain": segment.chain,
                "class_label": segment.class_label,
                "segment_index": segment.segment_index,
                "segment_id": f"{segment.chain}:{segment.segment_index}:{segment.residue_pair}",
                "residue_pair": segment.residue_pair,
                "res_i": segment.res_i_index,
                "res_j": segment.res_j_index,
                "parent_omega_deg": segment.parent_omega_deg,
                "scanned_omega_deg": omega,
                "omega_trans_deviation_deg": deviation,
                "omega_window_class": omega_window_class(deviation, value_is_deviation=True),
                "endpoint_distance_parent_A": parent_distance,
                "endpoint_distance_model_A": model_distance,
                "closure_residual_A": residual,
                "closure_class": closure_class(residual),
                "compatible_within_0p10_A": residual <= 0.10,
                "compatible_within_0p25_A": residual <= 0.25,
                "parent_theta_deg": segment.parent_theta_deg,
                "parent_phi_deg": segment.parent_phi_deg,
                "parent_psi_deg": segment.parent_psi_deg,
                "parent_omega_window_class": omega_window_class(segment.parent_omega_deg),
            }
        )
    return rows


def scan_segments(segments: list[PeptideSegment], omega_scan: list[float] | None = None) -> pd.DataFrame:
    """Run omega endpoint scan for all segments."""
    rows = [row for segment in segments for row in scan_segment(segment, omega_scan)]
    return pd.DataFrame(rows)


def summarize_group(scan: pd.DataFrame, group_name: str, group: str) -> dict[str, object]:
    """Summarize closure scan rows for one group."""
    parent = group.drop_duplicates("segment_id")
    within8 = group[group["omega_window_class"] == "within_8deg"]
    within10 = group[group["omega_window_class"].isin(["within_8deg", "within_10deg"])]
    best = group.sort_values(["closure_residual_A", "omega_trans_deviation_deg"]).iloc[0] if not group.empty else None
    parent_devs = parent["parent_omega_deg"].map(trans_deviation_deg).tolist()
    pattern = detect_every_other_pattern(parent_devs)
    return {
        "row_type": "summary",
        "group": group_name,
        "class_label": group_name if group_name in {"triketo_cyanuric_like", "triamino_melamine_like"} else "all",
        "analyzable_segment_count": int(parent["segment_id"].nunique()),
        "parent_omega_median_deg": float(parent["parent_omega_deg"].median()) if not parent.empty else float("nan"),
        "parent_omega_trans_deviation_median_deg": float(pd.Series(parent_devs).median()) if parent_devs else float("nan"),
        "parent_within_8deg_count": int(sum(value <= 8.0 for value in parent_devs)),
        "parent_within_10deg_count": int(sum(value <= 10.0 for value in parent_devs)),
        "parent_outside_10deg_count": int(sum(value > 10.0 for value in parent_devs)),
        "parent_every_other_detected": pattern["every_other_detected"],
        "best_scanned_omega_deg": float(best["scanned_omega_deg"]) if best is not None else float("nan"),
        "best_closure_residual_A": float(best["closure_residual_A"]) if best is not None else float("nan"),
        "good_closure_count_all_scan": int((group["closure_class"] == "good_closure").sum()),
        "borderline_or_good_count_all_scan": int((group["compatible_within_0p25_A"]).sum()),
        "good_closure_count_within_8deg_scan": int((within8["closure_class"] == "good_closure").sum()),
        "good_closure_count_within_10deg_scan": int((within10["closure_class"] == "good_closure").sum()),
        "segments_with_good_closure_within_8deg": int(within8[within8["closure_class"] == "good_closure"]["segment_id"].nunique()),
        "segments_with_good_closure_within_10deg": int(within10[within10["closure_class"] == "good_closure"]["segment_id"].nunique()),
        "notes": "endpoint closure surrogate with parent psi and explicit scanned omega",
    }


def summarize_scan(scan: pd.DataFrame) -> pd.DataFrame:
    """Build summary rows for all chains and class-separated subsets."""
    if scan.empty:
        return pd.DataFrame(
            [
                {
                    "row_type": "summary",
                    "group": "all_segments",
                    "analyzable_segment_count": 0,
                    "notes": "no analyzable segments",
                }
            ]
        )
    rows = [summarize_group(scan, "all_segments", scan)]
    for class_label in ["triketo_cyanuric_like", "triamino_melamine_like"]:
        subset = scan[scan["class_label"] == class_label]
        rows.append(summarize_group(subset, class_label, subset))
    for chain, subset in sorted(scan.groupby("chain")):
        rows.append(summarize_group(subset, f"chain_{chain}", subset))
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, columns: list[str], limit: int = 20) -> str:
    """Render a compact markdown table."""
    if df.empty:
        return "_None._"
    table = df.loc[:, [column for column in columns if column in df.columns]].head(limit).copy()
    for column in table.columns:
        table[column] = table[column].map(lambda value: "" if pd.isna(value) else value)
    header = "| " + " | ".join(table.columns) + " |"
    sep = "| " + " | ".join("---" for _ in table.columns) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in table.to_numpy()]
    if len(df) > limit:
        rows.append(f"| ... | {len(df) - limit} additional rows omitted |" + " |" * max(0, len(table.columns) - 2))
    return "\n".join([header, sep, *rows])


def build_report(scan: pd.DataFrame, summary: pd.DataFrame) -> str:
    """Build markdown report."""
    all_row = summary[summary["group"] == "all_segments"].iloc[0]
    tri_row = summary[summary["group"] == "triamino_melamine_like"].iloc[0]
    cy_row = summary[summary["group"] == "triketo_cyanuric_like"].iloc[0]
    feasible_8 = int(all_row["segments_with_good_closure_within_8deg"])
    feasible_10 = int(all_row["segments_with_good_closure_within_10deg"])
    total = int(all_row["analyzable_segment_count"])
    class_delta = abs(float(tri_row["segments_with_good_closure_within_10deg"]) - float(cy_row["segments_with_good_closure_within_10deg"]))
    class_note = (
        "A class-specific difference appears in the closure counts."
        if class_delta > 0
        else "No class-specific difference appears in this first closure-count surrogate."
    )
    every_other_note = (
        "Parent every-other omega behavior is detected in the all-segment summary and should be treated as a possible builder artifact."
        if bool(all_row["parent_every_other_detected"])
        else "Parent every-other omega behavior is not detected globally, although class-specific summaries should still be inspected for possible builder artifacts."
    )

    return f"""# Internal-Coordinate Endpoint Closure Prototype

This is a minimal endpoint-closure prototype, not a final structure. It is testing whether controlled omega variation can satisfy fixed endpoint geometry using a transparent local internal-coordinate surrogate.

The prototype is motivated by Asem's pNAB limitation and Nick's concern about every-other peptide non-planarity. Existing conservative coordinate scans monitored omega but did not rotate omega as a controlled degree of freedom. Success here means finding feasible endpoint closure under realistic omega constraints, not proving the physical hexaplex structure.

## Method

For each analyzable consecutive peptide segment, the upstream `N-CA-C` seed, parent bond lengths, parent bond angles, and parent psi are kept fixed. The downstream `N` and `CA` endpoint are reconstructed while scanning omega values around trans: -180, -175, -172, -170, -168, -165, and -160 degrees. The model endpoint is compared with the parent downstream C-alpha endpoint.

Closure thresholds:

- good_closure: <= 0.10 A
- borderline_closure: <= 0.25 A
- poor_closure: > 0.25 A

Omega windows:

- within +/- 8 deg
- within +/- 10 deg
- outside +/- 10 deg

## Key Results

- Analyzable peptide segments: {total}
- Parent omega median: {float(all_row['parent_omega_median_deg']):.3f} deg
- Parent median omega deviation from trans: {float(all_row['parent_omega_trans_deviation_median_deg']):.3f} deg
- Parent omega values within +/- 8 deg: {int(all_row['parent_within_8deg_count'])}/{total}
- Parent omega values within +/- 10 deg: {int(all_row['parent_within_10deg_count'])}/{total}
- Segments with good closure using scanned omega within +/- 8 deg: {feasible_8}/{total}
- Segments with good closure using scanned omega within +/- 10 deg: {feasible_10}/{total}
- {every_other_note}
- {class_note}

{markdown_table(summary, ['group', 'analyzable_segment_count', 'parent_omega_median_deg', 'parent_omega_trans_deviation_median_deg', 'parent_within_8deg_count', 'parent_within_10deg_count', 'parent_outside_10deg_count', 'parent_every_other_detected', 'segments_with_good_closure_within_8deg', 'segments_with_good_closure_within_10deg', 'best_scanned_omega_deg', 'best_closure_residual_A'], limit=12)}

## Interpretation

This first surrogate directly tests omega as a scanned degree of freedom, but it still holds parent psi and local bond geometry fixed. If closure is feasible inside +/- 8 to +/- 10 degrees, that supports proceeding toward an external two-class peptide-backbone builder. If closure fails for many segments inside those windows, the next builder must scan or optimize phi/psi and class-specific exit-vector geometry along with omega.

## Next Implementation Step

Build a slightly richer internal-coordinate fixture that solves phi/psi/omega together against fixed class-specific exit points, while preserving the recognition core/register and rejecting systematic every-other omega artifacts. Only after that closure fixture behaves well should a full atomistic hexaplex rebuild be attempted.

## Outputs

- Scan CSV: `outputs/metrics/internal_coordinate_endpoint_closure_scan.csv`
- Summary CSV: `outputs/metrics/internal_coordinate_endpoint_closure_summary.csv`
- Report: `outputs/reports/internal_coordinate_endpoint_closure_report.md`
"""


def run_closure(
    parent_pdb: Path = DEFAULT_PARENT_PDB,
    scan_csv: Path = DEFAULT_SCAN_CSV,
    summary_csv: Path = DEFAULT_SUMMARY_CSV,
    report_path: Path = DEFAULT_REPORT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run endpoint closure scan and write outputs."""
    by_chain = parse_residues(parent_pdb)
    segments = build_segments(by_chain)
    scan = scan_segments(segments)
    summary = summarize_scan(scan)
    scan_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    scan.to_csv(scan_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    report_path.write_text(build_report(scan, summary), encoding="utf-8")
    return scan, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-pdb", type=Path, default=DEFAULT_PARENT_PDB)
    parser.add_argument("--scan-csv", type=Path, default=DEFAULT_SCAN_CSV)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scan, summary = run_closure(args.parent_pdb, args.scan_csv, args.summary_csv, args.report)
    all_row = summary[summary["group"] == "all_segments"].iloc[0]
    print(f"Analyzable segments: {int(all_row['analyzable_segment_count'])}")
    print(f"Good closure within +/-8 deg: {int(all_row['segments_with_good_closure_within_8deg'])}")
    print(f"Good closure within +/-10 deg: {int(all_row['segments_with_good_closure_within_10deg'])}")
    print(f"Wrote {args.scan_csv}")
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
