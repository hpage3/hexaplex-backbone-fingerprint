"""Analyze peptide/backbone geometry by class-separated three-fold families.

This is an analysis-only diagnostic motivated by Asem's symmetry critique. It
does not build new coordinates and should not be interpreted as a new atomistic
reconstruction.
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

from hexaplex_backbone_fingerprint.geometry import angle_between_vectors, dihedral_degrees
from scripts.analyze_threefold_backbone_symmetry import (
    Residue,
    angle360,
    circular_gaps_deg,
    mean_exit_vector,
    parse_residues,
    peptide_plane_normal,
    symmetry_gap_rms_deg,
)
from scripts.run_parent_derived_rise_bridge import DEFAULT_PARENT_PDB, markdown_table


DEFAULT_METRICS = Path("outputs/metrics/class_separated_peptide_geometry_summary.csv")
DEFAULT_REPORT = Path("outputs/reports/class_separated_peptide_geometry_report.md")

TRIKETO_CHAINS = {"A", "C", "E"}
TRIAMINO_CHAINS = {"B", "D", "F"}


@dataclass(frozen=True)
class ChainGeometry:
    """Per-chain peptide/backbone geometry summary."""

    chain: str
    backbone_class: str
    residue_names: str
    ca_count: int
    omega_median_deg: float
    omega_trans_deviation_median_deg: float
    theta_median_deg: float
    theta_std_deg: float
    ca_rise_median_A: float
    ca_rise_std_A: float
    exit_vector_xy_angle_deg: float
    radial_angle_deg: float
    radial_radius_A: float
    interstrand_nn_ca_median_A: float


def class_for_chain(chain: str) -> str:
    """Return class assignment from the three-fold diagnostic."""
    if chain in TRIKETO_CHAINS:
        return "triketo_cyanuric_like"
    if chain in TRIAMINO_CHAINS:
        return "triamino_melamine_like"
    return "unclassified"


def finite_median(values: list[float]) -> float:
    """Return median for finite values or NaN."""
    arr = np.array([value for value in values if np.isfinite(value)], dtype=float)
    return float(np.median(arr)) if len(arr) else float("nan")


def finite_std(values: list[float]) -> float:
    """Return population std for finite values or NaN."""
    arr = np.array([value for value in values if np.isfinite(value)], dtype=float)
    return float(np.std(arr)) if len(arr) else float("nan")


def ca_coordinates(residues: list[Residue]) -> list[np.ndarray]:
    """Return C-alpha coordinates in coordinate order."""
    return [res.atoms["CA"] for res in residues if "CA" in res.atoms]


def ca_rise_values(residues: list[Residue]) -> list[float]:
    """Return absolute adjacent C-alpha z-step values."""
    cas = ca_coordinates(residues)
    return [abs(float(b[2] - a[2])) for a, b in zip(cas, cas[1:])]


def omega_values(residues: list[Residue]) -> list[float]:
    """Return peptide omega values where definable."""
    values = []
    for res_i, res_j in zip(residues, residues[1:]):
        if {"CA", "C"}.issubset(res_i.atoms) and {"N", "CA"}.issubset(res_j.atoms):
            values.append(dihedral_degrees(res_i.atoms["CA"], res_i.atoms["C"], res_j.atoms["N"], res_j.atoms["CA"]))
    return values


def theta_values(residues: list[Residue]) -> list[float]:
    """Return unsigned peptide-plane theta values along one chain."""
    normals = []
    for res_i, res_j in zip(residues, residues[1:]):
        normal = peptide_plane_normal(res_i, res_j)
        if normal is not None:
            normals.append(normal)
    return [angle_between_vectors(a, b) for a, b in zip(normals, normals[1:])]


def interstrand_nn_ca_values(chain: str, by_chain: dict[str, list[Residue]]) -> list[float]:
    """Return nearest C-alpha distances from one chain to any other chain."""
    own = ca_coordinates(by_chain[chain])
    other = [coord for other_chain, residues in by_chain.items() if other_chain != chain for coord in ca_coordinates(residues)]
    if not own or not other:
        return []
    other_arr = np.array(other, dtype=float)
    values = []
    for coord in own:
        distances = np.linalg.norm(other_arr - coord, axis=1)
        values.append(float(np.min(distances)))
    return values


def chain_geometry_rows(by_chain: dict[str, list[Residue]]) -> list[ChainGeometry]:
    """Compute per-chain peptide/backbone geometry rows."""
    all_cas = [coord for residues in by_chain.values() for coord in ca_coordinates(residues)]
    if not all_cas:
        raise ValueError("No C-alpha atoms found; cannot analyze peptide geometry.")
    center_xy = np.array(all_cas, dtype=float)[:, :2].mean(axis=0)
    rows = []
    for chain, residues in sorted(by_chain.items()):
        cas = np.array(ca_coordinates(residues), dtype=float)
        centroid = cas.mean(axis=0)
        radial_vec = centroid[:2] - center_xy
        exit_vector = mean_exit_vector(residues)
        exit_angle = (
            angle360(exit_vector[:2])
            if exit_vector is not None and np.linalg.norm(exit_vector[:2]) > 1e-12
            else float("nan")
        )
        omegas = omega_values(residues)
        omega_devs = [abs(abs(value) - 180.0) for value in omegas]
        thetas = theta_values(residues)
        rises = ca_rise_values(residues)
        nn = interstrand_nn_ca_values(chain, by_chain)
        rows.append(
            ChainGeometry(
                chain=chain,
                backbone_class=class_for_chain(chain),
                residue_names=",".join(sorted({res.resname for res in residues})),
                ca_count=len(cas),
                omega_median_deg=finite_median(omegas),
                omega_trans_deviation_median_deg=finite_median(omega_devs),
                theta_median_deg=finite_median(thetas),
                theta_std_deg=finite_std(thetas),
                ca_rise_median_A=finite_median(rises),
                ca_rise_std_A=finite_std(rises),
                exit_vector_xy_angle_deg=exit_angle,
                radial_angle_deg=angle360(radial_vec),
                radial_radius_A=float(np.linalg.norm(radial_vec)),
                interstrand_nn_ca_median_A=finite_median(nn),
            )
        )
    return rows


def summarize_group(model_id: str, group_name: str, rows: list[ChainGeometry]) -> dict[str, object]:
    """Summarize pooled/all or class-specific chain rows."""
    exit_angles = [row.exit_vector_xy_angle_deg for row in rows if np.isfinite(row.exit_vector_xy_angle_deg)]
    radial_angles = [row.radial_angle_deg for row in rows if np.isfinite(row.radial_angle_deg)]
    ideal_gap = 60.0 if group_name == "all_six_chains" else 120.0
    return {
        "model_id": model_id,
        "row_type": "summary",
        "group": group_name,
        "chains": ",".join(row.chain for row in rows),
        "chain_count": len(rows),
        "residue_name_sets": "; ".join(f"{row.chain}:{row.residue_names}" for row in rows),
        "omega_median_deg": finite_median([row.omega_median_deg for row in rows]),
        "omega_trans_deviation_median_deg": finite_median([row.omega_trans_deviation_median_deg for row in rows]),
        "theta_median_deg": finite_median([row.theta_median_deg for row in rows]),
        "theta_std_deg": finite_median([row.theta_std_deg for row in rows]),
        "ca_rise_median_A": finite_median([row.ca_rise_median_A for row in rows]),
        "ca_rise_std_A": finite_median([row.ca_rise_std_A for row in rows]),
        "exit_vector_angle_gap_rms_deg": symmetry_gap_rms_deg(exit_angles, ideal_gap),
        "exit_vector_angle_gaps_deg": ";".join(f"{value:.3f}" for value in circular_gaps_deg(exit_angles)),
        "radial_angle_gap_rms_deg": symmetry_gap_rms_deg(radial_angles, ideal_gap),
        "radial_radius_median_A": finite_median([row.radial_radius_A for row in rows]),
        "radial_radius_std_A": finite_std([row.radial_radius_A for row in rows]),
        "interstrand_nn_ca_median_A": finite_median([row.interstrand_nn_ca_median_A for row in rows]),
        "notes": "all chains pooled" if group_name == "all_six_chains" else "class-separated three-fold family",
    }


def difference_row(model_id: str, triamino: dict[str, object], triketo: dict[str, object]) -> dict[str, object]:
    """Return triamino-minus-triketo difference row for comparable metrics."""
    metric_names = [
        "omega_median_deg",
        "omega_trans_deviation_median_deg",
        "theta_median_deg",
        "theta_std_deg",
        "ca_rise_median_A",
        "ca_rise_std_A",
        "exit_vector_angle_gap_rms_deg",
        "radial_angle_gap_rms_deg",
        "radial_radius_median_A",
        "interstrand_nn_ca_median_A",
    ]
    row: dict[str, object] = {
        "model_id": model_id,
        "row_type": "difference",
        "group": "triamino_minus_triketo",
        "chains": f"{triamino['chains']} minus {triketo['chains']}",
        "chain_count": "",
        "notes": "positive values mean triamino/melamine-like exceeds triketo/cyanuric-like",
    }
    for name in metric_names:
        row[name] = float(triamino[name]) - float(triketo[name])
    return row


def summary_table(model_id: str, chain_rows: list[ChainGeometry]) -> pd.DataFrame:
    """Return class-separated peptide geometry summary table."""
    groups = {
        "all_six_chains": chain_rows,
        "triketo_cyanuric_like": [row for row in chain_rows if row.backbone_class == "triketo_cyanuric_like"],
        "triamino_melamine_like": [row for row in chain_rows if row.backbone_class == "triamino_melamine_like"],
    }
    summary_rows = [summarize_group(model_id, group, rows) for group, rows in groups.items() if rows]
    lookup = {row["group"]: row for row in summary_rows}
    if "triketo_cyanuric_like" in lookup and "triamino_melamine_like" in lookup:
        summary_rows.append(difference_row(model_id, lookup["triamino_melamine_like"], lookup["triketo_cyanuric_like"]))
    chain_dicts = [
        {
            "model_id": model_id,
            "row_type": "chain",
            "group": row.backbone_class,
            "chains": row.chain,
            "chain_count": 1,
            "residue_name_sets": row.residue_names,
            "omega_median_deg": row.omega_median_deg,
            "omega_trans_deviation_median_deg": row.omega_trans_deviation_median_deg,
            "theta_median_deg": row.theta_median_deg,
            "theta_std_deg": row.theta_std_deg,
            "ca_rise_median_A": row.ca_rise_median_A,
            "ca_rise_std_A": row.ca_rise_std_A,
            "exit_vector_xy_angle_deg": row.exit_vector_xy_angle_deg,
            "radial_angle_deg": row.radial_angle_deg,
            "radial_radius_median_A": row.radial_radius_A,
            "interstrand_nn_ca_median_A": row.interstrand_nn_ca_median_A,
            "notes": "per-chain diagnostic row",
        }
        for row in chain_rows
    ]
    return pd.DataFrame(summary_rows + chain_dicts)


def class_distinguishability(summary: pd.DataFrame) -> dict[str, object]:
    """Return compact interpretation values from the summary table."""
    difference = summary[(summary["row_type"] == "difference") & (summary["group"] == "triamino_minus_triketo")]
    if difference.empty:
        return {"has_difference": False, "max_abs_difference": float("nan"), "most_distinct_metric": ""}
    row = difference.iloc[0]
    metric_names = [
        "omega_trans_deviation_median_deg",
        "theta_median_deg",
        "ca_rise_median_A",
        "exit_vector_angle_gap_rms_deg",
        "radial_radius_median_A",
        "interstrand_nn_ca_median_A",
    ]
    values = {name: abs(float(row[name])) for name in metric_names if name in row and pd.notna(row[name])}
    if not values:
        return {"has_difference": True, "max_abs_difference": float("nan"), "most_distinct_metric": ""}
    metric = max(values, key=values.get)
    return {"has_difference": True, "max_abs_difference": values[metric], "most_distinct_metric": metric}


def build_report_text(parent_pdb: Path, summary: pd.DataFrame) -> str:
    """Build markdown report for class-separated peptide geometry."""
    summaries = summary[summary["row_type"].isin(["summary", "difference"])]
    distinguish = class_distinguishability(summary)
    table = markdown_table(
        summaries,
        [
            "group",
            "chains",
            "omega_median_deg",
            "omega_trans_deviation_median_deg",
            "theta_median_deg",
            "ca_rise_median_A",
            "exit_vector_angle_gap_rms_deg",
            "radial_angle_gap_rms_deg",
            "radial_radius_median_A",
            "interstrand_nn_ca_median_A",
            "notes",
        ],
    )
    chain_table = markdown_table(
        summary[summary["row_type"] == "chain"],
        [
            "chains",
            "group",
            "omega_median_deg",
            "omega_trans_deviation_median_deg",
            "theta_median_deg",
            "ca_rise_median_A",
            "exit_vector_xy_angle_deg",
            "radial_angle_deg",
            "radial_radius_median_A",
            "interstrand_nn_ca_median_A",
        ],
    )
    support = (
        "Yes, as a diagnostic next step: class-separated chain placement is already distinct, and the table quantifies whether peptide/backbone metrics differ enough to parameterize separately."
        if distinguish["has_difference"]
        else "Not yet; class differences could not be quantified from the available rows."
    )
    return f"""# Class-Separated Peptide Geometry Diagnostic

## Scope

This is a diagnostic analysis, not a new atomistic reconstruction and not a coordinate-generation step. It uses the same parent/reference coordinate model used by the parent-derived bridge and three-fold symmetry diagnostics:

`{parent_pdb}`

The class assignment follows the three-fold diagnostic:

- Triketo/cyanuric-like chains: A,C,E
- Triamino/melamine-like chains: B,D,F

## Summary

{table}

## Per-Chain Values

{chain_table}

## Interpretation Questions

- Are the two three-fold classes distinguishable in peptide/backbone geometry, not just chain placement? The difference row reports triamino-minus-triketo values. The largest absolute difference among the tracked peptide/backbone metrics is `{distinguish['most_distinct_metric']}` = {float(distinguish['max_abs_difference']):.4g}.
- Is there evidence of systematic alternating peptide-plane or omega behavior? Compare `theta_median_deg`, `omega_median_deg`, and `omega_trans_deviation_median_deg` between the class rows. Differences are diagnostic evidence for separate treatment, not final chemistry.
- Which class appears more strained or more distorted? The class with larger omega trans deviation, theta spread, or C-alpha rise spread is the more distorted class by this diagnostic.
- Does the result support building a two-class three-fold coordinate prototype? {support}
- What model degrees of freedom should be tested first? Start with class-specific exit-vector orientation, class-specific peptide-plane normal/theta, and class-specific axial/rise placement. Keep pNAB/YAML provenance separate from these controlled peptide-plane model tests.

## Output Files

- Metrics CSV: `outputs/metrics/class_separated_peptide_geometry_summary.csv`
- Report: `outputs/reports/class_separated_peptide_geometry_report.md`
"""


def run_analysis(parent_pdb: Path, metrics_path: Path, report_path: Path) -> pd.DataFrame:
    """Run class-separated peptide geometry analysis."""
    residues = parse_residues(parent_pdb)
    rows = chain_geometry_rows(residues)
    summary = summary_table(parent_pdb.stem, rows)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(metrics_path, index=False)
    report_path.write_text(build_report_text(parent_pdb, summary), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-pdb", type=Path, default=DEFAULT_PARENT_PDB)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_analysis(args.parent_pdb, args.metrics, args.report)
    print(f"Wrote {len(summary)} class-separated peptide geometry rows")
    print(f"Metrics: {args.metrics}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
