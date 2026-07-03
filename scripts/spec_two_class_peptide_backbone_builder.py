"""Specify a two-class peptide-backbone builder feasibility path.

This is a specification / feasibility audit, not a final structure builder.
It summarizes current parent-coordinate omega behavior, prior diagnostic
outputs, and the code/data inputs needed to build peptide backbones explicitly
outside pNAB's current peptide-omega limitation.
"""

from __future__ import annotations

import argparse
import math
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

from hexaplex_backbone_fingerprint.geometry import dihedral_degrees
from scripts.analyze_threefold_backbone_symmetry import parse_residues
from scripts.run_parent_derived_rise_bridge import DEFAULT_PARENT_PDB


DEFAULT_SUMMARY_CSV = Path("outputs/metrics/two_class_peptide_backbone_builder_spec_summary.csv")
DEFAULT_REPORT = Path("outputs/reports/two_class_peptide_backbone_builder_spec.md")

TRIKETO_CHAINS = {"A", "C", "E"}
TRIAMINO_CHAINS = {"B", "D", "F"}

PRIOR_OUTPUTS = {
    "threefold_symmetry_metrics": Path("outputs/metrics/threefold_backbone_symmetry_summary.csv"),
    "class_separated_geometry_metrics": Path("outputs/metrics/class_separated_peptide_geometry_summary.csv"),
    "two_class_peptide_plane_scores": Path("outputs/metrics/two_class_peptide_plane_prototype_scores.csv"),
    "two_class_backbone_anchor_scores": Path("outputs/metrics/two_class_backbone_anchor_scan_scores.csv"),
    "two_class_axial_theta_scores": Path("outputs/metrics/two_class_axial_theta_scan_scores.csv"),
    "parent_derived_fine_scan_scores": Path("outputs/metrics/parent_derived_rise_fine_scan_abcd_scores.csv"),
}


def trans_deviation_deg(omega_deg: float) -> float:
    """Return angular deviation from trans, handling +/-180 wraparound."""
    if omega_deg is None or not np.isfinite(float(omega_deg)):
        return float("nan")
    return abs(abs(float(omega_deg)) - 180.0)


def omega_window_class(deviation_deg: float) -> str:
    """Classify one omega trans deviation."""
    if deviation_deg is None or not np.isfinite(float(deviation_deg)):
        return "insufficient_data"
    value = float(deviation_deg)
    if value <= 8.0:
        return "within_8deg"
    if value <= 10.0:
        return "within_10deg"
    return "outside_10deg"


def detect_every_other_pattern(deviations: list[float], threshold_deg: float = 10.0) -> dict[str, object]:
    """Detect a simple alternating/every-other high-deviation pattern."""
    finite = [float(value) for value in deviations if np.isfinite(float(value))]
    if len(finite) < 4:
        return {
            "every_other_detected": False,
            "alternating_fraction": float("nan"),
            "even_median_deg": float("nan"),
            "odd_median_deg": float("nan"),
        }
    states = [value > threshold_deg for value in finite]
    transitions = [a != b for a, b in zip(states, states[1:])]
    alternating_fraction = float(sum(transitions) / len(transitions)) if transitions else float("nan")
    even_values = finite[0::2]
    odd_values = finite[1::2]
    even_median = float(np.median(even_values)) if even_values else float("nan")
    odd_median = float(np.median(odd_values)) if odd_values else float("nan")
    median_gap = abs(even_median - odd_median) if np.isfinite(even_median) and np.isfinite(odd_median) else 0.0
    detected = alternating_fraction >= 0.8 and median_gap >= threshold_deg * 0.5
    return {
        "every_other_detected": bool(detected),
        "alternating_fraction": alternating_fraction,
        "even_median_deg": even_median,
        "odd_median_deg": odd_median,
    }


def class_for_chain(chain: str) -> str:
    """Return fixed class assignment from the three-fold diagnostic."""
    if chain in TRIKETO_CHAINS:
        return "triketo_cyanuric_like"
    if chain in TRIAMINO_CHAINS:
        return "triamino_melamine_like"
    return "unclassified"


def omega_rows_from_pdb(path: Path, model_id: str) -> list[dict[str, object]]:
    """Extract per-peptide omega rows from a labeled PDB."""
    by_chain = parse_residues(path)
    rows: list[dict[str, object]] = []
    for chain, residues in sorted(by_chain.items()):
        for order, (res_i, res_j) in enumerate(zip(residues, residues[1:]), start=1):
            if {"CA", "C"}.issubset(res_i.atoms) and {"N", "CA"}.issubset(res_j.atoms):
                omega = dihedral_degrees(res_i.atoms["CA"], res_i.atoms["C"], res_j.atoms["N"], res_j.atoms["CA"])
                deviation = trans_deviation_deg(omega)
                rows.append(
                    {
                        "model_id": model_id,
                        "chain": chain,
                        "class_name": class_for_chain(chain),
                        "peptide_order": order,
                        "residue_pair": f"{res_i.resname}{res_i.resseq}->{res_j.resname}{res_j.resseq}",
                        "omega_deg": omega,
                        "omega_trans_deviation_deg": deviation,
                        "omega_window_class": omega_window_class(deviation),
                    }
                )
    return rows


def summarize_omega_rows(model_id: str, group: str, rows: list[dict[str, object]]) -> dict[str, object]:
    """Summarize omega rows for one model/group."""
    deviations = [float(row["omega_trans_deviation_deg"]) for row in rows if np.isfinite(float(row["omega_trans_deviation_deg"]))]
    omegas = [float(row["omega_deg"]) for row in rows if np.isfinite(float(row["omega_deg"]))]
    pattern = detect_every_other_pattern(deviations)
    total = len(deviations)
    within_8 = sum(value <= 8.0 for value in deviations)
    within_10 = sum(value <= 10.0 for value in deviations)
    outside_10 = sum(value > 10.0 for value in deviations)
    return {
        "row_type": "omega_summary",
        "model_id": model_id,
        "group": group,
        "omega_count": total,
        "omega_mean_deg": float(np.mean(omegas)) if omegas else float("nan"),
        "omega_median_deg": float(np.median(omegas)) if omegas else float("nan"),
        "omega_min_deg": float(np.min(omegas)) if omegas else float("nan"),
        "omega_max_deg": float(np.max(omegas)) if omegas else float("nan"),
        "trans_deviation_mean_deg": float(np.mean(deviations)) if deviations else float("nan"),
        "trans_deviation_median_deg": float(np.median(deviations)) if deviations else float("nan"),
        "trans_deviation_min_deg": float(np.min(deviations)) if deviations else float("nan"),
        "trans_deviation_max_deg": float(np.max(deviations)) if deviations else float("nan"),
        "within_8deg_count": within_8,
        "within_8deg_fraction": within_8 / total if total else float("nan"),
        "within_10deg_count": within_10,
        "within_10deg_fraction": within_10 / total if total else float("nan"),
        "outside_10deg_count": outside_10,
        "outside_10deg_fraction": outside_10 / total if total else float("nan"),
        "every_other_detected": pattern["every_other_detected"],
        "alternating_fraction": pattern["alternating_fraction"],
        "notes": "omega extracted from current labeled coordinate file",
    }


def summarize_parent_omega(parent_pdb: Path = DEFAULT_PARENT_PDB) -> pd.DataFrame:
    """Return parent omega summary rows for all chains and class-separated groups."""
    if not parent_pdb.exists():
        return pd.DataFrame(
            [
                {
                    "row_type": "omega_summary",
                    "model_id": "parent_reference",
                    "group": "all_six_chains",
                    "omega_count": 0,
                    "notes": f"parent PDB not found: {parent_pdb}",
                }
            ]
        )
    rows = omega_rows_from_pdb(parent_pdb, "parent_reference")
    summaries = [summarize_omega_rows("parent_reference", "all_six_chains", rows)]
    for class_name in ["triketo_cyanuric_like", "triamino_melamine_like"]:
        class_rows = [row for row in rows if row["class_name"] == class_name]
        summaries.append(summarize_omega_rows("parent_reference", class_name, class_rows))
    return pd.DataFrame(summaries)


def file_presence_rows(root: Path = ROOT) -> list[dict[str, object]]:
    """Summarize presence of prior outputs and likely pNAB/YAML inputs."""
    rows: list[dict[str, object]] = []
    for label, path in PRIOR_OUTPUTS.items():
        rows.append(
            {
                "row_type": "file_presence",
                "model_id": label,
                "group": "prior_output",
                "path": str(path),
                "present": path.exists(),
                "notes": "available prior diagnostic output" if path.exists() else "not found",
            }
        )

    pnab_like: list[Path] = []
    for pattern in ("*pnab*", "*.yaml", "*.yml", "*builder*", "*monomer*", "*building*block*"):
        pnab_like.extend(path for path in root.rglob(pattern) if ".git" not in path.parts and ".venv" not in path.parts)
    seen = set()
    for path in sorted(pnab_like):
        key = str(path)
        if key in seen or path.is_dir():
            continue
        if "scripts" in path.parts or "tests" in path.parts:
            continue
        suffix = path.suffix.lower()
        is_yaml = suffix in {".yaml", ".yml"}
        is_source_or_text = suffix in {".py", ".txt", ".md", ".json"}
        is_coordinate = suffix in {".pdb", ".xyz"}
        if not (is_yaml or is_source_or_text or is_coordinate):
            continue
        if "outputs" in path.parts and not (is_yaml or is_coordinate):
            continue
        seen.add(key)
        kind = "yaml_input" if is_yaml else "coordinate_or_model" if is_coordinate else "source_or_text"
        rows.append(
            {
                "row_type": "file_presence",
                "model_id": "pnab_or_builder_input",
                "group": "input_inventory",
                "path": str(path.relative_to(root)),
                "present": True,
                "notes": f"possible pNAB/YAML/building-block evidence ({kind}); inspect before treating as provenance",
            }
        )
    if not any(row["group"] == "input_inventory" for row in rows):
        rows.append(
            {
                "row_type": "file_presence",
                "model_id": "pnab_or_builder_input",
                "group": "input_inventory",
                "path": "",
                "present": False,
                "notes": "no pNAB/YAML/building-block inputs found by conservative filename scan",
            }
        )
    return rows


def scan_rotation_summary_rows() -> list[dict[str, object]]:
    """Summarize whether recent scans rotated omega or monitored it."""
    return [
        {
            "row_type": "scan_scope",
            "model_id": "two_class_peptide_plane_prototype",
            "group": "prior_scan",
            "rotated_omega": False,
            "notes": "conservative N/C/O peptide-plane perturbation monitored omega/theta; it did not implement omega as a controlled rotatable degree of freedom",
        },
        {
            "row_type": "scan_scope",
            "model_id": "two_class_backbone_anchor_scan",
            "group": "prior_scan",
            "rotated_omega": False,
            "notes": "N/CA/C/O radial backbone-anchor scan monitored omega/theta after coordinate shifts; omega was not independently scanned",
        },
        {
            "row_type": "scan_scope",
            "model_id": "two_class_axial_theta_scan",
            "group": "prior_scan",
            "rotated_omega": False,
            "notes": "N/CA/C/O axial-offset scan monitored omega/theta after coordinate shifts; omega was not independently scanned",
        },
        {
            "row_type": "builder_spec",
            "model_id": "external_two_class_peptide_backbone_builder",
            "group": "recommended_next_step",
            "rotated_omega": True,
            "notes": "minimal internal-coordinate fixture should scan omega explicitly within realistic trans windows before a full atomistic hexaplex rebuild",
        },
    ]


def build_summary_table(parent_pdb: Path = DEFAULT_PARENT_PDB) -> pd.DataFrame:
    """Build the complete specification summary table."""
    frames = [summarize_parent_omega(parent_pdb)]
    frames.append(pd.DataFrame(file_presence_rows()))
    frames.append(pd.DataFrame(scan_rotation_summary_rows()))
    return pd.concat(frames, ignore_index=True, sort=False)


def markdown_table(df: pd.DataFrame, columns: list[str], limit: int = 20) -> str:
    """Render a compact markdown table."""
    if df.empty:
        return "_None._"
    table = df.loc[:, [col for col in columns if col in df.columns]].head(limit).copy()
    for col in table.columns:
        table[col] = table[col].map(lambda value: "" if pd.isna(value) else value)
    header = "| " + " | ".join(table.columns) + " |"
    sep = "| " + " | ".join("---" for _ in table.columns) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in table.to_numpy()]
    if len(df) > limit:
        rows.append(f"| ... | {len(df) - limit} additional rows omitted |" + " |" * max(0, len(table.columns) - 2))
    return "\n".join([header, sep, *rows])


def brief_metric(summary: pd.DataFrame, group: str, column: str) -> str:
    """Return a formatted value from the summary table."""
    rows = summary[(summary["row_type"] == "omega_summary") & (summary["group"] == group)]
    if rows.empty or column not in rows.columns:
        return "not available"
    value = rows.iloc[0][column]
    if pd.isna(value):
        return "not available"
    if isinstance(value, (int, np.integer)):
        return str(value)
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3f}"
    return str(value)


def build_report(summary: pd.DataFrame) -> str:
    """Build the markdown specification report."""
    omega = summary[summary["row_type"] == "omega_summary"].copy()
    file_rows = summary[summary["row_type"] == "file_presence"].copy()
    scan_rows = summary[summary["row_type"] == "scan_scope"].copy()
    builder_rows = summary[summary["row_type"] == "builder_spec"].copy()

    parent_median = brief_metric(summary, "all_six_chains", "omega_median_deg")
    parent_dev = brief_metric(summary, "all_six_chains", "trans_deviation_median_deg")
    within8 = brief_metric(summary, "all_six_chains", "within_8deg_fraction")
    within10 = brief_metric(summary, "all_six_chains", "within_10deg_fraction")
    alternating = brief_metric(summary, "all_six_chains", "every_other_detected")

    return f"""# Two-Class Peptide-Backbone Builder Specification

This is a feasibility/specification audit for an external two-class peptide-backbone builder. It is not a final structure and it does not generate new atomistic coordinates.

## Why This Track Exists

Asem's pNAB update means changing polymer connection atoms alone cannot currently be trusted to fix peptide omega in this system. The peptide dihedral is defined by four atoms, and the current implementation can leave one of those atoms uncontrolled during polymer construction. Existing pNAB-derived structures should therefore be treated as diagnostic scaffolds, not physically decisive peptide-backbone models.

The parent-derived fine rise scan remains a useful diagnostic within a constrained six-fold parent-derived coordinate family, but it is not proof of exact pNAB/YAML provenance and not proof that pNAB determined the physical twist/rise or peptide-backbone geometry.

## Current Parent Omega Summary

- Parent/reference omega median: {parent_median} deg
- Parent/reference median deviation from trans: {parent_dev} deg
- Fraction within +/- 8 deg of trans: {within8}
- Fraction within +/- 10 deg of trans: {within10}
- Every-other omega pattern detected in parent/reference: {alternating}

{markdown_table(omega, ["model_id", "group", "omega_count", "omega_median_deg", "trans_deviation_median_deg", "within_8deg_fraction", "within_10deg_fraction", "outside_10deg_fraction", "every_other_detected"], limit=10)}

## Prior Output / Input Evidence

{markdown_table(file_rows, ["model_id", "group", "present", "path", "notes"], limit=25)}

## What Existing Scans Did

The recent conservative coordinate prototypes were useful negative controls, but they only monitored omega after coordinate perturbation. They did not implement omega as a controlled rotatable degree of freedom.

{markdown_table(scan_rows, ["model_id", "rotated_omega", "notes"], limit=10)}

## Proposed External Builder Design

Build outside the current pNAB limitation using two independent three-fold backbone classes:

- A,C,E: triketo/cyanuric-like class
- B,D,F: triamino/melamine-like class

Keep the recognition-core/register fixed initially. Reconstruct the peptide backbone with explicit internal-coordinate geometry rather than relying on pNAB to determine peptide omega. Treat omega as flexible but constrained near trans: preferred target +/-180 deg, realistic window +/- 8 deg, broader warning window +/- 10 deg. Outside +/- 10 deg should be flagged unless chemically justified.

The builder should distinguish realistic omega flexibility around trans from pNAB-induced every-other omega artifacts. It should also keep diagnostic coordinate transforms separate from physical atomistic reconstruction.

## Builder Controls Needed

The first builder should control or report:

- phi
- psi
- omega
- class-specific exit-vector geometry
- class-specific axial/rise geometry
- class-specific peptide-plane theta
- endpoint closure against fixed class-specific recognition-core anchors

Initial constraints:

- preserve carboxylates
- preserve residue/register labels
- preserve recognition-core placement initially
- keep omega within +/- 8 to +/- 10 deg unless justified
- reject systematic every-other omega artifacts
- report endpoint, bond-length, bond-angle, and peptide-plane deviations

## Recommended Next Implementation Step

Build a minimal internal-coordinate fixture/prototype that connects fixed class-specific exit points with an idealized peptide segment, scans omega in a realistic trans window, and reports whether the endpoints can be satisfied without large omega distortion. Do this before attempting a full atomistic hexaplex rebuild.

{markdown_table(builder_rows, ["model_id", "rotated_omega", "notes"], limit=5)}

## Questions For Asem / Nick

- Which atoms define the chemically meaningful exit points for the triketo/cyanuric-like and triamino/melamine-like classes?
- Which atoms must remain fixed to preserve the recognition core/register?
- Should carboxylate geometry be held rigid during the first internal-coordinate prototype?
- What omega tolerance should be considered chemically acceptable for this system: +/- 8 deg, +/- 10 deg, or a context-specific threshold?
- Are there original pNAB/YAML/building-block inputs that encode the intended monomer connection points, even if they cannot solve omega directly?

## Interpretation

This report supports a modest external two-class peptide-backbone builder specification. It does not revive the failed pseudo reconstructed bridge, does not claim pNAB provenance recovery, and does not claim a final physical structure. The next useful step is a small internal-coordinate closure prototype that explicitly scans omega, phi, and psi against fixed class-specific endpoints.
"""


def run_spec(summary_csv: Path = DEFAULT_SUMMARY_CSV, report_path: Path = DEFAULT_REPORT, parent_pdb: Path = DEFAULT_PARENT_PDB) -> pd.DataFrame:
    """Run the specification audit and write outputs."""
    summary = build_summary_table(parent_pdb)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, index=False)
    report_path.write_text(build_report(summary), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-pdb", type=Path, default=DEFAULT_PARENT_PDB)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_spec(args.summary_csv, args.report, args.parent_pdb)
    omega = summary[summary["row_type"] == "omega_summary"]
    parent = omega[omega["group"] == "all_six_chains"]
    if not parent.empty:
        row = parent.iloc[0]
        print(
            "Parent omega median "
            f"{row.get('omega_median_deg', math.nan):.3f} deg; "
            f"median trans deviation {row.get('trans_deviation_median_deg', math.nan):.3f} deg"
        )
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
