"""Analyze six-fold versus class-separated three-fold backbone symmetry.

This is a model-scope diagnostic for Asem's symmetry critique. It does not
generate new atomistic coordinates and should not be read as pNAB/YAML
provenance recovery.
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

from hexaplex_backbone_fingerprint.geometry import angle_between_vectors, dihedral_degrees, fit_plane
from scripts.generate_global_deformation_variants import parse_pdb_atom_lines
from scripts.run_parent_derived_rise_bridge import DEFAULT_PARENT_PDB, markdown_table


DEFAULT_METRICS = Path("outputs/metrics/threefold_backbone_symmetry_summary.csv")
DEFAULT_REPORT = Path("outputs/reports/threefold_backbone_symmetry_report.md")

MELAMINE_RESNAMES = {"MEP", "MEL", "TAP", "TRIAMINO", "MAM"}
CYANURIC_RESNAMES = {"CYP", "CY", "CYA", "CYANURIC", "TRIKETO"}
BACKBONE_NAMES = {"N", "CA", "C", "O"}


@dataclass(frozen=True)
class Residue:
    """One residue parsed from a PDB in coordinate order."""

    chain: str
    resseq: str
    resname: str
    atoms: dict[str, np.ndarray]


@dataclass(frozen=True)
class ChainFingerprint:
    """Per-chain geometry used by the symmetry summaries."""

    chain: str
    backbone_class: str
    classification_confidence: str
    residue_names: str
    ca_centroid: np.ndarray
    radial_angle_deg: float
    exit_vector: np.ndarray | None
    exit_vector_xy_angle_deg: float | None
    peptide_normal: np.ndarray | None
    peptide_normal_xy_angle_deg: float | None
    theta_median_deg: float | None
    omega_median_deg: float | None
    omega_trans_deviation_median_deg: float | None


def parse_residues(path: Path) -> dict[str, list[Residue]]:
    """Parse PDB residues grouped by chain in coordinate order."""
    _lines, atoms = parse_pdb_atom_lines(path)
    order: list[tuple[str, str]] = []
    residues: dict[tuple[str, str], dict[str, object]] = {}
    for atom in atoms:
        key = (atom.chain, atom.resseq)
        if key not in residues:
            residues[key] = {"resname": atom.resname, "atoms": {}}
            order.append(key)
        residues[key]["atoms"][atom.atom_name] = atom.coord
    by_chain: dict[str, list[Residue]] = {}
    for chain, resseq in order:
        info = residues[(chain, resseq)]
        by_chain.setdefault(chain, []).append(
            Residue(chain=chain, resseq=resseq, resname=str(info["resname"]), atoms=dict(info["atoms"]))
        )
    return by_chain


def classify_chain(residue_names: set[str]) -> tuple[str, str]:
    """Classify a chain as melamine/triamino-like or cyanuric/triketo-like."""
    has_melamine = bool(residue_names & MELAMINE_RESNAMES)
    has_cyanuric = bool(residue_names & CYANURIC_RESNAMES)
    if has_melamine and not has_cyanuric:
        return "triamino_melamine_like", "high"
    if has_cyanuric and not has_melamine:
        return "triketo_cyanuric_like", "high"
    if has_melamine and has_cyanuric:
        return "mixed_or_uncertain", "low"
    return "unclassified", "low"


def angle360(vector_xy: np.ndarray) -> float:
    """Return XY polar angle in degrees in [0, 360)."""
    return float((math.degrees(math.atan2(float(vector_xy[1]), float(vector_xy[0]))) + 360.0) % 360.0)


def circular_gaps_deg(angles_deg: list[float]) -> list[float]:
    """Return sorted circular gaps between angles."""
    if len(angles_deg) < 2:
        return []
    values = sorted(float(angle) % 360.0 for angle in angles_deg)
    return [values[i + 1] - values[i] for i in range(len(values) - 1)] + [values[0] + 360.0 - values[-1]]


def symmetry_gap_rms_deg(angles_deg: list[float], ideal_gap_deg: float) -> float:
    """Return RMS gap deviation from an ideal rotational symmetry gap."""
    gaps = circular_gaps_deg(angles_deg)
    if not gaps:
        return float("nan")
    return float(np.sqrt(np.mean([(gap - ideal_gap_deg) ** 2 for gap in gaps])))


def vector_angle_rms_deg(angles_deg: list[float], ideal_gap_deg: float) -> float:
    """Return RMS deviation of vector angle gaps from an ideal gap."""
    return symmetry_gap_rms_deg([angle for angle in angles_deg if not pd.isna(angle)], ideal_gap_deg)


def ca_atoms(residues: list[Residue]) -> list[np.ndarray]:
    """Return C-alpha coordinates in residue order."""
    return [res.atoms["CA"] for res in residues if "CA" in res.atoms]


def mean_exit_vector(residues: list[Residue]) -> np.ndarray | None:
    """Return mean same-chain C-alpha step vector."""
    cas = ca_atoms(residues)
    if len(cas) < 2:
        return None
    steps = np.diff(np.array(cas, dtype=float), axis=0)
    norms = np.linalg.norm(steps, axis=1)
    steps = steps[norms > 1e-12]
    if len(steps) == 0:
        return None
    vector = steps.mean(axis=0)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else None


def peptide_plane_normal(res_i: Residue, res_j: Residue) -> np.ndarray | None:
    """Return a peptide-plane normal for adjacent residues when atoms exist."""
    required_i = {"CA", "C", "O"}
    required_j = {"N", "CA"}
    if not required_i.issubset(res_i.atoms) or not required_j.issubset(res_j.atoms):
        return None
    points = np.array([res_i.atoms["CA"], res_i.atoms["C"], res_i.atoms["O"], res_j.atoms["N"], res_j.atoms["CA"]], dtype=float)
    _center, normal, _rms = fit_plane(points)
    return normal


def chain_plane_and_torsion_stats(residues: list[Residue]) -> dict[str, float | np.ndarray | None]:
    """Return chain-level peptide plane normal, theta, and omega stats."""
    normals: list[np.ndarray] = []
    omegas: list[float] = []
    for res_i, res_j in zip(residues, residues[1:]):
        normal = peptide_plane_normal(res_i, res_j)
        if normal is not None:
            normals.append(normal)
        if {"CA", "C"}.issubset(res_i.atoms) and {"N", "CA"}.issubset(res_j.atoms):
            omegas.append(dihedral_degrees(res_i.atoms["CA"], res_i.atoms["C"], res_j.atoms["N"], res_j.atoms["CA"]))
    theta_values = [angle_between_vectors(a, b) for a, b in zip(normals, normals[1:])]
    mean_normal = None
    if normals:
        raw = np.mean(np.array(normals, dtype=float), axis=0)
        norm = float(np.linalg.norm(raw))
        mean_normal = raw / norm if norm > 1e-12 else None
    omega_devs = [abs(abs(value) - 180.0) for value in omegas]
    return {
        "peptide_normal": mean_normal,
        "theta_median_deg": float(np.median(theta_values)) if theta_values else None,
        "omega_median_deg": float(np.median(omegas)) if omegas else None,
        "omega_trans_deviation_median_deg": float(np.median(omega_devs)) if omega_devs else None,
    }


def build_chain_fingerprints(by_chain: dict[str, list[Residue]]) -> list[ChainFingerprint]:
    """Build per-chain fingerprints from parsed residues."""
    all_cas = [coord for residues in by_chain.values() for coord in ca_atoms(residues)]
    if not all_cas:
        raise ValueError("No C-alpha atoms found; cannot compute symmetry fingerprint.")
    center_xy = np.array(all_cas, dtype=float)[:, :2].mean(axis=0)
    rows: list[ChainFingerprint] = []
    for chain, residues in sorted(by_chain.items()):
        names = {res.resname for res in residues}
        backbone_class, confidence = classify_chain(names)
        cas = np.array(ca_atoms(residues), dtype=float)
        centroid = cas.mean(axis=0)
        radial_angle = angle360(centroid[:2] - center_xy)
        exit_vector = mean_exit_vector(residues)
        exit_angle = angle360(exit_vector[:2]) if exit_vector is not None and np.linalg.norm(exit_vector[:2]) > 1e-12 else None
        stats = chain_plane_and_torsion_stats(residues)
        normal = stats["peptide_normal"]
        normal_angle = angle360(normal[:2]) if isinstance(normal, np.ndarray) and np.linalg.norm(normal[:2]) > 1e-12 else None
        rows.append(
            ChainFingerprint(
                chain=chain,
                backbone_class=backbone_class,
                classification_confidence=confidence,
                residue_names=",".join(sorted(names)),
                ca_centroid=centroid,
                radial_angle_deg=radial_angle,
                exit_vector=exit_vector,
                exit_vector_xy_angle_deg=exit_angle,
                peptide_normal=normal if isinstance(normal, np.ndarray) else None,
                peptide_normal_xy_angle_deg=normal_angle,
                theta_median_deg=stats["theta_median_deg"],
                omega_median_deg=stats["omega_median_deg"],
                omega_trans_deviation_median_deg=stats["omega_trans_deviation_median_deg"],
            )
        )
    return rows


def summarize_family(model_id: str, family: str, chains: list[ChainFingerprint], ideal_gap_deg: float, notes: str) -> dict[str, object]:
    """Summarize one rotational-symmetry family."""
    radial_angles = [row.radial_angle_deg for row in chains]
    exit_angles = [row.exit_vector_xy_angle_deg for row in chains if row.exit_vector_xy_angle_deg is not None]
    normal_angles = [row.peptide_normal_xy_angle_deg for row in chains if row.peptide_normal_xy_angle_deg is not None]
    theta_values = [row.theta_median_deg for row in chains if row.theta_median_deg is not None]
    omega_values = [row.omega_median_deg for row in chains if row.omega_median_deg is not None]
    omega_devs = [row.omega_trans_deviation_median_deg for row in chains if row.omega_trans_deviation_median_deg is not None]
    return {
        "model_id": model_id,
        "family": family,
        "chain_ids": ",".join(row.chain for row in chains),
        "backbone_classes": ",".join(sorted({row.backbone_class for row in chains})),
        "classification_confidence": "high" if all(row.classification_confidence == "high" for row in chains) else "low",
        "chain_count": len(chains),
        "ideal_gap_deg": ideal_gap_deg,
        "radial_angle_gap_rms_deg": symmetry_gap_rms_deg(radial_angles, ideal_gap_deg),
        "exit_vector_angle_gap_rms_deg": vector_angle_rms_deg(exit_angles, ideal_gap_deg),
        "peptide_normal_angle_gap_rms_deg": vector_angle_rms_deg(normal_angles, ideal_gap_deg),
        "theta_median_deg": float(np.median(theta_values)) if theta_values else np.nan,
        "theta_std_deg": float(np.std(theta_values)) if theta_values else np.nan,
        "omega_median_deg": float(np.median(omega_values)) if omega_values else np.nan,
        "omega_trans_deviation_median_deg": float(np.median(omega_devs)) if omega_devs else np.nan,
        "residue_name_sets": "; ".join(f"{row.chain}:{row.residue_names}" for row in chains),
        "notes": notes,
    }


def symmetry_summary(model_id: str, chains: list[ChainFingerprint]) -> pd.DataFrame:
    """Return six-fold and class-separated three-fold summary rows."""
    rows = [
        summarize_family(
            model_id,
            "forced_sixfold_all_chains",
            chains,
            60.0,
            "All chains treated as one forced six-fold backbone family.",
        )
    ]
    for family in ["triketo_cyanuric_like", "triamino_melamine_like", "mixed_or_uncertain", "unclassified"]:
        members = [row for row in chains if row.backbone_class == family]
        if members:
            rows.append(
                summarize_family(
                    model_id,
                    f"threefold_{family}",
                    members,
                    120.0,
                    "Class-separated three-fold family based on conservative residue-name classification.",
                )
            )
    return pd.DataFrame(rows)


def chain_table(chains: list[ChainFingerprint]) -> pd.DataFrame:
    """Return per-chain table for report context."""
    return pd.DataFrame(
        [
            {
                "chain": row.chain,
                "backbone_class": row.backbone_class,
                "classification_confidence": row.classification_confidence,
                "residue_names": row.residue_names,
                "radial_angle_deg": row.radial_angle_deg,
                "exit_vector_xy_angle_deg": row.exit_vector_xy_angle_deg,
                "peptide_normal_xy_angle_deg": row.peptide_normal_xy_angle_deg,
                "theta_median_deg": row.theta_median_deg,
                "omega_median_deg": row.omega_median_deg,
                "omega_trans_deviation_median_deg": row.omega_trans_deviation_median_deg,
            }
            for row in chains
        ]
    )


def interpretation(summary: pd.DataFrame) -> dict[str, object]:
    """Return cautious interpretation fields from summary rows."""
    six = summary[summary["family"] == "forced_sixfold_all_chains"].iloc[0]
    three = summary[summary["family"].str.startswith("threefold_")].copy()
    best_three_radial = float(pd.to_numeric(three["radial_angle_gap_rms_deg"], errors="coerce").min()) if not three.empty else float("nan")
    six_radial = float(six["radial_angle_gap_rms_deg"])
    supports_threefold = bool(np.isfinite(best_three_radial) and best_three_radial <= six_radial)
    return {
        "sixfold_radial_rms": six_radial,
        "best_threefold_radial_rms": best_three_radial,
        "supports_threefold_scope_concern": supports_threefold,
    }


def build_report_text(parent_pdb: Path, summary: pd.DataFrame, chains: pd.DataFrame) -> str:
    """Build markdown report for the symmetry diagnostic."""
    interp = interpretation(summary)
    six = summary[summary["family"] == "forced_sixfold_all_chains"].iloc[0]
    three = summary[summary["family"].str.startswith("threefold_")]
    row_notes = [
        f"- Forced six-fold radial RMS: {float(interp['sixfold_radial_rms']):.3f} deg",
        f"- Best class-separated three-fold radial RMS: {float(interp['best_threefold_radial_rms']):.3f} deg",
    ]
    answer = (
        "The class-separated three-fold fingerprint is at least as consistent as the forced six-fold diagnostic by radial chain placement."
        if interp["supports_threefold_scope_concern"]
        else "The forced six-fold diagnostic is not worse by radial placement alone; class chemistry and torsion/plane behavior still need inspection."
    )
    class_rows = markdown_table(
        summary,
        [
            "family",
            "chain_ids",
            "backbone_classes",
            "radial_angle_gap_rms_deg",
            "exit_vector_angle_gap_rms_deg",
            "peptide_normal_angle_gap_rms_deg",
            "theta_median_deg",
            "omega_median_deg",
            "omega_trans_deviation_median_deg",
        ],
    )
    chain_rows = markdown_table(
        chains,
        [
            "chain",
            "backbone_class",
            "classification_confidence",
            "residue_names",
            "radial_angle_deg",
            "exit_vector_xy_angle_deg",
            "theta_median_deg",
            "omega_median_deg",
            "omega_trans_deviation_median_deg",
        ],
    )
    return f"""# Three-Fold Backbone Symmetry Diagnostic

## Model Scope

This is a model-scope/symmetry analysis, not a new atomistic reconstruction. It uses the existing parent/reference coordinate model from the parent-derived bridge workflow:

`{parent_pdb}`

The diagnostic preserves these distinctions:

- Diagnostic coordinate transforms are useful for seeing how geometric perturbations move diffraction bands, but they are not physical provenance.
- The failed pseudo reconstructed bridge is not parent-equivalent and should not be revived as a source model.
- The validated parent-derived bridge is a controlled perturbation of the existing parent coordinate family.
- The fine parent-derived rise scan succeeded within the constrained six-fold parent-derived family, but it is not proof of exact original pNAB/YAML provenance and not proof of physical backbone symmetry.
- The class-separated three-fold analysis below is a new modeling hypothesis motivated by Asem's chemical critique.

## Asem Symmetry Hypothesis

The pNAB-derived construction imposed six-fold backbone symmetry because pNAB could not build two independent backbone types for two different strand classes. Asem flagged that melamine/triamino-side and cyanuric/triketo-side backbone exit vectors may not be chemically equivalent. The next hypothesis is therefore two independent three-fold peptide-backbone symmetry classes rather than one forced six-fold backbone family.

## Summary Metrics

{chr(10).join(row_notes)}

{class_rows}

## Chain Classification

{chain_rows}

## Questions

- Does the parent/reference model look more consistent with one six-fold backbone family or two separate three-fold families? {answer}
- Are the triamino/melamine-like and triketo/cyanuric-like exit vectors chemically/geometrically distinguishable? The residue-name classification separates the chains into distinct chemical classes when `MEP` and `CYP` are present; the exit-vector and peptide-normal RMS columns quantify whether their geometry should be treated separately.
- Is there evidence of alternating peptide-plane or omega behavior by class? Compare `theta_median_deg`, `omega_median_deg`, and `omega_trans_deviation_median_deg` between the two class-separated rows; differences here are diagnostic evidence, not final chemistry.
- Does this support Asem's concern that six-fold symmetry may be over-constraining the backbone? The report supports treating that concern as a serious modeling branch whenever class-separated rows have comparable or lower symmetry RMS, or class-specific torsion/plane statistics differ.
- What coordinate/modeling step should come next? Build a new peptide-plane model track with separate three-fold backbone classes for melamine/triamino-side and cyanuric/triketo-side strands, then use those peptide-plane results to guide any later atomistic model construction.

## Output Files

- Metrics CSV: `outputs/metrics/threefold_backbone_symmetry_summary.csv`
- Report: `outputs/reports/threefold_backbone_symmetry_report.md`
"""


def run_analysis(parent_pdb: Path, metrics_path: Path, report_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the three-fold versus six-fold symmetry diagnostic."""
    residues = parse_residues(parent_pdb)
    chains = build_chain_fingerprints(residues)
    model_id = parent_pdb.stem
    summary = symmetry_summary(model_id, chains)
    chain_df = chain_table(chains)
    combined = pd.concat([summary.assign(row_type="summary"), chain_df.assign(row_type="chain")], ignore_index=True, sort=False)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(metrics_path, index=False)
    report_path.write_text(build_report_text(parent_pdb, summary, chain_df), encoding="utf-8")
    return summary, chain_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-pdb", type=Path, default=DEFAULT_PARENT_PDB)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary, chain_df = run_analysis(args.parent_pdb, args.metrics, args.report)
    print(f"Analyzed {len(chain_df)} chains")
    print(f"Summary rows: {len(summary)}")
    print(f"Metrics: {args.metrics}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
