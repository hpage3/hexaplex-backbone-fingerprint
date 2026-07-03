"""Audit why reconstructed bridge baseline does not match the parent model."""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hexaplex_backbone_fingerprint.parametric_powder_scan import debye_profile, local_maxima, make_q_grid, nearest_peak


DEFAULT_PARENT_PDB = Path("outputs/coordinates/ideal_hexaflex_variants/ideal_hexaflex_backbone_plus_carboxylate.pdb")
DEFAULT_RECONSTRUCTED_PDB = Path("outputs/coordinates/reconstructed_rise_radius_bridge/reconstructed_rise_3p40.pdb")
DEFAULT_BRIDGE_SCORES = Path("outputs/metrics/reconstructed_rise_radius_bridge_abcd_scores.csv")
DEFAULT_SUMMARY_CSV = Path("outputs/metrics/reconstructed_bridge_parent_mismatch_summary.csv")
DEFAULT_PEAK_CSV = Path("outputs/metrics/reconstructed_bridge_parent_mismatch_peak_summary.csv")
DEFAULT_REPORT = Path("outputs/reports/reconstructed_bridge_parent_mismatch_audit.md")

TARGETS_A = {"A": 7.9, "B": 6.5, "C": 5.6, "D": 7.3}


@dataclass(frozen=True)
class AtomRecord:
    """Minimal PDB atom record."""

    serial: int
    atom_name: str
    resname: str
    chain: str
    resseq: str
    element: str
    x: float
    y: float
    z: float

    @property
    def coord(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

    @property
    def is_hydrogen(self) -> bool:
        return self.element.upper() == "H" or self.atom_name.upper().startswith("H")

    @property
    def is_ca(self) -> bool:
        return self.atom_name == "CA"


def parse_pdb_atoms(path: Path) -> list[AtomRecord]:
    """Parse ATOM/HETATM records from PDB."""
    atoms: list[AtomRecord] = []
    for line in path.read_text(encoding="ascii", errors="replace").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        atom_name = line[12:16].strip()
        element = (line[76:78].strip() or atom_name[:1]).upper()
        atoms.append(
            AtomRecord(
                serial=int(line[6:11]),
                atom_name=atom_name,
                resname=line[17:20].strip(),
                chain=line[21:22].strip(),
                resseq=line[22:26].strip(),
                element=element,
                x=float(line[30:38]),
                y=float(line[38:46]),
                z=float(line[46:54]),
            )
        )
    if not atoms:
        raise ValueError(f"No ATOM/HETATM records found in {path}.")
    return atoms


def coords(atoms: list[AtomRecord], heavy_only: bool = False) -> np.ndarray:
    """Return coordinate array."""
    selected = [atom for atom in atoms if not heavy_only or not atom.is_hydrogen]
    return np.array([atom.coord for atom in selected], dtype=float)


def residue_keys(atoms: list[AtomRecord]) -> list[tuple[str, str, str]]:
    """Return unique residue keys in file order."""
    seen: set[tuple[str, str, str]] = set()
    keys = []
    for atom in atoms:
        key = (atom.chain, atom.resseq, atom.resname)
        if key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


def chain_ids(atoms: list[AtomRecord]) -> list[str]:
    """Return sorted chain IDs."""
    return sorted({atom.chain for atom in atoms})


def z_span(atoms: list[AtomRecord]) -> float:
    """Return z-coordinate span."""
    arr = coords(atoms)
    return float(arr[:, 2].max() - arr[:, 2].min())


def xy_radius_summary(atoms: list[AtomRecord], ca_only: bool = True) -> dict[str, float]:
    """Return mean/median xy radius around selected atom xy centroid."""
    selected = [atom for atom in atoms if atom.is_ca] if ca_only else atoms
    if not selected:
        selected = atoms
    arr = coords(selected)
    center_xy = arr[:, :2].mean(axis=0)
    radii = np.linalg.norm(arr[:, :2] - center_xy, axis=1)
    return {"mean": float(np.mean(radii)), "median": float(np.median(radii))}


def residue_counts_by_chain(atoms: list[AtomRecord]) -> dict[str, int]:
    """Return residue counts by chain."""
    result: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for chain, resseq, resname in residue_keys(atoms):
        result[chain].add((resseq, resname))
    return {chain: len(values) for chain, values in sorted(result.items())}


def ca_counts_by_chain(atoms: list[AtomRecord]) -> dict[str, int]:
    """Return C-alpha counts by chain."""
    counter: Counter[str] = Counter(atom.chain for atom in atoms if atom.is_ca)
    return dict(sorted(counter.items()))


def residue_order_by_chain(atoms: list[AtomRecord], limit: int = 12) -> dict[str, str]:
    """Return concise residue order strings by chain."""
    rows: dict[str, list[str]] = defaultdict(list)
    for chain, _resseq, resname in residue_keys(atoms):
        rows[chain].append(resname)
    return {chain: "->".join(values[:limit]) + ("..." if len(values) > limit else "") for chain, values in sorted(rows.items())}


def composition_summary(atoms: list[AtomRecord]) -> dict[str, object]:
    """Return composition-level summary values."""
    atom_names = Counter(atom.atom_name for atom in atoms)
    resnames = Counter(atom.resname for atom in atoms)
    carboxylate_names = {"OE1", "OE2", "OD1", "OD2", "OXT"}
    backbone_names = {"N", "CA", "C", "O"}
    return {
        "atom_count": len(atoms),
        "heavy_atom_count": sum(1 for atom in atoms if not atom.is_hydrogen),
        "chain_count": len(chain_ids(atoms)),
        "chain_ids": ",".join(chain_ids(atoms)),
        "residue_count": len(residue_keys(atoms)),
        "residue_names_counts": compact_counter(Counter(key[2] for key in residue_keys(atoms))),
        "atom_names_counts": compact_counter(atom_names),
        "carboxylate_present": any(atom.atom_name in carboxylate_names for atom in atoms),
        "peptide_backbone_atoms_present": backbone_names.issubset(set(atom_names)),
        "residues_per_chain": compact_mapping(residue_counts_by_chain(atoms)),
        "ca_count_per_chain": compact_mapping(ca_counts_by_chain(atoms)),
        "z_span_A": z_span(atoms),
        "mean_ca_radius_A": xy_radius_summary(atoms)["mean"],
        "median_ca_radius_A": xy_radius_summary(atoms)["median"],
        "residue_order_by_chain": "; ".join(f"{k}:{v}" for k, v in residue_order_by_chain(atoms).items()),
    }


def compact_counter(counter: Counter[str], limit: int = 12) -> str:
    """Return compact counter text."""
    items = counter.most_common(limit)
    text = ";".join(f"{key}:{value}" for key, value in items)
    if len(counter) > limit:
        text += f";...({len(counter)} types)"
    return text


def compact_mapping(mapping: dict[str, object]) -> str:
    """Return stable key:value mapping text."""
    return ";".join(f"{key}:{value}" for key, value in sorted(mapping.items()))


def chain_centroids(atoms: list[AtomRecord]) -> dict[str, np.ndarray]:
    """Return C-alpha centroid by chain, falling back to all atoms."""
    result = {}
    for chain in chain_ids(atoms):
        selected = [atom for atom in atoms if atom.chain == chain and atom.is_ca]
        if not selected:
            selected = [atom for atom in atoms if atom.chain == chain]
        result[chain] = coords(selected).mean(axis=0)
    return result


def chain_centroid_radius_summary(atoms: list[AtomRecord]) -> dict[str, object]:
    """Return chain centroid radius and angular positions."""
    centroids = chain_centroids(atoms)
    arr = np.array(list(centroids.values()), dtype=float)
    center_xy = arr[:, :2].mean(axis=0)
    radii = np.linalg.norm(arr[:, :2] - center_xy, axis=1)
    angles = {
        chain: float((math.degrees(math.atan2(vec[1] - center_xy[1], vec[0] - center_xy[0])) + 360.0) % 360.0)
        for chain, vec in centroids.items()
    }
    return {
        "mean_chain_centroid_radius_A": float(np.mean(radii)),
        "median_chain_centroid_radius_A": float(np.median(radii)),
        "chain_centroid_angles_deg": ";".join(f"{chain}:{angle:.1f}" for chain, angle in sorted(angles.items())),
    }


def interstrand_nearest_neighbor_distances(atoms: list[AtomRecord]) -> np.ndarray:
    """Return nearest C-alpha distance from each chain to other chains."""
    cas = [atom for atom in atoms if atom.is_ca]
    values: list[float] = []
    for atom in cas:
        others = [other for other in cas if other.chain != atom.chain]
        if others:
            values.append(min(float(np.linalg.norm(atom.coord - other.coord)) for other in others))
    return np.asarray(values, dtype=float)


def estimate_chain_rise_twist(atoms: list[AtomRecord]) -> dict[str, float]:
    """Estimate median same-chain CA rise and angular twist around centroid."""
    cas = [atom for atom in atoms if atom.is_ca]
    if not cas:
        return {"median_ca_rise_A": float("nan"), "median_abs_ca_twist_deg": float("nan")}
    center_xy = coords(cas)[:, :2].mean(axis=0)
    rises: list[float] = []
    twists: list[float] = []
    by_chain: dict[str, list[AtomRecord]] = defaultdict(list)
    for atom in cas:
        by_chain[atom.chain].append(atom)
    for chain_atoms in by_chain.values():
        chain_atoms = sorted(chain_atoms, key=lambda atom: atom.serial)
        for a, b in zip(chain_atoms, chain_atoms[1:]):
            rises.append(abs(b.z - a.z))
            angle_a = math.atan2(a.y - center_xy[1], a.x - center_xy[0])
            angle_b = math.atan2(b.y - center_xy[1], b.x - center_xy[0])
            delta = math.degrees(math.atan2(math.sin(angle_b - angle_a), math.cos(angle_b - angle_a)))
            twists.append(abs(delta))
    return {
        "median_ca_rise_A": float(np.median(rises)) if rises else float("nan"),
        "median_abs_ca_twist_deg": float(np.median(twists)) if twists else float("nan"),
    }


def geometry_summary(atoms: list[AtomRecord]) -> dict[str, object]:
    """Return register/geometry organization summary values."""
    nn = interstrand_nearest_neighbor_distances(atoms)
    out = {
        "mean_interstrand_nn_ca_distance_A": float(np.mean(nn)) if len(nn) else float("nan"),
        "median_interstrand_nn_ca_distance_A": float(np.median(nn)) if len(nn) else float("nan"),
    }
    out.update(chain_centroid_radius_summary(atoms))
    out.update(estimate_chain_rise_twist(atoms))
    return out


def compare_metric(metric: str, parent_value: object, reconstructed_value: object, severity: str, interpretation: str) -> dict[str, object]:
    """Build one mismatch summary row."""
    note = ""
    if isinstance(parent_value, (int, float)) and isinstance(reconstructed_value, (int, float)):
        note = f"{float(reconstructed_value) - float(parent_value):.4g}"
    elif parent_value != reconstructed_value:
        note = "different"
    else:
        note = "same"
    return {
        "metric": metric,
        "parent_value": parent_value,
        "reconstructed_value": reconstructed_value,
        "delta_or_note": note,
        "severity": severity,
        "interpretation": interpretation,
    }


def severity_for_numeric_delta(parent_value: float, reconstructed_value: float, medium: float, high: float) -> str:
    """Return severity from absolute numeric delta."""
    delta = abs(float(reconstructed_value) - float(parent_value))
    if delta >= high:
        return "high"
    if delta >= medium:
        return "medium"
    return "low"


def build_mismatch_rows(parent_atoms: list[AtomRecord], reconstructed_atoms: list[AtomRecord]) -> pd.DataFrame:
    """Build composition and geometry mismatch summary rows."""
    parent_comp = composition_summary(parent_atoms)
    recon_comp = composition_summary(reconstructed_atoms)
    parent_geo = geometry_summary(parent_atoms)
    recon_geo = geometry_summary(reconstructed_atoms)
    rows: list[dict[str, object]] = []
    specs = [
        ("atom_count", "high", "different scattering content if atom counts differ strongly"),
        ("heavy_atom_count", "high", "heavy atom content controls Debye scattering"),
        ("chain_count", "high", "chain count should match for parent-equivalent model"),
        ("residue_count", "high", "residue model composition/register differs"),
        ("residue_names_counts", "high", "parent chemical residues differ from pseudo-residues"),
        ("atom_names_counts", "high", "atom-name inventory differs"),
        ("carboxylate_present", "high", "carboxylate content is central to clean explanatory model"),
        ("peptide_backbone_atoms_present", "medium", "backbone/peptide-plane atoms should be comparable"),
        ("residues_per_chain", "medium", "per-chain register/repeat count differs"),
        ("ca_count_per_chain", "medium", "C-alpha/register count differs"),
        ("residue_order_by_chain", "high", "sequence/order compatibility check"),
    ]
    for metric, default_severity, interpretation in specs:
        severity = "low" if parent_comp[metric] == recon_comp[metric] else default_severity
        rows.append(compare_metric(metric, parent_comp[metric], recon_comp[metric], severity, interpretation))
    for metric, medium, high, interpretation in [
        ("z_span_A", 2.0, 8.0, "axial extent differs"),
        ("mean_ca_radius_A", 1.0, 3.0, "mean C-alpha radius differs"),
        ("median_ca_radius_A", 1.0, 3.0, "median C-alpha radius differs"),
    ]:
        rows.append(
            compare_metric(
                metric,
                round(float(parent_comp[metric]), 4),
                round(float(recon_comp[metric]), 4),
                severity_for_numeric_delta(float(parent_comp[metric]), float(recon_comp[metric]), medium, high),
                interpretation,
            )
        )
    for metric, medium, high, interpretation in [
        ("mean_interstrand_nn_ca_distance_A", 0.5, 1.5, "inter-strand nearest-neighbor geometry differs"),
        ("median_interstrand_nn_ca_distance_A", 0.5, 1.5, "inter-strand nearest-neighbor geometry differs"),
        ("mean_chain_centroid_radius_A", 1.0, 3.0, "chain centroid radius differs"),
        ("median_chain_centroid_radius_A", 1.0, 3.0, "chain centroid radius differs"),
        ("median_ca_rise_A", 0.5, 1.5, "coordinate-estimated CA rise differs"),
        ("median_abs_ca_twist_deg", 5.0, 15.0, "coordinate-estimated CA twist differs"),
    ]:
        rows.append(
            compare_metric(
                metric,
                round(float(parent_geo[metric]), 4),
                round(float(recon_geo[metric]), 4),
                severity_for_numeric_delta(float(parent_geo[metric]), float(recon_geo[metric]), medium, high),
                interpretation,
            )
        )
    rows.append(compare_metric("chain_centroid_angles_deg", parent_geo["chain_centroid_angles_deg"], recon_geo["chain_centroid_angles_deg"], "medium", "chain angular organization comparison"))
    return pd.DataFrame(rows)


def resolve_reconstructed_path(default_path: Path, scores_csv: Path) -> Path:
    """Resolve reconstructed 3p40 PDB from scores CSV if needed."""
    if default_path.exists():
        return default_path
    if scores_csv.exists():
        scores = pd.read_csv(scores_csv)
        subset = scores[scores["variant_id"] == "reconstructed_rise_3p40"]
        if not subset.empty and "coordinate_path" in subset.columns:
            candidate = Path(str(subset.iloc[0]["coordinate_path"]))
            if candidate.exists():
                return candidate
    return default_path


def peak_summary_for_model(model_label: str, path: Path) -> pd.DataFrame:
    """Return concise nearest and top-window peaks for A/B/C/D targets."""
    arr = coords(parse_pdb_atoms(path), heavy_only=False)
    profile = debye_profile(arr, make_q_grid(d_min_A=2.5, d_max_A=12.0, q_step=0.01))
    peaks = local_maxima(profile)
    rows: list[dict[str, object]] = []
    for band, target in TARGETS_A.items():
        hit = nearest_peak(profile, target, tolerance_A=0.20)
        window = peaks[peaks["d_A"].between(target - 1.0, target + 1.0)].copy()
        top_peaks = window.sort_values("intensity", ascending=False).head(3)
        rows.append(
            {
                "model_label": model_label,
                "band": band,
                "target_d_A": target,
                "nearest_peak_d_A": hit.peak_d_A,
                "nearest_peak_error_A": hit.error_A,
                "nearest_peak_intensity": hit.intensity,
                "top_window_peak_d_A": ";".join(f"{value:.4f}" for value in top_peaks["d_A"].tolist()),
            }
        )
    return pd.DataFrame(rows)


def diagnose(summary: pd.DataFrame, peak_summary: pd.DataFrame) -> dict[str, str]:
    """Return diagnosis fields from mismatch and peak summaries."""
    high_metrics = set(summary.loc[summary["severity"] == "high", "metric"].astype(str))
    recon = peak_summary[peak_summary["model_label"] == "reconstructed_3p40"]
    collapsed = False
    if not recon.empty:
        observed = recon.set_index("band")["nearest_peak_d_A"].to_dict()
        collapsed = len({round(float(observed.get(band, float("nan"))), 3) for band in ["A", "B", "D"]}) == 1
    causes = []
    if {"atom_count", "heavy_atom_count", "atom_names_counts", "carboxylate_present"} & high_metrics:
        causes.append("missing_atoms_or_groups")
    if {"residue_names_counts", "residue_order_by_chain", "ca_count_per_chain"} & high_metrics:
        causes.append("different_register_or_orientation")
    if {"mean_ca_radius_A", "median_ca_rise_A", "median_abs_ca_twist_deg"} & set(summary.loc[summary["severity"].isin(["medium", "high"]), "metric"].astype(str)):
        causes.append("different_geometry_family")
    if collapsed:
        causes.append("scoring_or_peak_picking_issue")
    mismatch_source = causes[0] if len(causes) == 1 else ("multiple_causes" if causes else "unresolved")
    bridge_status = "not_parent_equivalent" if causes else "unresolved"
    if mismatch_source == "missing_atoms_or_groups":
        recommendation = "fix reconstructed generator atom content"
    elif mismatch_source in {"different_register_or_orientation", "different_geometry_family", "multiple_causes"}:
        recommendation = "use parent-derived coordinate generator instead"
    elif mismatch_source == "scoring_or_peak_picking_issue":
        recommendation = "fix scoring/peak picking"
    else:
        recommendation = "recover original pNAB/source files"
    return {
        "mismatch_source": mismatch_source,
        "bridge_status": bridge_status,
        "next_recommendation": recommendation,
        "collapsed_A_B_D_peak": str(collapsed),
    }


def markdown_table(df: pd.DataFrame, columns: list[str], limit: int = 20) -> str:
    """Render a compact markdown table."""
    if df.empty:
        return "_None._"
    columns = [column for column in columns if column in df.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for record in df[columns].head(limit).itertuples(index=False):
        values = [f"{value:.5g}" if isinstance(value, float) else str(value) for value in record]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_report(parent_path: Path, recon_path: Path, summary: pd.DataFrame, peak_summary: pd.DataFrame, diagnosis: dict[str, str]) -> str:
    """Build markdown report."""
    top = summary.sort_values("severity", key=lambda s: s.map({"high": 0, "medium": 1, "low": 2}).fillna(3)).head(12)
    return f"""# Reconstructed Bridge Parent Mismatch Audit

## Purpose

This audit explains why `reconstructed_rise_3p40` should not be treated as parent-equivalent simply because it uses a 3.40 A rise parameter.

## Paths

- Parent baseline PDB: `{parent_path}`
- Reconstructed 3.40 PDB: `{recon_path}`

## High-Level Composition And Geometry Findings

{markdown_table(top, ["metric", "parent_value", "reconstructed_value", "delta_or_note", "severity", "interpretation"], limit=12)}

## Peak/Profile Sanity Summary

{markdown_table(peak_summary, ["model_label", "band", "target_d_A", "nearest_peak_d_A", "nearest_peak_error_A", "top_window_peak_d_A"], limit=12)}

## Scoring Compatibility

Both models were evaluated here with the same direct point-scatterer Debye helper on all PDB ATOM/HETATM coordinates. The reconstructed bridge score CSV used the same helper family but scored the generated XYZ. The reconstructed model does not contain the same atom/residue content as the parent, so equal scoring code does not make the structures comparable.

The reconstructed 3.40 model collapses A/B/D nearest picked peaks onto the same d spacing: `{diagnosis['collapsed_A_B_D_peak']}`.

## Diagnosis

- mismatch_source: `{diagnosis['mismatch_source']}`
- bridge_status: `{diagnosis['bridge_status']}`
- next_recommendation: `{diagnosis['next_recommendation']}`

Conclusion: `reconstructed_rise_3p40` is not parent-equivalent under this audit. Treat the 3.35/3.38/3.40 reconstructed bridge family as a negative or partial diagnostic until atom content, register/orientation, and parent-derived geometry are repaired or original pNAB/source files are recovered.
"""


def run_audit(
    parent_pdb: Path = DEFAULT_PARENT_PDB,
    reconstructed_pdb: Path = DEFAULT_RECONSTRUCTED_PDB,
    bridge_scores_csv: Path = DEFAULT_BRIDGE_SCORES,
    summary_csv: Path = DEFAULT_SUMMARY_CSV,
    peak_csv: Path = DEFAULT_PEAK_CSV,
    report_path: Path = DEFAULT_REPORT,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Run mismatch audit and write outputs."""
    reconstructed_pdb = resolve_reconstructed_path(reconstructed_pdb, bridge_scores_csv)
    if not parent_pdb.exists():
        raise FileNotFoundError(f"Missing parent PDB: {parent_pdb}")
    if not reconstructed_pdb.exists():
        raise FileNotFoundError(f"Missing reconstructed PDB: {reconstructed_pdb}")
    parent_atoms = parse_pdb_atoms(parent_pdb)
    reconstructed_atoms = parse_pdb_atoms(reconstructed_pdb)
    summary = build_mismatch_rows(parent_atoms, reconstructed_atoms)
    peak_summary = pd.concat(
        [
            peak_summary_for_model("parent_baseline", parent_pdb),
            peak_summary_for_model("reconstructed_3p40", reconstructed_pdb),
        ],
        ignore_index=True,
    )
    diagnosis = diagnose(summary, peak_summary)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    peak_csv.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, index=False)
    peak_summary.to_csv(peak_csv, index=False)
    report_path.write_text(build_report(parent_pdb, reconstructed_pdb, summary, peak_summary, diagnosis), encoding="utf-8")
    return summary, peak_summary, diagnosis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-pdb", type=Path, default=DEFAULT_PARENT_PDB)
    parser.add_argument("--reconstructed-pdb", type=Path, default=DEFAULT_RECONSTRUCTED_PDB)
    parser.add_argument("--bridge-scores-csv", type=Path, default=DEFAULT_BRIDGE_SCORES)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--peak-csv", type=Path, default=DEFAULT_PEAK_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary, _peaks, diagnosis = run_audit(
        args.parent_pdb,
        args.reconstructed_pdb,
        args.bridge_scores_csv,
        args.summary_csv,
        args.peak_csv,
        args.report,
    )
    high = int((summary["severity"] == "high").sum())
    medium = int((summary["severity"] == "medium").sum())
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.peak_csv}")
    print(f"Wrote {args.report}")
    print(f"High severity findings: {high}; medium severity findings: {medium}")
    print(f"mismatch_source: {diagnosis['mismatch_source']}")
    print(f"bridge_status: {diagnosis['bridge_status']}")
    print(f"next_recommendation: {diagnosis['next_recommendation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
