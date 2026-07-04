"""Audit pNAB parallel versus anti-parallel model evidence.

This is a conservative compatibility/scaffold-filter audit. pNAB-derived files
are not treated as final structural proof, especially given known limitations
for the peptide-backed hexaplex case.
"""

from __future__ import annotations

import argparse
import re
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

from scripts.generate_global_deformation_variants import parse_pdb_atom_lines
from scripts.run_internal_coordinate_endpoint_closure import detect_every_other_pattern, trans_deviation_deg
from scripts.run_parent_derived_rise_bridge import (
    EXPECTED_PARENT_C_A,
    EXPECTED_PARENT_D_A,
    TARGETS_A,
    carboxylate_present,
    ca_rise_values,
    interstrand_nn_ca_distances,
    markdown_table,
    residue_keys,
    score_pdb_abcd,
)
from scripts.analyze_threefold_backbone_symmetry import parse_residues
from hexaplex_backbone_fingerprint.geometry import dihedral_degrees


DEFAULT_INVENTORY = Path("outputs/metrics/pnab_parallel_antiparallel_inventory.csv")
DEFAULT_CANDIDATE_SUMMARY = Path("outputs/metrics/pnab_parallel_antiparallel_candidate_summary.csv")
DEFAULT_GEOMETRY = Path("outputs/metrics/pnab_parallel_antiparallel_geometry.csv")
DEFAULT_ABCD = Path("outputs/metrics/pnab_parallel_antiparallel_abcd_scores.csv")
DEFAULT_REPORT = Path("outputs/reports/pnab_parallel_antiparallel_audit_report.md")

OMEGA_CLEAN_C_A = 5.6422
OMEGA_CLEAN_D_A = 7.2756
OMEGA_CLEAN_CD_ERROR_A = 0.0667

SEARCH_EXTENSIONS = {".pdb", ".ent", ".xyz", ".csv", ".tsv", ".txt", ".yaml", ".yml", ".json", ".md", ".pml"}
SKIP_DIR_NAMES = {".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache"}


def infer_orientation(text: str) -> str:
    """Infer parallel, anti_parallel, or unknown from path/name text."""
    value = text.lower()
    anti_patterns = ["anti-parallel", "anti_parallel", "antiparallel", "anti parallel", "anti-par"]
    if any(pattern in value for pattern in anti_patterns):
        return "anti_parallel"
    if "parallel" in value:
        return "parallel"
    return "unknown"


def infer_twist_deg(text: str) -> float | None:
    """Infer twist angle from filename/path text."""
    value = text.lower()
    patterns = [
        r"twist[_-]?([0-9]+(?:p[0-9]+|\.[0-9]+)?)",
        r"([0-9]+(?:p[0-9]+|\.[0-9]+)?)\s*deg",
        r"([0-9]+(?:p[0-9]+|\.[0-9]+)?)_degree",
        r"tw([0-9]+(?:p[0-9]+|\.[0-9]+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, value)
        if match:
            return float(match.group(1).replace("p", "."))
    return None


def infer_rise_A(text: str) -> float | None:
    """Infer rise from filename/path text."""
    value = text.lower()
    patterns = [
        r"rise[_-]?([0-9]+(?:p[0-9]+|\.[0-9]+)?)",
        r"([0-9]+p[0-9]+)\s*(?:a|ang|angstrom)?",
        r"([0-9]+\.[0-9]+)\s*(?:a|ang|angstrom)?",
    ]
    for pattern in patterns:
        match = re.search(pattern, value)
        if match:
            number = float(match.group(1).replace("p", "."))
            if 2.5 <= number <= 4.5:
                return number
    return None


def file_type(path: Path) -> str:
    """Return coarse file type."""
    suffix = path.suffix.lower()
    if suffix in {".pdb", ".ent", ".xyz"}:
        return "coordinate"
    if suffix in {".yaml", ".yml", ".json"}:
        return "input_or_config"
    if suffix in {".csv", ".tsv"}:
        return "metric_or_table"
    if suffix in {".txt", ".md"}:
        return "report_or_text"
    return suffix.lstrip(".") or "unknown"


def is_likely_candidate(path: Path) -> bool:
    """Return whether path is relevant to pNAB/orientation/twist audit."""
    text = str(path).lower()
    if path.suffix.lower() not in SEARCH_EXTENSIONS:
        return False
    direct_terms = ["pnab", "parallel", "antiparallel", "anti_parallel", "anti-parallel", "anti parallel"]
    if any(term in text for term in direct_terms):
        return True
    twist = infer_twist_deg(text)
    rise = infer_rise_A(text)
    return twist is not None and rise is not None and any(term in text for term in ["hexaplex", "hexaflex", "6strand"])


def iter_candidate_files(root: Path) -> list[Path]:
    """Find likely candidate files without following external inputs."""
    files: list[Path] = []
    for path in root.rglob("*"):
        if any(part in SKIP_DIR_NAMES for part in path.parts):
            continue
        if path.is_file() and is_likely_candidate(path):
            files.append(path)
    return sorted(files)


def inventory_row(path: Path, root: Path) -> dict[str, object]:
    """Build one inventory row."""
    rel = path.relative_to(root) if path.is_relative_to(root) else path
    ftype = file_type(path)
    is_visual_box = is_visual_box_file(path)
    return {
        "path": str(rel),
        "file_type": ftype,
        "inferred_orientation": infer_orientation(str(path)),
        "inferred_twist_deg": infer_twist_deg(str(path)),
        "inferred_rise_A": infer_rise_A(str(path)),
        "coordinate_file_exists": ftype == "coordinate" and path.exists() and not is_visual_box,
        "metric_or_scoring_file_exists": ftype == "metric_or_table" and path.exists(),
        "notes": "visual peptide-box helper, not scored as atomistic coordinate" if is_visual_box else "candidate found by conservative filename/path inference",
    }


def build_inventory(root: Path) -> pd.DataFrame:
    """Build inventory table."""
    rows = [inventory_row(path, root) for path in iter_candidate_files(root)]
    return pd.DataFrame(rows)


def parse_acceptance_table(path: Path) -> dict[str, object] | None:
    """Parse accepted/rejected counts and energy-like columns when possible."""
    if path.suffix.lower() not in {".csv", ".tsv"}:
        return None
    try:
        df = pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",", nrows=5000)
    except Exception:
        return None
    if df.empty:
        return None
    cols = {str(col).lower(): col for col in df.columns}
    status_cols = [col for low, col in cols.items() if any(term in low for term in ["status", "accept", "reject", "passed", "valid"])]
    accepted = 0
    rejected = 0
    if status_cols:
        text = df[status_cols[0]].astype(str).str.lower()
        accepted = int(text.str.contains("accept|pass|valid|ok|true", regex=True).sum())
        rejected = int(text.str.contains("reject|fail|invalid|false", regex=True).sum())
    energy_cols = [col for low, col in cols.items() if any(term in low for term in ["energy", "bond", "angle", "torsion", "vdw", "total"])]
    out: dict[str, object] = {
        "path": str(path),
        "candidate_count": len(df),
        "accepted_count": accepted,
        "rejected_count": rejected,
        "energy_columns": ",".join(map(str, energy_cols)),
    }
    for col in energy_cols[:8]:
        values = pd.to_numeric(df[col], errors="coerce")
        out[f"{col}_median"] = float(values.median()) if values.notna().any() else np.nan
    return out


def candidate_summary(inventory: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Build per-orientation/twist/rise candidate summary."""
    rows = []
    for path_text in inventory["path"].tolist() if not inventory.empty else []:
        path = root / path_text
        parsed = parse_acceptance_table(path)
        if parsed:
            parsed["inferred_orientation"] = infer_orientation(path_text)
            parsed["inferred_twist_deg"] = infer_twist_deg(path_text)
            parsed["inferred_rise_A"] = infer_rise_A(path_text)
            rows.append(parsed)
    if rows:
        return pd.DataFrame(rows)
    grouped = (
        inventory.groupby(["inferred_orientation", "inferred_twist_deg", "inferred_rise_A"], dropna=False)
        .agg(
            file_count=("path", "count"),
            coordinate_count=("coordinate_file_exists", "sum"),
            metric_count=("metric_or_scoring_file_exists", "sum"),
        )
        .reset_index()
        if not inventory.empty
        else pd.DataFrame(columns=["inferred_orientation", "inferred_twist_deg", "inferred_rise_A", "file_count", "coordinate_count", "metric_count"])
    )
    grouped["candidate_count"] = ""
    grouped["accepted_count"] = ""
    grouped["rejected_count"] = ""
    grouped["notes"] = "no parseable pNAB acceptance table found; summary is file-inventory based"
    return grouped


def omega_records(path: Path) -> list[float]:
    """Return omega values for a PDB when peptide atoms exist."""
    try:
        residues = parse_residues(path)
    except Exception:
        return []
    values: list[float] = []
    for chain_residues in residues.values():
        for res_i, res_j in zip(chain_residues, chain_residues[1:]):
            if {"CA", "C"}.issubset(res_i.atoms) and {"N", "CA"}.issubset(res_j.atoms):
                values.append(dihedral_degrees(res_i.atoms["CA"], res_i.atoms["C"], res_j.atoms["N"], res_j.atoms["CA"]))
    return values


def geometry_row(path: Path, root: Path) -> dict[str, object] | None:
    """Compute geometry sanity metrics for one PDB."""
    if path.suffix.lower() not in {".pdb", ".ent"}:
        return None
    if is_visual_box_file(path):
        return None
    try:
        _lines, atoms = parse_pdb_atom_lines(path)
    except Exception:
        return None
    omegas = omega_records(path)
    deviations = [trans_deviation_deg(value) for value in omegas]
    pattern = detect_every_other_pattern(deviations)
    rises = ca_rise_values(atoms)
    nn = interstrand_nn_ca_distances(atoms)
    chains = sorted({atom.chain for atom in atoms})
    return {
        "path": str(path.relative_to(root) if path.is_relative_to(root) else path),
        "inferred_orientation": infer_orientation(str(path)),
        "inferred_twist_deg": infer_twist_deg(str(path)),
        "inferred_rise_A": infer_rise_A(str(path)),
        "atom_count": len(atoms),
        "residue_count": len(residue_keys(atoms)),
        "chain_count": len(chains),
        "chain_ids": ",".join(chains),
        "carboxylate_present": carboxylate_present(atoms),
        "omega_count": len(omegas),
        "omega_median_deg": float(np.median(omegas)) if omegas else np.nan,
        "omega_trans_deviation_median_deg": float(np.median(deviations)) if deviations else np.nan,
        "omega_within_8_count": int(sum(value <= 8.0 for value in deviations)),
        "omega_within_10_count": int(sum(value <= 10.0 for value in deviations)),
        "omega_every_other_detected": bool(pattern["every_other_detected"]),
        "median_ca_rise_A": float(np.median(rises)) if len(rises) else np.nan,
        "median_interstrand_nn_ca_distance_A": float(np.median(nn)) if len(nn) else np.nan,
    }


def should_score_coordinate(path: Path) -> bool:
    """Score only likely pNAB/orientation coordinate candidates."""
    text = str(path).lower()
    if is_visual_box_file(path):
        return False
    return path.suffix.lower() in {".pdb", ".ent"} and (
        "pnab" in text or infer_orientation(text) != "unknown"
    )


def is_visual_box_file(path: Path) -> bool:
    """Return whether a PDB is a peptide-box visualization helper."""
    text = str(path).lower()
    return "visual_boxes" in text or "_boxes.pdb" in text or "\\output_boxes\\" in text or "/output_boxes/" in text


def abcd_row(path: Path, root: Path) -> dict[str, object] | None:
    """Compute A/B/C/D score for a coordinate file."""
    if not should_score_coordinate(path):
        return None
    try:
        scores = score_pdb_abcd(path)
    except Exception as exc:
        return {
            "path": str(path.relative_to(root) if path.is_relative_to(root) else path),
            "score_status": "failed",
            "notes": str(exc),
        }
    c_error = float(scores["observed_C_d_A"]) - TARGETS_A["C"]
    d_error = float(scores["observed_D_d_A"]) - TARGETS_A["D"]
    return {
        "path": str(path.relative_to(root) if path.is_relative_to(root) else path),
        "inferred_orientation": infer_orientation(str(path)),
        "inferred_twist_deg": infer_twist_deg(str(path)),
        "inferred_rise_A": infer_rise_A(str(path)),
        **scores,
        "C_error_A": c_error,
        "D_error_A": d_error,
        "combined_CD_abs_error_A": abs(c_error) + abs(d_error),
        "matches_parent_baseline": abs(float(scores["observed_C_d_A"]) - EXPECTED_PARENT_C_A) <= 0.05
        and abs(float(scores["observed_D_d_A"]) - EXPECTED_PARENT_D_A) <= 0.05,
        "approaches_omega_clean_plateau": abs(float(scores["observed_C_d_A"]) - OMEGA_CLEAN_C_A) <= 0.05
        and abs(float(scores["observed_D_d_A"]) - OMEGA_CLEAN_D_A) <= 0.05,
        "D_degrades_from_omega_clean": abs(float(scores["observed_D_d_A"]) - OMEGA_CLEAN_D_A) > 0.05,
        "score_status": "scored",
        "notes": "preliminary Debye score; pNAB is compatibility filter, not final proof",
    }


def classify_parallel_elimination(inventory: pd.DataFrame, summary: pd.DataFrame, scores: pd.DataFrame) -> str:
    """Classify whether parallel can be eliminated."""
    parallel_files = inventory[inventory["inferred_orientation"] == "parallel"] if not inventory.empty else pd.DataFrame()
    if parallel_files.empty:
        return "insufficient_data"
    parallel_scores = scores[scores.get("inferred_orientation", pd.Series(dtype=str)) == "parallel"] if not scores.empty else pd.DataFrame()
    if not parallel_scores.empty and bool((parallel_scores["approaches_omega_clean_plateau"] == True).any()):
        return "not_eliminated"
    if not parallel_scores.empty:
        return "disfavored_not_eliminated"
    return "insufficient_data"


def classify_anti_parallel_30(inventory: pd.DataFrame, scores: pd.DataFrame) -> str:
    """Classify anti-parallel 30-degree status."""
    anti30_files = inventory[
        (inventory["inferred_orientation"] == "anti_parallel")
        & (pd.to_numeric(inventory["inferred_twist_deg"], errors="coerce").sub(30).abs() <= 0.5)
    ] if not inventory.empty else pd.DataFrame()
    if anti30_files.empty:
        return "insufficient_data"
    anti30_scores = scores[
        (scores.get("inferred_orientation", pd.Series(dtype=str)) == "anti_parallel")
        & (pd.to_numeric(scores.get("inferred_twist_deg", pd.Series(dtype=float)), errors="coerce").sub(30).abs() <= 0.5)
    ] if not scores.empty else pd.DataFrame()
    if not anti30_scores.empty and bool((anti30_scores["approaches_omega_clean_plateau"] == True).any()):
        return "strongest_current_pnab_candidate"
    if not anti30_files.empty:
        return "plausible_candidate"
    return "insufficient_data"


def orientation_status(orientation: str, inventory: pd.DataFrame, scores: pd.DataFrame) -> str:
    """Return conservative orientation-level status."""
    files = inventory[inventory["inferred_orientation"] == orientation] if not inventory.empty else pd.DataFrame()
    if files.empty:
        return "insufficient_data"
    orient_scores = scores[scores.get("inferred_orientation", pd.Series(dtype=str)) == orientation] if not scores.empty else pd.DataFrame()
    if not orient_scores.empty and bool((orient_scores["approaches_omega_clean_plateau"] == True).any()):
        return "strongly_supported"
    if not orient_scores.empty:
        return "plausible"
    return "plausible"


def build_report(inventory: pd.DataFrame, summary: pd.DataFrame, geometry: pd.DataFrame, scores: pd.DataFrame) -> str:
    """Build markdown report."""
    parallel_status = classify_parallel_elimination(inventory, summary, scores)
    anti30_status = classify_anti_parallel_30(inventory, scores)
    orientation_rows = pd.DataFrame(
        [
            {"orientation": "parallel", "orientation_status": orientation_status("parallel", inventory, scores)},
            {"orientation": "anti_parallel", "orientation_status": orientation_status("anti_parallel", inventory, scores)},
            {"orientation": "unknown", "orientation_status": orientation_status("unknown", inventory, scores)},
        ]
    )
    inventory_counts = inventory.groupby(["inferred_orientation", "file_type"], dropna=False).size().reset_index(name="count") if not inventory.empty else pd.DataFrame()
    return f"""# pNAB Parallel Versus Anti-Parallel Compatibility Audit

## Scope

pNAB is being used as a compatibility/scaffold filter, not final structural proof. Current pNAB implementation has known limitations for this peptide-backed hexaplex case, including regularized symmetry behavior and the previously observed peptide/omega artifacts. Parallel candidates should only be eliminated if the data support elimination. If parallel candidates are merely worse but not eliminated, the status is `disfavored_not_eliminated`.

Exact original pNAB/YAML provenance in this repo is reported from available files only and may be partial or missing.

## Inventory Summary

{markdown_table(inventory_counts, ['inferred_orientation', 'file_type', 'count'])}

## Orientation Status

{markdown_table(orientation_rows, ['orientation', 'orientation_status'])}

- Parallel elimination status: `{parallel_status}`
- Anti-parallel 30 status: `{anti30_status}`

## Candidate Compatibility Summary

{markdown_table(summary.head(30), list(summary.columns[:10]))}

## Geometry Summary

{markdown_table(geometry.head(20), ['path', 'inferred_orientation', 'inferred_twist_deg', 'atom_count', 'chain_count', 'carboxylate_present', 'omega_count', 'omega_within_8_count', 'omega_within_10_count', 'omega_every_other_detected'])}

## A/B/C/D Diffraction Scores

{markdown_table(scores.head(20), ['path', 'inferred_orientation', 'inferred_twist_deg', 'observed_C_d_A', 'observed_D_d_A', 'combined_CD_abs_error_A', 'matches_parent_baseline', 'approaches_omega_clean_plateau', 'D_degrades_from_omega_clean', 'score_status'])}

## Interpretation Questions

- What pNAB parallel and anti-parallel data exist in the repo? See the inventory CSV and orientation count table.
- Are both orientations represented by source/input files, outputs, and coordinates? The inventory distinguishes `parallel`, `anti_parallel`, and `unknown`; missing orientation labels are not guessed.
- Does pNAB show a compatibility range by twist at rise 3.4 A? This can only be assessed where parseable pNAB output or coordinate files include twist/rise metadata.
- Which twist angles/orientations produce accepted or plausible candidates? See the candidate summary and score tables; acceptance counts are blank when no parseable pNAB acceptance table exists.
- Is anti-parallel 30 supported as a strong candidate by pNAB compatibility metrics? Current status: `{anti30_status}`.
- Are parallel candidates disfavored or eliminated? Current parallel elimination status: `{parallel_status}`.
- Can parallel structures be eliminated based on current repo data? Only if the status is `eliminated`; otherwise retain the caveat.
- If not eliminated, what missing data prevents elimination? Direct parallel pNAB coordinate/output tables with comparable twist/rise, geometry, and diffraction scores.
- How do pNAB-derived conclusions relate to omega-clean external-backbone results? pNAB remains a scaffold/compatibility filter. The omega-clean external-backbone pathway is the current better-controlled route for C/D-compatible structures because it removes the suspicious every-other omega artifact.
- What should the next step be? Locate or generate matched pNAB parallel and anti-parallel outputs at comparable rise/twist settings, then run this same audit on matched coordinate and acceptance tables.

## Outputs

- Inventory: `outputs/metrics/pnab_parallel_antiparallel_inventory.csv`
- Candidate summary: `outputs/metrics/pnab_parallel_antiparallel_candidate_summary.csv`
- Geometry: `outputs/metrics/pnab_parallel_antiparallel_geometry.csv`
- A/B/C/D scores: `outputs/metrics/pnab_parallel_antiparallel_abcd_scores.csv`
"""


def run_audit(
    root: Path = ROOT,
    inventory_path: Path = DEFAULT_INVENTORY,
    summary_path: Path = DEFAULT_CANDIDATE_SUMMARY,
    geometry_path: Path = DEFAULT_GEOMETRY,
    abcd_path: Path = DEFAULT_ABCD,
    report_path: Path = DEFAULT_REPORT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run audit and write outputs."""
    inventory = build_inventory(root)
    summary = candidate_summary(inventory, root)
    candidate_paths = [root / path for path in inventory["path"].tolist()] if not inventory.empty else []
    geometry = pd.DataFrame([row for row in (geometry_row(path, root) for path in candidate_paths) if row is not None])
    scores = pd.DataFrame([row for row in (abcd_row(path, root) for path in candidate_paths) if row is not None])
    for path in [inventory_path, summary_path, geometry_path, abcd_path, report_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(inventory_path, index=False)
    summary.to_csv(summary_path, index=False)
    geometry.to_csv(geometry_path, index=False)
    scores.to_csv(abcd_path, index=False)
    report_path.write_text(build_report(inventory, summary, geometry, scores), encoding="utf-8")
    return inventory, summary, geometry, scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--summary", type=Path, default=DEFAULT_CANDIDATE_SUMMARY)
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--abcd", type=Path, default=DEFAULT_ABCD)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inventory, summary, geometry, scores = run_audit(args.root, args.inventory, args.summary, args.geometry, args.abcd, args.report)
    parallel_status = classify_parallel_elimination(inventory, summary, scores)
    anti30_status = classify_anti_parallel_30(inventory, scores)
    print(f"Inventory files: {len(inventory)}")
    print(f"Candidate summary rows: {len(summary)}")
    print(f"Geometry rows: {len(geometry)}")
    print(f"A/B/C/D score rows: {len(scores)}")
    print(f"Parallel elimination status: {parallel_status}")
    print(f"Anti-parallel 30 status: {anti30_status}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
