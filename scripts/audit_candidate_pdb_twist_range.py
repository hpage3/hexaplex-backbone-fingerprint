"""Inventory candidate PDB twist/rise provenance across old and current sets.

This is an inventory/provenance audit, not a new structural conclusion.  Its
main purpose is to make sure older pNAB-derived twist-grid candidates are not
mistakenly excluded when comparing the current 30-degree-like publication-track
models against the wider candidate history.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


OUT_INVENTORY = Path("outputs/metrics/candidate_pdb_twist_inventory.csv")
OUT_CROSSWALK = Path("outputs/metrics/candidate_pdb_twist_abcd_crosswalk.csv")
OUT_SUMMARY = Path("outputs/metrics/candidate_pdb_twist_range_summary.csv")
OUT_REPORT = Path("outputs/reports/candidate_pdb_twist_range_audit_report.md")

PDB_SEARCH_ROOTS = [
    Path("inputs"),
    Path("outputs/coordinates"),
    Path("outputs/parametric_six_strand_peptide_plane_models"),
    Path("outputs/parametric_six_strand_peptide_plane_models_refined"),
    Path("outputs/parametric_six_strand_peptide_plane_models_alternating_zoffset"),
]

METRIC_FILES = [
    Path("outputs/metrics/pnab_parallel_antiparallel_inventory.csv"),
    Path("outputs/metrics/pnab_parallel_antiparallel_candidate_summary.csv"),
    Path("outputs/metrics/pnab_parallel_antiparallel_geometry.csv"),
    Path("outputs/metrics/omega_clean_rise_compression_scores.csv"),
    Path("outputs/metrics/guarded_full_chain_prototype_abcd_scores.csv"),
    Path("outputs/metrics/item6_candidate_filter_funnel.csv"),
    Path("outputs/metrics/twist_tightening_candidate_filters.csv"),
]


def normalize_path_text(value: str | Path) -> str:
    """Return a stable lower-case path-ish key."""
    return str(value).replace("/", "\\").lower()


def parse_number_token(token: str) -> float:
    """Parse file-name numeric tokens such as 3p38 or 30."""
    return float(token.lower().replace("p", "."))


def extract_twist_deg(path_or_name: str | Path) -> tuple[float | None, str]:
    """Infer twist from explicit path/name tokens.

    Numeric pNAB twist-grid directories are accepted only in the known
    sidechains_tleap_structures tree.
    """
    text = normalize_path_text(path_or_name)
    patterns = [
        (r"(?:^|[_\-\s\\])h?_?twist[_\-\s]?(\d+(?:p\d+|\.\d+)?)", "twist_token"),
        (r"(?:^|[_\-\s\\])tw(\d+(?:p\d+|\.\d+)?)", "tw_token"),
        (r"(?:^|[_\-\s\\])(\d+(?:p\d+|\.\d+)?)\s*deg(?:ree)?s?(?:[_\-\s\\.]|$)", "degree_token"),
        (r"(?:^|[_\-\s\\])(\d+(?:p\d+|\.\d+)?)_degree(?:[_\-\s\\.]|$)", "degree_token"),
    ]
    for pattern, source in patterns:
        match = re.search(pattern, text)
        if match:
            return parse_number_token(match.group(1)), source

    if "github_research_sidechains_tleap_structures" in text:
        parts = re.split(r"[\\/]+", str(path_or_name))
        for index, part in enumerate(parts):
            if part.lower() == "github_research_sidechains_tleap_structures" and index + 1 < len(parts):
                candidate = parts[index + 1]
                if re.fullmatch(r"\d+(?:\.\d+)?", candidate):
                    return float(candidate), "pnab_twist_grid_directory"
    return None, "unknown"


def extract_rise_A(path_or_name: str | Path) -> tuple[float | None, str]:
    """Infer rise from explicit path/name tokens."""
    text = normalize_path_text(path_or_name)
    patterns = [
        (r"(?:^|[_\-\s\\])h?_?rise[_\-\s]?(\d+(?:p\d+|\.\d+)?)", "rise_token"),
        (r"(?:^|[_\-\s\\])(3p\d+)(?:[_\-\s\\.]|$)", "p_token"),
        (r"(?:^|[_\-\s\\])(3\.\d+)(?:[_\-\s\\.]|$)", "decimal_token"),
    ]
    for pattern, source in patterns:
        match = re.search(pattern, text)
        if match:
            return parse_number_token(match.group(1)), source
    return None, "unknown"


def classify_candidate_set(path_or_name: str | Path) -> str:
    """Classify candidate provenance from conservative path/name evidence."""
    text = normalize_path_text(path_or_name)
    if "github_research_sidechains_tleap_structures" in text or "asem_original_3p4" in text:
        return "pnab_3p38_twist_grid"
    if "selected" in text and "345" in text:
        return "selected_from_345"
    if "omega_clean" in text or "guarded_full_chain" in text or "publication" in text or "parent_derived" in text:
        return "omega_clean_publication_track"
    if "abcd" in text or "powder" in text or "parametric_six_strand_peptide_plane_models" in text:
        return "powder_scored_candidate"
    return "unknown"


def count_pdb_atoms(path: Path) -> tuple[int, int]:
    """Count ATOM/HETATM records and hydrogen records in a PDB."""
    atom_count = 0
    hydrogen_count = 0
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if line.startswith(("ATOM", "HETATM")):
                    atom_count += 1
                    element = line[76:78].strip() if len(line) >= 78 else ""
                    atom_name = line[12:16].strip()
                    if element.upper() == "H" or atom_name.upper().startswith("H"):
                        hydrogen_count += 1
    except OSError:
        return 0, 0
    return atom_count, hydrogen_count


def iter_candidate_pdbs(search_roots: list[Path]) -> list[Path]:
    """Find candidate PDBs under focused coordinate roots."""
    found: list[Path] = []
    skip_dirs = {".git", ".pytest_cache", ".venv", "__pycache__"}
    for root in search_roots:
        abs_root = ROOT / root
        if not abs_root.exists():
            continue
        for path in abs_root.rglob("*.pdb"):
            if any(part in skip_dirs for part in path.parts):
                continue
            found.append(path)
    return sorted(found)


def inventory_row(path: Path) -> dict[str, Any]:
    """Build one candidate PDB inventory row."""
    rel = path.relative_to(ROOT)
    twist, twist_source = extract_twist_deg(rel)
    rise, rise_source = extract_rise_A(rel)
    candidate_set = classify_candidate_set(rel)
    atom_count, hydrogen_count = count_pdb_atoms(path)
    return {
        "candidate_id": path.stem if path.name != "initial.pdb" else path.parent.name,
        "path": str(rel),
        "filename": path.name,
        "candidate_set": candidate_set,
        "inferred_twist_deg": twist,
        "twist_source": twist_source,
        "inferred_rise_A": rise,
        "rise_source": rise_source,
        "atom_count": atom_count,
        "hydrogen_count": hydrogen_count,
        "file_size_bytes": path.stat().st_size,
    }


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    """Read a CSV if present."""
    full = ROOT / path
    if not full.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(full)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def first_present(row: pd.Series, names: list[str]) -> Any:
    """Return first non-null row value among names."""
    for name in names:
        if name in row and not pd.isna(row[name]):
            return row[name]
    return None


def score_rows_from_metric_file(path: Path) -> list[dict[str, Any]]:
    """Extract scoring/provenance rows from known metric CSVs."""
    df = read_csv_or_empty(path)
    if df.empty:
        return []
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        source_path = first_present(row, ["coordinate_path", "path", "input_pdb"])
        candidate_name = first_present(row, ["variant_id", "prototype_id", "candidate_name", "path"])
        key_text = str(source_path or candidate_name or "")
        twist = first_present(row, ["inferred_twist_deg", "twist_deg"])
        rise = first_present(row, ["inferred_rise_A", "rise_A", "nominal_rise_equiv_A"])
        candidate_set = classify_candidate_set(key_text or str(path))
        if twist is None:
            twist, _ = extract_twist_deg(key_text)
        if twist is None and candidate_set == "omega_clean_publication_track":
            twist = 30.0
        if rise is None:
            rise, _ = extract_rise_A(key_text)
        c_peak = first_present(row, ["observed_C_d_A", "C_peak_A", "C_peak_d_A"])
        d_peak = first_present(row, ["observed_D_d_A", "D_peak_A", "D_peak_d_A"])
        combined = first_present(row, ["combined_CD_abs_error_A", "combined_abs_error_A"])
        rows.append(
            {
                "metric_file": str(path),
                "candidate_name": candidate_name,
                "coordinate_path": source_path,
                "candidate_set": candidate_set,
                "inferred_twist_deg": twist,
                "inferred_rise_A": rise,
                "C_peak_A": c_peak,
                "D_peak_A": d_peak,
                "combined_CD_abs_error_A": combined,
                "scoring_available": c_peak is not None and d_peak is not None,
            }
        )
    return rows


def build_crosswalk(metric_files: list[Path] = METRIC_FILES) -> pd.DataFrame:
    """Build A/B/C/D scoring crosswalk from existing metric outputs."""
    rows: list[dict[str, Any]] = []
    for path in metric_files:
        rows.extend(score_rows_from_metric_file(path))
    return pd.DataFrame(rows)


def sorted_unique_text(values: pd.Series) -> str:
    """Return semicolon-joined numeric unique values."""
    numeric = pd.to_numeric(values, errors="coerce").dropna().unique()
    if len(numeric) == 0:
        return ""
    return ";".join(f"{value:g}" for value in sorted(numeric))


def summarize_inventory(inventory: pd.DataFrame, crosswalk: pd.DataFrame) -> pd.DataFrame:
    """Summarize twist/rise ranges by candidate set."""
    rows: list[dict[str, Any]] = []
    all_sets = sorted(set(inventory.get("candidate_set", [])) | set(crosswalk.get("candidate_set", [])))
    for candidate_set in all_sets:
        inv = inventory[inventory["candidate_set"] == candidate_set] if not inventory.empty else pd.DataFrame()
        scored = crosswalk[crosswalk["candidate_set"] == candidate_set] if not crosswalk.empty else pd.DataFrame()
        twists = pd.concat(
            [
                pd.to_numeric(inv.get("inferred_twist_deg", pd.Series(dtype=float)), errors="coerce"),
                pd.to_numeric(scored.get("inferred_twist_deg", pd.Series(dtype=float)), errors="coerce"),
            ],
            ignore_index=True,
        ).dropna()
        rises = pd.concat(
            [
                pd.to_numeric(inv.get("inferred_rise_A", pd.Series(dtype=float)), errors="coerce"),
                pd.to_numeric(scored.get("inferred_rise_A", pd.Series(dtype=float)), errors="coerce"),
            ],
            ignore_index=True,
        ).dropna()
        scored_available = scored[scored.get("scoring_available", False) == True] if not scored.empty else pd.DataFrame()
        best_name = ""
        best_error = math.nan
        if not scored_available.empty and "combined_CD_abs_error_A" in scored_available:
            errors = pd.to_numeric(scored_available["combined_CD_abs_error_A"], errors="coerce")
            if errors.notna().any():
                idx = errors.idxmin()
                best_error = float(errors.loc[idx])
                best_name = str(scored_available.loc[idx].get("candidate_name", ""))
        rows.append(
            {
                "candidate_set": candidate_set,
                "pdb_count": int(len(inv)),
                "scored_row_count": int(len(scored_available)),
                "twist_min_deg": float(twists.min()) if not twists.empty else math.nan,
                "twist_max_deg": float(twists.max()) if not twists.empty else math.nan,
                "twist_unique_values": ";".join(f"{value:g}" for value in sorted(twists.unique())) if not twists.empty else "",
                "rise_min_A": float(rises.min()) if not rises.empty else math.nan,
                "rise_max_A": float(rises.max()) if not rises.empty else math.nan,
                "rise_unique_values": ";".join(f"{value:g}" for value in sorted(rises.unique())) if not rises.empty else "",
                "best_scored_candidate": best_name,
                "best_combined_CD_abs_error_A": best_error,
            }
        )
    return pd.DataFrame(rows)


def has_18_to_32_grid(summary: pd.DataFrame) -> bool:
    """Return whether pNAB grid has explicit 18-32 twist coverage."""
    sub = summary[summary["candidate_set"] == "pnab_3p38_twist_grid"]
    if sub.empty:
        return False
    values = str(sub.iloc[0].get("twist_unique_values", "")).split(";")
    twists = {int(float(value)) for value in values if value}
    return set(range(18, 33)).issubset(twists)


def build_report(inventory: pd.DataFrame, crosswalk: pd.DataFrame, summary: pd.DataFrame) -> str:
    """Build markdown provenance audit report."""
    total = len(inventory)
    grid_found = has_18_to_32_grid(summary)
    selected = summary[summary["candidate_set"] == "selected_from_345"]
    selected_found = bool(not selected.empty and int(selected.iloc[0].get("pdb_count", 0)) > 0)
    scored_count = int(crosswalk["scoring_available"].sum()) if not crosswalk.empty and "scoring_available" in crosswalk else 0

    lines = [
        "# Candidate PDB Twist Range Audit",
        "",
        "This is an inventory/provenance audit, not a new structural conclusion. It checks whether older pNAB-derived twist-grid candidates are present alongside the later publication-track and powder-scored outputs.",
        "",
        "## Scope Distinction",
        "",
        "- Diagnostic coordinate transforms remain useful for asking how geometry moves diffraction bands.",
        "- The failed pseudo reconstructed bridge is not treated as a source model.",
        "- The validated parent-derived bridge and fine parent-derived rise scan remain constrained six-fold parent-family diagnostics.",
        "- The previous twist-tightening report only covered the current omega-clean / publication-track candidate family and therefore could miss older pNAB-derived twist-grid candidates.",
        "",
        "## Inventory Summary",
        "",
        f"- Total candidate PDB files found in focused coordinate roots: {total}",
        f"- A/B/C/D scoring rows crosswalked from existing metrics: {len(crosswalk)}",
        f"- Rows with C and D scoring values: {scored_count}",
        f"- pNAB-derived 18-32 twist-grid candidates found: {'yes' if grid_found else 'no'}",
        f"- Later selected subset from 345 clearly identifiable: {'yes' if selected_found else 'no'}",
        "",
        "Filename- or path-inferred twist values should be cross-checked against the original pNAB/YAML provenance before being used quantitatively.",
        "",
        "## Twist/Rise Range by Candidate Set",
        "",
        "| candidate_set | PDBs | scored rows | twist range | rise range | twist values |",
        "| --- | ---: | ---: | --- | --- | --- |",
    ]
    for _, row in summary.sort_values("candidate_set").iterrows():
        twist_range = ""
        if not pd.isna(row.get("twist_min_deg")):
            twist_range = f"{row['twist_min_deg']:g}-{row['twist_max_deg']:g}"
        rise_range = ""
        if not pd.isna(row.get("rise_min_A")):
            rise_range = f"{row['rise_min_A']:g}-{row['rise_max_A']:g}"
        lines.append(
            f"| {row['candidate_set']} | {int(row['pdb_count'])} | {int(row['scored_row_count'])} | {twist_range} | {rise_range} | {row['twist_unique_values']} |"
        )

    lines.extend(
        [
            "",
            "## Answers",
            "",
            f"- **Does the pNAB 3.38 A / 3.4 A twist-grid candidate set include 18-32 degree candidates?** {'Yes.' if grid_found else 'Not confirmed.'} The raw path inventory should be interpreted as provenance evidence, not as validated physical scoring.",
            f"- **Is the later selected subset from 345 candidates clearly identifiable?** {'Yes.' if selected_found else 'No.'} If this subset was recorded under a different naming convention, Nick/Asem should provide the selection manifest or trace labels.",
            "- **What did the A/B/C/D crosswalk find?** Existing scoring rows are concentrated in later powder-scored or omega-clean/publication-track outputs; the raw pNAB twist-grid PDBs do not all have direct C/D score rows in the currently discovered metrics.",
            "- **Why did the previous twist-tightening report miss these candidates?** It intentionally operated on the current omega-clean / publication-track family and did not inventory the raw pNAB-derived twist-grid PDB tree.",
            "",
            "## Caution",
            "",
            "This audit does not decide which twist is physically correct. It only establishes that pNAB-derived twist-grid candidates must be included when discussing historical candidate coverage, especially the 18-32 degree range near rise 3.38 A / 3.4 A.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(inventory: pd.DataFrame, crosswalk: pd.DataFrame, summary: pd.DataFrame, report: str) -> None:
    """Write audit outputs."""
    for path in [OUT_INVENTORY, OUT_CROSSWALK, OUT_SUMMARY, OUT_REPORT]:
        (ROOT / path).parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(ROOT / OUT_INVENTORY, index=False, quoting=csv.QUOTE_MINIMAL)
    crosswalk.to_csv(ROOT / OUT_CROSSWALK, index=False, quoting=csv.QUOTE_MINIMAL)
    summary.to_csv(ROOT / OUT_SUMMARY, index=False, quoting=csv.QUOTE_MINIMAL)
    (ROOT / OUT_REPORT).write_text(report, encoding="utf-8")


def run(search_roots: list[Path] = PDB_SEARCH_ROOTS) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Run the full audit."""
    inventory = pd.DataFrame([inventory_row(path) for path in iter_candidate_pdbs(search_roots)])
    crosswalk = build_crosswalk()
    summary = summarize_inventory(inventory, crosswalk)
    report = build_report(inventory, crosswalk, summary)
    return inventory, crosswalk, summary, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-write", action="store_true", help="Run audit without writing output files.")
    args = parser.parse_args()

    inventory, crosswalk, summary, report = run()
    if not args.no_write:
        write_outputs(inventory, crosswalk, summary, report)
    print(f"candidate_pdb_count={len(inventory)}")
    print(f"crosswalk_row_count={len(crosswalk)}")
    print(f"summary_rows={len(summary)}")
    print(f"report={OUT_REPORT}")


if __name__ == "__main__":
    main()
