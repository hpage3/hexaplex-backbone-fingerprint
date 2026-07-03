"""Audit repo files for source/generative structure parameters.

This is a provenance diagnostic. It searches concise source/config/docs files
for parameter terms that could regenerate the parent Hexaflex coordinate model,
while skipping heavy generated output directories by default.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HITS_CSV = Path("outputs/metrics/structure_generation_parameter_hits.csv")
DEFAULT_SUMMARY_CSV = Path("outputs/metrics/structure_generation_parameter_audit_summary.csv")
DEFAULT_REPORT = Path("outputs/reports/structure_generation_parameter_audit.md")

SEARCH_SUFFIXES = {".py", ".md", ".txt", ".yaml", ".yml", ".json", ".sh", ".ipynb"}
SKIP_DIRS = {
    ".git",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "outputs",
    "tests",
}
MAX_FILE_BYTES = 1_000_000

TERM_TYPES = {
    "h_rise": "rise",
    "rise": "rise",
    "rise_A": "rise",
    "helical_rise": "rise",
    "h_twist": "twist",
    "twist": "twist",
    "twist_deg": "twist",
    "helical_twist": "twist",
    "radius": "radius",
    "helix_radius": "radius",
    "helix_radius_A": "radius",
    "strand_orientation": "orientation",
    "orientation": "orientation",
    "antiparallel": "orientation",
    "anti_parallel": "orientation",
    "parallel": "orientation",
    "register": "register",
    "strand_z_offset": "register",
    "z_offset": "register",
    "sequence": "sequence",
    "backbone": "backbone",
    "hexad": "hexad",
    "pNAB": "pnab",
    "pnab": "pnab",
    "HelicalParameters": "pnab",
}

TERM_PATTERNS = {
    term: re.compile(rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])", re.IGNORECASE)
    for term in TERM_TYPES
}
ASSIGNMENT_RE = re.compile(r"[:=]\s*[-+]?\d+(?:\.\d+)?")


def is_generic_parallel_context(line: str) -> bool:
    """Return whether ``parallel`` refers to execution rather than strand orientation."""
    line_lower = line.lower()
    return "parallel" in line_lower and any(token in line_lower for token in ["worker", "workers", "execution", "run in parallel"])


def is_parameter_assignment(line: str, term: str) -> bool:
    """Return whether a term appears in a direct assignment/default context."""
    assignment = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])\s*(?:[:=]|,?\s*default=)")
    return bool(assignment.search(line))


@dataclass(frozen=True)
class ScanPlan:
    """Files considered and skipped by an audit run."""

    files: list[Path]
    skipped_dirs: list[Path]
    skipped_files: list[Path]


def likely_parameter_type(term: str) -> str:
    """Return the coarse parameter class for a matched term."""
    return TERM_TYPES.get(term, "other")


def context_snippet(line: str, max_chars: int = 220) -> str:
    """Return a compact single-line context snippet."""
    snippet = " ".join(line.strip().split())
    if len(snippet) <= max_chars:
        return snippet
    return snippet[: max_chars - 3] + "..."


def matched_terms(line: str) -> list[str]:
    """Return stable term matches for one source line."""
    hits = []
    for term, pattern in TERM_PATTERNS.items():
        if not pattern.search(line):
            continue
        if term.lower() == "parallel" and is_generic_parallel_context(line):
            continue
        hits.append(term)
    return sorted(hits, key=lambda value: (likely_parameter_type(value), value.lower()))


def confidence_for_hit(path: Path, line: str, term: str) -> str:
    """Classify one hit as high/medium/low-confidence provenance evidence."""
    suffix = path.suffix.lower()
    parameter_type = likely_parameter_type(term)
    line_lower = line.lower()
    if suffix in {".yaml", ".yml", ".json"} and (ASSIGNMENT_RE.search(line) or is_parameter_assignment(line, term)):
        return "high"
    if suffix in {".py", ".sh"} and is_parameter_assignment(line, term):
        return "high"
    if suffix == ".py" and parameter_type in {"rise", "twist", "radius", "orientation", "register"}:
        if "argparse" in line_lower or "dataclass" in line_lower or "manifest" in line_lower:
            return "high"
        return "medium"
    if suffix in {".md", ".txt"}:
        return "medium" if parameter_type in {"rise", "twist", "radius", "register", "pnab"} else "low"
    return "medium"


def notes_for_hit(path: Path, line: str, term: str) -> str:
    """Return concise notes for one hit."""
    notes: list[str] = []
    if is_parameter_assignment(line, term):
        notes.append("numeric_or_assignment_context")
    if path.parts and "legacy" in path.parts:
        notes.append("legacy_file")
    if path.suffix.lower() == ".ipynb":
        notes.append("notebook_json_context")
    if "parent" in line.lower() or "ideal_hexaflex" in line.lower() or "full_hexaplex" in line.lower():
        notes.append("mentions_parent_or_ideal_model")
    return ";".join(notes)


def should_skip_dir(path: Path, skip_dirs: set[str] = SKIP_DIRS) -> bool:
    """Return whether a directory should be skipped."""
    return path.name in skip_dirs


def build_scan_plan(root: Path, skip_dirs: set[str] = SKIP_DIRS) -> ScanPlan:
    """Return deterministic candidate files and skipped paths."""
    files: list[Path] = []
    skipped_dirs: list[Path] = []
    skipped_files: list[Path] = []
    for current, dirnames, filenames in sorted_walk(root):
        kept_dirs = []
        for dirname in sorted(dirnames):
            path = current / dirname
            if should_skip_dir(path, skip_dirs):
                skipped_dirs.append(path)
            else:
                kept_dirs.append(dirname)
        dirnames[:] = kept_dirs
        for filename in sorted(filenames):
            path = current / filename
            if path.suffix.lower() not in SEARCH_SUFFIXES:
                continue
            try:
                size = path.stat().st_size
            except OSError:
                skipped_files.append(path)
                continue
            if size > MAX_FILE_BYTES:
                skipped_files.append(path)
                continue
            files.append(path)
    return ScanPlan(files=files, skipped_dirs=skipped_dirs, skipped_files=skipped_files)


def sorted_walk(root: Path):
    """Yield ``Path``-based os.walk entries in deterministic order."""
    import os

    for current, dirnames, filenames in os.walk(root):
        dirnames.sort()
        filenames.sort()
        yield Path(current), dirnames, filenames


def scan_file(path: Path, root: Path) -> list[dict[str, object]]:
    """Scan one text-like file for parameter hits."""
    rows: list[dict[str, object]] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return rows
    for line_number, line in enumerate(lines, start=1):
        for term in matched_terms(line):
            rows.append(
                {
                    "file_path": path.relative_to(root).as_posix(),
                    "line_number": line_number,
                    "matched_term": term,
                    "context_snippet": context_snippet(line),
                    "likely_parameter_type": likely_parameter_type(term),
                    "confidence": confidence_for_hit(path, line, term),
                    "notes": notes_for_hit(path, line, term),
                }
            )
    return rows


def scan_repository(root: Path = ROOT) -> tuple[pd.DataFrame, ScanPlan]:
    """Scan source/config/docs files and return hit rows plus the scan plan."""
    plan = build_scan_plan(root)
    rows: list[dict[str, object]] = []
    for path in plan.files:
        rows.extend(scan_file(path, root))
    columns = [
        "file_path",
        "line_number",
        "matched_term",
        "context_snippet",
        "likely_parameter_type",
        "confidence",
        "notes",
    ]
    return pd.DataFrame(rows, columns=columns), plan


def high_confidence_parameter_types(hits: pd.DataFrame) -> set[str]:
    """Return parameter classes with high-confidence hits."""
    if hits.empty:
        return set()
    high = hits[hits["confidence"] == "high"]
    return set(high["likely_parameter_type"].dropna().astype(str))


def strongest_candidate_files(hits: pd.DataFrame, limit: int = 12) -> pd.DataFrame:
    """Rank candidate files by hit confidence and parameter diversity."""
    if hits.empty:
        return pd.DataFrame(columns=["file_path", "hit_count", "high_count", "parameter_types"])
    hits = hits[~hits["file_path"].astype(str).str.startswith("tests/")]
    if hits.empty:
        return pd.DataFrame(columns=["file_path", "hit_count", "high_count", "parameter_types"])
    rows = []
    for file_path, group in hits.groupby("file_path", sort=True):
        types = sorted(set(group["likely_parameter_type"].astype(str)))
        rows.append(
            {
                "file_path": file_path,
                "hit_count": int(len(group)),
                "high_count": int((group["confidence"] == "high").sum()),
                "parameter_types": ",".join(types),
            }
        )
    ranked = pd.DataFrame(rows).sort_values(["high_count", "hit_count", "file_path"], ascending=[False, False, True])
    return ranked.head(limit)


def recommendation_option(hits: pd.DataFrame) -> tuple[str, str]:
    """Return recommendation option and rationale."""
    high_types = high_confidence_parameter_types(hits)
    has_core = {"rise", "twist", "radius"}.issubset(high_types)
    has_orientation_or_register = bool({"orientation", "register"} & high_types)
    parent_link = False
    if not hits.empty:
        snippets = " ".join(hits["context_snippet"].astype(str).str.lower().tolist())
        parent_link = any(token in snippets for token in ["ideal_hexaflex", "full_hexaplex", "parent baseline"])
    pnab_link = bool(not hits.empty and (hits["likely_parameter_type"] == "pnab").any())
    if has_core and has_orientation_or_register and parent_link and pnab_link:
        return (
            "Option A",
            "source parameters appear to include core helical controls and a plausible parent/pNAB provenance link",
        )
    if has_core or parent_link:
        return (
            "Option B",
            "partial provenance or reusable parameterized generators were found, but the exact parent regeneration source is not proven",
        )
    return (
        "Option C",
        "no usable source/generative provenance was found for rebuilding the parent model from explicit physical parameters",
    )


def markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    """Render a compact markdown table without optional dependencies."""
    if not rows:
        return "_None._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def summarize_hits(hits: pd.DataFrame, plan: ScanPlan) -> dict[str, str]:
    """Build machine-readable audit summary values."""
    option, rationale = recommendation_option(hits)
    type_counts = Counter(hits["likely_parameter_type"].astype(str)) if not hits.empty else Counter()
    high_types = ",".join(sorted(high_confidence_parameter_types(hits)))
    return {
        "files_scanned": str(len(plan.files)),
        "directories_skipped": str(len(plan.skipped_dirs)),
        "oversize_or_unreadable_files_skipped": str(len(plan.skipped_files)),
        "hits_found": str(len(hits)),
        "high_confidence_parameter_types": high_types,
        "hit_counts_by_parameter_type": ";".join(f"{key}:{value}" for key, value in sorted(type_counts.items())),
        "recommendation": option,
        "recommendation_rationale": rationale,
    }


def write_summary_csv(summary: dict[str, str], path: Path) -> None:
    """Write machine-readable key/value summary CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["audit_key", "audit_value"])
        writer.writeheader()
        for key, value in summary.items():
            writer.writerow({"audit_key": key, "audit_value": value})


def build_report(hits: pd.DataFrame, plan: ScanPlan, root: Path) -> str:
    """Build markdown provenance audit report."""
    summary = summarize_hits(hits, plan)
    candidates = strongest_candidate_files(hits).to_dict("records")
    explicit = (
        hits[hits["confidence"].isin(["high", "medium"])]
        .sort_values(["confidence", "file_path", "line_number"], ascending=[True, True, True])
        .head(30)
        .to_dict("records")
        if not hits.empty
        else []
    )
    type_counts = Counter(hits["likely_parameter_type"].astype(str)) if not hits.empty else Counter()
    found_types = set(type_counts)
    core_text = []
    for parameter in ["rise", "twist", "radius", "orientation", "register", "pnab"]:
        status = "found" if parameter in found_types else "not found"
        core_text.append(f"- {parameter}: {status} ({type_counts.get(parameter, 0)} hit(s))")
    option = summary["recommendation"]
    if option == "Option A":
        buildable = "A physically parameterized 3.40/3.38/3.35 A rise family appears buildable from source parameters, pending verification against the parent coordinates."
    elif option == "Option B":
        buildable = "A 3.40/3.38/3.35 A family appears plausible as a reconstructed/coordinate-derived bridge, but exact parent-source provenance is not proven by this audit."
    else:
        buildable = "A physically parameterized 3.40/3.38/3.35 A rise family does not appear buildable from recovered source provenance yet."
    skipped_dirs = ", ".join(str(path.relative_to(root)) for path in sorted(plan.skipped_dirs)[:20]) or "none"
    if len(plan.skipped_dirs) > 20:
        skipped_dirs += f", ... ({len(plan.skipped_dirs)} total)"
    searched_roots = sorted({str(path.relative_to(root).parts[0]) if path.relative_to(root).parts else "." for path in plan.files})
    return f"""# Structure Generation Parameter Audit

## Summary

- Files scanned: {summary["files_scanned"]}
- Parameter hits found: {summary["hits_found"]}
- Searched top-level paths: {", ".join(searched_roots) if searched_roots else "none"}
- Skipped generated/heavy directories: {skipped_dirs}
- Oversize or unreadable files skipped: {summary["oversize_or_unreadable_files_skipped"]}

## Strongest Candidate Source/Generative Files

{markdown_table(candidates, ["file_path", "hit_count", "high_count", "parameter_types"])}

## Explicit Parameters Found

{chr(10).join(core_text)}

Representative line-level hits:

{markdown_table(explicit, ["file_path", "line_number", "matched_term", "likely_parameter_type", "confidence", "context_snippet"])}

## Can Source Parameters Be Changed?

This audit found the terms above in source/config/docs files. Hits in source scripts indicate whether parameterized or reconstructed generators exist in this repo; they do not by themselves prove that the current parent coordinate model was originally generated from those exact parameters.

- Rise/twist/radius/orientation/register changeability: inferred from the hit table above.
- pNAB/source provenance: requires a direct pNAB input/YAML/JSON or script tying the parent model to source generation settings.
- 3.40/3.38/3.35 A rise family: {buildable}

## Recommendation

{option}: {summary["recommendation_rationale"]}.

Use Option A only if subsequent manual inspection confirms exact source parameters for the parent/current model. Otherwise, keep the current best `parameterized_rise_0p9750` result labeled as an effective diagnostic coordinate transform until a chemically/register-defined or source-regenerated model is available.
"""


def run_audit(
    root: Path = ROOT,
    hits_csv: Path = DEFAULT_HITS_CSV,
    summary_csv: Path = DEFAULT_SUMMARY_CSV,
    report_path: Path = DEFAULT_REPORT,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Run the parameter provenance audit and write outputs."""
    hits, plan = scan_repository(root)
    hits_csv.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    hits.to_csv(hits_csv, index=False)
    summary = summarize_hits(hits, plan)
    write_summary_csv(summary, summary_csv)
    report_path.write_text(build_report(hits, plan, root), encoding="utf-8")
    return hits, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--hits-csv", type=Path, default=DEFAULT_HITS_CSV)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    hits, summary = run_audit(args.root, args.hits_csv, args.summary_csv, args.report)
    print(f"Wrote {args.hits_csv}")
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.report}")
    print(f"Files scanned: {summary['files_scanned']}")
    print(f"Hits found: {len(hits)}")
    print(f"Recommendation: {summary['recommendation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
