"""Create the manuscript-facing reproducibility track.

This pipeline is intentionally additive. It does not move or delete exploratory
research files; it copies selected stable outputs into ``outputs/manuscript``
and writes compact manuscript-facing summary tables.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


MANUSCRIPT_ROOT = Path("outputs/manuscript")
MANUSCRIPT_METRICS = MANUSCRIPT_ROOT / "metrics"
MANUSCRIPT_REPORTS = MANUSCRIPT_ROOT / "reports"
MANUSCRIPT_FIGURES = MANUSCRIPT_ROOT / "figures"
MANUSCRIPT_README = MANUSCRIPT_ROOT / "README.md"

ITEM_STATUS_CSV = MANUSCRIPT_METRICS / "manuscript_item_status.csv"
SURVIVING_FAMILY_CSV = MANUSCRIPT_METRICS / "manuscript_surviving_family_summary.csv"
SUMMARY_REPORT = MANUSCRIPT_REPORTS / "manuscript_reproducibility_summary.md"
COVERAGE_CSV = MANUSCRIPT_METRICS / "manuscript_pipeline_coverage.csv"
COVERAGE_REPORT = MANUSCRIPT_REPORTS / "manuscript_pipeline_coverage_report.md"


@dataclass(frozen=True)
class ManuscriptItem:
    """One manuscript claim-map item."""

    item: int
    claim: str
    computational_role: str
    primary_script: str
    primary_outputs: str
    current_status: str
    caveats: str
    support_type: str


def manuscript_items() -> list[ManuscriptItem]:
    """Return the manuscript claim map for items 1-8."""
    return [
        ManuscriptItem(
            1,
            "pNAB parallel versus anti-parallel scaffold compatibility audit.",
            "Inventory and conservative compatibility/provenance audit.",
            "scripts/audit_pnab_parallel_antiparallel_models.py",
            "outputs/metrics/pnab_parallel_antiparallel_*.csv; outputs/reports/pnab_parallel_antiparallel_audit_report.md",
            "parallel insufficient_data; anti-parallel 30 plausible_candidate",
            "Current repo evidence does not eliminate parallel models or prove anti-parallel 30 degrees strongest from pNAB alone.",
            "pending_external_data",
        ),
        ManuscriptItem(
            2,
            "Experiments support Hexaplex-like assembly and beta-sheet-like peptide conformation.",
            "Modeling provides context only; it does not replace CD, gels, mass spec, or AFM.",
            "N/A",
            "Experimental manuscript data outside this computational publication track.",
            "experimental context only",
            "Experimental evidence remains primary for this item.",
            "experimental_context",
        ),
        ManuscriptItem(
            3,
            "Powder band A supports stacking rise / dominant repeat near 3.4 A with plausible compressed rise near 3.35 A.",
            "Uses diffraction peak position as a rise constraint for modeled candidate space.",
            "scripts/run_omega_clean_rise_compression_scan.py",
            "outputs/metrics/omega_clean_rise_compression_scores.csv",
            "modeling-supported context for 3.4 A start and modest compression.",
            "Band A interpretation is linked to experimental peak shape and is not alone a full structure.",
            "modeling_supported",
        ),
        ManuscriptItem(
            4,
            "Bands B-D, especially C/D, are diagnostic within modeled candidate space.",
            "C/D act as structural filters rather than incidental peaks.",
            "scripts/report_item6_candidate_filter_funnel.py",
            "outputs/metrics/item6_candidate_filter_funnel.csv",
            "C/D filter selects omega-clean rise-compressed plateau.",
            "C/D agreement is necessary but not sufficient.",
            "modeling_supported",
        ),
        ManuscriptItem(
            5,
            "C/D report a coupled fingerprint involving rise, twist/rise coupling, backbone orientation, strand placement, and packing.",
            "Summarizes model scans showing C improves with modest rise compression while D guards against over-compression.",
            "scripts/run_omega_clean_rise_compression_scan.py; scripts/run_omega_clean_torsion_boundary_scan.py",
            "outputs/reports/omega_clean_rise_compression_report.md; outputs/reports/omega_clean_torsion_boundary_report.md",
            "D remains near 7.2756 A across the plateau; 0.9700 over-compression degrades D.",
            "Not yet reduced to one isolated structural variable.",
            "modeling_supported",
        ),
        ManuscriptItem(
            6,
            "Candidate-filter funnel combines pNAB/scaffold compatibility, C/D matching, and physical/chemical-sense filters.",
            "Identifies current surviving model family under combined available filters.",
            "scripts/report_item6_candidate_filter_funnel.py",
            "outputs/metrics/item6_candidate_filter_funnel.csv; outputs/metrics/item6_surviving_model_family_summary.csv",
            "omega-clean rise-compressed plateau survives with pNAB caveat.",
            "pNAB parallel-vs-anti-parallel evidence is insufficient to eliminate parallel models.",
            "modeling_supported",
        ),
        ManuscriptItem(
            7,
            "Local torsion-boundary scan estimates the finite compatible backbone range around the surviving family.",
            "Reports measured compatible ranges by torsion family and class perturbation.",
            "scripts/run_omega_clean_torsion_boundary_scan.py",
            "outputs/metrics/omega_clean_torsion_boundary_summary.csv; outputs/reports/omega_clean_torsion_boundary_report.md",
            "phi +/-8; psi +/-10; omega +/-8; combined symmetric +/-4; opposing class +/-6; phi/psi compensation +/-8.",
            "Whether the measured range is narrow enough for manuscript framing is a PI-level interpretation.",
            "modeling_supported",
        ),
        ManuscriptItem(
            8,
            "Small rise reductions from ideal 3.4 A improve C/D match.",
            "Compares omega-clean scale 1.0000 baseline to the 0.9825-0.9725 plateau and 0.9700 over-compression.",
            "scripts/run_omega_clean_rise_compression_scan.py",
            "outputs/metrics/omega_clean_rise_compression_scores.csv; outputs/reports/omega_clean_rise_compression_report.md",
            "baseline C 5.7454 A, D 7.2756 A, error 0.1698 A; plateau C 5.6422 A, D 7.2756 A, error 0.0667 A.",
            "0.9700 over-compresses because D shifts to 7.1923 A.",
            "modeling_supported",
        ),
    ]


def claim_map_dataframe() -> pd.DataFrame:
    """Return claim-map dataframe."""
    return pd.DataFrame([item.__dict__ for item in manuscript_items()])


def ensure_dirs() -> None:
    """Create manuscript output directories."""
    for path in [MANUSCRIPT_METRICS, MANUSCRIPT_REPORTS, MANUSCRIPT_FIGURES]:
        path.mkdir(parents=True, exist_ok=True)


def copy_if_present(source: Path, destination_dir: Path) -> str:
    """Copy source to destination directory if present and return status."""
    if not source.exists():
        return "missing_data"
    destination_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination_dir / source.name)
    return "reused_existing_output"


def run_python_script(script: Path) -> str:
    """Run a lightweight Python script and return status."""
    if not script.exists():
        return "missing_data"
    subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)
    return "reran_script"


def pipeline_steps() -> list[dict[str, object]]:
    """Return conservative manuscript pipeline steps."""
    return [
        {
            "step": "A",
            "name": "pNAB parallel/anti-parallel audit",
            "script": Path("scripts/audit_pnab_parallel_antiparallel_models.py"),
            "outputs": [
                Path("outputs/metrics/pnab_parallel_antiparallel_inventory.csv"),
                Path("outputs/metrics/pnab_parallel_antiparallel_candidate_summary.csv"),
                Path("outputs/metrics/pnab_parallel_antiparallel_geometry.csv"),
                Path("outputs/metrics/pnab_parallel_antiparallel_abcd_scores.csv"),
                Path("outputs/reports/pnab_parallel_antiparallel_audit_report.md"),
            ],
            "default_action": "reuse_existing_output",
        },
        {
            "step": "B",
            "name": "guarded full-chain prototype",
            "script": Path("scripts/build_guarded_full_chain_prototype.py"),
            "outputs": [
                Path("outputs/metrics/guarded_full_chain_prototype_segment_summary.csv"),
                Path("outputs/metrics/guarded_full_chain_prototype_chain_summary.csv"),
                Path("outputs/metrics/guarded_full_chain_prototype_geometry.csv"),
                Path("outputs/metrics/guarded_full_chain_prototype_abcd_scores.csv"),
                Path("outputs/reports/guarded_full_chain_prototype_report.md"),
            ],
            "default_action": "reuse_existing_output",
        },
        {
            "step": "C",
            "name": "omega-clean rise-compression scan",
            "script": Path("scripts/run_omega_clean_rise_compression_scan.py"),
            "outputs": [
                Path("outputs/metrics/omega_clean_rise_compression_scores.csv"),
                Path("outputs/metrics/omega_clean_rise_compression_geometry.csv"),
                Path("outputs/reports/omega_clean_rise_compression_report.md"),
            ],
            "default_action": "reuse_existing_output",
        },
        {
            "step": "D",
            "name": "omega-clean torsion-boundary scan",
            "script": Path("scripts/run_omega_clean_torsion_boundary_scan.py"),
            "outputs": [
                Path("outputs/metrics/omega_clean_torsion_boundary_scores.csv"),
                Path("outputs/metrics/omega_clean_torsion_boundary_geometry.csv"),
                Path("outputs/metrics/omega_clean_torsion_boundary_summary.csv"),
                Path("outputs/reports/omega_clean_torsion_boundary_report.md"),
            ],
            "default_action": "reuse_existing_output",
        },
        {
            "step": "E",
            "name": "item 6 candidate-filter funnel",
            "script": Path("scripts/report_item6_candidate_filter_funnel.py"),
            "outputs": [
                Path("outputs/metrics/item6_candidate_filter_funnel.csv"),
                Path("outputs/metrics/item6_surviving_model_family_summary.csv"),
                Path("outputs/reports/item6_candidate_filter_funnel_report.md"),
            ],
            "default_action": "rerun_script",
        },
    ]


def destination_for(source: Path) -> Path:
    """Return manuscript destination directory for a source output."""
    if "reports" in source.parts:
        return MANUSCRIPT_REPORTS
    if "figures" in source.parts:
        return MANUSCRIPT_FIGURES
    return MANUSCRIPT_METRICS


def execute_step(step: dict[str, object]) -> dict[str, object]:
    """Execute or reuse one pipeline step."""
    action = str(step["default_action"])
    script = Path(step["script"])
    outputs = [Path(path) for path in step["outputs"]]  # type: ignore[index]
    if action == "rerun_script":
        script_status = run_python_script(script)
    else:
        script_status = "not_rerun_reused_existing_output"

    copy_statuses = [copy_if_present(path, destination_for(path)) for path in outputs]
    if all(status == "reused_existing_output" for status in copy_statuses):
        output_status = "reused_existing_output" if action != "rerun_script" else "generated_and_copied"
    elif any(status == "reused_existing_output" for status in copy_statuses):
        output_status = "partial_reused_existing_output"
    else:
        output_status = "missing_data"
    return {
        "pipeline_step": step["step"],
        "pipeline_name": step["name"],
        "primary_script": str(script),
        "script_status": script_status,
        "output_status": output_status,
        "copied_output_count": sum(status == "reused_existing_output" for status in copy_statuses),
        "missing_output_count": sum(status == "missing_data" for status in copy_statuses),
    }


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    """Read CSV if available."""
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def item_status_dataframe(step_rows: Iterable[dict[str, object]]) -> pd.DataFrame:
    """Build item status table by combining claim map and pipeline status."""
    claim_map = claim_map_dataframe()
    step_df = pd.DataFrame(list(step_rows))
    status_by_item = {
        1: "partial_pnab_scaffold_filter_parallel_not_eliminated",
        2: "experimental_context",
        3: "modeling_context_available",
        4: "cd_filter_available",
        5: "coupled_fingerprint_supported_by_current_scans",
        6: "surviving_family_summarized",
        7: "measured_compatible_range_reported",
        8: "rise_compression_plateau_reported",
    }
    claim_map["publication_track_status"] = claim_map["item"].map(status_by_item)
    claim_map["pipeline_status"] = claim_map["item"].map(
        {
            1: step_df.loc[step_df["pipeline_step"] == "A", "output_status"].iloc[0] if not step_df.empty else "missing_data",
            6: step_df.loc[step_df["pipeline_step"] == "E", "output_status"].iloc[0] if not step_df.empty else "missing_data",
            7: step_df.loc[step_df["pipeline_step"] == "D", "output_status"].iloc[0] if not step_df.empty else "missing_data",
            8: step_df.loc[step_df["pipeline_step"] == "C", "output_status"].iloc[0] if not step_df.empty else "missing_data",
        }
    ).fillna("documented_context")
    return claim_map


def surviving_family_dataframe(source: Path = Path("outputs/metrics/item6_surviving_model_family_summary.csv")) -> pd.DataFrame:
    """Return manuscript surviving-family summary."""
    df = read_csv_or_empty(source)
    if df.empty:
        return pd.DataFrame(
            [
                {
                    "surviving_model_family": "missing_data",
                    "status": "missing_data",
                    "caveat": "Item 6 surviving-family summary is missing.",
                }
            ]
        )
    df = df.copy()
    df["publication_track_note"] = (
        "The omega-clean rise-compressed plateau is the current manuscript-track survivor under available "
        "C/D and physical/chemical filters, with pNAB caveat preserved."
    )
    return df


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    """Return a compact markdown table."""
    if df.empty:
        return "_None._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in df.iterrows():
        values = [str(row.get(column, "")) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_manuscript_readme() -> None:
    """Write outputs/manuscript README."""
    MANUSCRIPT_README.write_text(
        """# Manuscript Output Track

This directory contains copied or generated manuscript-facing outputs from the current repository. It is a publication track, not a cleanup of the exploratory workspace.

- `metrics/`: compact CSV summaries and copied stable metrics.
- `reports/`: manuscript-facing reports and copied stable reports.
- `figures/`: reserved for stable manuscript-facing figures.

No exploratory files were deleted or moved. Missing inputs are reported as `missing_data`; existing stable outputs copied into this tree are marked `reused_existing_output`.
""",
        encoding="utf-8",
    )


def build_summary_report(item_status: pd.DataFrame, survivor: pd.DataFrame, step_status: pd.DataFrame) -> str:
    """Build manuscript reproducibility summary report."""
    item_cols = ["item", "claim", "support_type", "publication_track_status", "pipeline_status", "caveats"]
    step_cols = ["pipeline_step", "pipeline_name", "script_status", "output_status", "copied_output_count", "missing_output_count"]
    survivor_cols = list(survivor.columns)
    return f"""# Manuscript Reproducibility Summary

This publication track is the clean entry point for manuscript-supporting computational results while the repository still contains exploratory research work. No exploratory files were deleted or moved.

## How To Read This Track

- `reused_existing_output` means the pipeline found and copied a stable current output.
- `generated_and_copied` means the pipeline reran a lightweight summary step and copied the result.
- `missing_data` means an expected script or output was absent and no value was invented.

## Pipeline Step Status

{markdown_table(step_status, step_cols)}

## Manuscript Item Status

{markdown_table(item_status[item_cols], item_cols)}

## Surviving Model Family

{markdown_table(survivor, survivor_cols)}

## Core Caveats

- pNAB is treated as a compatibility/scaffold filter, not final structural proof.
- The current pNAB audit reports `insufficient_data` for parallel elimination.
- Anti-parallel 30 degrees is a plausible candidate, not proven strongest solely from current pNAB data.
- C/D agreement is necessary but not sufficient; physical/chemical-sense filters are required.
- The omega-clean rise-compressed plateau is the current surviving family under the available filters.
- Whether the measured compatible range is sufficiently narrow for manuscript framing is a PI-level interpretation.

## Expected Outputs

- `outputs/manuscript/metrics/manuscript_item_status.csv`
- `outputs/manuscript/metrics/manuscript_surviving_family_summary.csv`
- `outputs/manuscript/reports/manuscript_reproducibility_summary.md`
"""


def run_pipeline() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the manuscript reproducibility pipeline."""
    ensure_dirs()
    step_rows = [execute_step(step) for step in pipeline_steps()]
    step_status = pd.DataFrame(step_rows)
    item_status = item_status_dataframe(step_rows)
    survivor = surviving_family_dataframe()

    item_status.to_csv(ITEM_STATUS_CSV, index=False)
    survivor.to_csv(SURVIVING_FAMILY_CSV, index=False)
    write_manuscript_readme()
    SUMMARY_REPORT.write_text(build_summary_report(item_status, survivor, step_status), encoding="utf-8")
    from scripts.audit_manuscript_pipeline_coverage import run as run_coverage

    coverage = run_coverage(COVERAGE_CSV, COVERAGE_REPORT)
    step_status = pd.concat(
        [
            step_status,
            pd.DataFrame(
                [
                    {
                        "pipeline_step": "F",
                        "pipeline_name": "manuscript pipeline coverage audit",
                        "primary_script": "scripts/audit_manuscript_pipeline_coverage.py",
                        "script_status": "reran_script",
                        "output_status": "generated_and_copied",
                        "copied_output_count": len(coverage),
                        "missing_output_count": int((coverage["status"] == "missing_expected_report").sum()) if not coverage.empty else 0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    SUMMARY_REPORT.write_text(build_summary_report(item_status, survivor, step_status), encoding="utf-8")
    return item_status, survivor, step_status


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run the manuscript reproducibility pipeline.")
    parser.parse_args()
    item_status, survivor, step_status = run_pipeline()
    print(f"Manuscript item rows: {len(item_status)}")
    print(f"Pipeline steps: {len(step_status)}")
    if not survivor.empty:
        print(f"Surviving family: {survivor.iloc[0].get('surviving_model_family')}")
        print(f"Status: {survivor.iloc[0].get('status')}")
    print(f"Summary report: {SUMMARY_REPORT}")


if __name__ == "__main__":
    main()
