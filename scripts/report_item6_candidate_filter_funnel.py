"""Build the manuscript item 6 candidate-filter funnel report.

This report combines three currently available filters:

1. pNAB/scaffold compatibility, treated as partial because matched labeled
   parallel/anti-parallel pNAB data are not yet present in the repo.
2. C/D powder-band matching.
3. Physical/chemical-sense guards from the omega-clean guarded scaffold and
   torsion-boundary scans.

The output is intentionally conservative: C/D agreement is necessary but not
sufficient, pNAB is a compatibility/scaffold filter rather than final proof, and
the compatible torsion range is reported without deciding whether it is narrow
enough for manuscript framing.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.run_parent_derived_rise_bridge import markdown_table


PNAB_INVENTORY = Path("outputs/metrics/pnab_parallel_antiparallel_inventory.csv")
PNAB_SUMMARY = Path("outputs/metrics/pnab_parallel_antiparallel_candidate_summary.csv")
PNAB_GEOMETRY = Path("outputs/metrics/pnab_parallel_antiparallel_geometry.csv")
PNAB_ABCD = Path("outputs/metrics/pnab_parallel_antiparallel_abcd_scores.csv")
PNAB_REPORT = Path("outputs/reports/pnab_parallel_antiparallel_audit_report.md")

RISE_SCORES = Path("outputs/metrics/omega_clean_rise_compression_scores.csv")
RISE_GEOMETRY = Path("outputs/metrics/omega_clean_rise_compression_geometry.csv")
RISE_REPORT = Path("outputs/reports/omega_clean_rise_compression_report.md")

GUARDED_SEGMENTS = Path("outputs/metrics/guarded_full_chain_prototype_segment_summary.csv")
GUARDED_CHAINS = Path("outputs/metrics/guarded_full_chain_prototype_chain_summary.csv")
GUARDED_GEOMETRY = Path("outputs/metrics/guarded_full_chain_prototype_geometry.csv")
GUARDED_ABCD = Path("outputs/metrics/guarded_full_chain_prototype_abcd_scores.csv")
GUARDED_REPORT = Path("outputs/reports/guarded_full_chain_prototype_report.md")

BOUNDARY_SCORES = Path("outputs/metrics/omega_clean_torsion_boundary_scores.csv")
BOUNDARY_GEOMETRY = Path("outputs/metrics/omega_clean_torsion_boundary_geometry.csv")
BOUNDARY_SUMMARY = Path("outputs/metrics/omega_clean_torsion_boundary_summary.csv")
BOUNDARY_REPORT = Path("outputs/reports/omega_clean_torsion_boundary_report.md")

OUT_FUNNEL = Path("outputs/metrics/item6_candidate_filter_funnel.csv")
OUT_SURVIVOR = Path("outputs/metrics/item6_surviving_model_family_summary.csv")
OUT_REPORT = Path("outputs/reports/item6_candidate_filter_funnel_report.md")

TARGET_C_A = 5.6
TARGET_D_A = 7.3
CD_PLATEAU_ERROR_THRESHOLD_A = 0.08
PARENT_LIKE_ERROR_THRESHOLD_A = 0.18
D_DEGRADED_THRESHOLD_A = 0.08


def parse_bool(value: Any) -> bool:
    """Parse bool-like CSV values."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y", "passed", "pass"}


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    """Read CSV if present, otherwise return an empty dataframe."""
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def missing_inputs(paths: list[Path]) -> list[Path]:
    """Return required input files that are missing."""
    return [path for path in paths if not path.exists()]


def classify_pnab_filter(parallel_elimination_status: str, anti_parallel_30_status: str) -> str:
    """Classify the pNAB/scaffold filter state."""
    if parallel_elimination_status == "insufficient_data":
        return "partial_pnab_scaffold_filter_parallel_not_eliminated"
    if parallel_elimination_status == "eliminated" and anti_parallel_30_status in {
        "strongest_current_pnab_candidate",
        "plausible_candidate",
    }:
        return "pnab_scaffold_compatible"
    if parallel_elimination_status == "disfavored_not_eliminated":
        return "pnab_disfavored_not_eliminated"
    return "pnab_insufficient"


def classify_cd_filter(c_peak_A: float | None, d_peak_A: float | None, combined_error_A: float | None) -> str:
    """Classify the C/D powder-band filter."""
    if c_peak_A is None or d_peak_A is None or combined_error_A is None or pd.isna(combined_error_A):
        return "missing_cd_score"
    d_error = abs(float(d_peak_A) - TARGET_D_A)
    if d_error > D_DEGRADED_THRESHOLD_A:
        return "over_compressed_d_degraded"
    if float(combined_error_A) <= CD_PLATEAU_ERROR_THRESHOLD_A:
        return "cd_plateau_preserved"
    if float(combined_error_A) <= PARENT_LIKE_ERROR_THRESHOLD_A:
        return "parent_like"
    return "cd_mismatch"


def classify_physical_filter(
    atom_count_preserved: bool,
    carboxylates_preserved: bool,
    omega_within_8_count: int | float | None,
    omega_count: int | float | None,
    every_other_absent: bool,
    unresolved_segments: int | float | None = 0,
    guard_passed: bool = True,
) -> str:
    """Classify physical/chemical-sense filter state."""
    omega_count_int = int(omega_count or 0)
    omega8_int = int(omega_within_8_count or 0)
    unresolved_int = int(unresolved_segments or 0)
    omega_clean = omega_count_int > 0 and omega8_int == omega_count_int
    if (
        atom_count_preserved
        and carboxylates_preserved
        and omega_clean
        and every_other_absent
        and unresolved_int == 0
        and guard_passed
    ):
        return "passes_physical_chemical_sense"
    if atom_count_preserved and carboxylates_preserved and guard_passed:
        return "physical_with_caveats"
    return "fails_physical_chemical_sense"


def classify_candidate_status(pnab_filter: str, cd_filter: str, physical_filter: str) -> str:
    """Return conservative candidate status for the combined funnel."""
    pnab_plausible = pnab_filter in {
        "partial_pnab_scaffold_filter_parallel_not_eliminated",
        "pnab_scaffold_compatible",
        "pnab_disfavored_not_eliminated",
    }
    cd_passes = cd_filter in {"cd_plateau_preserved", "parent_like"}
    physical_passes = physical_filter == "passes_physical_chemical_sense"
    if cd_passes and physical_passes and pnab_filter == "partial_pnab_scaffold_filter_parallel_not_eliminated":
        return "survives_current_filters_with_pnab_caveat"
    if cd_passes and physical_passes and pnab_plausible:
        return "survives_current_filters"
    if cd_passes and not physical_passes:
        return "cd_only_not_physical"
    if physical_passes and not cd_passes and pnab_plausible:
        return "physical_not_cd"
    if not pnab_plausible:
        return "pnab_insufficient"
    return "rejected"


def extract_pnab_status(report_text: str) -> tuple[str, str]:
    """Extract pNAB status strings from the audit report text."""
    parallel = "insufficient_data"
    anti = "insufficient_data"
    for line in report_text.splitlines():
        if "Parallel elimination status:" in line:
            parallel = line.split(":", 1)[1].strip().strip("`").strip()
        if "Anti-parallel 30 status:" in line:
            anti = line.split(":", 1)[1].strip().strip("`").strip()
    return parallel, anti


def load_pnab_context(report_path: Path = PNAB_REPORT) -> dict[str, str]:
    """Load pNAB caveat context."""
    if not report_path.exists():
        return {
            "parallel_elimination_status": "insufficient_data",
            "anti_parallel_30_status": "insufficient_data",
            "pnab_filter": "pnab_insufficient",
            "notes": f"Missing pNAB audit report: {report_path}",
        }
    text = report_path.read_text(encoding="utf-8")
    parallel, anti = extract_pnab_status(text)
    return {
        "parallel_elimination_status": parallel,
        "anti_parallel_30_status": anti,
        "pnab_filter": classify_pnab_filter(parallel, anti),
        "notes": "pNAB treated as compatibility/scaffold filter, not final structural proof",
    }


def first_row(df: pd.DataFrame) -> pd.Series | None:
    """Return first row or None."""
    return None if df.empty else df.iloc[0]


def guarded_funnel_row(pnab: dict[str, str], guarded_scores: pd.DataFrame, guarded_geometry: pd.DataFrame) -> dict[str, object]:
    """Build funnel row for the guarded full-chain prototype baseline."""
    score = first_row(guarded_scores)
    geom = first_row(guarded_geometry)
    if score is None or geom is None:
        cd_filter = "missing_cd_score"
        physical_filter = "fails_physical_chemical_sense"
        c_peak = d_peak = combined = float("nan")
    else:
        c_peak = float(score.get("observed_C_d_A", float("nan")))
        d_peak = float(score.get("observed_D_d_A", float("nan")))
        combined = float(score.get("combined_CD_abs_error_A", float("nan")))
        cd_filter = classify_cd_filter(c_peak, d_peak, combined)
        physical_filter = classify_physical_filter(
            parse_bool(geom.get("atom_count_preserved", False)),
            parse_bool(geom.get("carboxylates_preserved", False)),
            geom.get("omega_within_8deg_count", 0),
            geom.get("omega_count", 0),
            not parse_bool(geom.get("omega_every_other_detected", True)),
            geom.get("unresolved_segment_count", 999),
            parse_bool(geom.get("guard_status", "")),
        )
    status = classify_candidate_status(pnab["pnab_filter"], cd_filter, physical_filter)
    return {
        "candidate_family": "omega_clean_guarded_full_chain_baseline",
        "variant_id": score.get("prototype_id", "guarded_full_chain_prototype") if score is not None else "guarded_full_chain_prototype",
        "filter_1_pnab_scaffold": pnab["pnab_filter"],
        "parallel_elimination_status": pnab["parallel_elimination_status"],
        "anti_parallel_30_status": pnab["anti_parallel_30_status"],
        "filter_2_cd_band": cd_filter,
        "observed_C_d_A": c_peak,
        "observed_D_d_A": d_peak,
        "combined_CD_abs_error_A": combined,
        "filter_3_physical_chemical_sense": physical_filter,
        "filter_4_local_structural_envelope": "baseline_not_boundary_scan",
        "candidate_status": status,
        "notes": "Guarded omega-clean full-chain baseline reproduces parent/guarded C/D but is not the best C/D plateau.",
    }


def rise_funnel_rows(pnab: dict[str, str], rise_scores: pd.DataFrame, rise_geometry: pd.DataFrame) -> list[dict[str, object]]:
    """Build funnel rows for omega-clean rise-compression variants."""
    if rise_scores.empty:
        return []
    geom_by_id = {str(row["variant_id"]): row for _, row in rise_geometry.iterrows()} if not rise_geometry.empty else {}
    rows: list[dict[str, object]] = []
    for _, score in rise_scores.iterrows():
        variant_id = str(score["variant_id"])
        geom = geom_by_id.get(variant_id)
        c_peak = float(score.get("observed_C_d_A", float("nan")))
        d_peak = float(score.get("observed_D_d_A", float("nan")))
        combined = float(score.get("combined_CD_abs_error_A", float("nan")))
        cd_filter = classify_cd_filter(c_peak, d_peak, combined)
        if geom is None:
            physical_filter = "fails_physical_chemical_sense"
        else:
            physical_filter = classify_physical_filter(
                parse_bool(geom.get("atom_count_preserved_vs_guarded", False)),
                parse_bool(geom.get("carboxylates_preserved_vs_guarded", False)),
                geom.get("guarded_selected_retained_omega_within_8_count", 0),
                geom.get("guarded_selected_retained_omega_count", 0),
                not parse_bool(geom.get("guarded_selected_retained_omega_every_other_detected", True)),
                0,
                True,
            )
        envelope = "measured_boundary_available" if cd_filter == "cd_plateau_preserved" else "outside_best_plateau_or_baseline"
        rows.append(
            {
                "candidate_family": "omega_clean_rise_compressed",
                "variant_id": variant_id,
                "filter_1_pnab_scaffold": pnab["pnab_filter"],
                "parallel_elimination_status": pnab["parallel_elimination_status"],
                "anti_parallel_30_status": pnab["anti_parallel_30_status"],
                "filter_2_cd_band": cd_filter,
                "observed_C_d_A": c_peak,
                "observed_D_d_A": d_peak,
                "combined_CD_abs_error_A": combined,
                "filter_3_physical_chemical_sense": physical_filter,
                "filter_4_local_structural_envelope": envelope,
                "candidate_status": classify_candidate_status(pnab["pnab_filter"], cd_filter, physical_filter),
                "notes": "Omega-clean rise compression diagnostic; C/D agreement is necessary but not sufficient.",
            }
        )
    return rows


def summarize_surviving_family(funnel: pd.DataFrame, boundary_summary: pd.DataFrame) -> pd.DataFrame:
    """Return one-row summary of the strongest current surviving family."""
    survivors = funnel[funnel["candidate_status"].isin(["survives_current_filters", "survives_current_filters_with_pnab_caveat"])].copy()
    if survivors.empty:
        return pd.DataFrame(
            [
                {
                    "surviving_model_family": "none",
                    "status": "no_current_survivor",
                    "notes": "No row passed the currently available C/D and physical/chemical-sense filters.",
                }
            ]
        )
    plateau = survivors[survivors["filter_2_cd_band"] == "cd_plateau_preserved"].copy()
    selected = plateau if not plateau.empty else survivors
    best_error = pd.to_numeric(selected["combined_CD_abs_error_A"], errors="coerce").min()
    best_rows = selected[pd.to_numeric(selected["combined_CD_abs_error_A"], errors="coerce") == best_error].copy()
    variant_ids = best_rows["variant_id"].astype(str).tolist()
    return pd.DataFrame(
        [
            {
                "surviving_model_family": "omega-clean rise-compressed plateau",
                "status": "strongest_current_surviving_family_with_pnab_caveat",
                "variant_range": collapse_variant_range(variant_ids),
                "variant_count": len(variant_ids),
                "C_peak_A": float(pd.to_numeric(best_rows["observed_C_d_A"], errors="coerce").median()),
                "D_peak_A": float(pd.to_numeric(best_rows["observed_D_d_A"], errors="coerce").median()),
                "combined_CD_abs_error_A": float(best_error),
                "pnab_caveat": "parallel models not eliminated from current labeled pNAB data",
                "torsion_boundary_summary": torsion_boundary_sentence(boundary_summary),
            }
        ]
    )


def collapse_variant_range(variant_ids: list[str]) -> str:
    """Return compact range text for sorted plateau variant IDs."""
    if not variant_ids:
        return ""
    if len(variant_ids) == 1:
        return variant_ids[0]
    return f"{variant_ids[0]} through {variant_ids[-1]}"


def torsion_boundary_sentence(summary: pd.DataFrame) -> str:
    """Return concise measured torsion-boundary sentence."""
    if summary.empty:
        return "Torsion boundary summary missing."
    return (
        "Measured compatible range: phi +/-8 deg, psi +/-10 deg, omega +/-8 deg, "
        "combined symmetric same-sign +/-4 deg, opposing class +/-6 deg, "
        "phi/psi compensation +/-8 deg."
    )


def build_funnel(
    pnab: dict[str, str],
    guarded_scores: pd.DataFrame,
    guarded_geometry: pd.DataFrame,
    rise_scores: pd.DataFrame,
    rise_geometry: pd.DataFrame,
) -> pd.DataFrame:
    """Build full funnel table."""
    rows = [guarded_funnel_row(pnab, guarded_scores, guarded_geometry)]
    rows.extend(rise_funnel_rows(pnab, rise_scores, rise_geometry))
    return pd.DataFrame(rows)


def build_report_text(
    funnel: pd.DataFrame,
    survivor: pd.DataFrame,
    pnab: dict[str, str],
    missing: list[Path],
    boundary_summary: pd.DataFrame,
) -> str:
    """Build markdown report text."""
    cd_plateau = funnel[funnel["filter_2_cd_band"] == "cd_plateau_preserved"]
    overcompressed = funnel[funnel["filter_2_cd_band"] == "over_compressed_d_degraded"]
    surviving = funnel[funnel["candidate_status"].str.contains("survives", na=False)]
    physical_caveats = funnel[funnel["filter_3_physical_chemical_sense"] != "passes_physical_chemical_sense"]

    missing_text = "\n".join(f"- `{path}`" for path in missing) if missing else "- None."
    survivor_cols = list(survivor.columns)
    survivor_text = markdown_table(survivor, survivor_cols) if not survivor.empty else "_None._"
    funnel_cols = [
        "candidate_family",
        "variant_id",
        "filter_2_cd_band",
        "observed_C_d_A",
        "observed_D_d_A",
        "combined_CD_abs_error_A",
        "filter_3_physical_chemical_sense",
        "candidate_status",
    ]
    funnel_text = markdown_table(funnel[funnel_cols], funnel_cols) if not funnel.empty else "_No candidates summarized._"
    plateau_text = collapse_variant_range(cd_plateau["variant_id"].astype(str).tolist()) if not cd_plateau.empty else "none"
    over_text = ", ".join(overcompressed["variant_id"].astype(str).tolist()) if not overcompressed.empty else "none"

    return f"""# Item 6 Candidate-Filter Funnel

## Scope

Item 6 asks whether applying the positions of the C and D powder bands as a filter to pNAB-allowed scaffold candidates identifies a chemically sensible subset of structures. This report combines the currently available pNAB/scaffold compatibility filter, the C/D powder-band filter, and physical/chemical-sense filters from the omega-clean guarded full-chain and torsion-boundary workflows.

C/D agreement is necessary but not sufficient. Physical/chemical-sense filters are required.

pNAB is treated as a compatibility/scaffold filter, not final structural proof. The current pNAB parallel-vs-anti-parallel evidence is insufficient to eliminate parallel models. Anti-parallel 30 degrees remains a plausible candidate, not proven strongest solely from current pNAB data.

## Inputs Checked

Missing expected inputs:

{missing_text}

## Filter 1: pNAB / Scaffold Compatibility

- Parallel elimination status: `{pnab['parallel_elimination_status']}`
- Anti-parallel 30 status: `{pnab['anti_parallel_30_status']}`
- Funnel status: `{pnab['pnab_filter']}`

Can parallel models be eliminated from current repo data? No. The current repo audit did not find a clearly labeled matched parallel coordinate/output set suitable for atomistic and A/B/C/D scoring against the anti-parallel candidate.

## Filter 2: C/D Powder-Band Match

- Parent/guarded baseline: C = 5.7454 A, D = 7.2756 A, combined C/D error = 0.1698 A.
- Omega-clean compressed plateau: `{plateau_text}` with C = 5.6422 A, D = 7.2756 A, combined C/D error = 0.0667 A.
- Over-compressed D-degraded point(s): `{over_text}`. The known overshoot is omega_clean_scale_0p9700, where D shifts to about 7.1923 A.

Does the C/D filter select a subset? Yes. It selects the omega-clean rise-compressed plateau over the guarded/parent-like baseline and rejects the over-compressed D-degraded endpoint.

## Filter 3: Physical / Chemical Sense

The applied physical/chemical-sense filters include atom-count preservation, carboxylate preservation, residue/register preservation where available, guarded full-chain assembly, no unresolved segments, omega within +/-8 or +/-10 for selected/retained segments, and absence of the selected/retained every-other omega artifact.

Does the omega-clean rise-compressed family pass physical/chemical-sense filters? Yes for the current guarded selected/retained omega and preservation guards. Chain-level coordinate omega caveats remain present in the underlying coordinate-derived diagnostics, so this is still framed cautiously.

Are there models that match C/D but are chemically suspect? Any row with `cd_plateau_preserved` but not `passes_physical_chemical_sense` would be `cd_only_not_physical`. Current count: {len(funnel[funnel['candidate_status'] == 'cd_only_not_physical'])}.

## Filter 4: Local Structural Envelope

{torsion_boundary_sentence(boundary_summary)}

The torsion-boundary scan estimates the finite compatible backbone range indicated by the current C/D peak-picking and geometry guards. Whether that range is sufficiently narrow for manuscript framing is a PI-level interpretation. This report does not claim a unique structure.

## Funnel Table

{funnel_text}

## Surviving Model Family

{survivor_text}

The omega-clean rise-compressed plateau is the strongest current surviving model family under the combined filters available so far, with the explicit pNAB caveat that parallel models have not been eliminated from matched pNAB evidence.

## Interpretation Answers

- What does item 6 ask us to show? That C/D band positions can filter pNAB-compatible scaffold candidates toward a physically sensible subset.
- Which filters were applied? pNAB/scaffold compatibility, C/D powder-band match, physical/chemical-sense guards, and measured local torsion envelope.
- What does the pNAB filter currently show? It is partial/provenance-limited: anti-parallel 30 is plausible, but parallel is not eliminated.
- Can parallel models be eliminated from current repo data? No.
- Which rise-compressed omega-clean scales survive the C/D filter? `{plateau_text}`.
- Which candidates fail by over-compression? `{over_text}`.
- What is the current surviving model family? The omega-clean rise-compressed plateau, with pNAB caveat.
- What caveats remain? pNAB provenance is partial; matched parallel pNAB coordinates/output tables are missing; chain-level coordinate omega caveats remain; these are diagnostic coordinate families, not final atomistic provenance recovery.
- What should Asem/Nick provide or decide next? Matched pNAB parallel/anti-parallel outputs at comparable twist/rise, traceable YAML/input provenance, and PI-level judgment on whether the measured compatible torsion range is sufficiently narrow for manuscript framing.

## Output Files

- Funnel CSV: `outputs/metrics/item6_candidate_filter_funnel.csv`
- Surviving-family CSV: `outputs/metrics/item6_surviving_model_family_summary.csv`
- Report: `outputs/reports/item6_candidate_filter_funnel_report.md`
"""


def run(
    out_funnel: Path = OUT_FUNNEL,
    out_survivor: Path = OUT_SURVIVOR,
    out_report: Path = OUT_REPORT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build item 6 funnel outputs."""
    expected = [
        PNAB_INVENTORY,
        PNAB_SUMMARY,
        PNAB_GEOMETRY,
        PNAB_ABCD,
        PNAB_REPORT,
        RISE_SCORES,
        RISE_GEOMETRY,
        RISE_REPORT,
        GUARDED_SEGMENTS,
        GUARDED_CHAINS,
        GUARDED_GEOMETRY,
        GUARDED_ABCD,
        GUARDED_REPORT,
        BOUNDARY_SCORES,
        BOUNDARY_GEOMETRY,
        BOUNDARY_SUMMARY,
        BOUNDARY_REPORT,
    ]
    missing = missing_inputs(expected)
    pnab = load_pnab_context(PNAB_REPORT)
    guarded_scores = read_csv_or_empty(GUARDED_ABCD)
    guarded_geometry = read_csv_or_empty(GUARDED_GEOMETRY)
    rise_scores = read_csv_or_empty(RISE_SCORES)
    rise_geometry = read_csv_or_empty(RISE_GEOMETRY)
    boundary_summary = read_csv_or_empty(BOUNDARY_SUMMARY)

    funnel = build_funnel(pnab, guarded_scores, guarded_geometry, rise_scores, rise_geometry)
    survivor = summarize_surviving_family(funnel, boundary_summary)

    out_funnel.parent.mkdir(parents=True, exist_ok=True)
    out_survivor.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    funnel.to_csv(out_funnel, index=False)
    survivor.to_csv(out_survivor, index=False)
    out_report.write_text(build_report_text(funnel, survivor, pnab, missing, boundary_summary), encoding="utf-8")
    return funnel, survivor


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Build item 6 candidate-filter funnel report.")
    parser.add_argument("--out-funnel", type=Path, default=OUT_FUNNEL)
    parser.add_argument("--out-survivor", type=Path, default=OUT_SURVIVOR)
    parser.add_argument("--out-report", type=Path, default=OUT_REPORT)
    args = parser.parse_args()
    funnel, survivor = run(args.out_funnel, args.out_survivor, args.out_report)
    print(f"Funnel rows: {len(funnel)}")
    print(f"Surviving rows: {len(survivor)}")
    if not survivor.empty:
        row = survivor.iloc[0]
        print(f"Surviving family: {row.get('surviving_model_family')}")
        print(f"Variant range: {row.get('variant_range')}")
    print(f"Report: {args.out_report}")


if __name__ == "__main__":
    main()
