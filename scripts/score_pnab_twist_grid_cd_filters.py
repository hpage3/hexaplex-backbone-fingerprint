"""Score and filter pNAB twist-grid candidates against C/D targets.

This analysis corrects the earlier twist-tightening scan by including the raw
pNAB twist-grid candidates in addition to the later powder-scored/parametric
and omega-clean publication-track families. Missing physical-filter data are
marked unknown rather than treated as failure.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_candidate_pdb_twist_range import normalize_path_text
from scripts.rollup_rich_coordinate_cd_diagnostics import score_pdb_profile
from scripts.score_constrained_phi_psi_candidates_cd import TARGET_C, TARGET_D, TOLERANCE, combined_abs_error
from scripts.score_radial_axial_refinement_variant_cd import markdown_table


INVENTORY_CSV = Path("outputs/metrics/candidate_pdb_twist_inventory.csv")
AUDIT_SCRIPT = Path("scripts/audit_candidate_pdb_twist_range.py")

SCORE_CSV = Path("outputs/metrics/pnab_twist_grid_cd_scored_candidates.csv")
BY_TWIST_CSV = Path("outputs/metrics/pnab_twist_grid_cd_by_twist_summary.csv")
FILTER_CSV = Path("outputs/metrics/pnab_twist_grid_filter_outcomes.csv")
REPORT_PATH = Path("outputs/reports/pnab_twist_grid_cd_filter_report.md")
FIG_CD = Path("outputs/figures/pnab_twist_grid_cd_error_by_twist.png")
FIG_D = Path("outputs/figures/pnab_twist_grid_D_by_twist.png")
FIG_FILTERS = Path("outputs/figures/pnab_twist_grid_filter_pass_counts.png")

INCLUDED_SETS = {"pnab_3p38_twist_grid", "powder_scored_candidate", "omega_clean_publication_track"}
RISE_MIN_A = 3.3
RISE_MAX_A = 3.4
CD_PASS_THRESHOLD_A = 0.08
BASELINE_CD_ERROR_A = 0.1698
REJECTED_D_A = 7.1923
D_STABLE_CENTER_A = 7.2756
D_GUARDRAIL_TOLERANCE_A = 0.08

PARAMETRIC_SCORE_FILES = [
    Path("outputs/parametric_six_strand_powder_scan/parametric_powder_scan_summary.csv"),
    Path("outputs/parametric_six_strand_powder_scan_refined/parametric_powder_scan_summary.csv"),
    Path("outputs/parametric_six_strand_powder_scan_zoffset/parametric_powder_scan_summary.csv"),
    Path("outputs/parametric_six_strand_powder_scan_alternating_zoffset/parametric_powder_scan_summary.csv"),
]

PUBLICATION_SCORE_FILES = [
    Path("outputs/metrics/omega_clean_rise_compression_scores.csv"),
    Path("outputs/metrics/parent_derived_rise_fine_scan_abcd_scores.csv"),
    Path("outputs/metrics/guarded_full_chain_prototype_abcd_scores.csv"),
    Path("outputs/metrics/item6_candidate_filter_funnel.csv"),
]


def parse_bool(value: object) -> bool:
    """Parse common CSV bool values."""
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y", "pass", "passed"}


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    """Read a CSV relative to repo root, or return empty."""
    full = ROOT / path
    if not full.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(full)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load_inventory(path: Path = INVENTORY_CSV) -> pd.DataFrame:
    """Load candidate twist inventory or fail with instructions."""
    full = ROOT / path
    if not full.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run .\\.venv\\Scripts\\python.exe {AUDIT_SCRIPT} before this scoring audit."
        )
    return pd.read_csv(full)


def select_candidate_sets(inventory: pd.DataFrame) -> pd.DataFrame:
    """Return candidate rows from the explicit target candidate sets."""
    required = {"candidate_id", "path", "candidate_set", "inferred_twist_deg", "inferred_rise_A"}
    missing = required - set(inventory.columns)
    if missing:
        raise ValueError(f"Inventory missing required columns: {sorted(missing)}")
    out = inventory[inventory["candidate_set"].isin(INCLUDED_SETS)].copy()
    out["final_twist_deg"] = pd.to_numeric(out["inferred_twist_deg"], errors="coerce")
    out["final_rise_A"] = pd.to_numeric(out["inferred_rise_A"], errors="coerce")
    out["pdb_path"] = out["path"].astype(str)
    out["coordinate_exists"] = out["pdb_path"].map(lambda value: (ROOT / value).exists())
    out["inference_source"] = out.get("twist_source", "").astype(str) + ";" + out.get("rise_source", "").astype(str)
    out["provenance_confidence"] = out["candidate_set"].map(
        {
            "pnab_3p38_twist_grid": "path_inferred_raw_pnab_grid",
            "powder_scored_candidate": "path_or_metric_inferred_parametric",
            "omega_clean_publication_track": "publication_track_reference",
        }
    )
    return out[
        out["coordinate_exists"]
        & out["final_twist_deg"].notna()
        & out["final_rise_A"].notna()
    ].reset_index(drop=True)


def first_present(row: pd.Series, columns: list[str]) -> Any:
    """Return first non-null value from candidate column names."""
    for column in columns:
        if column in row and not pd.isna(row[column]):
            return row[column]
    return None


def score_record_from_row(row: pd.Series, metric_file: Path) -> dict[str, object]:
    """Normalize one existing score row."""
    pdb_path = first_present(row, ["pdb_path", "coordinate_path", "input_pdb", "path"])
    candidate_name = first_present(row, ["model_label", "variant_id", "prototype_id", "candidate_name", "path"])
    c_peak = first_present(row, ["nearest_C_peak_d_A", "observed_C_d_A", "C_peak_A", "C_peak_d_A"])
    d_peak = first_present(row, ["nearest_D_peak_d_A", "observed_D_d_A", "D_peak_A", "D_peak_d_A"])
    c_error = first_present(row, ["nearest_C_error_A", "C_error_A"])
    d_error = first_present(row, ["nearest_D_error_A", "D_error_A"])
    if c_error is None and c_peak is not None:
        c_error = float(c_peak) - TARGET_C
    if d_error is None and d_peak is not None:
        d_error = float(d_peak) - TARGET_D
    cd_error = first_present(row, ["CD_combined_abs_error_A", "combined_CD_abs_error_A", "combined_abs_error_A"])
    if cd_error is None and c_error is not None and d_error is not None:
        cd_error = combined_abs_error(c_error, d_error)
    candidate_set = "omega_clean_publication_track" if "omega_clean" in normalize_path_text(candidate_name or metric_file) else "powder_scored_candidate"
    twist = first_present(row, ["twist_deg", "inferred_twist_deg"])
    if twist is None and candidate_set == "omega_clean_publication_track":
        twist = 30.0
    rise = first_present(row, ["rise_A", "inferred_rise_A", "nominal_rise_equiv_A"])
    return {
        "metric_file": str(metric_file),
        "candidate_name": str(candidate_name or ""),
        "pdb_path": str(pdb_path or ""),
        "candidate_set": candidate_set,
        "final_twist_deg": twist,
        "final_rise_A": rise,
        "path_key": normalize_path_text(pdb_path or ""),
        "name_key": normalize_path_text(candidate_name or ""),
        "observed_C_d_A": c_peak,
        "observed_D_d_A": d_peak,
        "C_error_A": c_error,
        "D_error_A": d_error,
        "combined_C_D_error": cd_error,
        "C_score": first_present(row, ["nearest_C_intensity", "C_score", "C_peak_intensity_or_score"]),
        "D_score": first_present(row, ["nearest_D_intensity", "D_score", "D_peak_intensity_or_score"]),
    }


def load_existing_scores() -> pd.DataFrame:
    """Load all known existing C/D score tables into one normalized dataframe."""
    records: list[dict[str, object]] = []
    for metric_file in PARAMETRIC_SCORE_FILES + PUBLICATION_SCORE_FILES:
        df = read_csv_or_empty(metric_file)
        if df.empty:
            continue
        for _, row in df.iterrows():
            rec = score_record_from_row(row, metric_file)
            if rec["observed_C_d_A"] is not None and rec["observed_D_d_A"] is not None:
                records.append(rec)
    return pd.DataFrame(records)


def reference_rows_from_existing_scores(existing_scores: pd.DataFrame) -> pd.DataFrame:
    """Return omega-clean publication-track reference rows from recovered metrics."""
    if existing_scores.empty or "candidate_set" not in existing_scores:
        return pd.DataFrame()
    refs = existing_scores[existing_scores["candidate_set"] == "omega_clean_publication_track"].copy()
    rows: list[dict[str, object]] = []
    for _, row in refs.iterrows():
        rows.append(
            {
                "candidate_name": row.get("candidate_name", ""),
                "pdb_path": row.get("pdb_path", ""),
                "candidate_set": "omega_clean_publication_track",
                "final_twist_deg": pd.to_numeric(row.get("final_twist_deg", 30.0), errors="coerce"),
                "final_rise_A": pd.to_numeric(row.get("final_rise_A", math.nan), errors="coerce"),
                "inference_source": "existing_publication_metric",
                "provenance_confidence": "publication_track_reference",
                "target_C_d_A": TARGET_C,
                "target_D_d_A": TARGET_D,
                "observed_C_d_A": row.get("observed_C_d_A", math.nan),
                "observed_D_d_A": row.get("observed_D_d_A", math.nan),
                "C_error_A": row.get("C_error_A", math.nan),
                "D_error_A": row.get("D_error_A", math.nan),
                "combined_C_D_error": row.get("combined_C_D_error", math.nan),
                "C_score": row.get("C_score", math.nan),
                "D_score": row.get("D_score", math.nan),
                "score_source": "recovered_existing_metric",
                "score_status": "scored",
                "failure_reason": "",
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["rise_window_status"] = out["final_rise_A"].map(rise_window_status)
    out["cd_agreement_status"] = out["combined_C_D_error"].map(cd_agreement_status)
    out["D_guardrail_status"] = out["observed_D_d_A"].map(d_guardrail_status)
    out["physical_filter_status"] = "pass_available_publication_track"
    out["hbond_filter_status"] = "pass_hbond_proxy_reference"
    out["all_available_filters_status"] = out.apply(all_available_filters_status, axis=1)
    return out


def recover_score(candidate: pd.Series, existing_scores: pd.DataFrame) -> pd.Series | None:
    """Recover an existing score by exact path, filename/stem, or candidate name."""
    if existing_scores.empty:
        return None
    path_key = normalize_path_text(candidate["pdb_path"])
    name_key = normalize_path_text(candidate.get("candidate_id", ""))
    stem_key = normalize_path_text(Path(str(candidate["pdb_path"])).stem)
    parent_key = normalize_path_text(Path(str(candidate["pdb_path"])).parent.name)

    for column, key in [("path_key", path_key), ("name_key", name_key), ("name_key", stem_key), ("name_key", parent_key)]:
        matches = existing_scores[existing_scores[column] == key]
        if not matches.empty:
            return matches.iloc[0]
    filename = Path(str(candidate["pdb_path"])).name.lower()
    if filename != "initial.pdb":
        matches = existing_scores[existing_scores["pdb_path"].astype(str).str.lower().str.endswith(filename)]
        if not matches.empty:
            return matches.iloc[0]
    return None


def score_missing_candidate(
    candidate: pd.Series,
    target_c: float,
    target_d: float,
    tolerance: float,
    q_step: float,
    d_min: float,
    d_max: float,
) -> dict[str, object]:
    """Compute C/D score using the existing Debye point-scatterer utility."""
    score = score_pdb_profile(ROOT / str(candidate["pdb_path"]), target_c, target_d, tolerance, q_step, d_min, d_max)
    return {
        "observed_C_d_A": score["C_peak_d_A"],
        "observed_D_d_A": score["D_peak_d_A"],
        "C_error_A": score["C_error_A"],
        "D_error_A": score["D_error_A"],
        "combined_C_D_error": combined_abs_error(score["C_error_A"], score["D_error_A"]),
        "C_score": score["C_peak_intensity"],
        "D_score": score["D_peak_intensity"],
        "score_source": "computed_existing_debye_profile",
        "score_status": "scored",
        "failure_reason": "",
    }


def rise_window_status(rise_A: object) -> str:
    """Classify rise against Band-A-supported 3.3-3.4 A window."""
    value = pd.to_numeric(rise_A, errors="coerce")
    if pd.isna(value):
        return "unknown"
    if float(value) < RISE_MIN_A:
        return "fail_below_3p3"
    if float(value) > RISE_MAX_A:
        return "fail_above_3p4"
    return "pass"


def cd_agreement_status(error_A: object, threshold: float = CD_PASS_THRESHOLD_A) -> str:
    """Classify C/D agreement."""
    value = pd.to_numeric(error_A, errors="coerce")
    if pd.isna(value):
        return "unknown"
    if float(value) <= threshold:
        return "pass"
    if float(value) <= BASELINE_CD_ERROR_A:
        return "borderline_reference_like"
    return "fail"


def d_guardrail_status(d_peak_A: object) -> str:
    """Classify D-band stability against rejected over-compressed behavior."""
    value = pd.to_numeric(d_peak_A, errors="coerce")
    if pd.isna(value):
        return "unknown"
    if abs(float(value) - REJECTED_D_A) <= 0.025:
        return "fail_rejected_0p9700_like"
    if abs(float(value) - D_STABLE_CENTER_A) <= D_GUARDRAIL_TOLERANCE_A:
        return "pass"
    if abs(float(value) - TARGET_D) <= TOLERANCE:
        return "borderline_target_window"
    return "fail"


def physical_filter_status(candidate_set: str, candidate_name: str) -> str:
    """Return physical-filter status where available."""
    text = f"{candidate_set} {candidate_name}".lower()
    if "omega_clean" in text:
        return "pass_available_publication_track"
    if "pnab_3p38_twist_grid" in text:
        return "unknown_raw_pnab_not_available"
    if "powder_scored_candidate" in text:
        return "unknown_parametric_not_available"
    return "unknown"


def all_available_filters_status(row: pd.Series) -> str:
    """Classify all available filters without penalizing unknown physical filters."""
    required = [row["rise_window_status"], row["cd_agreement_status"], row["D_guardrail_status"]]
    if any(status.startswith("fail") for status in required):
        return "fail_available_filters"
    if row["cd_agreement_status"] == "borderline_reference_like" or row["D_guardrail_status"].startswith("borderline"):
        return "borderline_available_filters"
    if row["physical_filter_status"].startswith("pass") or row["physical_filter_status"].startswith("unknown"):
        if all(status == "pass" for status in required):
            return "pass_available_filters"
    return "unknown"


def build_scored_candidates(
    candidates: pd.DataFrame,
    existing_scores: pd.DataFrame,
    target_c: float = TARGET_C,
    target_d: float = TARGET_D,
    tolerance: float = TOLERANCE,
    q_step: float = 0.01,
    d_min: float = 2.5,
    d_max: float = 12.0,
    score_missing: bool = True,
    max_new_scores_per_twist: int = 0,
    max_atoms_for_new_score: int = 4500,
) -> pd.DataFrame:
    """Recover or compute C/D scores for candidates."""
    rows: list[dict[str, object]] = []
    new_scores_by_twist: dict[float, int] = {}
    for _, candidate in candidates.iterrows():
        base = {
            "candidate_name": candidate.get("candidate_id", ""),
            "pdb_path": candidate.get("pdb_path", ""),
            "candidate_set": candidate.get("candidate_set", ""),
            "final_twist_deg": candidate.get("final_twist_deg", math.nan),
            "final_rise_A": candidate.get("final_rise_A", math.nan),
            "inference_source": candidate.get("inference_source", ""),
            "provenance_confidence": candidate.get("provenance_confidence", ""),
            "target_C_d_A": target_c,
            "target_D_d_A": target_d,
        }
        recovered = recover_score(candidate, existing_scores)
        if recovered is not None:
            base.update(
                {
                    "observed_C_d_A": recovered["observed_C_d_A"],
                    "observed_D_d_A": recovered["observed_D_d_A"],
                    "C_error_A": recovered["C_error_A"],
                    "D_error_A": recovered["D_error_A"],
                    "combined_C_D_error": recovered["combined_C_D_error"],
                    "C_score": recovered["C_score"],
                    "D_score": recovered["D_score"],
                    "score_source": "recovered_existing_metric",
                    "score_status": "scored",
                    "failure_reason": "",
                }
            )
        elif score_missing:
            twist_key = float(candidate.get("final_twist_deg", math.nan))
            already_scored = new_scores_by_twist.get(twist_key, 0)
            atom_count = pd.to_numeric(candidate.get("atom_count", math.nan), errors="coerce")
            if already_scored >= max_new_scores_per_twist:
                base.update(
                    {
                        "observed_C_d_A": math.nan,
                        "observed_D_d_A": math.nan,
                        "C_error_A": math.nan,
                        "D_error_A": math.nan,
                        "combined_C_D_error": math.nan,
                        "C_score": math.nan,
                        "D_score": math.nan,
                        "score_source": "not_scored_runtime_guard",
                        "score_status": "missing_score",
                        "failure_reason": f"runtime guard: already computed {max_new_scores_per_twist} missing score(s) for twist {twist_key:g}",
                    }
                )
                rows.append(base)
                continue
            if pd.notna(atom_count) and float(atom_count) > max_atoms_for_new_score:
                base.update(
                    {
                        "observed_C_d_A": math.nan,
                        "observed_D_d_A": math.nan,
                        "C_error_A": math.nan,
                        "D_error_A": math.nan,
                        "combined_C_D_error": math.nan,
                        "C_score": math.nan,
                        "D_score": math.nan,
                        "score_source": "not_scored_runtime_guard",
                        "score_status": "missing_score",
                        "failure_reason": f"runtime guard: atom_count {float(atom_count):.0f} exceeds max_atoms_for_new_score {max_atoms_for_new_score}",
                    }
                )
                rows.append(base)
                continue
            try:
                base.update(score_missing_candidate(candidate, target_c, target_d, tolerance, q_step, d_min, d_max))
                new_scores_by_twist[twist_key] = already_scored + 1
            except Exception as exc:  # noqa: BLE001 - row-level scoring should continue.
                base.update(
                    {
                        "observed_C_d_A": math.nan,
                        "observed_D_d_A": math.nan,
                        "C_error_A": math.nan,
                        "D_error_A": math.nan,
                        "combined_C_D_error": math.nan,
                        "C_score": math.nan,
                        "D_score": math.nan,
                        "score_source": "computed_existing_debye_profile",
                        "score_status": "scoring_failed",
                        "failure_reason": str(exc),
                    }
                )
        else:
            base.update(
                {
                    "observed_C_d_A": math.nan,
                    "observed_D_d_A": math.nan,
                    "C_error_A": math.nan,
                    "D_error_A": math.nan,
                    "combined_C_D_error": math.nan,
                    "C_score": math.nan,
                    "D_score": math.nan,
                    "score_source": "not_scored",
                    "score_status": "missing_score",
                    "failure_reason": "score_missing disabled",
                }
            )
        rows.append(base)

    scored = pd.DataFrame(rows)
    scored["rise_window_status"] = scored["final_rise_A"].map(rise_window_status)
    scored["cd_agreement_status"] = scored["combined_C_D_error"].map(cd_agreement_status)
    scored["D_guardrail_status"] = scored["observed_D_d_A"].map(d_guardrail_status)
    scored["physical_filter_status"] = [
        physical_filter_status(row.candidate_set, row.candidate_name) for row in scored.itertuples()
    ]
    scored["hbond_filter_status"] = scored["candidate_set"].map(
        lambda value: "unknown_not_available" if value != "omega_clean_publication_track" else "pass_hbond_proxy_reference"
    )
    scored["all_available_filters_status"] = scored.apply(all_available_filters_status, axis=1)
    return scored


def summarize_by_twist(scored: pd.DataFrame) -> pd.DataFrame:
    """Group scored candidates by twist degree."""
    rows: list[dict[str, object]] = []
    for twist, sub in scored.groupby("final_twist_deg", dropna=False):
        scored_sub = sub[sub["score_status"] == "scored"].copy()
        errors = pd.to_numeric(scored_sub["combined_C_D_error"], errors="coerce")
        best = scored_sub.loc[errors.idxmin()] if not scored_sub.empty and errors.notna().any() else pd.Series(dtype=object)
        pass_count = int((sub["all_available_filters_status"] == "pass_available_filters").sum())
        cd_pass_count = int((sub["cd_agreement_status"] == "pass").sum())
        candidate_count = len(sub)
        scored_count = int((sub["score_status"] == "scored").sum())
        if scored_count == 0:
            status = "no_scored_candidates"
        elif pass_count > 0 and cd_pass_count == scored_count and scored_count == candidate_count:
            status = "strongly_supported_current_filters"
        elif pass_count > 0 or cd_pass_count > 0:
            status = "plausible_current_filters"
        elif scored_count > 0:
            status = "disfavored_current_filters"
        else:
            status = "insufficient_data"
        rows.append(
            {
                "twist_deg": twist,
                "candidate_count": candidate_count,
                "scored_candidate_count": scored_count,
                "scoring_failed_count": int((sub["score_status"] == "scoring_failed").sum()),
                "rise_pass_count": int((sub["rise_window_status"] == "pass").sum()),
                "cd_pass_count": cd_pass_count,
                "D_guardrail_pass_count": int((sub["D_guardrail_status"] == "pass").sum()),
                "physical_filter_pass_count": int(sub["physical_filter_status"].astype(str).str.startswith("pass").sum()),
                "all_available_filters_pass_count": pass_count,
                "best_C_D_error": float(errors.min()) if errors.notna().any() else math.nan,
                "best_C": best.get("observed_C_d_A", math.nan),
                "best_D": best.get("observed_D_d_A", math.nan),
                "best_candidate_name": best.get("candidate_name", ""),
                "median_C_D_error": float(errors.median()) if errors.notna().any() else math.nan,
                "min_C_D_error": float(errors.min()) if errors.notna().any() else math.nan,
                "max_C_D_error": float(errors.max()) if errors.notna().any() else math.nan,
                "unknown_physical_filter_count": int(sub["physical_filter_status"].astype(str).str.startswith("unknown").sum()),
                "twist_status": status,
            }
        )
    return pd.DataFrame(rows).sort_values("twist_deg").reset_index(drop=True)


def conclusion_logic(by_twist: pd.DataFrame, scored: pd.DataFrame) -> str:
    """Return conservative twist narrowing conclusion."""
    if scored.empty or int((scored["score_status"] == "scored").sum()) == 0:
        return "insufficient_scoring_data"
    publication = scored[scored["candidate_set"] == "omega_clean_publication_track"]
    if not publication.empty:
        best_pub = pd.to_numeric(publication["combined_C_D_error"], errors="coerce").min()
        if pd.isna(best_pub) or abs(float(best_pub) - 0.06665) > 0.03:
            return "scoring_method_needs_validation"

    scored_twists = set(pd.to_numeric(by_twist[by_twist["scored_candidate_count"] > 0]["twist_deg"], errors="coerce").dropna())
    passing_twists = set(
        pd.to_numeric(by_twist[by_twist["all_available_filters_pass_count"] > 0]["twist_deg"], errors="coerce").dropna()
    )
    if not set(range(18, 33)).intersection(scored_twists):
        return "insufficient_scoring_data"
    if not set(range(18, 33)).issubset(scored_twists):
        return "insufficient_scoring_data"
    if passing_twists and passing_twists == {30.0} and {28.0, 29.0, 31.0, 32.0}.issubset(scored_twists):
        return "narrowed_to_30_degree_like"
    if passing_twists and min(passing_twists) >= 28.0 and max(passing_twists) <= 32.0:
        return "narrowed_to_28_32_degree_family"
    if passing_twists and min(passing_twists) <= 18.0 and max(passing_twists) >= 32.0:
        return "broad_18_32_remains_plausible"
    return "insufficient_scoring_data"


def filter_outcomes(scored: pd.DataFrame, conclusion: str) -> pd.DataFrame:
    """Return compact per-candidate filter outcome table."""
    columns = [
        "candidate_name",
        "candidate_set",
        "final_twist_deg",
        "final_rise_A",
        "score_status",
        "score_source",
        "combined_C_D_error",
        "rise_window_status",
        "cd_agreement_status",
        "D_guardrail_status",
        "physical_filter_status",
        "hbond_filter_status",
        "all_available_filters_status",
        "failure_reason",
    ]
    out = scored[columns].copy()
    out["overall_conclusion"] = conclusion
    return out


def save_plots(scored: pd.DataFrame, by_twist: pd.DataFrame) -> None:
    """Save simple diagnostic plots."""
    for path in [FIG_CD, FIG_D, FIG_FILTERS]:
        path.parent.mkdir(parents=True, exist_ok=True)

    valid = scored[scored["score_status"] == "scored"].copy()
    valid["combined_C_D_error"] = pd.to_numeric(valid["combined_C_D_error"], errors="coerce")
    valid["observed_D_d_A"] = pd.to_numeric(valid["observed_D_d_A"], errors="coerce")
    valid["final_twist_deg"] = pd.to_numeric(valid["final_twist_deg"], errors="coerce")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for candidate_set, sub in valid.groupby("candidate_set"):
        ax.scatter(sub["final_twist_deg"], sub["combined_C_D_error"], s=14, label=candidate_set, alpha=0.65)
    ax.axhline(CD_PASS_THRESHOLD_A, color="0.3", ls="--", lw=1, label="C/D pass threshold")
    ax.set_xlabel("twist (deg)")
    ax.set_ylabel("combined |C|+|D| error (A)")
    ax.set_title("pNAB twist-grid C/D error by twist")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ROOT / FIG_CD, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for candidate_set, sub in valid.groupby("candidate_set"):
        ax.scatter(sub["final_twist_deg"], sub["observed_D_d_A"], s=14, label=candidate_set, alpha=0.65)
    ax.axhline(D_STABLE_CENTER_A, color="0.3", ls="--", lw=1, label="D stable reference")
    ax.axhline(REJECTED_D_A, color="crimson", ls=":", lw=1, label="rejected 0p9700-like D")
    ax.set_xlabel("twist (deg)")
    ax.set_ylabel("D peak (A)")
    ax.set_title("D-band guardrail by twist")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ROOT / FIG_D, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    plot = by_twist.copy()
    x = pd.to_numeric(plot["twist_deg"], errors="coerce")
    ax.bar(x - 0.18, plot["scored_candidate_count"], width=0.18, label="scored")
    ax.bar(x, plot["cd_pass_count"], width=0.18, label="C/D pass")
    ax.bar(x + 0.18, plot["all_available_filters_pass_count"], width=0.18, label="all available pass")
    ax.set_xlabel("twist (deg)")
    ax.set_ylabel("candidate count")
    ax.set_title("Filter pass counts by twist")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / FIG_FILTERS, dpi=180)
    plt.close(fig)


def build_report(scored: pd.DataFrame, by_twist: pd.DataFrame, conclusion: str) -> str:
    """Build markdown report."""
    set_counts = scored.groupby("candidate_set").agg(
        candidates=("candidate_name", "count"),
        scored=("score_status", lambda s: int((s == "scored").sum())),
        failed=("score_status", lambda s: int((s == "scoring_failed").sum())),
    ).reset_index()
    scored_sets = markdown_table(set_counts, ["candidate_set", "candidates", "scored", "failed"])
    by_twist_table = markdown_table(
        by_twist,
        [
            "twist_deg",
            "candidate_count",
            "scored_candidate_count",
            "cd_pass_count",
            "all_available_filters_pass_count",
            "best_C_D_error",
            "best_C",
            "best_D",
            "twist_status",
        ],
    )
    physical_unknown = int(scored["physical_filter_status"].astype(str).str.startswith("unknown").sum())
    recovered = int((scored["score_source"] == "recovered_existing_metric").sum())
    computed = int((scored["score_source"] == "computed_existing_debye_profile").sum())

    return f"""# pNAB Twist-Grid C/D Filter Report

This analysis corrects the earlier twist-tightening scan by including the raw pNAB twist-grid candidates. It remains a conservative diagnostic filter audit, not a unique-structure claim.

## Required Cautions

- Rise is constrained by Band A to 3.3-3.4 A.
- C/D agreement is necessary but not sufficient.
- H-bond scoring is a heavy-atom plausibility proxy, not affinity or free energy.
- Missing physical-filter data are marked unknown, not treated as failure.
- Candidate elimination should not rely on any single filter.
- If nearby twists remain viable, say so.
- If scoring could not be recovered or run for enough candidates, say insufficient_scoring_data.
- The later selected-from-345 subset was not clearly identifiable in the prior audit unless this task finds it.

## Candidate Sets Included

{scored_sets}

## Scoring Sources

- Existing C/D scores recovered: {recovered}
- Missing candidates scored with the existing Debye point-scatterer profile utility: {computed}
- Physical filters unavailable/unknown: {physical_unknown}

## Twist Summary

{by_twist_table}

## Conservative Conclusion

`{conclusion}`

## Interpretation Questions

- **Do C/D scores narrow 18-32 to 28-32?** The answer depends on whether raw pNAB twist-grid rows were successfully scored. If lower twists are unscored or physical filters are unavailable, the conclusion remains insufficient_scoring_data rather than a structural exclusion.
- **Do C/D scores further narrow 28-32 to ~30?** Only if 28, 29, 31, and 32 are scored and fail while 30 passes. Nearby twists that pass available filters remain plausible_current_filters.
- **Are physical filters available for raw pNAB candidates?** Mostly no; raw pNAB physical-filter data are marked unknown rather than failure.
- **Does H-bond plausibility add twist discrimination?** Not here unless a comparable H-bond proxy exists for the same candidate coordinates. Current H-bond scoring is a supporting heavy-atom plausibility proxy.
- **What data are still needed from Asem/Nick?** The selected-from-345 manifest, original pNAB/YAML metadata, trace labels, and comparable geometry/physical-filter outputs for raw pNAB twist-grid candidates.
"""


def run(
    inventory_csv: Path = INVENTORY_CSV,
    score_missing: bool = True,
    q_step: float = 0.01,
    d_min: float = 2.5,
    d_max: float = 12.0,
    max_new_scores_per_twist: int = 0,
    max_atoms_for_new_score: int = 4500,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Run scoring/filter audit and write outputs."""
    inventory = load_inventory(inventory_csv)
    candidates = select_candidate_sets(inventory)
    existing = load_existing_scores()
    scored = build_scored_candidates(
        candidates,
        existing,
        q_step=q_step,
        d_min=d_min,
        d_max=d_max,
        score_missing=score_missing,
        max_new_scores_per_twist=max_new_scores_per_twist,
        max_atoms_for_new_score=max_atoms_for_new_score,
    )
    references = reference_rows_from_existing_scores(existing)
    if not references.empty:
        scored = pd.concat([scored, references], ignore_index=True)
    by_twist = summarize_by_twist(scored)
    conclusion = conclusion_logic(by_twist, scored)
    filters = filter_outcomes(scored, conclusion)
    report = build_report(scored, by_twist, conclusion)

    for path in [SCORE_CSV, BY_TWIST_CSV, FILTER_CSV, REPORT_PATH, FIG_CD, FIG_D, FIG_FILTERS]:
        (ROOT / path).parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(ROOT / SCORE_CSV, index=False)
    by_twist.to_csv(ROOT / BY_TWIST_CSV, index=False)
    filters.to_csv(ROOT / FILTER_CSV, index=False)
    (ROOT / REPORT_PATH).write_text(report, encoding="utf-8")
    save_plots(scored, by_twist)
    return scored, by_twist, filters, conclusion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-csv", type=Path, default=INVENTORY_CSV)
    parser.add_argument("--no-score-missing", action="store_true")
    parser.add_argument("--q-step", type=float, default=0.01)
    parser.add_argument("--d-min", type=float, default=2.5)
    parser.add_argument("--d-max", type=float, default=12.0)
    parser.add_argument("--max-new-scores-per-twist", type=int, default=0)
    parser.add_argument("--max-atoms-for-new-score", type=int, default=4500)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scored, by_twist, _filters, conclusion = run(
        args.inventory_csv,
        score_missing=not args.no_score_missing,
        q_step=args.q_step,
        d_min=args.d_min,
        d_max=args.d_max,
        max_new_scores_per_twist=args.max_new_scores_per_twist,
        max_atoms_for_new_score=args.max_atoms_for_new_score,
    )
    print(f"candidates={len(scored)}")
    print(f"scored={(scored['score_status'] == 'scored').sum()}")
    print(f"failed={(scored['score_status'] == 'scoring_failed').sum()}")
    print(f"twists={','.join(str(v) for v in by_twist['twist_deg'].tolist())}")
    print(f"conclusion={conclusion}")
    print(f"report={REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
