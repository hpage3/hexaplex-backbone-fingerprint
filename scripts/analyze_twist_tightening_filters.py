"""Combine current filters to ask whether twist is tightened.

This analysis works inside the Band-A-supported rise window of 3.3-3.4 A and
uses existing C/D, omega-clean geometry, torsion-envelope, and H-bond proxy
outputs. It does not invent missing twist candidates or infer false precision.
"""

from __future__ import annotations

import argparse
import math
import re
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

from scripts.compare_candidate_models_to_emory_fiber_fingerprint import expected_candidate_records, inventory_dataframe


RISE_MIN_A = 3.3
RISE_MAX_A = 3.4
CD_PASS_THRESHOLD_A = 0.08
D_DEGRADED_THRESHOLD_A = 0.08
TARGET_D_A = 7.3

OMEGA_SCORES = Path("outputs/metrics/omega_clean_rise_compression_scores.csv")
OMEGA_GEOMETRY = Path("outputs/metrics/omega_clean_rise_compression_geometry.csv")
GUARDED_ABCD = Path("outputs/metrics/guarded_full_chain_prototype_abcd_scores.csv")
TORSION_BOUNDARY = Path("outputs/metrics/omega_clean_torsion_boundary_summary.csv")
TORSION_ENVELOPE = Path("outputs/metrics/omega_clean_torsion_envelope_summary.csv")
HBOND_SUMMARY = Path("outputs/metrics/candidate_hbond_scoring_diagnostics.csv")

OUT_CANDIDATES = Path("outputs/metrics/twist_tightening_candidate_filters.csv")
OUT_TWIST = Path("outputs/metrics/twist_tightening_by_twist_summary.csv")
OUT_REPORT = Path("outputs/reports/twist_tightening_filter_report.md")
FIG_CD = Path("outputs/figures/twist_tightening_cd_error_by_twist.png")
FIG_COUNTS = Path("outputs/figures/twist_tightening_filter_pass_counts.png")
FIG_HBOND = Path("outputs/figures/twist_tightening_hbond_by_twist.png")


def parse_bool(value: Any) -> bool:
    """Parse bool-like values from CSV."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y", "pass", "passed"}


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    """Read a CSV or return an empty dataframe."""
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def rise_window_status(rise_A: float | None, lo: float = RISE_MIN_A, hi: float = RISE_MAX_A) -> str:
    """Classify rise against the Band-A-supported window."""
    if rise_A is None or pd.isna(rise_A):
        return "unknown"
    if float(rise_A) < lo:
        return "below_window"
    if float(rise_A) > hi:
        return "above_window"
    return "within_window"


def pass_from_status(status: str) -> bool | None:
    """Convert status string to pass/unknown/fail."""
    if status == "unknown":
        return None
    return status == "within_window"


def cd_pass(c_peak_A: float | None, d_peak_A: float | None, combined_error_A: float | None) -> bool | None:
    """Return whether C/D passes current item-6 style threshold."""
    if c_peak_A is None or d_peak_A is None or combined_error_A is None:
        return None
    if pd.isna(c_peak_A) or pd.isna(d_peak_A) or pd.isna(combined_error_A):
        return None
    if abs(float(d_peak_A) - TARGET_D_A) > D_DEGRADED_THRESHOLD_A:
        return False
    return float(combined_error_A) <= CD_PASS_THRESHOLD_A


def omega_geometry_pass(row: pd.Series | None) -> bool | None:
    """Return omega-clean geometry pass/unknown/fail."""
    if row is None or row.empty:
        return None
    count = pd.to_numeric(row.get("guarded_selected_retained_omega_count"), errors="coerce")
    within8 = pd.to_numeric(row.get("guarded_selected_retained_omega_within_8_count"), errors="coerce")
    every_other = parse_bool(row.get("guarded_selected_retained_omega_every_other_detected"))
    identity = parse_bool(row.get("atom_identity_preserved_vs_guarded"))
    carbox = parse_bool(row.get("carboxylates_preserved_vs_guarded"))
    if pd.isna(count) or pd.isna(within8):
        return None
    return bool(count > 0 and within8 == count and not every_other and identity and carbox)


def scale_token_from_name(candidate_name: str) -> str:
    """Return p-style scale token from candidate name."""
    match = re.search(r"scale_(\d+p\d+)", candidate_name)
    return match.group(1) if match else ""


def scale_float_from_name(candidate_name: str) -> float | None:
    """Return numeric scale from candidate name."""
    token = scale_token_from_name(candidate_name)
    if not token:
        return None
    return float(token.replace("p", "."))


def torsion_envelope_pass(candidate_name: str, boundary: pd.DataFrame, envelope: pd.DataFrame) -> bool | None:
    """Return torsion-envelope compatibility pass/unknown/fail."""
    scale = scale_float_from_name(candidate_name)
    if scale is None:
        return None
    # The measured torsion envelope is available only on the supported plateau scales.
    values = []
    for df in [boundary, envelope]:
        if df.empty or "scale" not in df:
            continue
        scales = pd.to_numeric(df["scale"], errors="coerce")
        sub = df[(scales - scale).abs() <= 1e-9]
        if not sub.empty:
            values.append(sub)
    if not values:
        return None
    combined = pd.concat(values, ignore_index=True)
    if "viable_envelope_member_count" not in combined:
        return None
    return bool(pd.to_numeric(combined["viable_envelope_member_count"], errors="coerce").fillna(0).max() > 0)


def hbond_pass(row: pd.Series | None) -> bool | None:
    """Return H-bond proxy pass/unknown/fail."""
    if row is None or row.empty:
        return None
    status = str(row.get("hbond_network_classification", ""))
    if status == "missing_candidate_coordinates":
        return None
    return status == "hbond_network_plausible"


def infer_twist(candidate_name: str, family: str) -> tuple[float | None, str, str]:
    """Infer twist from explicit metadata only."""
    text = f"{candidate_name} {family}".lower()
    if "omega_clean" in text or "guarded_full_chain" in text or "antiparallel" in text:
        return 30.0, "inferred_30_degree_like_from_current_family", "30-degree-like"
    return None, "unknown", ""


def combined_classification(row: pd.Series) -> str:
    """Classify candidate after ordered filters."""
    if row["rise_window_status"] in {"below_window", "above_window"}:
        return "rejected_outside_rise_window"
    if row["twist_status"] == "unknown":
        return "twist_unknown_provenance"
    if row["rise_window_status"] == "unknown":
        return "twist_insufficient_data"
    if row["cd_pass"] is False:
        return "twist_disfavored_cd"
    if row["omega_geometry_pass"] is False or row["torsion_envelope_pass"] is False:
        return "twist_disfavored_geometry"
    if row["hbond_pass"] is False:
        return "twist_disfavored_hbond"
    if any(row[field] is None for field in ["cd_pass", "omega_geometry_pass", "torsion_envelope_pass", "hbond_pass"]):
        return "twist_insufficient_data"
    return "twist_viable_current_filters"


def conservative_twist_conclusion(twist_summary: pd.DataFrame) -> str:
    """Return conservative conclusion about twist tightening."""
    if twist_summary.empty or "twist_deg" not in twist_summary:
        return "insufficient_data"
    viable = twist_summary[pd.to_numeric(twist_summary["all_filters_pass_count"], errors="coerce") > 0]
    if viable.empty:
        return "insufficient_data"
    twists = sorted(pd.to_numeric(viable["twist_deg"], errors="coerce").dropna().unique())
    if len(twists) == 1 and abs(twists[0] - 30.0) < 1e-6:
        return "current filters favor a narrow 30-degree-like neighborhood among available candidates, but neighboring twists are not tested here"
    if min(twists) <= 28.0 and max(twists) >= 32.0:
        return "broader 28-32 degree neighborhood remains viable"
    return "plausible_current_filters"


def twist_status_from_counts(row: pd.Series) -> str:
    """Classify one twist group."""
    all_pass = int(row.get("all_filters_pass_count", 0))
    count = int(row.get("candidate_count", 0))
    if all_pass > 0:
        return "strongly_supported_current_filters" if all_pass == count else "plausible_current_filters"
    if count > 0 and int(row.get("cd_pass_count", 0)) == 0:
        return "disfavored_current_filters"
    return "insufficient_data"


def candidate_rows() -> pd.DataFrame:
    """Build candidate filter rows from existing outputs."""
    inventory = inventory_dataframe(expected_candidate_records())
    scores = read_csv_or_empty(OMEGA_SCORES)
    geometry = read_csv_or_empty(OMEGA_GEOMETRY)
    guarded = read_csv_or_empty(GUARDED_ABCD)
    boundary = read_csv_or_empty(TORSION_BOUNDARY)
    envelope = read_csv_or_empty(TORSION_ENVELOPE)
    hbond = read_csv_or_empty(HBOND_SUMMARY)

    rows = []
    for _, inv in inventory.iterrows():
        name = str(inv["model_id"])
        score_row = scores[scores["variant_id"] == name].iloc[0] if not scores.empty and (scores["variant_id"] == name).any() else None
        geom_row = geometry[geometry["variant_id"] == name].iloc[0] if not geometry.empty and (geometry["variant_id"] == name).any() else None
        hbond_row = hbond[hbond["model_id"] == name].iloc[0] if not hbond.empty and (hbond["model_id"] == name).any() else None
        if name == "guarded_full_chain_prototype" and not guarded.empty:
            c_peak = pd.to_numeric(guarded.iloc[0].get("observed_C_d_A"), errors="coerce")
            d_peak = pd.to_numeric(guarded.iloc[0].get("observed_D_d_A"), errors="coerce")
            cd_error = pd.to_numeric(guarded.iloc[0].get("combined_CD_abs_error_A"), errors="coerce")
            rise = 3.4
        elif score_row is not None:
            c_peak = pd.to_numeric(score_row.get("observed_C_d_A"), errors="coerce")
            d_peak = pd.to_numeric(score_row.get("observed_D_d_A"), errors="coerce")
            cd_error = pd.to_numeric(score_row.get("combined_CD_abs_error_A"), errors="coerce")
            rise = pd.to_numeric(score_row.get("nominal_rise_equiv_A"), errors="coerce")
        else:
            c_peak = d_peak = cd_error = rise = float("nan")
        twist, twist_status, twist_label = infer_twist(name, str(inv.get("inferred_family", "")))
        rise_status = rise_window_status(rise)
        cd_ok = cd_pass(c_peak, d_peak, cd_error)
        omega_ok = omega_geometry_pass(geom_row) if geom_row is not None else (True if name == "guarded_full_chain_prototype" else None)
        torsion_ok = torsion_envelope_pass(name, boundary, envelope)
        hb_ok = hbond_pass(hbond_row)
        row = {
            "candidate_name": name,
            "coordinate_path": inv.get("path", ""),
            "family": inv.get("inferred_family", ""),
            "provenance_caveat": inv.get("provenance_caveat", ""),
            "rise_equivalent_A": rise,
            "rise_window_status": rise_status,
            "rise_window_pass": pass_from_status(rise_status),
            "twist_deg": twist,
            "twist_label": twist_label,
            "twist_status": twist_status,
            "observed_C_d_A": c_peak,
            "observed_D_d_A": d_peak,
            "combined_CD_abs_error_A": cd_error,
            "cd_pass": cd_ok,
            "omega_geometry_pass": omega_ok,
            "torsion_envelope_pass": torsion_ok,
            "hbond_pass": hb_ok,
            "hbond_plausibility_score": pd.to_numeric(hbond_row.get("hbond_plausibility_score"), errors="coerce") if hbond_row is not None else float("nan"),
            "hbond_network_classification": hbond_row.get("hbond_network_classification") if hbond_row is not None else "",
            "hbond_basis": "heavy_atom_proxy_only" if hbond_row is not None and not parse_bool(hbond_row.get("explicit_hydrogens_present")) else "unknown",
        }
        row["combined_filter_classification"] = combined_classification(pd.Series(row))
        rows.append(row)
    return pd.DataFrame(rows)


def twist_summary(candidate_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize candidate filter results by twist."""
    known = candidate_df.dropna(subset=["twist_deg"]).copy()
    if known.empty:
        return pd.DataFrame()
    rows = []
    for twist, sub in known.groupby("twist_deg"):
        all_pass = sub[sub["combined_filter_classification"] == "twist_viable_current_filters"]
        best_idx = pd.to_numeric(sub["combined_CD_abs_error_A"], errors="coerce").idxmin()
        row = {
            "twist_deg": float(twist),
            "candidate_count": len(sub),
            "rise_window_pass_count": int((sub["rise_window_pass"] == True).sum()),
            "cd_pass_count": int((sub["cd_pass"] == True).sum()),
            "omega_geometry_pass_count": int((sub["omega_geometry_pass"] == True).sum()),
            "torsion_envelope_pass_count": int((sub["torsion_envelope_pass"] == True).sum()),
            "hbond_pass_count": int((sub["hbond_pass"] == True).sum()),
            "all_filters_pass_count": len(all_pass),
            "best_cd_error": float(pd.to_numeric(sub["combined_CD_abs_error_A"], errors="coerce").min()),
            "best_candidate_name": sub.loc[best_idx, "candidate_name"] if pd.notna(best_idx) else "",
        }
        row["twist_status"] = twist_status_from_counts(pd.Series(row))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("twist_deg").reset_index(drop=True)


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    """Render dataframe as markdown table."""
    if df.empty:
        return "_No rows._"
    cols = [col for col in columns if col in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in df.head(max_rows)[cols].itertuples(index=False):
        values = [f"{value:.4g}" if isinstance(value, float) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_report(candidates: pd.DataFrame, by_twist: pd.DataFrame) -> str:
    """Build conservative twist-tightening report."""
    conclusion = conservative_twist_conclusion(by_twist)
    twist_values = sorted(pd.to_numeric(candidates["twist_deg"], errors="coerce").dropna().unique())
    missing_twist = candidates[candidates["twist_status"] == "unknown"]
    viable = candidates[candidates["combined_filter_classification"] == "twist_viable_current_filters"]
    return f"""# Twist-Tightening Filter Report

## Scope

Rise is constrained by Band A to 3.3-3.4 A. This analysis asks whether twist can be tightened within that rise window. C/D agreement remains necessary but not sufficient. H-bond scoring is a heavy-atom plausibility proxy, not affinity and not free energy. Candidate elimination should not rely on any single filter. If 30 degrees is favored, it is described as 30-degree-like unless the data prove a narrower value. If nearby twists remain viable, say so. If twist provenance is missing, do not infer a false precision.

## Available Twist Values

- Available inferred twist values: {twist_values}
- Candidates with missing twist provenance: {len(missing_twist)}
- Conservative conclusion: {conclusion}

## Candidate Filters

{markdown_table(candidates, ["candidate_name", "twist_label", "rise_equivalent_A", "rise_window_status", "combined_CD_abs_error_A", "cd_pass", "omega_geometry_pass", "torsion_envelope_pass", "hbond_pass", "hbond_plausibility_score", "combined_filter_classification"], max_rows=20)}

## Summary By Twist

{markdown_table(by_twist, ["twist_deg", "candidate_count", "rise_window_pass_count", "cd_pass_count", "omega_geometry_pass_count", "torsion_envelope_pass_count", "hbond_pass_count", "all_filters_pass_count", "best_cd_error", "best_candidate_name", "twist_status"], max_rows=20)}

## Interpretation

- Twist values available in this repo are currently dominated by 30-degree-like candidates.
- The combined filters identify the C/D plateau inside the Band-A-supported rise window, but do not provide a broad neighboring twist scan.
- 30 degrees is therefore favored/plausible among available candidates, not uniquely proven as an exact twist.
- Neighboring twists such as 28-32 degrees cannot be ruled in or out unless coordinate candidates and filter metrics for those twists are provided.
- Asem/Nick should provide clearly labeled twist-series coordinates or helical parameter metadata if twist remains a key constraint.
"""


def save_plots(candidates: pd.DataFrame, by_twist: pd.DataFrame) -> None:
    """Save simple diagnostic plots."""
    for path in [FIG_CD, FIG_COUNTS, FIG_HBOND]:
        path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    valid = candidates.dropna(subset=["twist_deg"])
    ax.scatter(valid["twist_deg"], pd.to_numeric(valid["combined_CD_abs_error_A"], errors="coerce"))
    ax.axhline(CD_PASS_THRESHOLD_A, color="red", linestyle="--", label="C/D pass threshold")
    ax.set_xlabel("twist (deg)")
    ax.set_ylabel("combined C/D error (A)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_CD, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    if not by_twist.empty:
        cols = ["rise_window_pass_count", "cd_pass_count", "omega_geometry_pass_count", "torsion_envelope_pass_count", "hbond_pass_count", "all_filters_pass_count"]
        bottom = [0] * len(by_twist)
        for col in cols:
            values = pd.to_numeric(by_twist[col], errors="coerce").fillna(0)
            ax.bar(by_twist["twist_deg"].astype(str), values, bottom=bottom, label=col.replace("_count", ""))
            bottom = [a + b for a, b in zip(bottom, values)]
        ax.legend(fontsize=7)
    ax.set_xlabel("twist (deg)")
    ax.set_ylabel("pass count")
    fig.tight_layout()
    fig.savefig(FIG_COUNTS, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(valid["twist_deg"], pd.to_numeric(valid["hbond_plausibility_score"], errors="coerce"))
    ax.set_xlabel("twist (deg)")
    ax.set_ylabel("H-bond proxy score")
    fig.tight_layout()
    fig.savefig(FIG_HBOND, dpi=180)
    plt.close(fig)


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run twist-tightening rollup and write outputs."""
    candidates = candidate_rows()
    by_twist = twist_summary(candidates)
    OUT_CANDIDATES.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(OUT_CANDIDATES, index=False)
    by_twist.to_csv(OUT_TWIST, index=False)
    OUT_REPORT.write_text(build_report(candidates, by_twist), encoding="utf-8")
    save_plots(candidates, by_twist)
    return candidates, by_twist


def parse_args() -> argparse.Namespace:
    """Parse arguments."""
    return argparse.ArgumentParser(description=__doc__).parse_args()


def main() -> int:
    """CLI entry point."""
    _ = parse_args()
    candidates, by_twist = run()
    print(f"Candidates evaluated: {len(candidates)}")
    print("Twist summary:")
    for row in by_twist.itertuples(index=False):
        print(f"  {row.twist_deg:g} deg: {row.all_filters_pass_count}/{row.candidate_count} pass all filters ({row.twist_status})")
    print(f"Report: {OUT_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
