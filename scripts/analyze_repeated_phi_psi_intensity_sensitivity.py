"""Analyze C/D intensity sensitivity for safe repeated phi/psi variants."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_SCORES_CSV = Path("outputs/metrics/repeated_constrained_phi_psi_variant_cd_scores.csv")
DEFAULT_OUT_CSV = Path("outputs/metrics/repeated_phi_psi_intensity_sensitivity.csv")
DEFAULT_REPORT = Path("outputs/reports/repeated_phi_psi_intensity_sensitivity.md")
DEFAULT_FIGURE_BASE = Path("outputs/figures/repeated_phi_psi_intensity_sensitivity")
FLAT_TOLERANCE_FRACTION = 0.005


def require_intensity_columns(df: pd.DataFrame) -> None:
    """Raise a clear error if peak intensity columns are unavailable."""
    required = {"C_peak_intensity_or_score", "D_peak_intensity_or_score"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required intensity columns: {', '.join(missing)}")


def sort_by_delta(df: pd.DataFrame) -> pd.DataFrame:
    """Sort repeated variant rows by fixed torsion delta."""
    out = df.copy()
    out["fixed_torsion_delta_deg"] = pd.to_numeric(out["fixed_torsion_delta_deg"], errors="coerce")
    return out.sort_values(["fixed_torsion_delta_deg", "variant_id"]).reset_index(drop=True)


def baseline_value(df: pd.DataFrame, column: str) -> float:
    """Return the delta-zero baseline value for a column."""
    delta = pd.to_numeric(df["fixed_torsion_delta_deg"], errors="coerce")
    baseline = df[delta == 0]
    if baseline.empty:
        raise ValueError("No delta 0 baseline row found for intensity normalization.")
    return float(baseline.iloc[0][column])


def normalize_to_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Add relative and percent-change intensity columns."""
    require_intensity_columns(df)
    out = sort_by_delta(df)
    c_base = baseline_value(out, "C_peak_intensity_or_score")
    d_base = baseline_value(out, "D_peak_intensity_or_score")
    out["C_relative_to_baseline"] = pd.to_numeric(out["C_peak_intensity_or_score"], errors="coerce") / c_base
    out["D_relative_to_baseline"] = pd.to_numeric(out["D_peak_intensity_or_score"], errors="coerce") / d_base
    out["C_percent_change"] = (out["C_relative_to_baseline"] - 1.0) * 100.0
    out["D_percent_change"] = (out["D_relative_to_baseline"] - 1.0) * 100.0
    return out


def classify_trend(values: pd.Series, flat_tolerance_fraction: float = FLAT_TOLERANCE_FRACTION) -> str:
    """Classify relative intensity trend with a flatness tolerance."""
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_list()
    if len(numeric) < 3:
        return "insufficient"
    if max(abs(value - 1.0) for value in numeric) <= flat_tolerance_fraction:
        return "flat"
    diffs = [b - a for a, b in zip(numeric, numeric[1:])]
    if all(delta >= -flat_tolerance_fraction for delta in diffs):
        return "monotonic increasing"
    if all(delta <= flat_tolerance_fraction for delta in diffs):
        return "monotonic decreasing"
    return "asymmetric/non-monotonic"


def peak_positions_flat(df: pd.DataFrame) -> bool:
    """Return whether C and D peak positions are unchanged across rows."""
    c_unique = pd.to_numeric(df["C_peak_A"], errors="coerce").dropna().round(6).nunique()
    d_unique = pd.to_numeric(df["D_peak_A"], errors="coerce").dropna().round(6).nunique()
    return c_unique <= 1 and d_unique <= 1


def build_report_text(result: pd.DataFrame, flat_tolerance_fraction: float = FLAT_TOLERANCE_FRACTION) -> str:
    """Build markdown report text."""
    if result.empty:
        return "# Repeated Phi/Psi Intensity Sensitivity\n\nNo repeated variant scores were available.\n"
    c_trend = classify_trend(result["C_relative_to_baseline"], flat_tolerance_fraction)
    d_trend = classify_trend(result["D_relative_to_baseline"], flat_tolerance_fraction)
    c_span = float(pd.to_numeric(result["C_percent_change"], errors="coerce").max() - pd.to_numeric(result["C_percent_change"], errors="coerce").min())
    d_span = float(pd.to_numeric(result["D_percent_change"], errors="coerce").max() - pd.to_numeric(result["D_percent_change"], errors="coerce").min())
    more_sensitive = "C" if c_span > d_span else "D" if d_span > c_span else "neither"
    meaningful = "likely tiny/numerical at this pilot scale" if max(c_span, d_span) < 1.0 else "potentially meaningful enough to inspect"
    columns = [
        "variant_id",
        "fixed_torsion_delta_deg",
        "C_relative_to_baseline",
        "D_relative_to_baseline",
        "C_percent_change",
        "D_percent_change",
    ]
    return f"""# Repeated Phi/Psi Intensity Sensitivity

This report analyzes peak intensity/score changes for geometry-safe repeated CYP->GLU fixed-omega variants. Intensity sensitivity is not the same as peak-position movement.

- C/D peak positions flat: {'yes' if peak_positions_flat(result) else 'no'}
- C intensity trend: {c_trend}
- D intensity trend: {d_trend}
- C percent-change span: {c_span:.4f}%
- D percent-change span: {d_span:.4f}%
- More intensity-sensitive band in this fixed-omega pilot: {more_sensitive}
- Practical interpretation: changes are {meaningful}.

This is a tiny fixed-omega pilot, not the full systematic search. Omega sensitivity remains deferred.

## Relative Intensity Table

{markdown_table(result, columns)}
"""


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    """Render selected columns as markdown."""
    if df.empty:
        return "_None._"
    columns = [column for column in columns if column in df.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for record in df[columns].itertuples(index=False):
        values = []
        for value in record:
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def save_plot(result: pd.DataFrame, figure_base: Path) -> None:
    """Save relative C/D intensity plot."""
    if result.empty:
        return
    result = sort_by_delta(result)
    x = pd.to_numeric(result["fixed_torsion_delta_deg"], errors="coerce")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, result["C_relative_to_baseline"], marker="o", label="C relative intensity")
    ax.plot(x, result["D_relative_to_baseline"], marker="o", label="D relative intensity")
    ax.axhline(1.0, color="0.5", ls="--", lw=1, label="delta 0 baseline")
    ax.set_xlabel("fixed phi0 delta applied repeatedly (deg)")
    ax.set_ylabel("relative peak intensity / score")
    ax.set_title("Repeated phi/psi C/D intensity sensitivity")
    ax.legend(fontsize=8)
    fig.tight_layout()
    figure_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_base.with_suffix(".png"), dpi=180)
    fig.savefig(figure_base.with_suffix(".svg"))
    plt.close(fig)


def analyze(scores_csv: Path, out_csv: Path, report_path: Path, figure_base: Path) -> pd.DataFrame:
    """Analyze repeated variant intensity sensitivity and write outputs."""
    scores = pd.read_csv(scores_csv)
    result = normalize_to_baseline(scores)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_csv, index=False)
    report_path.write_text(build_report_text(result), encoding="utf-8")
    save_plot(result, figure_base)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-csv", type=Path, default=DEFAULT_SCORES_CSV)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--figure-base", type=Path, default=DEFAULT_FIGURE_BASE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = analyze(args.scores_csv, args.out_csv, args.report, args.figure_base)
    print(f"Analyzed {len(result)} repeated variant intensity rows")
    print(f"CSV: {args.out_csv}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
