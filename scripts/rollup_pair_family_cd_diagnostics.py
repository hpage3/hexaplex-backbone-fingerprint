"""Roll up pair-family C/D diagnostic summaries across models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATTERN = "*_pair_family_cd_pair_family_cd_summary.csv"
FOCUS_FAMILIES = [
    "same_strand_plusminus1_repeat",
    "adjacent_strand_same_register",
    "all_cross_strand",
    "all_same_strand",
    "alternating_interfaces_AB_CD_EF",
    "alternating_interfaces_BC_DE_FA",
]


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric column, using -inf if missing."""
    if column not in df.columns:
        return pd.Series(float("-inf"), index=df.index)
    return pd.to_numeric(df[column], errors="coerce").fillna(float("-inf"))


def top_family(df: pd.DataFrame, column: str) -> tuple[str, float]:
    """Return top family and value for a metric column."""
    values = numeric_series(df, column)
    if values.empty or values.max() == float("-inf"):
        return "", float("nan")
    idx = values.idxmax()
    return str(df.loc[idx, "family"]), float(values.loc[idx])


def focus_value(df: pd.DataFrame, family: str, column: str) -> float:
    """Return one focused family metric value, or 0/NaN if unavailable."""
    subset = df[df["family"] == family]
    if subset.empty or column not in df.columns:
        return 0.0
    value = pd.to_numeric(subset.iloc[0][column], errors="coerce")
    return float(value) if pd.notna(value) else float("nan")


def model_id_from_df_or_path(df: pd.DataFrame, path: Path) -> str:
    """Read model_id from the file when possible; otherwise infer from filename."""
    if "model_id" in df.columns and not df["model_id"].dropna().empty:
        return str(df["model_id"].dropna().iloc[0])
    suffix = "_pair_family_cd_pair_family_cd_summary"
    stem = path.stem
    return stem[: -len(suffix)] if stem.endswith(suffix) else stem


def summarize_file(path: Path) -> dict[str, object]:
    """Summarize one pair-family C/D summary CSV."""
    df = pd.read_csv(path)
    if "family" not in df.columns:
        raise ValueError(f"{path} is missing required column 'family'.")
    model_id = model_id_from_df_or_path(df, path)
    top_c_count, top_c_count_value = top_family(df, "C_pair_count")
    top_d_count, top_d_count_value = top_family(df, "D_pair_count")
    top_c_profile, top_c_profile_value = top_family(df, "C_profile_max_intensity")
    top_d_profile, top_d_profile_value = top_family(df, "D_profile_max_intensity")

    row: dict[str, object] = {
        "model_id": model_id,
        "source_file": str(path),
        "top_C_pair_count_family": top_c_count,
        "top_C_pair_count": top_c_count_value,
        "top_D_pair_count_family": top_d_count,
        "top_D_pair_count": top_d_count_value,
        "top_C_profile_family": top_c_profile,
        "top_C_profile_max_intensity": top_c_profile_value,
        "top_D_profile_family": top_d_profile,
        "top_D_profile_max_intensity": top_d_profile_value,
    }
    for family in FOCUS_FAMILIES:
        prefix = family
        row[f"{prefix}_C_pair_count"] = focus_value(df, family, "C_pair_count")
        row[f"{prefix}_D_pair_count"] = focus_value(df, family, "D_pair_count")
        row[f"{prefix}_C_profile_max_intensity"] = focus_value(df, family, "C_profile_max_intensity")
        row[f"{prefix}_D_profile_max_intensity"] = focus_value(df, family, "D_profile_max_intensity")
    row["alternating_AB_CD_EF_minus_BC_DE_FA_C_pair_count"] = (
        row["alternating_interfaces_AB_CD_EF_C_pair_count"]
        - row["alternating_interfaces_BC_DE_FA_C_pair_count"]
    )
    row["alternating_AB_CD_EF_minus_BC_DE_FA_D_pair_count"] = (
        row["alternating_interfaces_AB_CD_EF_D_pair_count"]
        - row["alternating_interfaces_BC_DE_FA_D_pair_count"]
    )
    return row


def write_report(rollup: pd.DataFrame, path: Path) -> None:
    """Write a concise markdown rollup report."""
    if rollup.empty:
        text = "# Pair-Family C/D Diagnostic Rollup\n\nNo diagnostic files were found.\n"
        path.write_text(text, encoding="utf-8")
        return

    top_table = markdown_table(rollup[
        [
            "model_id",
            "top_C_pair_count_family",
            "top_C_pair_count",
            "top_D_pair_count_family",
            "top_D_pair_count",
            "top_C_profile_family",
            "top_D_profile_family",
        ]
    ])
    alt_table = markdown_table(rollup[
        [
            "model_id",
            "alternating_interfaces_AB_CD_EF_C_pair_count",
            "alternating_interfaces_BC_DE_FA_C_pair_count",
            "alternating_AB_CD_EF_minus_BC_DE_FA_C_pair_count",
            "alternating_interfaces_AB_CD_EF_D_pair_count",
            "alternating_interfaces_BC_DE_FA_D_pair_count",
            "alternating_AB_CD_EF_minus_BC_DE_FA_D_pair_count",
        ]
    ])

    c_same = (rollup["top_C_pair_count_family"] == "same_strand_plusminus1_repeat").sum()
    c_local = rollup["top_C_pair_count_family"].str.startswith("same_strand").sum()
    d_cross = rollup["top_D_pair_count_family"].isin(["all_cross_strand", "all_adjacent_cross_strand"]).sum()
    d_register = (rollup["top_D_pair_count_family"] == "adjacent_strand_same_register").sum()
    c_split_median = rollup["alternating_AB_CD_EF_minus_BC_DE_FA_C_pair_count"].median()
    d_split_median = rollup["alternating_AB_CD_EF_minus_BC_DE_FA_D_pair_count"].median()

    text = f"""# Pair-Family C/D Diagnostic Rollup

## Files Summarized

- Diagnostic files summarized: {len(rollup)}

## Per-Model Top Families

{top_table}

## Alternating Interface Comparison

{alt_table}

## Cautious Interpretation

- C local-repeat tendency: `{c_local}` of `{len(rollup)}` models had a same-strand family as the top C pair-count family; `{c_same}` specifically had `same_strand_plusminus1_repeat`.
- D cross-strand/register tendency: `{d_cross}` of `{len(rollup)}` models had an all-cross/adjacent-cross family as the top D pair-count family; `{d_register}` specifically had `adjacent_strand_same_register`.
- Alternating interface split: median AB/CD/EF minus BC/DE/FA C-pair difference was `{c_split_median:g}`; median D-pair difference was `{d_split_median:g}`.

This rollup is diagnostic only. It summarizes pair counts and pair-only partial Debye profiles from the current generated/labeled models. It should be used to prioritize follow-up models, not as a final structural assignment.
"""
    path.write_text(text, encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> str:
    """Render a small dataframe as a GitHub-flavored markdown table without tabulate."""
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in df.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def rollup(metrics_dir: Path, output_prefix: str) -> tuple[pd.DataFrame, Path, Path]:
    """Create rollup CSV and markdown report."""
    files = sorted(metrics_dir.glob(DEFAULT_PATTERN))
    rows = [summarize_file(path) for path in files]
    df = pd.DataFrame(rows)
    csv_path = metrics_dir / f"{output_prefix}.csv"
    reports_dir = metrics_dir.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"{output_prefix}_report.md"
    df.to_csv(csv_path, index=False)
    write_report(df, report_path)
    return df, csv_path, report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", type=Path, default=Path("outputs/metrics"))
    parser.add_argument("--output-prefix", default="pair_family_cd_rollup")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df, csv_path, report_path = rollup(args.metrics_dir, args.output_prefix)
    print(f"Summarized {len(df)} diagnostic files")
    print(f"CSV: {csv_path}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
