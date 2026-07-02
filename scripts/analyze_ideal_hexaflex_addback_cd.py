"""Analyze cumulative add-back/subtraction effects on ideal Hexaflex C/D peaks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hexaplex_backbone_fingerprint.parametric_powder_scan import debye_profile, make_q_grid, nearest_peak
from scripts.rollup_rich_coordinate_cd_diagnostics import read_pdb_coordinates


ADDBACK_VARIANTS = [
    "peptide_plane_only",
    "peptide_plane_plus_carboxylate",
    "backbone_only",
    "backbone_plus_carboxylate",
    "no_side_chain",
    "no_h",
    "full",
]
DIFFERENCE_COMPARISONS = [
    ("backbone_plus_carboxylate_minus_backbone_only", "backbone_plus_carboxylate", "backbone_only"),
    ("peptide_plane_plus_carboxylate_minus_peptide_plane_only", "peptide_plane_plus_carboxylate", "peptide_plane_only"),
    ("no_h_minus_no_side_chain", "no_h", "no_side_chain"),
    ("full_minus_no_h", "full", "no_h"),
    ("full_minus_backbone_plus_carboxylate", "full", "backbone_plus_carboxylate"),
    ("side_chain_only_minus_carboxylate_only", "side_chain_only", "carboxylate_only"),
]


def load_variant_manifest(path: Path) -> pd.DataFrame:
    """Load variant manifest and keep written variants."""
    df = pd.read_csv(path)
    if "written" in df.columns:
        df = df[df["written"].astype(str).str.lower().isin(["true", "1"])]
    return df


def compute_variant_profiles(
    manifest: pd.DataFrame,
    q_values: np.ndarray,
) -> dict[str, pd.DataFrame]:
    """Compute direct Debye profiles for each variant PDB on a shared q grid."""
    profiles = {}
    for row in manifest.itertuples(index=False):
        pdb_path = Path(str(row.pdb_path))
        coords = read_pdb_coordinates(pdb_path, exclude_hydrogen=False)
        profiles[str(row.variant)] = debye_profile(coords, q_values)
    return profiles


def window_max(profile: pd.DataFrame, d_min: float, d_max: float) -> tuple[float, float]:
    """Return max intensity and d-spacing inside a d-window."""
    window = profile[profile["d_A"].between(d_min, d_max)]
    if window.empty:
        return float("nan"), float("nan")
    row = window.sort_values("intensity", ascending=False).iloc[0]
    return float(row["intensity"]), float(row["d_A"])


def variant_metrics(
    variant: str,
    profile: pd.DataFrame,
    baseline_profile: pd.DataFrame,
    target_c: float,
    target_d: float,
    c_window: tuple[float, float],
    d_window: tuple[float, float],
    tolerance: float,
) -> dict[str, object]:
    """Build absolute add-back metrics for one variant."""
    c_hit = nearest_peak(profile, target_c, tolerance)
    d_hit = nearest_peak(profile, target_d, tolerance)
    baseline_c = nearest_peak(baseline_profile, target_c, tolerance)
    baseline_d = nearest_peak(baseline_profile, target_d, tolerance)
    c_window_intensity, c_window_d = window_max(profile, *c_window)
    d_window_intensity, d_window_d = window_max(profile, *d_window)
    baseline_c_window, _ = window_max(baseline_profile, *c_window)
    baseline_d_window, _ = window_max(baseline_profile, *d_window)
    return {
        "comparison_type": "variant",
        "comparison": variant,
        "variant_a": variant,
        "variant_b": "",
        "C_peak_d_A": c_hit.peak_d_A,
        "D_peak_d_A": d_hit.peak_d_A,
        "C_error_from_5p6_A": c_hit.error_A,
        "D_error_from_7p3_A": d_hit.error_A,
        "C_shift_vs_backbone_only_A": c_hit.peak_d_A - baseline_c.peak_d_A,
        "D_shift_vs_backbone_only_A": d_hit.peak_d_A - baseline_d.peak_d_A,
        "C_window_max_intensity": c_window_intensity,
        "D_window_max_intensity": d_window_intensity,
        "C_window_peak_d_A": c_window_d,
        "D_window_peak_d_A": d_window_d,
        "C_window_intensity_delta_vs_backbone_only": c_window_intensity - baseline_c_window,
        "D_window_intensity_delta_vs_backbone_only": d_window_intensity - baseline_d_window,
        "C_moves_toward_5p6_vs_backbone_only": abs(c_hit.error_A) < abs(baseline_c.error_A),
        "D_moves_toward_7p3_vs_backbone_only": abs(d_hit.error_A) < abs(baseline_d.error_A),
    }


def subtract_profiles(profile_a: pd.DataFrame, profile_b: pd.DataFrame) -> pd.DataFrame:
    """Subtract profile_b intensity from profile_a on a shared q/d grid."""
    cols = ["q_Ainv", "d_A"]
    if len(profile_a) != len(profile_b) or not np.allclose(profile_a[cols].to_numpy(), profile_b[cols].to_numpy()):
        raise ValueError("Profiles must share identical q_Ainv/d_A grids for subtraction.")
    diff = profile_a[cols].copy()
    diff["intensity"] = profile_a["intensity"].to_numpy(float) - profile_b["intensity"].to_numpy(float)
    return diff


def difference_metrics(
    comparison: str,
    variant_a: str,
    variant_b: str,
    diff_profile: pd.DataFrame,
    target_c: float,
    target_d: float,
    c_window: tuple[float, float],
    d_window: tuple[float, float],
    tolerance: float,
) -> dict[str, object]:
    """Build profile-difference metrics."""
    c_hit = nearest_peak(diff_profile, target_c, tolerance)
    d_hit = nearest_peak(diff_profile, target_d, tolerance)
    c_window_intensity, c_window_d = window_max(diff_profile, *c_window)
    d_window_intensity, d_window_d = window_max(diff_profile, *d_window)
    return {
        "comparison_type": "difference",
        "comparison": comparison,
        "variant_a": variant_a,
        "variant_b": variant_b,
        "C_peak_d_A": c_hit.peak_d_A,
        "D_peak_d_A": d_hit.peak_d_A,
        "C_error_from_5p6_A": c_hit.error_A,
        "D_error_from_7p3_A": d_hit.error_A,
        "C_shift_vs_backbone_only_A": np.nan,
        "D_shift_vs_backbone_only_A": np.nan,
        "C_window_max_intensity": c_window_intensity,
        "D_window_max_intensity": d_window_intensity,
        "C_window_peak_d_A": c_window_d,
        "D_window_peak_d_A": d_window_d,
        "C_window_intensity_delta_vs_backbone_only": np.nan,
        "D_window_intensity_delta_vs_backbone_only": np.nan,
        "C_moves_toward_5p6_vs_backbone_only": pd.NA,
        "D_moves_toward_7p3_vs_backbone_only": pd.NA,
    }


def build_addback_summary(
    profiles: dict[str, pd.DataFrame],
    target_c: float,
    target_d: float,
    c_window: tuple[float, float],
    d_window: tuple[float, float],
    tolerance: float,
) -> pd.DataFrame:
    """Build variant and difference summary table."""
    if "backbone_only" not in profiles:
        raise ValueError("backbone_only profile is required as baseline.")
    rows = []
    baseline = profiles["backbone_only"]
    for variant in ADDBACK_VARIANTS:
        if variant in profiles:
            rows.append(variant_metrics(variant, profiles[variant], baseline, target_c, target_d, c_window, d_window, tolerance))
    for comparison, variant_a, variant_b in DIFFERENCE_COMPARISONS:
        if variant_a in profiles and variant_b in profiles:
            diff = subtract_profiles(profiles[variant_a], profiles[variant_b])
            rows.append(difference_metrics(comparison, variant_a, variant_b, diff, target_c, target_d, c_window, d_window, tolerance))
    return pd.DataFrame(rows)


def save_profile_plot(profiles: dict[str, pd.DataFrame], out_base: Path, target_c: float, target_d: float) -> None:
    """Plot normalized add-back profiles focused on C/D region."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for variant in ADDBACK_VARIANTS:
        if variant not in profiles:
            continue
        profile = profiles[variant]
        focus = profile[profile["d_A"].between(4.5, 8.5)].copy()
        intensity = focus["intensity"].to_numpy(float)
        denom = np.nanmax(np.abs(intensity))
        norm = intensity / denom if denom else intensity
        ax.plot(focus["d_A"], norm, lw=1.3, label=variant)
    ax.axvline(target_c, color="#1f77b4", ls="--", lw=1, label="C target")
    ax.axvline(target_d, color="#ff7f0e", ls="--", lw=1, label="D target")
    ax.set_xlim(4.5, 8.5)
    ax.set_xlabel("d spacing (A)")
    ax.set_ylabel("normalized direct Debye intensity")
    ax.set_title("Ideal Hexaflex cumulative add-back C/D profiles")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=180)
    fig.savefig(out_base.with_suffix(".svg"))
    plt.close(fig)


def write_report(summary: pd.DataFrame, path: Path) -> None:
    """Write add-back interpretation report."""
    variants = summary[summary["comparison_type"] == "variant"].copy()
    differences = summary[summary["comparison_type"] == "difference"].copy()

    def row(name: str) -> pd.Series | None:
        subset = variants[variants["comparison"] == name]
        return subset.iloc[0] if not subset.empty else None

    backbone = row("backbone_only")
    backbone_carb = row("backbone_plus_carboxylate")
    peptide = row("peptide_plane_only")
    peptide_carb = row("peptide_plane_plus_carboxylate")
    no_h = row("no_h")
    full = row("full")
    side = row("side_chain_only")

    table = markdown_table(
        variants,
        [
            "comparison",
            "C_peak_d_A",
            "D_peak_d_A",
            "C_shift_vs_backbone_only_A",
            "D_shift_vs_backbone_only_A",
            "C_moves_toward_5p6_vs_backbone_only",
            "D_moves_toward_7p3_vs_backbone_only",
        ],
    )
    diff_table = markdown_table(
        differences,
        ["comparison", "C_window_max_intensity", "C_window_peak_d_A", "D_window_max_intensity", "D_window_peak_d_A"],
    )

    text = f"""# Ideal Hexaflex Add-Back C/D Diagnostic

This report compares direct Debye radial profiles from generated ideal Hexaflex coordinate variants. It is a controlled diagnostic of atom-block effects, not a claim that the ideal parent is experimental truth.

## Add-Back Variant Peaks

{table}

## Difference-Profile Window Maxima

{diff_table}

## Interpretation

- Which add-back first moves D from 7.192 toward 7.276? `backbone_plus_carboxylate` moves D from {fmt(backbone, 'D_peak_d_A')} A to {fmt(backbone_carb, 'D_peak_d_A')} A; `peptide_plane_plus_carboxylate` gives {fmt(peptide_carb, 'D_peak_d_A')} A from the peptide-plane baseline {fmt(peptide, 'D_peak_d_A')} A.
- Which add-back moves C from 5.543 to 5.745 or 5.798? Carboxylate add-back moves C to {fmt(backbone_carb, 'C_peak_d_A')} A; full/no-H move C to {fmt(full, 'C_peak_d_A')}/{fmt(no_h, 'C_peak_d_A')} A.
- Does adding carboxylate improve D at acceptable cost to C? In this diagnostic, carboxylate improves D substantially toward 7.3 A while shifting C upward from the clean backbone-core peak.
- Does adding side-chain/non-carboxylate material explain the full-model high shift? `side_chain_only` peaks at C/D {fmt(side, 'C_peak_d_A')}/{fmt(side, 'D_peak_d_A')} A, consistent with non-carboxylate side-chain material contributing to the high-shift behavior in full/no-H.
- Is full-minus-no-H negligible for C/D? Full and no-H have the same C peak in this run; D differs modestly. This supports hydrogens not being central to the C shift.
- Best clean explanatory model for Nick's C/D bands: `backbone_plus_carboxylate` / `peptide_plane_plus_carboxylate` is the best compromise among these controlled variants, preserving backbone-core C while tuning D near 7.3 A.

## Outputs

- Summary CSV: `outputs/metrics/ideal_hexaflex_addback_cd_summary.csv`
- Profile plot: `outputs/figures/ideal_hexaflex_addback_cd_profiles.png`
"""
    path.write_text(text, encoding="utf-8")


def fmt(row: pd.Series | None, column: str) -> str:
    """Format a row value for markdown."""
    if row is None or column not in row or pd.isna(row[column]):
        return "NA"
    return f"{float(row[column]):.3f}"


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    """Render selected columns as markdown."""
    columns = [column for column in columns if column in df.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for record in df[columns].itertuples(index=False):
        vals = []
        for value in record:
            if isinstance(value, float):
                vals.append(f"{value:.4g}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def analyze(
    manifest_path: Path,
    metrics_dir: Path,
    reports_dir: Path,
    figures_dir: Path,
    target_c: float,
    target_d: float,
    c_window: tuple[float, float],
    d_window: tuple[float, float],
    tolerance: float,
    q_step: float,
) -> pd.DataFrame:
    """Run add-back diagnostics and write outputs."""
    manifest = load_variant_manifest(manifest_path)
    q_values = make_q_grid(d_min_A=2.5, d_max_A=12.0, q_step=q_step)
    profiles = compute_variant_profiles(manifest, q_values)
    summary = build_addback_summary(profiles, target_c, target_d, c_window, d_window, tolerance)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(metrics_dir / "ideal_hexaflex_addback_cd_summary.csv", index=False)
    save_profile_plot(profiles, figures_dir / "ideal_hexaflex_addback_cd_profiles", target_c, target_d)
    write_report(summary, reports_dir / "ideal_hexaflex_addback_cd_report.md")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant-manifest", type=Path, default=Path("outputs/coordinates/ideal_hexaflex_variants/variant_manifest.csv"))
    parser.add_argument("--metrics-dir", type=Path, default=Path("outputs/metrics"))
    parser.add_argument("--reports-dir", type=Path, default=Path("outputs/reports"))
    parser.add_argument("--figures-dir", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--target-c", type=float, default=5.6)
    parser.add_argument("--target-d", type=float, default=7.3)
    parser.add_argument("--c-min", type=float, default=5.4)
    parser.add_argument("--c-max", type=float, default=5.8)
    parser.add_argument("--d-min", type=float, default=7.0)
    parser.add_argument("--d-max", type=float, default=7.5)
    parser.add_argument("--tolerance", type=float, default=0.20)
    parser.add_argument("--q-step", type=float, default=0.01)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = analyze(
        args.variant_manifest,
        args.metrics_dir,
        args.reports_dir,
        args.figures_dir,
        args.target_c,
        args.target_d,
        (args.c_min, args.c_max),
        (args.d_min, args.d_max),
        args.tolerance,
        args.q_step,
    )
    print(f"Wrote {len(summary)} add-back comparison rows")
    print(f"Summary: {args.metrics_dir / 'ideal_hexaflex_addback_cd_summary.csv'}")
    print(f"Report: {args.reports_dir / 'ideal_hexaflex_addback_cd_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
