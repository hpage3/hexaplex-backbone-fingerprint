"""Run diagnostic Debye powder scans for parametric peptide-plane models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hexaplex_backbone_fingerprint.parametric_powder_scan import (
    debye_profile,
    load_xyz_coordinates,
    local_maxima,
    make_q_grid,
    nearest_peak,
    rank_powder_summary,
)


DEFAULT_MANIFEST = Path("outputs/parametric_six_strand_peptide_plane_models/model_manifest.csv")
DEFAULT_OUTDIR = Path("outputs/parametric_six_strand_powder_scan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--target-c", type=float, default=5.6)
    parser.add_argument("--target-d", type=float, default=7.3)
    parser.add_argument("--tolerance", type=float, default=0.20)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1, help="Reserved for future parallel execution.")
    parser.add_argument("--q-step", type=float, default=0.005)
    parser.add_argument("--d-min", type=float, default=2.5)
    parser.add_argument("--d-max", type=float, default=12.0)
    parser.add_argument("--write-profiles", action="store_true", help="Write every per-model radial profile CSV.")
    parser.add_argument("--baseline-summary", type=Path, help="Optional first-stage summary CSV for before/after comparison.")
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return ROOT / path


def analyze_model(row: pd.Series, q_values: np.ndarray, args: argparse.Namespace, profile_dir: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    xyz_path = resolve_path(str(row["xyz_path"]))
    coords = load_xyz_coordinates(xyz_path, exclude_hydrogen=False)
    profile = debye_profile(coords, q_values)
    peaks = local_maxima(profile)
    c_hit = nearest_peak(profile, args.target_c, args.tolerance)
    d_hit = nearest_peak(profile, args.target_d, args.tolerance)
    model_label = row["model_label"]
    if args.write_profiles:
        profile.to_csv(profile_dir / f"{model_label}_radial_profile.csv", index=False)

    peak_rows = []
    for peak in peaks.itertuples():
        peak_rows.append(
            {
                "model_label": model_label,
                "q_Ainv": peak.q_Ainv,
                "d_A": peak.d_A,
                "intensity": peak.intensity,
            }
        )

    summary = {
        "model_label": model_label,
        "xyz_path": row["xyz_path"],
        "pdb_path": row.get("pdb_path", ""),
        "twist_deg": row["twist_deg"],
        "rise_A": row["rise_A"],
        "helix_radius_A": row["helix_radius_A"],
        "plane_normal_to_axis_deg": row["plane_normal_to_axis_deg"],
        "plane_azimuth_deg": row["plane_azimuth_deg"],
        "in_plane_spin_deg": row["in_plane_spin_deg"],
        "uniform_adjacent_z_offset_A": row.get("uniform_adjacent_z_offset_A", row.get("strand_z_offset_A", 0.0)),
        "z_offset_mode": row.get("z_offset_mode", "uniform_adjacent"),
        "nearest_C_peak_d_A": c_hit.peak_d_A,
        "nearest_C_error_A": c_hit.error_A,
        "nearest_C_intensity": c_hit.intensity,
        "nearest_D_peak_d_A": d_hit.peak_d_A,
        "nearest_D_error_A": d_hit.error_A,
        "nearest_D_intensity": d_hit.intensity,
        "CD_combined_abs_error_A": abs(c_hit.error_A) + abs(d_hit.error_A),
        "C_found_within_tolerance": c_hit.found_within_tolerance,
        "D_found_within_tolerance": d_hit.found_within_tolerance,
        "both_C_and_D_found": c_hit.found_within_tolerance and d_hit.found_within_tolerance,
    }
    return summary, profile.assign(model_label=model_label), pd.DataFrame(peak_rows)


def pivot_metric(df: pd.DataFrame, metric: str, aggfunc: str = "mean") -> pd.DataFrame:
    return df.pivot_table(
        index="plane_normal_to_axis_deg",
        columns="plane_azimuth_deg",
        values=metric,
        aggfunc=aggfunc,
    ).sort_index(ascending=True).sort_index(axis=1)


def save_heatmap(df: pd.DataFrame, metric: str, path: Path, title: str, cmap: str = "viridis", aggfunc: str = "mean") -> None:
    pivot = pivot_metric(df, metric, aggfunc=aggfunc)
    fig, ax = plt.subplots(figsize=(6, 4.8))
    image = ax.imshow(pivot.values, origin="lower", aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(pivot.columns)), [f"{col:g}" for col in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [f"{idx:g}" for idx in pivot.index])
    ax.set_xlabel("plane_azimuth_deg")
    ax.set_ylabel("plane_normal_to_axis_deg")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label=metric)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_grouped_line(summary: pd.DataFrame, group_col: str, metrics: list[tuple[str, str]], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    grouped = summary.groupby(group_col)
    for metric, label in metrics:
        values = grouped[metric].min().sort_index()
        ax.plot(values.index, values.values, marker="o", label=label)
    ax.set_xlabel(group_col)
    ax.set_ylabel("best absolute error (A)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_hit_counts_by_group(summary: pd.DataFrame, group_col: str, path: Path, title: str) -> None:
    grouped = summary.groupby(group_col)[
        ["C_found_within_tolerance", "D_found_within_tolerance", "both_C_and_D_found"]
    ].sum().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(grouped.index))
    width = 0.25
    ax.bar(x - width, grouped["C_found_within_tolerance"], width=width, label="C")
    ax.bar(x, grouped["D_found_within_tolerance"], width=width, label="D")
    ax.bar(x + width, grouped["both_C_and_D_found"], width=width, label="C+D")
    ax.set_xticks(x, [f"{value:g}" for value in grouped.index])
    ax.set_xlabel(group_col)
    ax.set_ylabel("model count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_two_parameter_heatmap(
    summary: pd.DataFrame,
    x_col: str,
    y_col: str,
    metric: str,
    path: Path,
    title: str,
    aggfunc: str = "min",
) -> None:
    pivot = (
        summary.pivot_table(index=y_col, columns=x_col, values=metric, aggfunc=aggfunc)
        .sort_index(ascending=True)
        .sort_index(axis=1)
    )
    fig, ax = plt.subplots(figsize=(7, 4.8))
    image = ax.imshow(pivot.values, origin="lower", aspect="auto", cmap="magma_r")
    ax.set_xticks(range(len(pivot.columns)), [f"{col:g}" for col in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [f"{idx:g}" for idx in pivot.index])
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label=metric)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_best_profiles(profiles: dict[str, pd.DataFrame], best: pd.DataFrame, outdir: Path, target_c: float, target_d: float) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for row in best.head(5).itertuples():
        profile = profiles[row.model_label]
        intensity = profile["intensity"].to_numpy(float)
        norm_intensity = intensity / np.nanmax(intensity)
        ax.plot(profile["d_A"], norm_intensity, lw=1.2, label=row.model_label)
    ax.axvline(target_c, color="#1f77b4", ls="--", lw=1, label="C target")
    ax.axvline(target_d, color="#ff7f0e", ls="--", lw=1, label="D target")
    ax.set_xlim(4.5, 8.5)
    ax.set_xlabel("d spacing (A)")
    ax.set_ylabel("normalized intensity")
    ax.set_title("Best diagnostic Debye radial profiles")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(outdir / "best_model_radial_profiles.png", dpi=180)
    plt.close(fig)


def save_best_orientation_plot(best: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    subset = best.head(24).copy()
    colors = np.where(subset["both_C_and_D_found"], "#2ca02c", "#7f7f7f")
    sizes = 80 + 300 / (1.0 + subset["CD_combined_abs_error_A"].to_numpy(float))
    ax.scatter(subset["plane_azimuth_deg"], subset["plane_normal_to_axis_deg"], s=sizes, c=colors, alpha=0.75)
    ax.set_xlabel("plane_azimuth_deg")
    ax.set_ylabel("plane_normal_to_axis_deg")
    ax.set_title("Best models by peptide-plane orientation")
    fig.tight_layout()
    fig.savefig(outdir / "best_models_by_orientation.png", dpi=180)
    plt.close(fig)


def best_stats(summary: pd.DataFrame) -> dict[str, float]:
    ranked = rank_powder_summary(summary)
    return {
        "best_C_abs_error": float(summary["nearest_C_error_A"].abs().min()),
        "best_D_abs_error": float(summary["nearest_D_error_A"].abs().min()),
        "best_combined_abs_error": float(ranked.iloc[0]["CD_combined_abs_error_A"]),
        "C_hits": int(summary["C_found_within_tolerance"].sum()),
        "D_hits": int(summary["D_found_within_tolerance"].sum()),
        "both_hits": int(summary["both_C_and_D_found"].sum()),
    }


def dominant_values(best: pd.DataFrame, column: str, n: int = 12) -> str:
    subset = best.head(n)
    counts = subset[column].value_counts()
    return "; ".join(f"{idx:g}: {count}" for idx, count in counts.items())


def write_report(outdir: Path, summary: pd.DataFrame, best: pd.DataFrame, args: argparse.Namespace) -> None:
    both_count = int(summary["both_C_and_D_found"].sum())
    c_count = int(summary["C_found_within_tolerance"].sum())
    d_count = int(summary["D_found_within_tolerance"].sum())
    best_row = best.iloc[0]
    summary = summary.copy()
    summary["C_abs_error_A"] = summary["nearest_C_error_A"].abs()
    summary["D_abs_error_A"] = summary["nearest_D_error_A"].abs()
    best_c = summary.sort_values("C_abs_error_A").iloc[0]
    best_d = summary.sort_values("D_abs_error_A").iloc[0]
    has_z_offset = "uniform_adjacent_z_offset_A" in summary.columns
    z_offset_text = ""
    if has_z_offset:
        z_group = (
            summary.groupby("uniform_adjacent_z_offset_A")
            .agg(
                best_C_abs_error_A=("C_abs_error_A", "min"),
                best_D_abs_error_A=("D_abs_error_A", "min"),
                best_combined_abs_error_A=("CD_combined_abs_error_A", "min"),
                C_hits=("C_found_within_tolerance", "sum"),
                D_hits=("D_found_within_tolerance", "sum"),
                both_hits=("both_C_and_D_found", "sum"),
            )
            .sort_index()
        )
        favored_combined = z_group["best_combined_abs_error_A"].idxmin()
        favored_c = z_group["best_C_abs_error_A"].idxmin()
        favored_d = z_group["best_D_abs_error_A"].idxmin()
        hit_z = z_group[z_group["both_hits"] > 0].index.tolist()
        hit_z_text = ", ".join(f"{value:g}" for value in hit_z) if hit_z else "none"
        z_offset_text = f"""
## Z-Offset Effects

- Best combined z-offset: {favored_combined:g} A
- Best C-only z-offset: {favored_c:g} A
- Best D-only z-offset: {favored_d:g} A
- Z-offset values with simultaneous C/D hits: {hit_z_text}
- Best C-only model: z-offset {best_c.uniform_adjacent_z_offset_A:g} A, radius {best_c.helix_radius_A:g} A, normal/azimuth {best_c.plane_normal_to_axis_deg:g}/{best_c.plane_azimuth_deg:g}, spin {best_c.in_plane_spin_deg:g}; C peak {best_c.nearest_C_peak_d_A:.3f} A, D peak {best_c.nearest_D_peak_d_A:.3f} A.
- Best D-only model: z-offset {best_d.uniform_adjacent_z_offset_A:g} A, radius {best_d.helix_radius_A:g} A, normal/azimuth {best_d.plane_normal_to_axis_deg:g}/{best_d.plane_azimuth_deg:g}, spin {best_d.in_plane_spin_deg:g}; C peak {best_d.nearest_C_peak_d_A:.3f} A, D peak {best_d.nearest_D_peak_d_A:.3f} A.

In this panel, z-offset affects peak positions, not only intensities. The strongest C improvement occurs at large adjacent-strand z-offset, but the D-like feature shifts away from 7.3 A in those same models. D is preserved across several small-to-moderate z-offset values, but those models keep the nearest C-like feature too low.
"""
    baseline_text = ""
    if args.baseline_summary and args.baseline_summary.exists():
        baseline = pd.read_csv(args.baseline_summary)
        before = best_stats(baseline)
        after = best_stats(summary)
        baseline_label = "Previous refined sweep" if "refined" in str(args.baseline_summary).lower() else "Baseline sweep"
        baseline_text = f"""
## Baseline Comparison

| Metric | {baseline_label} | Current sweep |
|---|---:|---:|
| Best C absolute error | {before['best_C_abs_error']:.3f} | {after['best_C_abs_error']:.3f} |
| Best D absolute error | {before['best_D_abs_error']:.3f} | {after['best_D_abs_error']:.3f} |
| Best combined C/D absolute error | {before['best_combined_abs_error']:.3f} | {after['best_combined_abs_error']:.3f} |
| C hits within tolerance | {before['C_hits']} | {after['C_hits']} |
| D hits within tolerance | {before['D_hits']} | {after['D_hits']} |
| Both C and D hits | {before['both_hits']} | {after['both_hits']} |
"""
    text = f"""# Parametric Six-Strand Peptide-Bond-Plane Powder Scan

This is a direct forward-modeling test of Nick's hypothesis: simple six-stranded assemblies made only from peptide-bond-plane atoms may generate C and D powder features depending on peptide-plane orientation relative to the helical axis.

## Method

Existing diffraction tools were inspected in the neighboring `hexaplex-formation` repo. They include Debye/radial scripts, but they depend on that repo's internal PDB utilities and are not a clean drop-in for this generated XYZ panel. This run therefore uses a local **diagnostic point-scatterer Debye powder model**:

`I(q) = sum_i sum_j sinc(q r_ij / pi)`, with equal atom weights and self terms included.

This is a simplified first-pass powder model, not a full fiber diffraction or chemically weighted scattering calculation.

## Scan Size

- Models analyzed: {len(summary)}
- C target: {args.target_c:.3f} A
- D target: {args.target_d:.3f} A
- Tolerance: +/- {args.tolerance:.3f} A
- Models with C peak within tolerance: {c_count}
- Models with D peak within tolerance: {d_count}
- Models with both C and D peaks within tolerance: {both_count}

## Best Model

- Model: `{best_row.model_label}`
- twist/rise: {best_row.twist_deg:g} deg / {best_row.rise_A:g} A
- radius / z-offset: {best_row.helix_radius_A:g} A / {getattr(best_row, "uniform_adjacent_z_offset_A", 0.0):g} A
- plane normal / azimuth: {best_row.plane_normal_to_axis_deg:g} deg / {best_row.plane_azimuth_deg:g} deg
- in-plane spin: {best_row.in_plane_spin_deg:g} deg
- nearest C peak: {best_row.nearest_C_peak_d_A:.3f} A (error {best_row.nearest_C_error_A:.3f} A)
- nearest D peak: {best_row.nearest_D_peak_d_A:.3f} A (error {best_row.nearest_D_error_A:.3f} A)
- combined absolute C/D error: {best_row.CD_combined_abs_error_A:.3f} A
{baseline_text}

## Orientation Trends

- Favored plane-normal angles among top ranked models: {dominant_values(best, "plane_normal_to_axis_deg")}
- Favored plane azimuths among top ranked models: {dominant_values(best, "plane_azimuth_deg")}
{("- Favored z-offsets among top ranked models: " + dominant_values(best, "uniform_adjacent_z_offset_A")) if has_z_offset else ""}

## Parameter Effects

- Best C absolute error by radius is written to `best_C_D_combined_error_by_radius.png`.
- Best combined error by spin is written to `best_combined_error_by_spin.png`.
- Best normal/azimuth regions are summarized in the refined heatmaps.
{z_offset_text}

## Interpretation

The important question is whether C and D peak positions can arise from peptide-bond-plane orientation alone in a six-strand point-scatterer model. The best-ranked models show whether radius, in-plane spin, and finer orientation sampling can move the nearest C-like feature toward 5.6 A while preserving D near 7.3 A.

Changing radius and strand axial z-offset can move peak positions in this diagnostic model. In this sweep, axial strand phase rescues C-only feasibility in a few large-offset models, but it does not rescue simultaneous C/D recovery because D shifts away when C approaches 5.6 A. The best combined model remains the zero-offset D-preserving model.

This partially supports Nick's hypothesis with an added condition: strand register/phase matters for the simple peptide-bond-plane forward model. However, the current uniform adjacent z-offset is not sufficient by itself. The next missing parameters are likely nonuniform strand offsets, finer radius/register sampling around large z-offset models, longer repeats, a more realistic peptide-bond motif, or a full diffraction workflow.

This remains a diagnostic point-scatterer Debye forward model. It tests peak-position feasibility, not final experimental intensity.

Treat the intensity values and ranking cautiously: equal atom weights, finite model length, no solvent/background, and isotropic Debye averaging are all simplifications.

## Recommended Next Sweep

Because no models hit both targets, the next sweep should vary helix radius and strand phase/z-offset first, then in-plane spin and finer plane-normal/azimuth grids around the best D-favoring orientations. Longer repeats should follow after radius/phase sensitivity is understood.

## Output Files

- `parametric_powder_scan_summary.csv`
- `parametric_powder_peak_table.csv`
- `best_parametric_powder_models.csv`
- `best_model_radial_profiles.png`
- `heatmap_CD_error_by_normal_azimuth.png`
- `heatmap_best_CD_error_by_normal_azimuth.png`
- `heatmap_C_error_by_normal_azimuth.png`
- `heatmap_best_C_error_by_normal_azimuth.png`
- `heatmap_D_error_by_normal_azimuth.png`
- `heatmap_best_D_error_by_normal_azimuth.png`
- `heatmap_C_intensity_by_normal_azimuth.png`
- `heatmap_D_intensity_by_normal_azimuth.png`
- `best_models_by_orientation.png`
- `best_C_D_combined_error_by_radius.png`
- `best_combined_error_by_spin.png`
- `best_combined_error_by_z_offset.png`
- `best_C_error_by_z_offset.png`
- `best_D_error_by_z_offset.png`
- `C_and_D_hit_counts_by_z_offset.png`
- `heatmap_best_CD_error_by_z_offset_radius.png`
"""
    (outdir / "parametric_powder_scan_report.md").write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    profile_dir = args.outdir / "radial_profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    if args.limit is not None:
        manifest = manifest.head(args.limit).copy()

    q_values = make_q_grid(d_min_A=args.d_min, d_max_A=args.d_max, q_step=args.q_step)
    summaries = []
    peak_tables = []
    profiles_by_label = {}
    for row in manifest.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        summary, profile, peaks = analyze_model(row_series, q_values, args, profile_dir)
        summaries.append(summary)
        peak_tables.append(peaks)
        profiles_by_label[summary["model_label"]] = profile

    summary_df = pd.DataFrame(summaries)
    peak_df = pd.concat(peak_tables, ignore_index=True) if peak_tables else pd.DataFrame()
    best_df = rank_powder_summary(summary_df)
    summary_df.to_csv(args.outdir / "parametric_powder_scan_summary.csv", index=False)
    peak_df.to_csv(args.outdir / "parametric_powder_peak_table.csv", index=False)
    best_df.to_csv(args.outdir / "best_parametric_powder_models.csv", index=False)

    save_best_profiles(profiles_by_label, best_df, args.outdir, args.target_c, args.target_d)
    summary_df["C_abs_error_A"] = summary_df["nearest_C_error_A"].abs()
    summary_df["D_abs_error_A"] = summary_df["nearest_D_error_A"].abs()
    save_heatmap(summary_df, "CD_combined_abs_error_A", args.outdir / "heatmap_CD_error_by_normal_azimuth.png", "Mean C+D absolute error by orientation", cmap="magma_r")
    save_heatmap(summary_df, "CD_combined_abs_error_A", args.outdir / "heatmap_best_CD_error_by_normal_azimuth.png", "Best C+D absolute error by orientation", cmap="magma_r", aggfunc="min")
    save_heatmap(summary_df, "nearest_C_error_A", args.outdir / "heatmap_C_error_by_normal_azimuth.png", "Mean signed C error by orientation", cmap="coolwarm")
    save_heatmap(summary_df, "C_abs_error_A", args.outdir / "heatmap_best_C_error_by_normal_azimuth.png", "Best C absolute error by orientation", cmap="magma_r", aggfunc="min")
    save_heatmap(summary_df, "nearest_D_error_A", args.outdir / "heatmap_D_error_by_normal_azimuth.png", "Mean signed D error by orientation", cmap="coolwarm")
    save_heatmap(summary_df, "D_abs_error_A", args.outdir / "heatmap_best_D_error_by_normal_azimuth.png", "Best D absolute error by orientation", cmap="magma_r", aggfunc="min")
    save_heatmap(summary_df, "nearest_C_intensity", args.outdir / "heatmap_C_intensity_by_normal_azimuth.png", "Mean C peak intensity by orientation")
    save_heatmap(summary_df, "nearest_D_intensity", args.outdir / "heatmap_D_intensity_by_normal_azimuth.png", "Mean D peak intensity by orientation")
    save_grouped_line(
        summary_df,
        "helix_radius_A",
        [("C_abs_error_A", "C"), ("D_abs_error_A", "D"), ("CD_combined_abs_error_A", "C+D")],
        args.outdir / "best_C_D_combined_error_by_radius.png",
        "Best C/D error by helix radius",
    )
    save_grouped_line(
        summary_df,
        "in_plane_spin_deg",
        [("CD_combined_abs_error_A", "C+D")],
        args.outdir / "best_combined_error_by_spin.png",
        "Best combined C/D error by in-plane spin",
    )
    if "uniform_adjacent_z_offset_A" in summary_df.columns:
        save_grouped_line(
            summary_df,
            "uniform_adjacent_z_offset_A",
            [("CD_combined_abs_error_A", "C+D")],
            args.outdir / "best_combined_error_by_z_offset.png",
            "Best combined C/D error by adjacent-strand z-offset",
        )
        save_grouped_line(
            summary_df,
            "uniform_adjacent_z_offset_A",
            [("C_abs_error_A", "C")],
            args.outdir / "best_C_error_by_z_offset.png",
            "Best C error by adjacent-strand z-offset",
        )
        save_grouped_line(
            summary_df,
            "uniform_adjacent_z_offset_A",
            [("D_abs_error_A", "D")],
            args.outdir / "best_D_error_by_z_offset.png",
            "Best D error by adjacent-strand z-offset",
        )
        save_hit_counts_by_group(
            summary_df,
            "uniform_adjacent_z_offset_A",
            args.outdir / "C_and_D_hit_counts_by_z_offset.png",
            "C/D hit counts by adjacent-strand z-offset",
        )
        save_two_parameter_heatmap(
            summary_df,
            "helix_radius_A",
            "uniform_adjacent_z_offset_A",
            "CD_combined_abs_error_A",
            args.outdir / "heatmap_best_CD_error_by_z_offset_radius.png",
            "Best C+D error by z-offset and radius",
        )
    save_best_orientation_plot(best_df, args.outdir)
    write_report(args.outdir, summary_df, best_df, args)

    print(f"Analyzed {len(summary_df)} parametric models")
    print(f"Both C and D found: {int(summary_df['both_C_and_D_found'].sum())}")
    print(f"Best model: {best_df.iloc[0]['model_label']}")
    print(f"Output directory: {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
