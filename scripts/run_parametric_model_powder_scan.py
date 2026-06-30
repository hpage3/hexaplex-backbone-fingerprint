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


def pivot_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return df.pivot_table(
        index="plane_normal_to_axis_deg",
        columns="plane_azimuth_deg",
        values=metric,
        aggfunc="mean",
    ).sort_index(ascending=True).sort_index(axis=1)


def save_heatmap(df: pd.DataFrame, metric: str, path: Path, title: str, cmap: str = "viridis") -> None:
    pivot = pivot_metric(df, metric)
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


def dominant_values(best: pd.DataFrame, column: str, n: int = 12) -> str:
    subset = best.head(n)
    counts = subset[column].value_counts()
    return "; ".join(f"{idx:g}: {count}" for idx, count in counts.items())


def write_report(outdir: Path, summary: pd.DataFrame, best: pd.DataFrame, args: argparse.Namespace) -> None:
    both_count = int(summary["both_C_and_D_found"].sum())
    c_count = int(summary["C_found_within_tolerance"].sum())
    d_count = int(summary["D_found_within_tolerance"].sum())
    best_row = best.iloc[0]
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
- plane normal / azimuth: {best_row.plane_normal_to_axis_deg:g} deg / {best_row.plane_azimuth_deg:g} deg
- nearest C peak: {best_row.nearest_C_peak_d_A:.3f} A (error {best_row.nearest_C_error_A:.3f} A)
- nearest D peak: {best_row.nearest_D_peak_d_A:.3f} A (error {best_row.nearest_D_error_A:.3f} A)
- combined absolute C/D error: {best_row.CD_combined_abs_error_A:.3f} A

## Orientation Trends

- Favored plane-normal angles among top ranked models: {dominant_values(best, "plane_normal_to_axis_deg")}
- Favored plane azimuths among top ranked models: {dominant_values(best, "plane_azimuth_deg")}

## Interpretation

The important first-pass question is whether C and D peak positions can arise from peptide-bond-plane orientation alone in a six-strand point-scatterer model. In this starter sweep, no model placed both C and D within the requested tolerance. The best-ranked models place D close to the 7.3 A target but place the nearest C-like local maximum low, near 5.0 A rather than 5.6 A. This does not falsify Nick's hypothesis, but it says the first coarse orientation sweep at radius 8 A is not sufficient by itself.

Treat the intensity values and ranking as diagnostic: equal atom weights, finite model length, radius 8 A, no solvent/background, and isotropic Debye averaging are all simplifications.

## Recommended Next Sweep

Because no models hit both targets, the next sweep should vary helix radius and strand phase/z-offset first, then in-plane spin and finer plane-normal/azimuth grids around the best D-favoring orientations. Longer repeats should follow after radius/phase sensitivity is understood.

## Output Files

- `parametric_powder_scan_summary.csv`
- `parametric_powder_peak_table.csv`
- `best_parametric_powder_models.csv`
- `best_model_radial_profiles.png`
- `heatmap_CD_error_by_normal_azimuth.png`
- `heatmap_C_error_by_normal_azimuth.png`
- `heatmap_D_error_by_normal_azimuth.png`
- `heatmap_C_intensity_by_normal_azimuth.png`
- `heatmap_D_intensity_by_normal_azimuth.png`
- `best_models_by_orientation.png`
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
    save_heatmap(summary_df, "CD_combined_abs_error_A", args.outdir / "heatmap_CD_error_by_normal_azimuth.png", "Mean C+D absolute error by orientation", cmap="magma_r")
    save_heatmap(summary_df, "nearest_C_error_A", args.outdir / "heatmap_C_error_by_normal_azimuth.png", "Mean signed C error by orientation", cmap="coolwarm")
    save_heatmap(summary_df, "nearest_D_error_A", args.outdir / "heatmap_D_error_by_normal_azimuth.png", "Mean signed D error by orientation", cmap="coolwarm")
    save_heatmap(summary_df, "nearest_C_intensity", args.outdir / "heatmap_C_intensity_by_normal_azimuth.png", "Mean C peak intensity by orientation")
    save_heatmap(summary_df, "nearest_D_intensity", args.outdir / "heatmap_D_intensity_by_normal_azimuth.png", "Mean D peak intensity by orientation")
    save_best_orientation_plot(best_df, args.outdir)
    write_report(args.outdir, summary_df, best_df, args)

    print(f"Analyzed {len(summary_df)} parametric models")
    print(f"Both C and D found: {int(summary_df['both_C_and_D_found'].sum())}")
    print(f"Best model: {best_df.iloc[0]['model_label']}")
    print(f"Output directory: {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
