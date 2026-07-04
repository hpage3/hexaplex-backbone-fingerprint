"""Extract exploratory orientation fingerprints from Emory S1-02 X-ray images.

This second-stage analysis is intentionally conservative. It only reads the
three specified Emory image files, treats the TIFF in pixel-radius units, and
keeps rendered PNG subtraction previews as preview-only unless a future robust
panel crop is implemented.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_emory_s1_02_fiber_xray_images import (
    DEFAULT_INPUT_DIR,
    FIG_AZIMUTHAL,
    SPECIFIED_FILES,
    azimuthal_profile,
    ignored_image_files,
    infer_mask,
    load_image,
    provisional_center,
    radial_profile,
    sector_summary,
    specified_image_paths,
    strongest_radial_feature,
    validate_required_files,
)


OUT_FINGERPRINT = Path("outputs/metrics/emory_s1_02_fiber_orientation_fingerprint.csv")
OUT_ARCS = Path("outputs/metrics/emory_s1_02_fiber_arc_peaks.csv")
OUT_COMPARISON = Path("outputs/metrics/emory_s1_02_fiber_orientation_comparison.csv")
OUT_REPORT = Path("outputs/reports/emory_s1_02_fiber_orientation_fingerprint_report.md")
FIG_POLAR = Path("outputs/figures/emory_s1_02_fiber_orientation_polar_profile.png")
FIG_ARCS = Path("outputs/figures/emory_s1_02_fiber_arc_peak_overlay.png")
FIG_COMPARISON = Path("outputs/figures/emory_s1_02_fiber_orientation_comparison.png")


def is_allowed_filename(filename: str) -> bool:
    """Return whether the file is one of the three specified inputs."""
    return filename in SPECIFIED_FILES


def smooth_circular(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Return circular moving-average smoothed values."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    if window <= 1:
        return arr.copy()
    if window % 2 == 0:
        window += 1
    filled = arr.copy()
    if np.isnan(filled).any():
        median = np.nanmedian(filled)
        filled[np.isnan(filled)] = median if np.isfinite(median) else 0.0
    pad = window // 2
    padded = np.r_[filled[-pad:], filled, filled[:pad]]
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")


def detect_arc_peaks(azimuthal: pd.DataFrame, min_relative_height: float = 0.55) -> pd.DataFrame:
    """Detect local angular maxima/arcs in an azimuthal profile."""
    if azimuthal.empty:
        return pd.DataFrame(columns=["angle_deg", "relative_intensity", "arc_width_deg", "peak_intensity"])
    angles = pd.to_numeric(azimuthal["angle_deg"], errors="coerce").to_numpy(dtype=float)
    values = pd.to_numeric(azimuthal["mean_intensity"], errors="coerce").to_numpy(dtype=float)
    smoothed = smooth_circular(values, window=5)
    if smoothed.size == 0 or not np.isfinite(smoothed).any():
        return pd.DataFrame(columns=["angle_deg", "relative_intensity", "arc_width_deg", "peak_intensity"])
    vmax = float(np.nanmax(smoothed))
    vmin = float(np.nanmin(smoothed))
    dynamic = vmax - vmin
    if dynamic <= 0:
        return pd.DataFrame(columns=["angle_deg", "relative_intensity", "arc_width_deg", "peak_intensity"])
    threshold = vmin + min_relative_height * dynamic
    rows = []
    for idx, value in enumerate(smoothed):
        prev_value = smoothed[(idx - 1) % len(smoothed)]
        next_value = smoothed[(idx + 1) % len(smoothed)]
        if value >= threshold and value >= prev_value and value >= next_value:
            width = arc_width_deg(smoothed, idx, threshold, angle_step=float(np.nanmedian(np.diff(angles))) if len(angles) > 1 else 5.0)
            rows.append(
                {
                    "angle_deg": float(angles[idx] % 360.0),
                    "relative_intensity": float(value / vmax) if vmax else np.nan,
                    "arc_width_deg": width,
                    "peak_intensity": float(value),
                }
            )
    if not rows:
        idx = int(np.nanargmax(smoothed))
        rows.append(
            {
                "angle_deg": float(angles[idx] % 360.0),
                "relative_intensity": 1.0,
                "arc_width_deg": arc_width_deg(smoothed, idx, threshold, angle_step=float(np.nanmedian(np.diff(angles))) if len(angles) > 1 else 5.0),
                "peak_intensity": float(smoothed[idx]),
            }
        )
    return pd.DataFrame(rows).sort_values("relative_intensity", ascending=False).reset_index(drop=True)


def arc_width_deg(values: np.ndarray, peak_idx: int, threshold: float, angle_step: float) -> float:
    """Estimate peak width above threshold in degrees."""
    n = len(values)
    left = peak_idx
    while values[(left - 1) % n] >= threshold and (peak_idx - left) % n < n - 1:
        left = (left - 1) % n
    right = peak_idx
    while values[(right + 1) % n] >= threshold and (right - peak_idx) % n < n - 1:
        right = (right + 1) % n
    bins = ((right - left) % n) + 1
    return float(bins * angle_step)


def sector_means_from_summary(sectors: pd.DataFrame) -> dict[str, float]:
    """Return standard sector means from sector summary."""
    out = {
        "horizontal_sector_mean": np.nan,
        "vertical_sector_mean": np.nan,
        "diagonal_positive_sector_mean": np.nan,
        "diagonal_negative_sector_mean": np.nan,
    }
    mapping = {
        "horizontal": "horizontal_sector_mean",
        "vertical": "vertical_sector_mean",
        "diagonal_pos": "diagonal_positive_sector_mean",
        "diagonal_neg": "diagonal_negative_sector_mean",
    }
    for _, row in sectors.iterrows():
        key = mapping.get(str(row.get("sector", "")))
        if key:
            out[key] = float(row.get("mean_intensity", np.nan))
    return out


def anisotropy_ratio_from_azimuthal(azimuthal: pd.DataFrame) -> float:
    """Return max/mean azimuthal intensity ratio."""
    values = pd.to_numeric(azimuthal["mean_intensity"], errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) == 0 or np.nanmean(values) == 0:
        return float("nan")
    return float(np.nanmax(values) / np.nanmean(values))


def classify_orientation(anisotropy_ratio: float, number_of_arc_maxima: int) -> str:
    """Classify orientation fingerprint."""
    if not np.isfinite(anisotropy_ratio) or number_of_arc_maxima == 0:
        return "insufficient_quality"
    if anisotropy_ratio >= 2.0 and number_of_arc_maxima >= 1:
        return "fiber_oriented_arc_like"
    if anisotropy_ratio >= 1.35:
        return "mixed_orientation"
    return "weak_orientation"


def angular_difference(a: float, b: float) -> float:
    """Circular absolute angular difference in degrees."""
    return float(abs(((a - b + 180.0) % 360.0) - 180.0))


def opposite_angle(angle: float) -> float:
    """Return opposite azimuth angle."""
    return float((angle + 180.0) % 360.0)


def top_radial_features(profile: pd.DataFrame, count: int = 4, min_separation_px: float = 15.0) -> list[float]:
    """Return top separated radial feature radii in pixel units."""
    if profile.empty:
        return []
    sub = profile[pd.to_numeric(profile["radius_px"], errors="coerce") >= 10.0].copy()
    sub = sub[pd.to_numeric(sub["pixel_count"], errors="coerce") > 20]
    if sub.empty:
        return []
    values = pd.to_numeric(sub["mean_intensity"], errors="coerce").to_numpy(dtype=float)
    smoothed = smooth_circular(values, window=9)
    order = np.argsort(smoothed)[::-1]
    selected: list[float] = []
    radii = pd.to_numeric(sub["radius_px"], errors="coerce").to_numpy(dtype=float)
    for idx in order:
        radius = float(radii[idx])
        if all(abs(radius - prev) >= min_separation_px for prev in selected):
            selected.append(radius)
        if len(selected) >= count:
            break
    return sorted(selected)


def radii_to_probe(profile: pd.DataFrame) -> list[float]:
    """Return radii for azimuthal fingerprinting."""
    peak = strongest_radial_feature(profile)
    features = top_radial_features(profile, count=3)
    radii: list[float] = []
    if peak is not None:
        radii.extend([max(1.0, peak - 20.0), peak, peak + 20.0])
    radii.extend(features)
    deduped: list[float] = []
    for radius in radii:
        if all(abs(radius - existing) > 1.0 for existing in deduped):
            deduped.append(float(radius))
    return deduped


def analyze_tiff_record(record) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Analyze the sample-only TIFF orientation fingerprint."""
    center_x, center_y, center_method = provisional_center(record.array)
    mask = infer_mask(record.array)
    radial = radial_profile(record.array, (center_x, center_y), mask=mask)
    rows = []
    arc_rows = []
    comparison_rows = []
    for radius in radii_to_probe(radial):
        az = azimuthal_profile(record.array, (center_x, center_y), radius, mask=mask)
        az["smoothed_intensity"] = smooth_circular(pd.to_numeric(az["mean_intensity"], errors="coerce").to_numpy(dtype=float), window=5)
        arcs = detect_arc_peaks(az)
        sectors = sector_summary(record.array, (center_x, center_y), radius, mask=mask)
        sector_values = sector_means_from_summary(sectors)
        anisotropy = anisotropy_ratio_from_azimuthal(az)
        classification = classify_orientation(anisotropy, len(arcs))
        preferred_angle = float(arcs.iloc[0]["angle_deg"]) if not arcs.empty else np.nan
        opposite = opposite_angle(preferred_angle) if np.isfinite(preferred_angle) else np.nan
        width = float(arcs.iloc[0]["arc_width_deg"]) if not arcs.empty else np.nan
        row = {
            "filename": record.filename,
            "image_type": record.image_type,
            "analysis_status": "pixel_only_uncalibrated",
            "radius_px": radius,
            "beam_center_x_px": center_x,
            "beam_center_y_px": center_y,
            "beam_center_method": center_method,
            "preferred_angle_deg": preferred_angle,
            "opposite_angle_deg": opposite,
            "anisotropy_ratio": anisotropy,
            "number_of_arc_maxima": len(arcs),
            "arc_width_deg": width,
            "orientation_classification": classification,
            "calibration_status": "missing_calibration_pixel_radius_only",
            **sector_values,
        }
        rows.append(row)
        for _, arc in arcs.iterrows():
            arc_rows.append(
                {
                    "filename": record.filename,
                    "radius_px": radius,
                    "angle_deg": arc["angle_deg"],
                    "relative_intensity": arc["relative_intensity"],
                    "arc_width_deg": arc["arc_width_deg"],
                    "peak_intensity": arc["peak_intensity"],
                    "analysis_status": "pixel_only_uncalibrated",
                }
            )
        comparison_rows.append(
            {
                "filename": record.filename,
                "image_type": record.image_type,
                "radius_px": radius,
                "preferred_angle_deg": preferred_angle,
                "anisotropy_ratio": anisotropy,
                "orientation_classification": classification,
                "png_quantitative_status": "",
                "qualitative_agrees_with_tiff": "",
                "notes": "TIFF pixel-radius orientation fingerprint.",
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(arc_rows), pd.DataFrame(comparison_rows)


def preview_record_row(record, tiff_preferred_angle: float | None) -> dict[str, object]:
    """Return conservative preview PNG comparison row."""
    return {
        "filename": record.filename,
        "image_type": record.image_type,
        "radius_px": np.nan,
        "preferred_angle_deg": np.nan,
        "anisotropy_ratio": np.nan,
        "orientation_classification": "preview_only",
        "png_quantitative_status": "preview_only_no_robust_panel_crop",
        "qualitative_agrees_with_tiff": "not_quantitatively_tested",
        "notes": "Rendered PNG preview; Difference panel crop not robustly isolated, so not mixed into quantitative fingerprint.",
    }


def angle_stability(fingerprint: pd.DataFrame) -> dict[str, object]:
    """Summarize whether preferred angular sectors are stable across radii."""
    angles = pd.to_numeric(fingerprint["preferred_angle_deg"], errors="coerce").dropna().to_list()
    if len(angles) < 2:
        return {"stable_angle_count": len(angles), "max_preferred_angle_spread_deg": np.nan, "stable_across_nearby_radii": False}
    reference = angles[0]
    spreads = [angular_difference(angle, reference) for angle in angles]
    max_spread = float(max(spreads))
    return {
        "stable_angle_count": len(angles),
        "max_preferred_angle_spread_deg": max_spread,
        "stable_across_nearby_radii": max_spread <= 25.0,
    }


def plot_polar_profiles(fingerprint: pd.DataFrame, arc_peaks: pd.DataFrame, path: Path) -> None:
    """Plot polar arc peak positions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="polar")
    if not arc_peaks.empty:
        for radius, group in arc_peaks.groupby("radius_px"):
            theta = np.deg2rad(pd.to_numeric(group["angle_deg"], errors="coerce"))
            values = pd.to_numeric(group["relative_intensity"], errors="coerce")
            ax.scatter(theta, values, label=f"r={radius:.1f}px")
    ax.set_title("Emory S1-02 arc maxima (pixel radii)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15))
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_arc_overlay(fingerprint: pd.DataFrame, path: Path) -> None:
    """Plot preferred angles by radius."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    if not fingerprint.empty:
        ax.scatter(fingerprint["radius_px"], fingerprint["preferred_angle_deg"], label="preferred angle")
        ax.scatter(fingerprint["radius_px"], fingerprint["opposite_angle_deg"], label="opposite angle", marker="x")
    ax.set_xlabel("pixel radius")
    ax.set_ylabel("angle (deg)")
    ax.set_title("Preferred orientation angle by pixel radius")
    ax.set_ylim(0, 360)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_orientation_comparison(comparison: pd.DataFrame, path: Path) -> None:
    """Plot orientation classification / anisotropy summary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    sub = comparison[pd.to_numeric(comparison["anisotropy_ratio"], errors="coerce").notna()]
    if not sub.empty:
        ax.plot(sub["radius_px"], sub["anisotropy_ratio"], marker="o")
    ax.axhline(1.35, color="gray", linestyle="--", linewidth=1, label="mixed threshold")
    ax.axhline(2.0, color="black", linestyle=":", linewidth=1, label="arc-like threshold")
    ax.set_xlabel("pixel radius")
    ax.set_ylabel("anisotropy ratio")
    ax.set_title("Orientation anisotropy by pixel radius")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_report(fingerprint: pd.DataFrame, arcs: pd.DataFrame, comparison: pd.DataFrame, ignored: list[Path]) -> str:
    """Build conservative orientation fingerprint report."""
    stability = angle_stability(fingerprint)
    tiff = fingerprint[fingerprint["filename"].str.endswith(".tiff", na=False)]
    strongest = tiff.iloc[(pd.to_numeric(tiff["anisotropy_ratio"], errors="coerce")).idxmax()] if not tiff.empty else None
    ignored_text = "\n".join(f"- `{path.name}`" for path in ignored) if ignored else "- None."
    if strongest is not None:
        summary = (
            f"Strongest TIFF orientation fingerprint is at r={float(strongest['radius_px']):.1f} px, "
            f"preferred angle {float(strongest['preferred_angle_deg']):.1f} deg, "
            f"anisotropy ratio {float(strongest['anisotropy_ratio']):.3f}, "
            f"classified as `{strongest['orientation_classification']}`."
        )
    else:
        summary = "No quantitative TIFF orientation fingerprint was available."
    return f"""# Emory Sample S1-02 Fiber Orientation Fingerprint

## Scope

This is exploratory and provenance-limited. These images are fiber-like, not conventional powder patterns. pixel-radius features are not calibrated d-spacings, and this analysis is not structural proof.

John Bacsa's 3.4 A, 3.0 A, and 4.5 to 5 A features are external reference annotations, not derived by this script. Relative intensities remain approximate because the images were built from a small number of hand-picked frames. Nylon loop scattering/subtraction may affect low-angle features.

## Files

Analyzed only the three specified files:

- `HXC570-2D-subtraction-1pnt084.png`
- `HXC570-5frame-subtraction-preview.png`
- `HXC570-sample-only-5frame.tiff`

Ignored additional images:

{ignored_text}

## TIFF Orientation Fingerprint

{summary}

- Preferred angular sectors stable across nearby radii: `{stability['stable_across_nearby_radii']}`
- Max preferred-angle spread: `{stability['max_preferred_angle_spread_deg']}` deg
- Number of radii with preferred angles: `{stability['stable_angle_count']}`

## PNG Preview Handling

The PNG subtraction images are marked `preview_only_no_robust_panel_crop`. A robust Difference = sample only panel crop was not implemented here, so colorbars, titles, and margins are not mixed into quantitative analysis.

## Interpretation

- Do the images contain measurable angular anisotropy? The TIFF contains a measurable pixel-space anisotropy fingerprint.
- Does the TIFF show arc-like maxima around the strongest radial feature? The arc peak table reports angular maxima around the strongest and nearby pixel radii.
- Are preferred angular sectors stable across nearby radii? See the stability fields and comparison CSV.
- Do preview subtraction images qualitatively agree with the TIFF orientation pattern? They are preview-only in this pass; qualitative visual agreement should be assessed by eye against the generated summary, not used as a quantitative constraint.
- Could this provide additional constraints beyond A/B/C/D? Yes, if provenance and calibration are confirmed, the angular/orientation fingerprint could provide additional constraints beyond A/B/C/D radial band positions.
- What would be required to compare to models? Candidate models would need simulated 2D/fiber diffraction or oriented powder simulations, not just radial A/B/C/D scoring.

## Caveats

- Exploratory and provenance-limited.
- Fiber-like rather than conventional powder.
- Pixel-radius features are not calibrated d-spacings.
- Nylon loop subtraction/background may affect broad low-angle features.
- The analysis does not prove a structure.
"""


def run(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run orientation fingerprint analysis."""
    paths = validate_required_files(input_dir)
    ignored = ignored_image_files(input_dir)
    records = [load_image(path) for path in paths]
    fingerprint_frames = []
    arc_frames = []
    comparison_frames = []
    tiff_preferred = None
    for record in records:
        if record.image_type == "sample_only":
            fingerprint, arcs, comparison = analyze_tiff_record(record)
            fingerprint_frames.append(fingerprint)
            arc_frames.append(arcs)
            comparison_frames.append(comparison)
            if not fingerprint.empty:
                tiff_preferred = float(fingerprint.iloc[0]["preferred_angle_deg"])
        else:
            comparison_frames.append(pd.DataFrame([preview_record_row(record, tiff_preferred)]))
    fingerprint_df = pd.concat(fingerprint_frames, ignore_index=True) if fingerprint_frames else pd.DataFrame()
    arcs_df = pd.concat(arc_frames, ignore_index=True) if arc_frames else pd.DataFrame()
    comparison_df = pd.concat(comparison_frames, ignore_index=True) if comparison_frames else pd.DataFrame()

    for path in [OUT_FINGERPRINT, OUT_ARCS, OUT_COMPARISON]:
        path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint_df.to_csv(OUT_FINGERPRINT, index=False)
    arcs_df.to_csv(OUT_ARCS, index=False)
    comparison_df.to_csv(OUT_COMPARISON, index=False)
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text(build_report(fingerprint_df, arcs_df, comparison_df, ignored), encoding="utf-8")
    plot_polar_profiles(fingerprint_df, arcs_df, FIG_POLAR)
    plot_arc_overlay(fingerprint_df, FIG_ARCS)
    plot_orientation_comparison(comparison_df, FIG_COMPARISON)
    return fingerprint_df, arcs_df, comparison_df


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Analyze Emory S1-02 fiber orientation fingerprint.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    args = parser.parse_args()
    fingerprint, arcs, comparison = run(args.input_dir)
    print(f"Fingerprint rows: {len(fingerprint)}")
    print(f"Arc peak rows: {len(arcs)}")
    print(f"Comparison rows: {len(comparison)}")
    if not fingerprint.empty:
        best = fingerprint.sort_values("anisotropy_ratio", ascending=False).iloc[0]
        print(
            "Best TIFF orientation: "
            f"radius {float(best['radius_px']):.1f} px, "
            f"angle {float(best['preferred_angle_deg']):.1f} deg, "
            f"anisotropy {float(best['anisotropy_ratio']):.3f}, "
            f"class {best['orientation_classification']}"
        )
    print(f"Report: {OUT_REPORT}")


if __name__ == "__main__":
    main()
