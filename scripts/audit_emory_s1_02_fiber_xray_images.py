"""Audit the three Emory S1-02 fiber-like X-ray images.

This is an exploratory, provenance-limited image audit. The script only reads
the three explicitly named input files and reports pixel-radius/azimuthal
diagnostics unless calibration metadata are available. It does not modify raw
inputs and does not feed the images into the structural publication funnel.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageSequence


DEFAULT_INPUT_DIR = Path(
    r"C:\Users\hpage3\OneDrive - Georgia Institute of Technology\Documents\GitHub\fiber-diffraction-powder\inputs"
)
SPECIFIED_FILES = (
    "HXC570-2D-subtraction-1pnt084.png",
    "HXC570-5frame-subtraction-preview.png",
    "HXC570-sample-only-5frame.tiff",
)

OUT_INVENTORY = Path("outputs/metrics/emory_s1_02_fiber_xray_inventory.csv")
OUT_RADIAL = Path("outputs/metrics/emory_s1_02_fiber_xray_radial_profiles.csv")
OUT_AZIMUTHAL = Path("outputs/metrics/emory_s1_02_fiber_xray_azimuthal_profiles.csv")
OUT_SECTOR = Path("outputs/metrics/emory_s1_02_fiber_xray_sector_summary.csv")
OUT_REPORT = Path("outputs/reports/emory_s1_02_fiber_xray_audit_report.md")
FIG_RADIAL = Path("outputs/figures/emory_s1_02_fiber_xray_radial_profiles.png")
FIG_AZIMUTHAL = Path("outputs/figures/emory_s1_02_fiber_xray_azimuthal_profiles.png")
FIG_SECTOR = Path("outputs/figures/emory_s1_02_fiber_xray_sector_summary.png")


@dataclass(frozen=True)
class ImageRecord:
    """Loaded image data and metadata."""

    path: Path
    filename: str
    image_type: str
    array: np.ndarray
    mode: str
    dtype: str
    frame_count: int


def specified_image_paths(input_dir: Path = DEFAULT_INPUT_DIR) -> list[Path]:
    """Return exactly the three specified image paths."""
    return [input_dir / name for name in SPECIFIED_FILES]


def ignored_image_files(input_dir: Path = DEFAULT_INPUT_DIR) -> list[Path]:
    """Return image files in input_dir that are intentionally ignored."""
    image_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    specified = {name.lower() for name in SPECIFIED_FILES}
    if not input_dir.exists():
        return []
    return sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in image_exts and path.name.lower() not in specified)


def validate_required_files(input_dir: Path = DEFAULT_INPUT_DIR) -> list[Path]:
    """Validate that all three specified files exist, or raise a clear error."""
    paths = specified_image_paths(input_dir)
    missing = [path for path in paths if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing required Emory S1-02 X-ray image file(s):\n{missing_text}")
    return paths


def classify_image_type(filename: str) -> str:
    """Classify image type from the specified filename."""
    value = filename.lower()
    if "sample-only" in value:
        return "sample_only"
    if "5frame-subtraction-preview" in value:
        return "loop_subtraction_preview"
    if "2d-subtraction" in value:
        return "loop_scaled_difference"
    return "unknown"


def is_specified_filename(filename: str) -> bool:
    """Return whether filename is one of the three allowed inputs."""
    return filename in SPECIFIED_FILES


def load_image(path: Path) -> ImageRecord:
    """Load image as a floating grayscale array while preserving metadata."""
    with Image.open(path) as image:
        mode = image.mode
        frames = []
        for frame in ImageSequence.Iterator(image):
            frames.append(np.asarray(frame.convert("F"), dtype=float))
        if not frames:
            frames.append(np.asarray(image.convert("F"), dtype=float))
        array = np.mean(np.stack(frames, axis=0), axis=0)
        return ImageRecord(
            path=path,
            filename=path.name,
            image_type=classify_image_type(path.name),
            array=array,
            mode=mode,
            dtype=str(np.asarray(image).dtype),
            frame_count=len(frames),
        )


def image_color_class(mode: str, array: np.ndarray) -> str:
    """Classify source image color handling."""
    if mode in {"RGB", "RGBA", "P", "CMYK"}:
        return "RGB/RGBA_or_rendered_color"
    if array.ndim == 2:
        return "greyscale"
    return "unknown"


def provisional_center(array: np.ndarray) -> tuple[float, float, str]:
    """Return provisional beam center as image center."""
    height, width = array.shape[:2]
    return (width - 1) / 2.0, (height - 1) / 2.0, "image_center_provisional_no_calibration"


def infer_mask(array: np.ndarray) -> np.ndarray:
    """Infer a conservative detector mask from finite, non-saturated pixels."""
    finite = np.isfinite(array)
    if not finite.any():
        return finite
    lo, hi = np.nanpercentile(array[finite], [0.5, 99.8])
    return finite & (array > lo) & (array < hi)


def detect_beamstop_or_mask(array: np.ndarray, center: tuple[float, float], radius: float = 25.0) -> bool:
    """Infer whether a central blocked/beamstop region is visible."""
    x0, y0 = center
    yy, xx = np.indices(array.shape)
    rr = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)
    central = array[rr <= radius]
    annulus = array[(rr > radius * 2) & (rr <= radius * 4)]
    if len(central) == 0 or len(annulus) == 0:
        return False
    return float(np.nanmedian(central)) < 0.5 * float(np.nanmedian(annulus)) or float(np.nanstd(central)) < 0.1 * float(np.nanstd(annulus))


def radial_profile(array: np.ndarray, center: tuple[float, float], mask: np.ndarray | None = None, bin_width: float = 1.0) -> pd.DataFrame:
    """Compute masked radial profile in pixel units."""
    x0, y0 = center
    yy, xx = np.indices(array.shape)
    rr = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)
    if mask is None:
        mask = np.isfinite(array)
    valid = mask & np.isfinite(array)
    bins = np.floor(rr[valid] / bin_width).astype(int)
    values = array[valid]
    if len(values) == 0:
        return pd.DataFrame(columns=["radius_px", "mean_intensity", "pixel_count"])
    sums = np.bincount(bins, weights=values)
    counts = np.bincount(bins)
    radii = (np.arange(len(sums)) + 0.5) * bin_width
    means = np.divide(sums, counts, out=np.full_like(sums, np.nan, dtype=float), where=counts > 0)
    return pd.DataFrame({"radius_px": radii, "mean_intensity": means, "pixel_count": counts})


def strongest_radial_feature(profile: pd.DataFrame, min_radius: float = 10.0) -> float | None:
    """Return pixel radius of strongest smoothed radial feature."""
    if profile.empty:
        return None
    sub = profile[pd.to_numeric(profile["radius_px"], errors="coerce") >= min_radius].copy()
    sub = sub[pd.to_numeric(sub["pixel_count"], errors="coerce") > 20]
    if sub.empty:
        return None
    intensity = pd.to_numeric(sub["mean_intensity"], errors="coerce").to_numpy(dtype=float)
    if len(intensity) >= 7:
        kernel = np.ones(7) / 7
        smoothed = np.convolve(intensity, kernel, mode="same")
    else:
        smoothed = intensity
    idx = int(np.nanargmax(smoothed))
    return float(sub.iloc[idx]["radius_px"])


def azimuthal_profile(
    array: np.ndarray,
    center: tuple[float, float],
    target_radius_px: float,
    radius_half_width_px: float = 4.0,
    angle_bin_deg: float = 5.0,
    mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compute azimuthal profile around a target pixel radius."""
    x0, y0 = center
    yy, xx = np.indices(array.shape)
    rr = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)
    theta = (np.degrees(np.arctan2(yy - y0, xx - x0)) + 360.0) % 360.0
    if mask is None:
        mask = np.isfinite(array)
    valid = mask & np.isfinite(array) & (np.abs(rr - target_radius_px) <= radius_half_width_px)
    if not valid.any():
        return pd.DataFrame(columns=["angle_deg", "mean_intensity", "pixel_count", "radius_px"])
    bins = np.floor(theta[valid] / angle_bin_deg).astype(int)
    values = array[valid]
    sums = np.bincount(bins, weights=values, minlength=int(360 / angle_bin_deg))
    counts = np.bincount(bins, minlength=int(360 / angle_bin_deg))
    means = np.divide(sums, counts, out=np.full_like(sums, np.nan, dtype=float), where=counts > 0)
    angles = (np.arange(len(means)) + 0.5) * angle_bin_deg
    return pd.DataFrame({"angle_deg": angles, "mean_intensity": means, "pixel_count": counts, "radius_px": target_radius_px})


def sector_mask(theta_deg: np.ndarray, center_deg: float, half_width_deg: float = 15.0) -> np.ndarray:
    """Return angular sector mask with circular wraparound."""
    diff = np.abs(((theta_deg - center_deg + 180.0) % 360.0) - 180.0)
    return diff <= half_width_deg


def sector_summary(array: np.ndarray, center: tuple[float, float], target_radius_px: float, mask: np.ndarray | None = None) -> pd.DataFrame:
    """Compute sector averages around target radius."""
    x0, y0 = center
    yy, xx = np.indices(array.shape)
    rr = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)
    theta = (np.degrees(np.arctan2(yy - y0, xx - x0)) + 360.0) % 360.0
    if mask is None:
        mask = np.isfinite(array)
    annulus = mask & np.isfinite(array) & (np.abs(rr - target_radius_px) <= 5.0)
    sectors = {
        "horizontal": [0.0, 180.0],
        "vertical": [90.0, 270.0],
        "diagonal_pos": [45.0, 225.0],
        "diagonal_neg": [135.0, 315.0],
    }
    rows = []
    for name, centers in sectors.items():
        sec = np.zeros_like(annulus, dtype=bool)
        for angle in centers:
            sec |= sector_mask(theta, angle)
        values = array[annulus & sec]
        rows.append(
            {
                "sector": name,
                "target_radius_px": target_radius_px,
                "mean_intensity": float(np.nanmean(values)) if len(values) else np.nan,
                "pixel_count": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def anisotropy_classification(sectors: pd.DataFrame, flat_threshold: float = 1.25, arc_threshold: float = 1.75) -> tuple[float, str]:
    """Return anisotropy score and qualitative classification."""
    values = pd.to_numeric(sectors["mean_intensity"], errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) < 2 or np.nanmean(values) == 0:
        return float("nan"), "insufficient_quality"
    score = float(np.nanmax(values) / np.nanmean(values))
    if score >= arc_threshold:
        return score, "arc_like_oriented"
    if score <= flat_threshold:
        return score, "uniform_ring_like"
    return score, "mixed"


def png_preview_quantitative_status(record: ImageRecord) -> str:
    """Return conservative suitability for rendered PNG previews."""
    if record.path.suffix.lower() == ".png":
        return "preview_only_qualitative"
    return "qualitative_and_pixel_quantitative"


def inventory_row(record: ImageRecord, ignored_count: int = 0) -> dict[str, object]:
    """Build one image inventory row."""
    array = record.array
    center_x, center_y, center_method = provisional_center(array)
    mask = infer_mask(array)
    beamstop = detect_beamstop_or_mask(array, (center_x, center_y))
    color = image_color_class(record.mode, array)
    preview = png_preview_quantitative_status(record)
    return {
        "filename": record.filename,
        "path": str(record.path),
        "image_type": record.image_type,
        "width_px": int(array.shape[1]),
        "height_px": int(array.shape[0]),
        "mode": record.mode,
        "dtype": record.dtype,
        "frame_count": record.frame_count,
        "min_intensity": float(np.nanmin(array)),
        "max_intensity": float(np.nanmax(array)),
        "mean_intensity": float(np.nanmean(array)),
        "color_class": color,
        "beam_center_x_px": center_x,
        "beam_center_y_px": center_y,
        "beam_center_method": center_method,
        "beamstop_or_mask_inferred": bool(beamstop),
        "calibration_available": False,
        "quantitative_suitability": preview,
        "ignored_additional_image_count": ignored_count,
        "notes": "Rendered PNG preview; avoid colorbar/title/margin overinterpretation." if record.path.suffix.lower() == ".png" else "Sample-only TIFF analyzed in pixel-radius units.",
    }


def analyze_record(record: ImageRecord) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Analyze one image record."""
    center_x, center_y, _ = provisional_center(record.array)
    mask = infer_mask(record.array)
    inventory = inventory_row(record)
    if record.image_type == "sample_only":
        radial = radial_profile(record.array, (center_x, center_y), mask=mask)
        radius = strongest_radial_feature(radial)
        if radius is None:
            az = pd.DataFrame(columns=["angle_deg", "mean_intensity", "pixel_count", "radius_px"])
            sectors = pd.DataFrame(columns=["sector", "target_radius_px", "mean_intensity", "pixel_count"])
        else:
            az = azimuthal_profile(record.array, (center_x, center_y), radius, mask=mask)
            sectors = sector_summary(record.array, (center_x, center_y), radius, mask=mask)
            score, classification = anisotropy_classification(sectors)
            inventory["strongest_feature_radius_px"] = radius
            inventory["anisotropy_score"] = score
            inventory["feature_classification"] = classification
        radial["filename"] = record.filename
        az["filename"] = record.filename
        sectors["filename"] = record.filename
        return inventory, radial, az, sectors
    inventory["strongest_feature_radius_px"] = np.nan
    inventory["anisotropy_score"] = np.nan
    inventory["feature_classification"] = "preview_only"
    return (
        inventory,
        pd.DataFrame(columns=["radius_px", "mean_intensity", "pixel_count", "filename"]),
        pd.DataFrame(columns=["angle_deg", "mean_intensity", "pixel_count", "radius_px", "filename"]),
        pd.DataFrame(columns=["sector", "target_radius_px", "mean_intensity", "pixel_count", "filename"]),
    )


def plot_radial(radial: pd.DataFrame, path: Path) -> None:
    """Plot radial profiles."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    if not radial.empty:
        for filename, group in radial.groupby("filename"):
            ax.plot(group["radius_px"], group["mean_intensity"], label=filename)
    ax.set_xlabel("pixel radius (uncalibrated)")
    ax.set_ylabel("mean intensity")
    ax.set_title("Emory S1-02 sample-only radial profile")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_azimuthal(az: pd.DataFrame, path: Path) -> None:
    """Plot azimuthal profiles."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    if not az.empty:
        for filename, group in az.groupby("filename"):
            ax.plot(group["angle_deg"], group["mean_intensity"], label=filename)
    ax.set_xlabel("azimuth angle (deg)")
    ax.set_ylabel("mean intensity")
    ax.set_title("Azimuthal profile at strongest pixel-radius feature")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_sector(sector: pd.DataFrame, path: Path) -> None:
    """Plot sector summary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    if not sector.empty:
        sector = sector.copy()
        ax.bar(sector["sector"], sector["mean_intensity"])
    ax.set_ylabel("mean intensity")
    ax.set_title("Sector averages at strongest pixel-radius feature")
    fig.autofmt_xdate(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    """Return markdown table."""
    if df.empty:
        return "_None._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in df[columns].iterrows():
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def build_report(inventory: pd.DataFrame, sectors: pd.DataFrame, ignored: list[Path]) -> str:
    """Build conservative audit report."""
    sample = inventory[inventory["image_type"] == "sample_only"]
    classification = sample["feature_classification"].iloc[0] if not sample.empty and "feature_classification" in sample else "not_available"
    anisotropy = sample["anisotropy_score"].iloc[0] if not sample.empty and "anisotropy_score" in sample else np.nan
    ignored_text = "\n".join(f"- `{path.name}`" for path in ignored) if ignored else "- None."
    inv_cols = [
        "filename",
        "image_type",
        "width_px",
        "height_px",
        "mode",
        "color_class",
        "quantitative_suitability",
        "feature_classification",
    ]
    sector_text = markdown_table(sectors, ["filename", "sector", "target_radius_px", "mean_intensity", "pixel_count"]) if not sectors.empty else "_No quantitative sector profile was produced._"
    return f"""# Emory Sample S1-02 Fiber-Like X-ray Image Audit

## Scope

This limited audit analyzes only the three specified Emory X-ray image files for Sample S1-02. These images are fiber-like, not conventional powder patterns. Nick identified them as likely images from a fiber that Shibna made, not the expected powder pattern from Sajena's sample, so they are provenance-limited and not used as structural proof.

The observed arcs/preferred orientation may contain additional orientational information and additional constraints beyond A/B/C/D radial band positions, but only after provenance, calibration, and sample identity are confirmed.

## Images Analyzed

{markdown_table(inventory, inv_cols)}

## Files Explicitly Ignored

{ignored_text}

## Calibration And External Reference Peaks

No calibration metadata were available to this audit, so all image-derived radii are reported in pixels only. John Bacsa reported reliable peak positions at 3.4 A and 3.0 A, with a broad approximate 4.5 to 5 A feature. These values are external references from John's email, not calibrated values derived here.

Relative intensities are approximate because the images were built from a small number of hand-picked frames. The nylon loop contribution is a concern, especially near the broad 4.5 to 5 A feature.

## Fiber / Arc Assessment

- The PNG files are treated as rendered preview/subtraction images, not raw quantitative detector arrays.
- The TIFF sample-only image was analyzed in pixel-radius units.
- TIFF anisotropy classification: `{classification}`.
- TIFF anisotropy score: `{anisotropy}`.

John reported that the reflections show very strong preferred orientation and discrete arcs rather than uniform powder rings, which is consistent with a fiber-like sample. The automated sector score here is a simple diagnostic, not a calibrated fiber-diffraction analysis.

## Sector Summary

{sector_text}

## Interpretation

- What are the three images? One sample-only TIFF and two rendered PNG subtraction/difference previews.
- Which are raw-ish data and which are previews? The TIFF is treated as raw-ish sample-only image data; the PNGs are treated as preview_only unless robust panel cropping and metadata are provided.
- Do the images show oriented/arced reflections rather than uniform powder rings? Based on John's notes and the image context, yes, they are expected to show preferred orientation and arcs; the TIFF sector analysis tests this in pixel space.
- Are the 3.4 A, 3.0 A, and 4.5 to 5 A features visible or plausibly represented? They are plausible external-reference features from John's integrated 1D pattern, but this audit does not derive calibrated d-spacings from the images.
- Could these images add structural constraints beyond A/B/C/D? Yes, if provenance and calibration are confirmed, angular/orientation constraints could complement the powder A/B/C/D constraints.
- What metadata are needed? Detector geometry, beam center, pixel size, wavelength, distance, frame list, background/loop scaling, masks, exact sample identity, and original raw detector frames.
- Manuscript status: `provenance_limited_possible_constraint`, not `publication_ready_constraint`.
"""


def run(input_dir: Path = DEFAULT_INPUT_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the limited Emory S1-02 image audit."""
    paths = validate_required_files(input_dir)
    ignored = ignored_image_files(input_dir)
    records = [load_image(path) for path in paths]
    rows = []
    radial_frames = []
    az_frames = []
    sector_frames = []
    for record in records:
        row, radial, az, sectors = analyze_record(record)
        row["ignored_additional_image_count"] = len(ignored)
        rows.append(row)
        radial_frames.append(radial)
        az_frames.append(az)
        sector_frames.append(sectors)
    inventory = pd.DataFrame(rows)
    radial = pd.concat(radial_frames, ignore_index=True) if radial_frames else pd.DataFrame()
    az = pd.concat(az_frames, ignore_index=True) if az_frames else pd.DataFrame()
    sectors = pd.concat(sector_frames, ignore_index=True) if sector_frames else pd.DataFrame()

    for path in [OUT_INVENTORY, OUT_RADIAL, OUT_AZIMUTHAL, OUT_SECTOR]:
        path.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(OUT_INVENTORY, index=False)
    radial.to_csv(OUT_RADIAL, index=False)
    az.to_csv(OUT_AZIMUTHAL, index=False)
    sectors.to_csv(OUT_SECTOR, index=False)
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text(build_report(inventory, sectors, ignored), encoding="utf-8")
    plot_radial(radial, FIG_RADIAL)
    plot_azimuthal(az, FIG_AZIMUTHAL)
    plot_sector(sectors, FIG_SECTOR)
    return inventory, radial, az, sectors


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Audit the three Emory S1-02 fiber-like X-ray images.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    args = parser.parse_args()
    inventory, radial, az, sectors = run(args.input_dir)
    print(f"Files analyzed: {len(inventory)}")
    print(", ".join(inventory["filename"].tolist()))
    print(f"Radial profile rows: {len(radial)}")
    print(f"Azimuthal profile rows: {len(az)}")
    print(f"Sector rows: {len(sectors)}")
    print(f"Report: {OUT_REPORT}")


if __name__ == "__main__":
    main()
