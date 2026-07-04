from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from scripts.analyze_emory_s1_02_fiber_orientation_fingerprint import (
    build_report,
    classify_orientation,
    detect_arc_peaks,
    is_allowed_filename,
    preview_record_row,
    smooth_circular,
)
from scripts.audit_emory_s1_02_fiber_xray_images import (
    SPECIFIED_FILES,
    ImageRecord,
    azimuthal_profile,
    ignored_image_files,
    validate_required_files,
)


def write_png(path: Path, array: np.ndarray) -> None:
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(path)


def synthetic_arc_image(size: int = 128, radius: float = 32.0) -> tuple[np.ndarray, tuple[float, float]]:
    center = ((size - 1) / 2.0, (size - 1) / 2.0)
    yy, xx = np.indices((size, size))
    rr = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    theta = (np.degrees(np.arctan2(yy - center[1], xx - center[0])) + 360.0) % 360.0
    ring = np.exp(-((rr - radius) ** 2) / 8.0)
    arcs = np.exp(-((theta - 40.0) ** 2) / 120.0) + np.exp(-((theta - 220.0) ** 2) / 120.0)
    return ring * arcs, center


def synthetic_uniform_ring(size: int = 128, radius: float = 32.0) -> tuple[np.ndarray, tuple[float, float]]:
    center = ((size - 1) / 2.0, (size - 1) / 2.0)
    yy, xx = np.indices((size, size))
    rr = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    return np.exp(-((rr - radius) ** 2) / 8.0), center


def test_only_three_specified_filenames_are_accepted_and_extra_ignored(tmp_path: Path) -> None:
    assert all(is_allowed_filename(name) for name in SPECIFIED_FILES)
    assert not is_allowed_filename("extra.png")

    for name in SPECIFIED_FILES:
        write_png(tmp_path / name, np.zeros((8, 8), dtype=np.uint8))
    write_png(tmp_path / "extra.png", np.zeros((8, 8), dtype=np.uint8))
    assert [path.name for path in ignored_image_files(tmp_path)] == ["extra.png"]


def test_missing_specified_file_raises_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing required Emory S1-02 X-ray image"):
        validate_required_files(tmp_path)


def test_synthetic_arc_image_produces_detected_angular_maxima() -> None:
    image, center = synthetic_arc_image()
    az = azimuthal_profile(image, center, 32.0, radius_half_width_px=4.0, angle_bin_deg=5.0)
    peaks = detect_arc_peaks(az, min_relative_height=0.45)

    assert len(peaks) >= 1
    assert any(min(abs(angle - 42.5), abs(angle - 222.5)) < 15 for angle in peaks["angle_deg"])


def test_uniform_ring_has_lower_anisotropy_than_arc() -> None:
    arc, center = synthetic_arc_image()
    ring, _ = synthetic_uniform_ring()
    arc_az = azimuthal_profile(arc, center, 32.0)
    ring_az = azimuthal_profile(ring, center, 32.0)

    arc_ratio = float(arc_az["mean_intensity"].max() / arc_az["mean_intensity"].mean())
    ring_ratio = float(ring_az["mean_intensity"].max() / ring_az["mean_intensity"].mean())
    assert arc_ratio > ring_ratio


def test_pixel_only_mode_and_preview_png_status() -> None:
    record = ImageRecord(
        path=Path("HXC570-2D-subtraction-1pnt084.png"),
        filename="HXC570-2D-subtraction-1pnt084.png",
        image_type="loop_scaled_difference",
        array=np.zeros((10, 10)),
        mode="RGB",
        dtype="uint8",
        frame_count=1,
    )
    row = preview_record_row(record, None)
    assert row["orientation_classification"] == "preview_only"
    assert row["png_quantitative_status"] == "preview_only_no_robust_panel_crop"


def test_orientation_classification_logic() -> None:
    assert classify_orientation(float("nan"), 0) == "insufficient_quality"
    assert classify_orientation(1.1, 1) == "weak_orientation"
    assert classify_orientation(1.5, 1) == "mixed_orientation"
    assert classify_orientation(2.4, 2) == "fiber_oriented_arc_like"


def test_smooth_circular_preserves_length() -> None:
    values = np.array([0.0, 1.0, 3.0, 1.0])
    smoothed = smooth_circular(values, window=3)
    assert len(smoothed) == len(values)


def test_report_wording_contains_required_cautions() -> None:
    fingerprint = pd.DataFrame(
        [
            {
                "filename": "HXC570-sample-only-5frame.tiff",
                "radius_px": 262.5,
                "preferred_angle_deg": 42.5,
                "opposite_angle_deg": 222.5,
                "anisotropy_ratio": 2.1,
                "number_of_arc_maxima": 2,
                "arc_width_deg": 25.0,
                "orientation_classification": "fiber_oriented_arc_like",
            }
        ]
    )
    arcs = pd.DataFrame(
        [
            {
                "filename": "HXC570-sample-only-5frame.tiff",
                "radius_px": 262.5,
                "angle_deg": 42.5,
                "relative_intensity": 1.0,
            }
        ]
    )
    comparison = pd.DataFrame()
    text = build_report(fingerprint, arcs, comparison, [])
    for phrase in [
        "exploratory",
        "provenance-limited",
        "fiber-like",
        "pixel-radius",
        "not calibrated d-spacings",
        "not structural proof",
        "3.4 A",
        "3.0 A",
        "4.5 to 5 A",
        "Nylon loop",
        "additional constraints beyond A/B/C/D",
        "simulated 2D/fiber diffraction",
    ]:
        assert phrase in text
