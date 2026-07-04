from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from scripts.audit_emory_s1_02_fiber_xray_images import (
    SPECIFIED_FILES,
    anisotropy_classification,
    azimuthal_profile,
    build_report,
    classify_image_type,
    ignored_image_files,
    is_specified_filename,
    radial_profile,
    sector_summary,
    validate_required_files,
)


def write_png(path: Path, array: np.ndarray) -> None:
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(path)


def test_only_three_specified_filenames_are_accepted() -> None:
    assert all(is_specified_filename(name) for name in SPECIFIED_FILES)
    assert not is_specified_filename("extra_emory_image.png")


def test_additional_image_file_is_ignored(tmp_path: Path) -> None:
    for name in SPECIFIED_FILES:
        write_png(tmp_path / name, np.zeros((8, 8), dtype=np.uint8))
    write_png(tmp_path / "extra_emory_image.png", np.zeros((8, 8), dtype=np.uint8))

    ignored = ignored_image_files(tmp_path)
    assert [path.name for path in ignored] == ["extra_emory_image.png"]


def test_missing_specified_file_produces_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing required Emory S1-02 X-ray image"):
        validate_required_files(tmp_path)


def test_image_classification_from_filename() -> None:
    assert classify_image_type("HXC570-sample-only-5frame.tiff") == "sample_only"
    assert classify_image_type("HXC570-5frame-subtraction-preview.png") == "loop_subtraction_preview"
    assert classify_image_type("HXC570-2D-subtraction-1pnt084.png") == "loop_scaled_difference"


def test_synthetic_radial_profile_generation() -> None:
    yy, xx = np.indices((64, 64))
    rr = np.sqrt((xx - 31.5) ** 2 + (yy - 31.5) ** 2)
    image = np.exp(-((rr - 12.0) ** 2) / 4.0)
    profile = radial_profile(image, (31.5, 31.5))
    strongest = profile.loc[profile["mean_intensity"].idxmax(), "radius_px"]
    assert abs(strongest - 12.5) <= 2.0


def test_synthetic_arc_anisotropy_detection() -> None:
    yy, xx = np.indices((96, 96))
    center = (47.5, 47.5)
    rr = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    theta = (np.degrees(np.arctan2(yy - center[1], xx - center[0])) + 360) % 360
    ring = np.exp(-((rr - 20.0) ** 2) / 4.0)
    arc = ring * (np.exp(-((theta - 0.0) ** 2) / 100.0) + np.exp(-((theta - 180.0) ** 2) / 100.0))
    sectors = sector_summary(arc, center, 20.0)
    score, label = anisotropy_classification(sectors)
    assert score > 1.5
    assert label in {"arc_like_oriented", "mixed"}

    az = azimuthal_profile(arc, center, 20.0)
    assert not az.empty
    assert az["radius_px"].iloc[0] == 20.0


def test_missing_calibration_report_uses_pixel_only_outputs() -> None:
    inventory = pd.DataFrame(
        [
            {
                "filename": "HXC570-sample-only-5frame.tiff",
                "image_type": "sample_only",
                "width_px": 10,
                "height_px": 10,
                "mode": "I;16",
                "color_class": "greyscale",
                "quantitative_suitability": "qualitative_and_pixel_quantitative",
                "feature_classification": "arc_like_oriented",
                "anisotropy_score": 2.0,
            }
        ]
    )
    sectors = pd.DataFrame(
        [
            {
                "filename": "HXC570-sample-only-5frame.tiff",
                "sector": "horizontal",
                "target_radius_px": 20.0,
                "mean_intensity": 10.0,
                "pixel_count": 30,
            }
        ]
    )
    text = build_report(inventory, sectors, [])
    assert "pixel-radius" in text
    assert "No calibration metadata" in text


def test_report_wording_contains_required_cautions() -> None:
    inventory = pd.DataFrame(
        [
            {
                "filename": "HXC570-sample-only-5frame.tiff",
                "image_type": "sample_only",
                "width_px": 10,
                "height_px": 10,
                "mode": "I;16",
                "color_class": "greyscale",
                "quantitative_suitability": "qualitative_and_pixel_quantitative",
                "feature_classification": "arc_like_oriented",
                "anisotropy_score": 2.0,
            }
        ]
    )
    text = build_report(inventory, pd.DataFrame(), [])
    for phrase in [
        "Sample S1-02",
        "fiber-like",
        "preferred orientation",
        "discrete arcs rather than uniform powder rings",
        "provenance-limited",
        "not used as structural proof",
        "3.4 A",
        "3.0 A",
        "4.5 to 5 A",
        "nylon loop",
        "additional constraints beyond A/B/C/D",
    ]:
        assert phrase in text
