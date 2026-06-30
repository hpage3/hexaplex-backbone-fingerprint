from __future__ import annotations

import numpy as np
import pandas as pd

from hexaplex_backbone_fingerprint.parametric_powder_scan import (
    debye_profile,
    load_xyz_coordinates,
    make_q_grid,
    nearest_peak,
    rank_powder_summary,
)


def test_debye_profile_shape():
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    q_values = make_q_grid(d_min_A=3.0, d_max_A=6.0, q_step=0.1)

    profile = debye_profile(coords, q_values)

    assert list(profile.columns) == ["q_Ainv", "d_A", "intensity"]
    assert len(profile) == len(q_values)
    assert np.isfinite(profile["intensity"]).all()


def test_nearest_peak_finds_synthetic_peak():
    d_values = np.linspace(4.0, 8.0, 401)
    intensity = 1.0 + np.exp(-((d_values - 5.6) / 0.08) ** 2)
    profile = pd.DataFrame({"q_Ainv": 2.0 * np.pi / d_values, "d_A": d_values, "intensity": intensity})

    hit = nearest_peak(profile, target_d_A=5.6, tolerance_A=0.05)

    assert hit.found_within_tolerance
    assert abs(hit.peak_d_A - 5.6) < 0.02


def test_load_xyz_coordinates(tmp_path):
    xyz = tmp_path / "mini.xyz"
    xyz.write_text("2\nmini\nC 0 0 0\nH 1 0 0\n", encoding="ascii")

    coords = load_xyz_coordinates(xyz, exclude_hydrogen=True)

    assert coords.shape == (1, 3)


def test_rank_powder_summary_prefers_both_hits_then_error():
    summary = pd.DataFrame(
        [
            {
                "model_label": "a",
                "both_C_and_D_found": False,
                "CD_combined_abs_error_A": 0.01,
                "nearest_C_intensity": 10,
                "nearest_D_intensity": 10,
            },
            {
                "model_label": "b",
                "both_C_and_D_found": True,
                "CD_combined_abs_error_A": 0.20,
                "nearest_C_intensity": 1,
                "nearest_D_intensity": 1,
            },
            {
                "model_label": "c",
                "both_C_and_D_found": True,
                "CD_combined_abs_error_A": 0.10,
                "nearest_C_intensity": 1,
                "nearest_D_intensity": 1,
            },
        ]
    )

    ranked = rank_powder_summary(summary)

    assert ranked.iloc[0]["model_label"] == "c"
