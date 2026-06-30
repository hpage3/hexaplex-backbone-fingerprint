"""Diagnostic Debye powder scan helpers for parametric point-atom models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .xyz_parser import parse_xyz


@dataclass(frozen=True)
class PeakHit:
    """Nearest radial-profile peak to a target d spacing."""

    peak_d_A: float
    error_A: float
    intensity: float
    found_within_tolerance: bool


def load_xyz_coordinates(path: str | Path, exclude_hydrogen: bool = False) -> np.ndarray:
    """Load XYZ coordinates into an ``(n, 3)`` array."""
    atoms = parse_xyz(path)
    rows = []
    for atom in atoms:
        if exclude_hydrogen and atom.element.upper() == "H":
            continue
        rows.append((atom.x, atom.y, atom.z))
    if not rows:
        raise ValueError(f"No atoms available after filtering {path}.")
    return np.asarray(rows, dtype=float)


def make_q_grid(d_min_A: float = 2.5, d_max_A: float = 12.0, q_step: float = 0.005) -> np.ndarray:
    """Return an ascending q grid spanning the requested d-spacing range."""
    if d_min_A <= 0 or d_max_A <= 0 or d_min_A >= d_max_A:
        raise ValueError("Require 0 < d_min_A < d_max_A.")
    if q_step <= 0:
        raise ValueError("q_step must be positive.")
    q_min = 2.0 * np.pi / d_max_A
    q_max = 2.0 * np.pi / d_min_A
    return np.arange(q_min, q_max + 0.5 * q_step, q_step)


def pair_distances(coords: np.ndarray) -> np.ndarray:
    """Return upper-triangle pair distances for coordinates."""
    arr = np.asarray(coords, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("Coordinates must have shape (n, 3).")
    if arr.shape[0] < 2:
        return np.array([], dtype=float)
    row, col = np.triu_indices(arr.shape[0], k=1)
    return np.linalg.norm(arr[row] - arr[col], axis=1)


def debye_intensity_from_pair_distances(distances: np.ndarray, q_values: np.ndarray, atom_count: int) -> np.ndarray:
    """Compute equal-weight isotropic Debye intensity for pair distances."""
    distances = np.asarray(distances, dtype=float)
    q_values = np.asarray(q_values, dtype=float)
    intensities = np.full_like(q_values, fill_value=float(atom_count), dtype=float)
    if len(distances) == 0:
        return intensities
    for idx, q_value in enumerate(q_values):
        qr = q_value * distances
        intensities[idx] += 2.0 * np.sum(np.sinc(qr / np.pi))
    return intensities


def debye_profile(coords: np.ndarray, q_values: np.ndarray) -> pd.DataFrame:
    """Compute a diagnostic point-scatterer Debye profile."""
    distances = pair_distances(coords)
    intensities = debye_intensity_from_pair_distances(distances, q_values, atom_count=len(coords))
    profile = pd.DataFrame({"q_Ainv": q_values, "d_A": 2.0 * np.pi / q_values, "intensity": intensities})
    return profile.sort_values("d_A").reset_index(drop=True)


def local_maxima(profile: pd.DataFrame) -> pd.DataFrame:
    """Return simple local maxima from a radial profile."""
    if len(profile) < 3:
        return profile.iloc[0:0].copy()
    intensity = profile["intensity"].to_numpy(float)
    mask = (intensity[1:-1] > intensity[:-2]) & (intensity[1:-1] >= intensity[2:])
    peak_indices = np.where(mask)[0] + 1
    return profile.iloc[peak_indices].copy().reset_index(drop=True)


def nearest_peak(profile: pd.DataFrame, target_d_A: float, tolerance_A: float, search_half_width_A: float = 1.0) -> PeakHit:
    """Find the nearest local maximum to a target d spacing."""
    peaks = local_maxima(profile)
    window = peaks[(peaks["d_A"] >= target_d_A - search_half_width_A) & (peaks["d_A"] <= target_d_A + search_half_width_A)]
    if window.empty:
        window = profile[
            (profile["d_A"] >= target_d_A - search_half_width_A)
            & (profile["d_A"] <= target_d_A + search_half_width_A)
        ].copy()
        if window.empty:
            window = profile.copy()
        chosen = window.sort_values("intensity", ascending=False).iloc[0]
    else:
        chosen = window.iloc[(window["d_A"] - target_d_A).abs().argsort().iloc[0]]
    peak_d = float(chosen["d_A"])
    error = peak_d - target_d_A
    return PeakHit(
        peak_d_A=peak_d,
        error_A=error,
        intensity=float(chosen["intensity"]),
        found_within_tolerance=abs(error) <= tolerance_A,
    )


def rank_powder_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Rank model summaries by C/D hit status, error, then intensity."""
    df = summary.copy()
    df["combined_CD_intensity"] = pd.to_numeric(df["nearest_C_intensity"], errors="coerce").fillna(0) + pd.to_numeric(
        df["nearest_D_intensity"], errors="coerce"
    ).fillna(0)
    df["both_C_and_D_found"] = df["both_C_and_D_found"].astype(bool)
    return df.sort_values(
        ["both_C_and_D_found", "CD_combined_abs_error_A", "combined_CD_intensity"],
        ascending=[False, True, False],
    ).reset_index(drop=True)
