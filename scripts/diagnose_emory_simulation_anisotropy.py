"""Diagnose why first-pass Emory model simulations are weakly anisotropic.

This is a diagnostic of the first-pass simulator, not structural proof. It
tests whether weak Emory S1-02 model-orientation scores are caused by averaging,
axis choice, grid/radius choices, simulator controls, or weak single-orientation
anisotropy in the candidate coordinates.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_emory_s1_02_fiber_xray_images import azimuthal_profile, radial_profile, sector_summary
from scripts.compare_candidate_models_to_emory_fiber_fingerprint import (
    CandidateRecord,
    angular_lobe_score,
    expected_candidate_records,
    fingerprint_intensity_map,
    inventory_dataframe,
    projected_fft_intensity,
    read_pdb_coordinates,
    rotate_x,
    rotate_z,
    score_fingerprint_row,
)


CONTROL_ARC_MODEL_ID = "synthetic_arc_control"
CONTROL_RING_MODEL_ID = "synthetic_uniform_ring_control"
CONTROL_LATTICE_MODEL_ID = "synthetic_oriented_lattice_control"

DEFAULT_CANDIDATE_IDS = [
    "omega_clean_scale_0p9825",
    "omega_clean_scale_0p9775",
    "omega_clean_scale_0p9725",
    "omega_clean_scale_1p0000",
    "omega_clean_scale_0p9700",
    "guarded_full_chain_prototype",
]


def classify_simulated_orientation(anisotropy_ratio: float, number_of_arc_maxima: int) -> str:
    """Classify diagnostic simulated orientation fingerprints.

    The isotropic/weak boundary is slightly conservative to avoid treating
    square-grid pixelization of a synthetic uniform ring as meaningful
    orientation.
    """
    if not np.isfinite(anisotropy_ratio) or number_of_arc_maxima == 0:
        return "insufficient_quality"
    if number_of_arc_maxima >= 8 and anisotropy_ratio < 1.8:
        return "isotropic_ring_like"
    if anisotropy_ratio >= 2.2 and number_of_arc_maxima >= 1:
        return "fiber_oriented_arc_like"
    if anisotropy_ratio >= 1.45:
        return "mixed_orientation"
    if anisotropy_ratio >= 1.25:
        return "weak_orientation"
    return "isotropic_ring_like"

OUT_INVENTORY = Path("outputs/metrics/emory_simulation_anisotropy_candidate_inventory.csv")
OUT_AXIS = Path("outputs/metrics/emory_simulation_axis_sensitivity.csv")
OUT_LADDER = Path("outputs/metrics/emory_simulation_averaging_ladder.csv")
OUT_GRID = Path("outputs/metrics/emory_simulation_grid_sensitivity.csv")
OUT_RADIUS = Path("outputs/metrics/emory_simulation_radius_sensitivity.csv")
OUT_REPORT = Path("outputs/reports/emory_simulation_anisotropy_diagnostic_report.md")
FIG_AXIS = Path("outputs/figures/emory_simulation_axis_sensitivity.png")
FIG_LADDER = Path("outputs/figures/emory_simulation_averaging_ladder.png")
FIG_GRID = Path("outputs/figures/emory_simulation_grid_sensitivity.png")
FIG_MAPS = Path("outputs/figures/emory_simulation_example_maps.png")


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Return unit vector, or raise for a zero vector."""
    arr = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize zero vector.")
    return arr / norm


def rotation_matrix_from_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return rotation matrix that maps source vector onto target vector."""
    a = normalize_vector(source)
    b = normalize_vector(target)
    cross = np.cross(a, b)
    dot = float(np.dot(a, b))
    if np.linalg.norm(cross) <= 1e-12:
        if dot > 0:
            return np.eye(3)
        # 180-degree rotation around any perpendicular axis.
        axis = normalize_vector(np.cross(a, np.array([1.0, 0.0, 0.0])) if abs(a[0]) < 0.9 else np.cross(a, np.array([0.0, 1.0, 0.0])))
        return rotation_matrix_axis_angle(axis, 180.0)
    vx = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    s = float(np.linalg.norm(cross))
    return np.eye(3) + vx + vx @ vx * ((1.0 - dot) / (s * s))


def rotation_matrix_axis_angle(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Return axis-angle rotation matrix."""
    axis = normalize_vector(axis)
    angle = math.radians(angle_deg)
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    one = 1.0 - c
    return np.array(
        [
            [c + x * x * one, x * y * one - z * s, x * z * one + y * s],
            [y * x * one + z * s, c + y * y * one, y * z * one - x * s],
            [z * x * one - y * s, z * y * one + x * s, c + z * z * one],
        ],
        dtype=float,
    )


def principal_axes(coords: np.ndarray) -> dict[str, np.ndarray]:
    """Return principal component axes ordered by coordinate variance."""
    centered = coords - np.mean(coords, axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return {
        "principal_component_1": vt[0],
        "principal_component_2": vt[1],
        "principal_component_3": vt[2],
    }


def axis_vectors(coords: np.ndarray, include_helical: bool = True) -> dict[str, np.ndarray]:
    """Return candidate fiber-axis vectors."""
    axes = {
        "x_axis": np.array([1.0, 0.0, 0.0]),
        "y_axis": np.array([0.0, 1.0, 0.0]),
        "z_axis": np.array([0.0, 0.0, 1.0]),
        **principal_axes(coords),
    }
    if include_helical:
        axes["helical_axis_if_existing_utility_available"] = np.array([0.0, 0.0, 1.0])
    return axes


def orient_axis_to_z(coords: np.ndarray, axis_vector: np.ndarray) -> np.ndarray:
    """Center coordinates and rotate assumed fiber axis onto z."""
    centered = coords - np.mean(coords, axis=0)
    rot = rotation_matrix_from_vectors(axis_vector, np.array([0.0, 0.0, 1.0]))
    return centered @ rot.T


def simulate_axis_map(coords: np.ndarray, axis_vector: np.ndarray, grid_size: int) -> np.ndarray:
    """Simulate a single-orientation diagnostic map for one assumed axis."""
    oriented = orient_axis_to_z(coords, axis_vector)
    return projected_fft_intensity(rotate_x(oriented, 12.0), grid_size=grid_size)


def simulate_averaged_map(coords: np.ndarray, axis_vector: np.ndarray, grid_size: int, rotations: int = 1, tilt_deg: float = 12.0) -> np.ndarray:
    """Simulate an averaged diagnostic map around one assumed fiber axis."""
    oriented = orient_axis_to_z(coords, axis_vector)
    maps = []
    for idx in range(max(1, rotations)):
        angle = idx * 360.0 / max(1, rotations)
        maps.append(projected_fft_intensity(rotate_x(rotate_z(oriented, angle), tilt_deg), grid_size=grid_size))
    return normalize_image(np.mean(maps, axis=0))


def normalize_image(array: np.ndarray) -> np.ndarray:
    """Normalize an image to 0..1."""
    arr = np.asarray(array, dtype=float)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if hi <= lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def synthetic_arc_map(size: int = 128, radius: float = 34.0, angle_deg: float = 40.0) -> np.ndarray:
    """Return a synthetic two-lobe arc map."""
    center = ((size - 1) / 2.0, (size - 1) / 2.0)
    yy, xx = np.indices((size, size))
    rr = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    theta = (np.degrees(np.arctan2(yy - center[1], xx - center[0])) + 360.0) % 360.0
    ring = np.exp(-((rr - radius) ** 2) / 10.0)
    diff_a = np.abs(((theta - angle_deg + 180.0) % 360.0) - 180.0)
    diff_b = np.abs(((theta - (angle_deg + 180.0) + 180.0) % 360.0) - 180.0)
    return ring * (np.exp(-(diff_a**2) / 90.0) + np.exp(-(diff_b**2) / 90.0))


def synthetic_uniform_ring_map(size: int = 128, radius: float = 34.0) -> np.ndarray:
    """Return a synthetic isotropic ring map."""
    center = ((size - 1) / 2.0, (size - 1) / 2.0)
    yy, xx = np.indices((size, size))
    rr = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    return np.exp(-((rr - radius) ** 2) / 10.0)


def synthetic_oriented_lattice_coords(repeats: int = 18) -> np.ndarray:
    """Return an oriented toy coordinate set for averaging sanity checks."""
    coords = []
    for idx in range(repeats):
        z = idx * 1.2
        coords.extend(
            [
                [-4.0, 0.0, z],
                [4.0, 0.0, z],
                [-4.0, 1.1, z + 0.3],
                [4.0, -1.1, z + 0.3],
            ]
        )
    return np.asarray(coords, dtype=float)


def fingerprint_summary(model_id: str, mode: str, image: np.ndarray, grid_size: int, radius: float | None = None) -> dict[str, object]:
    """Return a compact fingerprint row for one image."""
    if radius is None:
        fp, _ = fingerprint_intensity_map(model_id, mode, image)
        if fp.empty:
            return empty_fingerprint(model_id, mode)
        row = fp.sort_values("anisotropy_ratio", ascending=False).iloc[0].to_dict()
        row["orientation_classification"] = classify_simulated_orientation(
            float(row.get("anisotropy_ratio", np.nan)),
            int(row.get("number_of_arc_maxima", 0)),
        )
    else:
        center = ((image.shape[1] - 1) / 2.0, (image.shape[0] - 1) / 2.0)
        az = azimuthal_profile(image, center, radius, radius_half_width_px=3.0, angle_bin_deg=5.0, mask=np.isfinite(image))
        values = pd.to_numeric(az["mean_intensity"], errors="coerce")
        anisotropy = float(values.max() / values.mean()) if len(values.dropna()) and values.mean() else float("nan")
        from scripts.analyze_emory_s1_02_fiber_orientation_fingerprint import detect_arc_peaks, sector_means_from_summary

        arcs = detect_arc_peaks(az, min_relative_height=0.45)
        sectors = sector_means_from_summary(sector_summary(image, center, radius, mask=np.isfinite(image)))
        row = {
            "model_id": model_id,
            "simulation_mode": mode,
            "radius_px": radius,
            "preferred_angle_deg": float(arcs.iloc[0]["angle_deg"]) if not arcs.empty else float("nan"),
            "opposite_angle_deg": float(arcs.iloc[1]["angle_deg"]) if len(arcs) > 1 else float("nan"),
            "anisotropy_ratio": anisotropy,
            "number_of_arc_maxima": len(arcs),
            "arc_width_deg": float(arcs.iloc[0]["arc_width_deg"]) if not arcs.empty else float("nan"),
            "orientation_classification": classify_simulated_orientation(anisotropy, len(arcs)),
            **sectors,
        }
    scores = score_fingerprint_row(pd.Series(row), grid_size=grid_size, provenance_caveat="")
    return {**row, **scores}


def empty_fingerprint(model_id: str, mode: str) -> dict[str, object]:
    """Return an empty fingerprint row."""
    return {
        "model_id": model_id,
        "simulation_mode": mode,
        "radius_px": float("nan"),
        "preferred_angle_deg": float("nan"),
        "opposite_angle_deg": float("nan"),
        "anisotropy_ratio": float("nan"),
        "number_of_arc_maxima": 0,
        "arc_width_deg": float("nan"),
        "orientation_classification": "insufficient_quality",
    }


def controlled_candidate_records() -> list[CandidateRecord]:
    """Return only the small controlled default candidate set."""
    by_id = {record.model_id: record for record in expected_candidate_records()}
    return [by_id[model_id] for model_id in DEFAULT_CANDIDATE_IDS if model_id in by_id]


def candidate_inventory(records: list[CandidateRecord], candidate_filter: str = "", max_candidates: int = 0) -> pd.DataFrame:
    """Return candidate inventory with atom counts and assumed-axis status."""
    inventory = inventory_dataframe(records)
    if candidate_filter:
        needle = candidate_filter.lower()
        inventory = inventory[
            inventory["model_id"].str.lower().str.contains(needle)
            | inventory["inferred_family"].str.lower().str.contains(needle)
        ]
    found_indices = inventory.index[inventory["status"] == "found"].tolist()
    if max_candidates > 0:
        keep_found = set(found_indices[:max_candidates])
        inventory = inventory[(inventory["status"] != "found") | inventory.index.isin(keep_found)]
    rows = []
    for _, row in inventory.iterrows():
        out = row.to_dict()
        if row["status"] == "found" and row["path"]:
            coords = read_pdb_coordinates(Path(row["path"]), exclude_hydrogen=True)
            out["coordinate_atom_count"] = len(coords)
            out["assumed_helical_fiber_axis"] = "z_axis"
            out["axis_status"] = "defaulted_and_principal_components_tested"
        else:
            out["coordinate_atom_count"] = 0
            out["assumed_helical_fiber_axis"] = ""
            out["axis_status"] = "unavailable"
        rows.append(out)
    return pd.DataFrame(rows).reset_index(drop=True)


def axis_sensitivity_rows(inventory: pd.DataFrame, grid_size: int) -> tuple[pd.DataFrame, dict[tuple[str, str], np.ndarray]]:
    """Run axis sensitivity simulation rows."""
    rows = []
    maps: dict[tuple[str, str], np.ndarray] = {}
    for item in inventory[inventory["status"] == "found"].itertuples(index=False):
        coords = read_pdb_coordinates(Path(item.path), exclude_hydrogen=True)
        for axis_name, axis_vector in axis_vectors(coords).items():
            image = simulate_axis_map(coords, axis_vector, grid_size)
            maps[(str(item.model_id), axis_name)] = image
            summary = fingerprint_summary(str(item.model_id), "single_orientation", image, grid_size)
            rows.append(
                {
                    **summary,
                    "axis_mode": axis_name,
                    "coordinate_atom_count": int(item.coordinate_atom_count),
                    "axis_status": item.axis_status,
                }
            )
    return pd.DataFrame(rows), maps


def ladder_levels(orientation_samples: int) -> list[tuple[str, int, float]]:
    """Return averaging ladder levels."""
    max_samples = max(1, int(orientation_samples))
    base = [
        ("single_orientation", 1, 12.0),
        ("rotate_about_fiber_axis_2", 2, 12.0),
        ("rotate_about_fiber_axis_4", 4, 12.0),
        ("rotate_about_fiber_axis_8", 8, 12.0),
        ("rotate_about_fiber_axis_16", 16, 12.0),
        ("tilt_plus_rotate_small", min(8, max_samples), 6.0),
        ("tilt_plus_rotate_medium", min(8, max_samples), 18.0),
        ("full_or_near_powder_average_if_available", max_samples, 30.0),
    ]
    return [(name, min(rotations, max_samples) if name != "single_orientation" else 1, tilt) for name, rotations, tilt in base]


def averaging_ladder_rows(inventory: pd.DataFrame, grid_size: int, orientation_samples: int, axis_mode: str) -> pd.DataFrame:
    """Run averaging ladder rows for selected axis."""
    rows = []
    for item in inventory[inventory["status"] == "found"].itertuples(index=False):
        coords = read_pdb_coordinates(Path(item.path), exclude_hydrogen=True)
        axes = axis_vectors(coords)
        axis_vector = axes.get(axis_mode, axes["z_axis"])
        previous_radius = None
        for level_name, rotations, tilt in ladder_levels(orientation_samples):
            image = simulate_averaged_map(coords, axis_vector, grid_size, rotations=rotations, tilt_deg=tilt)
            summary = fingerprint_summary(str(item.model_id), level_name, image, grid_size)
            radial_shift = abs(float(summary["radius_px"]) - previous_radius) if previous_radius is not None and np.isfinite(float(summary["radius_px"])) else 0.0
            previous_radius = float(summary["radius_px"]) if np.isfinite(float(summary["radius_px"])) else previous_radius
            rows.append(
                {
                    **summary,
                    "axis_mode": axis_mode,
                    "averaging_level": level_name,
                    "orientation_sample_count": rotations,
                    "tilt_deg": tilt,
                    "radial_feature_shift_from_previous_px": radial_shift,
                }
            )
    return pd.DataFrame(rows)


def grid_sensitivity_rows(inventory: pd.DataFrame, requested_grid_size: int, axis_mode: str) -> pd.DataFrame:
    """Run grid-size sensitivity rows."""
    grid_sizes = sorted(set([128, requested_grid_size] + ([512] if requested_grid_size >= 512 else [])))
    rows = []
    for item in inventory[inventory["status"] == "found"].head(3).itertuples(index=False):
        coords = read_pdb_coordinates(Path(item.path), exclude_hydrogen=True)
        axes = axis_vectors(coords)
        axis_vector = axes.get(axis_mode, axes["z_axis"])
        for size in grid_sizes:
            image = simulate_axis_map(coords, axis_vector, size)
            rows.append({**fingerprint_summary(str(item.model_id), "single_orientation", image, size), "axis_mode": axis_mode, "grid_size": size})
    return pd.DataFrame(rows)


def radius_sensitivity_rows(axis_df: pd.DataFrame, inventory: pd.DataFrame, grid_size: int, axis_mode: str) -> pd.DataFrame:
    """Run nearby-radius sensitivity rows around strongest simulated feature."""
    rows = []
    for item in inventory[inventory["status"] == "found"].itertuples(index=False):
        coords = read_pdb_coordinates(Path(item.path), exclude_hydrogen=True)
        axes = axis_vectors(coords)
        axis_vector = axes.get(axis_mode, axes["z_axis"])
        image = simulate_axis_map(coords, axis_vector, grid_size)
        best = axis_df[(axis_df["model_id"] == item.model_id) & (axis_df["axis_mode"] == axis_mode)]
        if best.empty:
            continue
        radius = float(best.sort_values("anisotropy_ratio", ascending=False).iloc[0]["radius_px"])
        for offset in [-20.0, 0.0, 20.0]:
            probe = max(2.0, radius + offset)
            rows.append({**fingerprint_summary(str(item.model_id), "radius_sensitivity", image, grid_size, radius=probe), "axis_mode": axis_mode, "radius_offset_px": offset})
    return pd.DataFrame(rows)


def control_rows(grid_size: int, orientation_samples: int) -> tuple[pd.DataFrame, bool]:
    """Run simulator sanity controls."""
    arc_summary = fingerprint_summary(CONTROL_ARC_MODEL_ID, "synthetic_control", synthetic_arc_map(size=grid_size), grid_size)
    ring_summary = fingerprint_summary(CONTROL_RING_MODEL_ID, "synthetic_control", synthetic_uniform_ring_map(size=grid_size), grid_size)
    coords = synthetic_oriented_lattice_coords()
    axis = np.array([0.0, 0.0, 1.0])
    single = fingerprint_summary(CONTROL_LATTICE_MODEL_ID, "single_orientation", simulate_averaged_map(coords, axis, grid_size, rotations=1), grid_size)
    averaged = fingerprint_summary(
        CONTROL_LATTICE_MODEL_ID,
        "rotate_about_fiber_axis_16",
        simulate_averaged_map(coords, axis, grid_size, rotations=max(2, min(16, orientation_samples))),
        grid_size,
    )
    controls = pd.DataFrame([arc_summary, ring_summary, single, averaged])
    controls_ok = (
        arc_summary["orientation_classification"] == "fiber_oriented_arc_like"
        and ring_summary["orientation_classification"] == "isotropic_ring_like"
        and float(single["anisotropy_ratio"]) >= float(averaged["anisotropy_ratio"])
    )
    return controls, controls_ok


def classify_failure_mode(axis_rows: pd.DataFrame, ladder_rows_df: pd.DataFrame, grid_rows_df: pd.DataFrame, radius_rows_df: pd.DataFrame, controls_ok: bool) -> str:
    """Conservatively classify anisotropy failure mode for one candidate."""
    if not controls_ok:
        return "simulator_control_failure"
    if axis_rows.empty:
        return "insufficient_data"
    axis_values = pd.to_numeric(axis_rows["anisotropy_ratio"], errors="coerce").dropna()
    if axis_values.empty:
        return "insufficient_data"
    max_axis = float(axis_values.max())
    min_axis = float(axis_values.min())
    if not ladder_rows_df.empty:
        single = ladder_rows_df[ladder_rows_df["averaging_level"] == "single_orientation"]
        strong_avg = ladder_rows_df[ladder_rows_df["averaging_level"].str.contains("8|16|powder", regex=True)]
        if not single.empty and not strong_avg.empty:
            single_max = float(pd.to_numeric(single["anisotropy_ratio"], errors="coerce").max())
            avg_min = float(pd.to_numeric(strong_avg["anisotropy_ratio"], errors="coerce").min())
            if single_max >= 3.0 and avg_min < 1.5:
                return "averaging_washes_out_arcs"
    if max_axis >= 3.0 and min_axis < 1.5:
        return "axis_choice_sensitive"
    if max_axis < 1.5:
        return "weak_even_single_orientation"
    if grid_sensitive(grid_rows_df):
        return "grid_resolution_sensitive"
    if radius_sensitive(radius_rows_df):
        return "radius_choice_sensitive"
    return "insufficient_data"


def grid_sensitive(rows: pd.DataFrame) -> bool:
    """Return whether grid-size choice changes anisotropy substantially."""
    if rows.empty:
        return False
    values = pd.to_numeric(rows["anisotropy_ratio"], errors="coerce").dropna()
    return len(values) > 1 and (float(values.max()) - float(values.min()) >= 1.0)


def radius_sensitive(rows: pd.DataFrame) -> bool:
    """Return whether nearby radius choice changes anisotropy substantially."""
    if rows.empty:
        return False
    values = pd.to_numeric(rows["anisotropy_ratio"], errors="coerce").dropna()
    return len(values) > 1 and (float(values.max()) - float(values.min()) >= 1.0)


def failure_mode_summary(axis_df: pd.DataFrame, ladder_df: pd.DataFrame, grid_df: pd.DataFrame, radius_df: pd.DataFrame, controls_ok: bool) -> pd.DataFrame:
    """Return one failure-mode row per candidate."""
    rows = []
    for model_id in sorted(set(axis_df["model_id"]) if not axis_df.empty else []):
        axis_rows_model = axis_df[axis_df["model_id"] == model_id]
        ladder_rows_model = ladder_df[ladder_df["model_id"] == model_id] if not ladder_df.empty else pd.DataFrame()
        grid_rows_model = grid_df[grid_df["model_id"] == model_id] if not grid_df.empty else pd.DataFrame()
        radius_rows_model = radius_df[radius_df["model_id"] == model_id] if not radius_df.empty else pd.DataFrame()
        rows.append(
            {
                "model_id": model_id,
                "anisotropy_failure_mode": classify_failure_mode(axis_rows_model, ladder_rows_model, grid_rows_model, radius_rows_model, controls_ok),
                "max_single_orientation_anisotropy": float(pd.to_numeric(axis_rows_model["anisotropy_ratio"], errors="coerce").max()),
                "min_single_orientation_anisotropy": float(pd.to_numeric(axis_rows_model["anisotropy_ratio"], errors="coerce").min()),
            }
        )
    return pd.DataFrame(rows)


def save_simple_plot(df: pd.DataFrame, path: Path, x: str, y: str, group: str, title: str) -> None:
    """Save a simple diagnostic line/scatter plot."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    if not df.empty and x in df and y in df:
        for key, sub in df.groupby(group):
            ax.plot(sub[x].astype(str), pd.to_numeric(sub[y], errors="coerce"), marker="o", label=str(key)[:28])
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=7)
    ax.axhspan(3.03, 4.80, color="#f58518", alpha=0.12, label="Emory anisotropy range")
    ax.set_ylabel(y)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_example_maps(maps: dict[tuple[str, str], np.ndarray], path: Path) -> None:
    """Save a few example axis maps."""
    if not maps:
        return
    keys = list(maps.keys())[:4]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 4), squeeze=False)
    for ax, key in zip(axes[0], keys):
        ax.imshow(maps[key], cmap="magma", origin="lower")
        ax.set_title(f"{key[0]}\n{key[1]}", fontsize=8)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 12) -> str:
    """Render a compact markdown table."""
    if df.empty:
        return "_No rows._"
    cols = [col for col in columns if col in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in df.head(max_rows)[cols].itertuples(index=False):
        values = [f"{value:.4g}" if isinstance(value, float) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_report(inventory: pd.DataFrame, controls: pd.DataFrame, axis_df: pd.DataFrame, ladder_df: pd.DataFrame, grid_df: pd.DataFrame, radius_df: pd.DataFrame, modes: pd.DataFrame, controls_ok: bool) -> str:
    """Build markdown diagnostic report."""
    control_status = "passed" if controls_ok else "failed"
    axis_best = axis_df.sort_values("anisotropy_ratio", ascending=False) if not axis_df.empty else axis_df
    ladder_summary = ladder_df.groupby("averaging_level", as_index=False)["anisotropy_ratio"].median() if not ladder_df.empty else pd.DataFrame()
    return f"""# Emory Simulation Anisotropy Diagnostic

## Scope

This is a diagnostic of the first-pass simulator, not structural proof. The Emory TIFF remains provenance-limited. Candidate elimination is not justified from this diagnostic alone. Weak simulated anisotropy may reflect simulator assumptions, orientation sampling, fiber-axis choice, missing detector geometry, or true model mismatch. A validated sufficiency-side filter would require confirmed sample provenance, detector calibration, beam center, loop subtraction, and a more physically faithful 2D/fiber diffraction simulator.

The goal is to determine whether the first-pass comparator is washing out arcs or failing to generate them.

## Simulator Controls

- Control status: {control_status}
- Failure modes considered: `averaging_washes_out_arcs`, `weak_even_single_orientation`, `axis_choice_sensitive`, `grid_resolution_sensitive`, `radius_choice_sensitive`, `simulator_control_failure`, `insufficient_data`

{markdown_table(controls, ["model_id", "simulation_mode", "anisotropy_ratio", "orientation_classification"], max_rows=10)}

## Candidate Inventory

{markdown_table(inventory, ["model_id", "status", "coordinate_atom_count", "axis_status", "path"], max_rows=12)}

## Axis Sensitivity

{markdown_table(axis_best, ["model_id", "axis_mode", "anisotropy_ratio", "orientation_classification", "preferred_angle_deg"], max_rows=15)}

## Averaging Ladder

{markdown_table(ladder_summary, ["averaging_level", "anisotropy_ratio"], max_rows=12)}

## Failure Mode Summary

{markdown_table(modes, ["model_id", "anisotropy_failure_mode", "max_single_orientation_anisotropy", "min_single_orientation_anisotropy"], max_rows=12)}

## Interpretation Questions

- Were synthetic anisotropy controls detected correctly? {control_status}.
- Do candidate models produce strong anisotropy in any single-orientation mode? See the axis sensitivity table; values near 3 to 5 would be comparable to the Emory TIFF.
- Does averaging erase anisotropy? See the averaging ladder; a high single-orientation value followed by low strongly averaged values is classified as `averaging_washes_out_arcs`.
- Is the result sensitive to assumed fiber axis? Candidates with large axis-dependent spread are classified as `axis_choice_sensitive`.
- Is the result sensitive to grid size? See `emory_simulation_grid_sensitivity.csv`; large changes trigger `grid_resolution_sensitive`.
- Is the result sensitive to radial feature selection? See `emory_simulation_radius_sensitivity.csv`; large nearby-radius changes trigger `radius_choice_sensitive`.

Before using Emory TIFF fingerprints as structural filters, the comparison needs confirmed sample provenance, detector calibration, beam center, loop subtraction provenance, and a more physically faithful simulated 2D/fiber diffraction workflow.
"""


def run(
    candidate_filter: str,
    max_candidates: int,
    grid_size: int,
    orientation_samples: int,
    radius_window: float,
    axis_mode: str,
    averaging_ladder: bool,
    write_example_maps: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full anisotropy diagnostic and write outputs."""
    _ = radius_window  # Reserved for future calibrated radius-window handling.
    inventory = candidate_inventory(controlled_candidate_records(), candidate_filter, max_candidates)
    controls, controls_ok = control_rows(grid_size, orientation_samples)
    axis_df, maps = axis_sensitivity_rows(inventory, grid_size)
    ladder_df = averaging_ladder_rows(inventory, grid_size, orientation_samples, axis_mode) if averaging_ladder else pd.DataFrame()
    grid_df = grid_sensitivity_rows(inventory, grid_size, axis_mode)
    radius_df = radius_sensitivity_rows(axis_df, inventory, grid_size, axis_mode)
    modes = failure_mode_summary(axis_df, ladder_df, grid_df, radius_df, controls_ok)

    for path, df in [
        (OUT_INVENTORY, inventory),
        (OUT_AXIS, axis_df),
        (OUT_LADDER, ladder_df),
        (OUT_GRID, grid_df),
        (OUT_RADIUS, radius_df),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text(build_report(inventory, controls, axis_df, ladder_df, grid_df, radius_df, modes, controls_ok), encoding="utf-8")
    save_simple_plot(axis_df, FIG_AXIS, "axis_mode", "anisotropy_ratio", "model_id", "Axis sensitivity")
    if not ladder_df.empty:
        save_simple_plot(ladder_df, FIG_LADDER, "averaging_level", "anisotropy_ratio", "model_id", "Averaging ladder")
    save_simple_plot(grid_df, FIG_GRID, "grid_size", "anisotropy_ratio", "model_id", "Grid sensitivity")
    if write_example_maps:
        save_example_maps(maps, FIG_MAPS)
    return inventory, controls, axis_df, ladder_df, grid_df, modes


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-filter", default="")
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--grid-size", type=int, default=256)
    parser.add_argument("--orientation-samples", type=int, default=8)
    parser.add_argument("--radius-window", type=float, default=20.0)
    parser.add_argument("--axis-mode", default="z_axis")
    parser.add_argument("--averaging-ladder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-example-maps", action="store_true")
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    inventory, controls, axis_df, ladder_df, _grid_df, modes = run(
        args.candidate_filter,
        args.max_candidates,
        args.grid_size,
        args.orientation_samples,
        args.radius_window,
        args.axis_mode,
        args.averaging_ladder,
        args.write_example_maps,
    )
    print(f"Candidates inventoried: {len(inventory)}")
    print(f"Controls: {len(controls)}")
    if not axis_df.empty:
        best_axis = axis_df.sort_values("anisotropy_ratio", ascending=False).iloc[0]
        print(
            f"Best axis anisotropy: {best_axis['model_id']} {best_axis['axis_mode']} "
            f"{float(best_axis['anisotropy_ratio']):.3f} {best_axis['orientation_classification']}"
        )
    if not ladder_df.empty:
        print(f"Averaging ladder rows: {len(ladder_df)}")
    print("Failure modes:")
    for row in modes.itertuples(index=False):
        print(f"  {row.model_id}: {row.anisotropy_failure_mode}")
    print(f"Report: {OUT_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
