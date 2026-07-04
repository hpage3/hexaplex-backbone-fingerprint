"""Compare candidate coordinate models to the Emory S1-02 fiber orientation fingerprint.

This is a first-pass exploratory model-to-image comparison. It preserves 2D
angular information with a lightweight projection/FFT diagnostic rather than a
validated full fiber-diffraction simulator. Scores are meant to prioritize
follow-up comparisons, not to eliminate structures.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_emory_s1_02_fiber_orientation_fingerprint import (
    anisotropy_ratio_from_azimuthal,
    classify_orientation,
    detect_arc_peaks,
    sector_means_from_summary,
    top_radial_features,
)
from scripts.audit_emory_s1_02_fiber_xray_images import (
    azimuthal_profile,
    radial_profile,
    sector_summary,
)


TARGET_DOMINANT_ANGLE_DEG = 342.5
TARGET_OPPOSITE_ANGLE_DEG = 162.5
TARGET_ANISOTROPY_MIN = 3.03
TARGET_ANISOTROPY_MAX = 4.80
TARGET_CLASSIFICATION = "fiber_oriented_arc_like"

DEFAULT_METRICS_DIR = Path("outputs/metrics")
DEFAULT_REPORTS_DIR = Path("outputs/reports")
DEFAULT_FIGURES_DIR = Path("outputs/figures")

OUT_INVENTORY = DEFAULT_METRICS_DIR / "emory_model_orientation_candidate_inventory.csv"
OUT_FINGERPRINTS = DEFAULT_METRICS_DIR / "emory_model_orientation_fingerprints.csv"
OUT_SCORES = DEFAULT_METRICS_DIR / "emory_model_orientation_similarity_scores.csv"
OUT_REPORT = DEFAULT_REPORTS_DIR / "emory_model_orientation_comparison_report.md"
FIG_BEST = DEFAULT_FIGURES_DIR / "emory_model_orientation_best_matches.png"
FIG_SCORE = DEFAULT_FIGURES_DIR / "emory_model_orientation_score_summary.png"
FIG_MAPS = DEFAULT_FIGURES_DIR / "emory_model_orientation_example_maps.png"

EXPECTED_OMEGA_CLEAN = [
    "omega_clean_scale_0p9825",
    "omega_clean_scale_0p9800",
    "omega_clean_scale_0p9775",
    "omega_clean_scale_0p9750",
    "omega_clean_scale_0p9725",
    "omega_clean_scale_1p0000",
    "omega_clean_scale_0p9700",
]


@dataclass(frozen=True)
class CandidateRecord:
    """One explicitly inventoried coordinate candidate."""

    model_id: str
    path: Path | None
    inferred_family: str
    inferred_scale: str
    inferred_twist: str
    inferred_rise: str
    inclusion_reason: str
    provenance_caveat: str
    status: str


def angular_difference(a: float, b: float) -> float:
    """Return circular absolute angular difference in degrees."""
    return float(abs(((a - b + 180.0) % 360.0) - 180.0))


def two_lobe_separation_error(angle_a: float, angle_b: float) -> float:
    """Return deviation from a 180-degree two-lobe pair."""
    return abs(angular_difference(angle_a, angle_b) - 180.0)


def normalize_array(values: np.ndarray) -> np.ndarray:
    """Normalize array to 0..1 for plotting/comparison."""
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=float)
    lo = float(np.nanmin(arr[finite]))
    hi = float(np.nanmax(arr[finite]))
    if hi <= lo:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


def parse_scale_from_model_id(model_id: str) -> str:
    """Infer scale token from a model id if present."""
    match = re.search(r"scale_(\d+p\d+)", model_id)
    return match.group(1).replace("p", ".") if match else ""


def expected_candidate_records() -> list[CandidateRecord]:
    """Return expected candidates without random filesystem discovery."""
    records: list[CandidateRecord] = []
    omega_dir = Path("outputs/coordinates/omega_clean_rise_compression_scan")
    for model_id in EXPECTED_OMEGA_CLEAN:
        path = omega_dir / f"{model_id}.pdb"
        records.append(
            CandidateRecord(
                model_id=model_id,
                path=path if path.exists() else None,
                inferred_family="omega_clean_rise_compressed",
                inferred_scale=parse_scale_from_model_id(model_id),
                inferred_twist="",
                inferred_rise="parent-derived scale",
                inclusion_reason="Expected omega-clean rise-compression plateau/baseline/end-point candidate from manuscript pipeline.",
                provenance_caveat="Diagnostic coordinate family; not proof of pNAB/YAML physical provenance.",
                status="found" if path.exists() else "missing_candidate_coordinates",
            )
        )

    guarded = Path("outputs/coordinates/guarded_full_chain_prototype/guarded_full_chain_prototype.pdb")
    records.append(
        CandidateRecord(
            model_id="guarded_full_chain_prototype",
            path=guarded if guarded.exists() else None,
            inferred_family="guarded_full_chain_prototype",
            inferred_scale="",
            inferred_twist="",
            inferred_rise="",
            inclusion_reason="Explicit guarded full-chain prototype requested as a candidate family.",
            provenance_caveat="Prototype coordinate model; diagnostic only.",
            status="found" if guarded.exists() else "missing_candidate_coordinates",
        )
    )

    pnab_candidates = [
        (
            "pnab_antiparallel_30_candidate",
            Path("outputs/coordinates/ideal_hexaflex_variants/ideal_hexaflex_backbone_plus_carboxylate.pdb"),
            "pNAB/parent anti-parallel 30-degree-like candidate if clearly identifiable by prior ideal Hexaflex workflow.",
            "anti-parallel 30-like parent-derived ideal Hexaflex; provenance remains partial.",
        ),
        (
            "pnab_parallel_candidate",
            None,
            "Parallel pNAB candidate requested only if clearly identifiable.",
            "No clearly labeled parallel pNAB coordinate was selected in this conservative inventory.",
        ),
    ]
    for model_id, path, reason, caveat in pnab_candidates:
        found = path is not None and path.exists()
        records.append(
            CandidateRecord(
                model_id=model_id,
                path=path if found else None,
                inferred_family="pnab_reference_candidate",
                inferred_scale="",
                inferred_twist="30" if "antiparallel" in model_id else "",
                inferred_rise="",
                inclusion_reason=reason,
                provenance_caveat=caveat,
                status="found" if found else "missing_candidate_coordinates",
            )
        )
    return records


def inventory_dataframe(records: Iterable[CandidateRecord]) -> pd.DataFrame:
    """Convert candidate records to an inventory dataframe."""
    return pd.DataFrame(
        [
            {
                "model_id": record.model_id,
                "path": str(record.path) if record.path else "",
                "inferred_family": record.inferred_family,
                "inferred_scale": record.inferred_scale,
                "inferred_twist": record.inferred_twist,
                "inferred_rise": record.inferred_rise,
                "inclusion_reason": record.inclusion_reason,
                "provenance_caveat": record.provenance_caveat,
                "status": record.status,
            }
            for record in records
        ]
    )


def read_pdb_coordinates(path: Path, exclude_hydrogen: bool = True) -> np.ndarray:
    """Read ATOM/HETATM coordinates from a PDB file."""
    coords: list[list[float]] = []
    for line in path.read_text(encoding="ascii").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        atom_name = line[12:16].strip()
        element = (line[76:78].strip() or atom_name[:1]).upper()
        if exclude_hydrogen and (element == "H" or atom_name.upper().startswith("H")):
            continue
        coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    if not coords:
        raise ValueError(f"No ATOM/HETATM coordinates found in {path}.")
    return np.asarray(coords, dtype=float)


def rotate_z(coords: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate coordinates about z axis."""
    angle = math.radians(angle_deg)
    rot = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return coords @ rot.T


def rotate_x(coords: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate coordinates about x axis."""
    angle = math.radians(angle_deg)
    rot = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(angle), -math.sin(angle)],
            [0.0, math.sin(angle), math.cos(angle)],
        ]
    )
    return coords @ rot.T


def projected_fft_intensity(coords: np.ndarray, grid_size: int = 256, padding_fraction: float = 0.15) -> np.ndarray:
    """Return a lightweight 2D projection/FFT intensity map.

    This is not a full calibrated fiber diffraction simulator. It bins projected
    atom density into the detector plane, then uses FFT magnitude as a diagnostic
    2D reciprocal-space texture.
    """
    centered = coords - np.mean(coords, axis=0)
    xy = centered[:, :2]
    extent = float(np.max(np.abs(xy))) if len(xy) else 1.0
    extent = max(extent * (1.0 + padding_fraction), 1.0)
    scaled = (xy / (2.0 * extent) + 0.5) * (grid_size - 1)
    ix = np.clip(np.rint(scaled[:, 0]).astype(int), 0, grid_size - 1)
    iy = np.clip(np.rint(scaled[:, 1]).astype(int), 0, grid_size - 1)
    density = np.zeros((grid_size, grid_size), dtype=float)
    np.add.at(density, (iy, ix), 1.0)
    density -= float(np.mean(density))
    intensity = np.abs(np.fft.fftshift(np.fft.fft2(density))) ** 2
    return normalize_array(np.log1p(intensity))


def simulate_2d_map(coords: np.ndarray, grid_size: int, orientation_samples: int, mode: str) -> np.ndarray:
    """Simulate one diagnostic 2D intensity map."""
    centered = coords - np.mean(coords, axis=0)
    if mode == "single_oriented":
        oriented = rotate_x(centered, 12.0)
        return projected_fft_intensity(oriented, grid_size=grid_size)
    if mode != "fiber_axis_average":
        raise ValueError(f"Unknown simulation mode: {mode}")
    samples = max(1, int(orientation_samples))
    maps = []
    for idx in range(samples):
        angle = idx * 360.0 / samples
        oriented = rotate_x(rotate_z(centered, angle), 12.0)
        maps.append(projected_fft_intensity(oriented, grid_size=grid_size))
    return normalize_array(np.mean(maps, axis=0))


def center_for_map(array: np.ndarray) -> tuple[float, float]:
    """Return image-center beam proxy for simulated maps."""
    height, width = array.shape[:2]
    return (width - 1) / 2.0, (height - 1) / 2.0


def feature_radii(profile: pd.DataFrame, count: int = 3) -> list[float]:
    """Return radial feature radii for simulated-map fingerprinting."""
    radii = top_radial_features(profile, count=count, min_separation_px=8.0)
    if radii:
        return radii
    values = pd.to_numeric(profile.get("mean_intensity", pd.Series(dtype=float)), errors="coerce")
    if profile.empty or values.dropna().empty:
        return []
    idx = values.idxmax()
    return [float(profile.loc[idx, "radius_px"])]


def fingerprint_intensity_map(model_id: str, mode: str, intensity: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract orientation fingerprint rows from one simulated 2D map."""
    center = center_for_map(intensity)
    mask = np.isfinite(intensity)
    radial = radial_profile(intensity, center, mask=mask, bin_width=1.0)
    rows: list[dict[str, object]] = []
    arcs: list[dict[str, object]] = []
    for radius in feature_radii(radial, count=3):
        az = azimuthal_profile(intensity, center, radius, radius_half_width_px=3.0, angle_bin_deg=5.0, mask=mask)
        arc_df = detect_arc_peaks(az, min_relative_height=0.45)
        sectors = sector_means_from_summary(sector_summary(intensity, center, radius, mask=mask))
        anisotropy = anisotropy_ratio_from_azimuthal(az)
        classification = classify_simulated_orientation(anisotropy, len(arc_df))
        preferred = float(arc_df.iloc[0]["angle_deg"]) if not arc_df.empty else float("nan")
        opposite = float(arc_df.iloc[1]["angle_deg"]) if len(arc_df) > 1 else (preferred + 180.0) % 360.0 if np.isfinite(preferred) else float("nan")
        rows.append(
            {
                "model_id": model_id,
                "simulation_mode": mode,
                "radius_px": radius,
                "preferred_angle_deg": preferred,
                "opposite_angle_deg": opposite,
                "anisotropy_ratio": anisotropy,
                "number_of_arc_maxima": len(arc_df),
                "arc_width_deg": float(arc_df.iloc[0]["arc_width_deg"]) if not arc_df.empty else float("nan"),
                "orientation_classification": classification,
                **sectors,
            }
        )
        for arc in arc_df.itertuples(index=False):
            arcs.append(
                {
                    "model_id": model_id,
                    "simulation_mode": mode,
                    "radius_px": radius,
                    "angle_deg": float(arc.angle_deg),
                    "relative_intensity": float(arc.relative_intensity),
                    "arc_width_deg": float(arc.arc_width_deg),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(arcs)


def classify_simulated_orientation(anisotropy_ratio: float, number_of_arc_maxima: int) -> str:
    """Classify simulated orientation fingerprints."""
    if not np.isfinite(anisotropy_ratio) or number_of_arc_maxima == 0:
        return "insufficient_quality"
    if anisotropy_ratio >= 2.2 and number_of_arc_maxima >= 1:
        return "fiber_oriented_arc_like"
    if anisotropy_ratio >= 1.45:
        return "mixed_orientation"
    if anisotropy_ratio >= 1.15:
        return "weak_orientation"
    return "isotropic_ring_like"


def best_rotational_alignment_error(sim_angle: float, target_angle: float = TARGET_DOMINANT_ANGLE_DEG) -> float:
    """Return angular error after allowing arbitrary detector rotation alignment."""
    if not np.isfinite(sim_angle):
        return 180.0
    rotation = (target_angle - sim_angle) % 360.0
    aligned = (sim_angle + rotation) % 360.0
    return angular_difference(aligned, target_angle)


def anisotropy_score(anisotropy_ratio: float, target_min: float = TARGET_ANISOTROPY_MIN, target_max: float = TARGET_ANISOTROPY_MAX) -> float:
    """Return 0..1 anisotropy similarity score."""
    if not np.isfinite(anisotropy_ratio) or anisotropy_ratio <= 0:
        return 0.0
    target_mid = 0.5 * (target_min + target_max)
    if target_min <= anisotropy_ratio <= target_max:
        return 1.0
    rel_error = abs(anisotropy_ratio - target_mid) / target_mid
    return float(max(0.0, 1.0 - rel_error))


def angular_lobe_score(preferred_angle: float, opposite_angle: float) -> float:
    """Score two-lobe geometry while allowing arbitrary detector rotation."""
    if not np.isfinite(preferred_angle):
        return 0.0
    alignment_error = best_rotational_alignment_error(preferred_angle)
    alignment_score = max(0.0, 1.0 - alignment_error / 90.0)
    if np.isfinite(opposite_angle):
        separation_score = max(0.0, 1.0 - two_lobe_separation_error(preferred_angle, opposite_angle) / 90.0)
    else:
        separation_score = 0.5
    return float(0.35 * alignment_score + 0.65 * separation_score)


def sector_pattern_score(row: pd.Series) -> float:
    """Score whether the sector pattern has a clear directional preference."""
    cols = [
        "horizontal_sector_mean",
        "vertical_sector_mean",
        "diagonal_positive_sector_mean",
        "diagonal_negative_sector_mean",
    ]
    values = pd.to_numeric(row[[col for col in cols if col in row.index]], errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) < 2 or np.nanmean(values) <= 0:
        return 0.0
    ratio = float(np.nanmax(values) / np.nanmean(values))
    return float(min(1.0, max(0.0, (ratio - 1.0) / 1.5)))


def radial_feature_score(radius_px: float, grid_size: int) -> float:
    """Score whether the feature is in a usable noncentral radial zone."""
    if not np.isfinite(radius_px) or grid_size <= 0:
        return 0.0
    normalized = radius_px / (0.5 * grid_size)
    if 0.25 <= normalized <= 0.85:
        return 1.0
    return float(max(0.0, 1.0 - min(abs(normalized - 0.25), abs(normalized - 0.85)) / 0.25))


def classification_match_score(classification: str) -> float:
    """Score classification similarity to the Emory target."""
    if classification == TARGET_CLASSIFICATION:
        return 1.0
    if classification == "mixed_orientation":
        return 0.7
    if classification == "weak_orientation":
        return 0.35
    if classification == "isotropic_ring_like":
        return 0.1
    return 0.0


def caveat_penalty(provenance_caveat: str, mode: str) -> float:
    """Return a small transparent penalty for weak provenance/diagnostic mode."""
    penalty = 0.0
    text = provenance_caveat.lower()
    if "partial" in text or "prototype" in text:
        penalty += 0.05
    if mode == "single_oriented":
        penalty += 0.05
    return min(0.15, penalty)


def score_fingerprint_row(row: pd.Series, grid_size: int, provenance_caveat: str) -> dict[str, float]:
    """Score one simulated fingerprint row against the Emory angular fingerprint."""
    angular = angular_lobe_score(float(row.get("preferred_angle_deg", np.nan)), float(row.get("opposite_angle_deg", np.nan)))
    anis = anisotropy_score(float(row.get("anisotropy_ratio", np.nan)))
    sector = sector_pattern_score(row)
    radial = radial_feature_score(float(row.get("radius_px", np.nan)), grid_size)
    cls = classification_match_score(str(row.get("orientation_classification", "")))
    penalty = caveat_penalty(provenance_caveat, str(row.get("simulation_mode", "")))
    total = 0.32 * angular + 0.26 * anis + 0.16 * sector + 0.12 * radial + 0.14 * cls - penalty
    return {
        "angular_lobe_score": float(angular),
        "anisotropy_score": float(anis),
        "sector_pattern_score": float(sector),
        "radial_feature_score": float(radial),
        "classification_match_score": float(cls),
        "caveat_penalty": float(penalty),
        "emory_orientation_similarity_score": float(max(0.0, min(1.0, total))),
    }


def top_score_rows(fingerprints: pd.DataFrame, inventory: pd.DataFrame, grid_size: int) -> pd.DataFrame:
    """Return best scored row per model/mode."""
    rows: list[dict[str, object]] = []
    caveats = dict(zip(inventory["model_id"], inventory["provenance_caveat"]))
    for _, row in fingerprints.iterrows():
        scores = score_fingerprint_row(row, grid_size, caveats.get(str(row["model_id"]), ""))
        rows.append({**row.to_dict(), **scores})
    scored = pd.DataFrame(rows)
    if scored.empty:
        return scored
    idx = scored.groupby(["model_id", "simulation_mode"])["emory_orientation_similarity_score"].idxmax()
    return scored.loc[idx].sort_values("emory_orientation_similarity_score", ascending=False).reset_index(drop=True)


def candidate_modes(mode: str) -> list[str]:
    """Return simulation modes requested by CLI."""
    if mode == "both":
        return ["single_oriented", "fiber_axis_average"]
    return [mode]


def filter_inventory(inventory: pd.DataFrame, candidate_filter: str) -> pd.DataFrame:
    """Filter found inventory by model id/family substring."""
    found = inventory[inventory["status"] == "found"].copy()
    if candidate_filter:
        needle = candidate_filter.lower()
        found = found[
            found["model_id"].str.lower().str.contains(needle)
            | found["inferred_family"].str.lower().str.contains(needle)
        ]
    return found.reset_index(drop=True)


def analyze_candidates(inventory: pd.DataFrame, max_candidates: int, grid_size: int, orientation_samples: int, mode: str, candidate_filter: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str], np.ndarray]]:
    """Run simulated 2D map and fingerprint extraction for selected candidates."""
    selected = filter_inventory(inventory, candidate_filter)
    if max_candidates > 0:
        selected = selected.head(max_candidates)
    fingerprint_frames: list[pd.DataFrame] = []
    arc_frames: list[pd.DataFrame] = []
    maps: dict[tuple[str, str], np.ndarray] = {}
    for row in selected.itertuples(index=False):
        coords = read_pdb_coordinates(Path(row.path), exclude_hydrogen=True)
        for sim_mode in candidate_modes(mode):
            intensity = simulate_2d_map(coords, grid_size=grid_size, orientation_samples=orientation_samples, mode=sim_mode)
            maps[(str(row.model_id), sim_mode)] = intensity
            fp, arcs = fingerprint_intensity_map(str(row.model_id), sim_mode, intensity)
            fingerprint_frames.append(fp)
            arc_frames.append(arcs)
    fingerprints = pd.concat(fingerprint_frames, ignore_index=True) if fingerprint_frames else pd.DataFrame()
    arcs = pd.concat(arc_frames, ignore_index=True) if arc_frames else pd.DataFrame()
    return fingerprints, arcs, maps


def save_score_summary_plot(scores: pd.DataFrame, path: Path) -> None:
    """Save score bar plot."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    if not scores.empty:
        plot_df = scores.head(12).copy()
        labels = [f"{row.model_id}\n{row.simulation_mode}" for row in plot_df.itertuples()]
        ax.bar(range(len(plot_df)), plot_df["emory_orientation_similarity_score"], color="#4c78a8")
        ax.set_xticks(range(len(plot_df)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("similarity score")
    ax.set_title("Emory S1-02 model orientation similarity scores")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_best_matches_plot(scores: pd.DataFrame, path: Path) -> None:
    """Save component score plot for top matches."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    components = [
        "angular_lobe_score",
        "anisotropy_score",
        "sector_pattern_score",
        "radial_feature_score",
        "classification_match_score",
    ]
    if not scores.empty:
        plot_df = scores.head(6).copy()
        bottom = np.zeros(len(plot_df))
        labels = [f"{row.model_id}\n{row.simulation_mode}" for row in plot_df.itertuples()]
        for component in components:
            values = pd.to_numeric(plot_df[component], errors="coerce").fillna(0).to_numpy(dtype=float) / len(components)
            ax.bar(range(len(plot_df)), values, bottom=bottom, label=component.replace("_score", ""))
            bottom += values
        ax.set_xticks(range(len(plot_df)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.legend(fontsize=7, ncols=2)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("component contribution")
    ax.set_title("Top first-pass orientation matches")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_example_maps(maps: dict[tuple[str, str], np.ndarray], scores: pd.DataFrame, path: Path) -> None:
    """Save example simulated 2D maps for top scored rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    selected_keys: list[tuple[str, str]] = []
    for row in scores.head(4).itertuples():
        key = (str(row.model_id), str(row.simulation_mode))
        if key in maps and key not in selected_keys:
            selected_keys.append(key)
    if not selected_keys:
        selected_keys = list(maps.keys())[:4]
    cols = max(1, len(selected_keys))
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4), squeeze=False)
    for ax, key in zip(axes[0], selected_keys):
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
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.4g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_report(inventory: pd.DataFrame, fingerprints: pd.DataFrame, scores: pd.DataFrame, mode: str) -> str:
    """Build the markdown comparison report."""
    found = inventory[inventory["status"] == "found"]
    missing = inventory[inventory["status"] != "found"]
    top = scores.head(8)
    strong_mismatch = scores[scores["emory_orientation_similarity_score"] < 0.25] if not scores.empty else scores
    classifications = fingerprints["orientation_classification"].value_counts().to_dict() if not fingerprints.empty else {}
    return f"""# Emory S1-02 Model Orientation Comparison

## Scope

This is a first-pass exploratory model-to-image comparison. It asks whether candidate structures generate simulated 2D/fiber diffraction-like intensity maps with angular anisotropy similar to the Emory S1-02 TIFF orientation fingerprint. The Emory TIFF is provenance-limited and lacks calibration metadata, so absolute d-spacing/radius agreement is not claimed unless calibration is supplied.

The comparison allows arbitrary detector rotation alignment. Similarity scores are diagnostic, not structural proof. No candidate should be eliminated solely from this analysis in this first pass. Strong mismatches may identify candidates for deprioritization or further testing. A validated comparison would require confirmed sample identity, detector calibration, beam center, loop subtraction provenance, and a more complete simulated 2D/fiber diffraction or oriented-powder simulator.

## Candidate Inventory

- Candidate coordinate files found: {len(found)}
- Expected candidates missing: {len(missing)}
- Simulation mode requested: `{mode}`

{markdown_table(inventory, ["model_id", "status", "inferred_family", "inferred_scale", "path", "provenance_caveat"], max_rows=20)}

## Missing Expected Candidates

{markdown_table(missing, ["model_id", "status", "inclusion_reason", "provenance_caveat"], max_rows=20)}

## Simulation Method

The first-pass simulator bins heavy-atom projected density onto a 2D grid and uses FFT magnitude as a diagnostic reciprocal-space texture. `single_oriented` uses one default model orientation. `fiber_axis_average` samples rotations about the model z/helical axis while preserving detector-plane anisotropy. This intentionally does not fully powder-average away angular information.

## Orientation Fingerprint Summary

- Experimental target dominant arc: {TARGET_DOMINANT_ANGLE_DEG:.1f} deg
- Experimental target opposite lobe: {TARGET_OPPOSITE_ANGLE_DEG:.1f} deg
- Experimental anisotropy range: {TARGET_ANISOTROPY_MIN:.2f} to {TARGET_ANISOTROPY_MAX:.2f}
- Simulated classification counts: {classifications}

## Top Similarity Scores

{markdown_table(top, ["model_id", "simulation_mode", "emory_orientation_similarity_score", "orientation_classification", "anisotropy_ratio", "preferred_angle_deg", "opposite_angle_deg"], max_rows=12)}

## Strong Mismatches

{markdown_table(strong_mismatch, ["model_id", "simulation_mode", "emory_orientation_similarity_score", "orientation_classification", "anisotropy_ratio"], max_rows=12)}

## Interpretation

This comparison tests possible additional constraints beyond A/B/C/D by preserving angular information. Models that produce arc-like anisotropy after arbitrary detector rotation alignment are better exploratory matches to the Emory S1-02 orientation fingerprint than models that produce isotropic rings or weak anisotropy. Because the image is fiber-like, provenance-limited, and uncalibrated, the result should be used to prioritize future simulated 2D/fiber diffraction comparisons rather than to make structural claims.
"""


def run(max_candidates: int, grid_size: int, orientation_samples: int, mode: str, candidate_filter: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full comparison and write outputs."""
    records = expected_candidate_records()
    inventory = inventory_dataframe(records)
    OUT_INVENTORY.parent.mkdir(parents=True, exist_ok=True)
    OUT_FINGERPRINTS.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(OUT_INVENTORY, index=False)

    fingerprints, _arcs, maps = analyze_candidates(inventory, max_candidates, grid_size, orientation_samples, mode, candidate_filter)
    scores = top_score_rows(fingerprints, inventory, grid_size) if not fingerprints.empty else pd.DataFrame()
    fingerprints.to_csv(OUT_FINGERPRINTS, index=False)
    scores.to_csv(OUT_SCORES, index=False)
    OUT_REPORT.write_text(build_report(inventory, fingerprints, scores, mode), encoding="utf-8")
    save_score_summary_plot(scores, FIG_SCORE)
    save_best_matches_plot(scores, FIG_BEST)
    save_example_maps(maps, scores, FIG_MAPS)
    return inventory, fingerprints, scores


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-candidates", type=int, default=0, help="Maximum found candidates to score; 0 means all.")
    parser.add_argument("--grid-size", type=int, default=512)
    parser.add_argument("--orientation-samples", type=int, default=8)
    parser.add_argument("--mode", choices=["single_oriented", "fiber_axis_average", "both"], default="both")
    parser.add_argument("--candidate-filter", default="")
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    inventory, fingerprints, scores = run(args.max_candidates, args.grid_size, args.orientation_samples, args.mode, args.candidate_filter)
    found_count = int((inventory["status"] == "found").sum())
    missing_count = int((inventory["status"] != "found").sum())
    print(f"Candidate records: {len(inventory)} ({found_count} found, {missing_count} missing)")
    print(f"Fingerprint rows: {len(fingerprints)}")
    if not scores.empty:
        print("Top scores:")
        for row in scores.head(5).itertuples():
            print(f"  {row.model_id} [{row.simulation_mode}] score={row.emory_orientation_similarity_score:.3f} class={row.orientation_classification}")
    print(f"Report: {OUT_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
