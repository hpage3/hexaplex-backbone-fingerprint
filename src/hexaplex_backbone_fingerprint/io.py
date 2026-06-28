"""Output helpers for backbone fingerprint analyses."""

from __future__ import annotations

from pathlib import Path
from statistics import mean, median

import numpy as np
import pandas as pd

from .band_mapping import BandCandidatePair
from .peptide_planes import PeptidePlane

PLANE_FEATURE_COLUMNS = [
    "model_label",
    "plane_index",
    "chain",
    "res_i",
    "res_j",
    "resname_i",
    "resname_j",
    "center_x",
    "center_y",
    "center_z",
    "normal_x",
    "normal_y",
    "normal_z",
    "rms",
    "ca_i_plane_dist",
    "c_i_plane_dist",
    "o_i_plane_dist",
    "n_j_plane_dist",
    "ca_j_plane_dist",
    "hn_j_plane_dist",
    "cno_to_peptide_normal_angle_deg",
    "cno_centroid_to_peptide_plane_signed_dist",
    "omega_like_deg",
    "omega_deviation_from_trans_deg",
]

BAND_CANDIDATE_COLUMNS = [
    "model_label",
    "band_name",
    "target_distance",
    "tolerance",
    "distance",
    "error_from_target",
    "plane_index_a",
    "plane_index_b",
    "chain_a",
    "chain_b",
    "res_i_a",
    "res_j_a",
    "res_i_b",
    "res_j_b",
    "sequence_separation",
    "same_chain",
]


def write_plane_features_csv(planes: list[PeptidePlane], path: str | Path, model_label: str) -> None:
    """Write peptide-plane feature records to CSV."""
    rows = []
    for idx, plane in enumerate(planes):
        rows.append(
            {
                "model_label": model_label,
                "plane_index": idx,
                "chain": plane.chain,
                "res_i": plane.res_i,
                "res_j": plane.res_j,
                "resname_i": plane.resname_i,
                "resname_j": plane.resname_j,
                "center_x": plane.center[0],
                "center_y": plane.center[1],
                "center_z": plane.center[2],
                "normal_x": plane.normal[0],
                "normal_y": plane.normal[1],
                "normal_z": plane.normal[2],
                "rms": plane.rms,
                "ca_i_plane_dist": plane.ca_i_plane_dist,
                "c_i_plane_dist": plane.c_i_plane_dist,
                "o_i_plane_dist": plane.o_i_plane_dist,
                "n_j_plane_dist": plane.n_j_plane_dist,
                "ca_j_plane_dist": plane.ca_j_plane_dist,
                "hn_j_plane_dist": plane.hn_j_plane_dist,
                "cno_to_peptide_normal_angle_deg": plane.cno_to_peptide_normal_angle_deg,
                "cno_centroid_to_peptide_plane_signed_dist": plane.cno_centroid_to_peptide_plane_signed_dist,
                "omega_like_deg": plane.omega_like_deg,
                "omega_deviation_from_trans_deg": plane.omega_deviation_from_trans_deg,
            }
        )
    _write_dataframe(rows, path, PLANE_FEATURE_COLUMNS)


def write_band_candidate_pairs_csv(
    candidates: list[BandCandidatePair],
    path: str | Path,
    model_label: str,
) -> None:
    """Write candidate band-distance pairs to CSV."""
    rows = []
    for candidate in candidates:
        rows.append(
            {
                "model_label": model_label,
                "band_name": candidate.band_name,
                "target_distance": candidate.target_distance,
                "tolerance": candidate.tolerance,
                "distance": candidate.distance,
                "error_from_target": candidate.error_from_target,
                "plane_index_a": candidate.plane_index_a,
                "plane_index_b": candidate.plane_index_b,
                "chain_a": candidate.chain_a,
                "chain_b": candidate.chain_b,
                "res_i_a": candidate.res_i_a,
                "res_j_a": candidate.res_j_a,
                "res_i_b": candidate.res_i_b,
                "res_j_b": candidate.res_j_b,
                "sequence_separation": candidate.sequence_separation,
                "same_chain": candidate.same_chain,
            }
        )
    _write_dataframe(rows, path, BAND_CANDIDATE_COLUMNS)


def write_summary_markdown(
    path: str | Path,
    input_file: str | Path,
    model_label: str,
    plane_count: int,
    c_target: float,
    d_target: float,
    tolerance: float,
    c_candidates: list[BandCandidatePair],
    d_candidates: list[BandCandidatePair],
    histogram_path: str | Path | None,
    distortion_plot_paths: list[str | Path],
    planes: list[PeptidePlane],
) -> None:
    """Write a compact Markdown summary for one analysis run."""
    lines = [
        "# Backbone Fingerprint Summary",
        "",
        f"- Input file: `{input_file}`",
        f"- Model label: `{model_label}`",
        f"- C target: {c_target:.3f} Angstrom",
        f"- D target: {d_target:.3f} Angstrom",
        f"- Tolerance: +/- {tolerance:.3f} Angstrom",
        f"- Peptide planes: {plane_count}",
        f"- C candidates: {len(c_candidates)}",
        f"- D candidates: {len(d_candidates)}",
        "",
        "## C Band",
        "",
        _band_summary(c_candidates),
        "",
        "## D Band",
        "",
        _band_summary(d_candidates),
        "",
        "## Plots",
        "",
        _plot_summary(histogram_path),
        *_distortion_plot_summary(distortion_plot_paths),
        "",
        "## Distortion Correlations",
        "",
        _correlation_summary(planes),
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def write_band_candidate_histogram(
    c_candidates: list[BandCandidatePair],
    d_candidates: list[BandCandidatePair],
    path: str | Path,
) -> Path | None:
    """Write a simple histogram of candidate distances, or return None when empty."""
    if not c_candidates and not d_candidates:
        return None

    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    if c_candidates:
        ax.hist([candidate.distance for candidate in c_candidates], bins="auto", alpha=0.65, label="C candidates")
    if d_candidates:
        ax.hist([candidate.distance for candidate in d_candidates], bins="auto", alpha=0.65, label="D candidates")
    ax.set_xlabel("Plane-center distance (Angstrom)")
    ax.set_ylabel("Candidate count")
    ax.set_title("Band Candidate Plane-Center Distances")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def write_distortion_scatter_plots(planes: list[PeptidePlane], outdir: str | Path) -> list[Path]:
    """Write simple scatter plots for peptide-plane distortion metrics."""
    output_dir = Path(outdir)
    plot_specs = [
        (
            "rms_vs_cno_angle.png",
            "rms",
            "cno_to_peptide_normal_angle_deg",
            "Plane RMS (Angstrom)",
            "CNO to peptide normal angle (deg)",
        ),
        (
            "rms_vs_omega_deviation.png",
            "rms",
            "omega_deviation_from_trans_deg",
            "Plane RMS (Angstrom)",
            "Omega deviation from trans (deg)",
        ),
        (
            "cno_angle_vs_omega_deviation.png",
            "cno_to_peptide_normal_angle_deg",
            "omega_deviation_from_trans_deg",
            "CNO to peptide normal angle (deg)",
            "Omega deviation from trans (deg)",
        ),
    ]

    written: list[Path] = []
    for filename, x_attr, y_attr, xlabel, ylabel in plot_specs:
        pairs = _finite_pairs(planes, x_attr, y_attr)
        if len(pairs) < 3:
            continue

        import matplotlib.pyplot as plt

        output_path = output_dir / filename
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter([pair[0] for pair in pairs], [pair[1] for pair in pairs], s=18, alpha=0.75)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(filename.replace("_", " ").replace(".png", "").title())
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        written.append(output_path)
    return written


def _write_dataframe(rows: list[dict], path: str | Path, columns: list[str]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(output_path, index=False)


def _band_summary(candidates: list[BandCandidatePair]) -> str:
    if not candidates:
        return "No candidates found for this band."

    distances = [candidate.distance for candidate in candidates]
    absolute_errors = [abs(candidate.error_from_target) for candidate in candidates]
    same_chain_count = sum(1 for candidate in candidates if candidate.same_chain)
    cross_chain_count = len(candidates) - same_chain_count
    return "\n".join(
        [
            f"- Minimum distance: {min(distances):.3f}",
            f"- Median distance: {median(distances):.3f}",
            f"- Mean distance: {mean(distances):.3f}",
            f"- Maximum distance: {max(distances):.3f}",
            f"- Median absolute error from target: {median(absolute_errors):.3f}",
            f"- Same-chain candidates: {same_chain_count}",
            f"- Cross-chain candidates: {cross_chain_count}",
        ]
    )


def _plot_summary(histogram_path: str | Path | None) -> str:
    if histogram_path is None:
        return "No candidate distances found, so no histogram was written."
    return f"- Candidate distance histogram: `{Path(histogram_path).name}`"


def _distortion_plot_summary(plot_paths: list[str | Path]) -> list[str]:
    if not plot_paths:
        return ["No distortion scatter plots were written because too few finite values were available."]
    return [f"- Distortion scatter plot: `{Path(path).name}`" for path in plot_paths]


def _correlation_summary(planes: list[PeptidePlane]) -> str:
    correlations = [
        (
            "rms vs cno_to_peptide_normal_angle_deg",
            "rms",
            "cno_to_peptide_normal_angle_deg",
            False,
        ),
        (
            "rms vs absolute cno_centroid_to_peptide_plane_signed_dist",
            "rms",
            "cno_centroid_to_peptide_plane_signed_dist",
            True,
        ),
        (
            "rms vs omega_deviation_from_trans_deg",
            "rms",
            "omega_deviation_from_trans_deg",
            False,
        ),
        (
            "cno_to_peptide_normal_angle_deg vs omega_deviation_from_trans_deg",
            "cno_to_peptide_normal_angle_deg",
            "omega_deviation_from_trans_deg",
            False,
        ),
    ]

    lines = []
    for label, x_attr, y_attr, use_abs_y in correlations:
        pairs = _finite_pairs(planes, x_attr, y_attr, use_abs_y=use_abs_y)
        if len(pairs) < 3:
            lines.append(f"- {label}: too few finite values for Pearson correlation.")
            continue
        x_values = np.array([pair[0] for pair in pairs], dtype=float)
        y_values = np.array([pair[1] for pair in pairs], dtype=float)
        if np.isclose(np.std(x_values), 0.0) or np.isclose(np.std(y_values), 0.0):
            lines.append(f"- {label}: undefined because one value series is constant.")
            continue
        corr = float(np.corrcoef(x_values, y_values)[0, 1])
        lines.append(f"- {label}: r = {corr:.3f} (n = {len(pairs)})")
    return "\n".join(lines)


def _finite_pairs(
    planes: list[PeptidePlane],
    x_attr: str,
    y_attr: str,
    use_abs_y: bool = False,
) -> list[tuple[float, float]]:
    pairs = []
    for plane in planes:
        x_value = getattr(plane, x_attr)
        y_value = getattr(plane, y_attr)
        if x_value is None or y_value is None:
            continue
        if use_abs_y:
            y_value = abs(y_value)
        if np.isfinite(x_value) and np.isfinite(y_value):
            pairs.append((float(x_value), float(y_value)))
    return pairs
