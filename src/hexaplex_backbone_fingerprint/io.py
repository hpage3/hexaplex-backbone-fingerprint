"""Output helpers for backbone fingerprint analyses."""

from __future__ import annotations

from pathlib import Path
from statistics import mean, median

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
