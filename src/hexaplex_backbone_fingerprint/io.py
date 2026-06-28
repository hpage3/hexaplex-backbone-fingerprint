"""Output helpers for backbone fingerprint analyses."""

from __future__ import annotations

from pathlib import Path
from statistics import mean, median

import pandas as pd

from .band_mapping import BandCandidatePair
from .peptide_planes import PeptidePlane


def write_plane_features_csv(planes: list[PeptidePlane], path: str | Path) -> None:
    """Write peptide-plane feature records to CSV."""
    rows = []
    for idx, plane in enumerate(planes):
        rows.append(
            {
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
                "c_i": plane.c_i,
                "o_i": plane.o_i,
                "n_j": plane.n_j,
                "ca_i": plane.ca_i,
                "ca_j": plane.ca_j,
                "hn_j": plane.hn_j,
            }
        )
    _write_dataframe(rows, path)


def write_band_candidate_pairs_csv(candidates: list[BandCandidatePair], path: str | Path) -> None:
    """Write candidate band-distance pairs to CSV."""
    rows = [candidate.__dict__ for candidate in candidates]
    _write_dataframe(rows, path)


def write_summary_markdown(
    path: str | Path,
    input_file: str | Path,
    plane_count: int,
    c_candidates: list[BandCandidatePair],
    d_candidates: list[BandCandidatePair],
) -> None:
    """Write a compact Markdown summary for one analysis run."""
    lines = [
        "# Backbone Fingerprint Summary",
        "",
        f"- Input file: `{input_file}`",
        f"- Peptide planes: {plane_count}",
        f"- C candidates: {len(c_candidates)}",
        f"- D candidates: {len(d_candidates)}",
        "",
        "## C Band",
        "",
        _distance_summary(c_candidates),
        "",
        "## D Band",
        "",
        _distance_summary(d_candidates),
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _write_dataframe(rows: list[dict], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def _distance_summary(candidates: list[BandCandidatePair]) -> str:
    if not candidates:
        return "No candidates found."
    distances = [candidate.distance for candidate in candidates]
    return "\n".join(
        [
            f"- Mean distance: {mean(distances):.3f}",
            f"- Median distance: {median(distances):.3f}",
            f"- Minimum distance: {min(distances):.3f}",
            f"- Maximum distance: {max(distances):.3f}",
        ]
    )
