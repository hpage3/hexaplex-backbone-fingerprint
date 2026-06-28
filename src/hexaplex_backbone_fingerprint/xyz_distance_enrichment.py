"""Coordinate-only distance enrichment analysis for unlabeled XYZ structures."""

from __future__ import annotations

from statistics import mean, median

import numpy as np

from .xyz_parser import XyzAtom

STRAND_COUNT_TERMS = {
    "monoplex": 1,
    "duplex": 2,
    "triplex": 3,
    "tetraplex": 4,
    "pentaplex": 5,
    "hexaplex": 6,
}


def infer_strand_count_from_filename(filename: str) -> int | None:
    """Infer strand count from common plex terms in a filename."""
    lower_name = filename.lower()
    for term, count in STRAND_COUNT_TERMS.items():
        if term in lower_name:
            return count
    return None


def count_distance_band_pairs(
    atoms: list[XyzAtom],
    target: float,
    tolerance: float,
    exclude_hydrogen: bool = True,
) -> dict[str, float | int | None]:
    """Count all coordinate-only atom pairs whose distance is near a target band."""
    selected_atoms = [
        atom for atom in atoms if not exclude_hydrogen or atom.element.upper() not in {"H", "D"}
    ]
    atom_count = len(selected_atoms)
    total_pairs = atom_count * (atom_count - 1) // 2
    if total_pairs == 0:
        return _empty_result(total_pairs)

    coordinates = np.array([(atom.x, atom.y, atom.z) for atom in selected_atoms], dtype=float)
    deltas = coordinates[:, None, :] - coordinates[None, :, :]
    distances = np.sqrt(np.sum(deltas * deltas, axis=2))
    upper_i, upper_j = np.triu_indices(atom_count, k=1)
    pair_distances = distances[upper_i, upper_j]
    mask = np.abs(pair_distances - target) <= tolerance
    candidate_distances = pair_distances[mask]

    if candidate_distances.size == 0:
        return _empty_result(total_pairs)

    errors = np.abs(candidate_distances - target)
    return {
        "candidate_pair_count": int(candidate_distances.size),
        "total_possible_pairs": int(total_pairs),
        "normalized_count": float(candidate_distances.size / total_pairs),
        "min_distance": float(np.min(candidate_distances)),
        "median_distance": float(np.median(candidate_distances)),
        "mean_distance": float(np.mean(candidate_distances)),
        "max_distance": float(np.max(candidate_distances)),
        "median_abs_error": float(np.median(errors)),
    }


def _empty_result(total_pairs: int) -> dict[str, float | int | None]:
    return {
        "candidate_pair_count": 0,
        "total_possible_pairs": int(total_pairs),
        "normalized_count": 0.0 if total_pairs else 0.0,
        "min_distance": None,
        "median_distance": None,
        "mean_distance": None,
        "max_distance": None,
        "median_abs_error": None,
    }
