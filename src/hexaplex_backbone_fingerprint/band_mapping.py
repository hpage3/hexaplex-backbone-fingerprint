"""Map peptide-plane features to candidate C/D band distances."""

from __future__ import annotations

from dataclasses import dataclass

from .geometry import distance
from .peptide_planes import PeptidePlane


@dataclass(frozen=True)
class BandCandidatePair:
    """A pair of peptide-plane features whose distance is near a target band."""

    band_name: str
    target_distance: float
    tolerance: float
    distance: float
    error_from_target: float
    feature_type: str
    plane_index_a: int
    plane_index_b: int
    chain_a: str
    chain_b: str
    res_i_a: int
    res_j_a: int
    res_i_b: int
    res_j_b: int
    sequence_separation: int | None
    same_chain: bool


def find_band_candidate_pairs(
    planes: list[PeptidePlane],
    target_distance: float,
    tolerance: float,
    band_name: str,
) -> list[BandCandidatePair]:
    """Find plane-center to plane-center pairs near a target distance.

    TODO: Add O-O, C-O, C/N/O subplane-center features.
    TODO: Add local sequence-neighborhood classification beyond simple separation.
    """
    candidates: list[BandCandidatePair] = []
    for idx_a, plane_a in enumerate(planes):
        for idx_b in range(idx_a + 1, len(planes)):
            plane_b = planes[idx_b]
            pair_distance = distance(plane_a.center, plane_b.center)
            error_from_target = pair_distance - target_distance
            if abs(error_from_target) <= tolerance:
                same_chain = plane_a.chain == plane_b.chain
                sequence_separation = abs(plane_b.res_i - plane_a.res_i) if same_chain else None
                candidates.append(
                    BandCandidatePair(
                        band_name=band_name,
                        target_distance=target_distance,
                        tolerance=tolerance,
                        distance=pair_distance,
                        error_from_target=error_from_target,
                        feature_type="plane_center_to_plane_center",
                        plane_index_a=idx_a,
                        plane_index_b=idx_b,
                        chain_a=plane_a.chain,
                        chain_b=plane_b.chain,
                        res_i_a=plane_a.res_i,
                        res_j_a=plane_a.res_j,
                        res_i_b=plane_b.res_i,
                        res_j_b=plane_b.res_j,
                        sequence_separation=sequence_separation,
                        same_chain=same_chain,
                    )
                )
    return candidates
