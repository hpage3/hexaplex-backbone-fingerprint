"""Map peptide-plane features to candidate C/D band distances."""

from __future__ import annotations

from dataclasses import dataclass

from .geometry import distance
from .peptide_planes import PeptidePlane


@dataclass(frozen=True)
class BandCandidatePair:
    """A pair of peptide-plane features whose distance is near a target band."""

    target_distance: float
    tolerance: float
    distance: float
    delta: float
    feature_type: str
    plane_i: int
    plane_j: int
    chain_i: str
    chain_j: str
    res_i: int
    res_j: int
    partner_res_i: int
    partner_res_j: int


def find_band_candidate_pairs(
    planes: list[PeptidePlane],
    target_distance: float,
    tolerance: float,
) -> list[BandCandidatePair]:
    """Find plane-center to plane-center pairs near a target distance.

    TODO: Add O-O, C-O, C/N/O subplane-center features.
    TODO: Add same-chain versus cross-chain classification.
    TODO: Add local sequence-separation classification.
    """
    candidates: list[BandCandidatePair] = []
    for idx_i, plane_i in enumerate(planes):
        for idx_j in range(idx_i + 1, len(planes)):
            plane_j = planes[idx_j]
            pair_distance = distance(plane_i.center, plane_j.center)
            delta = pair_distance - target_distance
            if abs(delta) <= tolerance:
                candidates.append(
                    BandCandidatePair(
                        target_distance=target_distance,
                        tolerance=tolerance,
                        distance=pair_distance,
                        delta=delta,
                        feature_type="plane_center_to_plane_center",
                        plane_i=idx_i,
                        plane_j=idx_j,
                        chain_i=plane_i.chain,
                        chain_j=plane_j.chain,
                        res_i=plane_i.res_i,
                        res_j=plane_i.res_j,
                        partner_res_i=plane_j.res_i,
                        partner_res_j=plane_j.res_j,
                    )
                )
    return candidates
