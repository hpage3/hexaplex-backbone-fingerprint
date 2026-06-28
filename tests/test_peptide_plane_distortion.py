import numpy as np

from hexaplex_backbone_fingerprint.pdb_parser import parse_pdb
from hexaplex_backbone_fingerprint.peptide_planes import build_peptide_planes


def test_cno_normal_angle_is_near_zero_for_planar_fixture():
    resmap = parse_pdb("tests/fixtures/mini_peptide.pdb")
    planes = build_peptide_planes(resmap)

    assert planes
    assert np.isclose(planes[0].cno_to_peptide_normal_angle_deg, 0.0)
    assert np.isclose(planes[0].cno_centroid_to_peptide_plane_signed_dist, 0.0)
