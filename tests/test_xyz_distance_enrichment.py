from pathlib import Path

import pytest

from hexaplex_backbone_fingerprint.xyz_distance_enrichment import (
    count_distance_band_pairs,
    infer_strand_count_from_filename,
)
from hexaplex_backbone_fingerprint.xyz_parser import parse_xyz


def test_parse_xyz_reads_atom_count_and_records():
    atoms = parse_xyz("tests/fixtures/mini_points.xyz")

    assert len(atoms) == 4
    assert atoms[0].atom_index == 1
    assert atoms[0].element == "C"
    assert atoms[2].element == "O"


def test_parse_xyz_malformed_atom_count_raises():
    with pytest.raises(ValueError, match="declared 3 atoms but parsed 2"):
        parse_xyz("tests/fixtures/bad_atom_count.xyz")


def test_distance_band_counting_finds_expected_pairs():
    atoms = parse_xyz("tests/fixtures/mini_points.xyz")

    stats = count_distance_band_pairs(atoms, target=1.0, tolerance=0.01)

    assert stats["candidate_pair_count"] == 1
    assert stats["total_possible_pairs"] == 3
    assert stats["normalized_count"] == pytest.approx(1 / 3)
    assert stats["median_distance"] == pytest.approx(1.0)
    assert stats["median_abs_error"] == pytest.approx(0.0)


def test_infer_strand_count_from_filename():
    assert infer_strand_count_from_filename("TetraplexWithCOOHTemp.xyz") == 4
    assert infer_strand_count_from_filename("unknown.xyz") is None
