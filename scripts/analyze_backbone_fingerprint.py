"""Analyze peptide-plane backbone fingerprints for one PDB model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hexaplex_backbone_fingerprint.band_mapping import find_band_candidate_pairs
from hexaplex_backbone_fingerprint.io import (
    write_band_candidate_pairs_csv,
    write_plane_features_csv,
    write_summary_markdown,
)
from hexaplex_backbone_fingerprint.pdb_parser import parse_pdb
from hexaplex_backbone_fingerprint.peptide_planes import build_peptide_planes


def main() -> int:
    """Run the command-line analysis."""
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    resmap = parse_pdb(args.pdb_file)
    planes = build_peptide_planes(resmap)
    c_candidates = find_band_candidate_pairs(planes, args.c_target, args.tol)
    d_candidates = find_band_candidate_pairs(planes, args.d_target, args.tol)

    write_plane_features_csv(planes, outdir / "plane_features.csv")
    write_band_candidate_pairs_csv(c_candidates + d_candidates, outdir / "band_candidate_pairs.csv")
    write_summary_markdown(
        outdir / "summary.md",
        input_file=args.pdb_file,
        plane_count=len(planes),
        c_candidates=c_candidates,
        d_candidates=d_candidates,
    )
    return 0


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdb_file", help="Input PDB coordinate file.")
    parser.add_argument("--outdir", default="outputs/model_test", help="Directory for generated outputs.")
    parser.add_argument("--c-target", type=float, default=5.6, help="C band target distance in Angstrom.")
    parser.add_argument("--d-target", type=float, default=7.3, help="D band target distance in Angstrom.")
    parser.add_argument("--tol", type=float, default=0.25, help="Distance tolerance in Angstrom.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
