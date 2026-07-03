"""Generate and score reconstructed rise/radius bridge models.

These are Option B reconstructed bridge models: they reuse the repository's
parametric six-strand peptide-plane source logic, but they are not exact
recovered pNAB-regenerated parent structures.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hexaplex_backbone_fingerprint.parametric_peptide_plane_models import (
    ModelParameters,
    generate_model_atoms,
    manifest_row,
    write_pdb,
    write_xyz,
)
from hexaplex_backbone_fingerprint.parametric_powder_scan import (
    debye_profile,
    load_xyz_coordinates,
    make_q_grid,
    nearest_peak,
)


DEFAULT_PARENT_PDB = Path("outputs/coordinates/ideal_hexaflex_variants/ideal_hexaflex_backbone_plus_carboxylate.pdb")
DEFAULT_OUTDIR = Path("outputs/coordinates/reconstructed_rise_radius_bridge")
DEFAULT_SCORE_CSV = Path("outputs/metrics/reconstructed_rise_radius_bridge_abcd_scores.csv")
DEFAULT_REPORT = Path("outputs/reports/reconstructed_rise_radius_bridge_summary.md")

REQUESTED_RISES_A = [3.40, 3.38, 3.35]
TARGETS_A = {
    "A": 7.9,
    "B": 6.5,
    "C": 5.6,
    "D": 7.3,
}
TOLERANCE_A = 0.20


@dataclass(frozen=True)
class BridgeCandidate:
    """One requested reconstructed bridge candidate."""

    variant_id: str
    requested_rise_A: float
    params: ModelParameters
    pdb_path: Path
    xyz_path: Path


def format_rise_variant_id(rise_A: float) -> str:
    """Return a stable variant ID for a requested rise value."""
    text = f"{rise_A:.2f}".replace(".", "p").replace("-", "m")
    return f"reconstructed_rise_{text}"


def bridge_coordinate_path(outdir: Path, variant_id: str, suffix: str = ".pdb") -> Path:
    """Return a candidate coordinate path under the bridge output directory."""
    return outdir / f"{variant_id}{suffix}"


def parse_parent_ca_coordinates(parent_pdb: Path) -> np.ndarray:
    """Parse parent C-alpha coordinates from a PDB file."""
    coords: list[list[float]] = []
    for line in parent_pdb.read_text(encoding="ascii", errors="replace").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        if line[12:16].strip() != "CA":
            continue
        coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    if not coords:
        raise ValueError(f"No C-alpha atoms found in parent PDB: {parent_pdb}")
    return np.asarray(coords, dtype=float)


def derive_parent_mean_ca_radius(parent_pdb: Path) -> float:
    """Derive mean C-alpha xy radius around the parent C-alpha xy centroid."""
    coords = parse_parent_ca_coordinates(parent_pdb)
    center_xy = coords[:, :2].mean(axis=0)
    radii = np.linalg.norm(coords[:, :2] - center_xy, axis=1)
    return float(np.mean(radii))


def derive_repeats_per_strand(parent_pdb: Path, fallback: int = 15) -> int:
    """Infer a conservative repeat count from parent C-alpha count per chain."""
    counts: dict[str, int] = {}
    for line in parent_pdb.read_text(encoding="ascii", errors="replace").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or line[12:16].strip() != "CA":
            continue
        chain = line[21:22].strip()
        counts[chain] = counts.get(chain, 0) + 1
    if not counts:
        return fallback
    median_ca = int(np.median(list(counts.values())))
    return max(1, median_ca // 2)


def controllable_parameters() -> dict[str, str]:
    """Return parameters controlled by reusable source logic."""
    return {
        "rise": "direct ModelParameters.rise_A",
        "radius": "direct ModelParameters.helix_radius_A",
        "twist": "direct ModelParameters.twist_deg",
        "strand_orientation": "plane_normal_to_axis_deg, plane_azimuth_deg, in_plane_spin_deg, handedness",
        "register": "uniform_adjacent_z_offset_A, alternating_z_offset_A, z_offset_mode",
        "chain_count": "direct ModelParameters.n_strands",
    }


def build_candidate(
    rise_A: float,
    outdir: Path,
    radius_A: float,
    twist_deg: float,
    repeats_per_strand: int,
    n_strands: int = 6,
) -> BridgeCandidate:
    """Build a candidate specification without writing coordinates."""
    variant_id = format_rise_variant_id(rise_A)
    params = ModelParameters(
        n_strands=n_strands,
        repeats_per_strand=repeats_per_strand,
        helix_radius_A=radius_A,
        twist_deg=twist_deg,
        rise_A=rise_A,
        plane_normal_to_axis_deg=60.0,
        plane_azimuth_deg=0.0,
        in_plane_spin_deg=0.0,
        handedness="right",
        z_offset_mode="uniform_adjacent",
        uniform_adjacent_z_offset_A=0.0,
    )
    return BridgeCandidate(
        variant_id=variant_id,
        requested_rise_A=rise_A,
        params=params,
        pdb_path=bridge_coordinate_path(outdir, variant_id, ".pdb"),
        xyz_path=bridge_coordinate_path(outdir, variant_id, ".xyz"),
    )


def write_candidate(candidate: BridgeCandidate) -> dict[str, object]:
    """Write one bridge candidate PDB/XYZ and return manifest-like metadata."""
    atoms = generate_model_atoms(candidate.params)
    write_pdb(atoms, candidate.pdb_path, candidate.params)
    write_xyz(atoms, candidate.xyz_path, comment=candidate.variant_id)
    row = manifest_row(candidate.params, candidate.pdb_path, candidate.xyz_path, len(atoms))
    row["variant_id"] = candidate.variant_id
    row["requested_rise_A"] = candidate.requested_rise_A
    return row


def score_xyz_abcd(
    xyz_path: Path,
    targets_A: dict[str, float] = TARGETS_A,
    tolerance_A: float = TOLERANCE_A,
    q_step: float = 0.01,
    d_min: float = 2.5,
    d_max: float = 12.0,
) -> dict[str, object]:
    """Score one XYZ model against A/B/C/D targets using existing Debye helpers."""
    coords = load_xyz_coordinates(xyz_path, exclude_hydrogen=False)
    q_values = make_q_grid(d_min_A=d_min, d_max_A=d_max, q_step=q_step)
    profile = debye_profile(coords, q_values)
    row: dict[str, object] = {}
    for band, target in targets_A.items():
        hit = nearest_peak(profile, target, tolerance_A)
        row[f"observed_{band}_d_A"] = hit.peak_d_A
        row[f"{band}_error_A"] = hit.error_A
        row[f"{band}_score"] = hit.intensity
        row[f"{band}_found_within_tolerance"] = hit.found_within_tolerance
    return row


def required_score_columns() -> list[str]:
    """Return required bridge score CSV columns."""
    return [
        "variant_id",
        "requested_rise_A",
        "realized_rise_A",
        "radius_parameter_A",
        "twist_parameter_deg",
        "source_logic",
        "coordinate_path",
        "observed_A_d_A",
        "observed_B_d_A",
        "observed_C_d_A",
        "observed_D_d_A",
        "A_error_A",
        "B_error_A",
        "C_error_A",
        "D_error_A",
        "combined_CD_abs_error_A",
        "combined_ABCD_abs_error_A",
        "status",
        "notes",
    ]


def bridge_summary_row(candidate: BridgeCandidate, manifest: dict[str, object], scores: dict[str, object]) -> dict[str, object]:
    """Build one output row for bridge scoring."""
    errors = {band: float(scores[f"{band}_error_A"]) for band in TARGETS_A}
    row = {
        "variant_id": candidate.variant_id,
        "requested_rise_A": candidate.requested_rise_A,
        "realized_rise_A": manifest["rise_A"],
        "radius_parameter_A": manifest["helix_radius_A"],
        "twist_parameter_deg": manifest["twist_deg"],
        "source_logic": "hexaplex_backbone_fingerprint.parametric_peptide_plane_models.ModelParameters",
        "coordinate_path": str(candidate.pdb_path),
        "combined_CD_abs_error_A": abs(errors["C"]) + abs(errors["D"]),
        "combined_ABCD_abs_error_A": sum(abs(value) for value in errors.values()),
        "status": "scored",
        "notes": "Option B reconstructed bridge model; not exact recovered pNAB provenance.",
    }
    row.update(scores)
    return row


def bridge_recommendation(scores: pd.DataFrame) -> str:
    """Classify bridge result based on C/D peak behavior."""
    if scores.empty:
        return "blocked"
    best = scores.loc[pd.to_numeric(scores["combined_CD_abs_error_A"], errors="coerce").idxmin()]
    best_cd = float(best["combined_CD_abs_error_A"])
    best_c_error = abs(float(best["C_error_A"]))
    best_d_error = abs(float(best["D_error_A"]))
    if best_c_error <= 0.20 and best_d_error <= 0.20 and best_cd <= 0.15:
        return "success"
    if best_c_error < 0.5 and best_d_error <= 0.35:
        return "partial"
    return "failure"


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    """Render selected columns as a compact markdown table."""
    if df.empty:
        return "_None._"
    columns = [column for column in columns if column in df.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for record in df[columns].itertuples(index=False):
        values = [f"{value:.4g}" if isinstance(value, float) else str(value) for value in record]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_report_text(scores: pd.DataFrame, parent_pdb: Path, radius_A: float, repeats_per_strand: int) -> str:
    """Build markdown report for reconstructed bridge workflow."""
    controls = controllable_parameters()
    controls_text = "\n".join(f"- {key}: {value}" for key, value in controls.items())
    recommendation = bridge_recommendation(scores)
    if scores.empty:
        best_text = "No candidates were scored."
        table = "_None._"
    else:
        best = scores.loc[pd.to_numeric(scores["combined_CD_abs_error_A"], errors="coerce").idxmin()]
        best_text = (
            f"Best by C/D error: `{best['variant_id']}` with C {float(best['observed_C_d_A']):.4f} A, "
            f"D {float(best['observed_D_d_A']):.4f} A, combined C/D error {float(best['combined_CD_abs_error_A']):.4f} A."
        )
        table = markdown_table(
            scores,
            [
                "variant_id",
                "requested_rise_A",
                "realized_rise_A",
                "radius_parameter_A",
                "twist_parameter_deg",
                "observed_A_d_A",
                "observed_B_d_A",
                "observed_C_d_A",
                "observed_D_d_A",
                "combined_CD_abs_error_A",
                "combined_ABCD_abs_error_A",
            ],
        )
    recommendation_text = {
        "success": "Bridge success: reconstructed 3.35/3.38/3.40 family reproduces C/D improvement while keeping D stable.",
        "partial": "Bridge partial: trend is directionally consistent or C/D is moderately close, but not as strong as the diagnostic compression.",
        "failure": "Bridge failure: reconstructed family does not reproduce the diagnostic C/D improvement.",
        "blocked": "Blocked: reusable source logic cannot generate a meaningful rise/radius family yet.",
    }[recommendation]
    return f"""# Reconstructed Rise/Radius Bridge Summary

## Purpose

This workflow tests whether the current diagnostic rise-like C/D improvement can be approximated using reusable source logic with explicit rise/radius parameters.

## Option B Provenance Caution

The source-parameter audit recommended Option B: partial provenance/reusable parameterized generators were found, but exact original pNAB provenance was not recovered. These are reconstructed bridge models, not exact pNAB-regenerated structures and not minimized physical structures.

## Source Logic Used

- Generator: `hexaplex_backbone_fingerprint.parametric_peptide_plane_models`
- Parent radius reference: `{parent_pdb}`
- Parent-derived mean C-alpha radius used as default radius: {radius_A:.4f} A
- Repeats per strand inferred from parent C-alpha count: {repeats_per_strand}

## Controllable Parameters

{controls_text}

## Generated Candidate Set

Exact requested rise set: 3.40 A, 3.38 A, 3.35 A.

{table}

## A/B/C/D Scoring Summary

Targets used by the bridge wrapper: A {TARGETS_A['A']:.2f} A, B {TARGETS_A['B']:.2f} A, C {TARGETS_A['C']:.2f} A, D {TARGETS_A['D']:.2f} A. Scoring uses the existing direct point-scatterer Debye profile and nearest-peak helper.

{best_text}

## Recommendation

{recommendation_text}

Preserve the distinction between this reconstructed bridge and the coordinate-derived diagnostic `parameterized_rise_0p9750`. If this bridge is partial or failed, the next step is to recover source pNAB/YAML settings or derive a chemically/register-defined generator from the parent coordinates.
"""


def run_bridge(
    parent_pdb: Path = DEFAULT_PARENT_PDB,
    outdir: Path = DEFAULT_OUTDIR,
    score_csv: Path = DEFAULT_SCORE_CSV,
    report_path: Path = DEFAULT_REPORT,
    requested_rises_A: list[float] | None = None,
    radius_A: float | None = None,
    twist_deg: float = 30.0,
) -> pd.DataFrame:
    """Generate and score the reconstructed rise/radius bridge family."""
    requested_rises_A = REQUESTED_RISES_A if requested_rises_A is None else requested_rises_A
    if radius_A is None:
        radius_A = derive_parent_mean_ca_radius(parent_pdb)
    repeats_per_strand = derive_repeats_per_strand(parent_pdb)
    outdir.mkdir(parents=True, exist_ok=True)
    score_csv.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for rise_A in requested_rises_A:
        candidate = build_candidate(rise_A, outdir, radius_A, twist_deg, repeats_per_strand)
        manifest = write_candidate(candidate)
        scores = score_xyz_abcd(candidate.xyz_path)
        rows.append(bridge_summary_row(candidate, manifest, scores))

    scores_df = pd.DataFrame(rows)
    scores_df = scores_df.reindex(columns=required_score_columns() + [column for column in scores_df.columns if column not in required_score_columns()])
    scores_df.to_csv(score_csv, index=False)
    report_path.write_text(build_report_text(scores_df, parent_pdb, radius_A, repeats_per_strand), encoding="utf-8")
    return scores_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-pdb", type=Path, default=DEFAULT_PARENT_PDB)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--score-csv", type=Path, default=DEFAULT_SCORE_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--radius-A", type=float, default=None)
    parser.add_argument("--twist-deg", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scores = run_bridge(args.parent_pdb, args.outdir, args.score_csv, args.report, radius_A=args.radius_A, twist_deg=args.twist_deg)
    print(f"Generated and scored {len(scores)} reconstructed bridge candidates")
    print(f"CSV: {args.score_csv}")
    print(f"Report: {args.report}")
    print(f"Recommendation: {bridge_recommendation(scores)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
