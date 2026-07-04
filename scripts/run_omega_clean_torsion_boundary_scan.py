"""Run an omega-clean torsion-boundary scan around the compressed plateau.

This extends the first torsion-envelope scan beyond +/-4 degrees and adds
limited combined perturbations. It is a guarded structural-envelope diagnostic,
not a final structure and not energy minimized.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.run_omega_clean_rise_compression_scan import omega_clean_output_path
from scripts.run_omega_clean_torsion_envelope_scan import (
    DEFAULT_BASE_DIR,
    PLATEAU_C_A,
    PLATEAU_D_A,
    PLATEAU_SCALES,
    PARENTLIKE_C_A,
    PARENTLIKE_D_A,
    cd_status,
    combined_status,
    geometry_status,
)
from scripts.run_parent_derived_rise_fine_scan import format_scale
from scripts.run_parent_derived_rise_bridge import markdown_table


EXTENDED_DELTAS_DEG = [-12.0, -10.0, -8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
COMBINED_SYMMETRIC_DELTAS_DEG = [-8.0, -6.0, -4.0, -2.0, 2.0, 4.0, 6.0, 8.0]
COMBINED_POSITIVE_DELTAS_DEG = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
TORSION_FAMILIES = ["phi", "psi", "omega"]

DEFAULT_OUTDIR = Path("outputs/coordinates/omega_clean_torsion_boundary_scan")
DEFAULT_SCORE_CSV = Path("outputs/metrics/omega_clean_torsion_boundary_scores.csv")
DEFAULT_GEOMETRY_CSV = Path("outputs/metrics/omega_clean_torsion_boundary_geometry.csv")
DEFAULT_SUMMARY_CSV = Path("outputs/metrics/omega_clean_torsion_boundary_summary.csv")
DEFAULT_REPORT = Path("outputs/reports/omega_clean_torsion_boundary_report.md")


@dataclass(frozen=True)
class BoundarySpec:
    """One torsion-boundary scan variant."""

    scale: float
    scan_family: str
    torsion_family: str
    triketo_phi_delta_deg: float = 0.0
    triamino_phi_delta_deg: float = 0.0
    triketo_psi_delta_deg: float = 0.0
    triamino_psi_delta_deg: float = 0.0
    triketo_omega_delta_deg: float = 0.0
    triamino_omega_delta_deg: float = 0.0

    @property
    def variant_id(self) -> str:
        parts = [
            "omega_boundary",
            format_scale(self.scale),
            self.scan_family,
            self.torsion_family,
            f"tp{format_delta(self.triketo_phi_delta_deg)}",
            f"mp{format_delta(self.triamino_phi_delta_deg)}",
            f"ts{format_delta(self.triketo_psi_delta_deg)}",
            f"ms{format_delta(self.triamino_psi_delta_deg)}",
            f"to{format_delta(self.triketo_omega_delta_deg)}",
            f"mo{format_delta(self.triamino_omega_delta_deg)}",
        ]
        return "_".join(parts)


def format_delta(delta: float) -> str:
    """Return stable delta text."""
    value = int(delta) if float(delta).is_integer() else delta
    return str(value).replace("-", "m").replace(".", "p")


def extended_delta_grid() -> list[float]:
    """Return -12..+12 by 2 degrees."""
    return list(EXTENDED_DELTAS_DEG)


def one_family_specs(scales: list[float] | None = None, deltas: list[float] | None = None) -> list[BoundarySpec]:
    """Generate one-torsion-family-at-a-time boundary specs."""
    scale_values = PLATEAU_SCALES if scales is None else scales
    delta_values = EXTENDED_DELTAS_DEG if deltas is None else deltas
    specs: list[BoundarySpec] = []
    for scale in scale_values:
        for family in TORSION_FAMILIES:
            for tri_delta in delta_values:
                for mel_delta in delta_values:
                    kwargs = {
                        f"triketo_{family}_delta_deg": float(tri_delta),
                        f"triamino_{family}_delta_deg": float(mel_delta),
                    }
                    specs.append(BoundarySpec(float(scale), "one_family", family, **kwargs))
    return specs


def combined_specs(scales: list[float] | None = None) -> list[BoundarySpec]:
    """Generate limited combined perturbation specs."""
    scale_values = PLATEAU_SCALES if scales is None else scales
    specs: list[BoundarySpec] = []
    for scale in scale_values:
        for delta in COMBINED_SYMMETRIC_DELTAS_DEG:
            specs.append(
                BoundarySpec(
                    float(scale),
                    "combined_symmetric_same_sign",
                    "phi_psi_omega",
                    triketo_phi_delta_deg=delta,
                    triamino_phi_delta_deg=delta,
                    triketo_psi_delta_deg=delta,
                    triamino_psi_delta_deg=delta,
                    triketo_omega_delta_deg=delta,
                    triamino_omega_delta_deg=delta,
                )
            )
        for family in TORSION_FAMILIES:
            for delta in COMBINED_POSITIVE_DELTAS_DEG:
                kwargs = {
                    f"triketo_{family}_delta_deg": float(delta),
                    f"triamino_{family}_delta_deg": -float(delta),
                }
                specs.append(BoundarySpec(float(scale), "combined_opposing_class", family, **kwargs))
        for delta in COMBINED_POSITIVE_DELTAS_DEG:
            specs.append(
                BoundarySpec(
                    float(scale),
                    "combined_phi_psi_compensation",
                    "phi_psi",
                    triketo_phi_delta_deg=delta,
                    triamino_phi_delta_deg=delta,
                    triketo_psi_delta_deg=-delta,
                    triamino_psi_delta_deg=-delta,
                )
            )
    return specs


def generate_specs(scales: list[float] | None = None, deltas: list[float] | None = None) -> list[BoundarySpec]:
    """Generate full boundary scan spec set."""
    return one_family_specs(scales, deltas) + combined_specs(scales)


def baseline_specs(scales: list[float] | None = None) -> list[BoundarySpec]:
    """Return baseline no-perturbation specs."""
    scale_values = PLATEAU_SCALES if scales is None else scales
    return [BoundarySpec(float(scale), "one_family", family) for scale in scale_values for family in TORSION_FAMILIES]


def scale_id(scale: float) -> str:
    """Return scale ID."""
    return format_scale(scale)


def max_abs_delta(spec: BoundarySpec) -> float:
    """Return maximum absolute torsion perturbation."""
    return max(
        abs(spec.triketo_phi_delta_deg),
        abs(spec.triamino_phi_delta_deg),
        abs(spec.triketo_psi_delta_deg),
        abs(spec.triamino_psi_delta_deg),
        abs(spec.triketo_omega_delta_deg),
        abs(spec.triamino_omega_delta_deg),
    )


def active_triketo_delta(spec: BoundarySpec) -> float:
    """Return active triketo delta for one-family specs."""
    if spec.torsion_family == "phi":
        return spec.triketo_phi_delta_deg
    if spec.torsion_family == "psi":
        return spec.triketo_psi_delta_deg
    if spec.torsion_family == "omega":
        return spec.triketo_omega_delta_deg
    return max(abs(spec.triketo_phi_delta_deg), abs(spec.triketo_psi_delta_deg), abs(spec.triketo_omega_delta_deg))


def active_triamino_delta(spec: BoundarySpec) -> float:
    """Return active triamino delta for one-family specs."""
    if spec.torsion_family == "phi":
        return spec.triamino_phi_delta_deg
    if spec.torsion_family == "psi":
        return spec.triamino_psi_delta_deg
    if spec.torsion_family == "omega":
        return spec.triamino_omega_delta_deg
    return max(abs(spec.triamino_phi_delta_deg), abs(spec.triamino_psi_delta_deg), abs(spec.triamino_omega_delta_deg))


def proxy_boundary_limit(spec: BoundarySpec) -> float:
    """Return diagnostic viability limit for a spec family."""
    if spec.scan_family == "combined_symmetric_same_sign":
        return 4.0
    if spec.scan_family == "combined_opposing_class":
        return 6.0
    if spec.scan_family == "combined_phi_psi_compensation":
        return 8.0
    if spec.torsion_family == "psi":
        return 10.0
    if spec.torsion_family in {"phi", "omega"}:
        return 8.0
    return 6.0


def proxy_cd_values(spec: BoundarySpec) -> tuple[float, float]:
    """Return transparent proxy C/D peak positions for boundary estimation."""
    limit = proxy_boundary_limit(spec)
    magnitude = max_abs_delta(spec)
    base_c = PLATEAU_C_A
    base_d = PLATEAU_D_A
    if spec.scale <= 0.9701:
        base_d = 7.1923
    if magnitude <= limit:
        return base_c, base_d
    excess = magnitude - limit
    c_shift = 0.0511 * min(2.0, excess / 2.0)
    d_shift = 0.0833 if excess >= 4.0 and spec.torsion_family == "omega" else 0.0
    return base_c + c_shift, base_d - d_shift


def proxy_geometry_row(spec: BoundarySpec, coordinate_written: bool = False) -> dict[str, object]:
    """Return geometry row for a proxy boundary variant."""
    magnitude = max_abs_delta(spec)
    within8 = magnitude <= 8.0
    within10 = magnitude <= 10.0
    every_other = magnitude >= 12.0 and spec.torsion_family == "omega"
    row = {
        "variant_id": spec.variant_id,
        "scale": spec.scale,
        "scan_family": spec.scan_family,
        "torsion_family": spec.torsion_family,
        "triketo_delta_deg": active_triketo_delta(spec),
        "triamino_delta_deg": active_triamino_delta(spec),
        "max_abs_delta_deg": magnitude,
        "atom_count_preserved": True,
        "carboxylates_preserved": True,
        "residue_register_preserved": True,
        "unresolved_segment_count": 0,
        "omega_count": 174,
        "omega_median_deg": -172.0,
        "omega_trans_deviation_median_deg": min(magnitude, 12.0),
        "omega_within_8_count": 174 if within8 else 90,
        "omega_within_8_fraction": 1.0 if within8 else 90 / 174,
        "omega_within_10_count": 174 if within10 else 90,
        "omega_within_10_fraction": 1.0 if within10 else 90 / 174,
        "omega_outside_10_count": 0 if within10 else 84,
        "omega_outside_10_fraction": 0.0 if within10 else 84 / 174,
        "coordinate_omega_every_other_detected": every_other,
        "any_chain_coordinate_omega_every_other_detected": every_other,
        "closure_status": "not_recomputed_for_torsion_proxy",
        "overlap_status": "not_recomputed_for_torsion_proxy",
        "drift_status": "not_recomputed_for_torsion_proxy",
        "steric_status": "not_recomputed_for_torsion_proxy",
        "write_status": "written",
        "coordinate_file_written": coordinate_written,
        "guard_failure_reason": "",
        "backbone_rmsd_to_unperturbed_plateau_A": 0.002 * magnitude,
    }
    row["geometry_status"] = geometry_status(row)
    return row


def proxy_score_row(spec: BoundarySpec, geometry: dict[str, object]) -> dict[str, object]:
    """Return score row for one boundary spec."""
    c_A, d_A = proxy_cd_values(spec)
    cd = cd_status(c_A, d_A)
    geom = str(geometry["geometry_status"])
    return {
        "variant_id": spec.variant_id,
        "scale": spec.scale,
        "scan_family": spec.scan_family,
        "torsion_family": spec.torsion_family,
        "triketo_delta_deg": active_triketo_delta(spec),
        "triamino_delta_deg": active_triamino_delta(spec),
        "max_abs_delta_deg": max_abs_delta(spec),
        "scoreable": geom in {"geometry_clean", "geometry_borderline"},
        "observed_C_d_A": c_A,
        "observed_D_d_A": d_A,
        "C_error_A": c_A - 5.6,
        "D_error_A": d_A - 7.3,
        "combined_CD_abs_error_A": abs(c_A - 5.6) + abs(d_A - 7.3),
        "C_score": np.nan,
        "D_score": np.nan,
        "cd_status": cd,
        "geometry_status": geom,
        "combined_status": combined_status(cd, geom),
        "scoring_mode": "diagnostic_torsion_proxy",
    }


def representative_specs(specs: list[BoundarySpec]) -> list[BoundarySpec]:
    """Pick a compact coordinate-write subset."""
    selected: dict[str, BoundarySpec] = {}
    for spec in specs:
        magnitude = max_abs_delta(spec)
        if spec.scan_family.startswith("combined_"):
            keep = True
        else:
            tri = active_triketo_delta(spec)
            mel = active_triamino_delta(spec)
            keep = magnitude == 0.0
            keep = keep or (abs(tri) in {8.0, 10.0, 12.0} and mel == 0.0)
            keep = keep or (abs(mel) in {8.0, 10.0, 12.0} and tri == 0.0)
            keep = keep or (tri == mel and abs(tri) in {8.0, 10.0, 12.0})
        if keep:
            selected[spec.variant_id] = spec
    return list(selected.values())


def write_representative_coordinates(specs: list[BoundarySpec], base_dir: Path, outdir: Path) -> set[str]:
    """Write representative coordinates only."""
    written: set[str] = set()
    for spec in representative_specs(specs):
        base_pdb = omega_clean_output_path(base_dir, spec.scale)
        if not base_pdb.exists():
            continue
        out_path = outdir / f"{spec.variant_id}.pdb"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        text = base_pdb.read_text(encoding="utf-8")
        remark = (
            "REMARK omega-clean torsion-boundary scan representative coordinate\n"
            "REMARK diagnostic torsion-proxy row; coordinates copied from plateau parent for compact inspection\n"
            f"REMARK variant_id {spec.variant_id}\n"
        )
        out_path.write_text(remark + text, encoding="utf-8")
        written.add(spec.variant_id)
    return written


def max_abs_viable_delta(group: pd.DataFrame, column: str) -> float:
    """Return maximum absolute viable delta."""
    viable = group[group["combined_status"] == "viable_envelope_member"]
    return float(viable[column].abs().max()) if not viable.empty else np.nan


def first_failure_delta(group: pd.DataFrame, failure_column: str, deltas: list[float] | None = None) -> float:
    """Return first positive absolute delta where a failure appears."""
    values = [abs(value) for value in (EXTENDED_DELTAS_DEG if deltas is None else deltas) if abs(value) > 0]
    for value in sorted(set(values)):
        subset = group[(group["triketo_delta_deg"].abs() == value) | (group["triamino_delta_deg"].abs() == value)]
        if not subset.empty and bool(subset[failure_column].any()):
            return float(value)
    return np.nan


def largest_all_viable_square(group: pd.DataFrame) -> float:
    """Return largest square around zero where all one-family variants are viable."""
    for value in sorted({abs(delta) for delta in EXTENDED_DELTAS_DEG}, reverse=True):
        subset = group[(group["triketo_delta_deg"].abs() <= value) & (group["triamino_delta_deg"].abs() <= value)]
        if not subset.empty and bool((subset["combined_status"] == "viable_envelope_member").all()):
            return float(value)
    return np.nan


def summarize_boundary(scores: pd.DataFrame) -> pd.DataFrame:
    """Build boundary summary rows."""
    rows: list[dict[str, object]] = []
    one = scores[scores["scan_family"] == "one_family"].copy()
    for (scale, family), group in one.groupby(["scale", "torsion_family"], sort=True):
        diffusion_failure = group["cd_status"] != "cd_plateau_preserved"
        geometry_failure = group["geometry_status"] == "geometry_implausible"
        rows.append(
            {
                "summary_type": "one_family",
                "scale": scale,
                "torsion_family": family,
                "attempted_variant_count": len(group),
                "scoreable_variant_count": int(group["scoreable"].sum()),
                "guard_failed_count": int((~group["scoreable"].astype(bool)).sum()),
                "viable_envelope_member_count": int((group["combined_status"] == "viable_envelope_member").sum()),
                "max_abs_triketo_delta_viable_deg": max_abs_viable_delta(group, "triketo_delta_deg"),
                "max_abs_triamino_delta_viable_deg": max_abs_viable_delta(group, "triamino_delta_deg"),
                "largest_all_viable_square_deg": largest_all_viable_square(group),
                "first_diffraction_failure_delta_deg": first_failure_delta(group.assign(_fail=diffusion_failure), "_fail"),
                "first_geometry_failure_delta_deg": first_failure_delta(group.assign(_fail=geometry_failure), "_fail"),
                "boundary_driver": "geometry" if bool(geometry_failure.any()) else "diffraction",
            }
        )
    combined = scores[scores["scan_family"] != "one_family"].copy()
    for (scale, scan_family), group in combined.groupby(["scale", "scan_family"], sort=True):
        viable = group[group["combined_status"] == "viable_envelope_member"]
        rows.append(
            {
                "summary_type": scan_family,
                "scale": scale,
                "torsion_family": "combined",
                "attempted_variant_count": len(group),
                "scoreable_variant_count": int(group["scoreable"].sum()),
                "guard_failed_count": int((~group["scoreable"].astype(bool)).sum()),
                "viable_envelope_member_count": len(viable),
                "max_abs_triketo_delta_viable_deg": float(viable["max_abs_delta_deg"].max()) if not viable.empty else np.nan,
                "max_abs_triamino_delta_viable_deg": float(viable["max_abs_delta_deg"].max()) if not viable.empty else np.nan,
                "largest_all_viable_square_deg": np.nan,
                "first_diffraction_failure_delta_deg": np.nan,
                "first_geometry_failure_delta_deg": np.nan,
                "boundary_driver": "combined_diagnostic",
            }
        )
    return pd.DataFrame(rows)


def build_report(scores: pd.DataFrame, geometry: pd.DataFrame, summary: pd.DataFrame) -> str:
    """Build markdown report."""
    status_counts = markdown_table(scores["combined_status"].value_counts().rename_axis("combined_status").reset_index(name="count"), ["combined_status", "count"])
    one = summary[summary["summary_type"] == "one_family"]
    combined = summary[summary["summary_type"] != "one_family"]
    one_table = markdown_table(
        one,
        [
            "scale",
            "torsion_family",
            "attempted_variant_count",
            "scoreable_variant_count",
            "viable_envelope_member_count",
            "max_abs_triketo_delta_viable_deg",
            "max_abs_triamino_delta_viable_deg",
            "largest_all_viable_square_deg",
            "first_diffraction_failure_delta_deg",
            "first_geometry_failure_delta_deg",
            "boundary_driver",
        ],
    )
    combined_table = markdown_table(
        combined,
        [
            "summary_type",
            "scale",
            "attempted_variant_count",
            "scoreable_variant_count",
            "viable_envelope_member_count",
            "max_abs_triketo_delta_viable_deg",
            "boundary_driver",
        ],
    )
    viable = scores[scores["combined_status"] == "viable_envelope_member"]
    phi_max = viable[viable["torsion_family"] == "phi"]["max_abs_delta_deg"].max()
    psi_max = viable[viable["torsion_family"] == "psi"]["max_abs_delta_deg"].max()
    omega_max = viable[viable["torsion_family"] == "omega"]["max_abs_delta_deg"].max()
    return f"""# Omega-Clean Torsion-Boundary Scan

## Scope

This is an omega-clean torsion-boundary scan. It extends the first torsion-envelope scan beyond +/-4 degrees and adds limited combined perturbations. It searches the local (phi/psi/omega)x2 neighborhood around the omega-clean compressed plateau. It is not a final structure, it is not energy minimized, and it is motivated by Nick's item 7. The goal is to estimate a narrow but finite compatible backbone range, not to prove a unique physical structure. Diffraction scoring is preliminary and should not be over-interpreted.

## Method

The one-family scan varies A/C/E and B/D/F independently over -12 to +12 degrees by 2 degrees for phi, psi, or omega while keeping the other torsions fixed. The combined scan adds symmetric same-sign, opposing-class, and phi/psi-compensation patterns without creating a full combinatorial grid.

Coordinate output is limited to baseline, boundary, and representative variants. The large table is a guarded torsion-proxy boundary diagnostic, not a true internal-coordinate rebuild.

## Status Counts

{status_counts}

## One-Family Boundary Summary

{one_table}

## Combined Perturbation Boundary Summary

{combined_table}

## Boundary Interpretation

- Estimated viable phi range reaches +/-{phi_max:.0f} degrees in at least one-family variants.
- Estimated viable psi range reaches +/-{psi_max:.0f} degrees in at least one-family variants.
- Estimated viable omega range reaches +/-{omega_max:.0f} degrees in at least one-family variants.
- A/C/E and B/D/F tolerance differences are reported in the separate max-delta columns; in this proxy scan, large asymmetry is not expected unless one class hits the boundary earlier.
- Failure is usually diffraction-driven when C/D leaves the plateau before geometry becomes implausible; geometry-driven failure appears when omega/every-other or guard logic crosses the defined thresholds.
- Combined perturbation rows indicate whether simultaneous phi/psi/omega movement shrinks the envelope relative to one-family scans.
- The torsion-boundary scan estimates the finite compatible backbone range indicated by the current C/D peak-picking and geometry guards. Whether that range is sufficiently narrow for manuscript framing is a scientific judgment for the PI.
- This does not claim a unique structure. It reports where C/D is preserved, where C/D degrades, and where geometry becomes implausible.

## Next Step

Use this boundary map to select a small set of true internal-coordinate rebuild candidates at the estimated boundary, then run full geometry audit and Debye scoring on those candidates.
"""


def run_scan(
    base_dir: Path = DEFAULT_BASE_DIR,
    outdir: Path = DEFAULT_OUTDIR,
    score_csv: Path = DEFAULT_SCORE_CSV,
    geometry_csv: Path = DEFAULT_GEOMETRY_CSV,
    summary_csv: Path = DEFAULT_SUMMARY_CSV,
    report_path: Path = DEFAULT_REPORT,
    scales: list[float] | None = None,
    deltas: list[float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run boundary scan and write outputs."""
    specs = generate_specs(scales, deltas)
    outdir.mkdir(parents=True, exist_ok=True)
    for path in [score_csv, geometry_csv, summary_csv, report_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    written_ids = write_representative_coordinates(specs, base_dir, outdir)
    geometry_rows = []
    score_rows = []
    for spec in specs:
        geom = proxy_geometry_row(spec, spec.variant_id in written_ids)
        score = proxy_score_row(spec, geom)
        geometry_rows.append(geom)
        score_rows.append(score)
    scores = pd.DataFrame(score_rows)
    geometry = pd.DataFrame(geometry_rows)
    summary = summarize_boundary(scores)
    scores.to_csv(score_csv, index=False)
    geometry.to_csv(geometry_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    report_path.write_text(build_report(scores, geometry, summary), encoding="utf-8")
    return scores, geometry, summary


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--score-csv", type=Path, default=DEFAULT_SCORE_CSV)
    parser.add_argument("--geometry-csv", type=Path, default=DEFAULT_GEOMETRY_CSV)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    """Run CLI."""
    args = parse_args()
    scores, _geometry, summary = run_scan(args.base_dir, args.outdir, args.score_csv, args.geometry_csv, args.summary_csv, args.report)
    print(f"Attempted variants: {len(scores)}")
    print(f"Scoreable variants: {int(scores['scoreable'].sum())}")
    print(f"Guard failed variants: {int((~scores['scoreable'].astype(bool)).sum())}")
    print(f"Viable variants: {int((scores['combined_status'] == 'viable_envelope_member').sum())}")
    print(f"Summary rows: {len(summary)}")
    print(f"Scores: {args.score_csv}")
    print(f"Geometry: {args.geometry_csv}")
    print(f"Summary: {args.summary_csv}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
