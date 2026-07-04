"""Run a local omega-clean torsion-envelope scan around the compressed plateau.

This is a guarded diagnostic scan, not a final structure and not energy
minimized. It probes one torsion family at a time for the two established
chain classes around the omega-clean rise-compression plateau.
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

from scripts.analyze_class_separated_peptide_geometry import chain_geometry_rows, summary_table
from scripts.analyze_threefold_backbone_symmetry import parse_residues
from scripts.generate_global_deformation_variants import format_pdb_coord_line, parse_pdb_atom_lines
from scripts.run_internal_coordinate_endpoint_closure import detect_every_other_pattern, trans_deviation_deg
from scripts.run_omega_clean_rise_compression_scan import (
    class_for_chain,
    coordinate_rmsd,
    ensure_guarded_pdb,
    omega_clean_output_path,
    omega_records,
)
from scripts.run_parent_derived_rise_bridge import TARGETS_A, carboxylate_present, markdown_table, score_pdb_abcd
from scripts.run_parent_derived_rise_fine_scan import best_score_rows, format_scale, plateau_text


PLATEAU_SCALES = [0.9825, 0.9800, 0.9775, 0.9750, 0.9725]
DELTAS_DEG = [-4.0, -2.0, 0.0, 2.0, 4.0]
TORSION_FAMILIES = ["phi", "psi", "omega"]
TRIKETO_CHAINS = {"A", "C", "E"}
TRIAMINO_CHAINS = {"B", "D", "F"}
BACKBONE_NAMES = {"N", "CA", "C", "O"}

DEFAULT_BASE_DIR = Path("outputs/coordinates/omega_clean_rise_compression_scan")
DEFAULT_OUTDIR = Path("outputs/coordinates/omega_clean_torsion_envelope_scan")
DEFAULT_SCORE_CSV = Path("outputs/metrics/omega_clean_torsion_envelope_scores.csv")
DEFAULT_GEOMETRY_CSV = Path("outputs/metrics/omega_clean_torsion_envelope_geometry.csv")
DEFAULT_SUMMARY_CSV = Path("outputs/metrics/omega_clean_torsion_envelope_summary.csv")
DEFAULT_REPORT = Path("outputs/reports/omega_clean_torsion_envelope_report.md")

PLATEAU_C_A = 5.6422
PLATEAU_D_A = 7.2756
PARENTLIKE_C_A = 5.7454
PARENTLIKE_D_A = 7.2756
PEAK_RESOLUTION_TOL_A = 0.001
PARENTLIKE_TOL_A = 0.01


@dataclass(frozen=True)
class EnvelopeSpec:
    """One class-level torsion-envelope perturbation."""

    scale: float
    torsion_family: str
    triketo_delta_deg: float
    triamino_delta_deg: float

    @property
    def variant_id(self) -> str:
        return (
            f"omega_clean_{format_scale(self.scale)}_{self.torsion_family}_"
            f"tri{format_delta(self.triketo_delta_deg)}_mel{format_delta(self.triamino_delta_deg)}"
        )


def format_delta(delta: float) -> str:
    """Return stable delta text."""
    value = int(delta) if float(delta).is_integer() else delta
    return str(value).replace("-", "m").replace("+", "p").replace(".", "p")


def plateau_scale_id(scale: float) -> str:
    """Return stable scale ID."""
    return format_scale(scale)


def generate_specs(scales: list[float] | None = None, deltas: list[float] | None = None) -> list[EnvelopeSpec]:
    """Generate one-torsion-family-at-a-time class-level perturbation specs."""
    scale_values = PLATEAU_SCALES if scales is None else scales
    delta_values = DELTAS_DEG if deltas is None else deltas
    return [
        EnvelopeSpec(float(scale), family, float(tri_delta), float(mel_delta))
        for scale in scale_values
        for family in TORSION_FAMILIES
        for tri_delta in delta_values
        for mel_delta in delta_values
    ]


def baseline_specs(scales: list[float] | None = None) -> list[EnvelopeSpec]:
    """Return no-perturbation specs, one per plateau scale and torsion family."""
    scale_values = PLATEAU_SCALES if scales is None else scales
    return [EnvelopeSpec(scale, family, 0.0, 0.0) for scale in scale_values for family in TORSION_FAMILIES]


def class_delta_for_chain(spec: EnvelopeSpec, chain: str) -> float:
    """Return class-level delta for a chain."""
    class_label = class_for_chain(chain)
    if class_label == "triketo_cyanuric_like":
        return spec.triketo_delta_deg
    if class_label == "triamino_melamine_like":
        return spec.triamino_delta_deg
    return 0.0


def rotate_point(point: np.ndarray, axis_a: np.ndarray, axis_b: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate point around axis A->B by angle degrees."""
    theta = np.radians(float(angle_deg))
    axis = np.asarray(axis_b, dtype=float) - np.asarray(axis_a, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12 or abs(angle_deg) <= 1e-12:
        return np.asarray(point, dtype=float).copy()
    unit = axis / norm
    p = np.asarray(point, dtype=float) - np.asarray(axis_a, dtype=float)
    rotated = p * np.cos(theta) + np.cross(unit, p) * np.sin(theta) + unit * np.dot(unit, p) * (1.0 - np.cos(theta))
    return np.asarray(axis_a, dtype=float) + rotated


def residue_atom_index(atoms) -> dict[tuple[str, str], dict[str, int]]:
    """Return atom indices keyed by chain/residue/atom name."""
    out: dict[tuple[str, str], dict[str, int]] = {}
    for i, atom in enumerate(atoms):
        out.setdefault((atom.chain, str(atom.resseq)), {})[atom.atom_name] = i
    return out


def sorted_residue_keys(atoms) -> list[tuple[str, str]]:
    """Return residue keys in coordinate order."""
    seen = set()
    keys = []
    for atom in atoms:
        key = (atom.chain, str(atom.resseq))
        if key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


def perturb_coords(atoms, spec: EnvelopeSpec) -> np.ndarray:
    """Apply a conservative coordinate-level torsion proxy to backbone atoms."""
    coords = np.array([atom.coord.copy() for atom in atoms], dtype=float)
    lookup = residue_atom_index(atoms)
    keys = sorted_residue_keys(atoms)
    by_chain: dict[str, list[str]] = {}
    for chain, resseq in keys:
        by_chain.setdefault(chain, []).append(resseq)
    for chain, residues in by_chain.items():
        delta = class_delta_for_chain(spec, chain)
        if abs(delta) <= 1e-12:
            continue
        for idx, resseq in enumerate(residues):
            current = lookup.get((chain, resseq), {})
            prev = lookup.get((chain, residues[idx - 1]), {}) if idx > 0 else {}
            nxt = lookup.get((chain, residues[idx + 1]), {}) if idx + 1 < len(residues) else {}
            if spec.torsion_family == "phi" and {"N", "CA", "C"}.issubset(current):
                for name in ["C", "O"]:
                    if name in current:
                        coords[current[name]] = rotate_point(coords[current[name]], coords[current["N"]], coords[current["CA"]], delta)
            elif spec.torsion_family == "psi" and {"CA", "C"}.issubset(current):
                for name, source in [("O", current), ("N", nxt)]:
                    if name in source:
                        coords[source[name]] = rotate_point(coords[source[name]], coords[current["CA"]], coords[current["C"]], delta)
            elif spec.torsion_family == "omega" and {"C"}.issubset(current) and {"N", "CA"}.issubset(nxt):
                coords[nxt["CA"]] = rotate_point(coords[nxt["CA"]], coords[current["C"]], coords[nxt["N"]], delta)
    return coords


def output_path(outdir: Path, spec: EnvelopeSpec) -> Path:
    """Return variant PDB path."""
    return outdir / f"{spec.variant_id}.pdb"


def write_variant(base_pdb: Path, out_path: Path, spec: EnvelopeSpec) -> dict[str, object]:
    """Write one coordinate variant and return write metadata."""
    lines, atoms = parse_pdb_atom_lines(base_pdb)
    coords = perturb_coords(atoms, spec)
    out_lines = list(lines)
    moved = 0
    for atom, coord in zip(atoms, coords):
        if atom.atom_name in BACKBONE_NAMES and not np.allclose(atom.coord, coord):
            moved += 1
        out_lines[atom.index] = format_pdb_coord_line(atom.line, coord)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return {"moved_backbone_atom_count": moved, "write_status": "written"}


def ensure_plateau_inputs(scales: list[float] | None = None) -> None:
    """Ensure omega-clean rise-compression inputs exist."""
    values = PLATEAU_SCALES if scales is None else scales
    missing = [omega_clean_output_path(DEFAULT_BASE_DIR, scale) for scale in values if not omega_clean_output_path(DEFAULT_BASE_DIR, scale).exists()]
    if missing:
        from scripts.run_omega_clean_rise_compression_scan import run_scan

        ensure_guarded_pdb()
        run_scan()


def omega_geometry(path: Path) -> dict[str, object]:
    """Return coordinate-derived omega metrics."""
    records = omega_records(path)
    df = pd.DataFrame(records)
    if df.empty:
        return {
            "omega_count": 0,
            "omega_median_deg": np.nan,
            "omega_trans_deviation_median_deg": np.nan,
            "omega_within_8_count": 0,
            "omega_within_8_fraction": np.nan,
            "omega_within_10_count": 0,
            "omega_within_10_fraction": np.nan,
            "omega_outside_10_count": 0,
            "omega_outside_10_fraction": np.nan,
            "coordinate_omega_every_other_detected": False,
            "any_chain_coordinate_omega_every_other_detected": False,
        }
    deviations = pd.to_numeric(df["omega_trans_deviation_deg"], errors="coerce").dropna().tolist()
    count = len(deviations)
    within8 = int(sum(value <= 8.0 for value in deviations))
    within10 = int(sum(value <= 10.0 for value in deviations))
    pattern = detect_every_other_pattern(deviations)
    chain_patterns = []
    for _chain, group in df.groupby("chain"):
        chain_patterns.append(bool(detect_every_other_pattern(group["omega_trans_deviation_deg"].tolist())["every_other_detected"]))
    return {
        "omega_count": count,
        "omega_median_deg": float(pd.to_numeric(df["omega_deg"], errors="coerce").median()),
        "omega_trans_deviation_median_deg": float(pd.Series(deviations).median()) if deviations else np.nan,
        "omega_within_8_count": within8,
        "omega_within_8_fraction": within8 / count if count else np.nan,
        "omega_within_10_count": within10,
        "omega_within_10_fraction": within10 / count if count else np.nan,
        "omega_outside_10_count": count - within10,
        "omega_outside_10_fraction": (count - within10) / count if count else np.nan,
        "coordinate_omega_every_other_detected": bool(pattern["every_other_detected"]),
        "any_chain_coordinate_omega_every_other_detected": bool(any(chain_patterns)),
    }


def class_geometry(path: Path) -> dict[str, object]:
    """Return class-separated geometry metrics."""
    try:
        table = summary_table(path.stem, chain_geometry_rows(parse_residues(path)))
    except Exception:
        return {}
    out: dict[str, object] = {}
    for group, prefix in [("all_six_chains", "all"), ("triketo_cyanuric_like", "triketo"), ("triamino_melamine_like", "triamino")]:
        row = table[(table["row_type"] == "summary") & (table["group"] == group)]
        if row.empty:
            continue
        r = row.iloc[0]
        for column in ["ca_rise_median_A", "exit_vector_angle_gap_rms_deg", "radial_angle_gap_rms_deg"]:
            out[f"{prefix}_{column}"] = r.get(column, np.nan)
    return out


def cd_status(c_A: float, d_A: float) -> str:
    """Classify C/D peak pattern."""
    if abs(c_A - PLATEAU_C_A) <= PEAK_RESOLUTION_TOL_A and abs(d_A - PLATEAU_D_A) <= PEAK_RESOLUTION_TOL_A:
        return "cd_plateau_preserved"
    if abs(c_A - PLATEAU_C_A) <= PEAK_RESOLUTION_TOL_A and abs(d_A - PLATEAU_D_A) > PEAK_RESOLUTION_TOL_A:
        return "c_preserved_d_degraded"
    if abs(c_A - PARENTLIKE_C_A) <= PARENTLIKE_TOL_A and abs(d_A - PARENTLIKE_D_A) <= PARENTLIKE_TOL_A:
        return "parent_like"
    return "degraded_other"


def geometry_status(row: dict[str, object]) -> str:
    """Classify geometry status from guard metrics."""
    if row.get("write_status") != "written":
        return "reconstruction_failed"
    if not row.get("atom_count_preserved", False) or not row.get("carboxylates_preserved", False):
        return "geometry_implausible"
    if bool(row.get("coordinate_omega_every_other_detected", False)):
        return "geometry_implausible"
    if int(row.get("omega_within_10_count", 0)) < int(row.get("omega_count", 0)):
        return "geometry_borderline"
    if int(row.get("omega_within_8_count", 0)) < int(row.get("omega_count", 0)):
        return "geometry_borderline"
    return "geometry_clean"


def combined_status(cd: str, geom: str) -> str:
    """Combine diffraction and geometry status."""
    if cd == "cd_plateau_preserved" and geom in {"geometry_clean", "geometry_borderline"}:
        return "viable_envelope_member"
    if cd == "cd_plateau_preserved" and geom == "geometry_implausible":
        return "diffraction_only_member"
    if cd != "cd_plateau_preserved" and geom in {"geometry_clean", "geometry_borderline"}:
        return "geometry_only_member"
    return "rejected"


def score_variant(spec: EnvelopeSpec, path: Path, scoreable: bool) -> dict[str, object]:
    """Score a variant when scoreable."""
    base = {
        "variant_id": spec.variant_id,
        "scale": spec.scale,
        "torsion_family": spec.torsion_family,
        "triketo_delta_deg": spec.triketo_delta_deg,
        "triamino_delta_deg": spec.triamino_delta_deg,
        "coordinate_path": str(path),
        "scoreable": scoreable,
    }
    if not scoreable:
        base.update(
            {
                "observed_C_d_A": np.nan,
                "observed_D_d_A": np.nan,
                "C_error_A": np.nan,
                "D_error_A": np.nan,
                "combined_CD_abs_error_A": np.nan,
                "C_score": np.nan,
                "D_score": np.nan,
                "cd_status": "not_scored_guard_failed",
            }
        )
        return base
    scores = score_pdb_abcd(path)
    c_error = float(scores["observed_C_d_A"]) - TARGETS_A["C"]
    d_error = float(scores["observed_D_d_A"]) - TARGETS_A["D"]
    base.update(
        {
            **scores,
            "C_error_A": c_error,
            "D_error_A": d_error,
            "combined_CD_abs_error_A": abs(c_error) + abs(d_error),
            "cd_status": cd_status(float(scores["observed_C_d_A"]), float(scores["observed_D_d_A"])),
        }
    )
    return base


def geometry_row(spec: EnvelopeSpec, base_pdb: Path, path: Path, write_info: dict[str, object]) -> dict[str, object]:
    """Return geometry row for a variant."""
    _base_lines, base_atoms = parse_pdb_atom_lines(base_pdb)
    _var_lines, var_atoms = parse_pdb_atom_lines(path)
    row = {
        "variant_id": spec.variant_id,
        "scale": spec.scale,
        "torsion_family": spec.torsion_family,
        "triketo_delta_deg": spec.triketo_delta_deg,
        "triamino_delta_deg": spec.triamino_delta_deg,
        "coordinate_path": str(path),
        "atom_count": len(var_atoms),
        "atom_count_preserved": len(base_atoms) == len(var_atoms),
        "carboxylates_preserved": carboxylate_present(base_atoms) and carboxylate_present(var_atoms),
        "backbone_rmsd_to_unperturbed_plateau_A": coordinate_rmsd(base_atoms, var_atoms),
        "phi_perturbation_abs_max_deg": max(abs(spec.triketo_delta_deg), abs(spec.triamino_delta_deg)) if spec.torsion_family == "phi" else 0.0,
        "psi_perturbation_abs_max_deg": max(abs(spec.triketo_delta_deg), abs(spec.triamino_delta_deg)) if spec.torsion_family == "psi" else 0.0,
        "omega_perturbation_abs_max_deg": max(abs(spec.triketo_delta_deg), abs(spec.triamino_delta_deg)) if spec.torsion_family == "omega" else 0.0,
        "selected_retained_omega_every_other_detected": False,
        "closure_status": "not_recomputed_for_torsion_proxy",
        "overlap_status": "not_recomputed_for_torsion_proxy",
        "drift_status": "not_recomputed_for_torsion_proxy",
        "steric_status": "not_recomputed_for_torsion_proxy",
        **write_info,
    }
    row.update(omega_geometry(path))
    row.update(class_geometry(path))
    row["geometry_status"] = geometry_status(row)
    return row


def summarize(scores: pd.DataFrame, geometry: pd.DataFrame) -> pd.DataFrame:
    """Build scale/torsion-family summary rows."""
    if "geometry_status" in scores.columns:
        merged = scores.copy()
    else:
        merged = scores.merge(geometry[["variant_id", "geometry_status"]], on="variant_id", how="left")
    rows = []
    for (scale, family), group in merged.groupby(["scale", "torsion_family"], sort=True):
        viable = group[group["combined_status"] == "viable_envelope_member"]
        rows.append(
            {
                "scale": scale,
                "torsion_family": family,
                "attempted_variant_count": len(group),
                "scoreable_variant_count": int(group["scoreable"].sum()),
                "guard_failed_count": int((~group["scoreable"].astype(bool)).sum()),
                "cd_plateau_preserved_count": int((group["cd_status"] == "cd_plateau_preserved").sum()),
                "geometry_clean_count": int(group["geometry_status"].isin(["geometry_clean", "geometry_borderline"]).sum()),
                "viable_envelope_member_count": len(viable),
                "max_abs_triketo_delta_viable_deg": float(viable["triketo_delta_deg"].abs().max()) if not viable.empty else np.nan,
                "max_abs_triamino_delta_viable_deg": float(viable["triamino_delta_deg"].abs().max()) if not viable.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_report(scores: pd.DataFrame, geometry: pd.DataFrame, summary: pd.DataFrame) -> str:
    """Build markdown report."""
    viable = scores[scores["combined_status"] == "viable_envelope_member"]
    best_rows = best_score_rows(scores[pd.notna(scores["combined_CD_abs_error_A"])]) if not scores.empty else pd.DataFrame()
    viable_ranges = (
        viable.groupby("torsion_family")
        .agg(
            max_abs_triketo_delta_viable_deg=("triketo_delta_deg", lambda s: float(s.abs().max()) if len(s) else np.nan),
            max_abs_triamino_delta_viable_deg=("triamino_delta_deg", lambda s: float(s.abs().max()) if len(s) else np.nan),
            viable_count=("variant_id", "count"),
        )
        .reset_index()
        if not viable.empty
        else pd.DataFrame(columns=["torsion_family", "max_abs_triketo_delta_viable_deg", "max_abs_triamino_delta_viable_deg", "viable_count"])
    )
    summary_table = markdown_table(
        summary,
        [
            "scale",
            "torsion_family",
            "attempted_variant_count",
            "scoreable_variant_count",
            "guard_failed_count",
            "cd_plateau_preserved_count",
            "geometry_clean_count",
            "viable_envelope_member_count",
            "max_abs_triketo_delta_viable_deg",
            "max_abs_triamino_delta_viable_deg",
        ],
    )
    range_table = markdown_table(
        viable_ranges,
        ["torsion_family", "max_abs_triketo_delta_viable_deg", "max_abs_triamino_delta_viable_deg", "viable_count"],
    )
    status_counts = markdown_table(scores["combined_status"].value_counts().rename_axis("combined_status").reset_index(name="count"), ["combined_status", "count"])
    best_text = plateau_text(best_rows) if not best_rows.empty else "not observed"
    narrowest = float(
        viable_ranges[["max_abs_triketo_delta_viable_deg", "max_abs_triamino_delta_viable_deg"]].min().min()
    ) if not viable_ranges.empty else np.nan
    return f"""# Omega-Clean Torsion-Envelope Scan

## Scope

This is an omega-clean torsion-envelope scan. It searches the local (phi/psi/omega) x 2 neighborhood around the omega-clean compressed plateau. It is not a final structure, it is not energy minimized, and it should not be interpreted as proof of the physical hexaplex structure. It is motivated by Nick's plan to test the local backbone structural range around the best solution. The goal is to define a defensible local torsion envelope compatible with C/D and geometry constraints.

## Method

Starting coordinates are the omega-clean rise-compression plateau variants from `omega_clean_scale_0p9825` through `omega_clean_scale_0p9725`. The scan uses established chain classes: A/C/E as triketo/cyanuric-like and B/D/F as triamino/melamine-like. One torsion family is perturbed at a time (`phi`, `psi`, or `omega`), with independent class-level deltas of -4, -2, 0, +2, and +4 degrees.

Coordinate variants are guarded diagnostic torsion-proxy perturbations. They preserve atom order, residue/register labels, chain IDs, and carboxylates, but they are not minimized and do not replace a full internal-coordinate builder.

## Summary By Scale And Torsion Family

{summary_table}

## Viable Envelope Ranges

{range_table}

## Combined Status Counts

{status_counts}

## Interpretation Questions

- Around `omega_clean_scale_0p9825` through `0p9725`, how much can phi vary independently for A/C/E and B/D/F before C/D degrades? See the `phi` row in the viable envelope table.
- How much can psi vary independently? See the `psi` row.
- How much can omega vary independently? See the `omega` row. Omega variants are expected to be the strictest because geometry classification enforces +/-8 and +/-10 trans windows.
- Which plateau scale is most robust? Compare viable counts by scale in the summary table.
- Which torsion dimension is most sensitive? The narrowest viable absolute class delta is {narrowest:.4g} degrees.
- Are A/C/E and B/D/F different in tolerance? Compare `max_abs_triketo_delta_viable_deg` and `max_abs_triamino_delta_viable_deg`.
- Do any torsion perturbations preserve C/D but make geometry implausible? Count `diffraction_only_member` in the status table.
- Do any torsion perturbations preserve geometry but lose C/D? Count `geometry_only_member` in the status table.
- Is the viable envelope narrow enough to support the claim that C/D bands constrain the local backbone structure? This first diagnostic supports that claim only if viable ranges are limited and guard failures rise quickly; treat this as a preliminary envelope, not a final model.
- Does this strengthen the omega-clean compressed model as the current best defensible structure family? It strengthens it if a nonzero local envelope exists around the C/D plateau without reintroducing selected/retained every-other omega artifacts.
- Best score plateau among scored variants: `{best_text}`.

## Next Implementation Step

Replace this torsion-proxy coordinate perturbation with a true two-class internal-coordinate builder that solves phi/psi/omega while preserving C-alpha/register constraints, then rerun this same envelope report on fully reconstructed candidates.

## Outputs

- Scores: `outputs/metrics/omega_clean_torsion_envelope_scores.csv`
- Geometry: `outputs/metrics/omega_clean_torsion_envelope_geometry.csv`
- Summary: `outputs/metrics/omega_clean_torsion_envelope_summary.csv`
- Coordinates: `outputs/coordinates/omega_clean_torsion_envelope_scan/`
"""


def run_scan(
    scales: list[float] | None = None,
    deltas: list[float] | None = None,
    base_dir: Path = DEFAULT_BASE_DIR,
    outdir: Path = DEFAULT_OUTDIR,
    score_csv: Path = DEFAULT_SCORE_CSV,
    geometry_csv: Path = DEFAULT_GEOMETRY_CSV,
    summary_csv: Path = DEFAULT_SUMMARY_CSV,
    report_path: Path = DEFAULT_REPORT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the omega-clean torsion-envelope scan."""
    scale_values = PLATEAU_SCALES if scales is None else scales
    ensure_plateau_inputs(scale_values)
    specs = generate_specs(scale_values, deltas)
    outdir.mkdir(parents=True, exist_ok=True)
    score_csv.parent.mkdir(parents=True, exist_ok=True)
    geometry_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    geometry_rows = []
    score_rows = []
    for spec in specs:
        base_pdb = omega_clean_output_path(base_dir, spec.scale)
        out_pdb = output_path(outdir, spec)
        write_info = write_variant(base_pdb, out_pdb, spec)
        geom = geometry_row(spec, base_pdb, out_pdb, write_info)
        scoreable = geom["geometry_status"] in {"geometry_clean", "geometry_borderline"}
        score = score_variant(spec, out_pdb, scoreable)
        score["geometry_status"] = geom["geometry_status"]
        score["combined_status"] = combined_status(score["cd_status"], geom["geometry_status"])
        geometry_rows.append(geom)
        score_rows.append(score)

    scores = pd.DataFrame(score_rows)
    geometry = pd.DataFrame(geometry_rows)
    summary = summarize(scores, geometry)
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
    scores, _geometry, summary = run_scan(
        base_dir=args.base_dir,
        outdir=args.outdir,
        score_csv=args.score_csv,
        geometry_csv=args.geometry_csv,
        summary_csv=args.summary_csv,
        report_path=args.report,
    )
    print(f"Attempted variants: {len(scores)}")
    print(f"Scored variants: {int(scores['scoreable'].sum())}")
    print(f"Viable envelope members: {int((scores['combined_status'] == 'viable_envelope_member').sum())}")
    print(f"Summary rows: {len(summary)}")
    print(f"Scores: {args.score_csv}")
    print(f"Geometry: {args.geometry_csv}")
    print(f"Summary: {args.summary_csv}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
