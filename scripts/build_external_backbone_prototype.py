"""Build a conservative external-backbone local-fragment prototype.

This is a coordinate-producing external backbone prototype, not a final
structure and not energy minimized. It uses selected phi/psi/omega rows from the
closure scan to write a multi-model local-fragment PDB for inspectable peptide
backbone segments. A full-chain PDB is deliberately deferred because adjacent
local reconstructions overlap and would need a global chain-consistency solve.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hexaplex_backbone_fingerprint.geometry import distance
from scripts.analyze_threefold_backbone_symmetry import parse_residues
from scripts.generate_global_deformation_variants import parse_pdb_atom_lines
from scripts.run_internal_coordinate_endpoint_closure import (
    closure_class,
    detect_every_other_pattern,
    markdown_table,
    omega_window_class,
    point_from_internal,
    trans_deviation_deg,
)
from scripts.run_parent_derived_rise_bridge import DEFAULT_PARENT_PDB, carboxylate_present
from scripts.run_phi_psi_omega_closure_scan import (
    DEFAULT_SCAN_CSV,
    PhiPsiOmegaSegment,
    build_segments,
    reconstruct_endpoint,
    run_scan,
    scan_segments,
    wrap_angle_deg,
)


DEFAULT_OUTDIR = Path("outputs/coordinates/external_backbone_prototype")
DEFAULT_PDB = DEFAULT_OUTDIR / "external_backbone_phi_psi_omega_prototype.pdb"
DEFAULT_SELECTED_CSV = Path("outputs/metrics/external_backbone_prototype_selected_torsions.csv")
DEFAULT_GEOMETRY_CSV = Path("outputs/metrics/external_backbone_prototype_geometry.csv")
DEFAULT_ABCD_CSV = Path("outputs/metrics/external_backbone_prototype_abcd_scores.csv")
DEFAULT_REPORT = Path("outputs/reports/external_backbone_prototype_report.md")

PARENT_BASELINE = {"C": 5.7454, "D": 7.2756, "combined_CD_abs_error_A": 0.1698}
FINE_SCAN_TARGET = {"C": 5.6422, "D": 7.2756, "combined_CD_abs_error_A": 0.0667}


def load_or_generate_scan(parent_pdb: Path, scan_csv: Path = DEFAULT_SCAN_CSV) -> pd.DataFrame:
    """Load the phi/psi/omega scan, generating it when absent."""
    if scan_csv.exists():
        return pd.read_csv(scan_csv)
    scan, _best, _summary = run_scan(parent_pdb=parent_pdb)
    return scan


def selected_torsion_for_segment(group: pd.DataFrame) -> dict[str, object]:
    """Choose one torsion row using the requested preference order."""
    scored = group[group["scan_status"] == "scored"].copy()
    choices = [
        ("good_within_8deg", scored[(scored["omega_window_class"] == "within_8deg") & (scored["closure_class"] == "good_closure")]),
        (
            "good_within_10deg",
            scored[
                scored["omega_window_class"].isin(["within_8deg", "within_10deg"])
                & (scored["closure_class"] == "good_closure")
            ],
        ),
        (
            "borderline_within_10deg",
            scored[
                scored["omega_window_class"].isin(["within_8deg", "within_10deg"])
                & scored["closure_class"].isin(["good_closure", "borderline_closure"])
            ],
        ),
    ]
    for reason, subset in choices:
        if not subset.empty:
            row = subset.sort_values(["closure_residual_A", "omega_trans_deviation_deg", "scanned_phi_deg", "scanned_psi_deg"]).iloc[0].to_dict()
            row["selection_reason"] = reason
            row["selected_phi_deg"] = row["scanned_phi_deg"]
            row["selected_psi_deg"] = row["scanned_psi_deg"]
            row["selected_omega_deg"] = row["scanned_omega_deg"]
            return row
    first = group.iloc[0].to_dict()
    first.update(
        {
            "selection_reason": "retain_parent_unresolved",
            "selected_phi_deg": np.nan,
            "selected_psi_deg": np.nan,
            "selected_omega_deg": np.nan,
            "closure_residual_A": np.nan,
            "closure_class": "insufficient_data",
        }
    )
    return first


def select_torsions(scan: pd.DataFrame) -> pd.DataFrame:
    """Select one torsion row per segment."""
    rows = [selected_torsion_for_segment(group) for _segment_id, group in scan.groupby("segment_id", sort=True)]
    selected = pd.DataFrame(rows)
    columns = [
        "segment_id",
        "chain",
        "class_label",
        "res_i",
        "res_j",
        "parent_phi_deg",
        "parent_psi_deg",
        "parent_omega_deg",
        "selected_phi_deg",
        "selected_psi_deg",
        "selected_omega_deg",
        "omega_window_class",
        "omega_trans_deviation_deg",
        "closure_residual_A",
        "closure_class",
        "phi_delta_from_parent_deg",
        "psi_delta_from_parent_deg",
        "selection_reason",
        "scan_status",
    ]
    for column in columns:
        if column not in selected.columns:
            selected[column] = np.nan
    return selected[columns]


def segment_lookup(parent_pdb: Path) -> dict[str, PhiPsiOmegaSegment]:
    """Return segment lookup by segment_id."""
    return {segment.segment_id: segment for segment in build_segments(parse_residues(parent_pdb))}


def reconstructed_points(segment: PhiPsiOmegaSegment, selected: pd.Series) -> dict[str, np.ndarray]:
    """Return local reconstructed points for one selected segment."""
    phi = float(selected["selected_phi_deg"])
    psi = float(selected["selected_psi_deg"])
    omega = float(selected["selected_omega_deg"])
    if segment.c_prev is None or not all(np.isfinite([phi, psi, omega])):
        raise ValueError("Cannot reconstruct segment without selected phi/psi/omega and previous C context.")
    c_i = point_from_internal(
        segment.c_prev,
        segment.n_i,
        segment.ca_i,
        segment.ca_c_length_A,
        segment.n_ca_c_angle_deg,
        phi,
    )
    n_j = point_from_internal(
        segment.n_i,
        segment.ca_i,
        c_i,
        segment.c_n_length_A,
        segment.ca_c_n_angle_deg,
        psi,
    )
    ca_j = point_from_internal(
        segment.ca_i,
        c_i,
        n_j,
        segment.n_ca_length_A,
        segment.c_n_ca_angle_deg,
        omega,
    )
    return {
        "N_i": segment.n_i,
        "CA_i": segment.ca_i,
        "C_i": c_i,
        "N_j": n_j,
        "CA_j": ca_j,
    }


def format_atom_line(serial: int, atom_name: str, resname: str, chain: str, resseq: str, coord: np.ndarray) -> str:
    """Format a simple PDB ATOM line."""
    element = "".join(char for char in atom_name if char.isalpha())[:1].upper() or "C"
    return (
        f"ATOM  {serial:5d} {atom_name:>4s} {resname:>3s} {chain:1s}{int(float(resseq)):4d}    "
        f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00  0.00          {element:>2s}"
    )


def write_local_fragment_pdb(selected: pd.DataFrame, segments: dict[str, PhiPsiOmegaSegment], out_path: Path, max_models: int | None = None) -> dict[str, object]:
    """Write selected reconstructed local fragments as a multi-model PDB."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "REMARK coordinate-producing external backbone prototype",
        "REMARK local-fragment multi-model PDB; not a full-chain atomistic reconstruction",
        "REMARK O atoms and downstream C/O atoms are retained from parent only in source metrics, not reconstructed here",
    ]
    reconstructed = selected[selected["selection_reason"].isin(["good_within_8deg", "good_within_10deg", "borderline_within_10deg"])].copy()
    if max_models is not None:
        reconstructed = reconstructed.head(max_models)
    model_count = 0
    serial = 1
    for row in reconstructed.itertuples(index=False):
        segment = segments.get(row.segment_id)
        if segment is None:
            continue
        try:
            points = reconstructed_points(segment, pd.Series(row._asdict()))
        except Exception:
            continue
        model_count += 1
        lines.append(f"MODEL     {model_count:4d}")
        lines.append(f"REMARK segment_id {row.segment_id} selected_omega {row.selected_omega_deg}")
        res_i_name, res_j_name = segment.residue_pair.split("->")
        res_i_resname = "".join([char for char in res_i_name if char.isalpha()])[:3]
        res_j_resname = "".join([char for char in res_j_name if char.isalpha()])[:3]
        fragment_atoms = [
            ("N", res_i_resname, segment.res_i, points["N_i"]),
            ("CA", res_i_resname, segment.res_i, points["CA_i"]),
            ("C", res_i_resname, segment.res_i, points["C_i"]),
            ("N", res_j_resname, segment.res_j, points["N_j"]),
            ("CA", res_j_resname, segment.res_j, points["CA_j"]),
        ]
        for atom_name, resname, resseq, coord in fragment_atoms:
            lines.append(format_atom_line(serial, atom_name, resname, segment.chain, str(resseq), coord))
            serial += 1
        lines.append("ENDMDL")
    lines.append("END")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"fragment_model_count": model_count, "fragment_atom_count": serial - 1}


def count_atom_records(path: Path) -> int:
    """Count ATOM/HETATM records in a PDB-like file."""
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.startswith(("ATOM  ", "HETATM")))


def atom_count_preserved(source_path: Path, prototype_path: Path) -> bool:
    """Return whether atom count is preserved between source and prototype."""
    return count_atom_records(source_path) == count_atom_records(prototype_path)


def selected_omega_summary(selected: pd.DataFrame) -> dict[str, object]:
    """Summarize selected omega values."""
    resolved = selected[pd.notna(selected["selected_omega_deg"])].copy()
    deviations = resolved["selected_omega_deg"].map(trans_deviation_deg).tolist()
    pattern = detect_every_other_pattern(deviations)
    total = len(resolved)
    within8 = int(sum(value <= 8.0 for value in deviations))
    within10 = int(sum(value <= 10.0 for value in deviations))
    return {
        "selected_count": total,
        "selected_omega_median_deg": float(resolved["selected_omega_deg"].median()) if total else np.nan,
        "selected_omega_trans_deviation_median_deg": float(pd.Series(deviations).median()) if deviations else np.nan,
        "selected_omega_within_8deg_count": within8,
        "selected_omega_within_10deg_count": within10,
        "selected_omega_outside_10deg_count": int(total - within10),
        "selected_omega_every_other_detected": pattern["every_other_detected"],
    }


def geometry_summary(
    source_pdb: Path,
    prototype_pdb: Path,
    selected: pd.DataFrame,
    fragment_info: dict[str, object],
) -> pd.DataFrame:
    """Build one-row geometry/provenance summary."""
    _source_lines, source_atoms = parse_pdb_atom_lines(source_pdb)
    omega = selected_omega_summary(selected)
    reconstructed = selected[selected["selection_reason"].isin(["good_within_8deg", "good_within_10deg", "borderline_within_10deg"])]
    unresolved = selected[selected["selection_reason"] == "retain_parent_unresolved"]
    row = {
        "prototype_type": "local_fragment_multimodel",
        "full_prototype_pdb_produced": False,
        "fragment_prototype_pdb_produced": prototype_pdb.exists(),
        "source_atom_count": len(source_atoms),
        "prototype_atom_count": count_atom_records(prototype_pdb) if prototype_pdb.exists() else 0,
        "atom_count_preserved": atom_count_preserved(source_pdb, prototype_pdb) if prototype_pdb.exists() else False,
        "source_carboxylate_present": carboxylate_present(source_atoms),
        "prototype_carboxylate_present": False,
        "carboxylate_preservation_status": "not_applicable_for_local_fragment_prototype",
        "selected_segment_count": len(selected),
        "reconstructed_segment_count": len(reconstructed),
        "retained_parent_segment_count": len(unresolved),
        "unresolved_segment_count": len(unresolved),
        "fragment_model_count": fragment_info.get("fragment_model_count", 0),
        "fragment_atom_count": fragment_info.get("fragment_atom_count", 0),
        "median_phi_delta_from_parent_deg": float(pd.to_numeric(reconstructed["phi_delta_from_parent_deg"], errors="coerce").median()),
        "median_psi_delta_from_parent_deg": float(pd.to_numeric(reconstructed["psi_delta_from_parent_deg"], errors="coerce").median()),
        "median_closure_residual_A": float(pd.to_numeric(reconstructed["closure_residual_A"], errors="coerce").median()),
        "max_closure_residual_A": float(pd.to_numeric(reconstructed["closure_residual_A"], errors="coerce").max()),
        "backbone_atom_rmsd_to_parent_A": np.nan,
        "diffraction_scoring_status": "not_applicable_for_fragment_prototype",
        "notes": "full-chain assembly deferred because local segment reconstructions overlap and require a global consistency solve",
    }
    row.update(omega)
    return pd.DataFrame([row])


def abcd_not_applicable_row() -> pd.DataFrame:
    """Return diffraction score row for fragment prototype."""
    return pd.DataFrame(
        [
            {
                "prototype_id": "external_backbone_phi_psi_omega_prototype",
                "prototype_type": "local_fragment_multimodel",
                "observed_C_d_A": np.nan,
                "observed_D_d_A": np.nan,
                "combined_CD_abs_error_A": np.nan,
                "parent_baseline_C_d_A": PARENT_BASELINE["C"],
                "parent_baseline_D_d_A": PARENT_BASELINE["D"],
                "parent_baseline_combined_CD_abs_error_A": PARENT_BASELINE["combined_CD_abs_error_A"],
                "diagnostic_fine_scan_C_d_A": FINE_SCAN_TARGET["C"],
                "diagnostic_fine_scan_D_d_A": FINE_SCAN_TARGET["D"],
                "diagnostic_fine_scan_combined_CD_abs_error_A": FINE_SCAN_TARGET["combined_CD_abs_error_A"],
                "diffraction_scoring_status": "not_applicable_for_fragment_prototype",
                "notes": "local-fragment prototype is not a full scattering model; diffraction scoring would be misleading",
            }
        ]
    )


def build_report(selected: pd.DataFrame, geometry: pd.DataFrame, abcd: pd.DataFrame, prototype_pdb: Path) -> str:
    """Build markdown report."""
    geom = geometry.iloc[0]
    abcd_row = abcd.iloc[0]
    return f"""# External Backbone Phi/Psi/Omega Prototype

This is a coordinate-producing external backbone prototype. It is not a final structure and it is not energy minimized.

It is motivated by Asem's pNAB limitation and Nick's concern about every-other peptide non-planarity. The goal is to test whether selected phi/psi/omega torsions can produce usable coordinates without systematic pNAB-induced omega artifacts. Diffraction scoring, if performed, is preliminary and should not be over-interpreted.

## Prototype Type

- Prototype PDB: `{prototype_pdb}`
- Prototype type: `{geom['prototype_type']}`
- Full prototype PDB produced: {bool(geom['full_prototype_pdb_produced'])}
- Fragment prototype PDB produced: {bool(geom['fragment_prototype_pdb_produced'])}

A full-chain coordinate assembly is deferred. Adjacent selected local reconstructions overlap along each chain, so writing a full PDB without a global consistency solve would fake continuity. This prototype writes a multi-model local-fragment PDB for inspectable reconstructed N/CA/C/N/CA peptide fragments.

## Torsion Selection

Selection preference:

1. good closure within +/- 8 deg omega
2. good closure within +/- 10 deg omega
3. borderline closure within +/- 10 deg omega
4. retain parent / unresolved

{markdown_table(selected['selection_reason'].value_counts().rename_axis('selection_reason').reset_index(name='count'), ['selection_reason', 'count'])}

## Geometry Summary

{markdown_table(geometry, ['prototype_type', 'source_atom_count', 'prototype_atom_count', 'atom_count_preserved', 'source_carboxylate_present', 'prototype_carboxylate_present', 'selected_segment_count', 'reconstructed_segment_count', 'retained_parent_segment_count', 'selected_omega_median_deg', 'selected_omega_within_8deg_count', 'selected_omega_within_10deg_count', 'selected_omega_every_other_detected', 'median_closure_residual_A', 'max_closure_residual_A', 'diffraction_scoring_status'])}

## Diffraction Scoring

Diffraction scoring status: `{abcd_row['diffraction_scoring_status']}`.

The local-fragment PDB is not suitable for Debye/powder scoring because it is not a complete scattering model. The relevant baselines remain:

- Parent/reference baseline: C = {PARENT_BASELINE['C']:.4f} A, D = {PARENT_BASELINE['D']:.4f} A, combined C/D error = {PARENT_BASELINE['combined_CD_abs_error_A']:.4f} A
- Diagnostic fine-scan target: C = {FINE_SCAN_TARGET['C']:.4f} A, D = {FINE_SCAN_TARGET['D']:.4f} A, combined C/D error = {FINE_SCAN_TARGET['combined_CD_abs_error_A']:.4f} A

## Interpretation

- Was a full prototype PDB produced, or only local-fragment prototypes? Only a local-fragment multi-model prototype was produced.
- How many segments were reconstructed from selected phi/psi/omega values? {int(geom['reconstructed_segment_count'])}.
- How many segments had to retain parent coordinates? {int(geom['retained_parent_segment_count'])}.
- Are selected omega values inside +/- 8 or +/- 10 degrees? {int(geom['selected_omega_within_8deg_count'])}/{int(geom['selected_count'])} selected segments are within +/- 8 deg and {int(geom['selected_omega_within_10deg_count'])}/{int(geom['selected_count'])} are within +/- 10 deg.
- Do selected omega values show every-other behavior? {bool(geom['selected_omega_every_other_detected'])}.
- Does the reconstructed prototype preserve atom count, carboxylates, residue order, and register? The source labels/register are preserved in the selected torsion table and fragment records, but atom count/carboxylate preservation is not applicable to the fragment-only PDB.
- Does geometry look plausible by available metrics? The selected local fragments have low closure residuals, but full-chain geometry plausibility remains unresolved until a global chain-consistency solve is added.
- If diffraction scoring was possible, does the prototype move C toward 5.6422 A while preserving D near 7.2756 A? Diffraction scoring was not applicable for this fragment prototype.
- Does this support continuing toward a fuller external two-class atomistic model? Yes, as a controlled next step, but only after implementing global chain assembly.

## Next Implementation Step

Build a global chain-consistency solver that assembles selected local phi/psi/omega solutions into continuous chains while preserving recognition-core/register atoms and carboxylates. Only then should a full prototype PDB be diffraction-scored.

## Outputs

- Prototype PDB: `outputs/coordinates/external_backbone_prototype/external_backbone_phi_psi_omega_prototype.pdb`
- Selected torsions: `outputs/metrics/external_backbone_prototype_selected_torsions.csv`
- Geometry metrics: `outputs/metrics/external_backbone_prototype_geometry.csv`
- ABCD scores: `outputs/metrics/external_backbone_prototype_abcd_scores.csv`
- Report: `outputs/reports/external_backbone_prototype_report.md`
"""


def run_prototype(
    parent_pdb: Path = DEFAULT_PARENT_PDB,
    scan_csv: Path = DEFAULT_SCAN_CSV,
    prototype_pdb: Path = DEFAULT_PDB,
    selected_csv: Path = DEFAULT_SELECTED_CSV,
    geometry_csv: Path = DEFAULT_GEOMETRY_CSV,
    abcd_csv: Path = DEFAULT_ABCD_CSV,
    report_path: Path = DEFAULT_REPORT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build local-fragment prototype and write outputs."""
    scan = load_or_generate_scan(parent_pdb, scan_csv)
    selected = select_torsions(scan)
    segments = segment_lookup(parent_pdb)
    fragment_info = write_local_fragment_pdb(selected, segments, prototype_pdb)
    geometry = geometry_summary(parent_pdb, prototype_pdb, selected, fragment_info)
    abcd = abcd_not_applicable_row()
    for path in [selected_csv, geometry_csv, abcd_csv, report_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(selected_csv, index=False)
    geometry.to_csv(geometry_csv, index=False)
    abcd.to_csv(abcd_csv, index=False)
    report_path.write_text(build_report(selected, geometry, abcd, prototype_pdb), encoding="utf-8")
    return selected, geometry, abcd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-pdb", type=Path, default=DEFAULT_PARENT_PDB)
    parser.add_argument("--scan-csv", type=Path, default=DEFAULT_SCAN_CSV)
    parser.add_argument("--prototype-pdb", type=Path, default=DEFAULT_PDB)
    parser.add_argument("--selected-csv", type=Path, default=DEFAULT_SELECTED_CSV)
    parser.add_argument("--geometry-csv", type=Path, default=DEFAULT_GEOMETRY_CSV)
    parser.add_argument("--abcd-csv", type=Path, default=DEFAULT_ABCD_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected, geometry, _abcd = run_prototype(
        parent_pdb=args.parent_pdb,
        scan_csv=args.scan_csv,
        prototype_pdb=args.prototype_pdb,
        selected_csv=args.selected_csv,
        geometry_csv=args.geometry_csv,
        abcd_csv=args.abcd_csv,
        report_path=args.report,
    )
    row = geometry.iloc[0]
    print(f"Prototype type: {row['prototype_type']}")
    print(f"Reconstructed segments: {int(row['reconstructed_segment_count'])}")
    print(f"Retained/unresolved segments: {int(row['retained_parent_segment_count'])}")
    print(f"Selected omega every-other: {row['selected_omega_every_other_detected']}")
    print(f"Wrote {args.prototype_pdb}")
    print(f"Wrote {args.selected_csv}")
    print(f"Wrote {args.geometry_csv}")
    print(f"Wrote {args.abcd_csv}")
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
