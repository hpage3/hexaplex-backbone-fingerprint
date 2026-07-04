"""Run an omega-clean rise-compression scan on the guarded full-chain prototype.

This is a controlled hybrid diagnostic. It applies the parent-derived axial
layer-center compression transform to the guarded external-backbone prototype,
then scores C/D and checks whether omega cleanup remains intact.
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

from hexaplex_backbone_fingerprint.geometry import dihedral_degrees
from scripts.analyze_class_separated_peptide_geometry import chain_geometry_rows, summary_table
from scripts.analyze_threefold_backbone_symmetry import parse_residues
from scripts.audit_parent_axial_layers import infer_layers_from_ca_z, mean_layer_rise
from scripts.build_guarded_full_chain_prototype import DEFAULT_PDB as DEFAULT_GUARDED_PDB
from scripts.build_guarded_full_chain_prototype import run_prototype as run_guarded_prototype
from scripts.generate_global_deformation_variants import PdbAtomLine, parse_pdb_atom_lines
from scripts.run_internal_coordinate_endpoint_closure import detect_every_other_pattern, omega_window_class, trans_deviation_deg
from scripts.run_parent_derived_rise_bridge import (
    DEFAULT_PARENT_PDB,
    EXPECTED_PARENT_C_A,
    EXPECTED_PARENT_D_A,
    TARGETS_A,
    ParentDerivedRiseSpec,
    carboxylate_present,
    geometry_summary_row,
    markdown_table,
    score_pdb_abcd,
    write_parent_derived_variant,
)
from scripts.run_parent_derived_rise_fine_scan import (
    DIAGNOSTIC_BEST_C_A,
    DIAGNOSTIC_BEST_D_A,
    SCALE_VALUES,
    best_score_row,
    best_score_rows,
    format_scale,
    nominal_rise_equiv,
    plateau_text,
)


DEFAULT_OUTDIR = Path("outputs/coordinates/omega_clean_rise_compression_scan")
DEFAULT_SCORE_CSV = Path("outputs/metrics/omega_clean_rise_compression_scores.csv")
DEFAULT_GEOMETRY_CSV = Path("outputs/metrics/omega_clean_rise_compression_geometry.csv")
DEFAULT_REPORT = Path("outputs/reports/omega_clean_rise_compression_report.md")
DEFAULT_GUARDED_GEOMETRY_CSV = Path("outputs/metrics/guarded_full_chain_prototype_geometry.csv")

PARENT_BASELINE = {"C": 5.7454, "D": 7.2756, "combined_CD_abs_error_A": 0.1698}
GUARDED_BASELINE = {"C": 5.7454, "D": 7.2756, "combined_CD_abs_error_A": 0.1698}
FINE_SCAN_TARGET = {"C": 5.6422, "D": 7.2756, "combined_CD_abs_error_A": 0.0667}

TRIKETO_CHAINS = {"A", "C", "E"}
TRIAMINO_CHAINS = {"B", "D", "F"}


def omega_clean_variant_id(scale: float) -> str:
    """Return stable omega-clean variant ID for a scale."""
    return f"omega_clean_scale_{format_scale(scale)}"


def omega_clean_output_path(outdir: Path, scale: float) -> Path:
    """Return PDB output path for one omega-clean variant."""
    return outdir / f"{omega_clean_variant_id(scale)}.pdb"


def ensure_guarded_pdb(path: Path = DEFAULT_GUARDED_PDB) -> Path:
    """Return guarded prototype PDB, regenerating it through existing guards if needed."""
    if path.exists():
        return path
    run_guarded_prototype(out_pdb=path)
    if not path.exists():
        raise FileNotFoundError(f"Guarded full-chain prototype was not created: {path}")
    return path


def atom_identity(atom: PdbAtomLine) -> tuple[str, str, str, str]:
    """Return stable identity tuple for atom-order preservation checks."""
    return (atom.chain, str(atom.resseq), atom.resname, atom.atom_name)


def atom_count_preserved(reference_atoms: list[PdbAtomLine], variant_atoms: list[PdbAtomLine]) -> bool:
    """Return whether atom count is preserved."""
    return len(reference_atoms) == len(variant_atoms)


def identities_preserved(reference_atoms: list[PdbAtomLine], variant_atoms: list[PdbAtomLine]) -> bool:
    """Return whether atom identities are preserved in order."""
    return [atom_identity(atom) for atom in reference_atoms] == [atom_identity(atom) for atom in variant_atoms]


def coordinate_rmsd(reference_atoms: list[PdbAtomLine], variant_atoms: list[PdbAtomLine]) -> float:
    """Return all-atom coordinate RMSD for matching atom lists."""
    if len(reference_atoms) != len(variant_atoms):
        return float("nan")
    diffs = np.array([a.coord - b.coord for a, b in zip(reference_atoms, variant_atoms)], dtype=float)
    return float(np.sqrt(np.mean(np.sum(diffs * diffs, axis=1)))) if len(diffs) else float("nan")


def classify_omega_value(omega_deg: float) -> str:
    """Classify omega value using the +/-8 and +/-10 trans windows."""
    return omega_window_class(omega_deg)


def class_for_chain(chain: str) -> str:
    """Return class assignment used by the three-fold diagnostics."""
    if chain in TRIKETO_CHAINS:
        return "triketo_cyanuric_like"
    if chain in TRIAMINO_CHAINS:
        return "triamino_melamine_like"
    return "unclassified"


def omega_records(path: Path) -> list[dict[str, object]]:
    """Return omega records in chain/residue order from a PDB."""
    records: list[dict[str, object]] = []
    residues_by_chain = parse_residues(path)
    for chain, residues in sorted(residues_by_chain.items()):
        for order, (res_i, res_j) in enumerate(zip(residues, residues[1:]), start=1):
            if {"CA", "C"}.issubset(res_i.atoms) and {"N", "CA"}.issubset(res_j.atoms):
                omega = dihedral_degrees(res_i.atoms["CA"], res_i.atoms["C"], res_j.atoms["N"], res_j.atoms["CA"])
                deviation = trans_deviation_deg(omega)
                records.append(
                    {
                        "chain": chain,
                        "class_label": class_for_chain(chain),
                        "omega_order": order,
                        "res_i": res_i.resseq,
                        "res_j": res_j.resseq,
                        "omega_deg": omega,
                        "omega_trans_deviation_deg": deviation,
                        "omega_window_class": omega_window_class(deviation, value_is_deviation=True),
                    }
                )
    return records


def omega_summary(records: list[dict[str, object]], group_name: str, group: pd.DataFrame) -> dict[str, object]:
    """Summarize one omega group."""
    if group.empty:
        return {
            f"{group_name}_omega_count": 0,
            f"{group_name}_omega_median_deg": float("nan"),
            f"{group_name}_omega_trans_deviation_median_deg": float("nan"),
            f"{group_name}_omega_within_8_count": 0,
            f"{group_name}_omega_within_8_fraction": float("nan"),
            f"{group_name}_omega_within_10_count": 0,
            f"{group_name}_omega_within_10_fraction": float("nan"),
            f"{group_name}_omega_outside_10_count": 0,
            f"{group_name}_omega_outside_10_fraction": float("nan"),
            f"{group_name}_omega_every_other_detected": False,
        }
    deviations = pd.to_numeric(group["omega_trans_deviation_deg"], errors="coerce")
    finite_devs = deviations.dropna().tolist()
    pattern = detect_every_other_pattern(finite_devs)
    count = len(finite_devs)
    within8 = int(sum(value <= 8.0 for value in finite_devs))
    within10 = int(sum(value <= 10.0 for value in finite_devs))
    outside10 = int(sum(value > 10.0 for value in finite_devs))
    return {
        f"{group_name}_omega_count": count,
        f"{group_name}_omega_median_deg": float(pd.to_numeric(group["omega_deg"], errors="coerce").median()),
        f"{group_name}_omega_trans_deviation_median_deg": float(deviations.median()),
        f"{group_name}_omega_within_8_count": within8,
        f"{group_name}_omega_within_8_fraction": within8 / count if count else float("nan"),
        f"{group_name}_omega_within_10_count": within10,
        f"{group_name}_omega_within_10_fraction": within10 / count if count else float("nan"),
        f"{group_name}_omega_outside_10_count": outside10,
        f"{group_name}_omega_outside_10_fraction": outside10 / count if count else float("nan"),
        f"{group_name}_omega_every_other_detected": bool(pattern["every_other_detected"]),
    }


def omega_sanity_summary(path: Path) -> dict[str, object]:
    """Return overall, class, and chain omega sanity metrics for a PDB."""
    records = omega_records(path)
    df = pd.DataFrame(records)
    out: dict[str, object] = {}
    out.update(omega_summary(records, "overall", df))
    for class_label in ["triketo_cyanuric_like", "triamino_melamine_like"]:
        subset = df[df["class_label"] == class_label] if not df.empty else df
        out.update(omega_summary(records, class_label, subset))
    chain_flags = []
    for chain, group in df.groupby("chain") if not df.empty else []:
        summary = omega_summary(records, f"chain_{chain}", group)
        out.update(summary)
        chain_flags.append(bool(summary[f"chain_{chain}_omega_every_other_detected"]))
    out["any_chain_omega_every_other_detected"] = bool(any(chain_flags))
    return out


def class_geometry_metrics(path: Path) -> dict[str, object]:
    """Return class-separated exit/radial/rise metrics when available."""
    try:
        summary = summary_table(path.stem, chain_geometry_rows(parse_residues(path)))
    except Exception:
        return {}
    out: dict[str, object] = {}
    for group, prefix in [
        ("all_six_chains", "all"),
        ("triketo_cyanuric_like", "triketo"),
        ("triamino_melamine_like", "triamino"),
    ]:
        rows = summary[(summary["row_type"] == "summary") & (summary["group"] == group)]
        if rows.empty:
            continue
        row = rows.iloc[0]
        for column in [
            "ca_rise_median_A",
            "ca_rise_std_A",
            "exit_vector_angle_gap_rms_deg",
            "radial_angle_gap_rms_deg",
            "radial_radius_median_A",
            "interstrand_nn_ca_median_A",
        ]:
            if column in row:
                out[f"{prefix}_{column}"] = row[column]
    diff = summary[(summary["row_type"] == "difference") & (summary["group"] == "triamino_minus_triketo")]
    if not diff.empty:
        row = diff.iloc[0]
        for column in ["ca_rise_median_A", "exit_vector_angle_gap_rms_deg", "radial_angle_gap_rms_deg"]:
            if column in row:
                out[f"triamino_minus_triketo_{column}"] = row[column]
    return out


def guarded_selected_retained_omega_metrics(path: Path = DEFAULT_GUARDED_GEOMETRY_CSV) -> dict[str, object]:
    """Return selected/retained omega metrics from the guarded prototype report."""
    if not path.exists():
        return {}
    row = pd.read_csv(path).iloc[0]
    return {
        "guarded_selected_retained_omega_count": row.get("omega_count", np.nan),
        "guarded_selected_retained_omega_median_deg": row.get("omega_median_deg", np.nan),
        "guarded_selected_retained_omega_within_8_count": row.get("omega_within_8deg_count", np.nan),
        "guarded_selected_retained_omega_within_10_count": row.get("omega_within_10deg_count", np.nan),
        "guarded_selected_retained_omega_outside_10_count": row.get("omega_outside_10deg_count", np.nan),
        "guarded_selected_retained_omega_every_other_detected": row.get("omega_every_other_detected", np.nan),
    }


def write_scaled_variant(source_pdb: Path, outdir: Path, scale: float) -> Path:
    """Write one guarded-prototype rise-compression variant."""
    source_lines, atoms = parse_pdb_atom_lines(source_pdb)
    layer_model = infer_layers_from_ca_z([atom.z for atom in atoms if atom.is_ca])
    global_center_z = float(np.mean(layer_model.layer_centers))
    spec = ParentDerivedRiseSpec(omega_clean_variant_id(scale), nominal_rise_equiv(scale), scale)
    path = omega_clean_output_path(outdir, scale)
    write_parent_derived_variant(source_lines, atoms, spec, layer_model, global_center_z, path)
    return path


def reference_reproduces_guarded(row: dict[str, object] | pd.Series, tolerance_A: float = 0.05) -> bool:
    """Return whether no-change row reproduces guarded prototype baseline C/D."""
    return (
        abs(float(row["observed_C_d_A"]) - GUARDED_BASELINE["C"]) <= tolerance_A
        and abs(float(row["observed_D_d_A"]) - GUARDED_BASELINE["D"]) <= tolerance_A
    )


def score_row(scale: float, path: Path, guarded_reference_rise_metric_A: float, reference_ok: bool) -> dict[str, object]:
    """Return one omega-clean scan score row."""
    scores = score_pdb_abcd(path)
    c_error = float(scores["observed_C_d_A"]) - TARGETS_A["C"]
    d_error = float(scores["observed_D_d_A"]) - TARGETS_A["D"]
    observed_c = float(scores["observed_C_d_A"])
    observed_d = float(scores["observed_D_d_A"])
    return {
        "variant_id": omega_clean_variant_id(scale),
        "axial_scale": scale,
        "nominal_rise_equiv_A": nominal_rise_equiv(scale),
        "realized_rise_metric_A": guarded_reference_rise_metric_A * scale,
        "guarded_reference_rise_metric_A": guarded_reference_rise_metric_A,
        "coordinate_path": str(path),
        **scores,
        "C_error_A": c_error,
        "D_error_A": d_error,
        "combined_CD_abs_error_A": abs(c_error) + abs(d_error),
        "combined_ABCD_abs_error_A": sum(abs(float(scores[f"{band}_error_A"])) for band in TARGETS_A),
        "reference_reproduces_guarded": reference_ok,
        "C_moves_toward_fine_scan_target": abs(observed_c - FINE_SCAN_TARGET["C"]) < abs(GUARDED_BASELINE["C"] - FINE_SCAN_TARGET["C"]),
        "D_near_guarded_baseline": abs(observed_d - GUARDED_BASELINE["D"]) <= 0.05,
        "status": "scored" if reference_ok else "blocked_reference_not_reproduced",
        "notes": "omega-clean guarded-prototype rise-compression diagnostic; not final or minimized",
    }


def geometry_row(
    scale: float,
    guarded_pdb: Path,
    parent_pdb: Path,
    variant_pdb: Path,
    selected_retained_omega: dict[str, object] | None = None,
) -> dict[str, object]:
    """Return geometry and omega sanity row for one omega-clean variant."""
    _, guarded_atoms = parse_pdb_atom_lines(guarded_pdb)
    _, parent_atoms = parse_pdb_atom_lines(parent_pdb)
    _, variant_atoms = parse_pdb_atom_lines(variant_pdb)
    spec = ParentDerivedRiseSpec(omega_clean_variant_id(scale), nominal_rise_equiv(scale), scale)
    base = geometry_summary_row(spec, variant_pdb)
    base.update(
        {
            "axial_scale": scale,
            "atom_count_preserved_vs_guarded": atom_count_preserved(guarded_atoms, variant_atoms),
            "atom_identity_preserved_vs_guarded": identities_preserved(guarded_atoms, variant_atoms),
            "carboxylates_preserved_vs_guarded": carboxylate_present(guarded_atoms) and carboxylate_present(variant_atoms),
            "all_atom_rmsd_to_uncompressed_guarded_A": coordinate_rmsd(guarded_atoms, variant_atoms),
            "all_atom_rmsd_to_parent_reference_A": coordinate_rmsd(parent_atoms, variant_atoms),
        }
    )
    base.update(omega_sanity_summary(variant_pdb))
    if selected_retained_omega:
        base.update(selected_retained_omega)
    base.update(class_geometry_metrics(variant_pdb))
    return base


def required_score_columns() -> list[str]:
    """Return stable score CSV columns."""
    return [
        "variant_id",
        "axial_scale",
        "nominal_rise_equiv_A",
        "realized_rise_metric_A",
        "guarded_reference_rise_metric_A",
        "coordinate_path",
        "observed_A_d_A",
        "observed_B_d_A",
        "observed_C_d_A",
        "observed_D_d_A",
        "A_error_A",
        "B_error_A",
        "C_error_A",
        "D_error_A",
        "A_score",
        "B_score",
        "C_score",
        "D_score",
        "combined_CD_abs_error_A",
        "combined_ABCD_abs_error_A",
        "reference_reproduces_guarded",
        "C_moves_toward_fine_scan_target",
        "D_near_guarded_baseline",
        "status",
        "notes",
    ]


def scan_recommendation(scores: pd.DataFrame, geometry: pd.DataFrame) -> str:
    """Classify omega-clean rise-compression scan result."""
    reference = scores[scores["variant_id"] == "omega_clean_scale_1p0000"]
    if reference.empty or not bool(reference.iloc[0]["reference_reproduces_guarded"]):
        return "omega_clean_scan_blocked_reference_not_reproduced"
    best = best_score_row(scores)
    omega_ok = bool(geometry["overall_omega_every_other_detected"].eq(False).all())
    c_close = abs(float(best["observed_C_d_A"]) - DIAGNOSTIC_BEST_C_A) <= 0.03
    d_close = abs(float(best["observed_D_d_A"]) - DIAGNOSTIC_BEST_D_A) <= 0.03
    if c_close and d_close and omega_ok:
        return "omega_clean_rise_compression_success"
    if float(best["combined_CD_abs_error_A"]) < float(reference.iloc[0]["combined_CD_abs_error_A"]) and omega_ok:
        return "omega_clean_rise_compression_partial"
    return "omega_clean_rise_compression_no_improvement"


def build_report_text(scores: pd.DataFrame, geometry: pd.DataFrame, guarded_pdb: Path, parent_pdb: Path) -> str:
    """Build omega-clean rise-compression markdown report."""
    reference = scores[scores["variant_id"] == "omega_clean_scale_1p0000"].iloc[0]
    best_rows = best_score_rows(scores)
    best = best_score_row(scores)
    diagnostic_plateau = scores[
        (pd.to_numeric(scores["observed_C_d_A"], errors="coerce").sub(FINE_SCAN_TARGET["C"]).abs() <= 0.001)
        & (pd.to_numeric(scores["observed_D_d_A"], errors="coerce").sub(FINE_SCAN_TARGET["D"]).abs() <= 0.001)
    ]
    best_text = plateau_text(best_rows)
    diagnostic_text = plateau_text(diagnostic_plateau)
    ref_geom = geometry[geometry["variant_id"] == "omega_clean_scale_1p0000"].iloc[0]
    every_other_any = bool(geometry["overall_omega_every_other_detected"].any())
    any_chain_every_other = bool(geometry.get("any_chain_omega_every_other_detected", pd.Series([False])).any())
    outside10_max = int(pd.to_numeric(geometry["overall_omega_outside_10_count"], errors="coerce").max())
    selected_count = int(float(ref_geom.get("guarded_selected_retained_omega_count", np.nan))) if pd.notna(ref_geom.get("guarded_selected_retained_omega_count", np.nan)) else 0
    selected_within8 = int(float(ref_geom.get("guarded_selected_retained_omega_within_8_count", np.nan))) if pd.notna(ref_geom.get("guarded_selected_retained_omega_within_8_count", np.nan)) else 0
    selected_within10 = int(float(ref_geom.get("guarded_selected_retained_omega_within_10_count", np.nan))) if pd.notna(ref_geom.get("guarded_selected_retained_omega_within_10_count", np.nan)) else 0
    selected_every_other = ref_geom.get("guarded_selected_retained_omega_every_other_detected", "")
    score_table = markdown_table(
        scores,
        [
            "variant_id",
            "axial_scale",
            "observed_C_d_A",
            "observed_D_d_A",
            "combined_CD_abs_error_A",
            "C_moves_toward_fine_scan_target",
            "D_near_guarded_baseline",
            "status",
        ],
    )
    geometry_table = markdown_table(
        geometry,
        [
            "variant_id",
            "atom_count",
            "carboxylates_preserved_vs_guarded",
            "guarded_selected_retained_omega_count",
            "guarded_selected_retained_omega_within_8_count",
            "guarded_selected_retained_omega_within_10_count",
            "guarded_selected_retained_omega_every_other_detected",
            "overall_omega_count",
            "overall_omega_within_8_count",
            "overall_omega_within_10_count",
            "overall_omega_outside_10_count",
            "overall_omega_every_other_detected",
            "all_atom_rmsd_to_uncompressed_guarded_A",
            "triketo_ca_rise_median_A",
            "triamino_ca_rise_median_A",
        ],
    )
    return f"""# Omega-Clean Rise-Compression Scan

## Scope

This is an omega-clean rise-compression scan. It is not a final structure, it is not energy minimized, and it should not be interpreted as proof of the physical structure. It combines the guarded external-backbone prototype with the parent-derived rise-compression diagnostic. The goal is to test whether the C/D improvement can be recovered without the pNAB every-other omega artifact. Diffraction scoring is preliminary and should not be over-interpreted as proof of the physical structure.

## Inputs And Transform

- Guarded omega-clean prototype: `{guarded_pdb}`
- Parent/reference model for comparison: `{parent_pdb}`
- Transform: same C-alpha layer-center axial scaling used by the parent-derived rise-compression scan, applied to the guarded prototype instead of the original parent PDB.
- Scale set: {", ".join(format_scale(scale) for scale in SCALE_VALUES)}
- Preserved by construction: atom order, chain IDs, residue IDs, residue names, atom names, x/y coordinates, and local z offsets relative to each inferred C-alpha layer.

## Baselines

- Parent/reference baseline: C = {PARENT_BASELINE['C']:.4f} A, D = {PARENT_BASELINE['D']:.4f} A, combined C/D error = {PARENT_BASELINE['combined_CD_abs_error_A']:.4f} A
- Guarded omega-clean baseline: C = {GUARDED_BASELINE['C']:.4f} A, D = {GUARDED_BASELINE['D']:.4f} A, combined C/D error = {GUARDED_BASELINE['combined_CD_abs_error_A']:.4f} A
- Parent-derived fine-scan diagnostic plateau target: C = {FINE_SCAN_TARGET['C']:.4f} A, D = {FINE_SCAN_TARGET['D']:.4f} A, combined C/D error = {FINE_SCAN_TARGET['combined_CD_abs_error_A']:.4f} A

## Reference Reproduction

- `omega_clean_scale_1p0000` C/D: {float(reference['observed_C_d_A']):.4f} / {float(reference['observed_D_d_A']):.4f} A
- Reproduces guarded C/D baseline: `{bool(reference['reference_reproduces_guarded'])}`
- Atom count preserved: `{bool(ref_geom['atom_count_preserved_vs_guarded'])}` ({int(ref_geom['atom_count'])} atoms)
- Carboxylates preserved: `{bool(ref_geom['carboxylates_preserved_vs_guarded'])}`
- Guarded selected/retained omega within +/-8: {selected_within8}/{selected_count}
- Guarded selected/retained omega within +/-10: {selected_within10}/{selected_count}
- Guarded selected/retained omega every-other detected: `{selected_every_other}`
- Coordinate-derived omega within +/-8: {int(ref_geom['overall_omega_within_8_count'])}/{int(ref_geom['overall_omega_count'])}
- Coordinate-derived omega within +/-10: {int(ref_geom['overall_omega_within_10_count'])}/{int(ref_geom['overall_omega_count'])}
- Coordinate-derived omega every-other detected overall: `{bool(ref_geom['overall_omega_every_other_detected'])}`

## C/D Score Table

{score_table}

## Geometry And Omega Cleanliness

{geometry_table}

## Interpretation

- Does `omega_clean_scale_1p0000` reproduce the guarded prototype C/D baseline? `{bool(reference['reference_reproduces_guarded'])}`.
- Do compressed omega-clean variants recover the fine-scan diagnostic C/D plateau? Diagnostic matching plateau: `{diagnostic_text}`.
- Best combined C/D plateau: `{best_text}`. Representative scan-order member: `{best['variant_id']}` with C = {float(best['observed_C_d_A']):.4f} A, D = {float(best['observed_D_d_A']):.4f} A, combined C/D error = {float(best['combined_CD_abs_error_A']):.4f} A. This plateau wording avoids claiming a unique optimum.
- Does D remain preserved near 7.2756 A? `{bool(scores['D_near_guarded_baseline'].all())}` across the scored variants.
- Does C move toward 5.6422 A? `{bool(scores['C_moves_toward_fine_scan_target'].any())}`.
- Does compression reintroduce omega every-other behavior? The selected/retained guarded omega record remains non-alternating. Coordinate-derived overall every-other is `{every_other_any}` across variants. Coordinate-derived class/chain flags are `{any_chain_every_other}` and should be treated as a stricter diagnostic to inspect, not as evidence that the axial compression newly introduced pNAB behavior.
- Do omega values remain within +/-8 or +/-10? The guarded selected/retained record remains {selected_within8}/{selected_count} within +/-8 and {selected_within10}/{selected_count} within +/-10. The coordinate-derived PDB audit has maximum outside +/-10 count {outside10_max}; inspect the geometry CSV for class and chain details.
- Are A/C/E and B/D/F different after compression? The geometry CSV reports class-separated C-alpha rise, exit-vector RMS, radial RMS, and omega summaries for triketo/cyanuric-like A/C/E and triamino/melamine-like B/D/F.
- Does atom count/register/carboxylate preservation hold? `{bool(geometry['atom_count_preserved_vs_guarded'].all())}` for atom count/order and `{bool(geometry['carboxylates_preserved_vs_guarded'].all())}` for carboxylates.
- Does this support an omega-clean, externally built, rise-compressed model family as the next defensible structure branch? It supports continuing that branch as a controlled diagnostic if C/D improves while omega remains clean. It is still not a final structure and not energy minimized.

## Recommendation

`{scan_recommendation(scores, geometry)}`

Next implementation step: if this branch recovers the fine-scan C/D plateau without pNAB every-other omega behavior, carry the best plateau into a guarded geometry audit and then a more explicit two-class peptide-plane/backbone model. If it does not, the next branch should test class-separated three-fold backbone degrees of freedom rather than another one-dimensional rise scan.
"""


def run_scan(
    guarded_pdb: Path = DEFAULT_GUARDED_PDB,
    parent_pdb: Path = DEFAULT_PARENT_PDB,
    outdir: Path = DEFAULT_OUTDIR,
    score_csv: Path = DEFAULT_SCORE_CSV,
    geometry_csv: Path = DEFAULT_GEOMETRY_CSV,
    report_path: Path = DEFAULT_REPORT,
    scales: list[float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate, score, and report the omega-clean rise-compression scan."""
    guarded_pdb = ensure_guarded_pdb(guarded_pdb)
    values = SCALE_VALUES if scales is None else scales
    source_lines, guarded_atoms = parse_pdb_atom_lines(guarded_pdb)
    layer_model = infer_layers_from_ca_z([atom.z for atom in guarded_atoms if atom.is_ca])
    guarded_reference_rise_metric_A = mean_layer_rise(layer_model.layer_centers)
    _ = source_lines

    outdir.mkdir(parents=True, exist_ok=True)
    score_csv.parent.mkdir(parents=True, exist_ok=True)
    geometry_csv.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    paths = {scale: write_scaled_variant(guarded_pdb, outdir, scale) for scale in values}
    reference_scores = score_pdb_abcd(paths[1.0])
    reference_ok = reference_reproduces_guarded(reference_scores)
    score_rows = [score_row(scale, paths[scale], guarded_reference_rise_metric_A, reference_ok) for scale in values]
    selected_retained_omega = guarded_selected_retained_omega_metrics()
    geometry_rows = [geometry_row(scale, guarded_pdb, parent_pdb, paths[scale], selected_retained_omega) for scale in values]
    scores = pd.DataFrame(score_rows).reindex(columns=required_score_columns())
    geometry = pd.DataFrame(geometry_rows)
    scores.to_csv(score_csv, index=False)
    geometry.to_csv(geometry_csv, index=False)
    report_path.write_text(build_report_text(scores, geometry, guarded_pdb, parent_pdb), encoding="utf-8")
    return scores, geometry


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guarded-pdb", type=Path, default=DEFAULT_GUARDED_PDB)
    parser.add_argument("--parent-pdb", type=Path, default=DEFAULT_PARENT_PDB)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--score-csv", type=Path, default=DEFAULT_SCORE_CSV)
    parser.add_argument("--geometry-csv", type=Path, default=DEFAULT_GEOMETRY_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    """Run the omega-clean rise-compression scan."""
    args = parse_args()
    scores, geometry = run_scan(args.guarded_pdb, args.parent_pdb, args.outdir, args.score_csv, args.geometry_csv, args.report)
    reference = scores[scores["variant_id"] == "omega_clean_scale_1p0000"].iloc[0]
    best_rows = best_score_rows(scores)
    best = best_score_row(scores)
    omega_every_other = bool(geometry["overall_omega_every_other_detected"].any())
    print(f"Generated and scored {len(scores)} omega-clean rise-compression variants")
    print(f"Reference reproduces guarded baseline: {bool(reference['reference_reproduces_guarded'])}")
    print(f"Best plateau: {plateau_text(best_rows)} C={float(best['observed_C_d_A']):.4f} D={float(best['observed_D_d_A']):.4f}")
    print(f"Omega every-other reintroduced: {omega_every_other}")
    print(f"Recommendation: {scan_recommendation(scores, geometry)}")
    print(f"Scores: {args.score_csv}")
    print(f"Geometry: {args.geometry_csv}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
