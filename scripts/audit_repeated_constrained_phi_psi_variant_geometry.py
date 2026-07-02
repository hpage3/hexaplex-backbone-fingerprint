"""Audit repeated constrained phi/psi variants before diffraction scoring."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_constrained_phi_psi_candidate_geometry import (
    Atom,
    atom_map,
    audit_candidate,
    distance,
    parse_pdb,
)


DEFAULT_MANIFEST = Path("outputs/metrics/repeated_constrained_phi_psi_variant_manifest.csv")
DEFAULT_SOURCE_PDB = Path("outputs/coordinates/ideal_hexaflex_variants/ideal_hexaflex_backbone_plus_carboxylate.pdb")
DEFAULT_OUT_CSV = Path("outputs/metrics/repeated_constrained_phi_psi_variant_geometry_audit.csv")
DEFAULT_REPORT = Path("outputs/reports/repeated_constrained_phi_psi_variant_geometry_audit.md")

MAX_CA_SHIFT_A = 1e-3
MAX_BACKBONE_BOND_DELTA_A = 0.05
MAX_BACKBONE_ANGLE_DELTA_DEG = 5.0
MAX_OMEGA_TRANS_DEVIATION_DEG = 15.0


def atom_label_sets_match(parent_atoms: list[Atom], variant_atoms: list[Atom]) -> bool:
    """Return whether source and variant preserve atom labels exactly."""
    return set(atom_map(parent_atoms)) == set(atom_map(variant_atoms))


def max_ca_shift(parent_atoms: list[Atom], variant_atoms: list[Atom]) -> float:
    """Return max C-alpha coordinate shift among matching atom labels."""
    parent = atom_map(parent_atoms)
    variant = atom_map(variant_atoms)
    shifts = [
        distance(parent[key].coord, variant[key].coord)
        for key in sorted(set(parent) & set(variant))
        if parent[key].name == "CA"
    ]
    return max(shifts) if shifts else float("nan")


def max_non_anchor_atom_shift(parent_atoms: list[Atom], variant_atoms: list[Atom]) -> float:
    """Return max non-CA atom coordinate shift among matching atom labels."""
    parent = atom_map(parent_atoms)
    variant = atom_map(variant_atoms)
    shifts = [
        distance(parent[key].coord, variant[key].coord)
        for key in sorted(set(parent) & set(variant))
        if parent[key].name != "CA"
    ]
    return max(shifts) if shifts else float("nan")


def safe_for_diffraction(audit: dict[str, object]) -> bool:
    """Classify whether a repeated variant passes fixed-omega geometry gates."""
    return (
        bool(audit.get("candidate_file_exists"))
        and bool(audit.get("atom_count_match"))
        and bool(audit.get("labels_preserved"))
        and float(audit.get("max_ca_shift_A", math.inf)) <= MAX_CA_SHIFT_A
        and float(audit.get("max_backbone_bond_delta_A", math.inf)) <= MAX_BACKBONE_BOND_DELTA_A
        and float(audit.get("max_backbone_angle_delta_deg", math.inf)) <= MAX_BACKBONE_ANGLE_DELTA_DEG
        and float(audit.get("max_omega_trans_deviation_deg", math.inf)) <= MAX_OMEGA_TRANS_DEVIATION_DEG
    )


def failure_reasons(audit: dict[str, object]) -> list[str]:
    """Return threshold failures for one audit row."""
    reasons = []
    if not audit.get("candidate_file_exists"):
        reasons.append("missing_variant_file")
    if not audit.get("atom_count_match"):
        reasons.append("atom_count_mismatch")
    if not audit.get("labels_preserved"):
        reasons.append("labels_not_preserved")
    if float(audit.get("max_ca_shift_A", math.inf)) > MAX_CA_SHIFT_A:
        reasons.append("ca_anchor_shift_exceeds_threshold")
    if float(audit.get("max_backbone_bond_delta_A", math.inf)) > MAX_BACKBONE_BOND_DELTA_A:
        reasons.append("backbone_bond_delta_exceeds_threshold")
    if float(audit.get("max_backbone_angle_delta_deg", math.inf)) > MAX_BACKBONE_ANGLE_DELTA_DEG:
        reasons.append("backbone_angle_delta_exceeds_threshold")
    if float(audit.get("max_omega_trans_deviation_deg", math.inf)) > MAX_OMEGA_TRANS_DEVIATION_DEG:
        reasons.append("omega_trans_deviation_exceeds_threshold")
    return reasons


def resolve_repo_path(path_text: object) -> Path:
    """Resolve a manifest path relative to the repo root if needed."""
    path = Path(str(path_text))
    return path if path.is_absolute() else ROOT / path


def join_manifest_audit(manifest_row: pd.Series, audit: dict[str, object]) -> dict[str, object]:
    """Merge manifest metadata with audit metrics in report column order."""
    safe = safe_for_diffraction(audit)
    reasons = failure_reasons(audit)
    return {
        "variant_id": manifest_row.get("variant_id", ""),
        "fixed_torsion_delta_deg": manifest_row.get("fixed_torsion_delta_deg", ""),
        "omega_policy": manifest_row.get("omega_policy", ""),
        "attempted_window_count": manifest_row.get("attempted_window_count", ""),
        "applied_window_count": manifest_row.get("applied_window_count", ""),
        "skipped_window_count": manifest_row.get("skipped_window_count", ""),
        "coordinate_path": manifest_row.get("coordinate_path", ""),
        "candidate_file_exists": audit.get("candidate_file_exists", False),
        "atom_count_parent": audit.get("atom_count_parent", ""),
        "atom_count_candidate": audit.get("atom_count_candidate", ""),
        "atom_count_match": audit.get("atom_count_match", False),
        "labels_preserved": audit.get("labels_preserved", False),
        "missing_label_count": audit.get("missing_label_count", ""),
        "extra_label_count": audit.get("extra_label_count", ""),
        "max_ca_shift_A": audit.get("max_ca_shift_A", ""),
        "max_atom_shift_A": audit.get("max_atom_shift_A", ""),
        "max_non_anchor_atom_shift_A": audit.get("max_non_anchor_atom_shift_A", ""),
        "max_backbone_bond_delta_A": audit.get("max_backbone_bond_delta_A", ""),
        "max_backbone_angle_delta_deg": audit.get("max_backbone_angle_delta_deg", ""),
        "omega_count": audit.get("omega_count", ""),
        "max_omega_trans_deviation_deg": audit.get("max_omega_trans_deviation_deg", ""),
        "median_omega_trans_deviation_deg": audit.get("median_omega_trans_deviation_deg", ""),
        "safe_for_diffraction_scoring": safe,
        "failure_reasons": ";".join(reasons),
        "notes": manifest_row.get("notes", ""),
    }


def audit_variant(parent_atoms: list[Atom], variant_path: Path) -> dict[str, object]:
    """Audit one repeated variant, handling missing files clearly."""
    if not variant_path.exists():
        return {
            "candidate_file_exists": False,
            "atom_count_parent": len(parent_atoms),
            "atom_count_candidate": 0,
            "atom_count_match": False,
            "labels_preserved": False,
            "missing_label_count": "",
            "extra_label_count": "",
            "max_ca_shift_A": math.inf,
            "max_atom_shift_A": math.inf,
            "max_non_anchor_atom_shift_A": math.inf,
            "max_backbone_bond_delta_A": math.inf,
            "max_backbone_angle_delta_deg": math.inf,
            "omega_count": 0,
            "max_omega_trans_deviation_deg": math.inf,
            "median_omega_trans_deviation_deg": math.inf,
        }
    audit = audit_candidate(parent_atoms, variant_path)
    variant_atoms = parse_pdb(variant_path)
    audit["max_non_anchor_atom_shift_A"] = max_non_anchor_atom_shift(parent_atoms, variant_atoms)
    audit["safe_for_diffraction_scoring"] = safe_for_diffraction(audit)
    return audit


def build_report_text(results: pd.DataFrame, source_pdb: Path) -> str:
    """Build markdown report for repeated variant geometry audit."""
    total = len(results)
    safe_count = int(results["safe_for_diffraction_scoring"].astype(bool).sum()) if total else 0
    max_ca = pd.to_numeric(results.get("max_ca_shift_A", pd.Series(dtype=float)), errors="coerce").max()
    max_bond = pd.to_numeric(results.get("max_backbone_bond_delta_A", pd.Series(dtype=float)), errors="coerce").max()
    max_angle = pd.to_numeric(results.get("max_backbone_angle_delta_deg", pd.Series(dtype=float)), errors="coerce").max()
    max_omega = pd.to_numeric(results.get("max_omega_trans_deviation_deg", pd.Series(dtype=float)), errors="coerce").max()
    failed = results[~results["safe_for_diffraction_scoring"].astype(bool)] if total else results
    fail_text = markdown_table(
        failed[
            ["variant_id", "fixed_torsion_delta_deg", "failure_reasons"]
        ]
    ) if not failed.empty else "_No variants failed the audit._"
    table = markdown_table(
        results[
            [
                "variant_id",
                "fixed_torsion_delta_deg",
                "max_ca_shift_A",
                "max_backbone_bond_delta_A",
                "max_backbone_angle_delta_deg",
                "max_omega_trans_deviation_deg",
                "safe_for_diffraction_scoring",
            ]
        ]
    ) if total else "_No repeated variants were audited._"
    return f"""# Repeated Constrained Phi/Psi Variant Geometry Audit

This audit compares repeated constrained phi/psi variants against the source backbone-plus-carboxylate model before diffraction scoring.

- Source PDB: `{source_pdb}`
- Repeated variants audited: {total}
- Safe for diffraction scoring: {safe_count}/{total}
- Maximum C-alpha shift: {max_ca:.6g} A
- Maximum backbone bond-length deviation from source: {max_bond:.6g} A
- Maximum backbone angle deviation from source: {max_angle:.6g} degrees
- Maximum omega trans deviation: {max_omega:.6g} degrees

## Thresholds

- Max C-alpha anchor shift: <= {MAX_CA_SHIFT_A:g} A
- Max backbone bond-length deviation: <= {MAX_BACKBONE_BOND_DELTA_A:g} A
- Max backbone angle deviation: <= {MAX_BACKBONE_ANGLE_DELTA_DEG:g} degrees
- Max omega trans deviation: <= {MAX_OMEGA_TRANS_DEVIATION_DEG:g} degrees

Omega is audited under Nick's current `fixed_180` policy. Omega sensitivity remains deferred.

## Per-Delta Summary

{table}

## Failed Variants

{fail_text}
"""


def markdown_table(df: pd.DataFrame) -> str:
    """Render a dataframe as markdown."""
    if df.empty:
        return "_No rows._"
    columns = list(df.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for record in df.itertuples(index=False):
        vals = []
        for value in record:
            if isinstance(value, float):
                vals.append(f"{value:.6g}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def run_audit(manifest_path: Path, source_pdb: Path, out_csv: Path, report_path: Path) -> pd.DataFrame:
    """Run repeated variant geometry audit and write outputs."""
    manifest = pd.read_csv(manifest_path)
    parent_atoms = parse_pdb(source_pdb)
    rows = []
    for _, manifest_row_data in manifest.iterrows():
        variant_path = resolve_repo_path(manifest_row_data["coordinate_path"])
        audit = audit_variant(parent_atoms, variant_path)
        rows.append(join_manifest_audit(manifest_row_data, audit))
    results = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_csv, index=False)
    report_path.write_text(build_report_text(results, source_pdb), encoding="utf-8")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-pdb", type=Path, default=DEFAULT_SOURCE_PDB)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = run_audit(args.manifest, args.source_pdb, args.out_csv, args.report)
    safe_count = int(results["safe_for_diffraction_scoring"].astype(bool).sum()) if not results.empty else 0
    print(f"Audited {len(results)} repeated variants")
    print(f"Safe for diffraction scoring: {safe_count}/{len(results)}")
    print(f"CSV: {args.out_csv}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
