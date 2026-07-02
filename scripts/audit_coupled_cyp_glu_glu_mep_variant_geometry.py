"""Audit coupled CYP->GLU plus GLU->MEP variants before C/D scoring."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_constrained_phi_psi_candidate_geometry import audit_candidate, parse_pdb
from scripts.audit_repeated_constrained_phi_psi_variant_geometry import failure_reasons, resolve_repo_path, safe_for_diffraction


DEFAULT_MANIFEST = Path("outputs/metrics/coupled_cyp_glu_glu_mep_variant_manifest.csv")
DEFAULT_SOURCE_PDB = Path("outputs/coordinates/ideal_hexaflex_variants/ideal_hexaflex_backbone_plus_carboxylate.pdb")
DEFAULT_OUT_CSV = Path("outputs/metrics/coupled_cyp_glu_glu_mep_variant_geometry_audit.csv")
DEFAULT_REPORT = Path("outputs/reports/coupled_cyp_glu_glu_mep_variant_geometry_audit.md")


def classify_geometry_metrics(metrics: dict[str, object]) -> tuple[bool, str]:
    """Classify metrics with existing geometry thresholds."""
    safe = safe_for_diffraction(metrics)
    return safe, ";".join(failure_reasons(metrics))


def audit_row(manifest_row: pd.Series, parent_atoms) -> dict[str, object]:
    """Audit one coupled manifest row."""
    output_pdb = resolve_repo_path(manifest_row["output_pdb"])
    metrics = audit_candidate(parent_atoms, output_pdb) if output_pdb.exists() else {
        "candidate_file_exists": False,
        "atom_count_match": False,
        "labels_preserved": False,
        "max_ca_shift_A": float("inf"),
        "max_backbone_bond_delta_A": float("inf"),
        "max_backbone_angle_delta_deg": float("inf"),
        "max_omega_trans_deviation_deg": float("inf"),
    }
    safe, failed = classify_geometry_metrics(metrics)
    return {
        "variant_id": manifest_row["variant_id"],
        "cyp_glu_delta_deg": manifest_row["cyp_glu_delta_deg"],
        "glu_mep_delta_deg": manifest_row["glu_mep_delta_deg"],
        "geometry_safe": safe,
        "max_ca_anchor_shift_A": metrics.get("max_ca_shift_A", ""),
        "max_bond_delta_A": metrics.get("max_backbone_bond_delta_A", ""),
        "max_angle_delta_deg": metrics.get("max_backbone_angle_delta_deg", ""),
        "max_omega_trans_deviation_deg": metrics.get("max_omega_trans_deviation_deg", ""),
        "failed_checks": failed,
        "output_pdb": manifest_row["output_pdb"],
    }


def run_audit(manifest_path: Path, source_pdb: Path, out_csv: Path, report_path: Path) -> pd.DataFrame:
    """Run coupled geometry audit."""
    manifest = pd.read_csv(manifest_path)
    parent_atoms = parse_pdb(source_pdb)
    results = pd.DataFrame([audit_row(row, parent_atoms) for _, row in manifest.iterrows()])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_csv, index=False)
    write_report(results, report_path)
    return results


def write_report(results: pd.DataFrame, path: Path) -> None:
    """Write coupled geometry audit report."""
    safe = results[results["geometry_safe"].astype(bool)]
    failed = results[~results["geometry_safe"].astype(bool)]
    text = f"""# Coupled CYP->GLU + GLU->MEP Geometry Audit

This is a coupled pilot geometry audit, not C/D diffraction scoring. Diffraction scoring should only be run on geometry-safe variants.

- Variants audited: {len(results)}
- Geometry-safe variants: {len(safe)}
- Safe variant IDs: {', '.join(safe['variant_id'].astype(str)) if not safe.empty else 'none'}

## Delta Pair Summary

{markdown_table(results[['variant_id', 'cyp_glu_delta_deg', 'glu_mep_delta_deg', 'geometry_safe', 'max_ca_anchor_shift_A', 'max_bond_delta_A', 'max_angle_delta_deg', 'max_omega_trans_deviation_deg', 'failed_checks']])}

## Failed Variants

{markdown_table(failed[['variant_id', 'cyp_glu_delta_deg', 'glu_mep_delta_deg', 'failed_checks']]) if not failed.empty else '_No variants failed._'}
"""
    path.write_text(text, encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> str:
    """Render dataframe as markdown."""
    columns = list(df.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for record in df.itertuples(index=False):
        lines.append("| " + " | ".join(f"{value:.6g}" if isinstance(value, float) else str(value) for value in record) + " |")
    return "\n".join(lines)


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
    print(f"Audited {len(results)} coupled variants")
    print(f"Geometry-safe variants: {int(results['geometry_safe'].astype(bool).sum())}/{len(results)}")
    print(f"CSV: {args.out_csv}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
