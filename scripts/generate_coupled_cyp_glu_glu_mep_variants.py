"""Generate small coupled CYP->GLU plus GLU->MEP constrained variants."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_backbone_torsion_repeat import parse_residues
from scripts.generate_constrained_phi_psi_candidates import read_pdb_lines, write_xyz_from_pdb_lines
from scripts.generate_repeated_constrained_phi_psi_variants import (
    apply_displacements,
    identify_cyp_glu_windows,
    max_ca_anchor_shift,
    safe_local_cyp_glu_rows,
    template_displacements,
)
from scripts.generate_repeated_glu_mep_baseline_omega_variants import (
    identify_glu_mep_windows,
    local_candidate_lines,
    safe_baseline_parent_glu_mep_rows,
    select_representative_rows,
)


DEFAULT_SOURCE_PDB = Path("outputs/coordinates/ideal_hexaflex_variants/ideal_hexaflex_backbone_plus_carboxylate.pdb")
DEFAULT_CYP_CD_SCORES = Path("outputs/metrics/constrained_phi_psi_candidate_cd_scores.csv")
DEFAULT_CYP_GEOMETRY_AUDIT = Path("outputs/metrics/constrained_phi_psi_candidate_geometry_audit.csv")
DEFAULT_GLU_COMPARISON = Path("outputs/metrics/glu_mep_omega_mode_closure_comparison.csv")
DEFAULT_OUTDIR = Path("outputs/coordinates/coupled_cyp_glu_glu_mep_variants")
DEFAULT_MANIFEST = Path("outputs/metrics/coupled_cyp_glu_glu_mep_variant_manifest.csv")
DEFAULT_REPORT = Path("outputs/reports/coupled_cyp_glu_glu_mep_variant_generation.md")
CYP_DELTAS = [-1.0, 0.0, 1.0]
GLU_DELTAS = [-1.0, 0.0]
CYP_OMEGA_MODE = "fixed_180"
GLU_OMEGA_MODE = "baseline_parent"


def coupled_delta_grid() -> list[tuple[float, float]]:
    """Return the deliberately small coupled delta grid."""
    return [(cyp, glu) for cyp in CYP_DELTAS for glu in GLU_DELTAS]


def delta_token(value: float) -> str:
    """Return a filename-safe signed delta token."""
    return f"{value:+g}".replace("+", "p").replace("-", "m").replace(".", "p")


def variant_id(cyp_delta: float, glu_delta: float) -> str:
    """Return stable coupled variant ID."""
    return f"cyp_glu_{delta_token(cyp_delta)}__glu_mep_{delta_token(glu_delta)}"


def output_path(out_dir: Path, variant_id_text: str) -> Path:
    """Return coupled output PDB path."""
    return out_dir / f"{variant_id_text}.pdb"


def rows_by_delta(rows: pd.DataFrame) -> dict[float, pd.Series]:
    """Map fixed torsion delta to row."""
    out = {}
    for _, row in rows.iterrows():
        out[float(row["fixed_torsion_delta_deg"])] = row
    return out


def manifest_row(
    variant_id_text: str,
    cyp_delta: float,
    glu_delta: float,
    source_pdb: Path,
    output_pdb: Path,
    cyp_found: int,
    cyp_applied: int,
    cyp_skipped: list[str],
    glu_found: int,
    glu_applied: int,
    glu_skipped: list[str],
    max_ca_shift: float,
) -> dict[str, object]:
    """Build coupled variant manifest row."""
    status = "ok" if not cyp_skipped and not glu_skipped else "partial"
    notes = []
    if cyp_skipped:
        notes.append("CYP->GLU skipped: " + "; ".join(cyp_skipped))
    if glu_skipped:
        notes.append("GLU->MEP skipped: " + "; ".join(glu_skipped))
    return {
        "variant_id": variant_id_text,
        "cyp_glu_delta_deg": cyp_delta,
        "glu_mep_delta_deg": glu_delta,
        "source_pdb": str(source_pdb),
        "output_pdb": str(output_pdb),
        "cyp_glu_windows_found": cyp_found,
        "cyp_glu_windows_attempted": cyp_found,
        "cyp_glu_windows_applied": cyp_applied,
        "cyp_glu_windows_skipped": len(cyp_skipped),
        "glu_mep_windows_found": glu_found,
        "glu_mep_windows_attempted": glu_found,
        "glu_mep_windows_applied": glu_applied,
        "glu_mep_windows_skipped": len(glu_skipped),
        "max_ca_anchor_shift_A": max_ca_shift,
        "cyp_glu_omega_mode": CYP_OMEGA_MODE,
        "glu_mep_omega_mode": GLU_OMEGA_MODE,
        "status": status,
        "notes": "; ".join(notes) if notes else "all_windows_applied; coupled_pilot_no_scoring",
    }


def generate_variants(
    source_pdb: Path,
    cyp_cd_scores: Path,
    cyp_geometry_audit: Path,
    glu_comparison: Path,
    out_dir: Path,
    manifest_path: Path,
    report_path: Path,
) -> pd.DataFrame:
    """Generate coupled variants."""
    source_lines = read_pdb_lines(source_pdb)
    residues_by_chain = parse_residues(source_pdb)
    cyp_windows = identify_cyp_glu_windows(residues_by_chain)
    glu_windows = identify_glu_mep_windows(residues_by_chain)
    cyp_rows = rows_by_delta(safe_local_cyp_glu_rows(pd.read_csv(cyp_cd_scores), pd.read_csv(cyp_geometry_audit)))
    glu_rows = rows_by_delta(select_representative_rows(safe_baseline_parent_glu_mep_rows(pd.read_csv(glu_comparison))))
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for cyp_delta, glu_delta in coupled_delta_grid():
        cyp_row = cyp_rows[cyp_delta]
        glu_row = glu_rows[glu_delta]
        cyp_candidate_lines = read_pdb_lines(ROOT / Path(str(cyp_row["coordinate_path"])))
        cyp_chain = str(cyp_row["source_chain"])
        cyp_start = int(pd.read_csv(cyp_geometry_audit).loc[pd.read_csv(cyp_geometry_audit)["candidate_id"] == cyp_row["candidate_id"], "repeat_start_index"].iloc[0])
        cyp_resseq_i = residues_by_chain[cyp_chain][cyp_start].resseq
        cyp_resseq_j = residues_by_chain[cyp_chain][cyp_start + 1].resseq
        cyp_disp = template_displacements(source_lines, cyp_candidate_lines, cyp_chain, cyp_resseq_i, cyp_resseq_j)

        glu_candidate_lines = local_candidate_lines(source_lines, residues_by_chain, glu_row)
        glu_chain = str(glu_row["chain_id"])
        glu_start = int(glu_row["repeat_start_index"])
        glu_resseq_i = residues_by_chain[glu_chain][glu_start].resseq
        glu_resseq_j = residues_by_chain[glu_chain][glu_start + 1].resseq
        glu_disp = template_displacements(source_lines, glu_candidate_lines, glu_chain, glu_resseq_i, glu_resseq_j)

        cyp_lines, cyp_applied, cyp_skipped = apply_displacements(source_lines, cyp_windows, cyp_disp)
        coupled_lines, glu_applied, glu_skipped = apply_displacements(cyp_lines, glu_windows, glu_disp)
        vid = variant_id(cyp_delta, glu_delta)
        pdb_path = output_path(out_dir, vid)
        xyz_path = pdb_path.with_suffix(".xyz")
        pdb_path.write_text("\n".join(coupled_lines) + "\n", encoding="ascii")
        write_xyz_from_pdb_lines(xyz_path, coupled_lines)
        rows.append(
            manifest_row(
                vid,
                cyp_delta,
                glu_delta,
                source_pdb,
                pdb_path,
                len(cyp_windows),
                cyp_applied,
                cyp_skipped,
                len(glu_windows),
                glu_applied,
                glu_skipped,
                max_ca_anchor_shift(source_lines, coupled_lines),
            )
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(manifest_path, index=False)
    write_report(manifest, report_path)
    return manifest


def write_report(manifest: pd.DataFrame, path: Path) -> None:
    """Write coupled generation report."""
    text = f"""# Coupled CYP->GLU + GLU->MEP Variant Generation

This is a small coupled pilot branch. It is generation plus geometry audit preparation only; no C/D diffraction scoring is run here.

- Coupled variants generated: {len(manifest)}
- CYP->GLU omega mode: `{CYP_OMEGA_MODE}`
- GLU->MEP omega mode: `{GLU_OMEGA_MODE}`
- CYP->GLU delta grid: {', '.join(f'{x:+g}' for x in CYP_DELTAS)}
- GLU->MEP delta grid: {', '.join(f'{x:+g}' for x in GLU_DELTAS)}
- Maximum C-alpha anchor shift: {manifest['max_ca_anchor_shift_A'].max():.6g} A

## Manifest

{markdown_table(manifest[['variant_id', 'cyp_glu_delta_deg', 'glu_mep_delta_deg', 'cyp_glu_windows_applied', 'glu_mep_windows_applied', 'max_ca_anchor_shift_A', 'status']])}
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
    parser.add_argument("--source-pdb", type=Path, default=DEFAULT_SOURCE_PDB)
    parser.add_argument("--cyp-cd-scores", type=Path, default=DEFAULT_CYP_CD_SCORES)
    parser.add_argument("--cyp-geometry-audit", type=Path, default=DEFAULT_CYP_GEOMETRY_AUDIT)
    parser.add_argument("--glu-comparison", type=Path, default=DEFAULT_GLU_COMPARISON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = generate_variants(
        args.source_pdb,
        args.cyp_cd_scores,
        args.cyp_geometry_audit,
        args.glu_comparison,
        args.out_dir,
        args.manifest,
        args.report,
    )
    print(f"Generated {len(manifest)} coupled variants")
    print(f"Manifest: {args.manifest}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
