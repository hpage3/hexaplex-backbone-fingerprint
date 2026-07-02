"""Generate repeated GLU->MEP variants from safe baseline-parent omega attempts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_backbone_torsion_repeat import parse_residues
from scripts.compare_glu_mep_omega_modes_closure import attempt_series
from scripts.generate_constrained_phi_psi_candidates import (
    read_pdb_lines,
    reconstruct_candidate_points,
    write_xyz_from_pdb_lines,
)
from scripts.generate_repeated_constrained_phi_psi_variants import (
    RepeatWindow,
    apply_displacements,
    max_ca_anchor_shift,
    template_displacements,
)
from scripts.prototype_constrained_phi_psi_closure import build_closure_window
from scripts.score_constrained_phi_psi_candidates_cd import parse_bool


DEFAULT_SOURCE_PDB = Path("outputs/coordinates/ideal_hexaflex_variants/ideal_hexaflex_backbone_plus_carboxylate.pdb")
DEFAULT_COMPARISON_CSV = Path("outputs/metrics/glu_mep_omega_mode_closure_comparison.csv")
DEFAULT_OUTDIR = Path("outputs/coordinates/repeated_glu_mep_baseline_omega_variants")
DEFAULT_MANIFEST = Path("outputs/metrics/repeated_glu_mep_baseline_omega_variant_manifest.csv")
DEFAULT_REPORT = Path("outputs/reports/repeated_glu_mep_baseline_omega_variant_generation.md")
OMEGA_MODE = "baseline_parent"
PREFERRED_DELTAS = [-2.0, -1.0, 0.0, 1.0, 2.0]
SOLVE_MODE_PRIORITY = {"one_torsion": 0, "two_torsion": 1, "local_refine": 2}


def identify_glu_mep_windows(residues_by_chain: dict[str, list]) -> list[RepeatWindow]:
    """Identify GLU->MEP windows using per-chain coordinate order."""
    windows: list[RepeatWindow] = []
    for chain, residues in residues_by_chain.items():
        for index, (first, second) in enumerate(zip(residues, residues[1:])):
            if first.resname == "GLU" and second.resname == "MEP":
                windows.append(
                    RepeatWindow(
                        chain=chain,
                        start_index=index,
                        resseq_i=first.resseq,
                        resseq_j=second.resseq,
                        resname_i=first.resname,
                        resname_j=second.resname,
                    )
                )
    return windows


def safe_baseline_parent_glu_mep_rows(comparison: pd.DataFrame) -> pd.DataFrame:
    """Filter safe baseline-parent GLU->MEP closure attempts."""
    rows = comparison[
        (comparison["omega_mode"] == OMEGA_MODE)
        & (comparison["repeat_type"] == "GLU->MEP")
        & comparison["geometry_safe"].map(parse_bool)
    ].copy()
    rows["fixed_torsion_delta_deg"] = pd.to_numeric(rows["fixed_torsion_delta_deg"], errors="coerce")
    return rows.sort_values(["fixed_torsion_delta_deg", "solve_mode"]).reset_index(drop=True)


def select_representative_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Select one small-pilot representative row per preferred delta."""
    selected = []
    for delta in PREFERRED_DELTAS:
        group = rows[rows["fixed_torsion_delta_deg"] == delta].copy()
        if group.empty:
            continue
        group["_priority"] = group["solve_mode"].map(SOLVE_MODE_PRIORITY).fillna(99)
        group["endpoint_error_A"] = pd.to_numeric(group["endpoint_error_A"], errors="coerce")
        selected.append(group.sort_values(["_priority", "endpoint_error_A"]).iloc[0].drop(labels=["_priority"]))
    return pd.DataFrame(selected).reset_index(drop=True) if selected else rows.head(0).copy()


def variant_id(row: pd.Series) -> str:
    """Build a stable variant ID."""
    delta = float(row["fixed_torsion_delta_deg"])
    token = f"{delta:+g}".replace("+", "p").replace("-", "m").replace(".", "p")
    mode = str(row["solve_mode"])
    return f"repeated_GLU_MEP_{OMEGA_MODE}_{mode}_phi0_deg_{token}"


def local_candidate_lines(source_lines: list[str], residues_by_chain: dict[str, list], row: pd.Series) -> list[str]:
    """Build local candidate PDB lines for one selected attempt."""
    chain = str(row["chain_id"])
    start = int(row["repeat_start_index"])
    window = build_closure_window(residues_by_chain[chain], chain, start)
    series = attempt_series(
        window,
        float(row["fixed_torsion_delta_deg"]),
        str(row["solve_mode"]),
        OMEGA_MODE,
        str(row["solved_torsion_1_name"]),
        float(row["solved_torsion_1_delta_deg"]),
        str(row.get("solved_torsion_2_name", "")),
        float(row["solved_torsion_2_delta_deg"]) if pd.notna(row.get("solved_torsion_2_delta_deg", np.nan)) and str(row.get("solved_torsion_2_delta_deg", "")) != "" else np.nan,
    )
    updates = reconstruct_candidate_points(window, series)
    output = []
    from scripts.generate_constrained_phi_psi_candidates import atom_key_from_line, update_pdb_coordinate_line

    for line in source_lines:
        key = atom_key_from_line(line)
        output.append(update_pdb_coordinate_line(line, updates[key]) if key in updates else line)
    if not output or output[-1] != "END":
        output.append("END")
    return output


def manifest_row(
    row: pd.Series,
    variant_id_text: str,
    source_model: Path,
    attempted: int,
    applied: int,
    skipped: list[str],
    max_ca_shift: float,
    coordinate_path: Path,
) -> dict[str, object]:
    """Build manifest row."""
    return {
        "variant_id": variant_id_text,
        "source_model": str(source_model),
        "repeat_type": "GLU->MEP",
        "fixed_torsion_name": row["fixed_torsion_name"],
        "fixed_torsion_delta_deg": float(row["fixed_torsion_delta_deg"]),
        "solve_mode": row["solve_mode"],
        "omega_mode": OMEGA_MODE,
        "attempted_window_count": attempted,
        "applied_window_count": applied,
        "skipped_window_count": len(skipped),
        "max_endpoint_error_A": row.get("endpoint_error_A", ""),
        "max_ca_anchor_shift_A": max_ca_shift,
        "coordinate_path": str(coordinate_path),
        "notes": "; ".join(skipped) if skipped else "all_equivalent_windows_applied; baseline_parent_omega_pilot",
    }


def generate_variants(
    source_pdb: Path,
    comparison_csv: Path,
    out_dir: Path,
    manifest_path: Path,
    report_path: Path,
) -> pd.DataFrame:
    """Generate repeated GLU->MEP baseline-parent omega variants."""
    comparison = pd.read_csv(comparison_csv)
    selected = select_representative_rows(safe_baseline_parent_glu_mep_rows(comparison))
    residues_by_chain = parse_residues(source_pdb)
    windows = identify_glu_mep_windows(residues_by_chain)
    source_lines = read_pdb_lines(source_pdb)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, row in selected.iterrows():
        candidate_lines = local_candidate_lines(source_lines, residues_by_chain, row)
        chain = str(row["chain_id"])
        start = int(row["repeat_start_index"])
        template_residues = residues_by_chain[chain]
        resseq_i = template_residues[start].resseq
        resseq_j = template_residues[start + 1].resseq
        displacements = template_displacements(source_lines, candidate_lines, chain, resseq_i, resseq_j)
        output_lines, applied, skipped = apply_displacements(source_lines, windows, displacements)
        vid = variant_id(row)
        pdb_path = out_dir / f"{vid}.pdb"
        xyz_path = out_dir / f"{vid}.xyz"
        pdb_path.write_text("\n".join(output_lines) + "\n", encoding="ascii")
        write_xyz_from_pdb_lines(xyz_path, output_lines)
        rows.append(
            manifest_row(
                row,
                vid,
                source_pdb,
                attempted=len(windows),
                applied=applied,
                skipped=skipped,
                max_ca_shift=max_ca_anchor_shift(source_lines, output_lines),
                coordinate_path=pdb_path,
            )
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(manifest_path, index=False)
    write_report(manifest, windows, report_path)
    return manifest


def write_report(manifest: pd.DataFrame, windows: list[RepeatWindow], path: Path) -> None:
    """Write variant generation report."""
    chains = ", ".join(sorted({window.chain for window in windows}))
    selected = (
        ", ".join(f"{row.fixed_torsion_delta_deg:+g}/{row.solve_mode}" for row in manifest.itertuples())
        if not manifest.empty
        else "none"
    )
    max_ca = manifest["max_ca_anchor_shift_A"].max() if not manifest.empty else float("nan")
    table = markdown_table(
        manifest[
            [
                "variant_id",
                "fixed_torsion_delta_deg",
                "solve_mode",
                "omega_mode",
                "attempted_window_count",
                "applied_window_count",
                "skipped_window_count",
                "max_ca_anchor_shift_A",
            ]
        ]
        if not manifest.empty
        else manifest
    )
    text = f"""# Repeated GLU->MEP Baseline-Omega Variant Generation

This is a baseline-parent-omega pilot, not an unconstrained omega scan. It uses only geometry-safe `baseline_parent` GLU->MEP closure attempts and applies selected perturbations coherently across equivalent GLU->MEP windows.

- Equivalent GLU->MEP windows found: {len(windows)}
- Chains containing GLU->MEP windows: {chains}
- Selected delta/solve-mode attempts: {selected}
- Repeated variants generated: {len(manifest)}
- Omega mode recorded as: `{OMEGA_MODE}`
- Maximum C-alpha anchor shift: {max_ca:.6g} A

## Variant Bookkeeping

{table}

## Interpretation

- C-alpha anchors are preserved by never updating `CA` atom coordinates.
- Chain IDs, residue IDs, residue names, and atom names are preserved from the source PDB.
- These variants are suitable for geometry audit before any diffraction scoring.
- This remains a small pilot; it is not a full omega or torsion scan.
"""
    path.write_text(text, encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> str:
    """Render dataframe as markdown."""
    if df.empty:
        return "_No rows._"
    columns = list(df.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for record in df.itertuples(index=False):
        values = [f"{value:.6g}" if isinstance(value, float) else str(value) for value in record]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pdb", type=Path, default=DEFAULT_SOURCE_PDB)
    parser.add_argument("--comparison-csv", type=Path, default=DEFAULT_COMPARISON_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = generate_variants(args.source_pdb, args.comparison_csv, args.out_dir, args.manifest, args.report)
    print(f"Generated {len(manifest)} repeated GLU->MEP baseline-omega variants")
    print(f"Manifest: {args.manifest}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
