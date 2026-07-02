"""Generate coherent repeated constrained phi/psi variants from safe local templates."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_backbone_torsion_repeat import parse_residues
from scripts.generate_constrained_phi_psi_candidates import (
    atom_key_from_line,
    read_pdb_lines,
    update_pdb_coordinate_line,
    write_xyz_from_pdb_lines,
)
from scripts.score_constrained_phi_psi_candidates_cd import parse_bool


OMEGA_POLICY = "fixed_180"
DEFAULT_CD_SCORES = Path("outputs/metrics/constrained_phi_psi_candidate_cd_scores.csv")
DEFAULT_GEOMETRY_AUDIT = Path("outputs/metrics/constrained_phi_psi_candidate_geometry_audit.csv")
DEFAULT_MANIFEST = Path("outputs/metrics/constrained_phi_psi_candidate_manifest.csv")
DEFAULT_SOURCE_PDB = Path("outputs/coordinates/ideal_hexaflex_variants/ideal_hexaflex_backbone_plus_carboxylate.pdb")
DEFAULT_OUTDIR = Path("outputs/coordinates/repeated_constrained_phi_psi_variants")
DEFAULT_OUT_MANIFEST = Path("outputs/metrics/repeated_constrained_phi_psi_variant_manifest.csv")
DEFAULT_REPORT = Path("outputs/reports/repeated_constrained_phi_psi_variant_generation.md")
ALLOWED_DELTAS = {-2.0, -1.0, 0.0, 1.0, 2.0, 3.0}
UPDATED_ATOMS = ("C", "O")
NEXT_UPDATED_ATOMS = ("N",)


@dataclass(frozen=True)
class RepeatWindow:
    """A two-residue repeat window identified by coordinate order."""

    chain: str
    start_index: int
    resseq_i: int
    resseq_j: int
    resname_i: str
    resname_j: str

    @property
    def repeat_type(self) -> str:
        return f"{self.resname_i}->{self.resname_j}"


def safe_local_cyp_glu_rows(cd_scores: pd.DataFrame, geometry_audit: pd.DataFrame) -> pd.DataFrame:
    """Return safe local CYP->GLU one-torsion rows for the repeated pilot."""
    audit_safe = geometry_audit[geometry_audit["safe_for_diffraction_scoring"].map(parse_bool)].copy()
    safe_ids = set(audit_safe["candidate_id"].astype(str))
    rows = cd_scores[
        cd_scores["candidate_id"].astype(str).isin(safe_ids)
        & (cd_scores["repeat_type"] == "CYP->GLU")
        & (cd_scores["solve_mode"] == "one_torsion")
        & (cd_scores["fixed_torsion_name"] == "phi0_deg")
    ].copy()
    rows["fixed_torsion_delta_deg"] = pd.to_numeric(rows["fixed_torsion_delta_deg"], errors="coerce")
    rows = rows[rows["fixed_torsion_delta_deg"].isin(ALLOWED_DELTAS)]
    return rows.sort_values("fixed_torsion_delta_deg").reset_index(drop=True)


def identify_cyp_glu_windows(residues_by_chain: dict[str, list]) -> list[RepeatWindow]:
    """Identify CYP->GLU windows using per-chain coordinate order."""
    windows: list[RepeatWindow] = []
    for chain, residues in residues_by_chain.items():
        for index, (first, second) in enumerate(zip(residues, residues[1:])):
            if first.resname == "CYP" and second.resname == "GLU":
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


def variant_id(delta: float) -> str:
    """Build stable repeated variant ID from the fixed torsion delta."""
    token = f"{delta:+g}".replace("+", "p").replace("-", "m").replace(".", "p")
    return f"repeated_CYP_GLU_one_torsion_phi0_deg_{token}"


def pdb_atom_lookup(lines: list[str]) -> dict[tuple[str, int, str], np.ndarray]:
    """Map chain/residue/atom to coordinates from PDB lines."""
    lookup = {}
    for line in lines:
        key = atom_key_from_line(line)
        if key is not None:
            lookup[key] = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=float)
    return lookup


def template_displacements(parent_lines: list[str], candidate_lines: list[str], template_chain: str, resseq_i: int, resseq_j: int) -> dict[tuple[int, str], np.ndarray]:
    """Return local atom displacement templates keyed by residue offset and atom name."""
    parent = pdb_atom_lookup(parent_lines)
    candidate = pdb_atom_lookup(candidate_lines)
    displacements: dict[tuple[int, str], np.ndarray] = {}
    for residue_offset, resseq, atom_names in [
        (0, resseq_i, UPDATED_ATOMS),
        (1, resseq_j, NEXT_UPDATED_ATOMS),
    ]:
        for atom_name in atom_names:
            key = (template_chain, resseq, atom_name)
            if key in parent and key in candidate:
                displacements[(residue_offset, atom_name)] = candidate[key] - parent[key]
    return displacements


def apply_displacements(
    source_lines: list[str],
    windows: list[RepeatWindow],
    displacements: dict[tuple[int, str], np.ndarray],
) -> tuple[list[str], int, list[str]]:
    """Apply local displacement template to all compatible windows."""
    output_lines = []
    applied_windows = 0
    skipped: list[str] = []
    remaining = {(window.chain, window.resseq_i, window.resseq_j) for window in windows}

    updates: dict[tuple[str, int, str], np.ndarray] = {}
    source_coords = pdb_atom_lookup(source_lines)
    for window in windows:
        required_keys = []
        for residue_offset, resseq in [(0, window.resseq_i), (1, window.resseq_j)]:
            for atom_name in [atom for offset, atom in displacements if offset == residue_offset]:
                required_keys.append((window.chain, resseq, atom_name, residue_offset))
        missing = [f"{chain}:{resseq}:{atom}" for chain, resseq, atom, _ in required_keys if (chain, resseq, atom) not in source_coords]
        if missing:
            skipped.append(f"{window.chain}{window.resseq_i}-{window.resseq_j}: missing {','.join(missing)}")
            continue
        for chain, resseq, atom_name, residue_offset in required_keys:
            updates[(chain, resseq, atom_name)] = source_coords[(chain, resseq, atom_name)] + displacements[(residue_offset, atom_name)]
        applied_windows += 1
        remaining.discard((window.chain, window.resseq_i, window.resseq_j))

    for line in source_lines:
        key = atom_key_from_line(line)
        if key in updates:
            output_lines.append(update_pdb_coordinate_line(line, updates[key]))
        else:
            output_lines.append(line)
    if not output_lines or output_lines[-1] != "END":
        output_lines.append("END")
    return output_lines, applied_windows, skipped


def max_ca_anchor_shift(source_lines: list[str], output_lines: list[str]) -> float:
    """Return maximum C-alpha coordinate change."""
    source = pdb_atom_lookup(source_lines)
    output = pdb_atom_lookup(output_lines)
    shifts = []
    for key, coord in source.items():
        if key[2] == "CA" and key in output:
            shifts.append(float(np.linalg.norm(output[key] - coord)))
    return max(shifts) if shifts else float("nan")


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
    """Build repeated-variant manifest row."""
    return {
        "variant_id": variant_id_text,
        "source_model": str(source_model),
        "repeat_type": "CYP->GLU",
        "fixed_torsion_name": row["fixed_torsion_name"],
        "fixed_torsion_delta_deg": float(row["fixed_torsion_delta_deg"]),
        "omega_policy": OMEGA_POLICY,
        "attempted_window_count": attempted,
        "applied_window_count": applied,
        "skipped_window_count": len(skipped),
        "max_endpoint_error_A": row.get("endpoint_error_A", ""),
        "max_ca_anchor_shift_A": max_ca_shift,
        "coordinate_path": str(coordinate_path),
        "notes": "; ".join(skipped) if skipped else "all_equivalent_windows_applied; coherent_fixed_omega_pilot",
    }


def write_report(manifest: pd.DataFrame, window_count: int, path: Path) -> None:
    """Write repeated-variant generation report."""
    deltas = ", ".join(f"{float(v):+g}" for v in manifest["fixed_torsion_delta_deg"]) if not manifest.empty else "none"
    max_ca = manifest["max_ca_anchor_shift_A"].max() if not manifest.empty else float("nan")
    table = markdown_table(
        manifest[
            [
                "variant_id",
                "fixed_torsion_delta_deg",
                "attempted_window_count",
                "applied_window_count",
                "skipped_window_count",
                "max_ca_anchor_shift_A",
            ]
        ]
        if not manifest.empty
        else manifest
    )
    text = f"""# Repeated Constrained Phi/Psi Variant Generation

This is a coherent fixed-omega pilot. It applies safe local CYP->GLU one-torsion displacement templates across equivalent CYP->GLU repeat windows in the backbone-plus-carboxylate source model. It is not the full systematic torsion scan.

- Equivalent CYP->GLU windows found: {window_count}
- Repeated variants generated: {len(manifest)}
- Generated deltas: {deltas}
- Omega policy: `{OMEGA_POLICY}` under Nick's current fixed/trans policy.
- Maximum C-alpha anchor shift: {max_ca:.6g} A

## Variant Bookkeeping

{table}

## Interpretation

- C-alpha anchors are preserved by never updating `CA` atom coordinates.
- Chain IDs, residue IDs, residue names, and atom names are preserved from the source PDB.
- These variants should be geometry-audited before diffraction scoring.
- Omega sensitivity remains deferred until after this fixed-omega coherent/repeated perturbation pilot.
"""
    path.write_text(text, encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> str:
    """Render dataframe as markdown."""
    if df.empty:
        return "_No rows._"
    columns = list(df.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for record in df.itertuples(index=False):
        values = [f"{v:.6g}" if isinstance(v, float) else str(v) for v in record]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def generate_variants(
    source_pdb: Path,
    cd_scores_csv: Path,
    geometry_audit_csv: Path,
    out_dir: Path,
    out_manifest: Path,
    report_path: Path,
) -> pd.DataFrame:
    """Generate repeated variants from safe local candidate templates."""
    cd_scores = pd.read_csv(cd_scores_csv)
    geometry_audit = pd.read_csv(geometry_audit_csv)
    rows = safe_local_cyp_glu_rows(cd_scores, geometry_audit)
    residues_by_chain = parse_residues(source_pdb)
    windows = identify_cyp_glu_windows(residues_by_chain)
    source_lines = read_pdb_lines(source_pdb)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for _, row in rows.iterrows():
        candidate_path = ROOT / Path(str(row["coordinate_path"]))
        candidate_lines = read_pdb_lines(candidate_path)
        template_chain = str(row["source_chain"])
        template_start = int(geometry_audit.loc[geometry_audit["candidate_id"] == row["candidate_id"], "repeat_start_index"].iloc[0])
        template_residues = residues_by_chain[template_chain]
        template_i = template_residues[template_start].resseq
        template_j = template_residues[template_start + 1].resseq
        displacements = template_displacements(source_lines, candidate_lines, template_chain, template_i, template_j)
        vid = variant_id(float(row["fixed_torsion_delta_deg"]))
        output_lines, applied, skipped = apply_displacements(source_lines, windows, displacements)
        pdb_path = out_dir / f"{vid}.pdb"
        xyz_path = out_dir / f"{vid}.xyz"
        pdb_path.write_text("\n".join(output_lines) + "\n", encoding="ascii")
        write_xyz_from_pdb_lines(xyz_path, output_lines)
        manifest_rows.append(
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
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(out_manifest, index=False)
    write_report(manifest, len(windows), report_path)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cd-scores", type=Path, default=DEFAULT_CD_SCORES)
    parser.add_argument("--geometry-audit", type=Path, default=DEFAULT_GEOMETRY_AUDIT)
    parser.add_argument("--local-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-pdb", type=Path, default=DEFAULT_SOURCE_PDB)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_OUT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = generate_variants(
        args.source_pdb,
        args.cd_scores,
        args.geometry_audit,
        args.out_dir,
        args.manifest,
        args.report,
    )
    print(f"Generated {len(manifest)} repeated variants")
    print(f"Manifest: {args.manifest}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
