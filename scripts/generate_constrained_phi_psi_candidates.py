"""Generate small coordinate candidates from accepted constrained phi/psi closures."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_backbone_torsion_repeat import parse_residues
from scripts.prototype_constrained_phi_psi_closure import (
    build_closure_window,
    find_parent_pdb,
    point_from_internal,
)


OMEGA_POLICY = "fixed_180"
DEFAULT_CLOSURE_CSV = Path("outputs/metrics/constrained_phi_psi_closure_prototype.csv")
DEFAULT_OUTDIR = Path("outputs/coordinates/constrained_phi_psi_candidates")
DEFAULT_MANIFEST = Path("outputs/metrics/constrained_phi_psi_candidate_manifest.csv")
DEFAULT_REPORT = Path("outputs/reports/constrained_phi_psi_candidate_generation.md")


def truthy(value: object) -> bool:
    """Parse pandas/string booleans."""
    return str(value).strip().lower() in {"true", "1", "yes"}


def accepted_closure_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return accepted closure rows only."""
    if "closure_success" not in df.columns:
        raise ValueError("Closure CSV is missing closure_success column.")
    return df[df["closure_success"].map(truthy)].copy()


def candidate_id(row: pd.Series, ordinal: int) -> str:
    """Build stable candidate ID from closure metadata."""
    repeat = re.sub(r"[^A-Za-z0-9]+", "_", str(row["residue_names"])).strip("_")
    mode = re.sub(r"[^A-Za-z0-9]+", "_", str(row["solve_mode"])).strip("_")
    delta = f"{float(row['fixed_torsion_delta_deg']):+g}".replace("+", "p").replace("-", "m").replace(".", "p")
    return f"cand_{ordinal:03d}_{row['chain_id']}_{repeat}_{mode}_{row['fixed_torsion_name']}_{delta}"


def select_candidate_rows(accepted: pd.DataFrame, max_candidates: int = 10) -> pd.DataFrame:
    """Select a small balanced candidate subset."""
    if accepted.empty or max_candidates <= 0:
        return accepted.head(0).copy()
    rows = []
    used = set()
    sorted_df = accepted.sort_values(["endpoint_error_A", "residue_names", "solve_mode", "fixed_torsion_delta_deg"])
    for _, group in sorted_df.groupby(["residue_names", "solve_mode"], sort=True):
        if len(rows) >= max_candidates:
            break
        idx = group.index[0]
        rows.append(idx)
        used.add(idx)
    for idx in sorted_df.index:
        if len(rows) >= max_candidates:
            break
        if idx not in used:
            rows.append(idx)
    return accepted.loc[rows].reset_index(drop=True)


def output_paths(out_dir: Path, candidate_id_text: str) -> tuple[Path, Path]:
    """Return PDB and XYZ output paths for a candidate."""
    return out_dir / f"{candidate_id_text}.pdb", out_dir / f"{candidate_id_text}.xyz"


def update_pdb_coordinate_line(line: str, coord: np.ndarray) -> str:
    """Return PDB line with updated XYZ coordinates."""
    return f"{line[:30]}{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}{line[54:]}"


def read_pdb_lines(path: Path) -> list[str]:
    """Read PDB lines."""
    return path.read_text(encoding="ascii").splitlines()


def atom_key_from_line(line: str) -> tuple[str, int, str] | None:
    """Return chain/residue/atom key for an ATOM/HETATM line."""
    if not line.startswith(("ATOM  ", "HETATM")):
        return None
    return line[21].strip(), int(line[22:26]), line[12:16].strip()


def solved_delta(row: pd.Series, torsion_name: str) -> float:
    """Return delta for a solved/fixed torsion in a closure row."""
    if str(row["fixed_torsion_name"]) == torsion_name:
        return float(row["fixed_torsion_delta_deg"])
    for idx in [1, 2]:
        name_col = f"solved_torsion_{idx}_name"
        delta_col = f"solved_torsion_{idx}_delta_deg"
        if name_col in row and str(row.get(name_col, "")) == torsion_name and pd.notna(row.get(delta_col)):
            return float(row[delta_col])
    return 0.0


def reconstruct_candidate_points(window, row: pd.Series) -> dict[tuple[str, int, str], np.ndarray]:
    """Reconstruct local backbone points while keeping CA anchors fixed."""
    omega0 = 180.0
    omega1 = 180.0
    phi0 = window.baseline_torsions["phi0_deg"] + solved_delta(row, "phi0_deg")
    psi0 = window.baseline_torsions["psi0_deg"] + solved_delta(row, "psi0_deg")
    phi1 = window.baseline_torsions.get("phi1_deg", 0.0) + solved_delta(row, "phi1_deg")
    psi1 = window.baseline_torsions.get("psi1_deg", 0.0) + solved_delta(row, "psi1_deg")

    p_c_prev = window.prev_residue.atoms["C"]
    p_n0 = window.first_residue.atoms["N"]
    p_ca0 = window.first_residue.atoms["CA"]
    p_c0 = point_from_internal(
        p_c_prev,
        p_n0,
        p_ca0,
        window.bond_lengths["CA0_C0"],
        window.bond_angles["N0_CA0_C0"],
        phi0,
    )
    p_n1 = point_from_internal(
        p_n0,
        p_ca0,
        p_c0,
        window.bond_lengths["C0_N1"],
        window.bond_angles["CA0_C0_N1"],
        psi0,
    )
    updates = {
        (window.first_residue.chain, window.first_residue.resseq, "C"): p_c0,
        (window.second_residue.chain, window.second_residue.resseq, "N"): p_n1,
    }
    if "O" in window.first_residue.atoms:
        updates[(window.first_residue.chain, window.first_residue.resseq, "O")] = (
            p_c0 + (window.first_residue.atoms["O"] - window.first_residue.atoms["C"])
        )
    if str(row.get("solve_mode", "")) == "two_torsion" and window.next_residue is not None:
        p_ca1 = window.second_residue.atoms["CA"]
        p_c1 = point_from_internal(
            p_c0,
            p_n1,
            p_ca1,
            window.bond_lengths["CA1_C1"],
            window.bond_angles["N1_CA1_C1"],
            phi1,
        )
        p_n2 = point_from_internal(
            p_n1,
            p_ca1,
            p_c1,
            window.bond_lengths["C1_N2"],
            window.bond_angles["CA1_C1_N2"],
            psi1,
        )
        updates[(window.second_residue.chain, window.second_residue.resseq, "C")] = p_c1
        updates[(window.next_residue.chain, window.next_residue.resseq, "N")] = p_n2
        if "O" in window.second_residue.atoms:
            updates[(window.second_residue.chain, window.second_residue.resseq, "O")] = (
                p_c1 + (window.second_residue.atoms["O"] - window.second_residue.atoms["C"])
            )
    return updates


def ca_anchor_shift(original_residues, candidate_lines: list[str], chain_id: str, start_index: int, solve_mode: str) -> float:
    """Return max CA anchor shift for selected window."""
    residues = original_residues[chain_id]
    anchor_indices = [start_index, start_index + 1]
    if solve_mode == "two_torsion" and start_index + 2 < len(residues):
        anchor_indices.append(start_index + 2)
    candidate_ca = {}
    for line in candidate_lines:
        key = atom_key_from_line(line)
        if key and key[2] == "CA":
            candidate_ca[(key[0], key[1])] = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    shifts = []
    for idx in anchor_indices:
        residue = residues[idx]
        original = residue.atoms["CA"]
        candidate = candidate_ca[(chain_id, residue.resseq)]
        shifts.append(float(np.linalg.norm(candidate - original)))
    return max(shifts) if shifts else float("nan")


def write_xyz_from_pdb_lines(path: Path, pdb_lines: list[str]) -> None:
    """Write XYZ file from PDB ATOM/HETATM lines."""
    atom_rows = []
    for line in pdb_lines:
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        atom_name = line[12:16].strip()
        element = (line[76:78].strip() or atom_name[:1]).upper()
        coord = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        atom_rows.append((element, coord))
    lines = [str(len(atom_rows)), f"generated from {path.with_suffix('.pdb').name}"]
    lines.extend(f"{element} {x:.6f} {y:.6f} {z:.6f}" for element, (x, y, z) in atom_rows)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def write_candidate_files(
    source_pdb: Path,
    row: pd.Series,
    candidate_id_text: str,
    out_dir: Path,
    residues_by_chain,
) -> dict[str, object]:
    """Write one candidate PDB/XYZ and return manifest data."""
    chain_id = str(row["chain_id"])
    start_index = int(row["repeat_start_index"])
    window = build_closure_window(residues_by_chain[chain_id], chain_id, start_index)
    updates = reconstruct_candidate_points(window, row)
    source_lines = read_pdb_lines(source_pdb)
    output_lines = []
    for line in source_lines:
        key = atom_key_from_line(line)
        if key in updates:
            output_lines.append(update_pdb_coordinate_line(line, updates[key]))
        else:
            output_lines.append(line)
    if not output_lines or output_lines[-1] != "END":
        output_lines.append("END")
    pdb_path, xyz_path = output_paths(out_dir, candidate_id_text)
    pdb_path.write_text("\n".join(output_lines) + "\n", encoding="ascii")
    write_xyz_from_pdb_lines(xyz_path, output_lines)
    anchor_shift = ca_anchor_shift(residues_by_chain, output_lines, chain_id, start_index, str(row["solve_mode"]))
    return {
        "coordinate_path": str(pdb_path),
        "xyz_path": str(xyz_path),
        "max_ca_anchor_shift_A": anchor_shift,
        "labels_preserved": True,
    }


def manifest_row(row: pd.Series, candidate_id_text: str, coordinate_path: str, notes: str, **extra) -> dict[str, object]:
    """Build candidate manifest row."""
    return {
        "candidate_id": candidate_id_text,
        "source_chain": row["chain_id"],
        "repeat_type": row["residue_names"],
        "repeat_start_index": int(row["repeat_start_index"]),
        "solve_mode": row["solve_mode"],
        "fixed_torsion_name": row["fixed_torsion_name"],
        "fixed_torsion_delta_deg": float(row["fixed_torsion_delta_deg"]),
        "solved_torsion_1_name": row.get("solved_torsion_1_name", ""),
        "solved_torsion_1_delta_deg": row.get("solved_torsion_1_delta_deg", np.nan),
        "solved_torsion_2_name": row.get("solved_torsion_2_name", ""),
        "solved_torsion_2_delta_deg": row.get("solved_torsion_2_delta_deg", np.nan),
        "omega_policy": OMEGA_POLICY,
        "endpoint_error_A": float(row["endpoint_error_A"]),
        "coordinate_path": coordinate_path,
        "notes": notes,
        **extra,
    }


def find_source_pdb() -> Path:
    """Find best-clean backbone-plus-carboxylate source PDB."""
    manifest = pd.read_csv(ROOT / "outputs/coordinates/ideal_hexaflex_variants/variant_manifest.csv")
    row = manifest[manifest["variant"] == "backbone_plus_carboxylate"]
    if row.empty:
        return find_parent_pdb()
    return ROOT / Path(str(row.iloc[0]["pdb_path"]))


def generate_candidates(
    source_pdb: Path,
    closure_csv: Path,
    out_dir: Path,
    manifest_path: Path,
    report_path: Path,
    max_candidates: int = 10,
) -> pd.DataFrame:
    """Generate candidate coordinate files from accepted closure rows."""
    closure_df = pd.read_csv(closure_csv)
    accepted = accepted_closure_rows(closure_df)
    selected = select_candidate_rows(accepted, max_candidates=max_candidates)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    residues_by_chain = parse_residues(source_pdb)
    rows = []
    for ordinal, (_, row) in enumerate(selected.iterrows(), start=1):
        cid = candidate_id(row, ordinal)
        written = write_candidate_files(source_pdb, row, cid, out_dir, residues_by_chain)
        rows.append(
            manifest_row(
                row,
                cid,
                written["coordinate_path"],
                f"generated_from_{source_pdb.name}; omega_sensitivity_deferred",
                xyz_path=written["xyz_path"],
                max_ca_anchor_shift_A=written["max_ca_anchor_shift_A"],
                labels_preserved=written["labels_preserved"],
            )
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(manifest_path, index=False)
    write_report(accepted, manifest, report_path)
    return manifest


def write_report(accepted: pd.DataFrame, manifest: pd.DataFrame, path: Path) -> None:
    """Write candidate generation report."""
    represented = (
        manifest.groupby(["repeat_type", "solve_mode"]).size().reset_index(name="candidate_count")
        if not manifest.empty
        else pd.DataFrame(columns=["repeat_type", "solve_mode", "candidate_count"])
    )
    max_anchor_shift = manifest["max_ca_anchor_shift_A"].max() if "max_ca_anchor_shift_A" in manifest else np.nan
    text = f"""# Constrained Phi/Psi Candidate Generation

This is a small coordinate-generation step for inspection and later diffraction scoring. It does not run the full diffraction scan.

- Accepted closure rows available: {len(accepted)}
- Coordinate candidates generated: {len(manifest)}
- Omega policy: `{OMEGA_POLICY}`
- Maximum C-alpha anchor shift among generated candidates: {max_anchor_shift:.6g} A
- Chain/residue/atom labels preserved: {bool(manifest['labels_preserved'].all()) if not manifest.empty else True}

## Representation

{markdown_table(represented)}

## Interpretation

- C-alpha anchors are preserved by leaving anchor `CA` atom coordinates fixed and recording the max anchor shift.
- Candidate PDB and XYZ files are suitable inputs for later Debye/pair-family diffraction scoring, subject to the prototype's local-geometry limitations.
- Omega is fixed at 180 degrees for this phase. Omega sensitivity is explicitly deferred.
"""
    path.write_text(text, encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> str:
    """Render dataframe as markdown table."""
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in df.itertuples(index=False):
        vals = [f"{v:.6g}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pdb", type=Path, default=None)
    parser.add_argument("--closure-csv", type=Path, default=DEFAULT_CLOSURE_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-candidates", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_pdb = args.source_pdb or find_source_pdb()
    manifest = generate_candidates(
        source_pdb,
        args.closure_csv,
        args.out_dir,
        args.manifest,
        args.report,
        max_candidates=args.max_candidates,
    )
    print(f"Generated {len(manifest)} candidates")
    print(f"Manifest: {args.manifest}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
