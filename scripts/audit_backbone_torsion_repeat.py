"""Audit whether ideal Hexaflex backbone torsion repeats are extractable."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hexaplex_backbone_fingerprint.geometry import dihedral_degrees, distance


PARENT_LABEL = "full_hexaplex_anti_parallel_30deg_ideal_deduped_6chain"
BACKBONE_ATOMS = {"N", "CA", "C", "O"}


@dataclass(frozen=True)
class Residue:
    """One parsed residue with atom coordinates."""

    chain: str
    resseq: int
    resname: str
    atoms: dict[str, np.ndarray]
    atom_names_in_order: tuple[str, ...]


def parse_residues(path: Path) -> dict[str, list[Residue]]:
    """Parse PDB residues in coordinate order."""
    order: list[tuple[str, int]] = []
    residues: dict[tuple[str, int], dict[str, object]] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        atom_name = line[12:16].strip()
        resname = line[17:20].strip()
        chain = line[21].strip()
        resseq = int(line[22:26])
        key = (chain, resseq)
        if key not in residues:
            residues[key] = {"resname": resname, "atoms": {}, "atom_names": []}
            order.append(key)
        residues[key]["atoms"][atom_name] = np.array(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=float
        )
        residues[key]["atom_names"].append(atom_name)
    if not residues:
        raise ValueError(f"No ATOM/HETATM residues found in {path}.")
    by_chain: dict[str, list[Residue]] = {}
    for chain, resseq in order:
        info = residues[(chain, resseq)]
        by_chain.setdefault(chain, []).append(
            Residue(
                chain=chain,
                resseq=resseq,
                resname=str(info["resname"]),
                atoms=dict(info["atoms"]),
                atom_names_in_order=tuple(info["atom_names"]),
            )
        )
    return by_chain


def missing_backbone_atoms(residue: Residue) -> list[str]:
    """Return missing standard peptide backbone atoms."""
    return sorted(BACKBONE_ATOMS - set(residue.atoms))


def residue_torsions(prev_res: Residue | None, res: Residue, next_res: Residue | None) -> dict[str, float]:
    """Extract phi/psi/omega when definable."""
    torsions = {"phi_deg": np.nan, "psi_deg": np.nan, "omega_deg": np.nan}
    if prev_res is not None and {"C"}.issubset(prev_res.atoms) and {"N", "CA", "C"}.issubset(res.atoms):
        torsions["phi_deg"] = dihedral_degrees(prev_res.atoms["C"], res.atoms["N"], res.atoms["CA"], res.atoms["C"])
    if next_res is not None and {"N", "CA", "C"}.issubset(res.atoms) and "N" in next_res.atoms:
        torsions["psi_deg"] = dihedral_degrees(res.atoms["N"], res.atoms["CA"], res.atoms["C"], next_res.atoms["N"])
    if next_res is not None and {"CA", "C"}.issubset(res.atoms) and {"N", "CA"}.issubset(next_res.atoms):
        torsions["omega_deg"] = dihedral_degrees(res.atoms["CA"], res.atoms["C"], next_res.atoms["N"], next_res.atoms["CA"])
    return torsions


def omega_near_trans(omega_deg: float, tolerance_deg: float = 20.0) -> bool:
    """Return whether omega is near trans peptide geometry."""
    if pd.isna(omega_deg):
        return False
    return abs(abs(float(omega_deg)) - 180.0) <= tolerance_deg


def identify_repeat_windows(residues: list[Residue], window_size: int = 2) -> list[dict[str, object]]:
    """Identify consecutive residue windows spanning base CA to base CA."""
    rows = []
    for i in range(0, max(0, len(residues) - window_size + 1)):
        window = residues[i : i + window_size]
        first = window[0]
        last = window[-1]
        has_ca_anchors = "CA" in first.atoms and "CA" in last.atoms
        rows.append(
            {
                "chain": first.chain,
                "repeat_start_order_index": i,
                "repeat_end_order_index": i + window_size - 1,
                "repeat_start_resseq": first.resseq,
                "repeat_end_resseq": last.resseq,
                "repeat_residue_names": "->".join(res.resname for res in window),
                "has_ca_anchors": has_ca_anchors,
                "ca_to_ca_distance_A": distance(first.atoms["CA"], last.atoms["CA"]) if has_ca_anchors else np.nan,
            }
        )
    return rows


def chain_order_summary(residues: list[Residue]) -> str:
    """Return residue ID order direction."""
    ids = [res.resseq for res in residues]
    diffs = np.diff(ids)
    if len(diffs) == 0:
        return "single"
    if np.all(diffs > 0):
        return "increasing"
    if np.all(diffs < 0):
        return "decreasing"
    return "mixed"


def audit_pdb(path: Path, source_label: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Audit residues, torsions, and repeat windows for one PDB."""
    by_chain = parse_residues(path)
    residue_rows = []
    repeat_rows = []
    chain_rows = []
    residue_sets = {chain: {res.resseq for res in residues} for chain, residues in by_chain.items()}
    first_set = next(iter(residue_sets.values()))
    for chain, residues in sorted(by_chain.items()):
        names = sorted({res.resname for res in residues})
        atom_name_sets = [" ".join(res.atom_names_in_order) for res in residues[:6]]
        chain_rows.append(
            {
                "source_label": source_label,
                "source_path": str(path),
                "chain": chain,
                "residue_count": len(residues),
                "residue_names": ",".join(names),
                "residue_id_min": min(res.resseq for res in residues),
                "residue_id_max": max(res.resseq for res in residues),
                "residue_id_order": chain_order_summary(residues),
                "equivalent_residue_set_to_first_chain": residue_sets[chain] == first_set,
                "example_atom_names_per_residue": " | ".join(atom_name_sets),
            }
        )
        repeat_rows.extend({"source_label": source_label, **row} for row in identify_repeat_windows(residues))
        for i, res in enumerate(residues):
            prev_res = residues[i - 1] if i > 0 else None
            next_res = residues[i + 1] if i + 1 < len(residues) else None
            torsions = residue_torsions(prev_res, res, next_res)
            residue_rows.append(
                {
                    "source_label": source_label,
                    "chain": chain,
                    "coordinate_order_index": i,
                    "residue_id": res.resseq,
                    "resname": res.resname,
                    "atom_names": " ".join(res.atom_names_in_order),
                    "missing_backbone_atoms": " ".join(missing_backbone_atoms(res)),
                    **torsions,
                    "omega_near_180": omega_near_trans(torsions["omega_deg"]),
                }
            )
    return pd.DataFrame(chain_rows), pd.DataFrame(residue_rows), pd.DataFrame(repeat_rows)


def summarize_range(values: pd.Series) -> str:
    """Format min/median/max for a numeric series."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return "NA"
    return f"{numeric.min():.2f} / {numeric.median():.2f} / {numeric.max():.2f}"


def markdown_table(df: pd.DataFrame, columns: list[str], limit: int = 12) -> str:
    """Render a compact markdown table."""
    if df.empty:
        return "_No rows._"
    columns = [col for col in columns if col in df.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in df.head(limit)[columns].itertuples(index=False):
        vals = [f"{value:.4g}" if isinstance(value, float) else str(value) for value in row]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(chain_df: pd.DataFrame, residue_df: pd.DataFrame, repeat_df: pd.DataFrame, path: Path) -> None:
    """Write feasibility markdown report."""
    parent_res = residue_df[residue_df["source_label"] == PARENT_LABEL]
    missing_count = int((parent_res["missing_backbone_atoms"].astype(str).str.len() > 0).sum())
    omega_values = parent_res["omega_deg"].dropna()
    omega_near = int(parent_res["omega_near_180"].sum())
    omega_total = int(parent_res["omega_deg"].notna().sum())
    repeat_summary = repeat_df.groupby(["source_label", "repeat_residue_names"], as_index=False).agg(
        count=("repeat_residue_names", "size"),
        median_ca_to_ca_distance_A=("ca_to_ca_distance_A", "median"),
    )
    text = f"""# Backbone Torsion Repeat Feasibility Audit

This audit checks whether the ideal anti-parallel Hexaflex PDB exposes enough standard backbone information for a future constrained phi/psi search. No candidate backbones were generated.

## Chain Summary

{markdown_table(chain_df, ['source_label', 'chain', 'residue_count', 'residue_names', 'residue_id_min', 'residue_id_max', 'residue_id_order', 'equivalent_residue_set_to_first_chain'], limit=20)}

## Torsion Ranges

For the parent PDB:

- Phi min/median/max: {summarize_range(parent_res['phi_deg'])}
- Psi min/median/max: {summarize_range(parent_res['psi_deg'])}
- Omega min/median/max: {summarize_range(parent_res['omega_deg'])}
- Omega near 180 deg: {omega_near}/{omega_total}
- Residues missing any N/CA/C/O atom: {missing_count}

## Candidate Two-Unit Repeat Windows

{markdown_table(repeat_summary, ['source_label', 'repeat_residue_names', 'count', 'median_ca_to_ca_distance_A'], limit=20)}

## Feasibility Assessment

- Can phi/psi/omega be defined directly from atom names? {'Yes' if missing_count == 0 and omega_total > 0 else 'Partially; inspect missing atoms.'}
- Is the structural repeat chemically regular enough to parameterize? The chain sequence alternates primarily CYP/GLU, so a two-residue CYP->GLU or GLU->CYP window is a plausible repeat unit.
- Are C-alpha anchors available and consistent across chains? Repeat-window rows report CA-to-CA spans for every consecutive two-residue window.
- What atoms would need to move during a constrained torsion search? At minimum the downstream backbone atoms affected by phi/psi rotations plus attached side-chain/base/carboxylate atoms, while selected CA anchors remain fixed and omega remains constrained near trans.
- Does anti-parallel chain direction affect residue order/repeat indexing? Raw residue IDs increase in every chain, but prior register audit showed raw IDs are offset by chain. A future torsion search should use per-chain coordinate-order indexing and include anti-parallel direction metadata.
- Existing diffraction scoring reuse: generated candidate PDB/XYZ coordinates can be passed through `scripts/analyze_backbone_pair_family_cd.py`, `scripts/rollup_rich_coordinate_cd_diagnostics.py`, and the parametric Debye helpers in `src/hexaplex_backbone_fingerprint/parametric_powder_scan.py`.

## Output Tables

- Metrics CSV: `outputs/metrics/backbone_torsion_repeat_audit.csv`
- Repeat-window rows are included in the same CSV with `row_type = repeat_window`.
"""
    path.write_text(text, encoding="utf-8")


def find_parent_pdb() -> Path:
    """Resolve parent PDB from first-panel summary."""
    summary = pd.read_csv(ROOT / "outputs/six_strand_first_panel/six_strand_first_panel_summary.csv")
    row = summary[summary["label"] == PARENT_LABEL]
    if row.empty:
        raise FileNotFoundError("Could not locate ideal parent PDB in first-panel summary.")
    return Path(str(row.iloc[0]["source_path"]))


def find_variant_pdb() -> Path | None:
    """Resolve optional backbone-plus-carboxylate variant PDB."""
    manifest_path = ROOT / "outputs/coordinates/ideal_hexaflex_variants/variant_manifest.csv"
    if not manifest_path.exists():
        return None
    manifest = pd.read_csv(manifest_path)
    row = manifest[manifest["variant"] == "backbone_plus_carboxylate"]
    if row.empty:
        return None
    return ROOT / Path(str(row.iloc[0]["pdb_path"]))


def run_audit(parent_pdb: Path, variant_pdb: Path | None, metrics_dir: Path, reports_dir: Path) -> pd.DataFrame:
    """Run torsion/repeat audit and write outputs."""
    frames = []
    report_chain_frames = []
    report_residue_frames = []
    report_repeat_frames = []
    inputs = [(PARENT_LABEL, parent_pdb)]
    if variant_pdb is not None:
        inputs.append(("ideal_hexaflex_backbone_plus_carboxylate", variant_pdb))
    for label, path in inputs:
        chain_df, residue_df, repeat_df = audit_pdb(path, label)
        report_chain_frames.append(chain_df)
        report_residue_frames.append(residue_df)
        report_repeat_frames.append(repeat_df)
        frames.append(chain_df.assign(row_type="chain_summary"))
        frames.append(residue_df.assign(row_type="residue_torsion"))
        frames.append(repeat_df.assign(row_type="repeat_window"))
    out = pd.concat(frames, ignore_index=True, sort=False)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(metrics_dir / "backbone_torsion_repeat_audit.csv", index=False)
    write_report(
        pd.concat(report_chain_frames, ignore_index=True),
        pd.concat(report_residue_frames, ignore_index=True),
        pd.concat(report_repeat_frames, ignore_index=True),
        reports_dir / "backbone_torsion_repeat_audit.md",
    )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-pdb", type=Path, default=None)
    parser.add_argument("--variant-pdb", type=Path, default=None)
    parser.add_argument("--metrics-dir", type=Path, default=Path("outputs/metrics"))
    parser.add_argument("--reports-dir", type=Path, default=Path("outputs/reports"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    parent = args.parent_pdb or find_parent_pdb()
    variant = args.variant_pdb if args.variant_pdb is not None else find_variant_pdb()
    out = run_audit(parent, variant, args.metrics_dir, args.reports_dir)
    print(f"Wrote {len(out)} audit rows")
    print(f"Metrics: {args.metrics_dir / 'backbone_torsion_repeat_audit.csv'}")
    print(f"Report: {args.reports_dir / 'backbone_torsion_repeat_audit.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
