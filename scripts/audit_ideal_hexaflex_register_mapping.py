"""Audit residue/register mapping assumptions for ideal anti-parallel Hexaflex."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_best_clean_model_register_interfaces import (
    alternating_interface_group,
    interface_label as rich_interface_label,
    register_offset_class as current_register_offset_class,
)
from scripts.analyze_ideal_hexaflex_atom_class_cd import RichAtom, atom_classes, canonical_pair_class, pair_class_keys


PARENT_LABEL = "full_hexaplex_anti_parallel_30deg_ideal_deduped_6chain"
VARIANT_LABEL = "ideal_hexaflex_backbone_plus_carboxylate_pair_family_cd"
FOCUS_CLASS_PAIRS = {("backbone", "backbone"), ("backbone", "carboxylate")}
INTERFACES = {"AB", "BC", "CD", "DE", "EF", "FA"}


@dataclass(frozen=True)
class AuditAtom:
    """PDB atom with raw and normalized residue/register labels."""

    atom_index: int
    atom_name: str
    resname: str
    element: str
    chain: str
    strand_index: int
    resseq: int
    coord: np.ndarray
    coordinate_repeat_index: int
    reversed_repeat_index: int
    antiparallel_repeat_index: int


def parse_pdb_atoms(path: Path) -> list[tuple[int, str, str, str, str, int, np.ndarray]]:
    """Parse ATOM/HETATM records needed for the audit."""
    rows = []
    for line in path.read_text(encoding="ascii").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        chain = line[21].strip()
        if not chain:
            raise ValueError(f"Atom without chain ID in {path}")
        rows.append(
            (
                int(line[6:11]),
                line[12:16].strip(),
                line[17:20].strip(),
                (line[76:78].strip() or line[12:16].strip()[:1]).upper(),
                chain,
                int(line[22:26]),
                np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=float),
            )
        )
    if not rows:
        raise ValueError(f"No ATOM/HETATM records found in {path}")
    return rows


def unique_residues_in_coordinate_order(rows: list[tuple[int, str, str, str, str, int, np.ndarray]]) -> dict[str, list[int]]:
    """Collect unique residue IDs in first-seen coordinate order for each chain."""
    residues: dict[str, list[int]] = defaultdict(list)
    seen: dict[str, set[int]] = defaultdict(set)
    for _, _, _, _, chain, resseq, _ in rows:
        if resseq not in seen[chain]:
            residues[chain].append(resseq)
            seen[chain].add(resseq)
    return dict(residues)


def repeat_index_maps(residues_by_chain: dict[str, list[int]]) -> dict[str, dict[int, tuple[int, int, int]]]:
    """Build coordinate, reversed, and odd-chain-reversed anti-parallel repeat maps."""
    maps: dict[str, dict[int, tuple[int, int, int]]] = {}
    for chain, residues in residues_by_chain.items():
        n = len(residues)
        strand_index = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789".index(chain)
        reverse_for_antiparallel = strand_index % 2 == 1
        chain_map = {}
        for idx, resseq in enumerate(residues):
            rev = n - 1 - idx
            anti = rev if reverse_for_antiparallel else idx
            chain_map[resseq] = (idx, rev, anti)
        maps[chain] = chain_map
    return maps


def build_audit_atoms(path: Path) -> list[AuditAtom]:
    """Parse a PDB and assign normalized repeat indices."""
    rows = parse_pdb_atoms(path)
    maps = repeat_index_maps(unique_residues_in_coordinate_order(rows))
    atoms = []
    for atom_index, atom_name, resname, element, chain, resseq, coord in rows:
        coord_idx, rev_idx, anti_idx = maps[chain][resseq]
        atoms.append(
            AuditAtom(
                atom_index=atom_index,
                atom_name=atom_name,
                resname=resname,
                element=element,
                chain=chain,
                strand_index="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789".index(chain),
                resseq=resseq,
                coord=coord,
                coordinate_repeat_index=coord_idx,
                reversed_repeat_index=rev_idx,
                antiparallel_repeat_index=anti_idx,
            )
        )
    return atoms


def to_rich(atom: AuditAtom, repeat_index: int | None = None) -> RichAtom:
    """Convert audit atom to RichAtom for existing class/interface helpers."""
    return RichAtom(
        atom_index=atom.atom_index,
        atom_name=atom.atom_name,
        resname=atom.resname,
        element=atom.element,
        chain=atom.chain,
        strand_index=atom.strand_index,
        repeat_index=atom.coordinate_repeat_index if repeat_index is None else repeat_index,
        coord=atom.coord,
    )


def interface_label(atom_a: RichAtom, atom_b: RichAtom) -> str:
    """Return interface label using the existing register-interface helper."""
    return rich_interface_label(atom_a, atom_b)


def classify_delta(delta: int, same_strand: bool = False) -> str:
    """Classify an absolute register/residue offset."""
    if same_strand:
        return "same_strand"
    delta = abs(int(delta))
    if delta == 0:
        return "same"
    if delta == 1:
        return "plusminus1"
    if delta == 2:
        return "plusminus2"
    return "plusminus3_or_more"


def register_classes(atom_a: AuditAtom, atom_b: AuditAtom) -> dict[str, object]:
    """Compare raw/current/coordinate/anti-parallel register schemes."""
    same_strand = atom_a.strand_index == atom_b.strand_index
    raw_delta = atom_b.resseq - atom_a.resseq
    coord_delta = atom_b.coordinate_repeat_index - atom_a.coordinate_repeat_index
    anti_delta = atom_b.antiparallel_repeat_index - atom_a.antiparallel_repeat_index
    current_a = to_rich(atom_a, repeat_index=(atom_a.resseq - 1) // 2)
    current_b = to_rich(atom_b, repeat_index=(atom_b.resseq - 1) // 2)
    return {
        "raw_residue_difference": raw_delta,
        "coordinate_repeat_index_difference": coord_delta,
        "antiparallel_repeat_index_difference": anti_delta,
        "current_register_offset_class": current_register_offset_class(current_a, current_b),
        "raw_register_offset_class": classify_delta(raw_delta, same_strand),
        "coordinate_order_register_offset_class": classify_delta(coord_delta, same_strand),
        "antiparallel_register_offset_class": classify_delta(anti_delta, same_strand),
    }


def chain_mapping_rows(label: str, path: Path) -> list[dict[str, object]]:
    """Summarize chain/residue mapping for one PDB."""
    residues = unique_residues_in_coordinate_order(parse_pdb_atoms(path))
    residue_sets = {chain: set(values) for chain, values in residues.items()}
    all_counts = {chain: len(values) for chain, values in residues.items()}
    first_set = next(iter(residue_sets.values()))
    rows = []
    for chain, values in sorted(residues.items()):
        diffs = np.diff(values)
        if len(diffs) == 0:
            order = "single"
        elif np.all(diffs > 0):
            order = "increasing"
        elif np.all(diffs < 0):
            order = "decreasing"
        else:
            order = "mixed"
        rows.append(
            {
                "source_label": label,
                "source_path": str(path),
                "chain": chain,
                "residue_count": len(values),
                "residue_id_min": min(values),
                "residue_id_max": max(values),
                "first_10_residue_ids": " ".join(map(str, values[:10])),
                "last_10_residue_ids": " ".join(map(str, values[-10:])),
                "same_residue_count_all_chains": len(set(all_counts.values())) == 1,
                "raw_residue_ids_aligned_across_chains": all(residue_sets[c] == first_set for c in residue_sets),
                "residue_id_order": order,
                "equivalent_residue_set": residue_sets[chain] == first_set,
            }
        )
    return rows


def atom_pair_classes(atom_a: AuditAtom, atom_b: AuditAtom) -> set[tuple[str, str]]:
    """Return canonical overlapping atom class pairs for audit atoms."""
    return pair_class_keys(atom_classes(to_rich(atom_a)), atom_classes(to_rich(atom_b)))


def focus_pair_class(atom_a: AuditAtom, atom_b: AuditAtom) -> list[tuple[str, str]]:
    """Return focused pair classes for backbone/backbone and backbone/carboxylate."""
    return sorted(atom_pair_classes(atom_a, atom_b).intersection(FOCUS_CLASS_PAIRS))


def audit_pair_rows(
    atoms: list[AuditAtom],
    c_window: tuple[float, float],
    d_window: tuple[float, float],
    sample_per_window: int = 80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create sampled pair rows and aggregate register-class comparison rows."""
    sample_rows = []
    aggregate: dict[tuple, dict[str, object]] = {}
    coords = np.array([atom.coord for atom in atoms])
    for i in range(len(atoms) - 1):
        atom_a = atoms[i]
        deltas = coords[i + 1 :] - atom_a.coord
        distances = np.linalg.norm(deltas, axis=1)
        for local_j, distance in enumerate(distances):
            in_c = c_window[0] <= distance <= c_window[1]
            in_d = d_window[0] <= distance <= d_window[1]
            if not (in_c or in_d):
                continue
            atom_b = atoms[i + 1 + local_j]
            interface = rich_interface_label(to_rich(atom_a), to_rich(atom_b))
            if interface not in INTERFACES:
                continue
            pair_classes = focus_pair_class(atom_a, atom_b)
            if not pair_classes:
                continue
            registers = register_classes(atom_a, atom_b)
            alt_group = alternating_interface_group(interface)
            for class_1, class_2 in pair_classes:
                for window_name, include in [("C", in_c), ("D", in_d)]:
                    if not include:
                        continue
                    row_base = {
                        "window": window_name,
                        "atom_class_1": class_1,
                        "atom_class_2": class_2,
                        "atom1_chain": atom_a.chain,
                        "atom1_residue_id": atom_a.resseq,
                        "atom1_coordinate_repeat_index": atom_a.coordinate_repeat_index,
                        "atom1_antiparallel_repeat_index": atom_a.antiparallel_repeat_index,
                        "atom1_atom_name": atom_a.atom_name,
                        "atom2_chain": atom_b.chain,
                        "atom2_residue_id": atom_b.resseq,
                        "atom2_coordinate_repeat_index": atom_b.coordinate_repeat_index,
                        "atom2_antiparallel_repeat_index": atom_b.antiparallel_repeat_index,
                        "atom2_atom_name": atom_b.atom_name,
                        "distance_A": float(distance),
                        "interface": interface,
                        "alternating_interface_group": alt_group,
                        **registers,
                    }
                    if len([r for r in sample_rows if r["window"] == window_name]) < sample_per_window:
                        sample_rows.append(row_base)
                    key = (
                        window_name,
                        interface,
                        alt_group,
                        f"{class_1} x {class_2}",
                        registers["current_register_offset_class"],
                        registers["coordinate_order_register_offset_class"],
                        registers["antiparallel_register_offset_class"],
                    )
                    if key not in aggregate:
                        aggregate[key] = {
                            "window": window_name,
                            "interface": interface,
                            "alternating_interface_group": alt_group,
                            "atom_class_pair": f"{class_1} x {class_2}",
                            "current_register_class": registers["current_register_offset_class"],
                            "coordinate_order_register_class": registers["coordinate_order_register_offset_class"],
                            "antiparallel_register_class": registers["antiparallel_register_offset_class"],
                            "pair_count": 0,
                            "median_distance_A_values": [],
                        }
                    aggregate[key]["pair_count"] += 1
                    aggregate[key]["median_distance_A_values"].append(float(distance))
    aggregate_rows = []
    for row in aggregate.values():
        values = row.pop("median_distance_A_values")
        row["median_distance_A"] = float(np.median(values)) if values else np.nan
        aggregate_rows.append(row)
    return pd.DataFrame(sample_rows), pd.DataFrame(aggregate_rows)


def markdown_table(df: pd.DataFrame, columns: list[str], limit: int = 12) -> str:
    """Render a small markdown table."""
    if df.empty:
        return "_No rows._"
    columns = [c for c in columns if c in df.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in df.head(limit)[columns].itertuples(index=False):
        vals = [f"{v:.4g}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(mapping: pd.DataFrame, aggregate: pd.DataFrame, report_path: Path) -> None:
    """Write register-mapping audit report."""
    parent = mapping[mapping["source_label"] == PARENT_LABEL]
    variant = mapping[mapping["source_label"] == VARIANT_LABEL]
    c_rows = aggregate[aggregate["window"] == "C"]
    d_rows = aggregate[aggregate["window"] == "D"]
    current = aggregate.groupby(["window", "current_register_class"])["pair_count"].sum().reset_index()
    coord = aggregate.groupby(["window", "coordinate_order_register_class"])["pair_count"].sum().reset_index()
    anti = aggregate.groupby(["window", "antiparallel_register_class"])["pair_count"].sum().reset_index()
    alt = aggregate.groupby(["window", "alternating_interface_group"])["pair_count"].sum().reset_index()
    text = f"""# Ideal Hexaflex Register Mapping Audit

This audit checks whether the prior `plusminus3_or_more` register result may be caused by raw residue numbering or anti-parallel chain orientation. It reports three candidate schemes side by side: raw residue IDs, coordinate-order residue indices, and an odd-chain-reversed anti-parallel normalization.

## Chain/Residue Mapping

Parent PDB:

{markdown_table(parent, ['chain', 'residue_count', 'residue_id_min', 'residue_id_max', 'first_10_residue_ids', 'last_10_residue_ids', 'same_residue_count_all_chains', 'raw_residue_ids_aligned_across_chains', 'residue_id_order', 'equivalent_residue_set'], limit=20)}

Backbone-plus-carboxylate variant:

{markdown_table(variant, ['chain', 'residue_count', 'residue_id_min', 'residue_id_max', 'first_10_residue_ids', 'last_10_residue_ids', 'same_residue_count_all_chains', 'raw_residue_ids_aligned_across_chains', 'residue_id_order', 'equivalent_residue_set'], limit=20)}

## Register-Class Counts

Current scheme:

{markdown_table(current.sort_values(['window', 'pair_count'], ascending=[True, False]), ['window', 'current_register_class', 'pair_count'], limit=20)}

Coordinate-order scheme:

{markdown_table(coord.sort_values(['window', 'pair_count'], ascending=[True, False]), ['window', 'coordinate_order_register_class', 'pair_count'], limit=20)}

Odd-chain-reversed anti-parallel scheme:

{markdown_table(anti.sort_values(['window', 'pair_count'], ascending=[True, False]), ['window', 'antiparallel_register_class', 'pair_count'], limit=20)}

## Alternating Interface Split

{markdown_table(alt.sort_values(['window', 'pair_count'], ascending=[True, False]), ['window', 'alternating_interface_group', 'pair_count'], limit=20)}

## Direct Answers

- Are residue/repeat labels aligned across chains? The chain tables above show whether raw residue sets and counts match for parent and variant.
- Do anti-parallel chains require index reversal? This remains a modeling choice; the audit provides the odd-chain-reversed scheme explicitly rather than assuming it.
- Does prior `plusminus3_or_more` remain after normalization? Compare the current, coordinate-order, and anti-parallel tables above.
- Are dominant C/D pairs true long-range register-offset pairs or numbering artifacts? If coordinate-order and anti-parallel schemes still concentrate in `plusminus3_or_more`, the result is less likely to be a raw-numbering artifact.
- Does AB/CD/EF vs BC/DE/FA survive normalization? The alternating interface table is independent of register indexing and should be checked directly.
- Suggested future scheme: use coordinate-order indexing as the default neutral scheme, and report an anti-parallel-normalized sensitivity column until physical chain direction is confirmed.
"""
    report_path.write_text(text, encoding="utf-8")


def find_parent_pdb() -> Path:
    """Find parent ideal PDB from existing first-panel summary."""
    summary = pd.read_csv(ROOT / "outputs/six_strand_first_panel/six_strand_first_panel_summary.csv")
    row = summary[summary["label"] == PARENT_LABEL]
    if row.empty:
        raise FileNotFoundError("Could not find parent ideal PDB in six_strand_first_panel_summary.csv")
    return Path(str(row.iloc[0]["source_path"]))


def find_variant_pdb() -> Path:
    """Find backbone-plus-carboxylate variant PDB from manifest."""
    manifest = pd.read_csv(ROOT / "outputs/coordinates/ideal_hexaflex_variants/variant_manifest.csv")
    row = manifest[manifest["variant"] == "backbone_plus_carboxylate"]
    if row.empty:
        raise FileNotFoundError("Could not find backbone_plus_carboxylate in variant_manifest.csv")
    return ROOT / Path(str(row.iloc[0]["pdb_path"]))


def run_audit(parent_pdb: Path, variant_pdb: Path, metrics_dir: Path, reports_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run mapping and pair-register audit."""
    metrics_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    mapping = pd.DataFrame(chain_mapping_rows(PARENT_LABEL, parent_pdb) + chain_mapping_rows(VARIANT_LABEL, variant_pdb))
    sample, aggregate = audit_pair_rows(build_audit_atoms(variant_pdb), (5.4, 5.8), (7.0, 7.5))
    sample.to_csv(metrics_dir / "ideal_hexaflex_register_mapping_audit.csv", index=False)
    aggregate.to_csv(metrics_dir / "ideal_hexaflex_register_mapping_register_class_comparison.csv", index=False)
    mapping.to_csv(metrics_dir / "ideal_hexaflex_register_mapping_chain_summary.csv", index=False)
    write_report(mapping, aggregate, reports_dir / "ideal_hexaflex_register_mapping_audit.md")
    return sample, aggregate


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
    variant = args.variant_pdb or find_variant_pdb()
    sample, aggregate = run_audit(parent, variant, args.metrics_dir, args.reports_dir)
    print(f"Sampled rows: {len(sample)}")
    print(f"Aggregate rows: {len(aggregate)}")
    print(f"Sample CSV: {args.metrics_dir / 'ideal_hexaflex_register_mapping_audit.csv'}")
    print(f"Aggregate CSV: {args.metrics_dir / 'ideal_hexaflex_register_mapping_register_class_comparison.csv'}")
    print(f"Report: {args.reports_dir / 'ideal_hexaflex_register_mapping_audit.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
