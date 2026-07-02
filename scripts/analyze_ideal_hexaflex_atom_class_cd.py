"""Decompose ideal Hexaflex C/D diagnostics by atom class and pair geometry.

This is a diagnostic layer on top of the existing pair-family decomposition.
Atom classes are intentionally overlapping: for example, a peptide-plane oxygen
belongs to `heavy_all`, `backbone`, `peptide_plane`, and `oxygen`.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hexaplex_backbone_fingerprint.parametric_powder_scan import local_maxima, make_q_grid

from scripts.analyze_backbone_pair_family_cd import CHAIN_IDS, classify_pair_families, partial_debye_profile
from scripts.build_ideal_hexaflex_selection_variants import (
    PdbAtomRecord,
    is_backbone_atom,
    is_carboxylate_atom,
    is_hydrogen,
    is_peptide_plane_atom,
    is_side_chain_atom,
)


GEOMETRY_FAMILIES = [
    "all_same_strand",
    "all_cross_strand",
    "adjacent_strand_same_register",
    "adjacent_strand_plusminus1_register",
    "adjacent_strand_plusminus2_or_more",
    "nonadjacent_cross_strand",
    "same_strand_plusminus1_repeat",
    "same_strand_plusminus2_or_more",
    "alternating_interfaces_AB_CD_EF",
    "alternating_interfaces_BC_DE_FA",
]
ATOM_CLASSES = [
    "H",
    "heavy_all",
    "backbone",
    "peptide_plane",
    "side_chain",
    "carboxylate",
    "backbone_non_peptide_plane",
    "side_chain_non_carboxylate",
    "carbon",
    "nitrogen",
    "oxygen",
    "other_heavy",
]
PAIR_PRIORITY = {name: idx for idx, name in enumerate(ATOM_CLASSES)}


@dataclass(frozen=True)
class RichAtom:
    """PDB atom with coordinates and geometry labels."""

    atom_index: int
    atom_name: str
    resname: str
    element: str
    chain: str
    strand_index: int
    repeat_index: int
    coord: np.ndarray


def safe_model_id(text: str | Path) -> str:
    """Return a filename-safe model ID."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def parse_rich_pdb(path: Path) -> list[RichAtom]:
    """Parse labeled PDB atoms, retaining hydrogens for class diagnostics."""
    atoms: list[RichAtom] = []
    for line in path.read_text(encoding="ascii").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        atom_name = line[12:16].strip()
        resname = line[17:20].strip()
        element = (line[76:78].strip() or atom_name[:1]).upper()
        chain = line[21].strip()
        if not chain:
            raise ValueError(f"PDB atom without chain ID in {path}; strand labels are required.")
        if chain not in CHAIN_IDS:
            raise ValueError(f"Unsupported one-character chain ID {chain!r} in {path}.")
        resseq = int(line[22:26])
        atoms.append(
            RichAtom(
                atom_index=int(line[6:11]),
                atom_name=atom_name,
                resname=resname,
                element=element,
                chain=chain,
                strand_index=CHAIN_IDS.index(chain),
                repeat_index=(resseq - 1) // 2,
                coord=np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=float),
            )
        )
    if not atoms:
        raise ValueError(f"No ATOM/HETATM records found in {path}.")
    return atoms


def rich_to_pdb_record(atom: RichAtom) -> PdbAtomRecord:
    """Convert a rich atom to the selection-helper record shape."""
    return PdbAtomRecord(
        line="",
        atom_name=atom.atom_name,
        resname=atom.resname,
        chain=atom.chain,
        resseq=atom.repeat_index,
        element=atom.element,
    )


def atom_classes(atom: RichAtom) -> tuple[str, ...]:
    """Return overlapping conservative atom classes for one atom."""
    record = rich_to_pdb_record(atom)
    classes: list[str] = []
    hydrogen = is_hydrogen(record)
    if hydrogen:
        classes.append("H")
        return tuple(classes)

    classes.append("heavy_all")
    if is_backbone_atom(record):
        classes.append("backbone")
    if is_peptide_plane_atom(record):
        classes.append("peptide_plane")
    if is_side_chain_atom(record):
        classes.append("side_chain")
    if is_carboxylate_atom(record):
        classes.append("carboxylate")
    if is_backbone_atom(record) and not is_peptide_plane_atom(record):
        classes.append("backbone_non_peptide_plane")
    if is_side_chain_atom(record) and not is_carboxylate_atom(record):
        classes.append("side_chain_non_carboxylate")

    element = atom.element.upper()
    if element == "C":
        classes.append("carbon")
    elif element == "N":
        classes.append("nitrogen")
    elif element == "O":
        classes.append("oxygen")
    else:
        classes.append("other_heavy")
    return tuple(classes)


def canonical_pair_class(class_a: str, class_b: str) -> tuple[str, str]:
    """Return an order-stable atom-class pair."""
    key_a = PAIR_PRIORITY.get(class_a, len(PAIR_PRIORITY)), class_a
    key_b = PAIR_PRIORITY.get(class_b, len(PAIR_PRIORITY)), class_b
    return (class_a, class_b) if key_a <= key_b else (class_b, class_a)


def pair_class_keys(classes_a: tuple[str, ...], classes_b: tuple[str, ...]) -> set[tuple[str, str]]:
    """Return canonical class-pair keys for two atoms."""
    return {canonical_pair_class(a, b) for a in classes_a for b in classes_b}


def filtered_geometry_families(atom_a: RichAtom, atom_b: RichAtom, n_strands: int = 6) -> list[str]:
    """Return the geometry families requested for atom-class diagnostics."""
    families = classify_pair_families(atom_a, atom_b, n_strands=n_strands)
    return [family for family in families if family in GEOMETRY_FAMILIES]


def compute_atom_class_family_distances(
    atoms: list[RichAtom], n_strands: int = 6
) -> dict[tuple[str, str, str], list[float]]:
    """Compute distances by atom-class pair and geometry family."""
    distances_by_key: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    coords = np.array([atom.coord for atom in atoms])
    class_cache = [atom_classes(atom) for atom in atoms]
    for i in range(len(atoms) - 1):
        deltas = coords[i + 1 :] - coords[i]
        distances = np.linalg.norm(deltas, axis=1)
        for local_j, distance in enumerate(distances):
            j = i + 1 + local_j
            geometry_families = filtered_geometry_families(atoms[i], atoms[j], n_strands=n_strands)
            if not geometry_families:
                continue
            for class_1, class_2 in pair_class_keys(class_cache[i], class_cache[j]):
                for family in geometry_families:
                    distances_by_key[(class_1, class_2, family)].append(float(distance))
    return distances_by_key


def window_profile_peak(profile: pd.DataFrame, d_min: float, d_max: float) -> tuple[float, float]:
    """Return max intensity and d-spacing in a d-window."""
    window = profile[(profile["d_A"] >= d_min) & (profile["d_A"] <= d_max)]
    if window.empty:
        return np.nan, np.nan
    maxima = local_maxima(window.rename(columns={"q": "q_Ainv"}))
    source = maxima if not maxima.empty else window
    row = source.sort_values("intensity", ascending=False).iloc[0]
    return float(row["intensity"]), float(row["d_A"])


def write_profiles(
    model_id: str,
    distances_by_key: dict[tuple[str, str, str], list[float]],
    q_values: np.ndarray,
    path: Path,
) -> pd.DataFrame:
    """Write atom-class partial radial profiles."""
    frames = []
    for (class_1, class_2, family), distances in sorted(distances_by_key.items()):
        profile = partial_debye_profile(np.asarray(distances, dtype=float), q_values)
        profile.insert(0, "geometry_family", family)
        profile.insert(0, "atom_class_2", class_2)
        profile.insert(0, "atom_class_1", class_1)
        profile.insert(0, "model_id", model_id)
        frames.append(profile)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    df.to_csv(path, index=False)
    return df


def write_summary(
    model_id: str,
    distances_by_key: dict[tuple[str, str, str], list[float]],
    profiles: pd.DataFrame,
    path: Path,
    c_window: tuple[float, float],
    d_window: tuple[float, float],
) -> pd.DataFrame:
    """Write C/D summary by atom class pair and geometry family."""
    rows = []
    for (class_1, class_2, family), distances in sorted(distances_by_key.items()):
        arr = np.asarray(distances, dtype=float)
        profile = profiles[
            (profiles["atom_class_1"] == class_1)
            & (profiles["atom_class_2"] == class_2)
            & (profiles["geometry_family"] == family)
        ]
        c_intensity, c_peak = window_profile_peak(profile, *c_window)
        d_intensity, d_peak = window_profile_peak(profile, *d_window)
        rows.append(
            {
                "model_id": model_id,
                "atom_class_1": class_1,
                "atom_class_2": class_2,
                "geometry_family": family,
                "C_pair_count": int(((arr >= c_window[0]) & (arr <= c_window[1])).sum()),
                "D_pair_count": int(((arr >= d_window[0]) & (arr <= d_window[1])).sum()),
                "C_profile_max_intensity": c_intensity,
                "C_profile_peak_d_A": c_peak,
                "D_profile_max_intensity": d_intensity,
                "D_profile_peak_d_A": d_peak,
                "D_minus_C_peak_strength_ratio": d_intensity / c_intensity if c_intensity not in {0.0, np.nan} else np.nan,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def top_row(summary: pd.DataFrame, column: str) -> pd.Series:
    """Return the row with the largest numeric value."""
    if summary.empty:
        raise ValueError("Cannot choose a top row from an empty summary.")
    values = pd.to_numeric(summary[column], errors="coerce").fillna(float("-inf"))
    return summary.loc[values.idxmax()]


def pair_label(row: pd.Series) -> str:
    """Format an atom-class/geometry row label."""
    return f"{row.atom_class_1} x {row.atom_class_2} / {row.geometry_family}"


def write_report(model_id: str, input_pdb: Path, summary: pd.DataFrame, path: Path) -> None:
    """Write focused atom-class C/D markdown report."""
    c_pair = top_row(summary, "C_pair_count")
    d_pair = top_row(summary, "D_pair_count")
    c_profile = top_row(summary.fillna({"C_profile_max_intensity": float("-inf")}), "C_profile_max_intensity")
    d_profile = top_row(summary.fillna({"D_profile_max_intensity": float("-inf")}), "D_profile_max_intensity")

    def count_for(class_1: str, class_2: str, family: str, column: str) -> int:
        a, b = canonical_pair_class(class_1, class_2)
        subset = summary[
            (summary["atom_class_1"] == a)
            & (summary["atom_class_2"] == b)
            & (summary["geometry_family"] == family)
        ]
        if subset.empty:
            return 0
        return int(pd.to_numeric(subset.iloc[0][column], errors="coerce") or 0)

    text = f"""# Atom-Class C/D Diagnostic: `{model_id}`

Input PDB: `{input_pdb}`

This is a diagnostic decomposition. Atom classes are overlapping, not mutually exclusive. For example, a peptide-plane oxygen contributes to `heavy_all`, `backbone`, `peptide_plane`, and `oxygen`.

## Top C and D Contributors

- Top C by pair count: `{pair_label(c_pair)}` ({int(c_pair.C_pair_count)} C-window pairs)
- Top D by pair count: `{pair_label(d_pair)}` ({int(d_pair.D_pair_count)} D-window pairs)
- Top C by partial-profile intensity: `{pair_label(c_profile)}` (peak d = {c_profile.C_profile_peak_d_A:.3f} A)
- Top D by partial-profile intensity: `{pair_label(d_profile)}` (peak d = {d_profile.D_profile_peak_d_A:.3f} A)

## Focused Checks

- `peptide_plane x peptide_plane` cross-strand C/D counts: {count_for('peptide_plane', 'peptide_plane', 'all_cross_strand', 'C_pair_count')}/{count_for('peptide_plane', 'peptide_plane', 'all_cross_strand', 'D_pair_count')}
- `backbone x backbone` cross-strand C/D counts: {count_for('backbone', 'backbone', 'all_cross_strand', 'C_pair_count')}/{count_for('backbone', 'backbone', 'all_cross_strand', 'D_pair_count')}
- `peptide_plane x carboxylate` cross-strand C/D counts: {count_for('peptide_plane', 'carboxylate', 'all_cross_strand', 'C_pair_count')}/{count_for('peptide_plane', 'carboxylate', 'all_cross_strand', 'D_pair_count')}
- `backbone x carboxylate` cross-strand C/D counts: {count_for('backbone', 'carboxylate', 'all_cross_strand', 'C_pair_count')}/{count_for('backbone', 'carboxylate', 'all_cross_strand', 'D_pair_count')}
- `side_chain_non_carboxylate x side_chain_non_carboxylate` C/D counts, all cross-strand: {count_for('side_chain_non_carboxylate', 'side_chain_non_carboxylate', 'all_cross_strand', 'C_pair_count')}/{count_for('side_chain_non_carboxylate', 'side_chain_non_carboxylate', 'all_cross_strand', 'D_pair_count')}
- `H x H` all-same/all-cross C counts: {count_for('H', 'H', 'all_same_strand', 'C_pair_count')}/{count_for('H', 'H', 'all_cross_strand', 'C_pair_count')}

## Interpretation Prompts

- Which atom-class pair dominates C? `{pair_label(c_pair)}` by raw pair count.
- Which atom-class pair dominates D? `{pair_label(d_pair)}` by raw pair count.
- Does C come from peptide-plane core or broader backbone? Compare peptide-plane and backbone rows in the summary CSV.
- Do carboxylates tune D? Inspect `peptide_plane/backbone x carboxylate` cross-strand and adjacent-register rows.
- Are side-chain non-carboxylate contacts pushing peaks high? Compare this model with the side-chain-only and full/no-H rollup rows.

Output summary CSV: `outputs/metrics/{model_id}_atom_class_cd_summary.csv`
Output radial profiles CSV: `outputs/metrics/{model_id}_atom_class_radial_profiles.csv`
"""
    path.write_text(text, encoding="utf-8")


def analyze(
    input_pdb: Path,
    model_id: str,
    out_dir: Path,
    figure_dir: Path,
    report_dir: Path,
    c_window: tuple[float, float],
    d_window: tuple[float, float],
    q_step: float = 0.01,
    d_profile_min: float = 2.5,
    d_profile_max: float = 12.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run atom-class C/D diagnostics for one PDB."""
    del figure_dir  # Reserved for future focused plots.
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    atoms = parse_rich_pdb(input_pdb)
    distances_by_key = compute_atom_class_family_distances(atoms)
    q_values = make_q_grid(d_min_A=d_profile_min, d_max_A=d_profile_max, q_step=q_step)
    profiles = write_profiles(
        model_id,
        distances_by_key,
        q_values,
        out_dir / f"{model_id}_atom_class_radial_profiles.csv",
    )
    summary = write_summary(
        model_id,
        distances_by_key,
        profiles,
        out_dir / f"{model_id}_atom_class_cd_summary.csv",
        c_window,
        d_window,
    )
    write_report(model_id, input_pdb, summary, report_dir / f"{model_id}_atom_class_cd_report.md")
    return summary, profiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-pdb", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/metrics"))
    parser.add_argument("--figure-dir", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--report-dir", type=Path, default=Path("outputs/reports"))
    parser.add_argument("--c-min", type=float, default=5.4)
    parser.add_argument("--c-max", type=float, default=5.8)
    parser.add_argument("--d-min", type=float, default=7.0)
    parser.add_argument("--d-max", type=float, default=7.5)
    parser.add_argument("--q-step", type=float, default=0.01)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_id = safe_model_id(args.model_id)
    summary, profiles = analyze(
        args.input_pdb,
        model_id,
        args.out_dir,
        args.figure_dir,
        args.report_dir,
        (args.c_min, args.c_max),
        (args.d_min, args.d_max),
        q_step=args.q_step,
    )
    print(f"Analyzed atom-class diagnostics for {model_id}")
    print(f"Summary rows: {len(summary)}")
    print(f"Profile rows: {len(profiles)}")
    print(f"Summary: {args.out_dir / f'{model_id}_atom_class_cd_summary.csv'}")
    print(f"Report: {args.report_dir / f'{model_id}_atom_class_cd_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
