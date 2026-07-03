"""Audit chemically/register-defined layer models against z-slice layers."""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.score_radial_axial_refinement_variant_cd import markdown_table
from scripts.validate_parent_axial_layer_assignments import (
    DEFAULT_SOURCE_PDB,
    build_atom_assignments,
    layer_centers_from_atoms,
    parse_parent_pdb,
)


DEFAULT_ZSLICE_ASSIGNMENTS = Path("outputs/metrics/parent_axial_layer_atom_assignments.csv")
DEFAULT_SUMMARY_CSV = Path("outputs/metrics/register_defined_layer_model_summary.csv")
DEFAULT_COMPOSITION_CSV = Path("outputs/metrics/register_defined_layer_composition.csv")
DEFAULT_MAPPING_CSV = Path("outputs/metrics/register_to_zslice_layer_mapping.csv")
DEFAULT_REPORT = Path("outputs/reports/register_defined_layer_model_audit.md")
DEFAULT_FIGURE_DIR = Path("outputs/figures")
BACKBONE_ATOMS = {"N", "CA", "C", "O"}


@dataclass(frozen=True)
class AtomRecord:
    """PDB atom record for register-layer auditing."""

    atom_serial: int
    atom_name: str
    residue_name: str
    chain_id: str
    residue_number: int
    insertion_code: str
    element: str
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class ResidueRecord:
    """Residue-level record with per-chain coordinate order."""

    chain_id: str
    residue_number: int
    insertion_code: str
    residue_name: str
    residue_order_index: int
    atoms: tuple[AtomRecord, ...]
    atom_count: int
    atom_names_present: tuple[str, ...]
    z_min: float
    z_max: float
    z_mean: float
    z_median: float
    ca_z: float | None
    has_N: bool
    has_CA: bool
    has_C: bool
    has_O: bool
    peptide_plane_z_mean: float | None


def infer_element(atom_name: str, line: str = "") -> str:
    """Infer element from PDB element column or atom name."""
    element = line[76:78].strip() if len(line) >= 78 else ""
    if element:
        return element.upper()
    match = re.search(r"[A-Za-z]+", atom_name.strip())
    if not match:
        return ""
    return match.group(0)[0].upper()


def parse_atom_line(line: str) -> AtomRecord:
    """Parse one ATOM/HETATM line."""
    return AtomRecord(
        atom_serial=int(line[6:11]),
        atom_name=line[12:16].strip(),
        residue_name=line[17:20].strip(),
        chain_id=line[21:22].strip(),
        residue_number=int(line[22:26]),
        insertion_code=line[26:27].strip(),
        element=infer_element(line[12:16].strip(), line),
        x=float(line[30:38]),
        y=float(line[38:46]),
        z=float(line[46:54]),
    )


def parse_pdb_atoms(path: Path) -> list[AtomRecord]:
    """Parse parent PDB atom records."""
    atoms = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            atoms.append(parse_atom_line(line))
    if not atoms:
        raise ValueError(f"No atoms found in {path}.")
    return atoms


def peptide_plane_z_mean(atoms: list[AtomRecord]) -> float | None:
    """Return mean z for N/CA/C/O atoms when complete."""
    by_name = {atom.atom_name: atom for atom in atoms}
    if not BACKBONE_ATOMS.issubset(by_name):
        return None
    return float(np.mean([by_name[name].z for name in sorted(BACKBONE_ATOMS)]))


def build_residues(atoms: list[AtomRecord]) -> list[ResidueRecord]:
    """Build residue records with per-chain coordinate/PDB order indices."""
    grouped: dict[tuple[str, int, str, str], list[AtomRecord]] = {}
    order: list[tuple[str, int, str, str]] = []
    for atom in atoms:
        key = (atom.chain_id, atom.residue_number, atom.insertion_code, atom.residue_name)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(atom)
    chain_counts: dict[str, int] = {}
    residues = []
    for key in order:
        chain_id, residue_number, insertion_code, residue_name = key
        idx = chain_counts.get(chain_id, 0)
        chain_counts[chain_id] = idx + 1
        group = grouped[key]
        z_values = [atom.z for atom in group]
        names = tuple(sorted({atom.atom_name for atom in group}))
        ca = [atom.z for atom in group if atom.atom_name == "CA"]
        residues.append(
            ResidueRecord(
                chain_id=chain_id,
                residue_number=residue_number,
                insertion_code=insertion_code,
                residue_name=residue_name,
                residue_order_index=idx,
                atoms=tuple(group),
                atom_count=len(group),
                atom_names_present=names,
                z_min=min(z_values),
                z_max=max(z_values),
                z_mean=float(np.mean(z_values)),
                z_median=float(np.median(z_values)),
                ca_z=ca[0] if ca else None,
                has_N="N" in names,
                has_CA="CA" in names,
                has_C="C" in names,
                has_O="O" in names,
                peptide_plane_z_mean=peptide_plane_z_mean(group),
            )
        )
    return residues


def residue_rows(residues: list[ResidueRecord]) -> pd.DataFrame:
    """Convert residue records to a dataframe."""
    return pd.DataFrame(
        [
            {
                "chain_id": r.chain_id,
                "residue_number": r.residue_number,
                "insertion_code": r.insertion_code,
                "residue_name": r.residue_name,
                "residue_order_index": r.residue_order_index,
                "atom_count": r.atom_count,
                "atom_names_present": ",".join(r.atom_names_present),
                "z_min_A": r.z_min,
                "z_max_A": r.z_max,
                "z_mean_A": r.z_mean,
                "z_median_A": r.z_median,
                "ca_z_A": r.ca_z,
                "has_N": r.has_N,
                "has_CA": r.has_CA,
                "has_C": r.has_C,
                "has_O": r.has_O,
                "peptide_plane_z_mean_A": r.peptide_plane_z_mean,
            }
            for r in residues
        ]
    )


def atoms_for_residues(residues: list[ResidueRecord]) -> list[AtomRecord]:
    """Flatten atoms from residues."""
    return [atom for residue in residues for atom in residue.atoms]


def layer_records_for_model(model_name: str, residues: list[ResidueRecord]) -> list[dict[str, object]]:
    """Build layer rows for one register-defined model."""
    rows = []
    if model_name == "residue_index_layer":
        groups = [(idx, [r for r in residues if r.residue_order_index == idx]) for idx in sorted({r.residue_order_index for r in residues})]
        for layer_id, group in groups:
            rows.append(layer_row_from_residues(model_name, layer_id, group, atoms_for_residues(group), [r.z_mean for r in group]))
    elif model_name == "ca_register_layer":
        groups = [(idx, [r for r in residues if r.residue_order_index == idx and r.ca_z is not None]) for idx in sorted({r.residue_order_index for r in residues})]
        for layer_id, group in groups:
            ca_atoms = [a for r in group for a in r.atoms if a.atom_name == "CA"]
            rows.append(layer_row_from_residues(model_name, layer_id, group, ca_atoms, [r.ca_z for r in group if r.ca_z is not None]))
    elif model_name == "repeat_pair_layer":
        pair_ids = sorted({r.residue_order_index // 2 for r in residues})
        for layer_id in pair_ids:
            group = [r for r in residues if r.residue_order_index // 2 == layer_id]
            rows.append(layer_row_from_residues(model_name, layer_id, group, atoms_for_residues(group), [r.z_mean for r in group]))
    elif model_name == "peptide_plane_layer":
        groups = [
            (idx, [r for r in residues if r.residue_order_index == idx and r.peptide_plane_z_mean is not None])
            for idx in sorted({r.residue_order_index for r in residues})
        ]
        for layer_id, group in groups:
            rows.append(layer_row_from_residues(model_name, layer_id, group, [], [r.peptide_plane_z_mean for r in group if r.peptide_plane_z_mean is not None]))
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return rows


def layer_row_from_residues(
    model_name: str,
    layer_id: int,
    residues: list[ResidueRecord],
    atoms: list[AtomRecord],
    representative_z_values: list[float],
) -> dict[str, object]:
    """Build one register layer composition row."""
    z_values = [atom.z for atom in atoms] if atoms else [z for z in representative_z_values if z is not None]
    chains = sorted({r.chain_id for r in residues})
    residue_names = sorted({r.residue_name for r in residues})
    order_indices = sorted({r.residue_order_index for r in residues})
    return {
        "model_name": model_name,
        "layer_id": layer_id,
        "atom_count": len(atoms),
        "residue_count": len(residues),
        "chain_count": len(chains),
        "chains_present": ",".join(chains),
        "residue_names_present": ",".join(residue_names),
        "residue_order_indices": ",".join(str(v) for v in order_indices),
        "z_mean_A": float(np.mean(z_values)) if z_values else np.nan,
        "z_median_A": float(np.median(z_values)) if z_values else np.nan,
        "z_min_A": min(z_values) if z_values else np.nan,
        "z_max_A": max(z_values) if z_values else np.nan,
        "z_thickness_A": (max(z_values) - min(z_values)) if z_values else np.nan,
        "representative_z_A": float(np.mean(representative_z_values)) if representative_z_values else np.nan,
        "notes": "" if residues else "empty_layer",
    }


def add_next_layer_delta(composition: pd.DataFrame) -> pd.DataFrame:
    """Add next-layer representative z deltas by model."""
    out = composition.sort_values(["model_name", "layer_id"]).copy()
    out["next_layer_delta_z_A"] = out.groupby("model_name")["representative_z_A"].shift(-1) - out["representative_z_A"]
    return out


def model_summary(model_name: str, composition: pd.DataFrame, split_residue_count: int, total_residues: int, expected_chains: list[str]) -> dict[str, object]:
    """Summarize one layer model."""
    sub = composition[composition["model_name"] == model_name]
    deltas = pd.to_numeric(sub["next_layer_delta_z_A"], errors="coerce").dropna()
    thickness = pd.to_numeric(sub["z_thickness_A"], errors="coerce").dropna()
    return {
        "model_name": model_name,
        "layer_count": len(sub),
        "expected_chains": ",".join(expected_chains),
        "mean_chains_per_layer": sub["chain_count"].mean(),
        "layers_with_all_expected_chains": int((sub["chain_count"] >= len(expected_chains)).sum()),
        "mean_atoms_per_layer": sub["atom_count"].mean(),
        "median_atoms_per_layer": sub["atom_count"].median(),
        "mean_residues_per_layer": sub["residue_count"].mean(),
        "median_residues_per_layer": sub["residue_count"].median(),
        "mean_z_A": sub["representative_z_A"].mean(),
        "z_span_A": sub["representative_z_A"].max() - sub["representative_z_A"].min(),
        "mean_layer_to_layer_delta_z_A": deltas.mean(),
        "median_layer_to_layer_delta_z_A": deltas.median(),
        "min_layer_to_layer_delta_z_A": deltas.min(),
        "max_layer_to_layer_delta_z_A": deltas.max(),
        "layer_delta_z_std_A": deltas.std(),
        "mean_layer_thickness_A": thickness.mean(),
        "median_layer_thickness_A": thickness.median(),
        "max_layer_thickness_A": thickness.max(),
        "split_residue_count": split_residue_count,
        "residues_split_fraction": split_residue_count / total_residues if total_residues else np.nan,
        "notes": "",
    }


def zslice_summary(zslice_assignments: pd.DataFrame, residue_df: pd.DataFrame, expected_chains: list[str]) -> dict[str, object]:
    """Summarize existing z-slice assignment."""
    comp = []
    for layer_id, group in zslice_assignments.groupby("layer_index"):
        residues = group[["chain_id", "residue_number", "residue_name"]].drop_duplicates()
        comp.append(
            {
                "model_name": "z_slice_layer",
                "layer_id": layer_id,
                "atom_count": len(group),
                "residue_count": len(residues),
                "chain_count": group["chain_id"].nunique(),
                "representative_z_A": group["z"].mean(),
                "z_thickness_A": group["z"].max() - group["z"].min(),
            }
        )
    comp_df = add_next_layer_delta(pd.DataFrame(comp))
    split_count = int((residue_df["zslice_layer_count"] > 1).sum())
    return model_summary("z_slice_layer", comp_df, split_count, len(residue_df), expected_chains)


def zslice_participation_by_residue(zslice_assignments: pd.DataFrame) -> pd.DataFrame:
    """Return z-slice participation for residue split counting."""
    rows = []
    for key, group in zslice_assignments.groupby(["chain_id", "residue_number", "residue_name"]):
        rows.append({**dict(zip(["chain_id", "residue_number", "residue_name"], key)), "zslice_layer_count": group["layer_index"].nunique()})
    return pd.DataFrame(rows)


def register_to_zslice_mapping(composition: pd.DataFrame, zslice_assignments: pd.DataFrame) -> pd.DataFrame:
    """Map register-defined layers onto z-slice layers."""
    rows = []
    for _, layer in composition[composition["model_name"] != "z_slice_layer"].iterrows():
        if layer["model_name"] == "ca_register_layer":
            mask = (zslice_assignments["atom_name"] == "CA") & (
                zslice_assignments["residue_number"].isin([])
            )
        chains = str(layer["chains_present"]).split(",") if str(layer["chains_present"]) else []
        order_indices = {int(v) for v in str(layer["residue_order_indices"]).split(",") if v != ""}
        # The assignment CSV lacks residue_order_index, so infer it within each chain from residue order.
        z = zslice_assignments.copy()
        residue_order = (
            z[["chain_id", "residue_number", "residue_name"]]
            .drop_duplicates()
            .sort_values(["chain_id", "residue_number"])
            .assign(residue_order_index=lambda df: df.groupby("chain_id").cumcount())
        )
        z = z.merge(residue_order, on=["chain_id", "residue_number", "residue_name"], how="left")
        if layer["model_name"] == "repeat_pair_layer":
            selected = z[z["residue_order_index"].floordiv(2).eq(int(layer["layer_id"]))]
        else:
            selected = z[z["residue_order_index"].isin(order_indices)]
        if layer["model_name"] == "ca_register_layer":
            selected = selected[selected["atom_name"] == "CA"]
        if selected.empty:
            counts = pd.Series(dtype=int)
        else:
            counts = selected["layer_index"].value_counts()
        primary = int(counts.idxmax()) if not counts.empty else ""
        primary_fraction = float(counts.max() / counts.sum()) if not counts.empty else np.nan
        rows.append(
            {
                "model_name": layer["model_name"],
                "register_layer_id": layer["layer_id"],
                "representative_z_A": layer["representative_z_A"],
                "atom_count": int(layer["atom_count"]),
                "residue_count": int(layer["residue_count"]),
                "chains_present": layer["chains_present"],
                "zslice_layers_overlapped": ",".join(str(int(v)) for v in sorted(counts.index)) if not counts.empty else "",
                "zslice_layer_count": len(counts),
                "primary_zslice_layer": primary,
                "primary_zslice_atom_fraction": primary_fraction,
                "notes": "diffuse_zslice_assignment" if len(counts) > 1 and primary_fraction < 0.8 else "",
            }
        )
    return pd.DataFrame(rows)


def build_register_layers(residues: list[ResidueRecord]) -> pd.DataFrame:
    """Build composition rows for all register-defined models."""
    rows = []
    for model_name in ["residue_index_layer", "ca_register_layer", "repeat_pair_layer", "peptide_plane_layer"]:
        rows.extend(layer_records_for_model(model_name, residues))
    return add_next_layer_delta(pd.DataFrame(rows))


def save_plots(summary: pd.DataFrame, mapping: pd.DataFrame, figure_dir: Path) -> list[Path]:
    """Save optional comparison plots."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(summary["model_name"], summary["median_layer_to_layer_delta_z_A"])
    ax.set_ylabel("median layer delta z (A)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    for suffix in [".png", ".svg"]:
        path = figure_dir / f"register_layer_model_spacing_comparison{suffix}"
        fig.savefig(path, dpi=180)
        outputs.append(path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(mapping["zslice_layer_count"].dropna(), bins=range(1, int(mapping["zslice_layer_count"].max()) + 3), align="left")
    ax.set_xlabel("z-slice layers overlapped")
    ax.set_ylabel("register layers")
    fig.tight_layout()
    for suffix in [".png", ".svg"]:
        path = figure_dir / f"register_to_zslice_overlap_histogram{suffix}"
        fig.savefig(path, dpi=180)
        outputs.append(path)
    plt.close(fig)
    return outputs


def structural_interpretation(summary: pd.DataFrame, mapping: pd.DataFrame) -> str:
    """Return conservative recommendation."""
    residue = summary[summary["model_name"] == "residue_index_layer"].iloc[0]
    repeat = summary[summary["model_name"] == "repeat_pair_layer"].iloc[0]
    clean_fraction = (mapping["primary_zslice_atom_fraction"] >= 0.8).mean()
    if clean_fraction < 0.5:
        return "z-slice model is better as a computational deformation coordinate, while register-defined model is better as a chemical/register coordinate"
    if repeat["median_layer_to_layer_delta_z_A"] > residue["median_layer_to_layer_delta_z_A"]:
        return "Register-defined layers are chemically cleaner but geometrically thicker/more diffuse than z-slices"
    return "Register-defined layers provide a plausible next layer model for physically parameterized rise variants"


def build_report_text(
    source_pdb: Path,
    atoms: list[AtomRecord],
    residues: list[ResidueRecord],
    summary: pd.DataFrame,
    mapping: pd.DataFrame,
    recommendation: str,
    plots: list[Path],
) -> str:
    """Build markdown audit report."""
    chains = sorted({atom.chain_id for atom in atoms})
    ca_count = sum(atom.atom_name == "CA" for atom in atoms)
    complete = sum(r.has_N and r.has_CA and r.has_C and r.has_O for r in residues)
    mapping_stats = mapping.groupby("model_name")["zslice_layer_count"].mean().reset_index()
    clean = mapping[mapping["primary_zslice_atom_fraction"] >= 0.8]
    plot_text = "\n".join(f"- `{path}`" for path in plots) if plots else "_Plots were not generated._"
    return f"""# Register-Defined Layer Model Audit

## 1. Purpose

This audit tests whether the rise parameter can be mapped from computational z-slices to chemically/register-defined layers.

## 2. Input Structure

- Parent PDB: `{source_pdb}`
- Atom count: {len(atoms)}
- Residue count: {len(residues)}
- Chain IDs: {', '.join(chains)}
- C-alpha count: {ca_count}
- Residues with complete N/CA/C/O: {complete}/{len(residues)}

## 3. Models Tested

- `z_slice_layer`: computational z-slices from the previous axial layer assignment.
- `residue_index_layer`: shared per-chain residue order index.
- `ca_register_layer`: shared per-chain C-alpha order index.
- `repeat_pair_layer`: adjacent two-residue repeat pair index.
- `peptide_plane_layer`: per-residue N/CA/C/O z-centers where complete.

## 4. Summary Comparison Table

{markdown_table(summary, ['model_name', 'layer_count', 'mean_chains_per_layer', 'mean_atoms_per_layer', 'mean_residues_per_layer', 'median_layer_to_layer_delta_z_A', 'median_layer_thickness_A', 'split_residue_count', 'residues_split_fraction'])}

## 5. z-slice Versus Register-Defined Contrast

The z-slice model has regular axial spacing but splits most residues. Residue/register models avoid splitting residues by construction, but they are geometrically thicker or more chemically aggregated than computational z-slices. These chemically/register-defined layers should not be interpreted as unique chemical hexad layers without additional register annotation.

## 6. Mapping To z-slice Layers

Mean z-slice overlap count per register model:

{markdown_table(mapping_stats, ['model_name', 'zslice_layer_count'])}

- Register layers with clean primary z-slice assignment (`primary_zslice_atom_fraction >= 0.80`): {len(clean)}/{len(mapping)}
- Diffuse assignments: {len(mapping) - len(clean)}

## 7. Structural Interpretation

{recommendation}. Further mapping to known hexad/register assignments is needed before calling these physical hexad layers.

## 8. Recommended Next Step

Based on this audit, keep `parameterized_rise_0p9750` language as computational z-layer compression unless a chemically defined layer model is selected and tested. A useful next branch would compare residue-index and repeat-pair rise variants without diffraction scoring first, then score only geometry-clean candidates.

## Optional Plots

{plot_text}
"""


def run(
    source_pdb: Path,
    zslice_assignments_path: Path,
    summary_csv: Path,
    composition_csv: Path,
    mapping_csv: Path,
    report_path: Path,
    figure_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, list[Path]]:
    """Run register-defined layer audit and write outputs."""
    atoms = parse_pdb_atoms(source_pdb)
    residues = build_residues(atoms)
    composition = build_register_layers(residues)
    expected_chains = sorted({atom.chain_id for atom in atoms})
    if zslice_assignments_path.exists():
        zslice = pd.read_csv(zslice_assignments_path)
    else:
        parent_atoms = parse_parent_pdb(source_pdb)
        centers = layer_centers_from_atoms(parent_atoms)
        zslice = build_atom_assignments(parent_atoms, centers, "ca_z_gap_nearest_layer_center")
    zslice_residue = zslice_participation_by_residue(zslice)
    summary_rows = [zslice_summary(zslice, zslice_residue, expected_chains)]
    total_residues = len(residues)
    for model_name in ["residue_index_layer", "ca_register_layer", "repeat_pair_layer", "peptide_plane_layer"]:
        split_count = 0
        summary_rows.append(model_summary(model_name, composition, split_count, total_residues, expected_chains))
    summary = pd.DataFrame(summary_rows)
    mapping = register_to_zslice_mapping(composition, zslice)
    recommendation = structural_interpretation(summary, mapping)
    plots = save_plots(summary, mapping, figure_dir)
    for path, df in [(summary_csv, summary), (composition_csv, composition), (mapping_csv, mapping)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report_text(source_pdb, atoms, residues, summary, mapping, recommendation, plots), encoding="utf-8")
    return summary, composition, mapping, recommendation, plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pdb", type=Path, default=DEFAULT_SOURCE_PDB)
    parser.add_argument("--zslice-assignments", type=Path, default=DEFAULT_ZSLICE_ASSIGNMENTS)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--composition-csv", type=Path, default=DEFAULT_COMPOSITION_CSV)
    parser.add_argument("--mapping-csv", type=Path, default=DEFAULT_MAPPING_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary, composition, mapping, recommendation, plots = run(
        args.source_pdb,
        args.zslice_assignments,
        args.summary_csv,
        args.composition_csv,
        args.mapping_csv,
        args.report,
        args.figure_dir,
    )
    print(f"Summary rows: {len(summary)}")
    print(f"Composition rows: {len(composition)}")
    print(f"Mapping rows: {len(mapping)}")
    print(f"Recommendation: {recommendation}")
    print(f"Report: {args.report}")
    print(f"Plots: {len(plots)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
