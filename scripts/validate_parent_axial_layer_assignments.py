"""Validate parent axial layer assignments used by parameterized rise diagnostics."""

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

from scripts.audit_parent_axial_layers import assign_layer_index, infer_layers_from_ca_z
from scripts.score_radial_axial_refinement_variant_cd import markdown_table


DEFAULT_SOURCE_PDB = Path("outputs/coordinates/ideal_hexaflex_variants/ideal_hexaflex_backbone_plus_carboxylate.pdb")
DEFAULT_ATOM_CSV = Path("outputs/metrics/parent_axial_layer_atom_assignments.csv")
DEFAULT_COMPOSITION_CSV = Path("outputs/metrics/parent_axial_layer_composition.csv")
DEFAULT_RESIDUE_CSV = Path("outputs/metrics/parent_axial_layer_residue_participation.csv")
DEFAULT_CHAIN_CSV = Path("outputs/metrics/parent_axial_layer_chain_coverage.csv")
DEFAULT_REPORT = Path("outputs/reports/parent_axial_layer_assignment_validation.md")
DEFAULT_FIGURE_DIR = Path("outputs/figures")
EXPECTED_CHAINS = ["A", "B", "C", "D", "E", "F"]


@dataclass(frozen=True)
class AtomRecord:
    """Minimal PDB atom record for layer assignment validation."""

    serial: int
    atom_name: str
    residue_name: str
    chain_id: str
    residue_number: int
    element: str
    x: float
    y: float
    z: float

    @property
    def residue_key(self) -> tuple[str, int, str]:
        return (self.chain_id, self.residue_number, self.residue_name)


def infer_element(atom_name: str, line: str = "") -> str:
    """Infer element from PDB element columns or atom name."""
    element = line[76:78].strip() if len(line) >= 78 else ""
    if element:
        return element.upper()
    match = re.search(r"[A-Za-z]+", atom_name.strip())
    if not match:
        return ""
    text = match.group(0).upper()
    if text.startswith(("CL", "BR")):
        return text[:2].title()
    return text[0]


def parse_atom_record(line: str) -> AtomRecord:
    """Parse one ATOM/HETATM line."""
    return AtomRecord(
        serial=int(line[6:11]),
        atom_name=line[12:16].strip(),
        residue_name=line[17:20].strip(),
        chain_id=line[21:22].strip(),
        residue_number=int(line[22:26]),
        element=infer_element(line[12:16].strip(), line),
        x=float(line[30:38]),
        y=float(line[38:46]),
        z=float(line[46:54]),
    )


def parse_parent_pdb(path: Path) -> list[AtomRecord]:
    """Parse ATOM/HETATM records from parent PDB."""
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            records.append(parse_atom_record(line))
    if not records:
        raise ValueError(f"No ATOM/HETATM records found in {path}.")
    return records


def layer_centers_from_atoms(atoms: list[AtomRecord]) -> list[float]:
    """Infer layer centers using the same C-alpha z-gap logic as parameterized rise."""
    ca_z = [atom.z for atom in atoms if atom.atom_name == "CA"]
    if not ca_z:
        raise ValueError("Cannot validate layer assignments without C-alpha atoms.")
    return infer_layers_from_ca_z(ca_z).layer_centers


def expected_chains_for(atoms: list[AtomRecord]) -> list[str]:
    """Return expected chain set, preferring A-F when present."""
    chains = sorted({atom.chain_id for atom in atoms})
    return EXPECTED_CHAINS if set(EXPECTED_CHAINS).issubset(chains) else chains


def balance_score(chains_present: set[str], expected_chains: list[str]) -> float:
    """Return simple chain balance score."""
    if not expected_chains:
        return 0.0
    return len(set(chains_present) & set(expected_chains)) / len(expected_chains)


def top_counter_text(values: list[str], limit: int = 6) -> str:
    """Return compact top-count summary."""
    counts = Counter(values)
    return ";".join(f"{value}:{count}" for value, count in counts.most_common(limit))


def build_atom_assignments(atoms: list[AtomRecord], layer_centers: list[float], assignment_method: str) -> pd.DataFrame:
    """Build atom-level assignment table."""
    base_rows = []
    for atom in atoms:
        layer_index = assign_layer_index(atom.z, layer_centers)
        base_rows.append({"atom": atom, "layer_index": layer_index})
    layer_stats = {}
    for layer_index in sorted({row["layer_index"] for row in base_rows}):
        z_values = [row["atom"].z for row in base_rows if row["layer_index"] == layer_index]
        layer_stats[layer_index] = (float(np.mean(z_values)), float(np.median(z_values)))
    rows = []
    for row in base_rows:
        atom = row["atom"]
        layer_index = int(row["layer_index"])
        z_mean, z_median = layer_stats[layer_index]
        rows.append(
            {
                "layer_index": layer_index,
                "atom_serial": atom.serial,
                "atom_name": atom.atom_name,
                "residue_name": atom.residue_name,
                "chain_id": atom.chain_id,
                "residue_number": atom.residue_number,
                "element": atom.element,
                "x": atom.x,
                "y": atom.y,
                "z": atom.z,
                "z_offset_from_layer_mean_A": atom.z - z_mean,
                "z_offset_from_layer_median_A": atom.z - z_median,
                "assignment_method": assignment_method,
            }
        )
    return pd.DataFrame(rows)


def residues_summary(group: pd.DataFrame, limit: int = 8) -> str:
    """Return compact residue summary for one layer."""
    residues = group[["chain_id", "residue_number", "residue_name"]].drop_duplicates()
    labels = [f"{r.chain_id}{int(r.residue_number)}:{r.residue_name}" for r in residues.itertuples(index=False)]
    suffix = "" if len(labels) <= limit else f";...(+{len(labels) - limit})"
    return ";".join(labels[:limit]) + suffix


def detect_outlier_layers(composition: pd.DataFrame) -> set[int]:
    """Detect outlier layers by z thickness using an IQR fence."""
    if composition.empty:
        return set()
    thickness = pd.to_numeric(composition["z_thickness_A"], errors="coerce")
    q1 = thickness.quantile(0.25)
    q3 = thickness.quantile(0.75)
    iqr = q3 - q1
    threshold = q3 + 1.5 * iqr
    return set(composition.loc[thickness > threshold, "layer_index"].astype(int))


def build_layer_composition(assignments: pd.DataFrame, expected_chains: list[str]) -> pd.DataFrame:
    """Build layer-level composition table."""
    rows = []
    layer_means = assignments.groupby("layer_index")["z"].mean().sort_index()
    for layer_index, group in assignments.groupby("layer_index", sort=True):
        chains = sorted(group["chain_id"].dropna().unique())
        residues = group[["chain_id", "residue_number", "residue_name"]].drop_duplicates()
        ca = group[group["atom_name"] == "CA"]
        z_values = pd.to_numeric(group["z"], errors="coerce")
        ca_z = pd.to_numeric(ca["z"], errors="coerce")
        next_delta = layer_means.get(layer_index + 1, np.nan) - layer_means.get(layer_index, np.nan)
        prev_delta = layer_means.get(layer_index, np.nan) - layer_means.get(layer_index - 1, np.nan)
        score = balance_score(set(chains), expected_chains)
        notes = []
        if score < 1.0:
            notes.append("missing_expected_chains")
        rows.append(
            {
                "layer_index": layer_index,
                "atom_count": len(group),
                "ca_count": len(ca),
                "chain_count": len(chains),
                "chains_present": ",".join(chains),
                "residue_count": len(residues),
                "residues_present_summary": residues_summary(group),
                "residue_names_present": ",".join(sorted(group["residue_name"].dropna().unique())),
                "atom_names_top": top_counter_text(group["atom_name"].astype(str).tolist()),
                "elements_top": top_counter_text(group["element"].astype(str).tolist()),
                "z_mean_A": float(z_values.mean()),
                "z_median_A": float(z_values.median()),
                "z_min_A": float(z_values.min()),
                "z_max_A": float(z_values.max()),
                "z_thickness_A": float(z_values.max() - z_values.min()),
                "next_layer_delta_z_mean_A": next_delta,
                "prev_layer_delta_z_mean_A": prev_delta,
                "ca_z_mean_A": float(ca_z.mean()) if len(ca_z) else np.nan,
                "ca_z_min_A": float(ca_z.min()) if len(ca_z) else np.nan,
                "ca_z_max_A": float(ca_z.max()) if len(ca_z) else np.nan,
                "ca_z_thickness_A": float(ca_z.max() - ca_z.min()) if len(ca_z) else np.nan,
                "balance_score": score,
                "notes": ";".join(notes),
            }
        )
    composition = pd.DataFrame(rows)
    outliers = detect_outlier_layers(composition)
    if outliers:
        composition["notes"] = composition.apply(
            lambda row: ";".join(filter(None, [str(row["notes"]), "thickness_outlier" if int(row["layer_index"]) in outliers else ""])),
            axis=1,
        )
    return composition


def build_residue_participation(assignments: pd.DataFrame) -> pd.DataFrame:
    """Build residue/layer participation table."""
    rows = []
    for (chain_id, residue_number, residue_name), group in assignments.groupby(["chain_id", "residue_number", "residue_name"], sort=True):
        counts = group["layer_index"].value_counts().sort_values(ascending=False)
        primary_layer = int(counts.index[0])
        layers = sorted(group["layer_index"].astype(int).unique())
        z_values = pd.to_numeric(group["z"], errors="coerce")
        notes = "split_across_layers" if len(layers) > 1 else ""
        rows.append(
            {
                "chain_id": chain_id,
                "residue_number": residue_number,
                "residue_name": residue_name,
                "atom_count": len(group),
                "layer_count": len(layers),
                "layers_present": ",".join(str(layer) for layer in layers),
                "primary_layer": primary_layer,
                "primary_layer_atom_fraction": counts.iloc[0] / len(group),
                "z_min_A": float(z_values.min()),
                "z_max_A": float(z_values.max()),
                "z_span_A": float(z_values.max() - z_values.min()),
                "notes": notes,
            }
        )
    return pd.DataFrame(rows)


def build_chain_coverage(assignments: pd.DataFrame) -> pd.DataFrame:
    """Build chain/layer coverage table."""
    rows = []
    for chain_id, group in assignments.groupby("chain_id", sort=True):
        layers = sorted(group["layer_index"].astype(int).unique())
        first_layer = min(layers)
        last_layer = max(layers)
        expected = set(range(first_layer, last_layer + 1))
        missing = sorted(expected - set(layers))
        residues = group[["residue_number", "residue_name"]].drop_duplicates()
        rows.append(
            {
                "chain_id": chain_id,
                "layer_count_present": len(layers),
                "first_layer": first_layer,
                "last_layer": last_layer,
                "layers_present": ",".join(str(layer) for layer in layers),
                "missing_layer_count_within_span": len(missing),
                "missing_layers_within_span": ",".join(str(layer) for layer in missing),
                "atom_count": len(group),
                "ca_count": int((group["atom_name"] == "CA").sum()),
                "residue_count": len(residues),
                "notes": "continuous_layer_span" if not missing else "missing_layers_within_span",
            }
        )
    return pd.DataFrame(rows)


def structural_interpretation(composition: pd.DataFrame, residue_participation: pd.DataFrame) -> str:
    """Return conservative structural interpretation."""
    split_fraction = (pd.to_numeric(residue_participation["layer_count"], errors="coerce") > 1).mean()
    balanced_fraction = (pd.to_numeric(composition["balance_score"], errors="coerce") >= 1.0).mean()
    thickness_median = pd.to_numeric(composition["z_thickness_A"], errors="coerce").median()
    if split_fraction > 0.5:
        return "layers are useful computational slices but should not be interpreted as unique chemical hexad levels"
    if balanced_fraction > 0.8 and thickness_median < 2.0:
        return "layers appear reasonably regular and useful as a diagnostic rise parameter"
    return "layer assignment is too ambiguous for physical interpretation"


def save_plots(composition: pd.DataFrame, figure_dir: Path) -> list[Path]:
    """Save optional simple layer validation plots."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    specs = [
        ("parent_axial_layer_z_means", "z_mean_A", "mean z (A)"),
        ("parent_axial_layer_thickness", "z_thickness_A", "z thickness (A)"),
        ("parent_axial_layer_chain_balance", "balance_score", "chain balance score"),
    ]
    for stem, column, ylabel in specs:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(composition["layer_index"], composition[column], marker="o", lw=1)
        ax.set_xlabel("layer index")
        ax.set_ylabel(ylabel)
        ax.set_title(stem.replace("_", " "))
        fig.tight_layout()
        for suffix in (".png", ".svg"):
            path = figure_dir / f"{stem}{suffix}"
            fig.savefig(path, dpi=180)
            outputs.append(path)
        plt.close(fig)
    return outputs


def build_report_text(
    source_pdb: Path,
    atoms: list[AtomRecord],
    composition: pd.DataFrame,
    residue_participation: pd.DataFrame,
    chain_coverage: pd.DataFrame,
    interpretation: str,
    plots: list[Path],
) -> str:
    """Build layer assignment validation report."""
    ca_count = sum(atom.atom_name == "CA" for atom in atoms)
    chains = sorted({atom.chain_id for atom in atoms})
    residues = {atom.residue_key for atom in atoms}
    z_values = [atom.z for atom in atoms]
    split = residue_participation[residue_participation["layer_count"] > 1]
    balanced = composition[composition["balance_score"] >= 1.0]
    layer_deltas = pd.to_numeric(composition["next_layer_delta_z_mean_A"], errors="coerce").dropna()
    thickness = pd.to_numeric(composition["z_thickness_A"], errors="coerce")
    plot_text = "\n".join(f"- `{path}`" for path in plots) if plots else "_Plots were not generated._"
    split_examples = markdown_table(split.head(10), ["chain_id", "residue_number", "residue_name", "layer_count", "layers_present"])
    return f"""# Parent Axial Layer Assignment Validation

## 1. Purpose

Validate whether the 45 inferred layers used in parameterized-rise modeling are structurally meaningful.

## 2. Input Structure

- Parent PDB: `{source_pdb}`
- Atom count: {len(atoms)}
- C-alpha count: {ca_count}
- Chains present: {', '.join(chains)}
- Residue count: {len(residues)}
- z span: {min(z_values):.4f} to {max(z_values):.4f} A ({max(z_values) - min(z_values):.4f} A)

## 3. Layer Assignment Method

The assignment reproduces the deterministic C-alpha z-gap layer inference used by `audit_parent_axial_layers.py` and the parameterized-rise generator. C-alpha atoms are sorted by z, gaps larger than the audit threshold define layer centers, and every atom is assigned to the nearest inferred layer center. Fallback was not used.

## 4. Layer Composition Summary

- Number of layers: {len(composition)}
- Mean/median atom count per layer: {composition['atom_count'].mean():.2f} / {composition['atom_count'].median():.2f}
- Mean/median C-alpha count per layer: {composition['ca_count'].mean():.2f} / {composition['ca_count'].median():.2f}
- Mean/median z thickness: {thickness.mean():.4f} / {thickness.median():.4f} A
- Mean/median layer-to-layer rise: {layer_deltas.mean():.4f} / {layer_deltas.median():.4f} A
- Layer-to-layer rise range: {layer_deltas.min():.4f} to {layer_deltas.max():.4f} A
- Layers with all expected chains: {len(balanced)}
- Layers missing one or more expected chains: {len(composition) - len(balanced)}

## 5. Residue Splitting Summary

- Number of residues: {len(residue_participation)}
- Residues contained in one layer: {int((residue_participation['layer_count'] == 1).sum())}
- Residues split across multiple layers: {len(split)}

Examples of split residues:

{split_examples}

This helps distinguish residue/hexad levels from atom-slice behavior.

## 6. Chain Coverage Summary

{markdown_table(chain_coverage, ['chain_id', 'layer_count_present', 'first_layer', 'last_layer', 'missing_layer_count_within_span', 'atom_count', 'ca_count', 'residue_count'])}

## 7. Structural Interpretation

{interpretation}.

## 8. Implications For parameterized_rise_0p9750

The 0.975 rise-scale result should be interpreted as effective computational z-layer compression unless the inferred layers are later mapped to known hexad/repeat register. The model is still not minimized, and the inferred 45 layers should not be interpreted as unique chemical hexad levels without further validation.

## 9. Recommended Next Steps

- Visually inspect representative layers if needed.
- Compare inferred layers to known hexad/repeat register if available.
- Consider a chemically defined layer model using residue/register information rather than purely z-derived layers.
- Only then use 0.975 as a physical design target.

## Optional Plots

{plot_text}
"""


def run(
    source_pdb: Path,
    atom_csv: Path,
    composition_csv: Path,
    residue_csv: Path,
    chain_csv: Path,
    report_path: Path,
    figure_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str, list[Path]]:
    """Run parent axial layer assignment validation and write outputs."""
    atoms = parse_parent_pdb(source_pdb)
    centers = layer_centers_from_atoms(atoms)
    assignments = build_atom_assignments(atoms, centers, "ca_z_gap_nearest_layer_center")
    expected = expected_chains_for(atoms)
    composition = build_layer_composition(assignments, expected)
    residue_participation = build_residue_participation(assignments)
    chain_coverage = build_chain_coverage(assignments)
    interpretation = structural_interpretation(composition, residue_participation)
    plots = save_plots(composition, figure_dir)
    for path, df in [
        (atom_csv, assignments),
        (composition_csv, composition),
        (residue_csv, residue_participation),
        (chain_csv, chain_coverage),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        build_report_text(source_pdb, atoms, composition, residue_participation, chain_coverage, interpretation, plots),
        encoding="utf-8",
    )
    return assignments, composition, residue_participation, chain_coverage, interpretation, plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pdb", type=Path, default=DEFAULT_SOURCE_PDB)
    parser.add_argument("--atom-csv", type=Path, default=DEFAULT_ATOM_CSV)
    parser.add_argument("--composition-csv", type=Path, default=DEFAULT_COMPOSITION_CSV)
    parser.add_argument("--residue-csv", type=Path, default=DEFAULT_RESIDUE_CSV)
    parser.add_argument("--chain-csv", type=Path, default=DEFAULT_CHAIN_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    assignments, composition, residue_participation, chain_coverage, interpretation, plots = run(
        args.source_pdb,
        args.atom_csv,
        args.composition_csv,
        args.residue_csv,
        args.chain_csv,
        args.report,
        args.figure_dir,
    )
    print(f"Assigned {len(assignments)} atoms across {len(composition)} layers")
    print(f"Split residues: {int((residue_participation['layer_count'] > 1).sum())}")
    print(f"Interpretation: {interpretation}")
    print(f"Report: {args.report}")
    print(f"Plots: {len(plots)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
