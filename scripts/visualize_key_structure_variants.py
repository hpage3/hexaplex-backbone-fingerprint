"""Create static visual summaries for key diagnostic structure variants."""

from __future__ import annotations

import argparse
import math
import sys
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

from scripts.generate_global_deformation_variants import PdbAtomLine, parse_pdb_atom_lines


DEFAULT_FIGURE_DIR = Path("outputs/figures")
DEFAULT_METRICS_DIR = Path("outputs/metrics")
DEFAULT_REPORT_DIR = Path("outputs/reports")

SUMMARY_CSV = DEFAULT_METRICS_DIR / "key_structure_variant_visual_summary.csv"
REPORT_PATH = DEFAULT_REPORT_DIR / "key_structure_variant_visualization.md"


@dataclass(frozen=True)
class StructureSpec:
    """Structure path plus labels used in visual outputs."""

    structure_id: str
    structure_label: str
    source_path: Path
    notes: str


def key_structure_specs(root: Path = ROOT) -> list[StructureSpec]:
    """Return the four key diagnostic structure specifications."""
    return [
        StructureSpec(
            "parent_baseline",
            "Parent baseline",
            root / "outputs/coordinates/ideal_hexaflex_variants/ideal_hexaflex_backbone_plus_carboxylate.pdb",
            "Best clean baseline: backbone plus carboxylate.",
        ),
        StructureSpec(
            "rise_like_0p9700",
            "Generic rise-like 0.9700",
            root / "outputs/coordinates/rise_like_variants/rise_like_0p9700.pdb",
            "Generic rise-like diagnostic compression.",
        ),
        StructureSpec(
            "parameterized_rise_0p9750",
            "Parameterized rise 0.9750",
            root / "outputs/coordinates/parameterized_rise_variants/parameterized_rise_0p9750.pdb",
            "Best current effective computational z-layer compression diagnostic.",
        ),
        StructureSpec(
            "parameterized_rise_0p9600",
            "Parameterized rise 0.9600",
            root / "outputs/coordinates/parameterized_rise_variants/parameterized_rise_0p9600.pdb",
            "Over-compressed parameterized-rise diagnostic example.",
        ),
    ]


def locate_key_structures(specs: list[StructureSpec] | None = None) -> list[StructureSpec]:
    """Validate that all key structure paths are available."""
    specs = key_structure_specs() if specs is None else specs
    missing = [str(spec.source_path) for spec in specs if not spec.source_path.exists()]
    if missing:
        raise FileNotFoundError("Missing key structure file(s): " + "; ".join(missing))
    return specs


def parse_structure(path: Path) -> list[PdbAtomLine]:
    """Parse PDB ATOM/HETATM records from a structure file."""
    _, atoms = parse_pdb_atom_lines(path)
    return atoms


def ca_atoms(atoms: list[PdbAtomLine]) -> list[PdbAtomLine]:
    """Return C-alpha atoms in file order."""
    return [atom for atom in atoms if atom.is_ca]


def coordinate_array(atoms: list[PdbAtomLine]) -> np.ndarray:
    """Return coordinates as an ``n x 3`` array."""
    return np.array([atom.coord for atom in atoms], dtype=float)


def z_bounds_and_span(atoms: list[PdbAtomLine]) -> tuple[float, float, float]:
    """Return min z, max z, and z span."""
    coords = coordinate_array(atoms)
    z_min = float(coords[:, 2].min())
    z_max = float(coords[:, 2].max())
    return z_min, z_max, z_max - z_min


def mean_ca_radius(atoms: list[PdbAtomLine]) -> float:
    """Return mean C-alpha xy radius around the C-alpha xy centroid."""
    cas = ca_atoms(atoms)
    chosen = cas if cas else atoms
    coords = coordinate_array(chosen)
    center_xy = coords[:, :2].mean(axis=0)
    return float(np.mean(np.linalg.norm(coords[:, :2] - center_xy, axis=1)))


def atom_identity(atom: PdbAtomLine) -> tuple[str, str, str, str]:
    """Return an identity key stable across derived variants."""
    return (atom.chain, atom.resseq, atom.resname, atom.atom_name)


def matching_coordinate_arrays(
    parent_atoms: list[PdbAtomLine], variant_atoms: list[PdbAtomLine], ca_only: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Return matching parent/variant coordinate arrays by atom identity."""
    parent_chosen = ca_atoms(parent_atoms) if ca_only else parent_atoms
    variant_chosen = ca_atoms(variant_atoms) if ca_only else variant_atoms
    parent_keys = [atom_identity(atom) for atom in parent_chosen]
    variant_map = {atom_identity(atom): atom for atom in variant_chosen}
    if any(key not in variant_map for key in parent_keys):
        raise ValueError("Parent and variant atom identities do not match.")
    parent_coords = np.array([atom.coord for atom in parent_chosen], dtype=float)
    variant_coords = np.array([variant_map[key].coord for key in parent_keys], dtype=float)
    return parent_coords, variant_coords


def rmsd_to_parent(parent_atoms: list[PdbAtomLine], variant_atoms: list[PdbAtomLine]) -> float:
    """Return all-atom RMSD to parent without alignment."""
    parent_coords, variant_coords = matching_coordinate_arrays(parent_atoms, variant_atoms)
    shifts = variant_coords - parent_coords
    return float(np.sqrt(np.mean(np.sum(shifts * shifts, axis=1))))


def ca_displacement_summary(parent_atoms: list[PdbAtomLine], variant_atoms: list[PdbAtomLine]) -> dict[str, float]:
    """Return C-alpha displacement summary relative to parent."""
    parent_coords, variant_coords = matching_coordinate_arrays(parent_atoms, variant_atoms, ca_only=True)
    distances = np.linalg.norm(variant_coords - parent_coords, axis=1)
    return {
        "max_ca_displacement_A": float(np.max(distances)) if len(distances) else float("nan"),
        "mean_ca_displacement_A": float(np.mean(distances)) if len(distances) else float("nan"),
        "median_ca_displacement_A": float(np.median(distances)) if len(distances) else float("nan"),
    }


def mean_interlayer_rise(cas: list[PdbAtomLine]) -> float:
    """Return a conservative mean absolute C-alpha z spacing by chain order."""
    if not cas:
        return float("nan")
    rises: list[float] = []
    for chain in sorted({atom.chain for atom in cas}):
        chain_cas = [atom for atom in cas if atom.chain == chain]
        chain_cas = sorted(chain_cas, key=lambda atom: atom.index)
        z_values = np.array([atom.z for atom in chain_cas], dtype=float)
        if len(z_values) > 1:
            rises.extend(np.abs(np.diff(z_values)).tolist())
    return float(np.mean(rises)) if rises else float("nan")


def structure_summary_row(
    spec: StructureSpec, atoms: list[PdbAtomLine], parent_atoms: list[PdbAtomLine]
) -> dict[str, object]:
    """Build one metrics CSV row for a structure."""
    z_min, z_max, z_span = z_bounds_and_span(atoms)
    cas = ca_atoms(atoms)
    if spec.structure_id == "parent_baseline":
        disp = {
            "max_ca_displacement_A": 0.0,
            "mean_ca_displacement_A": 0.0,
            "median_ca_displacement_A": 0.0,
        }
        rmsd = 0.0
    else:
        disp = ca_displacement_summary(parent_atoms, atoms)
        rmsd = rmsd_to_parent(parent_atoms, atoms)
    return {
        "structure_id": spec.structure_id,
        "source_path": str(spec.source_path),
        "structure_label": spec.structure_label,
        "z_min_A": z_min,
        "z_max_A": z_max,
        "z_span_A": z_span,
        "mean_ca_radius_A": mean_ca_radius(atoms),
        "ca_count": len(cas),
        "atom_count": len(atoms),
        "rmsd_to_parent_A": rmsd,
        **disp,
        "mean_interlayer_rise_A": mean_interlayer_rise(cas),
        "notes": spec.notes,
    }


def chain_color_map(atoms_by_id: dict[str, list[PdbAtomLine]]) -> dict[str, tuple[float, float, float, float]]:
    """Return stable colors by chain."""
    chains = sorted({atom.chain for atoms in atoms_by_id.values() for atom in atoms})
    cmap = plt.get_cmap("tab10")
    return {chain: cmap(i % 10) for i, chain in enumerate(chains)}


def centered_ca_by_chain(atoms: list[PdbAtomLine], center: np.ndarray) -> dict[str, np.ndarray]:
    """Return centered C-alpha coordinate arrays split by chain."""
    result: dict[str, np.ndarray] = {}
    for chain in sorted({atom.chain for atom in ca_atoms(atoms)}):
        chain_cas = [atom for atom in ca_atoms(atoms) if atom.chain == chain]
        chain_cas = sorted(chain_cas, key=lambda atom: atom.index)
        result[chain] = coordinate_array(chain_cas) - center
    return result


def axis_limits(centered: list[np.ndarray], dims: tuple[int, int], pad_fraction: float = 0.08) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return common padded axis limits for selected coordinate dimensions."""
    coords = np.vstack(centered)
    xs = coords[:, dims[0]]
    ys = coords[:, dims[1]]
    x_span = xs.max() - xs.min()
    y_span = ys.max() - ys.min()
    span = max(x_span, y_span)
    pad = span * pad_fraction if span > 0 else 1.0
    x_mid = (xs.max() + xs.min()) / 2
    y_mid = (ys.max() + ys.min()) / 2
    half = span / 2 + pad
    return (float(x_mid - half), float(x_mid + half)), (float(y_mid - half), float(y_mid + half))


def save_figure(fig: plt.Figure, png_path: Path, svg_path: Path) -> None:
    """Save a figure as PNG and SVG."""
    png_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


def plot_overview(specs: list[StructureSpec], atoms_by_id: dict[str, list[PdbAtomLine]], outdir: Path) -> None:
    """Create side/top overview panel for all key structures."""
    parent_center = coordinate_array(ca_atoms(atoms_by_id["parent_baseline"])).mean(axis=0)
    centered_all = [coordinate_array(ca_atoms(atoms)) - parent_center for atoms in atoms_by_id.values()]
    xz_limits = axis_limits(centered_all, (0, 2))
    xy_limits = axis_limits(centered_all, (0, 1))
    colors = chain_color_map(atoms_by_id)
    fig, axes = plt.subplots(len(specs), 2, figsize=(10.5, 14), constrained_layout=True)
    for row, spec in enumerate(specs):
        chains = centered_ca_by_chain(atoms_by_id[spec.structure_id], parent_center)
        for chain, coords in chains.items():
            axes[row, 0].plot(coords[:, 0], coords[:, 2], marker="o", ms=2.5, lw=0.8, color=colors[chain], label=chain)
            axes[row, 1].plot(coords[:, 0], coords[:, 1], marker="o", ms=2.5, lw=0.8, color=colors[chain], label=chain)
        axes[row, 0].set_title(f"{spec.structure_label}: side view")
        axes[row, 1].set_title(f"{spec.structure_label}: top view")
        axes[row, 0].set_xlabel("x centered on parent CA (A)")
        axes[row, 0].set_ylabel("z centered on parent CA (A)")
        axes[row, 1].set_xlabel("x centered on parent CA (A)")
        axes[row, 1].set_ylabel("y centered on parent CA (A)")
        axes[row, 0].set_xlim(*xz_limits[0])
        axes[row, 0].set_ylim(*xz_limits[1])
        axes[row, 1].set_xlim(*xy_limits[0])
        axes[row, 1].set_ylim(*xy_limits[1])
        axes[row, 0].set_aspect("equal", adjustable="box")
        axes[row, 1].set_aspect("equal", adjustable="box")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, title="Chain", loc="upper center", ncol=max(1, len(labels)))
    save_figure(fig, outdir / "key_structure_variant_overview.png", outdir / "key_structure_variant_overview.svg")


def plot_displacements(atoms_by_id: dict[str, list[PdbAtomLine]], outdir: Path) -> None:
    """Create parent/variant overlay plus displacement segments."""
    parent = atoms_by_id["parent_baseline"]
    parent_center = coordinate_array(ca_atoms(parent)).mean(axis=0)
    variants = [
        ("parameterized_rise_0p9750", "Parameterized rise 0.9750", "#1f77b4"),
        ("parameterized_rise_0p9600", "Parameterized rise 0.9600", "#d62728"),
    ]
    centered_all = [coordinate_array(ca_atoms(atoms)) - parent_center for atoms in atoms_by_id.values()]
    xz_limits = axis_limits(centered_all, (0, 2))
    xy_limits = axis_limits(centered_all, (0, 1))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)
    parent_ca = coordinate_array(ca_atoms(parent)) - parent_center
    for row, (variant_id, label, color) in enumerate(variants):
        _, variant_ca_raw = matching_coordinate_arrays(parent, atoms_by_id[variant_id], ca_only=True)
        variant_ca = variant_ca_raw - parent_center
        for col, dims, limits, view_label in [
            (0, (0, 2), xz_limits, "side view"),
            (1, (0, 1), xy_limits, "top view"),
        ]:
            ax = axes[row, col]
            ax.scatter(parent_ca[:, dims[0]], parent_ca[:, dims[1]], s=8, color="0.78", label="Parent")
            ax.scatter(variant_ca[:, dims[0]], variant_ca[:, dims[1]], s=8, color=color, label=label)
            stride = max(1, len(parent_ca) // 90)
            for start, end in zip(parent_ca[::stride], variant_ca[::stride]):
                ax.plot([start[dims[0]], end[dims[0]]], [start[dims[1]], end[dims[1]]], color=color, alpha=0.32, lw=0.6)
            ax.set_title(f"{label} displacement overlay: {view_label}")
            ax.set_xlim(*limits[0])
            ax.set_ylim(*limits[1])
            ax.set_aspect("equal", adjustable="box")
            ax.legend(loc="best", fontsize=8)
    save_figure(fig, outdir / "key_structure_variant_displacements.png", outdir / "key_structure_variant_displacements.svg")


def ca_z_by_chain_order(atoms: list[PdbAtomLine]) -> pd.DataFrame:
    """Return C-alpha z coordinates by chain and per-chain coordinate order."""
    rows = []
    for chain in sorted({atom.chain for atom in ca_atoms(atoms)}):
        chain_cas = sorted(
            [atom for atom in ca_atoms(atoms) if atom.chain == chain],
            key=lambda atom: atom.index,
        )
        for order, atom in enumerate(chain_cas, start=1):
            rows.append({"chain": chain, "order": order, "z": atom.z})
    return pd.DataFrame(rows)


def parent_relative_ca_displacement(
    parent_atoms: list[PdbAtomLine],
    variant_atoms: list[PdbAtomLine],
) -> pd.DataFrame:
    """Return C-alpha delta-z versus parent-centered parent z."""
    parent = ca_z_by_chain_order(parent_atoms).rename(columns={"z": "parent_z"})
    variant = ca_z_by_chain_order(variant_atoms).rename(columns={"z": "variant_z"})
    merged = parent.merge(variant, on=["chain", "order"], how="inner")
    parent_center = float(parent["parent_z"].mean())
    merged["parent_z_centered_A"] = merged["parent_z"] - parent_center
    merged["delta_z_A"] = merged["variant_z"] - merged["parent_z"]
    return merged.sort_values("parent_z_centered_A")


def plot_axial_profiles(specs: list[StructureSpec], atoms_by_id: dict[str, list[PdbAtomLine]], outdir: Path) -> None:
    """Plot parent-relative C-alpha axial displacement against parent z position."""
    parent_id = "parent_baseline"
    if parent_id not in atoms_by_id:
        raise ValueError("parent_baseline is required for parent-relative axial displacement plotting")

    fig, ax = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
    parent_atoms = atoms_by_id[parent_id]

    for spec in specs:
        if spec.structure_id == parent_id:
            continue

        displacement = parent_relative_ca_displacement(parent_atoms, atoms_by_id[spec.structure_id])

        ax.scatter(
            displacement["parent_z_centered_A"],
            displacement["delta_z_A"],
            s=12,
            alpha=0.8,
            label=spec.structure_label,
        )

        # Add a simple least-squares trend line to make compression direction visible.
        x = displacement["parent_z_centered_A"].to_numpy()
        y = displacement["delta_z_A"].to_numpy()
        if len(x) >= 2:
            slope, intercept = np.polyfit(x, y, 1)
            x_line = np.array([x.min(), x.max()])
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, linewidth=1.4, alpha=0.9)

    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.axvline(0.0, color="black", linestyle=":", linewidth=0.8)
    ax.set_title("Axial C-alpha displacement relative to parent")
    ax.set_xlabel("Parent-centered C-alpha z position (A)")
    ax.set_ylabel("Variant delta z vs parent (A)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    save_figure(fig, outdir / "key_structure_variant_axial_profiles.png", outdir / "key_structure_variant_axial_profiles.svg")

def plot_geometry_summary(summary: pd.DataFrame, outdir: Path) -> None:
    """Plot compact geometry metrics across structures."""
    labels = summary["structure_id"].tolist()
    metrics = [
        ("z_span_A", "z span (A)"),
        ("mean_ca_radius_A", "mean CA radius (A)"),
        ("rmsd_to_parent_A", "RMSD to parent (A)"),
        ("max_ca_displacement_A", "max CA displacement (A)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    for ax, (column, ylabel) in zip(axes.ravel(), metrics):
        ax.bar(labels, summary[column], color=["0.55", "#2ca02c", "#1f77b4", "#d62728"])
        ax.set_ylabel(ylabel)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
    save_figure(fig, outdir / "key_structure_variant_geometry_summary.png", outdir / "key_structure_variant_geometry_summary.svg")


def plot_chain_panels(atoms_by_id: dict[str, list[PdbAtomLine]], outdir: Path) -> None:
    """Optional parent versus best parameterized-rise chain-colored panel."""
    parent_center = coordinate_array(ca_atoms(atoms_by_id["parent_baseline"])).mean(axis=0)
    selected = [
        ("parent_baseline", "Parent baseline"),
        ("parameterized_rise_0p9750", "Parameterized rise 0.9750"),
    ]
    colors = chain_color_map(atoms_by_id)
    centered_all = [coordinate_array(ca_atoms(atoms_by_id[key])) - parent_center for key, _ in selected]
    limits = axis_limits(centered_all, (0, 2))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), constrained_layout=True)
    for ax, (structure_id, title) in zip(axes, selected):
        for chain, coords in centered_ca_by_chain(atoms_by_id[structure_id], parent_center).items():
            ax.plot(coords[:, 0], coords[:, 2], marker="o", ms=3, lw=0.9, color=colors[chain], label=chain)
        ax.set_title(title)
        ax.set_xlabel("x centered on parent CA (A)")
        ax.set_ylabel("z centered on parent CA (A)")
        ax.set_xlim(*limits[0])
        ax.set_ylim(*limits[1])
        ax.set_aspect("equal", adjustable="box")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Chain", loc="upper center", ncol=max(1, len(labels)))
    save_figure(fig, outdir / "key_structure_variant_chain_panels.png", outdir / "key_structure_variant_chain_panels.svg")


def markdown_table_from_df(df: pd.DataFrame, columns: list[str]) -> str:
    """Return a simple markdown table without requiring pandas/tabulate."""

    def fmt(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            if math.isnan(value):
                return ""
            return f"{value:.4f}"
        return str(value)

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]

    for row in df[columns].to_dict("records"):
        lines.append("| " + " | ".join(fmt(row.get(column, "")) for column in columns) + " |")

    return "\n".join(lines)

def build_report_text(summary: pd.DataFrame, specs: list[StructureSpec]) -> str:
    """Build markdown report for the static visualizations."""
    metric_cols = [
        "structure_id",
        "z_span_A",
        "mean_ca_radius_A",
        "rmsd_to_parent_A",
        "max_ca_displacement_A",
        "mean_interlayer_rise_A",
    ]
    table = markdown_table_from_df(summary, metric_cols)
    structures = "\n".join(f"- {spec.structure_label}: `{spec.source_path}`" for spec in specs)
    return f"""# Key Structure Variant Visualization

## Purpose

These static figures compare four diagnostic transformed structures that bracket the current best C/D forward-model result. They are intended to make the axial/rise-like character of the transformations inspectable without requiring PyMOL, VMD, Chimera, or an interactive viewer.

## Structures shown

{structures}

## What figures show

- `key_structure_variant_overview.png/svg`: side and top C-alpha views for all four structures with common centering and axis scaling.
- `key_structure_variant_displacements.png/svg`: parent overlays with the best parameterized-rise diagnostic and an over-compressed example.
- `key_structure_variant_axial_profiles.png/svg`: mean C-alpha z coordinate versus per-chain residue-order index.
- `key_structure_variant_geometry_summary.png/svg`: z span, mean C-alpha radius, RMSD to parent, and maximum C-alpha displacement.
- `key_structure_variant_chain_panels.png/svg`: optional chain-colored parent versus parameterized_rise_0p9750 side-view panel.

## Main visual takeaways

The parameterized_rise_0p9750 structure is best interpreted as an effective computational z-layer compression that moves the C/D diagnostics close to target while retaining the same broad six-chain organization. The comparison against parameterized_rise_0p9600 helps show what stronger compression looks like. The generic rise-like and parameterized-rise examples primarily alter axial spacing rather than gross radial organization.

Summary metrics:

{table}

## Cautions

- Do not treat these figures as proof of a physically minimized model.
- Do not interpret the 45 z-slices as validated physical hexad layers.
- These visuals are intended to interpret computational diagnostic transformations, not minimized physical structures.
- The current best parameterized_rise_0p9750 result should be described cautiously as effective computational z-layer compression pending mapping to a chemically and register-defined model.
"""


def run_visualization(
    figure_dir: Path = DEFAULT_FIGURE_DIR,
    metrics_csv: Path = SUMMARY_CSV,
    report_path: Path = REPORT_PATH,
) -> pd.DataFrame:
    """Run all key structure visual summaries and return the metrics table."""
    specs = locate_key_structures()
    atoms_by_id = {spec.structure_id: parse_structure(spec.source_path) for spec in specs}
    parent_atoms = atoms_by_id["parent_baseline"]
    summary = pd.DataFrame([structure_summary_row(spec, atoms_by_id[spec.structure_id], parent_atoms) for spec in specs])

    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(metrics_csv, index=False)
    report_path.write_text(build_report_text(summary, specs), encoding="utf-8")

    plot_overview(specs, atoms_by_id, figure_dir)
    plot_displacements(atoms_by_id, figure_dir)
    plot_axial_profiles(specs, atoms_by_id, figure_dir)
    plot_geometry_summary(summary, figure_dir)
    plot_chain_panels(atoms_by_id, figure_dir)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--metrics-csv", type=Path, default=SUMMARY_CSV)
    parser.add_argument("--report-path", type=Path, default=REPORT_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_visualization(args.figure_dir, args.metrics_csv, args.report_path)
    print(f"Wrote {args.metrics_csv}")
    print(f"Wrote {args.report_path}")
    print(f"Wrote figures under {args.figure_dir}")
    print(summary[["structure_id", "z_span_A", "mean_ca_radius_A", "rmsd_to_parent_A"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




