"""Generate simple six-strand parametric peptide-plane models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hexaplex_backbone_fingerprint.parametric_peptide_plane_models import (
    ModelParameters,
    generate_model_atoms,
    manifest_row,
    measured_normal_to_axis_angle,
    orientation_frame,
    repeat_center,
    write_pdb,
    write_xyz,
)


DEFAULT_OUTDIR = Path("outputs/parametric_six_strand_peptide_plane_models")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--n-strands", type=int, default=6)
    parser.add_argument("--repeats-per-strand", type=int, default=16)
    parser.add_argument("--helix-radius-A", nargs="*", type=float, default=[8.0])
    parser.add_argument("--twist-deg", nargs="*", type=float, default=[28.0, 30.0, 32.0])
    parser.add_argument("--rise-A", nargs="*", type=float, default=[3.38, 3.40])
    parser.add_argument("--plane-normal-to-axis-deg", nargs="*", type=float, default=[0.0, 30.0, 60.0, 90.0])
    parser.add_argument("--plane-azimuth-deg", nargs="*", type=float, default=[0.0, 30.0, 60.0, 90.0])
    parser.add_argument("--in-plane-spin-deg", nargs="*", type=float, default=[0.0])
    parser.add_argument("--uniform-adjacent-z-offset-A", nargs="*", type=float, default=[0.0])
    parser.add_argument("--alternating-z-offset-A", nargs="*", type=float)
    parser.add_argument("--z-offset-mode", choices=["uniform_adjacent", "alternating"], default="uniform_adjacent")
    parser.add_argument("--handedness", choices=["right", "left"], default="right")
    parser.add_argument("--write-xyz", action="store_true", default=True)
    return parser.parse_args()


def iter_parameter_sweep(args: argparse.Namespace):
    if args.z_offset_mode == "alternating":
        z_offsets = args.alternating_z_offset_A if args.alternating_z_offset_A is not None else args.uniform_adjacent_z_offset_A
    else:
        z_offsets = args.uniform_adjacent_z_offset_A
    for twist in args.twist_deg:
        for rise in args.rise_A:
            for radius in args.helix_radius_A:
                for normal_angle in args.plane_normal_to_axis_deg:
                    for azimuth in args.plane_azimuth_deg:
                        for spin in args.in_plane_spin_deg:
                            for z_offset in z_offsets:
                                yield ModelParameters(
                                    n_strands=args.n_strands,
                                    repeats_per_strand=args.repeats_per_strand,
                                    helix_radius_A=radius,
                                    twist_deg=twist,
                                    rise_A=rise,
                                    plane_normal_to_axis_deg=normal_angle,
                                    plane_azimuth_deg=azimuth,
                                    in_plane_spin_deg=spin,
                                    uniform_adjacent_z_offset_A=z_offset,
                                    alternating_z_offset_A=z_offset if args.z_offset_mode == "alternating" else None,
                                    z_offset_mode=args.z_offset_mode,
                                    handedness=args.handedness,
                                )


def write_readme(outdir: Path, manifest: pd.DataFrame) -> None:
    example_rows = manifest.head(5)
    example_lines = "\n".join(f"- `{row.model_label}`" for row in example_rows.itertuples())
    text = f"""# Parametric Six-Strand Peptide-Plane Models

These models are deliberately simple forward models for testing whether C/D powder-band behavior can arise from peptide-plane orientation relative to a six-strand helical axis.

## Minimal Motif

Each repeat is one idealized trans peptide-plane-like motif. The canonical motif lies in the local XY plane and contains:

- residue `PPI`: `CA`, `C`, `O`
- residue `PPJ`: `N`, `CA`, optional `H`

The C-N distance is peptide-like, so the existing peptide-plane parser can recover one plane per repeat. The motif is not intended to be full chemical realism; it is intended to be geometrically interpretable.

## Assembly Geometry

- The global helical axis is the z-axis.
- Six strands are equally spaced around the axis at 60 degree azimuthal offsets.
- Repeat centers lie on helices of radius `helix_radius_A`.
- Moving one repeat forward on a strand advances by `twist_deg` around the axis and `rise_A` along z.
- `uniform_adjacent_z_offset_A` staggers strand register axially: strand `s` receives `s * uniform_adjacent_z_offset_A` along z when `z_offset_mode` is `uniform_adjacent`.
- `alternating_z_offset_A` staggers complementary interfaces when `z_offset_mode` is `alternating`: strands A/C/E receive 0 A and B/D/F receive the offset.
- Right-handed models use positive azimuthal advance with increasing z. Left-handed models reverse that sign.

## Peptide-Plane Orientation Parameters

- `plane_normal_to_axis_deg`: angle between the peptide-plane normal and the global z-axis.
- `plane_azimuth_deg`: direction of the plane-normal projection in the local radial/tangential frame.
- `in_plane_spin_deg`: rotation of the motif within the already oriented peptide plane.

The normal orientation and in-plane spin are distinct: the first controls how the plane is tilted relative to the helix axis; the second controls how CA/C/O/N/CA are laid out inside that plane.

## Starter Sweep

This run generated {len(manifest)} models. The helix radius values in this run are: {', '.join(f'{value:.2f}' for value in sorted(manifest['helix_radius_A'].unique()))} Angstrom. The active z-offset values are: {', '.join(f'{value:.2f}' for value in sorted(manifest['active_z_offset_A'].unique()))} Angstrom. Radius and strand register are modeling assumptions and should be revisited before quantitative diffraction testing.

Example models:

{example_lines}

## Intended Use

Use these PDB/XYZ files as minimal, interpretable inputs for downstream visualization and diffraction testing. They are diagnostic models for isolating peptide-plane orientation effects, not replacements for chemically detailed Hexaflex/Hexaplex structures.

## Files

- `model_manifest.csv`: parameter table and generated file paths.
- `orientation_definition_schematic.png`: schematic of the normal-angle and azimuth definitions.
- `starter_model_preview.png`: top/side preview of the first generated model.
"""
    (outdir / "README.md").write_text(text, encoding="utf-8")


def plot_orientation_schematic(outdir: Path, params: ModelParameters) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    axis = np.array([0.0, 1.0])
    normal_angle = np.radians(params.plane_normal_to_axis_deg)
    normal = np.array([np.sin(normal_angle), np.cos(normal_angle)])
    ax.arrow(0, 0, axis[0], axis[1], head_width=0.05, length_includes_head=True, color="#333333")
    ax.arrow(0, 0, normal[0], normal[1], head_width=0.05, length_includes_head=True, color="#d95f02")
    ax.text(axis[0] + 0.03, axis[1], "helical axis z", color="#333333")
    ax.text(normal[0] + 0.03, normal[1], "plane normal", color="#d95f02")
    arc = np.linspace(0, normal_angle, 60)
    ax.plot(0.35 * np.sin(arc), 0.35 * np.cos(arc), color="#d95f02")
    ax.text(0.24, 0.36, "plane_normal_to_axis_deg", fontsize=9)
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_aspect("equal")
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.15, 1.2)
    ax.axis("off")
    ax.set_title("Peptide-plane orientation definition")
    fig.tight_layout()
    fig.savefig(outdir / "orientation_definition_schematic.png", dpi=200)
    plt.close(fig)


def plot_starter_preview(outdir: Path, params: ModelParameters) -> None:
    atoms = generate_model_atoms(params)
    coords = np.array([atom.coord for atom in atoms])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].scatter(coords[:, 0], coords[:, 1], s=5, alpha=0.55)
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("x (A)")
    axes[0].set_ylabel("y (A)")
    axes[0].set_title("top projection")
    axes[1].scatter(coords[:, 0], coords[:, 2], s=5, alpha=0.55)
    axes[1].set_xlabel("x (A)")
    axes[1].set_ylabel("z (A)")
    axes[1].set_title("side projection")
    fig.suptitle(params.model_label, fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "starter_model_preview.png", dpi=200)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    model_dir = args.outdir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    first_params: ModelParameters | None = None
    for params in iter_parameter_sweep(args):
        first_params = first_params or params
        atoms = generate_model_atoms(params)
        pdb_path = model_dir / f"{params.model_label}.pdb"
        xyz_path = model_dir / f"{params.model_label}.xyz" if args.write_xyz else None
        write_pdb(atoms, pdb_path, params)
        if xyz_path is not None:
            write_xyz(atoms, xyz_path, comment=params.model_label)
        row = manifest_row(params, pdb_path, xyz_path, len(atoms))
        row["measured_plane_normal_to_axis_deg"] = measured_normal_to_axis_angle(params)
        rows.append(row)

    manifest = pd.DataFrame(rows)
    manifest.to_csv(args.outdir / "model_manifest.csv", index=False)
    if first_params is not None:
        plot_orientation_schematic(args.outdir, first_params)
        plot_starter_preview(args.outdir, first_params)
    write_readme(args.outdir, manifest)

    print(f"Generated {len(manifest)} parametric peptide-plane models")
    print(f"Output directory: {args.outdir}")
    print(f"Manifest: {args.outdir / 'model_manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
