"""Scan controlled symmetry-breaking variants for C/D powder recovery.

This is a diagnostic/falsification scan for the simple six-strand
peptide-bond-plane models. It generates bounded coordinate perturbations around
the current D-preserving model, scores C/D Debye peak positions, and reuses the
pair-family diagnostic summary for each generated PDB.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hexaplex_backbone_fingerprint.parametric_peptide_plane_models import (
    ModelParameters,
    PlacedAtom,
    format_param,
    generate_model_atoms,
    write_pdb,
    write_xyz,
)
from hexaplex_backbone_fingerprint.parametric_powder_scan import debye_profile, make_q_grid, nearest_peak

from scripts.analyze_backbone_pair_family_cd import (
    compute_pair_family_distances,
    load_labeled_atoms,
    plot_histograms,
    plot_profiles,
    write_cd_summary,
    write_histograms,
    write_profiles,
    write_report as write_pair_family_report,
)


BASE_PARAMS = ModelParameters(
    n_strands=6,
    repeats_per_strand=16,
    helix_radius_A=8.0,
    twist_deg=32.0,
    rise_A=3.40,
    plane_normal_to_axis_deg=40.0,
    plane_azimuth_deg=90.0,
    in_plane_spin_deg=0.0,
    z_offset_mode="alternating",
    alternating_z_offset_A=0.75,
)
RADIUS_AMPLITUDES_A = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
Z_OFFSETS_A = [0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50]
REPEAT_PERTURBATIONS_A = [-0.10, -0.05, 0.0, 0.05, 0.10]


def radial_delta_pattern(pattern: str, amplitude_A: float, strand_index: int) -> float:
    """Return radial displacement for one strand under a named pattern."""
    if pattern == "none":
        return 0.0
    if pattern == "three_class_ACF":
        return [amplitude_A, 0.0, -amplitude_A, amplitude_A, 0.0, -amplitude_A][strand_index % 6]
    if pattern == "alternating_ACE_BDF":
        return amplitude_A if strand_index % 2 == 0 else -amplitude_A
    raise ValueError(f"Unknown radial pattern: {pattern}")


def z_offset_pattern(pattern: str, offset_A: float, strand_index: int) -> float:
    """Return axial strand offset for one symmetry-breaking z pattern."""
    if pattern == "none":
        return 0.0
    if pattern == "ACE_vs_BDF":
        return offset_A if strand_index % 2 == 1 else 0.0
    if pattern == "AB_CD_EF_biased":
        return [0.0, 0.0, offset_A, offset_A, 2.0 * offset_A, 2.0 * offset_A][strand_index % 6]
    raise ValueError(f"Unknown z-offset pattern: {pattern}")


def perturb_atom(
    atom: PlacedAtom,
    radial_pattern: str,
    radial_amplitude_A: float,
    z_pattern: str,
    z_offset_A: float,
    repeat_perturb_A: float,
) -> PlacedAtom:
    """Apply radial, strand-phase, and along-strand repeat perturbations to one atom."""
    coord = np.array(atom.coord, dtype=float).copy()
    xy = coord[:2]
    norm = np.linalg.norm(xy)
    if norm > 1e-12:
        radial = xy / norm
        coord[:2] = coord[:2] + radial_delta_pattern(radial_pattern, radial_amplitude_A, atom.strand_index) * radial
    coord[2] += z_offset_pattern(z_pattern, z_offset_A, atom.strand_index)
    coord[2] += repeat_perturb_A * atom.repeat_index
    return replace(atom, coord=coord)


def perturb_atoms(
    atoms: list[PlacedAtom],
    radial_pattern: str = "none",
    radial_amplitude_A: float = 0.0,
    z_pattern: str = "none",
    z_offset_A: float = 0.0,
    repeat_perturb_A: float = 0.0,
) -> list[PlacedAtom]:
    """Apply a symmetry-breaking perturbation to all atoms."""
    return [
        perturb_atom(atom, radial_pattern, radial_amplitude_A, z_pattern, z_offset_A, repeat_perturb_A)
        for atom in atoms
    ]


def iter_scan_specs() -> list[dict[str, object]]:
    """Return bounded scan specifications around the D-preserving base model."""
    specs: list[dict[str, object]] = []
    for amplitude in RADIUS_AMPLITUDES_A:
        specs.append(
            {
                "scan_family": "alternating_interface_radius",
                "radial_pattern": "three_class_ACF",
                "radial_amplitude_A": amplitude,
                "z_pattern": "none",
                "z_offset_A": 0.0,
                "repeat_perturb_A": 0.0,
            }
        )
    for pattern in ["ACE_vs_BDF", "AB_CD_EF_biased"]:
        for offset in Z_OFFSETS_A:
            specs.append(
                {
                    "scan_family": "alternating_z_phase",
                    "radial_pattern": "none",
                    "radial_amplitude_A": 0.0,
                    "z_pattern": pattern,
                    "z_offset_A": offset,
                    "repeat_perturb_A": 0.0,
                }
            )
    for perturb in REPEAT_PERTURBATIONS_A:
        specs.append(
            {
                "scan_family": "along_strand_repeat_perturbation",
                "radial_pattern": "none",
                "radial_amplitude_A": 0.0,
                "z_pattern": "none",
                "z_offset_A": 0.0,
                "repeat_perturb_A": perturb,
            }
        )
    for amplitude in RADIUS_AMPLITUDES_A:
        for offset in Z_OFFSETS_A:
            specs.append(
                {
                    "scan_family": "radius_plus_alternating_z_phase",
                    "radial_pattern": "three_class_ACF",
                    "radial_amplitude_A": amplitude,
                    "z_pattern": "ACE_vs_BDF",
                    "z_offset_A": offset,
                    "repeat_perturb_A": 0.0,
                }
            )
    return specs


def spec_label(spec: dict[str, object]) -> str:
    """Return compact label for a scan specification."""
    return (
        f"{spec['scan_family']}"
        f"_rpat{spec['radial_pattern']}_ramp{format_param(float(spec['radial_amplitude_A']))}"
        f"_zpat{spec['z_pattern']}_z{format_param(float(spec['z_offset_A']))}"
        f"_repdz{format_param(float(spec['repeat_perturb_A']))}"
    )


def coords_from_atoms(atoms: list[PlacedAtom]) -> np.ndarray:
    """Return coordinates for all atoms, including hydrogens to match prior scans."""
    return np.array([atom.coord for atom in atoms], dtype=float)


def top_family(summary: pd.DataFrame, column: str) -> str:
    """Return top family by a numeric column."""
    values = pd.to_numeric(summary[column], errors="coerce").fillna(float("-inf"))
    if values.empty:
        return ""
    return str(summary.loc[values.idxmax(), "family"])


def focus_count(summary: pd.DataFrame, family: str, column: str) -> int:
    """Return focused family count from pair-family summary."""
    subset = summary[summary["family"] == family]
    if subset.empty:
        return 0
    return int(pd.to_numeric(subset.iloc[0][column], errors="coerce"))


def score_model(c_error: float, d_error: float, c_family: str, d_family: str) -> float:
    """Score models by C/D proximity with a small family bonus."""
    family_bonus = 0.0
    if c_family.startswith("same_strand"):
        family_bonus += 0.05
    if d_family in {"all_cross_strand", "all_adjacent_cross_strand", "adjacent_strand_same_register", "adjacent_strand_plusminus1_register"}:
        family_bonus += 0.05
    return -(abs(c_error) + abs(d_error)) + family_bonus


def analyze_generated_model(
    model_id: str,
    atoms: list[PlacedAtom],
    pdb_path: Path,
    xyz_path: Path,
    args: argparse.Namespace,
    q_values: np.ndarray,
) -> dict[str, object]:
    """Write one generated model and return combined powder/pair-family metrics."""
    write_pdb(atoms, pdb_path, BASE_PARAMS)
    write_xyz(atoms, xyz_path, comment=model_id)
    profile = debye_profile(coords_from_atoms(atoms), q_values)
    c_hit = nearest_peak(profile, target_d_A=5.6, tolerance_A=0.20)
    d_hit = nearest_peak(profile, target_d_A=7.3, tolerance_A=0.20)

    labeled_atoms = load_labeled_atoms(pdb_path)
    distances_by_family = compute_pair_family_distances(labeled_atoms, n_strands=6)
    histograms = write_histograms(
        model_id,
        distances_by_family,
        args.outdir / "metrics" / f"{model_id}_pair_family_distance_histograms.csv",
        0.05,
    )
    profiles = write_profiles(
        model_id,
        distances_by_family,
        args.outdir / "metrics" / f"{model_id}_pair_family_radial_profiles.csv",
        q_values,
    )
    pair_summary = write_cd_summary(
        model_id,
        distances_by_family,
        profiles,
        args.outdir / "metrics" / f"{model_id}_pair_family_cd_summary.csv",
        (5.4, 5.8),
        (7.0, 7.5),
    )
    if args.write_pair_plots:
        plot_histograms(histograms, model_id, args.outdir / "figures" / f"{model_id}_pair_family_cd_histograms")
        plot_profiles(profiles, pair_summary, model_id, args.outdir / "figures" / f"{model_id}_pair_family_radial_profiles_C_D_focus")
        write_pair_family_report(model_id, pdb_path, pair_summary, args.outdir / "reports" / f"{model_id}_pair_family_cd_report.md")

    c_top = top_family(pair_summary, "C_pair_count")
    d_top = top_family(pair_summary, "D_pair_count")
    return {
        "model_id": model_id,
        "pdb_path": str(pdb_path),
        "xyz_path": str(xyz_path),
        "C_peak_d_A": c_hit.peak_d_A,
        "D_peak_d_A": d_hit.peak_d_A,
        "C_error_A": c_hit.error_A,
        "D_error_A": d_hit.error_A,
        "C_within_tolerance": c_hit.found_within_tolerance,
        "D_within_tolerance": d_hit.found_within_tolerance,
        "both_within_tolerance": c_hit.found_within_tolerance and d_hit.found_within_tolerance,
        "C_top_family": c_top,
        "D_top_family": d_top,
        "AB_CD_EF_C_count": focus_count(pair_summary, "alternating_interfaces_AB_CD_EF", "C_pair_count"),
        "AB_CD_EF_D_count": focus_count(pair_summary, "alternating_interfaces_AB_CD_EF", "D_pair_count"),
        "BC_DE_FA_C_count": focus_count(pair_summary, "alternating_interfaces_BC_DE_FA", "C_pair_count"),
        "BC_DE_FA_D_count": focus_count(pair_summary, "alternating_interfaces_BC_DE_FA", "D_pair_count"),
        "total_CD_score": score_model(c_hit.error_A, d_hit.error_A, c_top, d_top),
    }


def write_scan_report(scan: pd.DataFrame, path: Path) -> None:
    """Write markdown scan interpretation."""
    d_preserved = scan[scan["D_peak_d_A"].between(7.2, 7.3)].copy()
    best = scan.sort_values("total_CD_score", ascending=False).head(8)
    d_then_c = d_preserved.assign(C_abs_error=scan.loc[d_preserved.index, "C_error_A"].abs()).sort_values("C_abs_error").head(8)
    c_same_count = int(scan["C_top_family"].str.startswith("same_strand").sum())
    d_cross_count = int(scan["D_top_family"].isin(["all_cross_strand", "all_adjacent_cross_strand", "adjacent_strand_same_register", "adjacent_strand_plusminus1_register"]).sum())
    alt_c_split = (scan["AB_CD_EF_C_count"] - scan["BC_DE_FA_C_count"]).median()
    alt_d_split = (scan["AB_CD_EF_D_count"] - scan["BC_DE_FA_D_count"]).median()
    text = f"""# Symmetry-Breaking C/D Scan

This is a diagnostic scan around the current simple six-strand peptide-bond-plane model. It applies controlled symmetry-breaking perturbations and reuses the existing Debye scoring and pair-family C/D diagnostic summaries.

## Scan Size

- Models scanned: {len(scan)}
- Models with D peak in 7.2-7.3 A: {len(d_preserved)}
- Models with both C and D within +/-0.20 A: {int(scan['both_within_tolerance'].sum())}

## Best Scoring Models

{markdown_table(best[['model_id', 'scan_family', 'C_peak_d_A', 'D_peak_d_A', 'C_error_A', 'D_error_A', 'C_top_family', 'D_top_family', 'total_CD_score']])}

## D-Preserved Models With Best C Error

{markdown_table(d_then_c[['model_id', 'scan_family', 'C_peak_d_A', 'D_peak_d_A', 'C_error_A', 'D_error_A', 'C_top_family', 'D_top_family']]) if not d_then_c.empty else 'No models kept D in 7.2-7.3 A.'}

## Cautious Interpretation

- C improvement/locality: {c_same_count} of {len(scan)} models had a same-strand family as the top C pair-count family.
- D register/cross-strand tendency: {d_cross_count} of {len(scan)} models had a cross-strand/register-like top D family.
- Alternating interface asymmetry: median AB/CD/EF minus BC/DE/FA C-count split was {alt_c_split:g}; D-count split was {alt_d_split:g}.
- If high-ranking D-preserved models still keep C low, the current perturbations are not enough to tune the C-sensitive local-repeat geometry independently from D-sensitive cross-strand/register geometry.

This remains a direct forward-modeling diagnostic, not a final structural assignment.
"""
    path.write_text(text, encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> str:
    """Render a small dataframe as markdown without optional dependencies."""
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in df.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.4g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def plot_scan(scan: pd.DataFrame, path_base: Path) -> None:
    """Plot C vs D peak positions colored by score."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    scatter = ax.scatter(
        scan["C_peak_d_A"],
        scan["D_peak_d_A"],
        c=scan["total_CD_score"],
        s=40,
        cmap="viridis",
        alpha=0.8,
    )
    ax.axvspan(5.4, 5.8, color="#1f77b4", alpha=0.12)
    ax.axhspan(7.0, 7.5, color="#ff7f0e", alpha=0.12)
    ax.axhspan(7.2, 7.3, color="#2ca02c", alpha=0.10)
    ax.set_xlabel("C peak d spacing (A)")
    ax.set_ylabel("D peak d spacing (A)")
    ax.set_title("Symmetry-breaking scan: C vs D peak positions")
    fig.colorbar(scatter, ax=ax, label="total C/D score")
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".png"), dpi=180)
    fig.savefig(path_base.with_suffix(".svg"))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("outputs"))
    parser.add_argument("--max-models", type=int, default=None)
    parser.add_argument("--write-pair-plots", action="store_true", help="Write per-model pair-family plots/reports.")
    parser.add_argument("--q-step", type=float, default=0.005)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for subdir in ["metrics", "reports", "figures", "symmetry_breaking_cd_models"]:
        (args.outdir / subdir).mkdir(parents=True, exist_ok=True)
    specs = iter_scan_specs()
    if args.max_models is not None:
        specs = specs[: args.max_models]
    base_atoms = generate_model_atoms(BASE_PARAMS)
    q_values = make_q_grid(d_min_A=2.5, d_max_A=12.0, q_step=args.q_step)
    rows = []
    model_dir = args.outdir / "symmetry_breaking_cd_models"
    for idx, spec in enumerate(specs, start=1):
        label = f"symbreak_{idx:03d}_{spec_label(spec)}"
        atoms = perturb_atoms(
            base_atoms,
            radial_pattern=str(spec["radial_pattern"]),
            radial_amplitude_A=float(spec["radial_amplitude_A"]),
            z_pattern=str(spec["z_pattern"]),
            z_offset_A=float(spec["z_offset_A"]),
            repeat_perturb_A=float(spec["repeat_perturb_A"]),
        )
        row = {
            **spec,
            **analyze_generated_model(
                label,
                atoms,
                model_dir / f"{label}.pdb",
                model_dir / f"{label}.xyz",
                args,
                q_values,
            ),
        }
        rows.append(row)
    scan = pd.DataFrame(rows)
    scan_path = args.outdir / "metrics" / "symmetry_breaking_cd_scan.csv"
    scan.to_csv(scan_path, index=False)
    write_scan_report(scan, args.outdir / "reports" / "symmetry_breaking_cd_scan_report.md")
    plot_scan(scan, args.outdir / "figures" / "symmetry_breaking_cd_scan_C_vs_D")
    print(f"Scanned {len(scan)} symmetry-breaking models")
    print(f"CSV: {scan_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
