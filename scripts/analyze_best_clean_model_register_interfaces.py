"""Deep-dive best clean ideal Hexaflex models by atom class, interface, and register."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from scripts.analyze_ideal_hexaflex_atom_class_cd import (
    RichAtom,
    atom_classes,
    canonical_pair_class,
    pair_class_keys,
    parse_rich_pdb,
)


FOCUS_VARIANTS = ["backbone_plus_carboxylate", "peptide_plane_plus_carboxylate"]
COMPARISON_VARIANTS = ["backbone_only", "peptide_plane_only", "no_side_chain"]
REPORT_MODEL = "ideal_hexaflex_backbone_plus_carboxylate_pair_family_cd"
SUMMARY_NAME = "best_clean_model_register_interface_cd_summary.csv"


def ring_separation(strand_a: int, strand_b: int, n_strands: int = 6) -> int:
    """Return shortest ring separation."""
    raw = abs(strand_b - strand_a) % n_strands
    return min(raw, n_strands - raw)


def interface_label(atom_a: RichAtom, atom_b: RichAtom, n_strands: int = 6) -> str:
    """Return same_strand, nonadjacent, or AB/BC/.../FA interface label."""
    if atom_a.strand_index == atom_b.strand_index:
        return "same_strand"
    if ring_separation(atom_a.strand_index, atom_b.strand_index, n_strands=n_strands) != 1:
        return "nonadjacent"
    pair = {atom_a.strand_index, atom_b.strand_index}
    if pair == {0, n_strands - 1}:
        return "FA"
    low = min(pair)
    high = max(pair)
    return f"{CHAIN_IDS[low]}{CHAIN_IDS[high]}"


def alternating_interface_group(interface: str) -> str:
    """Classify adjacent interfaces into alternating groups."""
    if interface in {"AB", "CD", "EF"}:
        return "AB_CD_EF"
    if interface in {"BC", "DE", "FA"}:
        return "BC_DE_FA"
    if interface == "same_strand":
        return "same_strand"
    return "nonadjacent"


def register_offset_class(atom_a: RichAtom, atom_b: RichAtom) -> str:
    """Classify repeat/register offset."""
    if atom_a.strand_index == atom_b.strand_index:
        return "same_strand"
    delta = abs(atom_b.repeat_index - atom_a.repeat_index)
    if delta == 0:
        return "same"
    if delta == 1:
        return "plusminus1"
    if delta == 2:
        return "plusminus2"
    return "plusminus3_or_more"


def focused_geometry_families(atom_a: RichAtom, atom_b: RichAtom) -> list[str]:
    """Return existing geometry families for this pair."""
    return classify_pair_families(atom_a, atom_b, n_strands=6)


def aggregate_distances(atoms: list[RichAtom]) -> dict[tuple[str, str, str, str, str, str], list[float]]:
    """Aggregate distances by atom-class pair, interface/register labels, and geometry family."""
    distances_by_key: dict[tuple[str, str, str, str, str, str], list[float]] = defaultdict(list)
    coords = np.array([atom.coord for atom in atoms])
    class_cache = [atom_classes(atom) for atom in atoms]
    for i in range(len(atoms) - 1):
        deltas = coords[i + 1 :] - coords[i]
        distances = np.linalg.norm(deltas, axis=1)
        atom_a = atoms[i]
        for local_j, distance in enumerate(distances):
            j = i + 1 + local_j
            atom_b = atoms[j]
            interface = interface_label(atom_a, atom_b)
            alt_group = alternating_interface_group(interface)
            register = register_offset_class(atom_a, atom_b)
            geometries = focused_geometry_families(atom_a, atom_b)
            for class_1, class_2 in pair_class_keys(class_cache[i], class_cache[j]):
                for geometry in geometries:
                    key = (class_1, class_2, interface, alt_group, register, geometry)
                    distances_by_key[key].append(float(distance))
    return distances_by_key


def window_profile_peak(profile: pd.DataFrame, d_min: float, d_max: float) -> tuple[float, float]:
    """Return max intensity and d spacing inside a window."""
    window = profile[profile["d_A"].between(d_min, d_max)]
    if window.empty:
        return np.nan, np.nan
    maxima = local_maxima(window.rename(columns={"q": "q_Ainv"}))
    source = maxima if not maxima.empty else window
    row = source.sort_values("intensity", ascending=False).iloc[0]
    return float(row["intensity"]), float(row["d_A"])


def summarize_distances(
    model_id: str,
    distances_by_key: dict[tuple[str, str, str, str, str, str], list[float]],
    q_values: np.ndarray,
    c_window: tuple[float, float],
    d_window: tuple[float, float],
) -> pd.DataFrame:
    """Create C/D summary rows for each aggregate key."""
    rows = []
    for key, distances in sorted(distances_by_key.items()):
        class_1, class_2, interface, alt_group, register, geometry = key
        arr = np.asarray(distances, dtype=float)
        profile = partial_debye_profile(arr, q_values)
        c_intensity, c_peak = window_profile_peak(profile, *c_window)
        d_intensity, d_peak = window_profile_peak(profile, *d_window)
        rows.append(
            {
                "model_id": model_id,
                "atom_class_1": class_1,
                "atom_class_2": class_2,
                "interface": interface,
                "alternating_interface_group": alt_group,
                "register_offset_class": register,
                "geometry_family": geometry,
                "C_pair_count": int(((arr >= c_window[0]) & (arr <= c_window[1])).sum()),
                "D_pair_count": int(((arr >= d_window[0]) & (arr <= d_window[1])).sum()),
                "C_profile_peak_d_A": c_peak,
                "D_profile_peak_d_A": d_peak,
                "C_profile_max_intensity": c_intensity,
                "D_profile_max_intensity": d_intensity,
            }
        )
    return pd.DataFrame(rows)


def top_rows(df: pd.DataFrame, model_id: str, column: str, n: int = 8) -> pd.DataFrame:
    """Return top rows for one model/metric."""
    subset = df[df["model_id"] == model_id].copy()
    return subset.sort_values(column, ascending=False).head(n)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    """Render a small markdown table."""
    columns = [c for c in columns if c in df.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in df[columns].itertuples(index=False):
        vals = []
        for value in row:
            vals.append(f"{value:.4g}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def summarize_group(df: pd.DataFrame, model_id: str, group_cols: list[str], metric: str) -> pd.DataFrame:
    """Summarize a metric by selected columns for one model."""
    subset = df[df["model_id"] == model_id]
    return subset.groupby(group_cols, as_index=False)[metric].sum().sort_values(metric, ascending=False)


def write_report(summary: pd.DataFrame, path: Path) -> None:
    """Write focused markdown report."""
    model_id = REPORT_MODEL if REPORT_MODEL in set(summary["model_id"]) else str(summary["model_id"].iloc[0])
    top_c = top_rows(summary, model_id, "C_pair_count", n=10)
    top_d = top_rows(summary, model_id, "D_pair_count", n=10)
    by_class_c = summarize_group(summary, model_id, ["atom_class_1", "atom_class_2"], "C_pair_count").head(8)
    by_class_d = summarize_group(summary, model_id, ["atom_class_1", "atom_class_2"], "D_pair_count").head(8)
    by_alt_c = summarize_group(summary, model_id, ["alternating_interface_group"], "C_pair_count")
    by_alt_d = summarize_group(summary, model_id, ["alternating_interface_group"], "D_pair_count")
    by_register_c = summarize_group(summary, model_id, ["register_offset_class"], "C_pair_count")
    by_register_d = summarize_group(summary, model_id, ["register_offset_class"], "D_pair_count")

    text = f"""# Best Clean Model Register/Interface C/D Diagnostic

Focused model: `{model_id}`

This report combines overlapping atom-class pairs, strand interfaces, alternating interface groups, register offsets, and existing geometry-family labels. It is diagnostic and uses the ideal Hexaflex variants as controlled models, not experimental truth.

## Top C Combinations

{markdown_table(top_c, ['atom_class_1', 'atom_class_2', 'interface', 'alternating_interface_group', 'register_offset_class', 'geometry_family', 'C_pair_count', 'C_profile_peak_d_A'])}

## Top D Combinations

{markdown_table(top_d, ['atom_class_1', 'atom_class_2', 'interface', 'alternating_interface_group', 'register_offset_class', 'geometry_family', 'D_pair_count', 'D_profile_peak_d_A'])}

## Atom-Class Pair Totals

C:

{markdown_table(by_class_c, ['atom_class_1', 'atom_class_2', 'C_pair_count'])}

D:

{markdown_table(by_class_d, ['atom_class_1', 'atom_class_2', 'D_pair_count'])}

## Alternating Interface Groups

C:

{markdown_table(by_alt_c, ['alternating_interface_group', 'C_pair_count'])}

D:

{markdown_table(by_alt_d, ['alternating_interface_group', 'D_pair_count'])}

## Register Offset Classes

C:

{markdown_table(by_register_c, ['register_offset_class', 'C_pair_count'])}

D:

{markdown_table(by_register_d, ['register_offset_class', 'D_pair_count'])}

## Direct Answers

- For `backbone_plus_carboxylate`, dominant C combinations are listed in the Top C table; compare `backbone x backbone` and `backbone x carboxylate` rows to judge whether C is backbone-core or carboxylate-assisted.
- Dominant D combinations are listed in the Top D table, with register and interface labels preserved.
- D improvement after carboxylate add-back should be read from `backbone/carboxylate` and `peptide_plane/carboxylate` cross-strand rows, especially same/slipped-register labels.
- AB/CD/EF versus BC/DE/FA differences are summarized in the alternating-interface tables.
- Register specificity is summarized in the register-offset tables for C and D.

Outputs:
- Summary CSV: `outputs/metrics/best_clean_model_register_interface_cd_summary.csv`
- Heatmap: `outputs/figures/best_clean_model_register_interface_cd_heatmap.png`
"""
    path.write_text(text, encoding="utf-8")


def save_heatmap(summary: pd.DataFrame, path_base: Path) -> None:
    """Save focused C/D heatmaps by atom pair, interface group, and register."""
    focus = summary[summary["model_id"] == REPORT_MODEL].copy()
    if focus.empty:
        focus = summary.copy()
    focus["atom_pair"] = focus["atom_class_1"] + " x " + focus["atom_class_2"]
    focus["group_register"] = focus["alternating_interface_group"] + " / " + focus["register_offset_class"]
    focus_pairs = (
        focus.groupby("atom_pair")["C_pair_count"].sum().add(focus.groupby("atom_pair")["D_pair_count"].sum(), fill_value=0)
        .sort_values(ascending=False)
        .head(12)
        .index
    )
    focus = focus[focus["atom_pair"].isin(focus_pairs)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, metric, title in zip(axes, ["C_pair_count", "D_pair_count"], ["C-window pair counts", "D-window pair counts"]):
        pivot = focus.pivot_table(index="atom_pair", columns="group_register", values=metric, aggfunc="sum", fill_value=0)
        pivot = pivot.sort_index().sort_index(axis=1)
        image = ax.imshow(np.log10(pivot.to_numpy(float) + 1.0), aspect="auto", cmap="viridis")
        ax.set_title(title)
        ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(pivot.index)), pivot.index, fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="log10(count + 1)")
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".png"), dpi=180)
    fig.savefig(path_base.with_suffix(".svg"))
    plt.close(fig)


def load_models(manifest_path: Path, include_comparisons: bool) -> list[tuple[str, Path]]:
    """Load model IDs and PDB paths for requested variants."""
    wanted = set(FOCUS_VARIANTS + (COMPARISON_VARIANTS if include_comparisons else []))
    manifest = pd.read_csv(manifest_path)
    rows = []
    for row in manifest.itertuples(index=False):
        if str(row.variant) in wanted and str(row.written).lower() in {"true", "1"}:
            rows.append((str(row.model_id), Path(str(row.pdb_path))))
    return rows


def analyze(
    manifest_path: Path,
    metrics_dir: Path,
    reports_dir: Path,
    figures_dir: Path,
    c_window: tuple[float, float],
    d_window: tuple[float, float],
    include_comparisons: bool = True,
    q_step: float = 0.01,
) -> pd.DataFrame:
    """Run register/interface diagnostics for selected models."""
    q_values = make_q_grid(d_min_A=2.5, d_max_A=12.0, q_step=q_step)
    frames = []
    for model_id, pdb_path in load_models(manifest_path, include_comparisons):
        atoms = parse_rich_pdb(pdb_path)
        distances = aggregate_distances(atoms)
        frames.append(summarize_distances(model_id, distances, q_values, c_window, d_window))
    summary = pd.concat(frames, ignore_index=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(metrics_dir / SUMMARY_NAME, index=False)
    write_report(summary, reports_dir / "best_clean_model_register_interface_cd_report.md")
    save_heatmap(summary, figures_dir / "best_clean_model_register_interface_cd_heatmap")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant-manifest", type=Path, default=Path("outputs/coordinates/ideal_hexaflex_variants/variant_manifest.csv"))
    parser.add_argument("--metrics-dir", type=Path, default=Path("outputs/metrics"))
    parser.add_argument("--reports-dir", type=Path, default=Path("outputs/reports"))
    parser.add_argument("--figures-dir", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--c-min", type=float, default=5.4)
    parser.add_argument("--c-max", type=float, default=5.8)
    parser.add_argument("--d-min", type=float, default=7.0)
    parser.add_argument("--d-max", type=float, default=7.5)
    parser.add_argument("--q-step", type=float, default=0.01)
    parser.add_argument("--no-comparisons", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = analyze(
        args.variant_manifest,
        args.metrics_dir,
        args.reports_dir,
        args.figures_dir,
        (args.c_min, args.c_max),
        (args.d_min, args.d_max),
        include_comparisons=not args.no_comparisons,
        q_step=args.q_step,
    )
    print(f"Wrote {len(summary)} register/interface rows")
    print(f"Summary: {args.metrics_dir / SUMMARY_NAME}")
    print(f"Report: {args.reports_dir / 'best_clean_model_register_interface_cd_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
