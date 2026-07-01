"""Diagnose pair-distance families behind parametric powder features."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
DEFAULT_OUTDIR = Path("outputs/parametric_peak_pair_family_diagnostics")
TARGET_WINDOWS = [
    ("near_5p0", 5.0, 0.10),
    ("C_5p6", 5.6, 0.10),
    ("D_7p3", 7.3, 0.10),
    ("near_8p3", 8.3, 0.10),
]
REPRESENTATIVE_LABELS = [
    "6strand_tw32_rise3p4_rad8_norm40_az90_spin0_zoff0p75_alternating_rep16_right",
    "6strand_tw32_rise3p4_rad9_norm30_az70_spin0_zoff2p5_alternating_rep16_right",
    "6strand_tw32_rise3p4_rad8_norm40_az70_spin120_rep16_right",
]
MANIFEST_PATHS = [
    Path("outputs/parametric_six_strand_peptide_plane_models_alternating_zoffset/model_manifest.csv"),
    Path("outputs/parametric_six_strand_peptide_plane_models_zoffset/model_manifest.csv"),
    Path("outputs/parametric_six_strand_peptide_plane_models_refined/model_manifest.csv"),
]
BEST_MODEL_TABLES = [
    Path("outputs/parametric_six_strand_powder_scan_alternating_zoffset/best_parametric_powder_models.csv"),
    Path("outputs/parametric_six_strand_powder_scan_zoffset/best_parametric_powder_models.csv"),
    Path("outputs/parametric_six_strand_powder_scan_refined/best_parametric_powder_models.csv"),
]


@dataclass(frozen=True)
class ParametricAtom:
    """One labeled atom parsed from a generated parametric PDB."""

    serial: int
    atom_name: str
    resname: str
    chain: str
    resseq: int
    element: str
    coord: np.ndarray

    @property
    def strand_index(self) -> int:
        return CHAIN_IDS.index(self.chain)

    @property
    def repeat_index(self) -> int:
        return (self.resseq - 1) // 2

    @property
    def motif_atom_label(self) -> str:
        return f"{self.resname}:{self.atom_name}"


def parse_parametric_pdb(path: Path) -> list[ParametricAtom]:
    """Parse ATOM records from a generated parametric model PDB."""
    atoms: list[ParametricAtom] = []
    for line in path.read_text(encoding="ascii").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        atoms.append(
            ParametricAtom(
                serial=int(line[6:11]),
                atom_name=line[12:16].strip(),
                resname=line[17:20].strip(),
                chain=line[21].strip(),
                resseq=int(line[22:26]),
                element=line[76:78].strip() or line[12:16].strip()[0],
                coord=np.array(
                    [
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ],
                    dtype=float,
                ),
            )
        )
    if not atoms:
        raise ValueError(f"No ATOM/HETATM records found in {path}")
    return atoms


def ring_separation(strand_a: int, strand_b: int, n_strands: int = 6) -> int:
    """Return shortest separation around a cyclic strand ring."""
    raw = abs(strand_b - strand_a) % n_strands
    return min(raw, n_strands - raw)


def pair_family(atom_a: ParametricAtom, atom_b: ParametricAtom) -> dict[str, object]:
    """Classify one atom pair by strand, repeat, and motif atom family."""
    same_strand = atom_a.chain == atom_b.chain
    strand_pair = "-".join(sorted([atom_a.chain, atom_b.chain]))
    repeat_offset = atom_b.repeat_index - atom_a.repeat_index
    abs_repeat_offset = abs(repeat_offset)
    atom_pair_type = "-".join(sorted([atom_a.motif_atom_label, atom_b.motif_atom_label]))
    if abs_repeat_offset == 0:
        repeat_class = "same_repeat"
    elif abs_repeat_offset == 1:
        repeat_class = "neighboring_repeat"
    else:
        repeat_class = "longer_offset"
    return {
        "same_strand": same_strand,
        "strand_pair": strand_pair,
        "ring_separation": ring_separation(atom_a.strand_index, atom_b.strand_index),
        "repeat_offset": repeat_offset,
        "abs_repeat_offset": abs_repeat_offset,
        "repeat_class": repeat_class,
        "atom_pair_type": atom_pair_type,
    }


def load_manifest_index() -> pd.DataFrame:
    """Load available parametric model manifests into one model-label index."""
    rows = []
    for manifest_path in MANIFEST_PATHS:
        path = ROOT / manifest_path
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["manifest_path"] = str(manifest_path)
        rows.append(df)
    if not rows:
        raise FileNotFoundError("No parametric model manifests were found.")
    merged = pd.concat(rows, ignore_index=True)
    return merged.drop_duplicates("model_label", keep="first")


def select_models(manifest: pd.DataFrame, include_top_n: int) -> pd.DataFrame:
    """Select required representatives plus optional top-ranked models."""
    labels: list[str] = []
    for label in REPRESENTATIVE_LABELS:
        if label in set(manifest["model_label"]):
            labels.append(label)
    for table_path in BEST_MODEL_TABLES:
        path = ROOT / table_path
        if not path.exists():
            continue
        best = pd.read_csv(path)
        for label in best["model_label"].head(include_top_n):
            if label in set(manifest["model_label"]):
                labels.append(label)
    labels = list(dict.fromkeys(labels))
    selected = manifest[manifest["model_label"].isin(labels)].copy()
    selected["selection_order"] = selected["model_label"].map({label: i for i, label in enumerate(labels)})
    return selected.sort_values("selection_order")


def resolve_repo_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def pair_detail_rows(model_row: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-window pair detail rows and histogram distances for one model."""
    atoms = parse_parametric_pdb(resolve_repo_path(str(model_row["pdb_path"])))
    coords = np.array([atom.coord for atom in atoms])
    detail_rows = []
    histogram_rows = []
    for i in range(len(atoms) - 1):
        deltas = coords[i + 1 :] - coords[i]
        distances = np.linalg.norm(deltas, axis=1)
        for distance in distances:
            histogram_rows.append({"model_label": model_row["model_label"], "distance": float(distance)})
        for window_name, target, tolerance in TARGET_WINDOWS:
            mask = np.abs(distances - target) <= tolerance
            if not np.any(mask):
                continue
            q = 2.0 * np.pi / target
            for local_j, distance in zip(np.flatnonzero(mask), distances[mask]):
                atom_a = atoms[i]
                atom_b = atoms[i + 1 + int(local_j)]
                family = pair_family(atom_a, atom_b)
                contribution = float(np.sinc((q * distance) / np.pi))
                detail_rows.append(
                    {
                        "model_label": model_row["model_label"],
                        "model_role": model_role(str(model_row["model_label"])),
                        "pdb_path": model_row["pdb_path"],
                        "target_window": window_name,
                        "target_d_A": target,
                        "window_tolerance_A": tolerance,
                        "distance_A": float(distance),
                        "debye_pair_contribution": contribution,
                        "atom_serial_a": atom_a.serial,
                        "atom_serial_b": atom_b.serial,
                        "chain_a": atom_a.chain,
                        "chain_b": atom_b.chain,
                        "repeat_i": atom_a.repeat_index,
                        "repeat_j": atom_b.repeat_index,
                        "atom_a": atom_a.motif_atom_label,
                        "atom_b": atom_b.motif_atom_label,
                        **family,
                    }
                )
    return pd.DataFrame(detail_rows), pd.DataFrame(histogram_rows)


def model_role(label: str) -> str:
    """Return a compact human role for key representative labels."""
    if label == REPRESENTATIVE_LABELS[0]:
        return "best_combined_alternating"
    if label == REPRESENTATIVE_LABELS[1]:
        return "best_C_only_alternating"
    if label == REPRESENTATIVE_LABELS[2]:
        return "best_refined_zero_offset"
    return "top_ranked_optional"


def _top_counts(series: pd.Series, n: int = 5) -> str:
    counts = series.value_counts().head(n)
    return "; ".join(f"{idx}: {count}" for idx, count in counts.items())


def summarize_details(details: pd.DataFrame) -> pd.DataFrame:
    """Summarize pair families for every model and target window."""
    rows = []
    group_cols = ["model_label", "model_role", "target_window", "target_d_A"]
    for keys, group in details.groupby(group_cols, dropna=False):
        same = int(group["same_strand"].sum())
        cross = int(len(group) - same)
        contribution = group["debye_pair_contribution"]
        rows.append(
            {
                "model_label": keys[0],
                "model_role": keys[1],
                "target_window": keys[2],
                "target_d_A": keys[3],
                "pair_count": len(group),
                "same_strand_count": same,
                "cross_strand_count": cross,
                "cross_strand_fraction": cross / len(group) if len(group) else np.nan,
                "dominant_strand_pair": group["strand_pair"].value_counts().idxmax(),
                "dominant_ring_separation": group["ring_separation"].value_counts().idxmax(),
                "dominant_repeat_offset": group["repeat_offset"].value_counts().idxmax(),
                "dominant_abs_repeat_offset": group["abs_repeat_offset"].value_counts().idxmax(),
                "dominant_atom_pair_type": group["atom_pair_type"].value_counts().idxmax(),
                "dominant_repeat_class": group["repeat_class"].value_counts().idxmax(),
                "median_distance_A": group["distance_A"].median(),
                "distance_std_A": group["distance_A"].std(ddof=0),
                "debye_contribution_sum": contribution.sum(),
                "debye_positive_sum": contribution[contribution > 0].sum(),
                "debye_negative_sum": contribution[contribution < 0].sum(),
                "top_strand_pairs": _top_counts(group["strand_pair"]),
                "top_repeat_offsets": _top_counts(group["repeat_offset"]),
                "top_atom_pair_types": _top_counts(group["atom_pair_type"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["model_role", "model_label", "target_d_A"])


def save_histogram_plot(histograms: pd.DataFrame, outdir: Path) -> None:
    selected = histograms[histograms["distance"].between(4.0, 9.0)]
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, group in selected.groupby("model_label"):
        ax.hist(group["distance"], bins=np.linspace(4.0, 9.0, 101), histtype="step", lw=1.2, label=short_label(label))
    for _name, target, _tol in TARGET_WINDOWS:
        ax.axvline(target, color="#666666", lw=0.8, ls="--")
    ax.set_xlabel("pair distance (A)")
    ax.set_ylabel("pair count")
    ax.set_title("Pair-distance histograms for representative parametric models")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(outdir / "pair_distance_histograms_representative_models.png", dpi=180)
    plt.close(fig)


def short_label(label: str) -> str:
    """Make a plot legend label compact."""
    return (
        label.replace("6strand_tw32_rise3p4_", "")
        .replace("_rep16_right", "")
        .replace("norm", "n")
        .replace("spin", "s")
    )


def save_count_plots(details: pd.DataFrame, summary: pd.DataFrame, outdir: Path) -> None:
    counts = summary.pivot_table(index="model_role", columns="target_window", values="pair_count", aggfunc="sum").fillna(0)
    fig, ax = plt.subplots(figsize=(9, 5))
    counts.plot(kind="bar", ax=ax)
    ax.set_ylabel("pair count")
    ax.set_title("Pair-family counts by target window")
    fig.tight_layout()
    fig.savefig(outdir / "pair_family_counts_by_target_window.png", dpi=180)
    plt.close(fig)

    repeat_counts = details.groupby(["model_role", "target_window", "repeat_offset"]).size().reset_index(name="count")
    repeat_counts["panel"] = repeat_counts["model_role"] + " / " + repeat_counts["target_window"]
    pivot = repeat_counts.pivot_table(index="panel", columns="repeat_offset", values="count", aggfunc="sum").fillna(0)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    image = ax.imshow(pivot.values, aspect="auto", cmap="magma")
    ax.set_xticks(range(len(pivot.columns)), [str(col) for col in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_xlabel("repeat_offset")
    ax.set_title("Dominant repeat offsets by model and target")
    fig.colorbar(image, ax=ax, label="pair count")
    fig.tight_layout()
    fig.savefig(outdir / "dominant_repeat_offsets_by_model_and_target.png", dpi=180)
    plt.close(fig)

    top_atom_types = details["atom_pair_type"].value_counts().head(12).index
    atom_counts = (
        details[details["atom_pair_type"].isin(top_atom_types)]
        .groupby(["target_window", "atom_pair_type"])
        .size()
        .reset_index(name="count")
    )
    pivot = atom_counts.pivot_table(index="atom_pair_type", columns="target_window", values="count", aggfunc="sum").fillna(0)
    fig, ax = plt.subplots(figsize=(9, 6))
    image = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_title("Atom-pair type counts by target window")
    fig.colorbar(image, ax=ax, label="pair count")
    fig.tight_layout()
    fig.savefig(outdir / "atom_pair_type_counts_by_target_window.png", dpi=180)
    plt.close(fig)


def summary_row(summary: pd.DataFrame, role: str, window: str) -> pd.Series | None:
    subset = summary[(summary["model_role"] == role) & (summary["target_window"] == window)]
    if subset.empty:
        return None
    return subset.sort_values("pair_count", ascending=False).iloc[0]


def write_report(outdir: Path, selected: pd.DataFrame, summary: pd.DataFrame) -> None:
    d_row = summary_row(summary, "best_combined_alternating", "D_7p3")
    c_row = summary_row(summary, "best_C_only_alternating", "C_5p6")
    five_row = summary_row(summary, "best_combined_alternating", "near_5p0")
    eight_row = summary_row(summary, "best_C_only_alternating", "near_8p3")

    def describe(row: pd.Series | None) -> str:
        if row is None:
            return "No pairs found in this target window."
        return (
            f"{int(row.pair_count)} pairs; {row.cross_strand_fraction:.2%} cross-strand; "
            f"dominant strand pair {row.dominant_strand_pair}; ring separation {row.dominant_ring_separation}; "
            f"dominant repeat offset {row.dominant_repeat_offset}; dominant atom pair {row.dominant_atom_pair_type}; "
            f"median distance {row.median_distance_A:.3f} A; Debye pair contribution sum {row.debye_contribution_sum:.2f}."
        )

    c_family = "cross-strand" if c_row is not None and c_row.cross_strand_fraction >= 0.5 else "same-strand"
    five_family = "cross-strand" if five_row is not None and five_row.cross_strand_fraction >= 0.5 else "same-strand"
    d_family = "cross-strand" if d_row is not None and d_row.cross_strand_fraction >= 0.5 else "same-strand"

    selected_lines = "\n".join(f"- `{row.model_label}` ({model_role(row.model_label)})" for row in selected.itertuples())
    text = f"""# Parametric Peak Pair-Family Diagnostics

This is a direct forward-modeling diagnostic for the simple six-stranded peptide-bond-plane point models. It decomposes real-space pair-distance windows near 5.0, 5.6, 7.3, and 8.3 A to explain which atom/strand/repeat families feed the diagnostic Debye powder features.

## Models Analyzed

{selected_lines}

## Main Pair-Family Findings

- D-like 7.3 A feature in the D-preserving alternating model: {describe(d_row)}
- C-like 5.6 A feature in the C-only large-offset model: {describe(c_row)}
- 5.0 A feature in the D-preserving model: {describe(five_row)}
- 8.3 A feature in the C-only model: {describe(eight_row)}

## Interpretation

The D-preserving model's 7.3 A window is {d_family}, dominated by adjacent-ring strand pairs and mostly same-repeat or near-register contacts. The C-only large-offset model's 5.6 A window is {c_family}. In the current representative best C-only model, the 5.6 A window is dominated by same-strand neighboring-repeat motif contacts rather than by the same cross-strand family that produces D.

The 5.0 A feature in the D-preserving model is {five_family} and is not compositionally identical to the same-strand 5.6 A family in the C-only model. That means the simple model is not cleanly shifting one C-family from 5.0 to 5.6 while leaving D alone. Instead, large radius/offset settings create a local same-strand 5.6 A family and simultaneously move important cross-strand families toward the 8.3 A window.

This explains the observed tradeoff: D depends on adjacent cross-strand geometry near 7.3 A, while the easiest way this simple model makes a C-like 5.6 A feature is by changing local repeat geometry and stretching cross-strand contacts. The result does not support a single uniform or simple alternating model as sufficient for both bands.

The most justified next refinements are custom six-strand offset patterns or unequal interface classes so one interface family can preserve D while a complementary family is tuned toward C. A two-radius model is also plausible because the best C-only models favor larger radius while D-preserving models favor radius 8 A. Longer repeats and a more realistic motif should come after the interface/register degrees of freedom are tested.

## Output Tables

- `representative_model_pair_family_summary.csv`: one row per model and target window.
- `representative_model_pair_family_details.csv`: one row per atom pair in each target window.

## Plots

- `pair_distance_histograms_representative_models.png`
- `pair_family_counts_by_target_window.png`
- `dominant_repeat_offsets_by_model_and_target.png`
- `atom_pair_type_counts_by_target_window.png`
"""
    (outdir / "parametric_peak_pair_family_report.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--include-top-n", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest_index()
    selected = select_models(manifest, include_top_n=args.include_top_n)
    if selected.empty:
        raise ValueError("No representative parametric models were found.")

    detail_frames = []
    histogram_frames = []
    for row in selected.itertuples(index=False):
        details, histograms = pair_detail_rows(pd.Series(row._asdict()))
        detail_frames.append(details)
        histogram_frames.append(histograms)
    details = pd.concat(detail_frames, ignore_index=True)
    histograms = pd.concat(histogram_frames, ignore_index=True)
    summary = summarize_details(details)

    details.to_csv(outdir / "representative_model_pair_family_details.csv", index=False)
    summary.to_csv(outdir / "representative_model_pair_family_summary.csv", index=False)
    save_histogram_plot(histograms, outdir)
    save_count_plots(details, summary, outdir)
    write_report(outdir, selected, summary)

    print(f"Analyzed {selected['model_label'].nunique()} representative models")
    print(f"Wrote outputs to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
