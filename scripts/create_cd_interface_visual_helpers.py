"""Create PyMOL helpers and compact summaries for full-ideal C/D interfaces."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FULL_IDEAL_LABEL = "full_hexaplex_anti_parallel_30deg_ideal_deduped_6chain"
DEFAULT_SPATIAL_DIR = Path("outputs/cd_candidate_spatial_register")
DEFAULT_PLANE_FEATURES = Path("outputs/six_strand_first_panel") / FULL_IDEAL_LABEL / "plane_features.csv"
C_INTERFACES = ["A-B", "C-D", "E-F"]
D_INTERFACES = ["A-F", "B-C", "D-E"]
CHAIN_ORDER = ["A", "B", "C", "D", "E", "F"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create full-ideal C/D interface visualization helpers.")
    parser.add_argument("--spatial-dir", type=Path, default=DEFAULT_SPATIAL_DIR)
    parser.add_argument("--plane-features", type=Path, default=DEFAULT_PLANE_FEATURES)
    return parser.parse_args()


def load_full_ideal_pairs(spatial_dir: Path) -> pd.DataFrame:
    c_pairs = pd.read_csv(spatial_dir / "full_ideal_C_top_pairs.csv")
    d_pairs = pd.read_csv(spatial_dir / "full_ideal_D_top_pairs.csv")
    return pd.concat([c_pairs, d_pairs], ignore_index=True)


def add_box_resi_map(pairs: pd.DataFrame, plane_features: pd.DataFrame) -> pd.DataFrame:
    planes = plane_features.copy()
    planes["plane_index"] = planes["plane_index"].astype(int)
    planes = planes.sort_values(["chain", "res_i", "res_j", "plane_index"]).copy()
    planes["box_resi"] = planes.groupby("chain").cumcount() + 1
    box_resi_by_plane = planes.set_index("plane_index")["box_resi"].to_dict()
    pairs = pairs.copy()
    pairs["box_resi_a"] = pairs["plane_index_a"].astype(int).map(box_resi_by_plane)
    pairs["box_resi_b"] = pairs["plane_index_b"].astype(int).map(box_resi_by_plane)
    return pairs


def selection_terms_for_boxes(rows: pd.DataFrame, suffix: str) -> list[str]:
    terms = set()
    for row in rows.itertuples():
        chain = getattr(row, f"chain_{suffix}")
        box_resi = getattr(row, f"box_resi_{suffix}")
        if pd.isna(box_resi):
            continue
        terms.add(f"(chain {chain} and resi {int(box_resi)})")
    return sorted(terms)


def selection_terms_for_model_residues(rows: pd.DataFrame, suffix: str) -> list[str]:
    terms = set()
    for row in rows.itertuples():
        chain = getattr(row, f"chain_{suffix}")
        for residue_field in [f"res_i_{suffix}", f"res_j_{suffix}"]:
            residue = getattr(row, residue_field)
            if pd.isna(residue):
                continue
            terms.add(f"(chain {chain} and resi {int(float(residue))})")
    return sorted(terms)


def write_chunked_selection(handle, object_name: str, selection_name: str, terms: list[str], chunk_size: int = 40) -> None:
    if not terms:
        handle.write(f"# No terms available for {selection_name}.\n")
        handle.write(f"select {selection_name}, none\n")
        return
    chunk_names = []
    for index in range(0, len(terms), chunk_size):
        chunk_name = f"{selection_name}_{index // chunk_size + 1}"
        chunk_names.append(chunk_name)
        handle.write(f"select {chunk_name}, {object_name} and ({' or '.join(terms[index:index + chunk_size])})\n")
    handle.write(f"select {selection_name}, {' or '.join(chunk_names)}\n")


def write_interface_pml(
    path: Path,
    title: str,
    pairs: pd.DataFrame,
    band: str,
    color: str,
    residue_color: str,
    include_low_low: bool = False,
) -> None:
    rows = pairs[pairs["band_name"] == band].copy()
    low_low = rows[rows["pair_rms_class"] == "low_low"].copy()
    box_terms = selection_terms_for_boxes(rows, "a") + selection_terms_for_boxes(rows, "b")
    residue_terms = selection_terms_for_model_residues(rows, "a") + selection_terms_for_model_residues(rows, "b")
    low_low_box_terms = selection_terms_for_boxes(low_low, "a") + selection_terms_for_boxes(low_low, "b")
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"# {title}\n")
        handle.write("# Assumes actual PDB is loaded as `model` and peptide boxes PDB is loaded as `boxes`.\n")
        handle.write("# Box selections use chain plus chain-local PLN residue number derived from plane_features.csv.\n")
        handle.write("# If your boxes object has different residue numbering, use the companion CSV tables.\n")
        handle.write("hide everything, boxes\n")
        handle.write("show cartoon, model\n")
        handle.write("color gray70, model\n")
        handle.write("show sticks, boxes\n")
        handle.write("color gray80, boxes\n")
        write_chunked_selection(handle, "boxes", f"{band}_candidate_boxes", sorted(set(box_terms)))
        write_chunked_selection(handle, "model", f"{band}_candidate_model_residues", sorted(set(residue_terms)))
        handle.write(f"color {color}, {band}_candidate_boxes\n")
        handle.write(f"color {residue_color}, {band}_candidate_model_residues\n")
        handle.write(f"show sticks, {band}_candidate_model_residues\n")
        if include_low_low:
            write_chunked_selection(handle, "boxes", f"{band}_low_low_boxes", sorted(set(low_low_box_terms)))
            handle.write(f"color yellow, {band}_low_low_boxes\n")
            handle.write(f"# {len(low_low)} {band} candidate rows are low_low.\n")
        handle.write(f"zoom {band}_candidate_boxes\n")


def write_c_vs_d_pml(path: Path, pairs: pd.DataFrame) -> None:
    c_rows = pairs[pairs["band_name"] == "C"].copy()
    d_rows = pairs[pairs["band_name"] == "D"].copy()
    c_box_terms = set(selection_terms_for_boxes(c_rows, "a") + selection_terms_for_boxes(c_rows, "b"))
    d_box_terms = set(selection_terms_for_boxes(d_rows, "a") + selection_terms_for_boxes(d_rows, "b"))
    shared_box_terms = sorted(c_box_terms & d_box_terms)
    c_only_box_terms = sorted(c_box_terms - d_box_terms)
    d_only_box_terms = sorted(d_box_terms - c_box_terms)
    c_residue_terms = sorted(
        set(selection_terms_for_model_residues(c_rows, "a") + selection_terms_for_model_residues(c_rows, "b"))
    )
    d_residue_terms = sorted(
        set(selection_terms_for_model_residues(d_rows, "a") + selection_terms_for_model_residues(d_rows, "b"))
    )
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Full ideal C vs D interface helper.\n")
        handle.write("# Assumes actual PDB is loaded as `model` and peptide boxes PDB is loaded as `boxes`.\n")
        handle.write("hide everything, boxes\n")
        handle.write("show cartoon, model\n")
        handle.write("color gray75, model\n")
        handle.write("show sticks, boxes\n")
        handle.write("color gray85, boxes\n")
        write_chunked_selection(handle, "boxes", "C_only_candidate_boxes", c_only_box_terms)
        write_chunked_selection(handle, "boxes", "D_only_candidate_boxes", d_only_box_terms)
        write_chunked_selection(handle, "boxes", "C_D_shared_candidate_boxes", shared_box_terms)
        write_chunked_selection(handle, "model", "C_candidate_model_residues", c_residue_terms)
        write_chunked_selection(handle, "model", "D_candidate_model_residues", d_residue_terms)
        handle.write("color cyan, C_only_candidate_boxes\n")
        handle.write("color orange, D_only_candidate_boxes\n")
        handle.write("color magenta, C_D_shared_candidate_boxes\n")
        handle.write("color marine, C_candidate_model_residues\n")
        handle.write("color tv_orange, D_candidate_model_residues\n")
        handle.write("show sticks, C_candidate_model_residues or D_candidate_model_residues\n")
        handle.write("zoom C_only_candidate_boxes or D_only_candidate_boxes or C_D_shared_candidate_boxes\n")


def draw_interface_cartoon(path: Path) -> None:
    theta = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, len(CHAIN_ORDER), endpoint=False)
    coords = {chain: np.array([np.cos(angle), np.sin(angle)]) for chain, angle in zip(CHAIN_ORDER, theta)}
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for chain, xy in coords.items():
        circle = plt.Circle(xy, 0.12, color="#f4f4f4", ec="black", lw=1.2, zorder=3)
        ax.add_patch(circle)
        ax.text(xy[0], xy[1], chain, ha="center", va="center", fontsize=13, weight="bold", zorder=4)
    for pair in C_INTERFACES:
        a, b = pair.split("-")
        ax.plot([coords[a][0], coords[b][0]], [coords[a][1], coords[b][1]], color="#1f9ac9", lw=5, alpha=0.85)
        midpoint = (coords[a] + coords[b]) / 2
        ax.text(midpoint[0] * 1.08, midpoint[1] * 1.08, "C", color="#0b6f91", fontsize=12, weight="bold")
    for pair in D_INTERFACES:
        a, b = pair.split("-")
        ax.plot([coords[a][0], coords[b][0]], [coords[a][1], coords[b][1]], color="#e07a23", lw=5, alpha=0.85)
        midpoint = (coords[a] + coords[b]) / 2
        ax.text(midpoint[0] * 1.08, midpoint[1] * 1.08, "D", color="#a64b00", fontsize=12, weight="bold")
    ax.text(-1.25, -1.28, "C interfaces: A-B, C-D, E-F", color="#0b6f91", fontsize=11)
    ax.text(-1.25, -1.42, "D interfaces: A-F, B-C, D-E", color="#a64b00", fontsize=11)
    ax.set_title("Full ideal C/D interface pattern")
    ax.set_aspect("equal")
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.55, 1.35)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def draw_d_register_ladder(path: Path, d_pairs: pd.DataFrame) -> None:
    d_pairs = d_pairs.dropna(subset=["register_offset_i", "register_offset_j"]).copy()
    d_pairs["register_label"] = (
        d_pairs["register_offset_i"].astype(int).astype(str)
        + "/"
        + d_pairs["register_offset_j"].astype(int).astype(str)
    )
    counts = d_pairs.groupby(["chain_pair", "register_label"]).size().reset_index(name="count")
    pivot = counts.pivot(index="register_label", columns="chain_pair", values="count").fillna(0)
    pivot["sort_key"] = [int(label.split("/")[0]) for label in pivot.index]
    pivot = pivot.sort_values("sort_key").drop(columns=["sort_key"])
    fig, ax = plt.subplots(figsize=(7.5, max(5.0, 0.25 * len(pivot))))
    image = ax.imshow(pivot.values, aspect="auto", cmap="Oranges")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_xlabel("D chain-pair interface")
    ax.set_ylabel("register offset i/j")
    ax.set_title("Full ideal D candidate register ladder")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = int(pivot.iat[i, j])
            if value:
                ax.text(j, i, str(value), ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="candidate count")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_interpretation(path: Path, boxes_path: Path) -> None:
    lines = [
        "# Full Ideal C/D Visual Interpretation",
        "",
        "## Load In PyMOL",
        "Load the actual full ideal PDB as object `model`, then load the peptide-box PDB as object `boxes`.",
        "",
        "```pml",
        "load <full_ideal_source_pdb>, model",
        f"load {boxes_path.as_posix()}, boxes",
        "run outputs/cd_candidate_spatial_register/full_ideal_highlight_C_vs_D_interfaces.pml",
        "```",
        "",
        "Useful alternatives:",
        "- `run outputs/cd_candidate_spatial_register/full_ideal_highlight_C_interfaces.pml`",
        "- `run outputs/cd_candidate_spatial_register/full_ideal_highlight_D_interfaces.pml`",
        "",
        "## What To Look For",
        "- C candidates should emphasize the A-B, C-D, and E-F interfaces.",
        "- D candidates should emphasize the complementary A-F, B-C, and D-E interfaces.",
        "- In the D helper, low_low boxes are colored distinctly because D is strongly associated with the low-RMS/flatter side of the alternating peptide-plane pattern.",
        "- Non-candidate boxes are muted gray so the interface pattern can be inspected against the full peptide-box scaffold.",
        "",
        "## Cautious Interpretation",
        "D appears to be a regular low-RMS inter-strand register feature in the full ideal six-chain model. C appears to use a complementary interface set. This visualization is diagnostic: it highlights candidate planes and residues, but the molecular interpretation still depends on confirming the loaded source PDB and box object match this run.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    spatial_dir = args.spatial_dir
    pairs = load_full_ideal_pairs(spatial_dir)
    plane_features = pd.read_csv(args.plane_features)
    pairs = add_box_resi_map(pairs, plane_features)

    write_interface_pml(
        spatial_dir / "full_ideal_highlight_C_interfaces.pml",
        "Full ideal C candidate interfaces",
        pairs,
        "C",
        "cyan",
        "marine",
    )
    write_interface_pml(
        spatial_dir / "full_ideal_highlight_D_interfaces.pml",
        "Full ideal D candidate interfaces",
        pairs,
        "D",
        "orange",
        "tv_orange",
        include_low_low=True,
    )
    write_c_vs_d_pml(spatial_dir / "full_ideal_highlight_C_vs_D_interfaces.pml", pairs)
    draw_interface_cartoon(spatial_dir / "full_ideal_interface_cartoon_summary.png")
    draw_d_register_ladder(
        spatial_dir / "full_ideal_D_register_ladder.png",
        pairs[pairs["band_name"] == "D"],
    )
    boxes_path = (
        Path("outputs/six_strand_first_panel_visual_boxes")
        / FULL_IDEAL_LABEL
        / f"{FULL_IDEAL_LABEL}_boxes.pdb"
    )
    write_interpretation(spatial_dir / "full_ideal_visual_interpretation.md", boxes_path)
    print(f"Wrote full-ideal C/D visualization helpers to {spatial_dir}")


if __name__ == "__main__":
    main()
