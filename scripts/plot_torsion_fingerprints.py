
"""Plot torsion fingerprints from existing six-strand first-panel outputs.

This script is intentionally output-oriented: it reads the numeric peptide-plane
CSV files and the legacy peptide-box adjacent-angle CSV files already produced
for selected models, then writes plots, top-plane tables, PyMOL highlight scripts,
and short markdown summaries.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Iterable

import matplotlib.pyplot as plt

DEFAULT_MODELS = [
    "full_hexaplex_anti_parallel_30deg_ideal_deduped_6chain",
    "central6_loose_initial_0000",
    "pnab_hexaplex_twist30_rise3p38",
]

PLANE_COLUMNS = [
    "model_label",
    "rank_by_rms",
    "rank_by_cno_angle",
    "plane_index",
    "chain",
    "res_i",
    "res_j",
    "rms",
    "cno_to_peptide_normal_angle_deg",
    "cno_centroid_to_peptide_plane_signed_dist",
    "omega_like_deg",
    "omega_deviation_from_trans_deg",
    "included_for",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create torsion fingerprint plots from existing first-panel CSV outputs."
    )
    parser.add_argument(
        "--numeric-root",
        type=Path,
        default=Path("outputs/six_strand_first_panel"),
        help="Directory containing per-model plane_features.csv outputs.",
    )
    parser.add_argument(
        "--visual-root",
        type=Path,
        default=Path("outputs/six_strand_first_panel_visual_boxes"),
        help="Directory containing legacy peptide-box outputs.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("outputs/six_strand_first_panel_torsion_fingerprints"),
        help="Output directory for plots, tables, PML, and summaries.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=DEFAULT_MODELS,
        help="Model labels to process. Defaults to the three requested representative models.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def f(row: dict[str, str], column: str) -> float:
    value = row.get(column, "")
    return float(value) if value not in (None, "") else math.nan


def i(row: dict[str, str], column: str) -> int:
    value = row.get(column, "")
    return int(float(value)) if value not in (None, "") else -1


def safe_median(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return median(finite) if finite else math.nan


def safe_mean(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return mean(finite) if finite else math.nan


def safe_max(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return max(finite) if finite else math.nan


def short_title(label: str, max_len: int = 58) -> str:
    if len(label) <= max_len:
        return label
    return label[: max_len - 3] + "..."


def chain_position_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["chain"]].append(row)

    positioned: list[dict[str, object]] = []
    for chain, chain_rows in sorted(grouped.items()):
        sorted_rows = sorted(chain_rows, key=lambda r: i(r, "plane_index"))
        for pos, row in enumerate(sorted_rows, start=1):
            enriched = dict(row)
            enriched["chain_position"] = pos
            positioned.append(enriched)
    return positioned


def plot_metric_by_chain(
    rows: list[dict[str, object]],
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
    highlight_rows: list[dict[str, object]] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    chains = sorted({str(row["chain"]) for row in rows})
    for chain in chains:
        chain_rows = sorted(
            [row for row in rows if str(row["chain"]) == chain],
            key=lambda row: int(row["chain_position"]),
        )
        x = [int(row["chain_position"]) for row in chain_rows]
        y = [float(row[metric]) for row in chain_rows]
        ax.plot(x, y, marker="o", linewidth=1.4, markersize=3, label=f"Chain {chain}")

    if highlight_rows:
        hx = [int(row["chain_position"]) for row in highlight_rows]
        hy = [float(row[metric]) for row in highlight_rows]
        ax.scatter(hx, hy, s=55, facecolors="none", edgecolors="black", linewidths=1.2, zorder=5)

    ax.set_xlabel("Plane position within chain")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.28)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_theta_by_chain(
    rows: list[dict[str, str]],
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> bool:
    if not rows or metric not in rows[0]:
        return False

    fig, ax = plt.subplots(figsize=(11, 6))
    chains = sorted({row["chain"] for row in rows})
    for chain in chains:
        chain_rows = sorted(
            [row for row in rows if row["chain"] == chain],
            key=lambda row: i(row, "plane_index_A"),
        )
        x = [i(row, "plane_index_A") + 1 for row in chain_rows]
        y = [f(row, metric) for row in chain_rows]
        ax.plot(x, y, marker="o", linewidth=1.4, markersize=3, label=f"Chain {chain}")

    ax.set_xlabel("Adjacent plane pair start position within chain")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.28)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return True


def top_torsion_rows(model_label: str, rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_rms = sorted(rows, key=lambda row: float(row["rms"]), reverse=True)[:20]
    by_cno = sorted(
        rows,
        key=lambda row: float(row["cno_to_peptide_normal_angle_deg"]),
        reverse=True,
    )[:20]
    rms_rank = {int(row["plane_index"]): rank for rank, row in enumerate(by_rms, start=1)}
    cno_rank = {int(row["plane_index"]): rank for rank, row in enumerate(by_cno, start=1)}

    combined = {int(row["plane_index"]): row for row in by_rms + by_cno}
    ordered = sorted(
        combined.values(),
        key=lambda row: (
            min(rms_rank.get(int(row["plane_index"]), 999), cno_rank.get(int(row["plane_index"]), 999)),
            int(row["plane_index"]),
        ),
    )

    output_rows: list[dict[str, object]] = []
    for row in ordered:
        plane_index = int(row["plane_index"])
        in_rms = plane_index in rms_rank
        in_cno = plane_index in cno_rank
        included_for = "both" if in_rms and in_cno else "top_rms" if in_rms else "top_cno_angle"
        output_rows.append(
            {
                "model_label": model_label,
                "rank_by_rms": rms_rank.get(plane_index, ""),
                "rank_by_cno_angle": cno_rank.get(plane_index, ""),
                "plane_index": plane_index,
                "chain": row["chain"],
                "res_i": row["res_i"],
                "res_j": row["res_j"],
                "rms": row["rms"],
                "cno_to_peptide_normal_angle_deg": row["cno_to_peptide_normal_angle_deg"],
                "cno_centroid_to_peptide_plane_signed_dist": row["cno_centroid_to_peptide_plane_signed_dist"],
                "omega_like_deg": row["omega_like_deg"],
                "omega_deviation_from_trans_deg": row["omega_deviation_from_trans_deg"],
                "included_for": included_for,
            }
        )
    return output_rows


def write_pymol_highlight_script(path: Path, top_rows: list[dict[str, object]]) -> None:
    rms_rows = [row for row in top_rows if row["included_for"] in ("top_rms", "both")]
    cno_rows = [row for row in top_rows if row["included_for"] in ("top_cno_angle", "both")]

    def selection(name: str, rows: list[dict[str, object]]) -> list[str]:
        clauses = []
        for row in rows:
            chain = row["chain"]
            # Legacy boxes preserve the source residue ID for each PLN
            # box. Numeric plane_index is zero-based globally, so chain +
            # res_i is the reliable box residue selector.
            box_resi = int(row["res_i"])
            clauses.append(f"(boxes and resn PLN and chain {chain} and resi {box_resi})")
        if not clauses:
            return [f"select {name}, none"]
        lines = [f"select {name}, " + clauses[0]]
        for clause in clauses[1:]:
            lines.append(f"select {name}, {name} or {clause}")
        return lines

    lines = [
        "# Highlight peptide-plane box fingerprints with high torsion metrics.",
        "# Assumes the original structure is loaded as object 'model'.",
        "# Assumes the visual peptide-plane boxes are loaded as object 'boxes'.",
        "hide everything, boxes",
        "show sticks, boxes",
        "color gray70, boxes",
        "set stick_radius, 0.10, boxes",
        "",
        "# Top RMS/torsion planes: red",
        *selection("top_rms_planes", rms_rows),
        "color red, top_rms_planes",
        "set stick_radius, 0.26, top_rms_planes",
        "",
        "# Top CNO torsion planes: magenta; overlap with RMS remains visually bright.",
        *selection("top_cno_planes", cno_rows),
        "color magenta, top_cno_planes",
        "set stick_radius, 0.22, top_cno_planes",
        "",
        "# Optional view helpers",
        "show cartoon, model",
        "color gray85, model",
        "zoom boxes",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def localization_note(values: list[float], metric_name: str) -> str:
    finite = sorted([v for v in values if math.isfinite(v)])
    if not finite:
        return f"No finite {metric_name} values were available."
    med = median(finite)
    mx = finite[-1]
    if med == 0:
        ratio = math.inf if mx > 0 else 1.0
    else:
        ratio = mx / med
    if ratio >= 8:
        return f"{metric_name} appears localized: the maximum is much larger than the median."
    if ratio >= 3:
        return f"{metric_name} shows a few elevated planes above a quieter background."
    return f"{metric_name} appears broadly distributed rather than dominated by a single outlier."


def write_model_summary(
    path: Path,
    label: str,
    rows: list[dict[str, object]],
    top_rows: list[dict[str, object]],
    theta_generated: bool,
) -> dict[str, float]:
    rms_values = [float(row["rms"]) for row in rows]
    cno_values = [float(row["cno_to_peptide_normal_angle_deg"]) for row in rows]
    omega_values = [float(row["omega_deviation_from_trans_deg"]) for row in rows]

    top_rms = sorted(rows, key=lambda row: float(row["rms"]), reverse=True)[:5]
    top_cno = sorted(
        rows,
        key=lambda row: float(row["cno_to_peptide_normal_angle_deg"]),
        reverse=True,
    )[:5]

    def bullet(row: dict[str, object], metric: str) -> str:
        return (
            f"- plane {row['plane_index']} chain {row['chain']} "
            f"res {row['res_i']}-{row['res_j']}: {metric}={float(row[metric]):.6g}"
        )

    lines = [
        f"# Torsion fingerprint summary: {label}",
        "",
        f"- Median RMS: {safe_median(rms_values):.6g}",
        f"- Max RMS: {safe_max(rms_values):.6g}",
        f"- Median CNO angle: {safe_median(cno_values):.6g} deg",
        f"- Max CNO angle: {safe_max(cno_values):.6g} deg",
        f"- Median omega deviation: {safe_median(omega_values):.6g} deg",
        f"- Max omega deviation: {safe_max(omega_values):.6g} deg",
        f"- Theta fingerprints generated: {'yes' if theta_generated else 'no'}",
        "",
        "## Top 5 RMS Planes",
        *[bullet(row, "rms") for row in top_rms],
        "",
        "## Top 5 CNO-Angle Planes",
        *[bullet(row, "cno_to_peptide_normal_angle_deg") for row in top_cno],
        "",
        "## Distribution Note",
        localization_note(rms_values, "RMS/torsion"),
        localization_note(cno_values, "CNO torsion"),
        "",
        "## PyMOL",
        "Open the legacy box PDB as object `boxes`, the original structure as object `model`, then run `highlight_top_torsion_planes.pml`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "median_rms": safe_median(rms_values),
        "max_rms": safe_max(rms_values),
        "median_cno": safe_median(cno_values),
        "max_cno": safe_max(cno_values),
        "median_omega": safe_median(omega_values),
        "max_omega": safe_max(omega_values),
    }


def process_model(label: str, numeric_root: Path, visual_root: Path, out_root: Path) -> dict[str, object]:
    model_outdir = out_root / label
    model_outdir.mkdir(parents=True, exist_ok=True)

    plane_path = numeric_root / label / "plane_features.csv"
    if not plane_path.exists():
        raise FileNotFoundError(f"Missing plane features for {label}: {plane_path}")

    rows = chain_position_rows(read_csv(plane_path))
    top_rms_rows = sorted(rows, key=lambda row: float(row["rms"]), reverse=True)[:10]
    top_cno_rows = sorted(
        rows,
        key=lambda row: float(row["cno_to_peptide_normal_angle_deg"]),
        reverse=True,
    )[:10]

    plot_metric_by_chain(
        rows,
        "rms",
        "Plane RMS (A)",
        f"RMS by chain: {short_title(label)}",
        model_outdir / "rms_by_chain.png",
        top_rms_rows,
    )
    plot_metric_by_chain(
        rows,
        "cno_to_peptide_normal_angle_deg",
        "CNO-to-peptide normal angle (deg)",
        f"CNO angle by chain: {short_title(label)}",
        model_outdir / "cno_angle_by_chain.png",
        top_cno_rows,
    )
    plot_metric_by_chain(
        rows,
        "omega_deviation_from_trans_deg",
        "Omega-like deviation from trans (deg)",
        f"Omega-like deviation by chain: {short_title(label)}",
        model_outdir / "omega_deviation_by_chain.png",
    )

    theta_path = visual_root / label / f"{label}_boxes_adjacent_angles.csv"
    theta_generated = False
    if theta_path.exists():
        theta_rows = read_csv(theta_path)
        theta_unsigned = plot_theta_by_chain(
            theta_rows,
            "angle_unsigned_deg",
            "Adjacent plane angle, unsigned (deg)",
            f"Theta unsigned by chain: {short_title(label)}",
            model_outdir / "theta_unsigned_by_chain.png",
        )
        theta_metric = "angle_signed_deg" if theta_rows and "angle_signed_deg" in theta_rows[0] else "dihedral_deg"
        theta_signed = plot_theta_by_chain(
            theta_rows,
            theta_metric,
            f"Adjacent plane {theta_metric.replace('_', ' ')} (deg)",
            f"Theta signed by chain: {short_title(label)}",
            model_outdir / "theta_signed_by_chain.png",
        )
        theta_generated = theta_unsigned and theta_signed

    top_rows = top_torsion_rows(label, rows)
    write_csv(model_outdir / "top_torsion_planes.csv", top_rows, PLANE_COLUMNS)
    write_pymol_highlight_script(model_outdir / "highlight_top_torsion_planes.pml", top_rows)
    stats = write_model_summary(
        model_outdir / "torsion_fingerprint_summary.md",
        label,
        rows,
        top_rows,
        theta_generated,
    )

    return {
        "label": label,
        "outdir": model_outdir,
        "theta_generated": theta_generated,
        **stats,
    }


def write_overview(path: Path, results: list[dict[str, object]]) -> None:
    def highest(metric: str) -> dict[str, object]:
        return max(results, key=lambda row: float(row[metric]))

    lines = [
        "# Six-strand torsion fingerprint overview",
        "",
        "This overview is generated from existing peptide-plane and legacy peptide-box outputs. It does not rerun the structural analysis.",
        "",
        "## Models processed",
    ]
    for result in results:
        label = result["label"]
        outdir = result["outdir"]
        lines.extend(
            [
                f"- `{label}`",
                f"  - plots/tables: `{outdir}`",
                f"  - theta fingerprints generated: {'yes' if result['theta_generated'] else 'no'}",
            ]
        )

    highest_rms = highest("max_rms")
    highest_cno = highest("max_cno")
    highest_omega = highest("max_omega")
    lines.extend(
        [
            "",
            "## Highest observed distortions",
            f"- Highest RMS model: `{highest_rms['label']}` max RMS {float(highest_rms['max_rms']):.6g}",
            f"- Highest CNO-angle model: `{highest_cno['label']}` max CNO angle {float(highest_cno['max_cno']):.6g} deg",
            f"- Highest omega-deviation model: `{highest_omega['label']}` max omega deviation {float(highest_omega['max_omega']):.6g} deg",
            "",
            "## Plot files per model",
            "Each model directory contains `rms_by_chain.png`, `cno_angle_by_chain.png`, `omega_deviation_by_chain.png`, `theta_unsigned_by_chain.png`, and `theta_signed_by_chain.png` when the adjacent-angle CSV was available.",
            "",
            "## PyMOL highlight files",
            "Each model directory contains `highlight_top_torsion_planes.pml`. Load the original model as `model`, load the legacy `*_boxes.pdb` as `boxes`, then run the PML script.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    results = [process_model(label, args.numeric_root, args.visual_root, args.outdir) for label in args.models]
    write_overview(args.outdir / "torsion_fingerprint_overview.md", results)
    print(f"Processed {len(results)} models")
    for result in results:
        print(f"- {result['label']}: {result['outdir']}")


if __name__ == "__main__":
    main()
