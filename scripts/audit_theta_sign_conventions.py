"""Audit peptide-plane theta sign conventions without changing production code."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hexaplex_backbone_fingerprint.geometry import fit_plane, normalize  # noqa: E402
from hexaplex_backbone_fingerprint.pdb_parser import (  # noqa: E402
    Residue,
    ResidueKey,
    is_peptide_linked,
    iter_residue_pairs,
    parse_pdb,
)


DEFAULT_MODELS = {
    "full_hexaplex_anti_parallel_30deg_ideal_deduped_6chain": (
        r"C:\Users\hpage3\OneDrive - Georgia Institute of Technology\Documents\GitHub\research"
        r"\hexaplex-formation\outputs\intermediates\ai_candidate_inputs"
        r"\full_hexaplex_anti_parallel_30deg_ideal_deduped_6chain.pdb"
    ),
    "pnab_hexaplex_twist30_rise3p38": (
        r"C:\Users\hpage3\OneDrive - Georgia Institute of Technology\Documents\GitHub\research"
        r"\hexaplex-formation\outputs\nick_handoff\pnab_hexaplex_twist30_rise3p38.pdb"
    ),
    "central6_loose_initial_0000": (
        r"C:\Users\hpage3\OneDrive - Georgia Institute of Technology\Documents\GitHub\research"
        r"\hexaplex-formation\outputs\seed_formation\ensembles\central6_loose_initial_0000.pdb"
    ),
}

CONTROL_IDS = ["1AL1", "1TEN", "1UBQ"]
SEARCH_ROOTS = [
    Path(r"C:\Users\hpage3\research"),
    Path(r"C:\Users\hpage3\OneDrive - Georgia Institute of Technology\Documents\GitHub\research"),
    Path(r"C:\Users\hpage3\OneDrive - Georgia Institute of Technology\Hud Lab"),
    Path(r"C:\Users\hpage3\OneDrive - Georgia Institute of Technology\Documents\GitHub"),
]

METHOD_COLUMNS = [
    "current_signed",
    "unsigned",
    "continuity_signed",
    "backbone_axis_signed",
    "continuity_backbone_signed",
    "continuity_sign_only_preserve_magnitude",
]


@dataclass
class AuditPlane:
    index: int
    chain: str
    res_i: int
    res_j: int
    normal: np.ndarray
    center: np.ndarray
    u_axis: np.ndarray
    backbone_axis: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit theta sign conventions on PDB peptide planes.")
    parser.add_argument("--outdir", type=Path, default=Path("outputs/theta_sign_audit"))
    parser.add_argument(
        "--pdb",
        nargs=2,
        action="append",
        metavar=("LABEL", "PATH"),
        help="Additional or replacement model to audit. May be repeated.",
    )
    parser.add_argument("--skip-defaults", action="store_true", help="Only audit --pdb inputs.")
    return parser.parse_args()


def safe_normalize(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    try:
        return normalize(vector)
    except ValueError:
        if fallback is None:
            return np.array([1.0, 0.0, 0.0], dtype=float)
        return normalize(fallback)


def ortho_axes(ca_i: np.ndarray, o_i: np.ndarray, ca_j: np.ndarray, normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = safe_normalize(normal)
    u_raw = ca_j - ca_i
    u = safe_normalize(u_raw - np.dot(u_raw, n) * n, np.array([1.0, 0.0, 0.0]))
    v_raw = o_i - ca_i
    v = v_raw - np.dot(v_raw, n) * n
    v = v - np.dot(v, u) * u
    v = safe_normalize(v, np.cross(n, u))
    return u, v


def build_audit_planes(pdb_path: Path) -> list[AuditPlane]:
    resmap = parse_pdb(pdb_path)
    planes: list[AuditPlane] = []
    for key_i, residue_i, key_j, residue_j in iter_residue_pairs(resmap):
        if key_i[0] != key_j[0] or not is_peptide_linked(residue_i, residue_j):
            continue
        plane = build_audit_plane(len(planes), key_i, residue_i, key_j, residue_j)
        if plane is not None:
            planes.append(plane)
    return planes


def build_audit_plane(
    index: int,
    key_i: ResidueKey,
    residue_i: Residue,
    key_j: ResidueKey,
    residue_j: Residue,
) -> AuditPlane | None:
    required = [residue_i.get("CA"), residue_i.get("C"), residue_i.get("O"), residue_j.get("N"), residue_j.get("CA")]
    if any(atom is None for atom in required):
        return None

    hn_atom = residue_j.get("HN") or residue_j.get("H")
    atoms = required + ([hn_atom] if hn_atom is not None else [])
    points = np.array([atom.coord for atom in atoms if atom is not None], dtype=float)
    center, normal, _ = fit_plane(points)

    ca_i = np.array(residue_i["CA"].coord, dtype=float)
    o_i = np.array(residue_i["O"].coord, dtype=float)
    ca_j = np.array(residue_j["CA"].coord, dtype=float)
    c_i = np.array(residue_i["C"].coord, dtype=float)
    n_j = np.array(residue_j["N"].coord, dtype=float)

    u_axis, v_axis = ortho_axes(ca_i, o_i, ca_j, normal)
    oriented_normal = safe_normalize(normal)
    handed_normal = safe_normalize(np.cross(u_axis, v_axis), oriented_normal)
    if float(np.dot(oriented_normal, handed_normal)) < 0.0:
        oriented_normal = -oriented_normal

    return AuditPlane(
        index=index,
        chain=key_i[0],
        res_i=key_i[1],
        res_j=key_j[1],
        normal=oriented_normal,
        center=center,
        u_axis=u_axis,
        backbone_axis=safe_normalize(n_j - c_i, ca_j - ca_i),
    )


def legacy_signed_angle(n1: np.ndarray, n2: np.ndarray, reference_axis: np.ndarray) -> float:
    n1_unit = safe_normalize(n1)
    n2_unit = safe_normalize(n2)
    dot_product = float(np.clip(np.dot(n1_unit, n2_unit), -1.0, 1.0))
    cross_product = np.cross(n1_unit, n2_unit)
    sign_indicator = float(np.dot(cross_product, safe_normalize(reference_axis)))
    sin_component = float(np.linalg.norm(cross_product))
    cos_component = dot_product
    if dot_product < 0:
        cos_component = -abs(cos_component)
    if sign_indicator < 0:
        sin_component = -sin_component
    return wrap_angle(float(np.degrees(np.arctan2(sin_component, cos_component))))


def unsigned_normal_angle(n1: np.ndarray, n2: np.ndarray) -> float:
    dot = abs(float(np.clip(np.dot(safe_normalize(n1), safe_normalize(n2)), -1.0, 1.0)))
    return float(np.degrees(np.arccos(dot)))


def signed_about_axis(n1: np.ndarray, n2: np.ndarray, axis: np.ndarray) -> float:
    n1_unit = safe_normalize(n1)
    n2_unit = safe_normalize(n2)
    axis_unit = safe_normalize(axis)
    sin_component = float(np.dot(np.cross(n1_unit, n2_unit), axis_unit))
    cos_component = float(np.clip(np.dot(n1_unit, n2_unit), -1.0, 1.0))
    return wrap_angle(float(np.degrees(np.arctan2(sin_component, cos_component))))


def sign_only_preserve_magnitude(n1: np.ndarray, n2: np.ndarray, axis: np.ndarray) -> float:
    """Use the raw 0..180 normal angle magnitude, assigning sign separately."""
    n1_unit = safe_normalize(n1)
    n2_unit = safe_normalize(n2)
    axis_unit = safe_normalize(axis)
    dot = float(np.clip(np.dot(n1_unit, n2_unit), -1.0, 1.0))
    magnitude = float(np.degrees(np.arccos(dot)))
    sign_indicator = float(np.dot(np.cross(n1_unit, n2_unit), axis_unit))
    sign = -1.0 if sign_indicator < 0 else 1.0
    return sign * magnitude


def wrap_angle(angle: float) -> float:
    wrapped = ((angle + 180.0) % 360.0) - 180.0
    return 180.0 if wrapped == -180.0 and angle > 0 else wrapped


def continuity_normals(planes: list[AuditPlane]) -> tuple[dict[int, np.ndarray], set[int]]:
    by_chain: dict[str, list[AuditPlane]] = defaultdict(list)
    for plane in planes:
        by_chain[plane.chain].append(plane)

    normals: dict[int, np.ndarray] = {}
    flipped: set[int] = set()
    for chain_planes in by_chain.values():
        previous: np.ndarray | None = None
        for plane in sorted(chain_planes, key=lambda p: (p.res_i, p.res_j, p.index)):
            current = plane.normal.copy()
            if previous is not None and float(np.dot(previous, current)) < 0.0:
                current = -current
                flipped.add(plane.index)
            normals[plane.index] = current
            previous = current
    return normals, flipped


def adjacent_pairs(planes: list[AuditPlane]) -> list[tuple[AuditPlane, AuditPlane]]:
    pairs: list[tuple[AuditPlane, AuditPlane]] = []
    by_chain: dict[str, list[AuditPlane]] = defaultdict(list)
    for plane in planes:
        by_chain[plane.chain].append(plane)
    for chain_planes in by_chain.values():
        ordered = sorted(chain_planes, key=lambda p: (p.res_i, p.res_j, p.index))
        for a, b in zip(ordered, ordered[1:]):
            if a.res_j == b.res_i:
                pairs.append((a, b))
    return pairs


def audit_model(label: str, pdb_path: Path, outdir: Path) -> dict[str, object]:
    planes = build_audit_planes(pdb_path)
    continuity, flipped = continuity_normals(planes)
    rows: list[dict[str, object]] = []
    for plane_a, plane_b in adjacent_pairs(planes):
        n_a = plane_a.normal
        n_b = plane_b.normal
        cn_a = continuity[plane_a.index]
        cn_b = continuity[plane_b.index]
        local_axis = safe_normalize(plane_b.center - plane_a.center, plane_a.backbone_axis)
        raw_dot = float(np.dot(safe_normalize(n_a), safe_normalize(n_b)))
        continuity_dot = float(np.dot(safe_normalize(cn_a), safe_normalize(cn_b)))
        rows.append(
            {
                "model_label": label,
                "chain": plane_a.chain,
                "plane_index_A": plane_a.index,
                "plane_index_B": plane_b.index,
                "res_i_A": plane_a.res_i,
                "res_j_A": plane_a.res_j,
                "res_i_B": plane_b.res_i,
                "res_j_B": plane_b.res_j,
                "current_signed": legacy_signed_angle(n_a, n_b, plane_a.u_axis),
                "unsigned": unsigned_normal_angle(n_a, n_b),
                "continuity_signed": legacy_signed_angle(cn_a, cn_b, plane_a.u_axis),
                "backbone_axis_signed": signed_about_axis(n_a, n_b, local_axis),
                "continuity_backbone_signed": signed_about_axis(cn_a, cn_b, local_axis),
                "continuity_sign_only_preserve_magnitude": sign_only_preserve_magnitude(n_a, n_b, local_axis),
                "normal_dot_raw": raw_dot,
                "normal_dot_after_continuity": continuity_dot,
                "was_normal_flipped_B": plane_b.index in flipped,
                "local_axis_x": float(local_axis[0]),
                "local_axis_y": float(local_axis[1]),
                "local_axis_z": float(local_axis[2]),
            }
        )

    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"{label}_theta_sign_audit.csv"
    write_rows(csv_path, rows)
    summary = summarize_rows(label, pdb_path, rows)
    summary_path = outdir / f"{label}_theta_sign_audit_summary.md"
    summary_path.write_text(render_model_summary(label, pdb_path, summary), encoding="utf-8")
    plot_path = outdir / f"{label}_theta_sign_comparison.png"
    plot_model(label, rows, plot_path)
    return {
        "label": label,
        "pdb_path": str(pdb_path),
        "csv_path": str(csv_path),
        "summary_path": str(summary_path),
        "plot_path": str(plot_path),
        **summary,
    }


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "model_label",
        "chain",
        "plane_index_A",
        "plane_index_B",
        "res_i_A",
        "res_j_A",
        "res_i_B",
        "res_j_B",
        *METHOD_COLUMNS,
        "normal_dot_raw",
        "normal_dot_after_continuity",
        "was_normal_flipped_B",
        "local_axis_x",
        "local_axis_y",
        "local_axis_z",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sign_change_count(rows: list[dict[str, object]], method: str) -> int:
    count = 0
    by_chain: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_chain[str(row["chain"])].append(row)
    for chain_rows in by_chain.values():
        ordered = sorted(chain_rows, key=lambda r: (int(r["res_i_A"]), int(r["plane_index_A"])))
        previous_sign = 0
        for row in ordered:
            value = float(row[method])
            sign = 1 if value > 0 else -1 if value < 0 else 0
            if previous_sign and sign and sign != previous_sign:
                count += 1
            if sign:
                previous_sign = sign
    return count


def abrupt_jump_count(rows: list[dict[str, object]], method: str, threshold: float = 150.0) -> int:
    count = 0
    by_chain: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_chain[str(row["chain"])].append(row)
    for chain_rows in by_chain.values():
        ordered = sorted(chain_rows, key=lambda r: (int(r["res_i_A"]), int(r["plane_index_A"])))
        values = [float(row[method]) for row in ordered]
        count += sum(1 for a, b in zip(values, values[1:]) if abs(b - a) > threshold)
    return count


def summarize_rows(label: str, pdb_path: Path, rows: list[dict[str, object]]) -> dict[str, object]:
    method_stats = {}
    for method in METHOD_COLUMNS:
        values = np.array([float(row[method]) for row in rows], dtype=float)
        method_stats[method] = {
            "min": float(np.min(values)) if len(values) else math.nan,
            "median": float(np.median(values)) if len(values) else math.nan,
            "max": float(np.max(values)) if len(values) else math.nan,
            "sign_changes": sign_change_count(rows, method),
            "abrupt_jumps_gt_150": abrupt_jump_count(rows, method),
        }
    flip_count = sum(1 for row in rows if row["was_normal_flipped_B"])
    recommendation = recommend_method(method_stats)
    return {
        "pair_count": len(rows),
        "flip_count": flip_count,
        "flip_fraction": (flip_count / len(rows)) if rows else 0.0,
        "method_stats": method_stats,
        "recommendation": recommendation,
    }


def recommend_method(method_stats: dict[str, dict[str, float]]) -> str:
    candidates = [
        "continuity_sign_only_preserve_magnitude",
        "continuity_backbone_signed",
        "backbone_axis_signed",
        "continuity_signed",
        "current_signed",
    ]
    return min(
        candidates,
        key=lambda method: (
            method_stats[method]["abrupt_jumps_gt_150"],
            method_stats[method]["sign_changes"],
        ),
    )


def render_model_summary(label: str, pdb_path: Path, summary: dict[str, object]) -> str:
    lines = [
        f"# Theta sign audit summary: {label}",
        "",
        f"- Source PDB: `{pdb_path}`",
        f"- Adjacent plane pairs: {summary['pair_count']}",
        f"- Normal flips required by continuity: {summary['flip_count']} ({summary['flip_fraction']:.3f})",
        f"- Most stable diagnostic convention in this audit: `{summary['recommendation']}`",
        "",
        "## Method Statistics",
        "",
        "| method | min | median | max | sign changes | jumps >150 deg |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, stats in summary["method_stats"].items():
        lines.append(
            f"| `{method}` | {stats['min']:.3f} | {stats['median']:.3f} | {stats['max']:.3f} | "
            f"{stats['sign_changes']} | {stats['abrupt_jumps_gt_150']} |"
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            f"`{summary['recommendation']}` is the most stable convention in this diagnostic, but it is not yet validated as the manuscript theta-pp definition without controls or the original Loren/Howard implementation.",
        ]
    )
    return "\n".join(lines) + "\n"


def plot_model(label: str, rows: list[dict[str, object]], path: Path, gap: int = 5) -> None:
    serial_x = []
    offset = 0
    last_chain = None
    sorted_rows = sorted(rows, key=lambda r: (str(r["chain"]), int(r["res_i_A"]), int(r["plane_index_A"])))
    for row in sorted_rows:
        if last_chain is not None and row["chain"] != last_chain:
            offset += gap
        serial_x.append(len(serial_x) + offset)
        last_chain = row["chain"]

    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
    plot_methods = [
        "current_signed",
        "continuity_signed",
        "backbone_axis_signed",
        "continuity_backbone_signed",
        "continuity_sign_only_preserve_magnitude",
    ]
    for ax, method in zip(axes, plot_methods):
        ax.plot(serial_x, [float(row[method]) for row in sorted_rows], marker="o", markersize=2.5, linewidth=1.0)
        ax.set_ylabel(method.replace("_", "\n"), fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
        for idx in range(len(sorted_rows) - 1):
            if sorted_rows[idx]["chain"] != sorted_rows[idx + 1]["chain"]:
                ax.axvline(serial_x[idx] + gap / 2, color="red", linestyle="--", alpha=0.35)
    axes[-1].set_xlabel("Adjacent peptide-plane pairs, chains laid out serially")
    fig.suptitle(f"Theta sign convention comparison: {label}")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def find_local_controls() -> tuple[dict[str, str], list[str]]:
    found: dict[str, str] = {}
    missing: list[str] = []
    for control_id in CONTROL_IDS:
        matches: list[Path] = []
        patterns = [f"*{control_id}*.pdb", f"*{control_id.lower()}*.pdb"]
        for root in SEARCH_ROOTS:
            if not root.exists():
                continue
            for pattern in patterns:
                matches.extend(root.rglob(pattern))
        if matches:
            found[control_id] = str(sorted(set(matches), key=lambda p: (len(str(p)), str(p)))[0])
        else:
            missing.append(control_id)
    return found, missing


def write_overview(outdir: Path, results: list[dict[str, object]], controls_found: dict[str, str], controls_missing: list[str]) -> None:
    current = "\n".join(
        [
            "- Legacy visual script orients each SVD normal to match `cross(u_axis, v_axis)` for that plane.",
            "- `angle_unsigned_deg` is `arccos(abs(dot(n1, n2)))`.",
            "- `angle_signed_deg` uses raw adjacent normals and the first plane's `u_axis` as the sign reference.",
            "- The legacy CSV computes a pairwise flipped dot product, but that flipped normal is not used in `angle_signed_deg`.",
            "- No chain-wide normal continuity is enforced before the current signed angle calculation.",
            "- `continuity_backbone_signed` flips adjacent normals to positive dot products before `atan2`, which removes large jumps but folds obtuse angles into acute complements.",
            "- `continuity_sign_only_preserve_magnitude` preserves the raw 0..180 degree inter-plane magnitude and assigns sign separately with the local backbone propagation axis.",
        ]
    )
    lines = [
        "# Theta sign audit overview",
        "",
        "## What Was Inspected",
        "- `legacy/peptide_box/planes_from_backbone_ortho_boxes.py`",
        "- `src/hexaplex_backbone_fingerprint/geometry.py`",
        "- `src/hexaplex_backbone_fingerprint/peptide_planes.py`",
        "",
        "## Current Method Summary",
        current,
        "",
        "## Controls",
        f"- Controls found locally: {', '.join(controls_found) if controls_found else 'none'}",
        f"- Controls missing locally: {', '.join(controls_missing) if controls_missing else 'none'}",
        "- No web download was attempted.",
        "",
        "## Audited Models",
    ]
    for result in results:
        lines.extend(
            [
                f"### {result['label']}",
                f"- PDB: `{result['pdb_path']}`",
                f"- CSV: `{result['csv_path']}`",
                f"- Summary: `{result['summary_path']}`",
                f"- Plot: `{result['plot_path']}`",
                f"- Normal flips required: {result['flip_count']} / {result['pair_count']} ({result['flip_fraction']:.3f})",
                f"- Recommended convention from diagnostic stability: `{result['recommendation']}`",
            ]
        )
        current_jumps = result["method_stats"]["current_signed"]["abrupt_jumps_gt_150"]
        continuity_jumps = result["method_stats"]["continuity_backbone_signed"]["abrupt_jumps_gt_150"]
        preserve_obtuse = sum(
            1
            for value in [
                result["method_stats"]["continuity_sign_only_preserve_magnitude"]["min"],
                result["method_stats"]["continuity_sign_only_preserve_magnitude"]["max"],
            ]
            if abs(value) > 90.0
        )
        lines.append(
            f"- Current signed jumps >150 deg: {current_jumps}; continuity-backbone jumps >150 deg: {continuity_jumps}"
        )
        lines.append(f"- Preserve-magnitude method has obtuse extrema present: {'yes' if preserve_obtuse else 'no'}")
        lines.append("")

    current_total_jumps = sum(r["method_stats"]["current_signed"]["abrupt_jumps_gt_150"] for r in results)
    corrected_total_jumps = sum(r["method_stats"]["continuity_backbone_signed"]["abrupt_jumps_gt_150"] for r in results)
    preserve_total_jumps = sum(
        r["method_stats"]["continuity_sign_only_preserve_magnitude"]["abrupt_jumps_gt_150"] for r in results
    )
    lines.extend(
        [
            "## Interpretation",
            f"- Across audited Hexaplex models, current signed abrupt jumps >150 deg: {current_total_jumps}.",
            f"- Across audited Hexaplex models, continuity-backbone abrupt jumps >150 deg: {corrected_total_jumps}.",
            f"- Across audited Hexaplex models, preserve-magnitude abrupt jumps >150 deg: {preserve_total_jumps}.",
            "- If the current signed plot alternates between positive and negative values where a beta-like segment should remain mostly negative, the absence of chain-wide normal continuity is a plausible contributor.",
            "- The previous continuity/backbone method reduced abrupt jumps by forcing adjacent dot products positive, but that also collapses real obtuse beta-like angles toward acute complements.",
            "- The preserve-magnitude method restores obtuse angle magnitudes, but still requires validation against alpha/beta controls before being treated as the real manuscript theta-pp convention.",
            "",
            "## Recommended Next Step",
            "- Do not keep treating the current legacy `angle_signed_deg` as trustworthy without controls.",
            "- Compare these diagnostics against the original Loren/Howard theta-pp implementation if available.",
            "- For the next diagnostic plot regeneration, include `continuity_sign_only_preserve_magnitude` as the leading candidate because it preserves obtuse angle magnitude while assigning sign from local backbone propagation, but label it diagnostic until controls pass.",
        ]
    )
    (outdir / "theta_sign_audit_overview.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    models: dict[str, str] = {}
    if not args.skip_defaults:
        models.update(DEFAULT_MODELS)
    if args.pdb:
        for label, path in args.pdb:
            models[label] = path

    controls_found, controls_missing = find_local_controls()
    models.update(controls_found)

    results = []
    for label, path_text in models.items():
        pdb_path = Path(path_text)
        if not pdb_path.exists():
            print(f"[warn] missing PDB for {label}: {pdb_path}")
            continue
        print(f"[audit] {label}: {pdb_path}")
        results.append(audit_model(label, pdb_path, args.outdir))

    write_overview(args.outdir, results, controls_found, controls_missing)
    print(f"Wrote theta sign audit outputs to {args.outdir}")
    if controls_missing:
        print(f"Missing local controls: {', '.join(controls_missing)}")


if __name__ == "__main__":
    main()
