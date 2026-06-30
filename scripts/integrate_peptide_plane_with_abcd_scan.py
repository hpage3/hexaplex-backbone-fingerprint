"""Integrate diagnostic peptide-plane descriptors with the A/B/C/D scan table."""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hexaplex_backbone_fingerprint.band_mapping import find_band_candidate_pairs
from hexaplex_backbone_fingerprint.io import write_band_candidate_pairs_csv, write_plane_features_csv
from hexaplex_backbone_fingerprint.pdb_parser import parse_pdb
from hexaplex_backbone_fingerprint.peptide_planes import PeptidePlane, build_peptide_planes


DEFAULT_SCAN_FILE = Path(
    r"C:\Users\hpage3\OneDrive - Georgia Institute of Technology\Documents\GitHub\research"
    r"\hexaplex-formation\outputs\asem_matched_twist_scan_28_32\statistical_analysis\abcd_candidate_scores.csv"
)
DEFAULT_PDB_ROOT = Path(
    r"C:\Users\hpage3\OneDrive - Georgia Institute of Technology\Documents\GitHub\research\hexaplex-formation"
)
DEFAULT_OUTDIR = Path("outputs/abcd_scan_peptide_plane_integration")
SCORE_METRICS = ["A_score", "B_score", "C_score", "D_score", "CD_score", "ABCD_score"]
ERROR_METRICS = ["A_abs_error", "B_abs_error", "C_abs_error", "D_abs_error", "cd_rmsd", "helical_abcd_rmsd"]
PEPTIDE_FEATURES_FOR_GROUPS = [
    "theta_90_130_fraction",
    "theta_100_120_fraction",
    "std_theta_pp",
    "median_rms",
    "rms_alternation_fraction",
    "high_low_rms_gap",
    "D_low_low_fraction",
    "D_top_interface_fraction",
    "D_register_repetition_score",
    "D_structural_organization_score",
    "C_structural_organization_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-file", type=Path, default=DEFAULT_SCAN_FILE)
    parser.add_argument("--pdb-root", type=Path, default=DEFAULT_PDB_ROOT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--limit", type=int, default=None, help="Limit scan rows for debugging.")
    parser.add_argument("--force", action="store_true", help="Recompute cached per-candidate plane features.")
    parser.add_argument("--c-target", type=float, default=5.6)
    parser.add_argument("--d-target", type=float, default=7.3)
    parser.add_argument("--tol", type=float, default=0.25)
    return parser.parse_args()


def safe_model_id(value: object) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def resolve_pdb_path(row: pd.Series, pdb_root: Path) -> Path | None:
    value = row.get("coordinate_file", "")
    if pd.isna(value) or not str(value).strip():
        return None
    path = Path(str(value))
    candidates = [path] if path.is_absolute() else [pdb_root / path, ROOT / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def compute_sign_stabilized_theta(planes_df: pd.DataFrame) -> pd.Series:
    theta = pd.Series(np.nan, index=planes_df.index, dtype=float)
    for _, chain_df in planes_df.sort_values(["chain", "res_i", "res_j", "plane_index"]).groupby("chain"):
        normals = chain_df[["normal_x", "normal_y", "normal_z"]].to_numpy(float)
        centers = chain_df[["center_x", "center_y", "center_z"]].to_numpy(float)
        indices = chain_df.index.to_list()
        previous: float | None = None
        for idx in range(len(chain_df) - 1):
            normal_a = normalize(normals[idx])
            normal_b = normalize(normals[idx + 1])
            magnitude = float(np.degrees(np.arccos(np.clip(np.dot(normal_a, normal_b), -1.0, 1.0))))
            local_axis = normalize(centers[idx + 1] - centers[idx])
            sign_metric = float(np.dot(np.cross(normal_a, normal_b), local_axis))
            raw_signed = magnitude if sign_metric >= 0 else -magnitude
            candidates = [raw_signed, -raw_signed]
            accepted = candidates[0] if previous is None else min(candidates, key=lambda value: abs(value - previous))
            theta.loc[indices[idx]] = accepted
            previous = accepted
    return theta


def plane_features_dataframe(planes: list[PeptidePlane], model_id: str) -> pd.DataFrame:
    rows = []
    for idx, plane in enumerate(planes):
        rows.append(
            {
                "model_label": model_id,
                "plane_index": idx,
                "chain": plane.chain,
                "res_i": plane.res_i,
                "res_j": plane.res_j,
                "resname_i": plane.resname_i,
                "resname_j": plane.resname_j,
                "center_x": plane.center[0],
                "center_y": plane.center[1],
                "center_z": plane.center[2],
                "normal_x": plane.normal[0],
                "normal_y": plane.normal[1],
                "normal_z": plane.normal[2],
                "rms": plane.rms,
                "cno_to_peptide_normal_angle_deg": plane.cno_to_peptide_normal_angle_deg,
                "omega_deviation_from_trans_deg": plane.omega_deviation_from_trans_deg,
            }
        )
    return pd.DataFrame(rows)


def rms_state(rms: float) -> str:
    if rms >= 0.03:
        return "high_rms"
    if rms <= 0.005:
        return "low_rms"
    return "mid_rms"


def add_plane_annotations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["chain", "res_i", "res_j", "plane_index"]).copy()
    df["step_type"] = df["resname_i"].astype(str) + "->" + df["resname_j"].astype(str)
    df["rms_state"] = [rms_state(value) for value in df["rms"]]
    df["within_chain_order"] = df.groupby("chain").cumcount()
    df["within_chain_parity"] = np.where(df["within_chain_order"] % 2 == 0, "even", "odd")
    df["sign_stabilized_preserve_magnitude_theta_pp"] = compute_sign_stabilized_theta(df)
    return df


def fraction(mask: pd.Series) -> float:
    if len(mask) == 0:
        return np.nan
    return float(mask.mean())


def safe_corr(x: pd.Series, y: pd.Series) -> float:
    pair = pd.concat([pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1).dropna()
    if len(pair) < 3 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return np.nan
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))


def top_value(values: pd.Series) -> str:
    clean = values.dropna().astype(str)
    if clean.empty:
        return ""
    return Counter(clean).most_common(1)[0][0]


def rms_alternation_fraction(df: pd.DataFrame) -> float:
    transitions = 0
    alternating = 0
    for _, chain_df in df.sort_values(["chain", "res_i", "res_j", "plane_index"]).groupby("chain"):
        states = chain_df["rms_state"].tolist()
        for left, right in zip(states, states[1:]):
            transitions += 1
            if {left, right} == {"high_rms", "low_rms"}:
                alternating += 1
    return alternating / transitions if transitions else np.nan


def parity_consistency(df: pd.DataFrame) -> float:
    high = df[df["rms_state"] == "high_rms"]
    if high.empty:
        return np.nan
    counts = high["within_chain_parity"].value_counts(normalize=True)
    return float(counts.max())


def peptide_descriptors(planes_df: pd.DataFrame) -> dict[str, object]:
    theta = planes_df["sign_stabilized_preserve_magnitude_theta_pp"].dropna()
    abs_theta = theta.abs()
    rms = planes_df["rms"]
    high = planes_df["rms_state"] == "high_rms"
    low = planes_df["rms_state"] == "low_rms"
    chain_medians = planes_df.groupby("chain")["sign_stabilized_preserve_magnitude_theta_pp"].median()
    chain_means = planes_df.groupby("chain")["sign_stabilized_preserve_magnitude_theta_pp"].mean()
    even = planes_df["within_chain_parity"] == "even"
    odd = planes_df["within_chain_parity"] == "odd"
    high_rms = planes_df[high]
    low_rms = planes_df[low]
    return {
        "n_planes": len(planes_df),
        "n_chains": planes_df["chain"].nunique(),
        "median_abs_theta_pp": abs_theta.median(),
        "mean_abs_theta_pp": abs_theta.mean(),
        "std_theta_pp": theta.std(),
        "theta_signed_median": theta.median(),
        "theta_signed_mean": theta.mean(),
        "theta_90_130_fraction": fraction((abs_theta >= 90.0) & (abs_theta <= 130.0)),
        "theta_100_120_fraction": fraction((abs_theta >= 100.0) & (abs_theta <= 120.0)),
        "theta_chain_median_std": chain_medians.std(),
        "theta_chain_mean_std": chain_means.std(),
        "median_rms": rms.median(),
        "mean_rms": rms.mean(),
        "std_rms": rms.std(),
        "rms_high_fraction": fraction(high),
        "rms_low_fraction": fraction(low),
        "high_low_rms_gap": high_rms["rms"].median() - low_rms["rms"].median() if not high_rms.empty and not low_rms.empty else np.nan,
        "rms_alternation_fraction": rms_alternation_fraction(planes_df),
        "odd_even_rms_gap": planes_df.loc[odd, "rms"].median() - planes_df.loc[even, "rms"].median(),
        "within_chain_high_rms_parity_consistency": parity_consistency(planes_df),
        "cno_angle_median": planes_df["cno_to_peptide_normal_angle_deg"].median(),
        "cno_angle_mean": planes_df["cno_to_peptide_normal_angle_deg"].mean(),
        "cno_angle_std": planes_df["cno_to_peptide_normal_angle_deg"].std(),
        "omega_deviation_median": planes_df["omega_deviation_from_trans_deg"].median(),
        "omega_deviation_mean": planes_df["omega_deviation_from_trans_deg"].mean(),
        "omega_deviation_std": planes_df["omega_deviation_from_trans_deg"].std(),
        "rms_cno_corr": safe_corr(planes_df["rms"], planes_df["cno_to_peptide_normal_angle_deg"]),
        "rms_omega_corr": safe_corr(planes_df["rms"], planes_df["omega_deviation_from_trans_deg"]),
        "top_step_type": top_value(planes_df["step_type"]),
        "GLU_CYP_fraction": fraction(planes_df["step_type"] == "GLU->CYP"),
        "CYP_GLU_fraction": fraction(planes_df["step_type"] == "CYP->GLU"),
        "GLU_ending_step_fraction": fraction(planes_df["resname_j"].astype(str) == "GLU"),
        "high_rms_top_step_type": top_value(high_rms["step_type"]),
        "low_rms_top_step_type": top_value(low_rms["step_type"]),
    }


def entropy_fraction(values: pd.Series) -> float:
    counts = values.dropna().astype(str).value_counts()
    total = counts.sum()
    if total == 0 or len(counts) < 2:
        return 0.0
    probs = counts / total
    return float(-(probs * np.log(probs)).sum() / np.log(len(counts)))


def pair_rms_class(state_a: str, state_b: str) -> str:
    states = {state_a, state_b}
    if "mid_rms" in states:
        return "includes_mid"
    if states == {"high_rms"}:
        return "high_high"
    if states == {"low_rms"}:
        return "low_low"
    return "high_low"


def band_pair_descriptors(planes_df: pd.DataFrame, planes: list[PeptidePlane], c_target: float, d_target: float, tol: float) -> dict[str, object]:
    plane_lookup = planes_df.set_index("plane_index")
    descriptors: dict[str, object] = {}
    for band, target in [("C", c_target), ("D", d_target)]:
        candidates = find_band_candidate_pairs(planes, target, tol, band)
        rows = []
        for candidate in candidates:
            pa = plane_lookup.loc[candidate.plane_index_a]
            pb = plane_lookup.loc[candidate.plane_index_b]
            chain_pair = "-".join(sorted([candidate.chain_a, candidate.chain_b]))
            offset_i = candidate.res_i_b - candidate.res_i_a
            offset_j = candidate.res_j_b - candidate.res_j_a
            rows.append(
                {
                    "same_chain": candidate.same_chain,
                    "chain_pair": chain_pair,
                    "register_offset_i": offset_i,
                    "register_offset_j": offset_j,
                    "pair_rms_class": pair_rms_class(pa["rms_state"], pb["rms_state"]),
                    "pair_mean_rms": np.mean([pa["rms"], pb["rms"]]),
                    "pair_mean_cno": np.mean([pa["cno_to_peptide_normal_angle_deg"], pb["cno_to_peptide_normal_angle_deg"]]),
                    "pair_mean_omega": np.mean([pa["omega_deviation_from_trans_deg"], pb["omega_deviation_from_trans_deg"]]),
                }
            )
        pair_df = pd.DataFrame(rows)
        if pair_df.empty:
            for suffix in [
                "pair_count",
                "cross_chain_fraction",
                "low_low_fraction",
                "high_high_fraction",
                "high_low_fraction",
                "interface_entropy",
                "top_interface_fraction",
                "register_repetition_score",
                "same_register_fraction",
                "mean_pair_rms",
                "mean_pair_cno",
                "mean_pair_omega",
                "structural_organization_score",
            ]:
                descriptors[f"{band}_{suffix}"] = 0 if suffix == "pair_count" else np.nan
            continue
        top_interface_fraction = pair_df["chain_pair"].value_counts(normalize=True).max()
        register_labels = pair_df["register_offset_i"].astype(str) + "/" + pair_df["register_offset_j"].astype(str)
        register_repetition = register_labels.value_counts(normalize=True).max()
        cross_chain_fraction = fraction(~pair_df["same_chain"])
        low_low_fraction = fraction(pair_df["pair_rms_class"] == "low_low")
        descriptors.update(
            {
                f"{band}_pair_count": len(pair_df),
                f"{band}_cross_chain_fraction": cross_chain_fraction,
                f"{band}_low_low_fraction": low_low_fraction,
                f"{band}_high_high_fraction": fraction(pair_df["pair_rms_class"] == "high_high"),
                f"{band}_high_low_fraction": fraction(pair_df["pair_rms_class"] == "high_low"),
                f"{band}_interface_entropy": entropy_fraction(pair_df["chain_pair"]),
                f"{band}_top_interface_fraction": float(top_interface_fraction),
                f"{band}_register_repetition_score": float(register_repetition),
                f"{band}_same_register_fraction": fraction(pair_df["register_offset_i"] == pair_df["register_offset_j"]),
                f"{band}_mean_pair_rms": pair_df["pair_mean_rms"].mean(),
                f"{band}_mean_pair_cno": pair_df["pair_mean_cno"].mean(),
                f"{band}_mean_pair_omega": pair_df["pair_mean_omega"].mean(),
            }
        )
        descriptors[f"{band}_structural_organization_score"] = (
            cross_chain_fraction * low_low_fraction * top_interface_fraction * register_repetition
        )
    return descriptors


def analyze_candidate(row: pd.Series, pdb_path: Path, outdir: Path, args: argparse.Namespace) -> dict[str, object]:
    model_id = str(row["model_id"])
    safe_id = safe_model_id(model_id)
    feature_dir = outdir / "per_candidate_plane_features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    plane_csv = feature_dir / f"{safe_id}_plane_features.csv"
    pair_csv = feature_dir / f"{safe_id}_band_candidate_pairs.csv"
    try:
        resmap = parse_pdb(pdb_path)
        planes = build_peptide_planes(resmap)
        if not planes:
            raise ValueError("No peptide-compatible planes were built from this PDB.")
        write_plane_features_csv(planes, plane_csv, model_label=model_id)
        c_candidates = find_band_candidate_pairs(planes, args.c_target, args.tol, "C")
        d_candidates = find_band_candidate_pairs(planes, args.d_target, args.tol, "D")
        write_band_candidate_pairs_csv(c_candidates + d_candidates, pair_csv, model_label=model_id)
        planes_df = add_plane_annotations(plane_features_dataframe(planes, model_id))
        descriptors = peptide_descriptors(planes_df)
        descriptors.update(band_pair_descriptors(planes_df, planes, args.c_target, args.d_target, args.tol))
        descriptors.update(
            {
                "peptide_plane_success": True,
                "cd_pair_features_success": True,
                "peptide_plane_error": "",
                "plane_features_cache": str(plane_csv),
                "band_candidate_pairs_cache": str(pair_csv),
            }
        )
        return descriptors
    except Exception as exc:  # noqa: BLE001 - report and continue by design.
        return {
            "peptide_plane_success": False,
            "cd_pair_features_success": False,
            "peptide_plane_error": f"{type(exc).__name__}: {exc}",
        }


def rank_spearman(x: pd.Series, y: pd.Series) -> float:
    return safe_corr(x.rank(), y.rank())


def correlation_table(df: pd.DataFrame, feature_columns: list[str], metric_columns: list[str]) -> pd.DataFrame:
    rows = []
    for feature in feature_columns:
        for metric in metric_columns:
            if feature not in df.columns or metric not in df.columns:
                continue
            rows.append(
                {
                    "feature": feature,
                    "metric": metric,
                    "pearson": safe_corr(df[feature], df[metric]),
                    "spearman": rank_spearman(df[feature], df[metric]),
                    "n": int(pd.concat([pd.to_numeric(df[feature], errors="coerce"), pd.to_numeric(df[metric], errors="coerce")], axis=1).dropna().shape[0]),
                }
            )
    return pd.DataFrame(rows).sort_values("spearman", key=lambda s: s.abs(), ascending=False)


def cell_summary(df: pd.DataFrame, feature_columns: list[str], metric_columns: list[str]) -> pd.DataFrame:
    group_cols = [col for col in ["twist_deg", "rise_A", "sidechain_variant"] if col in df.columns]
    agg: dict[str, tuple[str, str]] = {"candidate_count": ("model_id", "count")}
    for col in metric_columns + feature_columns:
        if col in df.columns:
            agg[f"mean_{col}"] = (col, "mean")
            agg[f"median_{col}"] = (col, "median")
    return df.groupby(group_cols, dropna=False).agg(**agg).reset_index()


def classify_cd_cell_groups(cells: pd.DataFrame, metric_col: str = "mean_CD_score") -> pd.DataFrame:
    cells = cells.copy()
    valid = cells[metric_col].dropna()
    if valid.empty:
        cells["cd_cell_group"] = ""
        return cells
    q1, q2 = valid.quantile([1 / 3, 2 / 3])
    cells["cd_cell_group"] = "middle"
    cells.loc[cells[metric_col] <= q1, "cd_cell_group"] = "bottom"
    cells.loc[cells[metric_col] >= q2, "cd_cell_group"] = "top"
    return cells


def group_comparison(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, group_df in cells.groupby("cd_cell_group"):
        if not group:
            continue
        row: dict[str, object] = {"cd_cell_group": group, "cell_count": len(group_df)}
        for feature in PEPTIDE_FEATURES_FOR_GROUPS:
            mean_col = f"mean_{feature}"
            if mean_col in group_df.columns:
                row[f"mean_{feature}"] = group_df[mean_col].mean()
                row[f"median_{feature}"] = group_df[mean_col].median()
        rows.append(row)
    return pd.DataFrame(rows)


def save_scatter(df: pd.DataFrame, x: str, y: str, path: Path, title: str) -> bool:
    if x not in df.columns or y not in df.columns:
        return False
    plot_df = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(plot_df) < 3:
        return False
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(plot_df[x], plot_df[y], s=22, alpha=0.7)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def save_heatmap(corr: pd.DataFrame, path: Path, title: str) -> bool:
    if corr.empty:
        return False
    pivot = corr.pivot(index="feature", columns="metric", values="spearman").fillna(0.0)
    if pivot.empty:
        return False
    fig, ax = plt.subplots(figsize=(8, max(6, 0.22 * len(pivot))))
    image = ax.imshow(pivot.values, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index, fontsize=7)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="Spearman correlation")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def save_cell_twist_plot(cells: pd.DataFrame, path: Path) -> bool:
    required = {"twist_deg", "rise_A", "sidechain_variant", "mean_CD_score"}
    if not required <= set(cells.columns):
        return False
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for variant, group in cells.groupby("sidechain_variant"):
        ax.plot(group["twist_deg"], group["mean_CD_score"], marker="o", linestyle="-", label=str(variant))
    ax.set_xlabel("twist_deg")
    ax.set_ylabel("mean_CD_score")
    ax.set_title("Cell mean CD score by twist/rise/variant")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def save_group_bar(comp: pd.DataFrame, columns: list[str], path: Path, title: str) -> bool:
    available = [col for col in columns if col in comp.columns]
    if comp.empty or not available:
        return False
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(available))
    groups = ["bottom", "middle", "top"]
    width = 0.25
    for idx, group in enumerate(groups):
        row = comp[comp["cd_cell_group"] == group]
        if row.empty:
            continue
        values = [float(row.iloc[0][col]) if pd.notna(row.iloc[0][col]) else np.nan for col in available]
        ax.bar(x + (idx - 1) * width, values, width=width, label=group)
    ax.set_xticks(x, [col.replace("mean_", "") for col in available], rotation=35, ha="right")
    ax.set_title(title)
    ax.legend(title="CD cell group")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def top_correlations(corr: pd.DataFrame, metric: str, n: int = 5) -> str:
    subset = corr[corr["metric"] == metric].dropna(subset=["spearman"]).copy()
    if subset.empty:
        return "none"
    subset["abs_spearman"] = subset["spearman"].abs()
    parts = []
    for row in subset.sort_values("abs_spearman", ascending=False).head(n).itertuples():
        parts.append(f"{row.feature}: rho={row.spearman:.3f}")
    return "; ".join(parts)


def write_report(
    outdir: Path,
    scan_file: Path,
    scan_rows: int,
    mapped_count: int,
    feature_count: int,
    pair_feature_count: int,
    unmapped_examples: pd.DataFrame,
    feature_failures: pd.DataFrame,
    feature_successes: pd.DataFrame,
    individual_corr: pd.DataFrame,
    cell_corr: pd.DataFrame,
    cell_groups: pd.DataFrame,
    plot_status: dict[str, bool],
) -> None:
    lines = [
        "# A/B/C/D Scan Peptide-Plane Integration",
        "",
        "This is a diagnostic, exploratory integration of peptide-like backbone-plane descriptors with the matched Asem 28-32 degree A/B/C/D powder-peak scan. These descriptors are structural summaries, not diffraction simulations, and do not prove experimental correctness.",
        "",
        "## Inputs And Mapping",
        f"- Scan file used: `{scan_file}`",
        f"- Scan rows: {scan_rows}",
        f"- Rows mapped to PDBs: {mapped_count}",
        f"- Rows with successful peptide-plane features: {feature_count}",
        f"- Rows with successful C/D pair organization features: {pair_feature_count}",
        "- Performance metrics used: `A_score`, `B_score`, `C_score`, `D_score`, `CD_score`, `ABCD_score` where higher is better.",
        "- Error columns were retained separately and not mixed with score direction.",
        "",
        "## Unmapped Examples",
    ]
    if unmapped_examples.empty:
        lines.append("All scan rows mapped to PDB paths.")
    else:
        for row in unmapped_examples.head(10).itertuples():
            lines.append(f"- {row.model_id}: {getattr(row, 'coordinate_file', '')}")

    lines.append("")
    lines.append("## Peptide-Plane Feature Failures")
    if feature_failures.empty:
        lines.append("All mapped rows produced peptide-plane features.")
    else:
        if "sidechain_variant" in feature_failures.columns:
            for variant, group in feature_failures.groupby("sidechain_variant", dropna=False):
                lines.append(f"- {variant}: {len(group)} failed rows")
        lines.append("- Example failures:")
        for row in feature_failures.head(8).itertuples():
            lines.append(f"  - {row.model_id}: {getattr(row, 'peptide_plane_error', '')}")

    lines.extend(
        [
            "",
            "## Methods",
            "- Peptide planes were built with the existing backbone fingerprint parser from linked residue pairs.",
            "- `sign_stabilized_preserve_magnitude_theta_pp` is diagnostic: it preserves adjacent-plane angle magnitude and stabilizes sign continuity within each chain.",
            "- RMS alternation is the fraction of adjacent chain-local plane pairs that alternate high/low RMS states.",
            "- C/D organization scores use plane-center candidate pairs near 5.6 and 7.3 Angstrom with the existing tolerance. The score is cross-chain fraction x low-low fraction x top-interface fraction x register-repetition score.",
        ]
    )
    if not feature_successes.empty and "n_chains" in feature_successes.columns:
        chain_counts = feature_successes["n_chains"].value_counts(dropna=False).to_dict()
        chain_text = ", ".join(f"{key}: {value}" for key, value in sorted(chain_counts.items()))
        cross_chain = feature_successes[["C_cross_chain_fraction", "D_cross_chain_fraction"]].apply(
            pd.to_numeric, errors="coerce"
        )
        lines.extend(
            [
                "",
                "## Structural Organization Limitation",
                f"- Chain-count distribution for successful peptide-plane rows: {chain_text}.",
                f"- Median C cross-chain fraction: {cross_chain['C_cross_chain_fraction'].median():.3f}; median D cross-chain fraction: {cross_chain['D_cross_chain_fraction'].median():.3f}.",
                "- In this scan subset, successful peptide-plane PDBs are single-chain structures, so cross-chain interface/register organization cannot be interpreted the same way as in the six-strand first-panel analysis.",
            ]
        )

    lines.extend(
        [
            "",
            "## Individual-Candidate Results",
            f"- Strongest C associations: {top_correlations(individual_corr, 'C_score')}",
            f"- Strongest D associations: {top_correlations(individual_corr, 'D_score')}",
            f"- Strongest C/D associations: {top_correlations(individual_corr, 'CD_score')}",
            f"- Strongest ABCD associations: {top_correlations(individual_corr, 'ABCD_score')}",
            "",
            "## Cell-Level Results",
            f"- Strongest cell C associations: {top_correlations(cell_corr, 'mean_C_score')}",
            f"- Strongest cell D associations: {top_correlations(cell_corr, 'mean_D_score')}",
            f"- Strongest cell C/D associations: {top_correlations(cell_corr, 'mean_CD_score')}",
            f"- Strongest cell ABCD associations: {top_correlations(cell_corr, 'mean_ABCD_score')}",
        ]
    )
    if not cell_groups.empty:
        lines.append("")
        lines.append("## High-Vs-Low C/D Cells")
        for row in cell_groups.itertuples():
            values = []
            for feature in PEPTIDE_FEATURES_FOR_GROUPS:
                column = f"mean_{feature}"
                if hasattr(row, column):
                    value = getattr(row, column)
                    if pd.notna(value):
                        values.append(f"{feature}={value:.3f}")
            lines.append(f"- {row.cd_cell_group} cells ({int(row.cell_count)}): " + "; ".join(values[:8]))

    lines.extend(
        [
            "",
            "## C vs D Distinction",
            "- Treat C and D separately in interpretation. The current peptide-plane work indicates D is more tied to low-RMS/register organization, while C uses complementary interfaces and can be more mixed.",
            "- If individual correlations are weak but cell-level correlations are stronger, that is consistent with the prior A/B/C/D statistical result that cell-level geometry is more informative than candidate-level scores alone.",
            "",
            "## Plots",
        ]
    )
    for name, written in plot_status.items():
        lines.append(f"- {name}: {'written' if written else 'skipped'}")
    (outdir / "abcd_peptide_plane_integration_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    scan = pd.read_csv(args.scan_file)
    if args.limit is not None:
        scan = scan.head(args.limit).copy()

    mapped_paths = []
    descriptors = []
    for _, row in scan.iterrows():
        pdb_path = resolve_pdb_path(row, args.pdb_root)
        mapped_paths.append(str(pdb_path) if pdb_path is not None else "")
        if pdb_path is None:
            descriptors.append({"mapping_success": False, "peptide_plane_success": False, "cd_pair_features_success": False})
            continue
        result = analyze_candidate(row, pdb_path, args.outdir, args)
        result["mapping_success"] = True
        descriptors.append(result)

    joined = scan.copy()
    joined["mapped_pdb_path"] = mapped_paths
    descriptor_df = pd.DataFrame(descriptors)
    joined = pd.concat([joined.reset_index(drop=True), descriptor_df.reset_index(drop=True)], axis=1)
    joined.to_csv(args.outdir / "abcd_scan_with_peptide_plane_features.csv", index=False)

    feature_columns = [
        col
        for col in joined.columns
        if col
        not in set(scan.columns)
        | {"mapped_pdb_path", "mapping_success", "peptide_plane_success", "cd_pair_features_success", "peptide_plane_error", "plane_features_cache", "band_candidate_pairs_cache"}
        and pd.api.types.is_numeric_dtype(joined[col])
    ]
    metric_columns = [col for col in SCORE_METRICS if col in joined.columns]
    individual_corr = correlation_table(joined[joined["peptide_plane_success"] == True], feature_columns, metric_columns)  # noqa: E712
    individual_corr.to_csv(args.outdir / "individual_candidate_correlations.csv", index=False)

    cells = cell_summary(joined[joined["peptide_plane_success"] == True], feature_columns, metric_columns)  # noqa: E712
    cells = classify_cd_cell_groups(cells)
    cells.to_csv(args.outdir / "cell_level_peptide_plane_summary.csv", index=False)
    cell_feature_columns = [col for col in cells.columns if col.startswith("mean_") and col.replace("mean_", "") in feature_columns]
    cell_metric_columns = [f"mean_{col}" for col in metric_columns if f"mean_{col}" in cells.columns]
    cell_corr = correlation_table(cells, cell_feature_columns, cell_metric_columns)
    cell_corr.to_csv(args.outdir / "cell_level_correlations.csv", index=False)
    comparison = group_comparison(cells)
    comparison.to_csv(args.outdir / "cd_cell_group_comparison.csv", index=False)

    plot_status: dict[str, bool] = {}
    plot_status["cd_metric_vs_theta_90_130_fraction.png"] = save_scatter(joined, "theta_90_130_fraction", "CD_score", args.outdir / "cd_metric_vs_theta_90_130_fraction.png", "CD score vs theta 90-130 fraction")
    plot_status["cd_metric_vs_theta_std.png"] = save_scatter(joined, "std_theta_pp", "CD_score", args.outdir / "cd_metric_vs_theta_std.png", "CD score vs theta std")
    plot_status["cd_metric_vs_rms_alternation_fraction.png"] = save_scatter(joined, "rms_alternation_fraction", "CD_score", args.outdir / "cd_metric_vs_rms_alternation_fraction.png", "CD score vs RMS alternation")
    plot_status["d_metric_vs_D_low_low_fraction.png"] = save_scatter(joined, "D_low_low_fraction", "D_score", args.outdir / "d_metric_vs_D_low_low_fraction.png", "D score vs D low-low fraction")
    plot_status["d_metric_vs_D_structural_organization_score.png"] = save_scatter(joined, "D_structural_organization_score", "D_score", args.outdir / "d_metric_vs_D_structural_organization_score.png", "D score vs D organization")
    plot_status["c_metric_vs_C_structural_organization_score.png"] = save_scatter(joined, "C_structural_organization_score", "C_score", args.outdir / "c_metric_vs_C_structural_organization_score.png", "C score vs C organization")
    plot_status["cell_mean_cd_by_twist_rise_variant.png"] = save_cell_twist_plot(cells, args.outdir / "cell_mean_cd_by_twist_rise_variant.png")
    plot_status["cell_cd_metric_vs_theta_90_130_fraction.png"] = save_scatter(cells, "mean_theta_90_130_fraction", "mean_CD_score", args.outdir / "cell_cd_metric_vs_theta_90_130_fraction.png", "Cell CD score vs theta 90-130 fraction")
    plot_status["cell_cd_metric_vs_rms_alternation_fraction.png"] = save_scatter(cells, "mean_rms_alternation_fraction", "mean_CD_score", args.outdir / "cell_cd_metric_vs_rms_alternation_fraction.png", "Cell CD score vs RMS alternation")
    plot_status["cell_d_metric_vs_D_structural_organization_score.png"] = save_scatter(cells, "mean_D_structural_organization_score", "mean_D_score", args.outdir / "cell_d_metric_vs_D_structural_organization_score.png", "Cell D score vs D organization")
    plot_status["peptide_plane_features_by_cd_cell_group.png"] = save_group_bar(
        comparison,
        [f"mean_{col}" for col in ["theta_90_130_fraction", "theta_100_120_fraction", "std_theta_pp", "rms_alternation_fraction", "high_low_rms_gap"]],
        args.outdir / "peptide_plane_features_by_cd_cell_group.png",
        "Peptide-plane features by CD cell group",
    )
    plot_status["structural_organization_by_cd_cell_group.png"] = save_group_bar(
        comparison,
        [f"mean_{col}" for col in ["D_low_low_fraction", "D_top_interface_fraction", "D_register_repetition_score", "D_structural_organization_score", "C_structural_organization_score"]],
        args.outdir / "structural_organization_by_cd_cell_group.png",
        "Structural organization by CD cell group",
    )
    plot_status["individual_correlation_heatmap.png"] = save_heatmap(individual_corr, args.outdir / "individual_correlation_heatmap.png", "Individual candidate Spearman correlations")
    plot_status["cell_level_correlation_heatmap.png"] = save_heatmap(cell_corr, args.outdir / "cell_level_correlation_heatmap.png", "Cell-level Spearman correlations")

    unmapped = joined[joined["mapped_pdb_path"] == ""]
    feature_failures = joined[(joined["mapped_pdb_path"] != "") & (~joined["peptide_plane_success"].fillna(False))]
    write_report(
        args.outdir,
        args.scan_file,
        len(scan),
        int((joined["mapped_pdb_path"] != "").sum()),
        int(joined["peptide_plane_success"].fillna(False).sum()),
        int(joined["cd_pair_features_success"].fillna(False).sum()),
        unmapped,
        feature_failures,
        joined[joined["peptide_plane_success"].fillna(False)],
        individual_corr,
        cell_corr,
        comparison,
        plot_status,
    )

    print(f"Scan file: {args.scan_file}")
    print(f"Rows: {len(scan)}")
    print(f"Mapped PDBs: {int((joined['mapped_pdb_path'] != '').sum())}")
    print(f"Peptide-plane features: {int(joined['peptide_plane_success'].fillna(False).sum())}")
    print(f"C/D organization features: {int(joined['cd_pair_features_success'].fillna(False).sum())}")
    print(f"Output directory: {args.outdir}")


if __name__ == "__main__":
    main()
