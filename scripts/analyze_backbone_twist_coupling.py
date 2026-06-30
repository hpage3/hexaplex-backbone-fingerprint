"""Exploratory coupling of peptide-plane state to local twist/rise geometry."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_MODELS = [
    "full_hexaplex_anti_parallel_30deg_ideal_deduped_6chain",
    "pnab_hexaplex_twist30_rise3p38",
    "hexaplex_base_length_scale_1p00",
    "central6_loose_initial_0000",
]
DEFAULT_INPUT_ROOT = Path("outputs/six_strand_first_panel")
DEFAULT_CD_CSV = Path("outputs/cd_candidates_by_torsion_state/cd_candidate_pair_torsion_state.csv")
DEFAULT_OUTDIR = Path("outputs/backbone_twist_coupling")
FULL_IDEAL_LABEL = "full_hexaplex_anti_parallel_30deg_ideal_deduped_6chain"
NOMINAL_TWIST_DEG = 30.0

CHAIN_COLORS = {
    "A": "#1f77b4",
    "B": "#ff7f0e",
    "C": "#2ca02c",
    "D": "#d62728",
    "E": "#9467bd",
    "F": "#8c564b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze peptide-plane theta/RMS coupling to local twist/rise.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--cd-candidates", type=Path, default=DEFAULT_CD_CSV)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--low-rms-threshold", type=float, default=0.005)
    parser.add_argument("--high-rms-threshold", type=float, default=0.03)
    parser.add_argument("--gap", type=int, default=5)
    return parser.parse_args()


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def fit_model_axis(centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    origin = centers.mean(axis=0)
    _, _, vh = np.linalg.svd(centers - origin, full_matrices=False)
    axis = normalize(vh[0])
    return origin, axis


def orient_axis_for_positive_rise(df: pd.DataFrame, axis: np.ndarray) -> tuple[np.ndarray, float]:
    rises: list[float] = []
    for _, chain_df in df.groupby("chain", sort=True):
        chain_df = chain_df.sort_values(["res_i", "res_j", "plane_index"])
        centers = chain_df[["center_x", "center_y", "center_z"]].to_numpy(float)
        if len(centers) < 2:
            continue
        rises.extend(np.diff(centers, axis=0) @ axis)
    median_rise = float(np.nanmedian(rises)) if rises else np.nan
    if np.isfinite(median_rise) and median_rise < 0:
        axis = -axis
        median_rise = -median_rise
    return axis, median_rise


def axis_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    trial = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(trial, axis))) > 0.9:
        trial = np.array([0.0, 1.0, 0.0])
    basis_x = normalize(trial - np.dot(trial, axis) * axis)
    basis_y = normalize(np.cross(axis, basis_x))
    return basis_x, basis_y


def signed_angle_deg(vec_a: np.ndarray, vec_b: np.ndarray, axis: np.ndarray) -> float:
    a_norm = normalize(vec_a)
    b_norm = normalize(vec_b)
    if np.linalg.norm(a_norm) == 0 or np.linalg.norm(b_norm) == 0:
        return np.nan
    sine = float(np.dot(axis, np.cross(a_norm, b_norm)))
    cosine = float(np.dot(a_norm, b_norm))
    return float(np.degrees(np.arctan2(sine, cosine)))


def classify_rms(rms: float, low_threshold: float, high_threshold: float) -> str:
    if rms <= low_threshold:
        return "low_rms"
    if rms >= high_threshold:
        return "high_rms"
    return "mid_rms"


def compute_theta_for_chain(chain_df: pd.DataFrame) -> pd.Series:
    chain_df = chain_df.sort_values(["res_i", "res_j", "plane_index"])
    normals = chain_df[["normal_x", "normal_y", "normal_z"]].to_numpy(float)
    centers = chain_df[["center_x", "center_y", "center_z"]].to_numpy(float)
    indices = chain_df.index.to_list()
    theta = pd.Series(np.nan, index=chain_df.index, dtype=float)
    previous: float | None = None
    for local_idx in range(len(chain_df) - 1):
        normal_a = normalize(normals[local_idx])
        normal_b = normalize(normals[local_idx + 1])
        magnitude = float(np.degrees(np.arccos(np.clip(np.dot(normal_a, normal_b), -1.0, 1.0))))
        local_axis = normalize(centers[local_idx + 1] - centers[local_idx])
        sign_metric = float(np.dot(np.cross(normal_a, normal_b), local_axis))
        sign = 1.0 if sign_metric >= 0 else -1.0
        candidates = [magnitude * sign, -magnitude * sign]
        if previous is None:
            accepted = candidates[0]
        else:
            accepted = min(candidates, key=lambda value: abs(value - previous))
        theta.loc[indices[local_idx]] = accepted
        previous = accepted
    return theta


def build_cd_plane_annotations(cd_df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    cd_df = cd_df[cd_df["model_label"].isin(models)].copy()
    for row in cd_df.itertuples():
        for suffix in ["a", "b"]:
            plane_index = int(getattr(row, f"plane_index_{suffix}"))
            partner_chain = getattr(row, "chain_b" if suffix == "a" else "chain_a")
            rows.append(
                {
                    "model_label": row.model_label,
                    "plane_index": plane_index,
                    "band_name": row.band_name,
                    "partner_chain": partner_chain,
                    "pair_rms_class": row.pair_rms_class,
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "model_label",
                "plane_index",
                "number_of_C_pairs",
                "number_of_D_pairs",
                "dominant_C_partner_chain",
                "dominant_D_partner_chain",
                "involved_in_C",
                "involved_in_D",
                "D_low_low_involvement",
            ]
        )
    exploded = pd.DataFrame(rows)
    annotations: list[dict[str, object]] = []
    for (model, plane_index), group in exploded.groupby(["model_label", "plane_index"]):
        c_group = group[group["band_name"] == "C"]
        d_group = group[group["band_name"] == "D"]
        annotations.append(
            {
                "model_label": model,
                "plane_index": plane_index,
                "number_of_C_pairs": len(c_group),
                "number_of_D_pairs": len(d_group),
                "dominant_C_partner_chain": dominant_partner(c_group),
                "dominant_D_partner_chain": dominant_partner(d_group),
                "involved_in_C": len(c_group) > 0,
                "involved_in_D": len(d_group) > 0,
                "D_low_low_involvement": bool((d_group["pair_rms_class"] == "low_low").any()),
            }
        )
    return pd.DataFrame(annotations)


def dominant_partner(group: pd.DataFrame) -> str:
    if group.empty:
        return ""
    return Counter(group["partner_chain"].astype(str)).most_common(1)[0][0]


def add_serial_layout(df: pd.DataFrame, gap: int) -> pd.DataFrame:
    df = df.sort_values(["chain", "res_i", "res_j", "plane_index"]).reset_index(drop=True).copy()
    serial_x: list[int] = []
    offset = 0
    previous_chain = None
    for position, row in df.iterrows():
        if previous_chain is not None and row["chain"] != previous_chain:
            offset += gap
        serial_x.append(position + offset)
        previous_chain = row["chain"]
    df["serial_x"] = serial_x
    return df


def analyze_model(
    model: str,
    input_root: Path,
    annotations: pd.DataFrame,
    low_threshold: float,
    high_threshold: float,
    gap: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    plane_path = input_root / model / "plane_features.csv"
    planes = pd.read_csv(plane_path)
    required = {
        "plane_index",
        "chain",
        "res_i",
        "res_j",
        "resname_i",
        "resname_j",
        "center_x",
        "center_y",
        "center_z",
        "normal_x",
        "normal_y",
        "normal_z",
        "rms",
        "cno_to_peptide_normal_angle_deg",
        "omega_deviation_from_trans_deg",
    }
    missing = sorted(required - set(planes.columns))
    if missing:
        raise ValueError(f"{plane_path} is missing required columns: {', '.join(missing)}")

    planes = planes.sort_values(["chain", "res_i", "res_j", "plane_index"]).copy()
    centers = planes[["center_x", "center_y", "center_z"]].to_numpy(float)
    origin, axis = fit_model_axis(centers)
    axis, median_oriented_rise = orient_axis_for_positive_rise(planes, axis)
    basis_x, basis_y = axis_basis(axis)

    planes["step_type"] = planes["resname_i"].astype(str) + "->" + planes["resname_j"].astype(str)
    planes["rms_state"] = [classify_rms(value, low_threshold, high_threshold) for value in planes["rms"]]
    planes["theta_pp_diagnostic_deg"] = np.nan
    for _, chain_df in planes.groupby("chain", sort=True):
        theta = compute_theta_for_chain(chain_df)
        planes.loc[theta.index, "theta_pp_diagnostic_deg"] = theta

    trace_rows: list[pd.DataFrame] = []
    negative_rise_intervals = 0
    total_intervals = 0
    chain_axis_angles: list[float] = []
    for chain, chain_df in planes.groupby("chain", sort=True):
        chain_df = chain_df.sort_values(["res_i", "res_j", "plane_index"]).copy()
        chain_df["within_chain_order"] = np.arange(len(chain_df))
        chain_centers = chain_df[["center_x", "center_y", "center_z"]].to_numpy(float)
        if len(chain_centers) >= 2:
            _, chain_axis = fit_model_axis(chain_centers)
            chain_axis_angles.append(
                float(np.degrees(np.arccos(np.clip(abs(np.dot(chain_axis, axis)), -1.0, 1.0))))
            )
        radial_vectors = chain_centers - origin
        projected_axial = radial_vectors @ axis
        perpendicular = radial_vectors - np.outer(projected_axial, axis)
        chain_df["radial_distance"] = np.linalg.norm(perpendicular, axis=1)
        chain_df["angular_phase_deg"] = np.degrees(
            np.arctan2(perpendicular @ basis_y, perpendicular @ basis_x)
        )
        local_rise = np.full(len(chain_df), np.nan)
        local_twist = np.full(len(chain_df), np.nan)
        for idx in range(len(chain_df) - 1):
            delta = chain_centers[idx + 1] - chain_centers[idx]
            local_rise[idx] = float(np.dot(delta, axis))
            twist = signed_angle_deg(perpendicular[idx], perpendicular[idx + 1], axis)
            local_twist[idx] = twist
            total_intervals += 1
            if np.isfinite(local_rise[idx]) and local_rise[idx] < 0:
                negative_rise_intervals += 1
        chain_df["local_rise"] = local_rise
        chain_df["local_twist_deg"] = local_twist
        chain_df["abs_local_twist_deg"] = np.abs(local_twist)
        chain_df["twist_deviation_from_30_deg"] = np.abs(np.abs(local_twist) - NOMINAL_TWIST_DEG)
        trace_rows.append(chain_df)

    trace = pd.concat(trace_rows, ignore_index=True)
    trace = trace.merge(
        annotations[annotations["model_label"] == model],
        on=["model_label", "plane_index"],
        how="left",
    )
    for column in ["number_of_C_pairs", "number_of_D_pairs"]:
        trace[column] = trace[column].fillna(0).astype(int)
    for column in ["involved_in_C", "involved_in_D", "D_low_low_involvement"]:
        trace[column] = trace[column].fillna(False).astype(bool)
    for column in ["dominant_C_partner_chain", "dominant_D_partner_chain"]:
        trace[column] = trace[column].fillna("")
    trace = add_serial_layout(trace, gap)

    diagnostics = {
        "model_label": model,
        "axis_x": axis[0],
        "axis_y": axis[1],
        "axis_z": axis[2],
        "median_oriented_chain_rise": median_oriented_rise,
        "negative_local_rise_intervals": negative_rise_intervals,
        "total_local_intervals": total_intervals,
        "negative_local_rise_fraction": negative_rise_intervals / total_intervals if total_intervals else np.nan,
        "median_chain_axis_angle_to_model_axis": float(np.nanmedian(chain_axis_angles)) if chain_axis_angles else np.nan,
    }
    return trace, diagnostics


def summarize_trace(trace: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["model_label", "chain", "rms_state", "step_type", "involved_in_C", "involved_in_D"]
    metric_columns = {
        "theta_pp_diagnostic_deg": "median_theta",
        "rms": "median_rms",
        "local_rise": "median_local_rise",
        "abs_local_twist_deg": "median_abs_local_twist",
        "twist_deviation_from_30_deg": "median_twist_deviation_from_30",
        "radial_distance": "median_radial_distance",
        "cno_to_peptide_normal_angle_deg": "median_cno_angle",
        "omega_deviation_from_trans_deg": "median_omega_deviation",
    }
    summary = (
        trace.groupby(group_columns, dropna=False)
        .agg(count=("plane_index", "count"), **{name: (column, "median") for column, name in metric_columns.items()})
        .reset_index()
    )
    return summary.sort_values(group_columns)


def median_by_mask(trace: pd.DataFrame, mask: pd.Series, column: str) -> float:
    values = trace.loc[mask, column]
    return float(values.median()) if not values.dropna().empty else np.nan


def comparative_lines(trace: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    for model, model_df in trace.groupby("model_label", sort=True):
        high = model_df["rms_state"] == "high_rms"
        low = model_df["rms_state"] == "low_rms"
        d_in = model_df["involved_in_D"]
        d_out = ~model_df["involved_in_D"]
        c_in = model_df["involved_in_C"]
        c_out = ~model_df["involved_in_C"]
        lines.append(f"### {model}")
        lines.append(
            "- high_rms vs low_rms median abs twist: "
            f"{median_by_mask(model_df, high, 'abs_local_twist_deg'):.3f} vs "
            f"{median_by_mask(model_df, low, 'abs_local_twist_deg'):.3f}; "
            "median rise: "
            f"{median_by_mask(model_df, high, 'local_rise'):.3f} vs "
            f"{median_by_mask(model_df, low, 'local_rise'):.3f}"
        )
        lines.append(
            "- D-involved vs non-D median twist deviation from 30: "
            f"{median_by_mask(model_df, d_in, 'twist_deviation_from_30_deg'):.3f} vs "
            f"{median_by_mask(model_df, d_out, 'twist_deviation_from_30_deg'):.3f}; "
            "median rise: "
            f"{median_by_mask(model_df, d_in, 'local_rise'):.3f} vs "
            f"{median_by_mask(model_df, d_out, 'local_rise'):.3f}"
        )
        lines.append(
            "- C-involved vs non-C median twist deviation from 30: "
            f"{median_by_mask(model_df, c_in, 'twist_deviation_from_30_deg'):.3f} vs "
            f"{median_by_mask(model_df, c_out, 'twist_deviation_from_30_deg'):.3f}; "
            "median rise: "
            f"{median_by_mask(model_df, c_in, 'local_rise'):.3f} vs "
            f"{median_by_mask(model_df, c_out, 'local_rise'):.3f}"
        )
    return lines


def color_for_chain(chain: str) -> str:
    return CHAIN_COLORS.get(str(chain), "#333333")


def scatter_plot(trace: pd.DataFrame, x: str, y: str, outpath: Path, xlabel: str, ylabel: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for chain, group in trace.groupby("chain"):
        ax.scatter(group[x], group[y], s=20, alpha=0.65, color=color_for_chain(chain), label=str(chain))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="chain", ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def boxplot_by_state(trace: pd.DataFrame, outpath: Path) -> None:
    states = ["low_rms", "mid_rms", "high_rms"]
    models = list(trace["model_label"].drop_duplicates())
    fig, axes = plt.subplots(len(models), 1, figsize=(8, max(3, 2.2 * len(models))), sharex=True)
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        model_df = trace[trace["model_label"] == model]
        data = [model_df.loc[model_df["rms_state"] == state, "abs_local_twist_deg"].dropna() for state in states]
        ax.boxplot(data, tick_labels=states, showfliers=False)
        ax.set_ylabel("abs twist")
        ax.set_title(model, fontsize=9)
    axes[-1].set_xlabel("RMS state")
    fig.suptitle("Local twist by RMS state and model", y=0.995)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def trace_plot(
    trace: pd.DataFrame,
    model: str,
    columns: list[tuple[str, str, str]],
    outpath: Path,
    title: str,
    ylabel: str,
) -> None:
    model_df = trace[trace["model_label"] == model].sort_values("serial_x")
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for column, label, color in columns:
        ax.plot(model_df["serial_x"], model_df[column], label=label, color=color, lw=1.4)
    chain_mids = []
    chain_labels = []
    for chain, group in model_df.groupby("chain", sort=False):
        chain_mids.append((group["serial_x"].min() + group["serial_x"].max()) / 2)
        chain_labels.append(str(chain))
        ax.axvline(group["serial_x"].max() + 2.5, color="#cccccc", lw=0.8)
    ax.set_xticks(chain_mids, chain_labels)
    ax.set_xlabel("chain")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def distribution_plot(trace: pd.DataFrame, outpath: Path, mode: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if mode == "D":
        groups = [("D-involved", trace[trace["involved_in_D"]]), ("not D-involved", trace[~trace["involved_in_D"]])]
    else:
        groups = [
            ("C-involved", trace[trace["involved_in_C"]]),
            ("D-involved", trace[trace["involved_in_D"]]),
        ]
    data = [group["abs_local_twist_deg"].dropna() for _, group in groups]
    ax.boxplot(data, tick_labels=[name for name, _ in groups], showfliers=False)
    ax.axhline(NOMINAL_TWIST_DEG, color="#777777", lw=1, ls="--", label="30 deg")
    ax.set_ylabel("absolute local twist (deg)")
    ax.set_title("Local twist distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def write_report(trace: pd.DataFrame, summary: pd.DataFrame, diagnostics: pd.DataFrame, outdir: Path) -> None:
    lines = [
        "# Backbone Twist Coupling Analysis",
        "",
        "This is an exploratory, descriptive, diagnostic analysis. It reuses Aleph-style local geometry machinery for peptide-plane centers, but it is not a diffraction simulator and does not prove experimental correctness.",
        "",
        "## Available Columns",
        "- `plane_features.csv` already contained plane centers (`center_x/y/z`) and normals (`normal_x/y/z`), plus RMS, CNO angle, and omega deviation.",
        "- Diagnostic peptide-plane theta was computed directly from adjacent plane normals using preserve-magnitude sign stabilization, so all primary models could be handled consistently.",
        "",
        "## Axis Handling",
        "- A model principal axis was fit through all peptide-plane centers by SVD/PCA.",
        "- The axis was oriented so the median chain-local center-to-center rise was positive.",
        "- Local twist was computed as the signed rotation of adjacent radial center vectors around that model axis.",
        "",
        "## Axis Diagnostics",
    ]
    for row in diagnostics.itertuples():
        lines.append(
            f"- {row.model_label}: negative local-rise intervals {int(row.negative_local_rise_intervals)}/"
            f"{int(row.total_local_intervals)} ({row.negative_local_rise_fraction:.3f}); "
            f"median chain-axis angle to model axis {row.median_chain_axis_angle_to_model_axis:.3f} deg"
        )

    lines.extend(["", "## Comparisons"])
    lines.extend(comparative_lines(trace))

    high = trace["rms_state"] == "high_rms"
    low = trace["rms_state"] == "low_rms"
    d_in = trace["involved_in_D"]
    c_in = trace["involved_in_C"]
    d_dev = median_by_mask(trace, d_in, "twist_deviation_from_30_deg")
    non_d_dev = median_by_mask(trace, ~d_in, "twist_deviation_from_30_deg")
    c_dev = median_by_mask(trace, c_in, "twist_deviation_from_30_deg")

    lines.extend(
        [
            "",
            "## Interpretation",
            f"- Do high-RMS planes have different local twist than low-RMS planes? Yes descriptively: aggregate median absolute twist is {median_by_mask(trace, high, 'abs_local_twist_deg'):.3f} deg for high_rms and {median_by_mask(trace, low, 'abs_local_twist_deg'):.3f} deg for low_rms.",
            f"- Do D-involved planes have local twist closer to nominal 30 degrees? In this diagnostic calculation, D-involved median twist deviation is {d_dev:.3f} deg versus {non_d_dev:.3f} deg for non-D planes.",
            f"- Do C and D candidates occupy different twist/rise/phase regimes? They differ in interface/register space already; here their aggregate twist deviations are C {c_dev:.3f} deg and D {d_dev:.3f} deg. Inspect the C-vs-D distribution plot before treating this as structural mechanism.",
            "- Does the full model show signs of layer-ordering or antiparallel ordering issues? The chain-local ordering avoids global layer-order assumptions. Negative local-rise fractions and chain-axis angles above are the main diagnostics; nonzero negative rises should be treated as axis/order caution flags.",
            "- Is Aleph useful here? Yes as local geometry machinery for center-axis rise/twist/phase descriptors, but not as a whole-helix fingerprint imposed on these antiparallel or multi-chain models.",
            "",
            "## Output Files",
            "- `backbone_twist_coupling_trace.csv`",
            "- `backbone_twist_coupling_summary.csv`",
            "- `theta_vs_local_twist.png`",
            "- `rms_vs_local_twist.png`",
            "- `rms_state_local_twist_by_model.png`",
            "- `local_twist_trace_full_ideal.png`",
            "- `local_rise_trace_full_ideal.png`",
            "- `theta_rms_twist_trace_full_ideal.png`",
            "- `D_involved_twist_distribution.png`",
            "- `C_vs_D_twist_distribution.png`",
        ]
    )
    (outdir / "backbone_twist_coupling_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    cd_df = pd.read_csv(args.cd_candidates)
    annotations = build_cd_plane_annotations(cd_df, args.models)
    traces: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []
    for model in args.models:
        trace, model_diagnostics = analyze_model(
            model,
            args.input_root,
            annotations,
            args.low_rms_threshold,
            args.high_rms_threshold,
            args.gap,
        )
        traces.append(trace)
        diagnostics.append(model_diagnostics)

    trace_df = pd.concat(traces, ignore_index=True)
    diagnostics_df = pd.DataFrame(diagnostics)
    summary_df = summarize_trace(trace_df)
    trace_df.to_csv(args.outdir / "backbone_twist_coupling_trace.csv", index=False)
    summary_df.to_csv(args.outdir / "backbone_twist_coupling_summary.csv", index=False)
    diagnostics_df.to_csv(args.outdir / "backbone_twist_coupling_axis_diagnostics.csv", index=False)

    scatter_plot(
        trace_df,
        "abs_local_twist_deg",
        "theta_pp_diagnostic_deg",
        args.outdir / "theta_vs_local_twist.png",
        "absolute local twist (deg)",
        "diagnostic theta-pp (deg)",
        "Theta vs local twist",
    )
    scatter_plot(
        trace_df,
        "abs_local_twist_deg",
        "rms",
        args.outdir / "rms_vs_local_twist.png",
        "absolute local twist (deg)",
        "plane RMS",
        "RMS vs local twist",
    )
    boxplot_by_state(trace_df, args.outdir / "rms_state_local_twist_by_model.png")
    trace_plot(
        trace_df,
        FULL_IDEAL_LABEL,
        [("abs_local_twist_deg", "abs local twist", "#4c78a8")],
        args.outdir / "local_twist_trace_full_ideal.png",
        "Full ideal local twist trace",
        "absolute local twist (deg)",
    )
    trace_plot(
        trace_df,
        FULL_IDEAL_LABEL,
        [("local_rise", "local rise", "#59a14f")],
        args.outdir / "local_rise_trace_full_ideal.png",
        "Full ideal local rise trace",
        "local rise along model axis",
    )
    trace_plot(
        trace_df,
        FULL_IDEAL_LABEL,
        [
            ("theta_pp_diagnostic_deg", "theta", "#4c78a8"),
            ("rms", "RMS", "#e15759"),
            ("abs_local_twist_deg", "abs twist", "#f28e2b"),
        ],
        args.outdir / "theta_rms_twist_trace_full_ideal.png",
        "Full ideal theta/RMS/twist diagnostic trace",
        "mixed units",
    )
    distribution_plot(trace_df, args.outdir / "D_involved_twist_distribution.png", "D")
    distribution_plot(trace_df, args.outdir / "C_vs_D_twist_distribution.png", "C_vs_D")
    write_report(trace_df, summary_df, diagnostics_df, args.outdir)

    print(f"Wrote backbone twist coupling analysis to {args.outdir}")
    print(f"Trace rows: {len(trace_df)}")
    print("Plane centers/normals: available in plane_features.csv")


if __name__ == "__main__":
    main()
