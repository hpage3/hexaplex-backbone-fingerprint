"""Decompose simple backbone/peptide-plane powder features by pair families.

This is a diagnostic/falsification script. It uses labeled coordinates to ask
which real-space pair families contribute distances and Debye-style partial
profiles near the C and D windows. It does not make final structural claims.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hexaplex_backbone_fingerprint.parametric_powder_scan import local_maxima, make_q_grid
from hexaplex_backbone_fingerprint.xyz_parser import parse_xyz


CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
BASE_FAMILIES = [
    "same_strand_same_repeat",
    "same_strand_plusminus1_repeat",
    "same_strand_plusminus2_or_more",
    "adjacent_strand_same_register",
    "adjacent_strand_plusminus1_register",
    "adjacent_strand_plusminus2_or_more",
    "nonadjacent_cross_strand",
    "interface_AB",
    "interface_BC",
    "interface_CD",
    "interface_DE",
    "interface_EF",
    "interface_FA",
    "alternating_interfaces_AB_CD_EF",
    "alternating_interfaces_BC_DE_FA",
    "all_same_strand",
    "all_cross_strand",
    "all_adjacent_cross_strand",
]


@dataclass(frozen=True)
class LabeledAtom:
    """One coordinate with conservative strand/repeat/atom metadata."""

    atom_index: int
    atom_name: str
    element: str
    chain: str
    strand_index: int
    repeat_index: int
    coord: np.ndarray


def safe_model_id(path: Path, explicit: str | None = None) -> str:
    """Return a filename-safe model id."""
    text = explicit or path.stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def parse_labeled_pdb(path: Path) -> list[LabeledAtom]:
    """Parse a labeled PDB using the parametric model chain/resseq convention."""
    atoms: list[LabeledAtom] = []
    for line in path.read_text(encoding="ascii").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        atom_name = line[12:16].strip()
        element = (line[76:78].strip() or atom_name[0]).upper()
        if element == "H":
            continue
        chain = line[21].strip()
        if not chain:
            raise ValueError(f"PDB atom without chain ID in {path}; strand labels are required.")
        if chain not in CHAIN_IDS:
            raise ValueError(f"Unsupported one-character chain ID {chain!r} in {path}.")
        resseq = int(line[22:26])
        atoms.append(
            LabeledAtom(
                atom_index=int(line[6:11]),
                atom_name=atom_name,
                element=element,
                chain=chain,
                strand_index=CHAIN_IDS.index(chain),
                repeat_index=(resseq - 1) // 2,
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
        raise ValueError(f"No heavy ATOM/HETATM records found in {path}.")
    return atoms


def parse_mapping_csv(path: Path) -> pd.DataFrame:
    """Read a coordinate mapping CSV for unlabeled XYZ-like files."""
    df = pd.read_csv(path)
    required = {"atom_index", "strand_index", "repeat_index"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Mapping CSV is missing required columns: {sorted(missing)}")
    return df


def load_labeled_atoms(coord_path: Path, mapping_csv: Path | None = None) -> list[LabeledAtom]:
    """Load labeled heavy atoms from PDB, or XYZ plus an explicit mapping CSV."""
    suffix = coord_path.suffix.lower()
    if suffix in {".pdb", ".ent"}:
        return parse_labeled_pdb(coord_path)
    if suffix == ".xyz":
        if mapping_csv is None:
            raise ValueError(
                "XYZ files do not contain strand/repeat labels. Provide --mapping-csv "
                "with atom_index,strand_index,repeat_index, and optional atom_name/chain columns."
            )
        xyz_atoms = parse_xyz(coord_path)
        mapping = parse_mapping_csv(mapping_csv).set_index("atom_index")
        atoms = []
        for xyz_atom in xyz_atoms:
            if xyz_atom.element.upper() == "H":
                continue
            if xyz_atom.atom_index not in mapping.index:
                raise ValueError(f"Mapping CSV has no row for XYZ atom_index {xyz_atom.atom_index}.")
            row = mapping.loc[xyz_atom.atom_index]
            strand_index = int(row["strand_index"])
            chain = str(row["chain"]) if "chain" in row and pd.notna(row["chain"]) else CHAIN_IDS[strand_index]
            atom_name = str(row["atom_name"]) if "atom_name" in row and pd.notna(row["atom_name"]) else xyz_atom.element
            atoms.append(
                LabeledAtom(
                    atom_index=xyz_atom.atom_index,
                    atom_name=atom_name,
                    element=xyz_atom.element.upper(),
                    chain=chain,
                    strand_index=strand_index,
                    repeat_index=int(row["repeat_index"]),
                    coord=np.array([xyz_atom.x, xyz_atom.y, xyz_atom.z], dtype=float),
                )
            )
        if not atoms:
            raise ValueError(f"No heavy atoms available after filtering {coord_path}.")
        return atoms
    raise ValueError(f"Unsupported coordinate format {suffix!r}. Use PDB, or XYZ with --mapping-csv.")


def ring_separation(strand_a: int, strand_b: int, n_strands: int = 6) -> int:
    """Return shortest separation around a six-strand ring."""
    raw = abs(strand_b - strand_a) % n_strands
    return min(raw, n_strands - raw)


def interface_name(atom_a: LabeledAtom, atom_b: LabeledAtom, n_strands: int = 6) -> str | None:
    """Return interface_AB style name for adjacent-strand pairs, including F-A."""
    if atom_a.strand_index == atom_b.strand_index:
        return None
    if ring_separation(atom_a.strand_index, atom_b.strand_index, n_strands=n_strands) != 1:
        return None
    low = min(atom_a.strand_index, atom_b.strand_index)
    high = max(atom_a.strand_index, atom_b.strand_index)
    if {low, high} == {0, n_strands - 1}:
        return "interface_FA"
    return f"interface_{CHAIN_IDS[low]}{CHAIN_IDS[high]}"


def classify_pair_families(atom_a: LabeledAtom, atom_b: LabeledAtom, n_strands: int = 6) -> list[str]:
    """Return all overlapping geometry families for one atom pair."""
    families: list[str] = []
    same_strand = atom_a.strand_index == atom_b.strand_index
    repeat_delta = atom_b.repeat_index - atom_a.repeat_index
    abs_delta = abs(repeat_delta)
    if same_strand:
        families.append("all_same_strand")
        if abs_delta == 0:
            families.append("same_strand_same_repeat")
        elif abs_delta == 1:
            families.append("same_strand_plusminus1_repeat")
        else:
            families.append("same_strand_plusminus2_or_more")
        return families

    families.append("all_cross_strand")
    separation = ring_separation(atom_a.strand_index, atom_b.strand_index, n_strands=n_strands)
    if separation != 1:
        families.append("nonadjacent_cross_strand")
        return families

    families.append("all_adjacent_cross_strand")
    if abs_delta == 0:
        families.append("adjacent_strand_same_register")
    elif abs_delta == 1:
        families.append("adjacent_strand_plusminus1_register")
    else:
        families.append("adjacent_strand_plusminus2_or_more")

    iface = interface_name(atom_a, atom_b, n_strands=n_strands)
    if iface is not None:
        families.append(iface)
        if iface in {"interface_AB", "interface_CD", "interface_EF"}:
            families.append("alternating_interfaces_AB_CD_EF")
        if iface in {"interface_BC", "interface_DE", "interface_FA"}:
            families.append("alternating_interfaces_BC_DE_FA")
    return families


def compute_pair_family_distances(atoms: list[LabeledAtom], n_strands: int = 6) -> dict[str, list[float]]:
    """Compute heavy-atom pair distances assigned to overlapping families."""
    distances_by_family = {family: [] for family in BASE_FAMILIES}
    coords = np.array([atom.coord for atom in atoms])
    for i in range(len(atoms) - 1):
        deltas = coords[i + 1 :] - coords[i]
        distances = np.linalg.norm(deltas, axis=1)
        for local_j, distance in enumerate(distances):
            atom_b = atoms[i + 1 + local_j]
            for family in classify_pair_families(atoms[i], atom_b, n_strands=n_strands):
                distances_by_family[family].append(float(distance))
    return distances_by_family


def partial_debye_profile(distances: np.ndarray, q_values: np.ndarray) -> pd.DataFrame:
    """Compute equal-weight pair-only Debye partial profile for one family."""
    distances = np.asarray(distances, dtype=float)
    intensities = np.zeros_like(q_values, dtype=float)
    if len(distances):
        for idx, q in enumerate(q_values):
            intensities[idx] = 2.0 * np.sum(np.sinc((q * distances) / np.pi))
    return pd.DataFrame({"q": q_values, "d_A": 2.0 * np.pi / q_values, "intensity": intensities}).sort_values("d_A")


def window_profile_peak(profile: pd.DataFrame, d_min: float, d_max: float) -> tuple[float, float]:
    """Return max intensity and its d spacing inside a d-spacing window."""
    window = profile[(profile["d_A"] >= d_min) & (profile["d_A"] <= d_max)]
    if window.empty:
        return np.nan, np.nan
    maxima = local_maxima(window.rename(columns={"q": "q_Ainv"}))
    source = maxima if not maxima.empty else window
    row = source.sort_values("intensity", ascending=False).iloc[0]
    return float(row["intensity"]), float(row["d_A"])


def write_histograms(model_id: str, distances_by_family: dict[str, list[float]], path: Path, bin_width: float) -> pd.DataFrame:
    """Write pair-distance histogram table."""
    max_distance = max((max(values) for values in distances_by_family.values() if values), default=0.0)
    upper = max(12.0, np.ceil(max_distance / bin_width) * bin_width)
    edges = np.arange(0.0, upper + bin_width, bin_width)
    rows = []
    for family, values in distances_by_family.items():
        counts, _ = np.histogram(values, bins=edges)
        centers = 0.5 * (edges[:-1] + edges[1:])
        for center, count in zip(centers, counts):
            rows.append(
                {
                    "model_id": model_id,
                    "family": family,
                    "d_bin_center_A": center,
                    "pair_count": int(count),
                    "weighted_pair_count_if_available": int(count),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def write_profiles(
    model_id: str,
    distances_by_family: dict[str, list[float]],
    path: Path,
    q_values: np.ndarray,
) -> pd.DataFrame:
    """Write Debye-style partial radial profiles by family."""
    frames = []
    for family, values in distances_by_family.items():
        profile = partial_debye_profile(np.asarray(values, dtype=float), q_values)
        profile.insert(0, "family", family)
        profile.insert(0, "model_id", model_id)
        frames.append(profile)
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(path, index=False)
    return df


def write_cd_summary(
    model_id: str,
    distances_by_family: dict[str, list[float]],
    profiles: pd.DataFrame,
    path: Path,
    c_window: tuple[float, float],
    d_window: tuple[float, float],
) -> pd.DataFrame:
    """Write C/D window summary by family."""
    rows = []
    for family, values in distances_by_family.items():
        distances = np.asarray(values, dtype=float)
        family_profile = profiles[profiles["family"] == family]
        c_intensity, c_peak = window_profile_peak(family_profile, *c_window)
        d_intensity, d_peak = window_profile_peak(family_profile, *d_window)
        rows.append(
            {
                "model_id": model_id,
                "family": family,
                "C_window_min_A": c_window[0],
                "C_window_max_A": c_window[1],
                "D_window_min_A": d_window[0],
                "D_window_max_A": d_window[1],
                "C_pair_count": int(((distances >= c_window[0]) & (distances <= c_window[1])).sum()),
                "D_pair_count": int(((distances >= d_window[0]) & (distances <= d_window[1])).sum()),
                "C_profile_max_intensity": c_intensity,
                "C_profile_peak_d_A": c_peak,
                "D_profile_max_intensity": d_intensity,
                "D_profile_peak_d_A": d_peak,
                "D_minus_C_peak_strength_ratio": d_intensity / c_intensity if c_intensity not in {0.0, np.nan} else np.nan,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def plot_histograms(histograms: pd.DataFrame, model_id: str, path_base: Path) -> None:
    """Plot family histograms focused on C/D region."""
    focus = histograms[(histograms["d_bin_center_A"] >= 4.5) & (histograms["d_bin_center_A"] <= 8.5)]
    important = (
        focus.groupby("family")["pair_count"].sum().sort_values(ascending=False).head(10).index.tolist()
    )
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for family in important:
        group = focus[focus["family"] == family]
        ax.plot(group["d_bin_center_A"], group["pair_count"], lw=1.2, label=family)
    ax.axvspan(5.4, 5.8, color="#1f77b4", alpha=0.12, label="C window")
    ax.axvspan(7.0, 7.5, color="#ff7f0e", alpha=0.12, label="D window")
    ax.set_xlabel("pair distance (A)")
    ax.set_ylabel("pair count")
    ax.set_title(f"{model_id}: pair-family distance histograms")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".png"), dpi=180)
    fig.savefig(path_base.with_suffix(".svg"))
    plt.close(fig)


def plot_profiles(profiles: pd.DataFrame, summary: pd.DataFrame, model_id: str, path_base: Path) -> None:
    """Plot partial radial profiles focused on C/D region."""
    top_families = (
        summary.assign(total_pairs=summary["C_pair_count"] + summary["D_pair_count"])
        .sort_values("total_pairs", ascending=False)
        .head(10)["family"]
        .tolist()
    )
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for family in top_families:
        group = profiles[(profiles["family"] == family) & profiles["d_A"].between(4.5, 8.5)]
        ax.plot(group["d_A"], group["intensity"], lw=1.2, label=family)
    ax.axvspan(5.4, 5.8, color="#1f77b4", alpha=0.12, label="C window")
    ax.axvspan(7.0, 7.5, color="#ff7f0e", alpha=0.12, label="D window")
    ax.set_xlabel("d spacing (A)")
    ax.set_ylabel("pair-only partial intensity")
    ax.set_title(f"{model_id}: pair-family Debye profiles")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".png"), dpi=180)
    fig.savefig(path_base.with_suffix(".svg"))
    plt.close(fig)


def _top_family(summary: pd.DataFrame, column: str) -> pd.Series:
    return summary.sort_values(column, ascending=False).iloc[0]


def write_report(model_id: str, coord_path: Path, summary: pd.DataFrame, report_path: Path) -> None:
    """Write concise markdown interpretation for C/D pair-family diagnostic."""
    c_pair = _top_family(summary, "C_pair_count")
    d_pair = _top_family(summary, "D_pair_count")
    c_profile = _top_family(summary.fillna({"C_profile_max_intensity": -np.inf}), "C_profile_max_intensity")
    d_profile = _top_family(summary.fillna({"D_profile_max_intensity": -np.inf}), "D_profile_max_intensity")
    same_family = c_pair["family"] == d_pair["family"]
    alt1 = summary[summary["family"] == "alternating_interfaces_AB_CD_EF"].iloc[0]
    alt2 = summary[summary["family"] == "alternating_interfaces_BC_DE_FA"].iloc[0]
    adjacent_same = summary[summary["family"] == "adjacent_strand_same_register"].iloc[0]
    same_pm1 = summary[summary["family"] == "same_strand_plusminus1_repeat"].iloc[0]
    adjacent_pm1 = summary[summary["family"] == "adjacent_strand_plusminus1_register"].iloc[0]
    text = f"""# Pair-Family C/D Diagnostic: `{model_id}`

Input coordinate file: `{coord_path}`

This is a diagnostic/falsification decomposition of heavy-atom pair distances and pair-only Debye partial profiles. It should not be read as a final structural assignment.

## C Window

- C distance window: {c_pair.C_window_min_A:.2f}-{c_pair.C_window_max_A:.2f} A
- Most C-window pairs: `{c_pair.family}` ({int(c_pair.C_pair_count)} pairs)
- Strongest C-window partial profile: `{c_profile.family}` (peak d = {c_profile.C_profile_peak_d_A:.3f} A, intensity = {c_profile.C_profile_max_intensity:.3f})

## D Window

- D distance window: {d_pair.D_window_min_A:.2f}-{d_pair.D_window_max_A:.2f} A
- Most D-window pairs: `{d_pair.family}` ({int(d_pair.D_pair_count)} pairs)
- Strongest D-window partial profile: `{d_profile.family}` (peak d = {d_profile.D_profile_peak_d_A:.3f} A, intensity = {d_profile.D_profile_max_intensity:.3f})

## Interpretation Questions

- Which pair families dominate distances near C? `{c_pair.family}` by raw pair count, with profile support from `{c_profile.family}`.
- Which pair families dominate distances near D? `{d_pair.family}` by raw pair count, with profile support from `{d_profile.family}`.
- Are C and D coming from the same family? {'Yes, by raw pair-count winner.' if same_family else 'No, the raw pair-count winners differ.'}
- Do alternating interfaces AB/CD/EF differ from BC/DE/FA? AB/CD/EF has C/D counts {int(alt1.C_pair_count)}/{int(alt1.D_pair_count)}; BC/DE/FA has C/D counts {int(alt2.C_pair_count)}/{int(alt2.D_pair_count)}.
- Is D primarily adjacent-strand same-register? Adjacent same-register has {int(adjacent_same.D_pair_count)} D-window pairs.
- Is C primarily same-strand or slipped-register? Same-strand +/-1 repeat has {int(same_pm1.C_pair_count)} C-window pairs; adjacent-strand +/-1 register has {int(adjacent_pm1.C_pair_count)} C-window pairs.

## Output Tables

- Pair-distance histogram CSV: `outputs/metrics/{model_id}_pair_family_distance_histograms.csv`
- Pair-family radial profile CSV: `outputs/metrics/{model_id}_pair_family_radial_profiles.csv`
- C/D summary CSV: `outputs/metrics/{model_id}_pair_family_cd_summary.csv`
"""
    report_path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coordinate_file", type=Path)
    parser.add_argument("--mapping-csv", type=Path, help="Required for unlabeled XYZ files.")
    parser.add_argument("--model-id", help="Optional model id for output filenames.")
    parser.add_argument("--outdir", type=Path, default=Path("outputs"))
    parser.add_argument("--c-window-min", type=float, default=5.4)
    parser.add_argument("--c-window-max", type=float, default=5.8)
    parser.add_argument("--d-window-min", type=float, default=7.0)
    parser.add_argument("--d-window-max", type=float, default=7.5)
    parser.add_argument("--d-min", type=float, default=2.5)
    parser.add_argument("--d-max", type=float, default=12.0)
    parser.add_argument("--q-step", type=float, default=0.005)
    parser.add_argument("--hist-bin-width", type=float, default=0.05)
    parser.add_argument("--n-strands", type=int, default=6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_id = safe_model_id(args.coordinate_file, args.model_id)
    metrics_dir = args.outdir / "metrics"
    figures_dir = args.outdir / "figures"
    reports_dir = args.outdir / "reports"
    for path in [metrics_dir, figures_dir, reports_dir]:
        path.mkdir(parents=True, exist_ok=True)

    atoms = load_labeled_atoms(args.coordinate_file, args.mapping_csv)
    distances_by_family = compute_pair_family_distances(atoms, n_strands=args.n_strands)
    q_values = make_q_grid(d_min_A=args.d_min, d_max_A=args.d_max, q_step=args.q_step)
    histograms = write_histograms(
        model_id,
        distances_by_family,
        metrics_dir / f"{model_id}_pair_family_distance_histograms.csv",
        args.hist_bin_width,
    )
    profiles = write_profiles(
        model_id,
        distances_by_family,
        metrics_dir / f"{model_id}_pair_family_radial_profiles.csv",
        q_values,
    )
    summary = write_cd_summary(
        model_id,
        distances_by_family,
        profiles,
        metrics_dir / f"{model_id}_pair_family_cd_summary.csv",
        (args.c_window_min, args.c_window_max),
        (args.d_window_min, args.d_window_max),
    )
    plot_histograms(histograms, model_id, figures_dir / f"{model_id}_pair_family_cd_histograms")
    plot_profiles(profiles, summary, model_id, figures_dir / f"{model_id}_pair_family_radial_profiles_C_D_focus")
    write_report(model_id, args.coordinate_file, summary, reports_dir / f"{model_id}_pair_family_cd_report.md")

    print(f"Analyzed {len(atoms)} labeled heavy atoms")
    print(f"Metrics: {metrics_dir}")
    print(f"Figures: {figures_dir}")
    print(f"Report: {reports_dir / f'{model_id}_pair_family_cd_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
