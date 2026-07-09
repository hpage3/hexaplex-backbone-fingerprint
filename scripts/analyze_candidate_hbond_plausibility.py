"""Score candidate structures with a geometric hydrogen-bond plausibility filter.

This is a hydrogen-bond plausibility score, not a true affinity or free-energy
calculation. It is intended as a comparative physical-sense filter that can
complement C/D matching, omega geometry, carboxylate preservation, and torsion
boundary checks.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_candidate_models_to_emory_fiber_fingerprint import expected_candidate_records, inventory_dataframe


OUT_PAIRS = Path("outputs/metrics/candidate_hbond_pairs.csv")
OUT_SUMMARY = Path("outputs/metrics/candidate_hbond_summary.csv")
OUT_DIAGNOSTICS = Path("outputs/metrics/candidate_hbond_scoring_diagnostics.csv")
OUT_TOP_PAIRS = Path("outputs/metrics/candidate_hbond_top_pairs.csv")
OUT_REPORT = Path("outputs/reports/candidate_hbond_plausibility_report.md")
FIG_SCORE = Path("outputs/figures/candidate_hbond_score_summary.png")
FIG_DISTANCE = Path("outputs/figures/candidate_hbond_distance_distribution.png")


@dataclass(frozen=True)
class AtomRecord:
    """Parsed PDB atom record."""

    serial: int
    atom_name: str
    resname: str
    chain: str
    resseq: str
    icode: str
    element: str
    coord: np.ndarray

    @property
    def residue_key(self) -> tuple[str, str, str]:
        return (self.chain, self.resseq, self.icode)

    @property
    def atom_id(self) -> str:
        return f"{self.chain}:{self.resname}{self.resseq}{self.icode}:{self.atom_name}"


def infer_element(atom_name: str, element_field: str = "") -> str:
    """Infer element from PDB element field or atom name."""
    value = element_field.strip().upper()
    if value:
        return value
    stripped = atom_name.strip()
    if not stripped:
        return ""
    if stripped[0].isdigit() and len(stripped) > 1:
        return stripped[1].upper()
    return stripped[0].upper()


def parse_pdb_atoms(path: Path) -> list[AtomRecord]:
    """Parse ATOM/HETATM records from a PDB file."""
    atoms: list[AtomRecord] = []
    for line in path.read_text(encoding="ascii").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        atoms.append(
            AtomRecord(
                serial=int(line[6:11]),
                atom_name=line[12:16].strip(),
                resname=line[17:20].strip(),
                chain=line[21:22].strip(),
                resseq=line[22:26].strip(),
                icode=line[26:27].strip(),
                element=infer_element(line[12:16], line[76:78] if len(line) >= 78 else ""),
                coord=np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=float),
            )
        )
    if not atoms:
        raise ValueError(f"No ATOM/HETATM records found in {path}.")
    return atoms


def is_hydrogen(atom: AtomRecord) -> bool:
    """Return whether atom is hydrogen."""
    return atom.element == "H" or atom.atom_name.upper().startswith("H")


def is_carboxylate_oxygen(atom: AtomRecord) -> bool:
    """Conservatively identify likely carboxylate oxygens."""
    name = atom.atom_name.upper()
    res = atom.resname.upper()
    return atom.element == "O" and (name in {"OE1", "OE2", "OD1", "OD2", "OXT"} or res in {"GLU", "ASP"} and name.startswith(("OE", "OD")))


def is_likely_donor(atom: AtomRecord) -> bool:
    """Return whether heavy atom is a conservative H-bond donor candidate."""
    if is_hydrogen(atom):
        return False
    name = atom.atom_name.upper()
    if atom.element == "N":
        return not name.startswith("C")
    if atom.element == "O":
        return name.startswith(("OH", "OG", "OG1", "NE", "NH"))
    return False


def is_likely_acceptor(atom: AtomRecord) -> bool:
    """Return whether heavy atom is a conservative H-bond acceptor candidate."""
    if is_hydrogen(atom):
        return False
    name = atom.atom_name.upper()
    if atom.element == "O":
        return True
    if atom.element == "N":
        return not name.startswith(("N", "NH", "NE"))
    return False


def donor_hydrogens(donor: AtomRecord, atoms_by_residue: dict[tuple[str, str, str], list[AtomRecord]]) -> list[AtomRecord]:
    """Return explicit hydrogens close to the donor within the same residue."""
    hydrogens = []
    for atom in atoms_by_residue.get(donor.residue_key, []):
        if is_hydrogen(atom) and np.linalg.norm(atom.coord - donor.coord) <= 1.25:
            hydrogens.append(atom)
    return hydrogens


def pair_type(donor: AtomRecord, acceptor: AtomRecord) -> str:
    """Classify donor/acceptor relationship."""
    if donor.residue_key == acceptor.residue_key:
        return "intraresidue"
    if donor.chain and acceptor.chain and donor.chain != acceptor.chain:
        return "interchain"
    if donor.chain == acceptor.chain:
        return "intrachain"
    return "unknown"


def residue_number(atom: AtomRecord) -> int | None:
    """Parse residue number when possible."""
    try:
        return int(atom.resseq)
    except ValueError:
        return None


def should_exclude_pair(donor: AtomRecord, acceptor: AtomRecord) -> bool:
    """Exclude obvious same-residue or neighboring covalent/trivial pairs."""
    if donor.serial == acceptor.serial:
        return True
    if donor.residue_key == acceptor.residue_key:
        return True
    if donor.chain and donor.chain == acceptor.chain:
        d_res = residue_number(donor)
        a_res = residue_number(acceptor)
        if d_res is not None and a_res is not None and abs(d_res - a_res) <= 1:
            return True
    return False


def pair_exclusion_reason(donor: AtomRecord, acceptor: AtomRecord) -> str:
    """Return why a donor/acceptor pair is excluded, or empty string."""
    if donor.serial == acceptor.serial:
        return "same_atom_or_invalid"
    if donor.residue_key == acceptor.residue_key:
        return "same_residue"
    if donor.chain and donor.chain == acceptor.chain:
        d_res = residue_number(donor)
        a_res = residue_number(acceptor)
        if d_res is not None and a_res is not None and abs(d_res - a_res) <= 1:
            return "close_sequence_or_trivial"
    return ""


def classify_distance(distance_A: float) -> str:
    """Classify heavy-atom donor/acceptor distance."""
    if distance_A < 2.4:
        return "rejected_too_close"
    if 2.6 <= distance_A <= 3.1:
        return "strong_geometry"
    if 2.5 <= distance_A <= 3.4:
        return "plausible_geometry"
    if 3.4 < distance_A <= 3.7:
        return "weak_geometry"
    return "rejected_too_far"


def angle_degrees(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return angle ABC in degrees."""
    ba = a - b
    bc = c - b
    denom = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom <= 1e-12:
        return float("nan")
    cosang = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return math.degrees(math.acos(cosang))


def classify_angle(distance_class: str, angle_deg: float) -> str:
    """Classify explicit donor-H-acceptor geometry."""
    if distance_class in {"rejected_too_close", "rejected_too_far"}:
        return distance_class
    if not np.isfinite(angle_deg):
        return distance_class
    if angle_deg >= 150.0 and distance_class in {"strong_geometry", "plausible_geometry"}:
        return "strong_geometry"
    if angle_deg >= 120.0 and distance_class != "weak_geometry":
        return "plausible_geometry"
    if angle_deg >= 100.0:
        return "weak_geometry"
    return "rejected_too_far"


def bad_same_type_contacts(atoms: list[AtomRecord], predicate, cutoff_A: float = 2.4) -> int:
    """Count close same-type donor/donor or acceptor/acceptor contacts."""
    selected = [atom for atom in atoms if predicate(atom)]
    count = 0
    for idx, atom_a in enumerate(selected):
        for atom_b in selected[idx + 1 :]:
            if should_exclude_pair(atom_a, atom_b):
                continue
            if np.linalg.norm(atom_a.coord - atom_b.coord) < cutoff_A:
                count += 1
    return count


def pair_score_contribution(distance_class: str) -> int:
    """Return score contribution from one distance class."""
    return {
        "strong_geometry": 3,
        "plausible_geometry": 2,
        "weak_geometry": 1,
        "rejected_too_close": -3,
        "rejected_too_far": 0,
    }.get(distance_class, 0)


def atom_classification_counts(atoms: list[AtomRecord]) -> dict[str, int]:
    """Return donor/acceptor atom classification counts."""
    known = {"H", "C", "N", "O", "S", "P", "CL", "NA", "MG", "ZN", "FE", "CA"}
    acceptors = [atom for atom in atoms if is_likely_acceptor(atom)]
    return {
        "donor_atom_count": sum(1 for atom in atoms if is_likely_donor(atom)),
        "acceptor_atom_count": len(acceptors),
        "carboxylate_acceptor_count": sum(1 for atom in acceptors if is_carboxylate_oxygen(atom)),
        "unknown_element_or_untyped_atom_count": sum(1 for atom in atoms if atom.element.upper() not in known),
    }


def hbond_pair_rows(model_id: str, atoms: list[AtomRecord]) -> pd.DataFrame:
    """Return post-exclusion donor/acceptor pair rows for a model."""
    donors = [atom for atom in atoms if is_likely_donor(atom)]
    acceptors = [atom for atom in atoms if is_likely_acceptor(atom)]
    atoms_by_residue: dict[tuple[str, str, str], list[AtomRecord]] = {}
    for atom in atoms:
        atoms_by_residue.setdefault(atom.residue_key, []).append(atom)
    rows = []
    for donor in donors:
        hydrogens = donor_hydrogens(donor, atoms_by_residue)
        for acceptor in acceptors:
            if pair_exclusion_reason(donor, acceptor):
                continue
            distance = float(np.linalg.norm(donor.coord - acceptor.coord))
            distance_class = classify_distance(distance)
            h_acceptor_distance = float("nan")
            angle = float("nan")
            if hydrogens:
                best = min(hydrogens, key=lambda h: np.linalg.norm(h.coord - acceptor.coord))
                h_acceptor_distance = float(np.linalg.norm(best.coord - acceptor.coord))
                angle = angle_degrees(donor.coord, best.coord, acceptor.coord)
                hbond_class = classify_angle(distance_class, angle)
                geometry_basis = "explicit_hydrogen_angle"
            else:
                hbond_class = "missing_hydrogen_proxy" if distance_class not in {"rejected_too_close", "rejected_too_far"} else distance_class
                geometry_basis = "hydrogen_missing_geometry_proxy"
            rows.append(
                {
                    "model_id": model_id,
                    "candidate_name": model_id,
                    "donor_atom": donor.atom_id,
                    "acceptor_atom": acceptor.atom_id,
                    "donor_chain": donor.chain,
                    "donor_resname": donor.resname,
                    "donor_resseq": donor.resseq,
                    "donor_atom_name": donor.atom_name,
                    "donor_element": donor.element,
                    "acceptor_chain": acceptor.chain,
                    "acceptor_resname": acceptor.resname,
                    "acceptor_resseq": acceptor.resseq,
                    "acceptor_atom_name": acceptor.atom_name,
                    "acceptor_element": acceptor.element,
                    "donor_acceptor_distance_A": distance,
                    "H_acceptor_distance_A": h_acceptor_distance,
                    "donor_H_acceptor_angle_deg": angle,
                    "pair_type": pair_type(donor, acceptor),
                    "distance_hbond_class": distance_class,
                    "hbond_class": hbond_class,
                    "geometry_basis": geometry_basis,
                    "involves_carboxylate": is_carboxylate_oxygen(donor) or is_carboxylate_oxygen(acceptor),
                    "pair_score_contribution": pair_score_contribution(distance_class),
                    "reason_included": "post_exclusion_distance_evaluated",
                }
            )
    return pd.DataFrame(rows)


def pair_generation_diagnostics(atoms: list[AtomRecord], pair_df: pd.DataFrame) -> dict[str, object]:
    """Return pair-generation diagnostic counts."""
    donors = [atom for atom in atoms if is_likely_donor(atom)]
    acceptors = [atom for atom in atoms if is_likely_acceptor(atom)]
    excluded_same = 0
    excluded_close = 0
    excluded_invalid = 0
    for donor in donors:
        for acceptor in acceptors:
            reason = pair_exclusion_reason(donor, acceptor)
            if reason == "same_residue":
                excluded_same += 1
            elif reason == "close_sequence_or_trivial":
                excluded_close += 1
            elif reason == "same_atom_or_invalid":
                excluded_invalid += 1
    return {
        "total_possible_donor_acceptor_pairs": len(donors) * len(acceptors),
        "excluded_same_residue_pairs": excluded_same,
        "excluded_close_sequence_or_trivial_pairs": excluded_close,
        "excluded_same_atom_or_invalid_pairs": excluded_invalid,
        "pairs_considered_after_exclusions": len(pair_df),
        "interchain_pairs_considered": int((pair_df["pair_type"] == "interchain").sum()) if not pair_df.empty else 0,
        "intrachain_pairs_considered": int((pair_df["pair_type"] == "intrachain").sum()) if not pair_df.empty else 0,
    }


def score_summary_row(model_id: str, atoms: list[AtomRecord], pair_df: pd.DataFrame, inventory_row: pd.Series) -> dict[str, object]:
    """Return candidate-level H-bond summary row."""
    hydrogens_present = any(is_hydrogen(atom) for atom in atoms)
    distance_classes = pair_df["distance_hbond_class"] if not pair_df.empty else pd.Series(dtype=str)
    class_counts = atom_classification_counts(atoms)
    pair_diag = pair_generation_diagnostics(atoms, pair_df)
    bad_acceptor = bad_same_type_contacts(atoms, is_likely_acceptor)
    bad_donor = bad_same_type_contacts(atoms, is_likely_donor)
    bad_contacts = bad_acceptor + bad_donor
    strong = int((distance_classes == "strong_geometry").sum())
    plausible = int((distance_classes == "plausible_geometry").sum())
    weak = int((distance_classes == "weak_geometry").sum())
    too_far = int((distance_classes == "rejected_too_far").sum())
    too_close = int((distance_classes == "rejected_too_close").sum())
    missing_proxy = int((pair_df["hbond_class"] == "missing_hydrogen_proxy").sum()) if not pair_df.empty else 0
    raw_positive = 3 * strong + 2 * plausible + weak
    raw_penalty = 3 * too_close + 2 * bad_contacts
    raw_score = raw_positive - raw_penalty
    residue_count = len({atom.residue_key for atom in atoms})
    chain_count = len({atom.chain for atom in atoms if atom.chain})
    normalized = raw_score / residue_count if residue_count else float("nan")
    interchain = int((pair_df["pair_type"] == "interchain").sum()) if not pair_df.empty else 0
    intrachain = int((pair_df["pair_type"] == "intrachain").sum()) if not pair_df.empty else 0
    balance = interchain / max(1, strong + plausible + weak)
    if class_counts["donor_atom_count"] == 0 or class_counts["acceptor_atom_count"] == 0:
        classification = "insufficient_atom_annotation"
    elif not pair_df.empty and (strong + plausible + weak) >= 10 and too_close <= 2 and bad_contacts <= 5:
        classification = "hbond_network_plausible"
    elif not pair_df.empty and (strong + plausible + weak) >= 3:
        classification = "hbond_network_marginal"
    else:
        classification = "hbond_network_poor"
    caveat = "explicit_hydrogen_angles_used" if hydrogens_present else "hydrogen_missing_geometry_proxy; missing hydrogens limit angle-based interpretation"
    return {
        "model_id": model_id,
        "candidate_name": model_id,
        "status": inventory_row.get("status", "found"),
        "path": inventory_row.get("path", ""),
        "coordinate_path": inventory_row.get("path", ""),
        "inferred_family": inventory_row.get("inferred_family", ""),
        "family": inventory_row.get("inferred_family", ""),
        "inferred_scale": inventory_row.get("inferred_scale", ""),
        "scale": inventory_row.get("inferred_scale", ""),
        "atom_count": len(atoms),
        "residue_count": residue_count,
        "chain_count": chain_count,
        "explicit_hydrogens_present": hydrogens_present,
        **class_counts,
        **pair_diag,
        "total_candidate_pairs_considered": len(pair_df),
        "strong_geometry_count": strong,
        "plausible_geometry_count": plausible,
        "weak_geometry_count": weak,
        "strong_hbond_count": strong,
        "plausible_hbond_count": plausible,
        "weak_hbond_count": weak,
        "rejected_too_far_count": too_far,
        "rejected_too_close_count": too_close,
        "missing_hydrogen_proxy_count": missing_proxy,
        "interchain_hbond_count": interchain,
        "intrachain_hbond_count": intrachain,
        "carboxylate_contact_count": int(pair_df["involves_carboxylate"].sum()) if not pair_df.empty else 0,
        "possible_bad_acceptor_acceptor_contact_count": bad_acceptor,
        "possible_bad_donor_donor_contact_count": bad_donor,
        "min_donor_acceptor_distance_A": float(pd.to_numeric(pair_df["donor_acceptor_distance_A"], errors="coerce").min()) if not pair_df.empty else float("nan"),
        "median_donor_acceptor_distance_A": float(pd.to_numeric(pair_df["donor_acceptor_distance_A"], errors="coerce").median()) if not pair_df.empty else float("nan"),
        "max_plausible_distance_A": float(pd.to_numeric(pair_df.loc[pair_df["distance_hbond_class"].isin(["strong_geometry", "plausible_geometry", "weak_geometry"]), "donor_acceptor_distance_A"], errors="coerce").max()) if not pair_df.empty else float("nan"),
        "raw_positive_score_components": raw_positive,
        "raw_penalty_components": raw_penalty,
        "hbond_plausibility_score": raw_score,
        "hbond_plausibility_score_per_residue": normalized,
        "normalized_hbond_score": normalized,
        "hbond_network_balance_score": balance,
        "hbond_network_classification": classification,
        "hbond_caveat": caveat,
    }


def missing_summary_row(inventory_row: pd.Series) -> dict[str, object]:
    """Return summary row for missing coordinate candidate."""
    return {
        "model_id": inventory_row["model_id"],
        "candidate_name": inventory_row["model_id"],
        "status": "missing_candidate_coordinates",
        "path": inventory_row.get("path", ""),
        "coordinate_path": inventory_row.get("path", ""),
        "inferred_family": inventory_row.get("inferred_family", ""),
        "family": inventory_row.get("inferred_family", ""),
        "inferred_scale": inventory_row.get("inferred_scale", ""),
        "scale": inventory_row.get("inferred_scale", ""),
        "atom_count": 0,
        "residue_count": 0,
        "chain_count": 0,
        "explicit_hydrogens_present": False,
        "donor_atom_count": 0,
        "acceptor_atom_count": 0,
        "carboxylate_acceptor_count": 0,
        "unknown_element_or_untyped_atom_count": 0,
        "total_possible_donor_acceptor_pairs": 0,
        "excluded_same_residue_pairs": 0,
        "excluded_close_sequence_or_trivial_pairs": 0,
        "excluded_same_atom_or_invalid_pairs": 0,
        "pairs_considered_after_exclusions": 0,
        "interchain_pairs_considered": 0,
        "intrachain_pairs_considered": 0,
        "total_candidate_pairs_considered": 0,
        "strong_geometry_count": 0,
        "plausible_geometry_count": 0,
        "weak_geometry_count": 0,
        "strong_hbond_count": 0,
        "plausible_hbond_count": 0,
        "weak_hbond_count": 0,
        "rejected_too_far_count": 0,
        "rejected_too_close_count": 0,
        "missing_hydrogen_proxy_count": 0,
        "interchain_hbond_count": 0,
        "intrachain_hbond_count": 0,
        "carboxylate_contact_count": 0,
        "possible_bad_acceptor_acceptor_contact_count": 0,
        "possible_bad_donor_donor_contact_count": 0,
        "min_donor_acceptor_distance_A": float("nan"),
        "median_donor_acceptor_distance_A": float("nan"),
        "max_plausible_distance_A": float("nan"),
        "raw_positive_score_components": 0,
        "raw_penalty_components": 0,
        "hbond_plausibility_score": 0,
        "hbond_plausibility_score_per_residue": float("nan"),
        "normalized_hbond_score": float("nan"),
        "hbond_network_balance_score": float("nan"),
        "hbond_network_classification": "missing_candidate_coordinates",
        "hbond_caveat": "missing_candidate_coordinates",
    }


def analyze_inventory(inventory: pd.DataFrame, max_candidates: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze candidate inventory rows."""
    rows_to_analyze = inventory.copy()
    if max_candidates > 0:
        found = rows_to_analyze[rows_to_analyze["status"] == "found"].head(max_candidates)
        missing = rows_to_analyze[rows_to_analyze["status"] != "found"]
        rows_to_analyze = pd.concat([found, missing], ignore_index=True)
    pair_frames = []
    summary_rows = []
    for _, row in rows_to_analyze.iterrows():
        if row["status"] != "found" or not str(row.get("path", "")):
            summary_rows.append(missing_summary_row(row))
            continue
        atoms = parse_pdb_atoms(Path(str(row["path"])))
        pairs = hbond_pair_rows(str(row["model_id"]), atoms)
        if not pairs.empty:
            pair_frames.append(pairs)
        summary_rows.append(score_summary_row(str(row["model_id"]), atoms, pairs, row))
    pair_df = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame()
    return pair_df, pd.DataFrame(summary_rows)


def top_pairs_table(pairs: pd.DataFrame, max_per_candidate: int = 20) -> pd.DataFrame:
    """Return top donor-acceptor pairs ranked by class and distance."""
    if pairs.empty:
        return pd.DataFrame()
    priority = {
        "strong_geometry": 0,
        "plausible_geometry": 1,
        "weak_geometry": 2,
        "missing_hydrogen_proxy": 3,
        "rejected_too_close": 4,
        "rejected_too_far": 5,
    }
    out = pairs.copy()
    out["_priority"] = out["hbond_class"].map(priority).fillna(9)
    out["_distance_sort"] = pd.to_numeric(out["donor_acceptor_distance_A"], errors="coerce")
    out = out.sort_values(["model_id", "_priority", "_distance_sort"]).copy()
    rows = []
    for model_id, sub in out.groupby("model_id", sort=False):
        for rank, (_, row) in enumerate(sub.head(max_per_candidate).iterrows(), start=1):
            rows.append(
                {
                    "candidate_name": model_id,
                    "rank": rank,
                    "donor_chain": row["donor_chain"],
                    "donor_residue_id": row["donor_resseq"],
                    "donor_residue_name": row["donor_resname"],
                    "donor_atom_name": row["donor_atom_name"],
                    "donor_element": row["donor_element"],
                    "acceptor_chain": row["acceptor_chain"],
                    "acceptor_residue_id": row["acceptor_resseq"],
                    "acceptor_residue_name": row["acceptor_resname"],
                    "acceptor_atom_name": row["acceptor_atom_name"],
                    "acceptor_element": row["acceptor_element"],
                    "pair_type": row["pair_type"],
                    "donor_acceptor_distance_A": row["donor_acceptor_distance_A"],
                    "hbond_class": row["hbond_class"],
                    "pair_score_contribution": row["pair_score_contribution"],
                    "reason_included": row["reason_included"],
                }
            )
    return pd.DataFrame(rows)


def plot_score_summary(summary: pd.DataFrame, path: Path) -> None:
    """Save score summary plot."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    found = summary[summary["status"] == "found"].sort_values("hbond_plausibility_score", ascending=False)
    ax.bar(found["model_id"], pd.to_numeric(found["hbond_plausibility_score"], errors="coerce"), color="#4c78a8")
    ax.set_ylabel("H-bond plausibility score")
    ax.set_title("Candidate H-bond plausibility score")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_distance_distribution(pairs: pd.DataFrame, path: Path) -> None:
    """Save donor-acceptor distance distribution plot."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    if not pairs.empty:
        for model_id, sub in pairs.groupby("model_id"):
            ax.hist(pd.to_numeric(sub["donor_acceptor_distance_A"], errors="coerce"), bins=20, alpha=0.35, label=str(model_id)[:24])
        ax.legend(fontsize=7)
    ax.axvspan(2.6, 3.1, color="#54a24b", alpha=0.15, label="strong")
    ax.axvspan(3.1, 3.4, color="#f2cf5b", alpha=0.12, label="plausible")
    ax.set_xlabel("donor-acceptor distance (A)")
    ax.set_ylabel("pair count")
    ax.set_title("Candidate H-bond donor-acceptor distances")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 12) -> str:
    """Render a compact markdown table."""
    if df.empty:
        return "_No rows._"
    cols = [col for col in columns if col in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in df.head(max_rows)[cols].itertuples(index=False):
        values = [f"{value:.4g}" if isinstance(value, float) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_report(summary: pd.DataFrame, pairs: pd.DataFrame) -> str:
    """Build H-bond plausibility markdown report."""
    found = summary[summary["status"] == "found"]
    missing = summary[summary["status"] != "found"]
    top = found.sort_values("hbond_plausibility_score", ascending=False)
    poor = found[found["hbond_network_classification"].isin(["hbond_network_poor", "hbond_network_marginal"])]
    hydrogens = found["explicit_hydrogens_present"].value_counts().to_dict() if not found.empty else {}
    plateau = found[found["model_id"].str.contains("omega_clean_scale_0p98|omega_clean_scale_0p97", regex=True)]
    plateau_classes = plateau["hbond_network_classification"].value_counts().to_dict() if not plateau.empty else {}
    focus_ids = [
        "omega_clean_scale_1p0000",
        "guarded_full_chain_prototype",
        "omega_clean_scale_0p9825",
        "omega_clean_scale_0p9725",
        "omega_clean_scale_0p9700",
    ]
    focus = summary[summary["model_id"].isin(focus_ids)]
    return f"""# Candidate Hydrogen-Bond Plausibility Report

## Scope And Caveats

This is a hydrogen-bond plausibility score, not a true affinity/free-energy calculation and not a free-energy calculation. Missing hydrogens limit angle-based interpretation. Protonation states and solvent are not modeled. Scores are useful as a comparative physical-sense filter, not as absolute binding energies. Candidate elimination should not rely on this score alone. This filter can complement C/D matching, omega geometry, carboxylate preservation, and torsion-boundary filters.

## Candidate Coverage

- Candidates analyzed: {len(found)}
- Missing candidates: {len(missing)}
- Explicit hydrogens present by candidate count: {hydrogens}

{markdown_table(summary, ["model_id", "status", "explicit_hydrogens_present", "total_candidate_pairs_considered", "hbond_network_classification", "hbond_plausibility_score", "hbond_caveat"], max_rows=14)}

## Missing Candidate Coordinates

{markdown_table(missing, ["model_id", "status", "inferred_family", "path"], max_rows=8)}

## Top H-Bond Plausibility Scores

{markdown_table(top, ["model_id", "hbond_plausibility_score", "hbond_plausibility_score_per_residue", "strong_hbond_count", "plausible_hbond_count", "weak_hbond_count", "interchain_hbond_count", "carboxylate_contact_count"], max_rows=10)}

## Marginal Or Poor Candidates

{markdown_table(poor, ["model_id", "hbond_network_classification", "hbond_plausibility_score", "possible_bad_acceptor_acceptor_contact_count", "possible_bad_donor_donor_contact_count", "hbond_caveat"], max_rows=10)}

## Why Candidates Scored Differently

This section is intended to explain score provenance, not to prove a structure. The hydrogen-bond score remains a heavy-atom proxy because explicit hydrogens are absent. If a poor score is caused by missing atoms, different coordinate representation, or parsing/provenance artifact, it should not be interpreted chemically.

{markdown_table(focus, ["model_id", "donor_atom_count", "acceptor_atom_count", "total_possible_donor_acceptor_pairs", "excluded_same_residue_pairs", "excluded_close_sequence_or_trivial_pairs", "pairs_considered_after_exclusions", "strong_geometry_count", "plausible_geometry_count", "weak_geometry_count", "rejected_too_far_count", "missing_hydrogen_proxy_count", "raw_positive_score_components", "raw_penalty_components", "hbond_plausibility_score", "hbond_network_classification"], max_rows=10)}

- Did the baseline/guarded candidates have donors and acceptors? See `donor_atom_count` and `acceptor_atom_count` above.
- Did they generate candidate pairs? See `pairs_considered_after_exclusions`.
- Were pairs excluded unexpectedly? Compare `excluded_same_residue_pairs` and `excluded_close_sequence_or_trivial_pairs` with total possible pairs.
- Were donor-acceptor distances outside the threshold windows? See `rejected_too_far_count` and the distance summaries in `candidate_hbond_scoring_diagnostics.csv`.
- Were they parsed differently from compressed candidates? Compare atom, residue, chain, donor, and acceptor counts in the diagnostic CSV.
- Does a 0 score look chemically meaningful or likely a parsing/provenance artifact? This diagnostic reports the mechanics, but interpretation should remain cautious until representation and atom typing are confirmed.
- Does 0p9700 scoring like the plateau mean the H-bond filter does not discriminate over-compression? In this heavy-atom proxy pass, yes: if 0p9700 has the same score components, this filter is not discriminating that over-compressed endpoint.
- The hydrogen-bond filter should be used only as a plausibility/supporting filter rather than a rejection filter at this stage.

## Interpretation

- Explicit hydrogens were generally absent in these candidate coordinates, so scores are based on heavy-atom donor-acceptor geometry proxies rather than true donor-H-acceptor angles.
- The omega-clean rise-compressed plateau classifications are: {plateau_classes}.
- Candidates with many strong/plausible interchain contacts and few bad same-type close contacts pass this physical-sense filter more cleanly.
- The over-compressed endpoint should be checked against the plateau rows for degraded H-bond plausibility or increased bad contacts.
- C/D-compatible candidates with marginal or poor H-bond geometry remain chemically suspect and should be treated as lower-priority until atom typing, hydrogens, protonation states, and solvent are handled more explicitly.
- This H-bond plausibility score can help item 6 physical-sense filtering, but only as one comparative diagnostic beside C/D agreement and geometry guards.
"""


def run(max_candidates: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run H-bond plausibility analysis and write outputs."""
    inventory = inventory_dataframe(expected_candidate_records())
    pairs, summary = analyze_inventory(inventory, max_candidates=max_candidates)
    top_pairs = top_pairs_table(pairs)
    OUT_PAIRS.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(OUT_PAIRS, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    summary.to_csv(OUT_DIAGNOSTICS, index=False)
    top_pairs.to_csv(OUT_TOP_PAIRS, index=False)
    OUT_REPORT.write_text(build_report(summary, pairs), encoding="utf-8")
    plot_score_summary(summary, FIG_SCORE)
    plot_distance_distribution(pairs, FIG_DISTANCE)
    return pairs, summary


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-candidates", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    pairs, summary = run(max_candidates=args.max_candidates)
    found = summary[summary["status"] == "found"]
    print(f"Candidates summarized: {len(summary)} ({len(found)} found)")
    print(f"H-bond pair rows: {len(pairs)}")
    for row in found.sort_values("hbond_plausibility_score", ascending=False).head(5).itertuples(index=False):
        print(f"  {row.model_id}: score={row.hbond_plausibility_score} class={row.hbond_network_classification}")
    print(f"Report: {OUT_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
