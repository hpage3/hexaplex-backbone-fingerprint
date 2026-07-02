from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]

MANIFEST = ROOT / "outputs" / "metrics" / "constrained_phi_psi_candidate_manifest.csv"
OUT_CSV = ROOT / "outputs" / "metrics" / "constrained_phi_psi_candidate_geometry_audit.csv"
OUT_REPORT = ROOT / "outputs" / "reports" / "constrained_phi_psi_candidate_geometry_audit.md"


@dataclass(frozen=True)
class Atom:
    record: str
    serial: int
    name: str
    altloc: str
    resname: str
    chain: str
    resseq: int
    icode: str
    x: float
    y: float
    z: float

    @property
    def key(self) -> tuple:
        return (
            self.record,
            self.chain,
            self.resseq,
            self.icode,
            self.resname,
            self.name,
            self.altloc,
        )

    @property
    def coord(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


def parse_pdb(path: Path) -> list[Atom]:
    atoms: list[Atom] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            atoms.append(
                Atom(
                    record=line[0:6].strip(),
                    serial=int(line[6:11]),
                    name=line[12:16].strip(),
                    altloc=line[16:17].strip(),
                    resname=line[17:20].strip(),
                    chain=line[21:22].strip(),
                    resseq=int(line[22:26]),
                    icode=line[26:27].strip(),
                    x=float(line[30:38]),
                    y=float(line[38:46]),
                    z=float(line[46:54]),
                )
            )
    return atoms


def distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def vec(a, b):
    return (b[0] - a[0], b[1] - a[1], b[2] - a[2])


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm(a):
    return math.sqrt(dot(a, a))


def angle_deg(a, b, c) -> float:
    ba = vec(b, a)
    bc = vec(b, c)
    denom = norm(ba) * norm(bc)
    if denom == 0:
        return float("nan")
    cosang = max(-1.0, min(1.0, dot(ba, bc) / denom))
    return math.degrees(math.acos(cosang))


def cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def dihedral_deg(p0, p1, p2, p3) -> float:
    b0 = vec(p1, p0)
    b1 = vec(p1, p2)
    b2 = vec(p2, p3)

    b1n = norm(b1)
    if b1n == 0:
        return float("nan")
    b1u = tuple(x / b1n for x in b1)

    v = tuple(b0[i] - dot(b0, b1u) * b1u[i] for i in range(3))
    w = tuple(b2[i] - dot(b2, b1u) * b1u[i] for i in range(3))

    x = dot(v, w)
    y = dot(cross(b1u, v), w)
    return math.degrees(math.atan2(y, x))


def trans_deviation_deg(omega: float) -> float:
    return abs(180.0 - abs(omega))


def find_parent_pdb() -> Path:
    path = (
        ROOT
        / "outputs"
        / "coordinates"
        / "ideal_hexaflex_variants"
        / "ideal_hexaflex_backbone_plus_carboxylate.pdb"
    )
    if not path.exists():
        raise FileNotFoundError(f"Could not find parent/source PDB: {path}")
    return path

def atom_map(atoms: list[Atom]) -> dict[tuple, Atom]:
    seen: dict[tuple, int] = {}
    mapped: dict[tuple, Atom] = {}
    for atom in atoms:
        base = atom.key
        idx = seen.get(base, 0)
        seen[base] = idx + 1
        mapped[base + (idx,)] = atom
    return mapped


def by_residue(atoms: list[Atom]) -> dict[tuple[str, int, str, str], dict[str, Atom]]:
    residues: dict[tuple[str, int, str, str], dict[str, Atom]] = {}
    for atom in atoms:
        key = (atom.chain, atom.resseq, atom.icode, atom.resname)
        residues.setdefault(key, {})[atom.name] = atom
    return residues


def sorted_residue_keys(residues: dict) -> list[tuple[str, int, str, str]]:
    return sorted(residues.keys(), key=lambda k: (k[0], k[1], k[2], k[3]))


def backbone_bonds(atoms: list[Atom]) -> list[tuple[str, tuple, tuple, float]]:
    residues = by_residue(atoms)
    keys = sorted_residue_keys(residues)
    rows = []

    for key in keys:
        r = residues[key]
        for a_name, b_name in [("N", "CA"), ("CA", "C"), ("C", "O")]:
            if a_name in r and b_name in r:
                rows.append((f"{a_name}-{b_name}", key, key, distance(r[a_name].coord, r[b_name].coord)))

    by_chain: dict[str, list[tuple[str, int, str, str]]] = {}
    for key in keys:
        by_chain.setdefault(key[0], []).append(key)

    for chain, chain_keys in by_chain.items():
        for k1, k2 in zip(chain_keys, chain_keys[1:]):
            r1 = residues[k1]
            r2 = residues[k2]
            if "C" in r1 and "N" in r2:
                rows.append(("C-N_next", k1, k2, distance(r1["C"].coord, r2["N"].coord)))

    return rows


def backbone_angles(atoms: list[Atom]) -> list[tuple[str, tuple, tuple, float]]:
    residues = by_residue(atoms)
    keys = sorted_residue_keys(residues)
    rows = []

    for key in keys:
        r = residues[key]
        if all(name in r for name in ("N", "CA", "C")):
            rows.append(("N-CA-C", key, key, angle_deg(r["N"].coord, r["CA"].coord, r["C"].coord)))

    by_chain: dict[str, list[tuple[str, int, str, str]]] = {}
    for key in keys:
        by_chain.setdefault(key[0], []).append(key)

    for chain_keys in by_chain.values():
        for k1, k2 in zip(chain_keys, chain_keys[1:]):
            r1 = residues[k1]
            r2 = residues[k2]
            if all(name in r1 for name in ("CA", "C")) and all(name in r2 for name in ("N", "CA")):
                rows.append(("CA-C-N_next", k1, k2, angle_deg(r1["CA"].coord, r1["C"].coord, r2["N"].coord)))
                rows.append(("C-N_next-CA_next", k1, k2, angle_deg(r1["C"].coord, r2["N"].coord, r2["CA"].coord)))

    return rows


def omega_values(atoms: list[Atom]) -> list[float]:
    residues = by_residue(atoms)
    by_chain: dict[str, list[tuple[str, int, str, str]]] = {}
    for key in sorted_residue_keys(residues):
        by_chain.setdefault(key[0], []).append(key)

    omegas = []
    for chain_keys in by_chain.values():
        for k1, k2 in zip(chain_keys, chain_keys[1:]):
            r1 = residues[k1]
            r2 = residues[k2]
            if all(name in r1 for name in ("CA", "C")) and all(name in r2 for name in ("N", "CA")):
                omegas.append(dihedral_deg(r1["CA"].coord, r1["C"].coord, r2["N"].coord, r2["CA"].coord))
    return [x for x in omegas if not math.isnan(x)]


def max_abs_delta(parent_rows, cand_rows) -> float:
    parent_lookup = {(name, k1, k2): val for name, k1, k2, val in parent_rows}
    deltas = []
    for name, k1, k2, val in cand_rows:
        old = parent_lookup.get((name, k1, k2))
        if old is not None and not math.isnan(val) and not math.isnan(old):
            deltas.append(abs(val - old))
    return max(deltas) if deltas else float("nan")


def audit_candidate(parent_atoms: list[Atom], candidate_path: Path) -> dict[str, object]:
    candidate_atoms = parse_pdb(candidate_path)

    parent_map = atom_map(parent_atoms)
    candidate_map = atom_map(candidate_atoms)

    parent_keys = set(parent_map)
    candidate_keys = set(candidate_map)
    common_keys = sorted(parent_keys & candidate_keys)

    ca_shifts = []
    all_shifts = []
    for key in common_keys:
        p = parent_map[key]
        c = candidate_map[key]
        shift = distance(p.coord, c.coord)
        all_shifts.append(shift)
        if p.name == "CA":
            ca_shifts.append(shift)

    parent_bonds = backbone_bonds(parent_atoms)
    cand_bonds = backbone_bonds(candidate_atoms)
    parent_angles = backbone_angles(parent_atoms)
    cand_angles = backbone_angles(candidate_atoms)
    omegas = omega_values(candidate_atoms)
    omega_devs = [trans_deviation_deg(o) for o in omegas]

    atom_count_match = len(parent_atoms) == len(candidate_atoms)
    labels_preserved = parent_keys == candidate_keys
    max_ca_shift = max(ca_shifts) if ca_shifts else float("nan")
    max_atom_shift = max(all_shifts) if all_shifts else float("nan")
    max_bond_delta = max_abs_delta(parent_bonds, cand_bonds)
    max_angle_delta = max_abs_delta(parent_angles, cand_angles)
    max_omega_trans_dev = max(omega_devs) if omega_devs else float("nan")
    median_omega_trans_dev = median(omega_devs) if omega_devs else float("nan")

    safe = (
        atom_count_match
        and labels_preserved
        and max_ca_shift <= 1e-3
        and max_bond_delta <= 0.05
        and max_angle_delta <= 5.0
        and max_omega_trans_dev <= 15.0
    )

    return {
        "candidate_file_exists": candidate_path.exists(),
        "atom_count_parent": len(parent_atoms),
        "atom_count_candidate": len(candidate_atoms),
        "atom_count_match": atom_count_match,
        "labels_preserved": labels_preserved,
        "missing_label_count": len(parent_keys - candidate_keys),
        "extra_label_count": len(candidate_keys - parent_keys),
        "max_ca_shift_A": max_ca_shift,
        "max_atom_shift_A": max_atom_shift,
        "max_backbone_bond_delta_A": max_bond_delta,
        "max_backbone_angle_delta_deg": max_angle_delta,
        "omega_count": len(omegas),
        "max_omega_trans_deviation_deg": max_omega_trans_dev,
        "median_omega_trans_deviation_deg": median_omega_trans_dev,
        "safe_for_diffraction_scoring": safe,
    }


def main() -> None:
    parent_pdb = find_parent_pdb()
    parent_atoms = parse_pdb(parent_pdb)

    rows = []
    with MANIFEST.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            candidate_path = ROOT / row["coordinate_path"]
            audit = audit_candidate(parent_atoms, candidate_path)
            rows.append({**row, "parent_pdb": str(parent_pdb.relative_to(ROOT)), **audit})

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys()) if rows else []
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    safe_count = sum(str(r["safe_for_diffraction_scoring"]) == "True" for r in rows)
    total = len(rows)

    max_ca = max(float(r["max_ca_shift_A"]) for r in rows) if rows else float("nan")
    max_bond = max(float(r["max_backbone_bond_delta_A"]) for r in rows) if rows else float("nan")
    max_angle = max(float(r["max_backbone_angle_delta_deg"]) for r in rows) if rows else float("nan")
    max_omega = max(float(r["max_omega_trans_deviation_deg"]) for r in rows) if rows else float("nan")

    report = [
        "# Constrained Phi/Psi Candidate Geometry Audit",
        "",
        "This audit checks whether generated constrained phi/psi coordinate candidates are geometrically safe before diffraction scoring.",
        "",
        f"- Parent PDB: `{parent_pdb.relative_to(ROOT)}`",
        f"- Candidates audited: {total}",
        f"- Safe for diffraction scoring: {safe_count}/{total}",
        f"- Maximum C-alpha shift: {max_ca:.6g} A",
        f"- Maximum backbone bond-length deviation from parent: {max_bond:.6g} A",
        f"- Maximum backbone angle deviation from parent: {max_angle:.6g} degrees",
        f"- Maximum omega trans deviation: {max_omega:.6g} degrees",
        "",
        "## Candidate summary",
        "",
        "| candidate_id | repeat_type | solve_mode | endpoint_error_A | max_CA_shift_A | max_bond_delta_A | max_angle_delta_deg | max_omega_trans_dev_deg | safe |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for r in rows:
        report.append(
            "| {candidate_id} | {repeat_type} | {solve_mode} | {endpoint_error_A} | "
            "{max_ca_shift_A:.6g} | {max_backbone_bond_delta_A:.6g} | "
            "{max_backbone_angle_delta_deg:.6g} | {max_omega_trans_deviation_deg:.6g} | "
            "{safe_for_diffraction_scoring} |".format(
                **{
                    **r,
                    "max_ca_shift_A": float(r["max_ca_shift_A"]),
                    "max_backbone_bond_delta_A": float(r["max_backbone_bond_delta_A"]),
                    "max_backbone_angle_delta_deg": float(r["max_backbone_angle_delta_deg"]),
                    "max_omega_trans_deviation_deg": float(r["max_omega_trans_deviation_deg"]),
                }
            )
        )

    report.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Candidates passing this audit are suitable for the next small diffraction-scoring comparison.",
            "- Omega is audited against Nick's current fixed-180-degree policy. Candidates exceeding the trans-deviation threshold are not considered safe for diffraction scoring in this phase.",
            "- Omega sensitivity remains deferred until after the fixed-omega candidate scoring pass.",
        ]
    )

    OUT_REPORT.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"Audited {total} candidates")
    print(f"Safe for diffraction scoring: {safe_count}/{total}")
    print(f"CSV: {OUT_CSV.relative_to(ROOT)}")
    print(f"Report: {OUT_REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()