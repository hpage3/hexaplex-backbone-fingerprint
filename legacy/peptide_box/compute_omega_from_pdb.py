#!/usr/bin/env python3
"""
extract_omega.py – compute ω (CA–C–N–CA) dihedral angles from a PDB file.

Usage:
    python extract_omega.py my_structure.pdb
Outputs:
    <basename>_omega.csv  with columns: chain, res_i, res_j, omega_deg
"""

import sys, math, numpy as np, pandas as pd
from pathlib import Path

def dihedral(p1, p2, p3, p4):
    """Return dihedral angle in degrees for 4 points (CA–C–N–CA)."""
    b0 = -1.0 * (p2 - p1)
    b1 = p3 - p2
    b2 = p4 - p3
    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return math.degrees(math.atan2(y, x))

def extract_omega_from_pdb(pdb_path):
    residues = {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            atom = line[12:16].strip()
            resname = line[17:20].strip()
            chain = line[21].strip()
            resseq = int(line[22:26])
            x, y, z = map(float, (line[30:38], line[38:46], line[46:54]))
            residues.setdefault((chain, resseq, resname), {})[atom] = np.array([x, y, z])

    records = []
    keys = sorted(residues.keys(), key=lambda k: (k[0], k[1]))
    for i in range(len(keys)-1):
        ch_i, res_i, _ = keys[i]
        ch_j, res_j, _ = keys[i+1]
        if ch_i != ch_j:
            continue
        at_i = residues[keys[i]]
        at_j = residues[keys[i+1]]
        if not all(a in at_i for a in ("CA", "C")) or not all(a in at_j for a in ("N", "CA")):
            continue
        CAi, Ci, Nj, CAj = at_i["CA"], at_i["C"], at_j["N"], at_j["CA"]
        omega = dihedral(CAi, Ci, Nj, CAj)
        records.append({"chain": ch_i, "res_i": res_i, "res_j": res_j, "omega_deg": omega})
    return pd.DataFrame(records)

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_omega.py <pdb_file>")
        sys.exit(1)

    pdb_path = Path(sys.argv[1])
    if not pdb_path.exists():
        print(f"Error: file not found {pdb_path}")
        sys.exit(2)

    df = extract_omega_from_pdb(pdb_path)
    out_csv = pdb_path.with_name(pdb_path.stem + "_omega.csv")
    df.to_csv(out_csv, index=False)
    print(f"[ok] Wrote {out_csv} with {len(df)} omega angles")

if __name__ == "__main__":
    main()
