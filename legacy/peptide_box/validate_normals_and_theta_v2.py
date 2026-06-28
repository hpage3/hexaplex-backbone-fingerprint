#!/usr/bin/env python3
# Validate boxer normals (from *_boxes.pdb) against normals recomputed from the
# original source PDB, and compute the theta_pp fingerprint. ASCII-only.
#
# Usage (Windows examples):
#   python validate_normals_and_theta_v2.py ^
#     --boxes output\4jea_boxes.pdb ^
#     --src   input_data\4jea.pdb ^
#     --out-prefix output\4jea ^
#     --angle-thresh 8
# Optional: --chain A

import argparse
import math
from pathlib import Path
from typing import Dict, Tuple, List, Optional

# ----------------- small vector helpers -----------------
def vdot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def vnorm(a):
    return math.sqrt(vdot(a, a))

def vunit(a):
    n = vnorm(a)
    if n == 0.0:
        return (0.0, 0.0, 0.0)
    return (a[0]/n, a[1]/n, a[2]/n)

def vcross(a, b):
    return (
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    )

def angle_deg_abs(u, v):
    nu = vnorm(u); nv = vnorm(v)
    if nu == 0.0 or nv == 0.0:
        return float('nan')
    d = abs(vdot(u, v) / (nu*nv))
    d = max(-1.0, min(1.0, d))
    return math.degrees(math.acos(d))

# ----------------- source PDB parsing -----------------
class Atom:
    __slots__ = ("name","resname","chain","resseq","icode","x","y","z")
    def __init__(self, name, resname, chain, resseq, icode, x, y, z):
        self.name=name; self.resname=resname; self.chain=chain; self.resseq=resseq; self.icode=icode
        self.x=float(x); self.y=float(y); self.z=float(z)

Key = Tuple[str, int, str]  # (chain, resseq, icode)

def parse_source_pdb(path: Path, chain_sel: Optional[str]) -> Dict[Key, Dict[str, Atom]]:
    bb: Dict[Key, Dict[str, Atom]] = {}
    with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
        for ln in fh:
            if not (ln.startswith('ATOM') or ln.startswith('HETATM')):
                continue
            name = ln[12:16].strip(); resname = ln[17:20].strip()
            chain = (ln[21] or ' ').strip() or ' '
            if chain_sel and chain != chain_sel:
                continue
            try:
                resseq = int(ln[22:26])
            except Exception:
                continue
            icode = (ln[26] or '.').strip() or '.'
            try:
                x = float(ln[30:38]); y = float(ln[38:46]); z = float(ln[46:54])
            except Exception:
                continue
            key = (chain, resseq, icode)
            if key not in bb:
                bb[key] = {}
            bb[key][name] = Atom(name, resname, chain, resseq, icode, x, y, z)
    # keep only residues with at least these atoms used below
    return bb

# ----------------- boxes PDB parsing -----------------
class BoxRecord:
    __slots__ = ("chain","resi","center","pn1","pn2","v1","v2","v3","v4")
    def __init__(self, chain: str, resi: int):
        self.chain = chain; self.resi = resi
        self.center=None; self.pn1=None; self.pn2=None
        self.v1=None; self.v2=None; self.v3=None; self.v4=None

Key2 = Tuple[str, int]  # (chain, resi_i)

def parse_boxes_pdb(path: Path, chain_sel: Optional[str]) -> Dict[Key2, BoxRecord]:
    def xyz(ln: str):
        return (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
    recs: Dict[Key2, BoxRecord] = {}
    with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
        for ln in fh:
            if not (ln.startswith('ATOM') or ln.startswith('HETATM')):
                continue
            name = ln[12:16].strip()
            chain = (ln[21] or ' ').strip() or ' '
            if chain_sel and chain != chain_sel:
                continue
            try:
                resi = int(ln[22:26])
            except Exception:
                continue
            key = (chain, resi)
            if key not in recs:
                recs[key] = BoxRecord(chain, resi)
            if name == 'PCEN': recs[key].center = xyz(ln)
            elif name == 'PN1': recs[key].pn1 = xyz(ln)
            elif name == 'PN2': recs[key].pn2 = xyz(ln)
            elif name == 'PV1': recs[key].v1 = xyz(ln)
            elif name == 'PV2': recs[key].v2 = xyz(ln)
            elif name == 'PV3': recs[key].v3 = xyz(ln)
            elif name == 'PV4': recs[key].v4 = xyz(ln)
    return recs

# ----------------- normals -----------------

def recompute_link_normal(bb, chain, resi_i):
    """
    True peptide-link plane normal for i -> i+1 using exactly (C_i, O_i, N_{i+1}).
    - Enforces immediate successor (i+1) — never skips across gaps.
    - Ignores CA entirely (prevents tilt).
    - Returns a unit vector or None if any atom is missing.
    bb keys are expected like: (chain, resseq, icode?) -> {'N','CA','C','O': atom}
    """

    # find best key for residue (chain, i) and (chain, i+1), respecting insertion codes
    def find_key(bb, ch, resi):
        cand = [k for k in bb.keys() if k[0] == ch and k[1] == resi]
        if not cand: 
            return None
        # prefer blank icode first, then alphabetical
        def icode_key(k):
            ic = k[2] if len(k) > 2 else ''
            return (0, '') if ic in (None, '', ' ') else (1, str(ic))
        return sorted(cand, key=icode_key)[0]

    ki = find_key(bb, chain, resi_i)
    kj = find_key(bb, chain, resi_i + 1)
    if ki is None or kj is None:
        return None  # require immediate neighbor only

    at_i = bb.get(ki, {})
    at_j = bb.get(kj, {})

    Ci = at_i.get('C');  Oi = at_i.get('O');  Nj = at_j.get('N')
    if not (Ci and Oi and Nj):
        return None

    # vectors in the SAME plane: C_i→O_i and C_i→N_{i+1}
    v1 = (Oi.x - Ci.x, Oi.y - Ci.y, Oi.z - Ci.z)
    v2 = (Nj.x - Ci.x, Nj.y - Ci.y, Nj.z - Ci.z)

    # cross gives plane normal; normalize
    n = vcross(v1, v2)
    return vunit(n) if vnorm(n) > 1e-12 else None

# theta_pp from residue-plane normals (N-CA-C)

def residue_plane_normal(at: Dict[str,Atom]) -> Optional[Tuple[float,float,float]]:
    if not ('N' in at and 'CA' in at and 'C' in at):
        return None
    u = (at['CA'].x - at['N'].x,  at['CA'].y - at['N'].y,  at['CA'].z - at['N'].z)
    v = (at['C'].x  - at['CA'].x, at['C'].y  - at['CA'].y, at['C'].z  - at['CA'].z)
    return vunit(vcross(u, v))

# ----------------- main -----------------

def main():
    ap = argparse.ArgumentParser(description='Validate boxer normals vs recomputed normals and compute theta_pp')
    ap.add_argument('--boxes', required=True)
    ap.add_argument('--src', required=True)
    ap.add_argument('--out-prefix', required=True)
    ap.add_argument('--chain', default=None)
    ap.add_argument('--angle-thresh', type=float, default=8.0)
    args = ap.parse_args()

    boxes_p = Path(args.boxes).resolve(); src_p = Path(args.src).resolve()
    out_prefix = args.out_prefix

    bb = parse_source_pdb(src_p, args.chain)
    recs = parse_boxes_pdb(boxes_p, args.chain)

    # choose boxer normals per record
    def boxer_normal(rec: BoxRecord) -> Tuple[Optional[Tuple[float,float,float]], str]:
        if rec.pn1 and rec.pn2:
            v = (rec.pn2[0]-rec.pn1[0], rec.pn2[1]-rec.pn1[1], rec.pn2[2]-rec.pn1[2])
            return vunit(v), 'pn'
        if rec.v1 and rec.v2 and rec.v4:
            e1 = (rec.v2[0]-rec.v1[0], rec.v2[1]-rec.v1[1], rec.v2[2]-rec.v1[2])
            e2 = (rec.v4[0]-rec.v1[0], rec.v4[1]-rec.v1[1], rec.v4[2]-rec.v1[2])
            return vunit(vcross(e1, e2)), 'pv'
        return None, 'none'

    # per-link validation table
    rows = []
    flagged = []
    keys = sorted(recs.keys(), key=lambda k:(k[0], k[1]))
    for chain, resi_i in keys:
        rec = recs[(chain, resi_i)]
        n_box, src = boxer_normal(rec)
        n_calc = recompute_link_normal(bb, chain, resi_i)
        if not n_calc or not n_box:
            continue
        ang = angle_deg_abs(n_calc, n_box)
        dot = abs(vdot(n_calc, n_box))
        rows.append((chain, resi_i, src, f"{ang:.3f}", f"{dot:.5f}"))
        if ang > args.angle_thresh:
            flagged.append((chain, resi_i, f"{ang:.3f}"))

    # write tables
    # write tables
    if rows:
        with open(f"{out_prefix}_normals_validation.tsv", "w", encoding="utf-8") as fh:
            fh.write("chain\tresseq\tsource\tangle_err_deg\tdot\n")
            for r in rows:
                fh.write("\t".join(map(str, r)) + "\n")   # <-- map(str, r)

    if flagged:
        with open(f"{out_prefix}_flags.tsv", "w", encoding="utf-8") as fh:
            fh.write("chain\tresseq\tangle_err_deg\n")
            for r in flagged:
                fh.write("\t".join(map(str, r)) + "\n")   # already safe, keep as-is

    # summary
    mean_err = sum(float(r[3]) for r in rows)/len(rows) if rows else float('nan')
    worst = max(float(r[3]) for r in rows) if rows else float('nan')
    with open(f"{out_prefix}_summary.txt", 'w', encoding='utf-8') as fh:
        fh.write(f"Validated links: {len(rows)}\n")
        fh.write(f"Flagged > {args.angle_thresh} deg: {len(flagged)}\n")
        fh.write(f"Mean abs error (deg): {mean_err:.3f}\n")
        fh.write(f"Worst-case error (deg): {worst:.3f}\n")

    # theta_pp from residue-plane normals (N-CA-C) along each chain
    # build ordered residue lists per chain
    per_chain: Dict[str, List[Key]] = {}
    for k in bb.keys():
        per_chain.setdefault(k[0], []).append(k)
    for c in per_chain:
        per_chain[c].sort(key=lambda k:(k[1], k[2]))
    with open(f"{out_prefix}_theta_pp.tsv", 'w', encoding='utf-8') as fh:
        fh.write("index_along_chain\ttheta_pp_deg\tchain\tresseq\ticode\n")
        for c, klst in per_chain.items():
            # compute normals per residue
            rn = []  # (key, normal)
            for k in klst:
                n = residue_plane_normal(bb[k])
                if n is not None:
                    rn.append((k, n))
            for i in range(len(rn)-1):
                k1, n1 = rn[i]; k2, n2 = rn[i+1]
                th = angle_deg_abs(n1, n2)
                fh.write(f"{i}\t{th:.3f}\t{c}\t{k1[1]}\t{k1[2]}\n")

    print(f"[ok] Wrote: {out_prefix}_normals_validation.tsv")
    print(f"[ok] Wrote: {out_prefix}_flags.tsv")
    print(f"[ok] Wrote: {out_prefix}_theta_pp.tsv")
    print(f"[ok] Wrote: {out_prefix}_summary.txt")

if __name__ == '__main__':
    main()
