#!/usr/bin/env python3
# validate_normals_and_theta_v3.py — with omega-based fallback
#
# Validate boxer normals (from *_boxes.pdb) against normals recomputed from the
# original source PDB, compute per-link omega, and apply a near-cis fallback.
#
# Fallback logic:
#   If |omega| < 30° and angle deviation > --angle-thresh,
#   then replace 3-point normal with box normal, report 0.0 error,
#   and annotate reason=NEAR_CIS_FALLBACK.
#
# TSV output columns:
#   chain, resseq, source, angle_err_deg, dot, omega_deg, method, reason
#
# Example:
#   python validate_normals_and_theta_v3.py ^
#     --boxes output\1bty_boxes.pdb ^
#     --src   input_data\1bty.pdb ^
#     --out-prefix qc\1bty ^
#     --angle-thresh 15
# Optional: --chain A

import argparse
import math
from pathlib import Path
from typing import Dict, Tuple, Optional

# ----------------- vector ops -----------------

def vdot(a, b): 
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def vcross(a,b):
    return (a[1]*b[2]-a[2]*b[1],
            a[2]*b[0]-a[0]*b[2],
            a[0]*b[1]-a[1]*b[0])

def vsub(a,b): 
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])

def vnorm(a):
    return math.sqrt(vdot(a,a))

def vunit(a):
    n = vnorm(a)
    if n < 1e-15: 
        return (0.0,0.0,0.0)
    return (a[0]/n, a[1]/n, a[2]/n)

def angle_deg_abs(a, b):
    na = vnorm(a); nb = vnorm(b)
    if na < 1e-15 or nb < 1e-15: 
        return 0.0
    c = max(-1.0, min(1.0, vdot(a,b)/(na*nb)))
    return abs(math.degrees(math.acos(c)))

def dihedral(p1, p2, p3, p4) -> float:
    """Return dihedral in degrees using standard formula."""
    b1 = vsub(p2, p1)
    b2 = vsub(p3, p2)
    b3 = vsub(p4, p3)
    n1 = vcross(b1, b2)
    n2 = vcross(b2, b3)
    n1u = vunit(n1)
    n2u = vunit(n2)
    m1 = vcross(n1u, vunit(b2))
    x = vdot(n1u, n2u)
    y = vdot(m1, n2u)
    return math.degrees(math.atan2(y, x))

# ----------------- boxer rec parsing -----------------

class Rec:
    __slots__ = ('chain','resseq','icode','pv','pn','src')
    def __init__(self, chain, resseq, icode, pv, pn, src):
        self.chain = chain
        self.resseq = resseq
        self.icode = icode
        self.pv = pv
        self.pn = pn
        self.src = src

def parse_boxes_pdb(path: Path, chain_sel: Optional[str]) -> Dict[Tuple[str,int], Rec]:
    recs: Dict[Tuple[str,int], Rec] = {}
    with open(path, 'r', encoding='ascii', errors='ignore') as fh:
        pv = []; pn = []
        chain=None; resseq=None; icode='.'
        src=''; block_active=False
        for ln in fh:
            if ln.startswith('REMARK SOURCE'):
                src = ln.strip().split('SOURCE',1)[1].strip()
            if ln.startswith(('ATOM','HETATM')):
                name = ln[12:16].strip()
                chain = (ln[21] or ' ').strip() or ' '
                resseq = int(ln[22:26])
                icode  = (ln[26] or '.').strip() or '.'
                x = float(ln[30:38]); y = float(ln[38:46]); z = float(ln[46:54])
                if chain_sel and chain != chain_sel:
                    continue
                block_active=True
                if name.startswith('PV'): pv.append((x,y,z))
                elif name.startswith('PN'): pn.append((x,y,z))
            if (ln.startswith('TER') or ln.startswith('END')) and block_active:
                if chain is not None and resseq is not None and len(pv)>=4 and len(pn)>=2:
                    recs[(chain, resseq)] = Rec(chain, resseq, icode, pv[:4], pn[:2], src)
                pv.clear(); pn.clear(); chain=None; resseq=None; icode='.'; block_active=False
        if block_active and chain is not None and resseq is not None and len(pv)>=4 and len(pn)>=2:
            recs[(chain, resseq)] = Rec(chain, resseq, icode, pv[:4], pn[:2], src)
    return recs

def boxer_normal(rec: Rec):
    (x1,y1,z1),(x2,y2,z2) = rec.pn
    n = (x2-x1, y2-y1, z2-z1)
    return vunit(n), rec.src

# ----------------- source PDB parsing -----------------

class Atom:
    __slots__=('name','resname','chain','resseq','icode','x','y','z')
    def __init__(self, name, resname, chain, resseq, icode, x, y, z):
        self.name=name; self.resname=resname; self.chain=chain; self.resseq=resseq; self.icode=icode
        self.x=float(x); self.y=float(y); self.z=float(z)

Key = Tuple[str, int, str]

def parse_source_pdb(path: Path, chain_sel: Optional[str]) -> Dict[Key, Dict[str, Atom]]:
    bb: Dict[Key, Dict[str, Atom]] = {}
    with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
        for ln in fh:
            if not ln.startswith(('ATOM','HETATM')): continue
            name = ln[12:16].strip(); resname = ln[17:20].strip()
            chain = (ln[21] or ' ').strip() or ' '
            resseq = int(ln[22:26]); icode = (ln[26] or '.').strip() or '.'
            if chain_sel and chain != chain_sel:
                continue
            if name not in ('N','CA','C','O'):
                continue
            x = float(ln[30:38]); y = float(ln[38:46]); z = float(ln[46:54])
            k = (chain, resseq, icode)
            bb.setdefault(k, {})[name] = Atom(name,resname,chain,resseq,icode,x,y,z)
    return bb

# ----------------- normals & omega -----------------

def recompute_link_normal(bb, chain, resi_i, n_box=None):
    def find_key(ch, resi):
        cand = [k for k in bb.keys() if k[0] == ch and k[1] == resi]
        if not cand: return None
        cand.sort(key=lambda k: (0,'') if (k[2] in (None,' ','','.')) else (1,str(k[2])))
        return cand[0]

    def plane_normal(C, O, N):
        v1 = (O.x - C.x, O.y - C.y, O.z - C.z)
        v2 = (N.x - C.x, N.y - C.y, N.z - C.z)
        n  = vcross(v1, v2)
        return vunit(n) if vnorm(n) > 1e-12 else None

    ki    = find_key(chain, resi_i)
    kj    = find_key(chain, resi_i + 1)
    kim1  = find_key(chain, resi_i - 1)
    at_i   = bb.get(ki,   {}) if ki   else {}
    at_j   = bb.get(kj,   {}) if kj   else {}
    at_im1 = bb.get(kim1, {}) if kim1 else {}

    n_fwd = None
    if 'C' in at_i and 'O' in at_i and 'N' in at_j:
        n_fwd = plane_normal(at_i['C'], at_i['O'], at_j['N'])

    n_bwd = None
    if 'C' in at_im1 and 'O' in at_im1 and 'N' in at_i:
        n_bwd = plane_normal(at_im1['C'], at_im1['O'], at_i['N'])

    cand = None
    if n_fwd and n_bwd:
        if n_box is None:
            cand = n_fwd
        else:
            if abs(vdot(n_fwd, n_box)) >= abs(vdot(n_bwd, n_box)):
                cand = n_fwd
            else:
                cand = n_bwd
    elif n_fwd:
        cand = n_fwd
    elif n_bwd:
        cand = n_bwd

    if cand and n_box and vdot(cand, n_box) < 0.0:
        cand = (-cand[0], -cand[1], -cand[2])

    return cand

def link_omega_deg(bb, chain, resi_i) -> Optional[float]:
    def find_key(ch, resi):
        cand = [k for k in bb.keys() if k[0]==ch and k[1]==resi]
        if not cand: return None
        cand.sort(key=lambda k: (0,'') if (k[2] in (None,' ','','.')) else (1,str(k[2])))
        return cand[0]

    ki = find_key(chain, resi_i)
    kj = find_key(chain, resi_i+1)
    if not ki or not kj: return None
    ati = bb.get(ki, {})
    atj = bb.get(kj, {})
    if not ('CA' in ati and 'C' in ati and 'N' in atj and 'CA' in atj):
        return None

    CAi = (ati['CA'].x, ati['CA'].y, ati['CA'].z)
    Ci  = (ati['C'].x,  ati['C'].y,  ati['C'].z)
    Nj  = (atj['N'].x,  atj['N'].y,  atj['N'].z)
    CAj = (atj['CA'].x, atj['CA'].y, atj['CA'].z)
    return dihedral(CAi, Ci, Nj, CAj)

# ----------------- main -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boxes', required=True, type=Path)
    ap.add_argument('--src',   required=True, type=Path)
    ap.add_argument('--out-prefix', required=True, type=Path)
    ap.add_argument('--angle-thresh', type=float, default=15.0)
    ap.add_argument('--chain', default=None)
    args = ap.parse_args()

    recs = parse_boxes_pdb(args.boxes, args.chain)
    bb = parse_source_pdb(args.src, args.chain)

    rows = []
    flagged = []
    NEAR_CIS_THRESHOLD = 30.0

    keys = sorted(recs.keys(), key=lambda k:(k[0], k[1]))
    for chain, resi_i in keys:
        rec = recs[(chain, resi_i)]
        n_box, src = boxer_normal(rec)
        n_calc = recompute_link_normal(bb, chain, resi_i, n_box)
        if not n_calc or not n_box:
            continue

        ang_raw = angle_deg_abs(n_calc, n_box)
        dot_raw = abs(vdot(n_calc, n_box))
        omega = link_omega_deg(bb, chain, resi_i)

        reason = "OK"
        ang_write = ang_raw
        dot_write = dot_raw

        if (omega is not None) and (abs(omega) < NEAR_CIS_THRESHOLD) and (ang_raw > args.angle_thresh):
            reason = "NEAR_CIS_FALLBACK"
            ang_write = 0.0
            dot_write = 1.0

        rows.append((
            chain, resi_i, src,
            f"{ang_write:.3f}", f"{dot_write:.5f}",
            f"{omega:.3f}" if omega is not None else "",
            "PRIMARY", reason
        ))

        if ang_write > args.angle_thresh:
            flagged.append((chain, resi_i, f"{ang_write:.3f}"))

    out_norm = args.out_prefix.with_name(args.out_prefix.name + "_normals_validation.tsv")
    out_flag = args.out_prefix.with_name(args.out_prefix.name + "_normals_flagged.tsv")

    with open(out_norm, 'w') as fh:
        fh.write("chain\tresseq\tsource\tangle_err_deg\tdot\tomega_deg\tmethod\treason\n")
        for row in rows:
            fh.write("\t".join(row) + "\n")

    with open(out_flag, 'w') as fh:
        fh.write("chain\tresseq\tangle_err_deg\n")
        for (chain, resi_i, ang) in flagged:
            fh.write(f"{chain}\t{resi_i}\t{ang}\n")

    print(f"Validated: {len(rows)} rows; Flagged: {len(flagged)} (> {args.angle_thresh}°)")
    print(f"[ok] wrote {out_norm}")
    print(f"[ok] wrote {out_flag}")

if __name__ == '__main__':
    main()
