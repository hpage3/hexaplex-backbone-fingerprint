#!/usr/bin/env python3
# Validate boxer normals (from *_boxes.pdb) against normals recomputed from the
# original source PDB, and compute per-link omega. ASCII-only.
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
from typing import Dict, Tuple, List, Optional

# ----------------- small vector ops -----------------

def vdot(a, b): return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def vcross(a,b):
    return (a[1]*b[2]-a[2]*b[1],
            a[2]*b[0]-a[0]*b[2],
            a[0]*b[1]-a[1]*b[0])

def vsub(a,b): return (a[0]-b[0], a[1]-b[1], a[2]-b[2])

def vnorm(a):
    return math.sqrt(vdot(a,a))

def vunit(a):
    n = vnorm(a)
    if n < 1e-15: return (0.0,0.0,0.0)
    return (a[0]/n, a[1]/n, a[2]/n)

def angle_deg_abs(a, b):
    na = vnorm(a); nb = vnorm(b)
    if na < 1e-15 or nb < 1e-15: return 0.0
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
    ang = math.degrees(math.atan2(y, x))
    return ang

# ----------------- boxer rec parsing -----------------

class Rec:
    __slots__ = ('chain','resseq','icode','pv','pn','src')
    def __init__(self, chain, resseq, icode, pv, pn, src):
        self.chain = chain
        self.resseq = resseq
        self.icode = icode
        self.pv = pv         # 4 in-plane corners [(x,y,z)*4]
        self.pn = pn         # PN1, PN2 points [(x,y,z)*2]
        self.src = src       # source tag

def parse_boxes_pdb(path: Path, chain_sel: Optional[str]) -> Dict[Tuple[str,int], Rec]:
    recs: Dict[Tuple[str,int], Rec] = {}
    with open(path, 'r', encoding='ascii', errors='ignore') as fh:
        pv = []; pn = []; chain=None; resseq=None; icode='.'
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
        # final flush
        if block_active and chain is not None and resseq is not None and len(pv)>=4 and len(pn)>=2:
            recs[(chain, resseq)] = Rec(chain, resseq, icode, pv[:4], pn[:2], src)
    return recs

def boxer_normal(rec: Rec):
    # PN1->PN2 defines the normal direction
    (x1,y1,z1),(x2,y2,z2) = rec.pn
    n = (x2-x1, y2-y1, z2-z1)
    return vunit(n), rec.src

# ----------------- source PDB parsing -----------------

class Atom:
    __slots__=('name','resname','chain','resseq','icode','x','y','z')
    def __init__(self, name, resname, chain, resseq, icode, x, y, z):
        self.name=name; self.resname=resname; self.chain=chain; self.resseq=resseq; self.icode=icode
        self.x=float(x); self.y=float(y); self.z=float(z)

Key = Tuple[str, int, str]  # (chain, resseq, icode)

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
    """
    Return a peptide-link plane normal that aligns with the boxer/SVD normal.
    Strategy:
      - Compute both planes:
          forward  : (C_i,     O_i,     N_{i+1})
          backward : (C_{i-1}, O_{i-1}, N_i)
      - If n_box is provided, pick the candidate with the larger |dot(n, n_box)|
        and flip its sign if needed so it lies in the same hemisphere as n_box.
    Adds a safeguard: if the O atom is >0.8 Å out of the canonical C–CA–N plane,
    mark as "oxygen skew" and skip (return None).
    """

    OXY_SKEW_THRESH = 0.8  # angstroms

    def find_key(bb, ch, resi):
        cand = [k for k in bb.keys() if k[0] == ch and k[1] == resi]
        if not cand: 
            return None
        def icode_key(k):
            ic = k[2] if len(k) > 2 else ''
            blank = (ic is None) or (ic in ('', ' ', '.'))
            return (0, '') if blank else (1, str(ic))
        return sorted(cand, key=icode_key)[0]

    def plane_normal(C, O, N):
        v1 = (O.x - C.x, O.y - C.y, O.z - C.z)
        v2 = (N.x - C.x, N.y - C.y, N.z - C.z)
        n  = vcross(v1, v2)
        return vunit(n) if vnorm(n) > 1e-12 else None

    def oxygen_offset(C, CA, N, O):
        # normal of canonical peptide plane (C,CA,N)
        v1 = (CA.x - C.x, CA.y - C.y, CA.z - C.z)
        v2 = (N.x - C.x,  N.y - C.y,  N.z - C.z)
        n  = vunit(vcross(v1, v2))
        vO = (O.x - C.x, O.y - C.y, O.z - C.z)
        return abs(vdot(vO, n))

    # keys and atom maps
    ki    = find_key(bb, chain, resi_i)
    kj    = find_key(bb, chain, resi_i + 1)
    kim1  = find_key(bb, chain, resi_i - 1)
    at_i   = bb.get(ki,   {}) if ki   else {}
    at_j   = bb.get(kj,   {}) if kj   else {}
    at_im1 = bb.get(kim1, {}) if kim1 else {}

    # forward (i -> i+1): (C_i, O_i, N_{i+1})
    n_fwd = None
    Ci, Oi, Nj, CAi = at_i.get('C'), at_i.get('O'), at_j.get('N'), at_i.get('CA')
    if Ci and Oi and Nj:
        if CAi:
            d = oxygen_offset(Ci, CAi, Nj, Oi)
            if d <= OXY_SKEW_THRESH:
                n_fwd = plane_normal(Ci, Oi, Nj)
            else:
                # O is skewed too far, treat as unreliable
                n_fwd = None
        else:
            n_fwd = plane_normal(Ci, Oi, Nj)

    # backward (i-1 -> i): (C_{i-1}, O_{i-1}, N_i)
    n_bwd = None
    Cim1, Oim1, Ni, CAim1 = at_im1.get('C'), at_im1.get('O'), at_i.get('N'), at_im1.get('CA')
    if Cim1 and Oim1 and Ni:
        if CAim1:
            d = oxygen_offset(Cim1, CAim1, Ni, Oim1)
            if d <= OXY_SKEW_THRESH:
                n_bwd = plane_normal(Cim1, Oim1, Ni)
            else:
                n_bwd = None
        else:
            n_bwd = plane_normal(Cim1, Oim1, Ni)

    # choose candidate: prefer forward by default; if n_box provided, pick closest
    cand = n_fwd or n_bwd
    if n_fwd and n_bwd:
        cand = n_fwd if n_box is None else (n_fwd if abs(vdot(n_fwd, n_box)) >= abs(vdot(n_bwd, n_box)) else n_bwd)

    return cand
