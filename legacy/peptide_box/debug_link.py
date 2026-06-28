# debug_link.py  -- draw true C-O-N link-plane normal and print diagnostics
from pymol import cmd
import math

def _vsub(a,b): return (a[0]-b[0], a[1]-b[1], a[2]-b[2])
def _vdot(a,b): return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]
def _vlen(a): return math.sqrt(max(1e-18, _vdot(a,a)))
def _vunit(a): L=_vlen(a); return (a[0]/L, a[1]/L, a[2]/L)
def _vcross(a,b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])

def _cgo_arrow(name, p0, p1, radius=0.15, head=0.45, color=(0.2,0.8,1.0)):
    from pymol.cgo import CYLINDER, CONE
    ax = _vsub(p1, p0); au = _vunit(ax); L = _vlen(ax)
    mid = (p0[0]+(L-head)*au[0], p0[1]+(L-head)*au[1], p0[2]+(L-head)*au[2])
    obj = [
        CYLINDER, p0[0],p0[1],p0[2], mid[0],mid[1],mid[2], radius,
        color[0],color[1],color[2], color[0],color[1],color[2],
        CONE, mid[0],mid[1],mid[2], p1[0],p1[1],p1[2], radius*2.0, 0.0,
        color[0],color[1],color[2], color[0],color[1],color[2], 1.0, 0.0
    ]
    cmd.load_cgo(obj, name)

def debug_link(obj, chain, i, scale=2.5):
    """Inspect peptide link i -> i+1 using the C_i, O_i, N_{i+1} plane."""
    selC   = f"{obj} and chain {chain} and resi {i}   and name C"
    selO   = f"{obj} and chain {chain} and resi {i}   and name O"
    selNn  = f"{obj} and chain {chain} and resi {i+1} and name N"
    selCA  = f"{obj} and chain {chain} and resi {i}   and name CA"
    selCAn = f"{obj} and chain {chain} and resi {i+1} and name CA"

    try:
        C   = cmd.get_atom_coords(selC)
        O   = cmd.get_atom_coords(selO)
        Nn  = cmd.get_atom_coords(selNn)
        CA  = cmd.get_atom_coords(selCA)
        CAn = cmd.get_atom_coords(selCAn)
    except Exception as e:
        print(f"[warn] missing atom(s) for {chain}{i}->{i+1}: {e}")
        return

    # true link-plane normal
    CO, CNn = _vsub(O, C), _vsub(Nn, C)
    n_true  = _vunit(_vcross(CO, CNn))

    # draw cyan arrow
    base = C
    tip  = (C[0]+scale*n_true[0], C[1]+scale*n_true[1], C[2]+scale*n_true[2])
    name = f"true_normal_{obj}_{chain}_{i}"
    cmd.delete(name)
    _cgo_arrow(name, base, tip, color=(0.2,0.8,1.0))  # cyan

    # perpendicularity to C->N (0 is perfect)
    perp = abs(_vdot(n_true, _vunit(CNn)))

    # omega dihedral: CA_i - C_i - N_{i+1} - CA_{i+1}
    try:
        omega = cmd.get_dihedral(selCA, selC, selNn, selCAn)
    except:
        omega = float("nan")

    # show the defining triangle
    cmd.show("sticks", f"({selC}) or ({selO}) or ({selNn})")
    cmd.color("tv_orange", f"({selC}) or ({selO})")
    cmd.color("marine", selNn)
    cmd.set("stick_radius", 0.08, f"({selC}) or ({selO}) or ({selNn})")

    print(f"[{obj} {chain}{i}->{i+1}]  perp_to_CN={perp:.3f}   omega={omega:.1f} deg  (180=trans, 0=cis)")

cmd.extend("debug_link", debug_link)
