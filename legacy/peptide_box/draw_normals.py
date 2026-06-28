from pymol import cmd, cgo
import csv

def get_coord(selection):
    if not cmd.count_atoms(selection):
        return None
    return tuple(cmd.get_atom_coords(selection))

def load_normals_split_abs(csv_file, scale=1.5, radius=0.15,
                           low_cutoff=0.03, high_cutoff=0.06):
    """
    Draw normals as arrows split into three objects using absolute RMS cutoffs.

    - normals_low  (blue)  : rms < low_cutoff
    - normals_mid  (purple): low_cutoff <= rms < high_cutoff
    - normals_high (red)   : rms >= high_cutoff

    Args:
        csv_file    : *_boxes_normals.csv file
        scale       : arrow length
        radius      : arrow thickness
        low_cutoff  : RMS cutoff for blue→purple
        high_cutoff : RMS cutoff for purple→red
    """
    scale   = float(scale)
    radius  = float(radius)
    low_cutoff  = float(low_cutoff)
    high_cutoff = float(high_cutoff)

    objs = {"low": [], "mid": [], "high": []}

    with open(csv_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            chain = row["chain"].strip()
            res_i = row["res_i"].strip()
            res_j = row["res_j"].strip()

            Ci = get_coord(f"chain {chain} and resi {res_i} and name C")
            Nj = get_coord(f"chain {chain} and resi {res_j} and name N")
            if Ci is None or Nj is None:
                continue

            Ci_x, Ci_y, Ci_z = Ci
            Nj_x, Nj_y, Nj_z = Nj
            cx = (Ci_x + Nj_x) / 2.0
            cy = (Ci_y + Nj_y) / 2.0
            cz = (Ci_z + Nj_z) / 2.0

            nx, ny, nz = float(row["nx"]), float(row["ny"]), float(row["nz"])
            rms = float(row["rms"])

            # assign bucket
            if rms < low_cutoff:
                color = (0.0, 0.0, 1.0)  # blue
                bucket = "low"
            elif rms < high_cutoff:
                color = (0.5, 0.0, 0.5)  # purple
                bucket = "mid"
            else:
                color = (1.0, 0.0, 0.0)  # red
                bucket = "high"

            ex = cx + nx * scale
            ey = cy + ny * scale
            ez = cz + nz * scale

            objs[bucket].extend([
                cgo.CYLINDER, cx, cy, cz, ex, ey, ez,
                radius, *color, *color,
                cgo.CONE, ex, ey, ez,
                ex+nx*0.5, ey+ny*0.5, ez+nz*0.5,
                radius*1.6, 0.0, *color, *color, 1.0, 0.0
            ])

    # delete old and load new
    for name in ["low", "mid", "high"]:
        obj_name = f"normals_{name}"
        cmd.delete(obj_name)
        if objs[name]:
            cmd.load_cgo(objs[name], obj_name)

cmd.extend("load_normals_split_abs", load_normals_split_abs)
