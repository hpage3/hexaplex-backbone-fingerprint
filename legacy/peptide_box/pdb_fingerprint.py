import os
import argparse
import pandas as pd
from Bio.PDB import PDBParser

AA3_TO_1 = {
    "ALA":"A","CYS":"C","ASP":"D","GLU":"E","PHE":"F",
    "GLY":"G","HIS":"H","ILE":"I","LYS":"K","LEU":"L",
    "MET":"M","ASN":"N","PRO":"P","GLN":"Q","ARG":"R",
    "SER":"S","THR":"T","VAL":"V","TRP":"W","TYR":"Y"
}

def get_residue_map(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", pdb_file)
    mapping = {}
    for model in structure:
        for chain in model:
            for res in chain:
                if res.id[0] == " ":  # standard residue
                    mapping[(chain.id, res.id[1])] = AA3_TO_1.get(res.resname.strip().upper(), "X")
    return mapping

def process_pdb(pdb_id, input_dir, angles_dir, boxes_dir, out_dir):
    pdb_path   = os.path.join(input_dir,  f"{pdb_id}.pdb")
    angles_csv = os.path.join(angles_dir, f"{pdb_id}_boxes_adjacent_angles.csv")
    boxes_csv  = os.path.join(boxes_dir,  f"{pdb_id}_boxes_normals.csv")

    if not (os.path.exists(pdb_path) and os.path.exists(angles_csv) and os.path.exists(boxes_csv)):
        print(f"[skip] Missing files for {pdb_id}")
        return

    # θpp table
    ang = pd.read_csv(angles_csv)
    # Use A-side indices as the link identity (i -> j)
    ang["res_i"] = ang["res_i_A"]
    ang["res_j"] = ang["res_j_A"]
    ang = ang.rename(columns={"angle_signed_deg":"theta_pp_deg"})

    # Box RMS table (standard headers are chain,res_i,res_j,rms)
    box = pd.read_csv(boxes_csv)
    # Normalize columns if needed
    rmap = {}
    for c in box.columns:
        cl = c.lower()
        if cl in ("chain",): rmap[c] = "chain"
        elif cl in ("res_i","resseq","i"): rmap[c] = "res_i"
        elif cl in ("res_j","j"): rmap[c] = "res_j"
        elif cl in ("rms","fit_rms","box_rms"): rmap[c] = "box_rms"
    box = box.rename(columns=rmap)
    box = box[["chain","res_i","res_j","box_rms"]].drop_duplicates()

    # Merge RMS onto θpp using the same (chain,res_i,res_j) link identity
    ang = ang.merge(box, on=["chain","res_i","res_j"], how="left")

    # Map one-letter residue for i
    aa_map = get_residue_map(pdb_path)
    ang["aa_i"] = [aa_map.get((c, i), "X") for c, i in zip(ang["chain"], ang["res_i"])]

    # Build final, deduped per-residue fingerprint (anchor on res_i)
    out = ang[["chain","res_i","aa_i","theta_pp_deg","box_rms"]].copy()
    out.insert(0, "pdb_id", pdb_id)
    out = out.drop_duplicates(subset=["pdb_id","chain","res_i"]).sort_values(["chain","res_i"])

    # Write per-PDB fingerprint file
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{pdb_id}_fingerprint.csv")
    out.to_csv(out_path, index=False)
    print(f"[ok] wrote {out_path}")

def main():
    ap = argparse.ArgumentParser(description="Build per-PDB θpp fingerprints with RMS")
    ap.add_argument("--ids", default="ids.txt", help="Text file of PDB IDs (one per line)")
    ap.add_argument("--input-dir", default="input_data")
    ap.add_argument("--angles-dir", default="output_boxes")
    ap.add_argument("--boxes-dir",  default="output_boxes")
    ap.add_argument("--outdir",     default="fingerprints")
    args = ap.parse_args()

    with open(args.ids) as f:
        ids = [ln.strip() for ln in f if ln.strip()]

    for pid in ids:
        process_pdb(pid, args.input_dir, args.angles_dir, args.boxes_dir, args.outdir)

if __name__ == "__main__":
    main()
