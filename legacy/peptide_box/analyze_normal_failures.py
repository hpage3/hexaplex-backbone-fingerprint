#!/usr/bin/env python3
"""
Analyze flagged normal mismatches across many PDBs.

Usage:
    python analyze_normal_failures.py --ids ids.txt --qc-dir qc --pdb-dir input_data --out qc_analysis
"""

import argparse, os
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from Bio import PDB

def parse_args():
    ap = argparse.ArgumentParser(description="Analyze flagged normal mismatches")
    ap.add_argument("--ids", required=True, help="File containing PDB IDs (one per line)")
    ap.add_argument("--qc-dir", default="qc", help="Directory containing *_normals_validation.tsv files")
    ap.add_argument("--pdb-dir", default="input_data", help="Directory containing source PDBs")
    ap.add_argument("--out", default="qc_analysis", help="Output directory for reports/plots")
    ap.add_argument("--angle-col", default="angle_err_deg", help="Column holding angular error")
    ap.add_argument("--threshold", type=float, default=8.0, help="Flag threshold (deg)")
    return ap.parse_args()

def load_residue_map(pdb_path):
    """Map (chain, resseq) → residue name from a PDB file."""
    parser = PDB.PDBParser(QUIET=True)
    try:
        struct = parser.get_structure("x", pdb_path)
    except Exception as e:
        print(f"[warn] could not parse {pdb_path}: {e}")
        return {}
    res_map = {}
    for model in struct:
        for chain in model:
            for res in chain:
                if PDB.is_aa(res, standard=True):
                    res_map[(chain.id, res.id[1])] = res.resname
    return res_map

def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    # read IDs from file
    with open(args.ids) as fh:
        pdb_ids = [line.strip() for line in fh if line.strip()]

    all_rows = []
    aa_errors = defaultdict(list)

    for pdb_id in pdb_ids:
        qc_path = os.path.join(args.qc_dir, f"{pdb_id}_normals_validation.tsv")
        pdb_path = os.path.join(args.pdb_dir, f"{pdb_id}.pdb")

        if not os.path.exists(qc_path):
            print(f"[warn] missing QC file {qc_path}, skipping")
            continue

        res_map = {}
        if os.path.exists(pdb_path):
            res_map = load_residue_map(pdb_path)
        else:
            print(f"[warn] missing PDB {pdb_path}, residue names will be UNK")

        df = pd.read_csv(qc_path, sep="\t")
        if args.angle_col not in df.columns:
            print(f"[warn] {qc_path} missing {args.angle_col}")
            continue

        flagged = df[df[args.angle_col] > args.threshold].copy()
        for _, row in flagged.iterrows():
            chain = str(row["chain"])
            resi  = int(row["resseq"])
            resn  = res_map.get((chain, resi), "UNK")
            ang   = float(row[args.angle_col])
            row["resn"] = resn
            row["pdb_id"] = pdb_id
            all_rows.append(row)
            aa_errors[resn].append(ang)

    if not all_rows:
        print("No flagged rows found above threshold.")
        return

    flagged_df = pd.DataFrame(all_rows)
    flagged_df.to_csv(os.path.join(args.out, "all_flagged.tsv"), sep="\t", index=False)

    # Histogram of errors
    plt.figure(figsize=(6,4))
    flagged_df[args.angle_col].hist(bins=50, color="tomato", alpha=0.7)
    plt.xlabel("Angular error (deg)")
    plt.ylabel("Count (flagged residues)")
    plt.title("Distribution of flagged angular errors")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "error_histogram.png"), dpi=200)
    plt.close()

    # Error by residue type
    rows = []
    for aa, vals in sorted(aa_errors.items(), key=lambda kv: kv[0]):
        rows.append({
            "resn": aa,
            "count": len(vals),
            "mean_err": sum(vals)/len(vals),
            "median_err": pd.Series(vals).median(),
            "max_err": max(vals)
        })
    by_res = pd.DataFrame(rows)
    by_res.sort_values("mean_err", ascending=False, inplace=True)
    by_res.to_csv(os.path.join(args.out, "error_by_residue.tsv"), sep="\t", index=False)

    # Top 50 worst residues
    top50 = flagged_df.sort_values(args.angle_col, ascending=False).head(50)
    top50.to_csv(os.path.join(args.out, "top_50_worst_cases.tsv"), sep="\t", index=False)

    # Text summary
    with open(os.path.join(args.out, "error_summary.txt"), "w") as fh:
        fh.write(f"TOTAL flagged residues: {len(flagged_df)}\n")
        fh.write(f"Mean error: {flagged_df[args.angle_col].mean():.2f} deg\n")
        fh.write(f"Median error: {flagged_df[args.angle_col].median():.2f} deg\n")
        fh.write(f"Max error: {flagged_df[args.angle_col].max():.2f} deg\n\n")
        fh.write("Residue types with highest mean error:\n")
        fh.write(by_res.head(10).to_string(index=False))

    print(f"[ok] Analysis complete → results in {args.out}")

if __name__ == "__main__":
    main()
