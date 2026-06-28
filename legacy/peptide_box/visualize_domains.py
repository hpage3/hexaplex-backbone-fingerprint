import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.cm import get_cmap
import os
import math

def main():
    ap = argparse.ArgumentParser(description="Visualize ECOD/CATH domain distribution")
    ap.add_argument("--pdb", required=True, help="PDB ID (e.g., 1awp)")
    ap.add_argument("--fingerprint-dir", default="fingerprints",
                    help="Directory containing *_fingerprint.csv and *_augmented_fingerprint.csv")
    ap.add_argument("--domain-type", choices=["ecod", "cath"], default="ecod",
                    help="Which domain system to visualize")
    ap.add_argument("--out-dir", default="visuals",
                    help="Directory to save plots and PyMOL scripts")
    ap.add_argument("--residues-per-line", type=int, default=200,
                    help="Residues per row before wrapping (default: 200)")
    args = ap.parse_args()

    pdb_id = args.pdb.lower()
    domain_col = f"{args.domain_type}_domain"

    # --- Locate input files
    fp_path = os.path.join(args.fingerprint_dir, f"{pdb_id}_fingerprint.csv")
    aug_path = os.path.join(args.fingerprint_dir, f"{pdb_id}_augmented_fingerprint.csv")
    print(f"🔍 Searching for: {fp_path}")
    print(f"🔍 Searching for: {aug_path}")

    if not os.path.exists(fp_path) or not os.path.exists(aug_path):
        print(f"❌ Missing files for {pdb_id}")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Load & normalize
    df_fp = pd.read_csv(fp_path)
    df_aug = pd.read_csv(aug_path)
    df_fp.columns = [c.lower().strip() for c in df_fp.columns]
    df_aug.columns = [c.lower().strip() for c in df_aug.columns]

    # Normalize join keys
    if "pdb_id" in df_fp.columns: df_fp.rename(columns={"pdb_id": "pdb"}, inplace=True)
    if "pdb_id" in df_aug.columns: df_aug.rename(columns={"pdb_id": "pdb"}, inplace=True)
    for d in (df_fp, df_aug):
        d["pdb"] = d["pdb"].astype(str).str.lower()
        d["chain"] = d["chain"].astype(str).str.upper()
        d["res_i"] = d["res_i"].astype(int)

    df = pd.merge(df_fp, df_aug, on=["pdb", "chain", "res_i"], how="inner")
    if df.empty:
        print(f"⚠️ No overlapping residues found for {pdb_id}")
        return

    # --- Summary
    domain_counts = df[domain_col].fillna("Unassigned").value_counts().to_dict()
    print(f"📊 Domains found ({len(domain_counts)} total):")
    for dom, count in domain_counts.items():
        print(f"  {dom}: {count} residues")

    # --- Plot sequence map with multiple lines
    domains = list(domain_counts.keys())
    cmap = get_cmap("tab20", len(domains))
    color_map = {d: cmap(i) for i, d in enumerate(domains)}

    total_res = df["res_i"].max()
    lines = math.ceil(total_res / args.residues_per_line)

    fig, axes = plt.subplots(lines, 1, figsize=(12, 1.2 * lines),
                             sharey=True, constrained_layout=True)
    if lines == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        start = i * args.residues_per_line + 1
        end = min((i + 1) * args.residues_per_line, total_res)
        sub = df[(df["res_i"] >= start) & (df["res_i"] <= end)]
        for dom, grp in sub.groupby(domain_col):
            ax.bar(grp["res_i"], [1]*len(grp),
                   color=color_map[dom], width=1.0)
        ax.set_xlim(start, end)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("Residue index")
        if i == 0:
            ax.set_title(f"{pdb_id.upper()} — {args.domain_type.upper()} domain distribution")

    # --- Legend below all plots
    handles = [mpatches.Patch(color=color_map[d], label=d) for d in domains]
    fig.legend(handles=handles, loc="lower center", ncol=6,
               fontsize="small", frameon=False)
    fig.subplots_adjust(bottom=0.15 + 0.05 * lines)

    plot_path = os.path.join(args.out_dir, f"{pdb_id}_{args.domain_type}_domains.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"✅ Saved domain plot: {plot_path}")

if __name__ == "__main__":
    main()
