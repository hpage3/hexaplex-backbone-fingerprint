import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load fingerprint embedding
embed = pd.read_csv(r".\fingerprint_analysis\fingerprint_embedding.tsv", sep="\t")

# Split pdb_id (e.g. "1AWP_A") into pdb and chain
embed[['pdb', 'chain']] = embed['pdb_id'].str.split("_", n=1, expand=True)
embed['pdb'] = embed['pdb'].str.lower()

# --- Load ECOD mapping (robust parser)
ecod_cols = [
    "uid","ecod_domain_id","manual_rep","f_id","pdb","chain","pdb_range","seqid_range",
    "unp_acc","arch_name","x_name","h_name","t_name","f_name","asm_status","ligand"
]

ecod = pd.read_csv(
    r"..\HHSearch\ecod.latest.domains_2023.tsv",
    sep="\t",
    comment="#",
    names=ecod_cols,
    engine="python",
    quoting=3,          # don't try to interpret quotes
    on_bad_lines="warn" # tolerate messy rows
)

# Clean up f_name (sometimes has stray tabs)
ecod["f_name"] = ecod["f_name"].astype(str).str.replace("\t", ",").str.strip()

# Normalize pdb field
ecod["pdb"] = ecod["pdb"].str.lower()

# Deduplicate to (pdb, chain) -> arch_name, x_name
ecod_map = ecod[["pdb", "chain", "arch_name", "x_name"]].drop_duplicates()

# --- Merge embeddings with ECOD
merged = embed.merge(ecod_map, on=["pdb", "chain"], how="left")
merged["arch_name"] = merged["arch_name"].fillna("UNK")
merged["x_name"]   = merged["x_name"].fillna("UNK")

# --- Plot by Architecture
plt.figure(figsize=(10, 8))
sns.scatterplot(data=merged, x="x", y="y", hue="arch_name", s=40, alpha=0.8)
plt.title("Protein Fingerprint Map colored by ECOD Architecture")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
plt.tight_layout()
plt.savefig("fingerprint_by_arch.png", dpi=300)
plt.close()

# --- Plot by X-group
plt.figure(figsize=(10, 8))
sns.scatterplot(data=merged, x="x", y="y", hue="x_name", s=40, alpha=0.8)
plt.title("Protein Fingerprint Map colored by ECOD X-group")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small", ncol=5 )
plt.tight_layout()
plt.savefig("fingerprint_by_xname.png", dpi=300)
plt.close()

# --- Save merged data
merged.to_csv("fingerprint_with_ecod.tsv", sep="\t", index=False)

print("[ok] Wrote:")
print(" - fingerprint_by_arch.png")
print(" - fingerprint_by_xname.png")
print(" - fingerprint_with_ecod.tsv (embedding + ECOD labels)")
