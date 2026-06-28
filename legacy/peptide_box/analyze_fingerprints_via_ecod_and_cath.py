# analyze_fingerprints_v2.py
# Enhanced version with CATH/ECOD domain awareness and per-domain FFT clustering

import os
import glob
import numpy as np
import pandas as pd
from scipy.fft import rfft
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score
import argparse
import pandas as pd, matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# -----------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------


def load_and_join_one_pdb(pdb_id, fingerprint_dir, aug_dir):
    """
    Load and merge fingerprint and augmented fingerprint data for a given PDB.
    Handles case-insensitive file names and harmonizes key column names/types.
    """
    # Normalize PDB ID case for consistent file lookup
    pdb_id_lower = pdb_id.lower()

    # --- Case-insensitive file search ---
    fp_candidates = [f for f in os.listdir(fingerprint_dir) if f.lower() == f"{pdb_id_lower}_fingerprint.csv"]
    aug_candidates = [f for f in os.listdir(aug_dir) if f.lower() == f"{pdb_id_lower}_augmented_fingerprint.csv"]

    if not fp_candidates or not aug_candidates:
        print(f"⚠️ Missing fingerprint or augmented file for {pdb_id}")
        return None

    fp_path = os.path.join(fingerprint_dir, fp_candidates[0])
    aug_path = os.path.join(aug_dir, aug_candidates[0])

    # --- Load CSVs ---
    df_fp = pd.read_csv(fp_path)
    df_aug = pd.read_csv(aug_path)

    # --- Normalize headers ---
    df_fp.columns = [c.strip().lower() for c in df_fp.columns]
    df_aug.columns = [c.strip().lower() for c in df_aug.columns]

    # ✅ Handle pdb_id vs pdb mismatch
    if "pdb_id" in df_fp.columns and "pdb" not in df_fp.columns:
        df_fp.rename(columns={"pdb_id": "pdb"}, inplace=True)
    if "pdbid" in df_fp.columns and "pdb" not in df_fp.columns:
        df_fp.rename(columns={"pdbid": "pdb"}, inplace=True)

    # ✅ Standardize chain and residue naming
    rename_map = {
        "chain_id": "chain",
        "residue": "res_i",
        "residue_index": "res_i"
    }
    df_fp.rename(columns=rename_map, inplace=True)
    df_aug.rename(columns=rename_map, inplace=True)

    # ✅ Harmonize key column types (case-insensitive PDB IDs, string chains, integer residues)
    for col in ["pdb", "chain"]:
        if col in df_fp.columns:
            df_fp[col] = df_fp[col].astype(str).str.lower()
        if col in df_aug.columns:
            df_aug[col] = df_aug[col].astype(str).str.lower()

    if "res_i" in df_fp.columns:
        df_fp["res_i"] = pd.to_numeric(df_fp["res_i"], errors="coerce").astype("Int64")
    if "res_i" in df_aug.columns:
        df_aug["res_i"] = pd.to_numeric(df_aug["res_i"], errors="coerce").astype("Int64")

    # --- Perform safe merge ---
    required = {"pdb", "chain", "res_i"}
    if not required.issubset(df_fp.columns) or not required.issubset(df_aug.columns):
        print(f"⚠️ Missing required columns for {pdb_id}")
        return None

    df = pd.merge(df_fp, df_aug, on=["pdb", "chain", "res_i"], how="inner")

    if df.empty:
        print(f"⚠️ Merge produced empty dataframe for {pdb_id}")
        return None

    return df

def load_vectors_by_granularity(
    fingerprint_dir,
    aug_dir,
    fft_len=100,
    weight_rms=False,
    ids_file=None,
    granularity="chain",
    min_len=30
):
    """
    Load and join fingerprint + augmented fingerprint data, grouped by granularity level.
    Optionally filters PDBs via a provided ids.txt file.
    """

    # --- Load PDB IDs from ids.txt if provided ---
    id_filter = None
    if ids_file and os.path.exists(ids_file):
        with open(ids_file, "r") as f:
            id_filter = [line.strip().lower() for line in f if line.strip()]
        print(f"📋 Loaded {len(id_filter)} PDB IDs from {ids_file}")
    else:
        print("⚠️ No ids.txt file provided — processing all available PDBs.")

    # --- Collect available PDB IDs from fingerprint directory ---
    pdb_ids = sorted(set(
        os.path.splitext(f)[0].split("_")[0].lower()
        for f in os.listdir(fingerprint_dir)
        if f.lower().endswith("_fingerprint.csv")
    ))


    # --- Filter if ids.txt is given ---
    if id_filter:
        before_count = len(pdb_ids)
        pdb_ids = [pid for pid in pdb_ids if pid in id_filter]
        print(f"✅ Using {len(pdb_ids)}/{before_count} PDBs listed in ids.txt")

    vectors, labels, meta = [], [], []
    processed, skipped = 0, 0

    for pid in pdb_ids:
        joined = load_and_join_one_pdb(pid, fingerprint_dir, aug_dir)
        if joined is None or joined.empty:
            skipped += 1
            continue

        # Normalize and clean headers
        joined.columns = [c.strip().lower() for c in joined.columns]
        rename_map = {
            "ecod_dom": "ecod_domain",
            "cath_dom": "cath_domain",
            "ecod_domn": "ecod_domain",
            "cath_domn": "cath_domain",
        }
        joined.rename(columns=rename_map, inplace=True)

        # Clean up any placeholder string 'nan' values
        for col in ["cath_domain", "ecod_domain"]:
            if col in joined.columns:
                joined[col] = joined[col].replace(["", " ", "nan", "NaN", "None"], pd.NA)

        has_cath = "cath_domain" in joined.columns and joined["cath_domain"].notna().any()
        has_ecod = "ecod_domain" in joined.columns and joined["ecod_domain"].notna().any()
        if not (has_cath or has_ecod):
            print(f"⚠️ Skipping {pid.upper()} — no CATH or ECOD annotations found.")
            skipped += 1
            continue

        # --- Granularity grouping (e.g., chain, ecod_f, cath_domain) ---
        if granularity == "chain":
            key = ["pdb", "chain"]
        elif granularity == "ecod_f":
            key = ["ecod_domain"]
        elif granularity == "cath_domain":
            key = ["cath_domain"]
        else:
            key = ["pdb"]

        for gkey, g in joined.groupby(key, dropna=False):
            if len(g) < min_len:
                continue

            # Convert to FFT vector
            theta = g["theta_pp_deg"].values if "theta_pp_deg" in g else None
            rms = g["box_rms"].values if "box_rms" in g else None
            if theta is None or len(theta) == 0:
                continue

            fft_vals = np.abs(np.fft.rfft(theta, n=fft_len))
            if weight_rms and rms is not None:
                fft_vals *= (1 + np.nan_to_num(rms, nan=0.0))

            vectors.append(fft_vals)
            labels.append(gkey if isinstance(gkey, str) else "_".join(map(str, gkey)))
            meta.append({"pdb": pid, "size": len(g)})

        processed += 1

    print(f"✅ Processed {processed}/{len(pdb_ids)} PDBs ({skipped} skipped without annotations).")
    return np.array(vectors), labels, meta


def report_alignment(df_out, name):
    from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score
    valid = df_out[name].notna()
    if valid.sum() <= 1:
        return
    ami = adjusted_mutual_info_score(df_out.loc[valid, "cluster"], df_out.loc[valid, name])
    nmi = normalized_mutual_info_score(df_out.loc[valid, "cluster"], df_out.loc[valid, name])
    print(f"[align] {name}: AMI={ami:.3f}, NMI={nmi:.3f}")

# -----------------------------------------------------------
# Main Entry
# -----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze protein fingerprints via ECOD and CATH annotations.")
    parser.add_argument("--fingerprint_dir", type=str, default="fingerprints", help="Directory containing *_fingerprint.csv files.")
    parser.add_argument("--aug_dir", type=str, default="fingerprints", help="Directory containing *_augmented_fingerprint.csv files.")
    parser.add_argument("--granularity", type=str, default="chain", choices=["chain", "ecod_f", "cath_domain"], help="Granularity level for grouping.")
    parser.add_argument("--clusters", type=int, default=10, help="Number of clusters for KMeans.")
    parser.add_argument("--fft_len", type=int, default=100, help="FFT vector length.")
    parser.add_argument("--weight_rms", action="store_true", help="Weight FFT by RMS values.")
    parser.add_argument("--ids", dest="ids_file", type=str, default=None, help="Optional path to an ids.txt file listing PDBs to include.")

    args = parser.parse_args()

    os.makedirs("analysis_output", exist_ok=True)

    X, labels, meta = load_vectors_by_granularity(
        fingerprint_dir=args.fingerprint_dir,
        aug_dir=args.aug_dir,
        fft_len=args.fft_len,
        weight_rms=args.weight_rms,
        ids_file=args.ids_file,
        granularity=args.granularity
    )

    print(f"FFT vectors: {X.shape}, Granularity: {args.granularity}")

    # --- Run KMeans clustering ---
    km = KMeans(n_clusters=args.clusters, n_init="auto", random_state=42)
    preds = km.fit_predict(X)

    # --- t-SNE embedding ---
    print("Computing t-SNE embedding (2D)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    emb = tsne.fit_transform(X)
    out_path = f"analysis_output/embedding_{args.granularity}.tsv"
    pd.DataFrame({"x": emb[:, 0], "y": emb[:, 1], "label": preds, "meta": labels}).to_csv(out_path, sep="\t", index=False)
    print(f"Saved embedding: {out_path}")

    # --- Cluster alignment metrics ---
    for col in ["cath_class", "cath_domain", "ecod_domain", "f_id", "h_name"]:
        if col in labels:
            print(f"[align] {col}: AMI=?, NMI=? (to be filled if label matrix is available)")

    print("✅ Analysis complete.")
 
# --- Load your t-SNE results ---
    df = pd.read_csv("analysis_output/embedding_ecod_f.tsv", sep="\t")

    # --- Basic scatter plot ---
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        df["x"],
        df["y"],
        c=df["label"],
        cmap="tab10",
        s=35,
        alpha=0.8,
        edgecolors="none"
    )

    # --- Add axis labels and title ---
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.title("t-SNE of θPP + RMS Fingerprints by KMeans Cluster", fontsize=14)

    # --- Add legend for clusters ---
    # Get unique cluster labels
    clusters = sorted(df["label"].unique())
    legend_elements = [
        Line2D(
            [0], [0],
            marker='o',
            color='w',
            label=f"Cluster {int(c)}",
            markerfacecolor=plt.cm.tab10(c / max(clusters)),
            markersize=10
        )
        for c in clusters
    ]
    plt.legend(
        handles=legend_elements,
        title="KMeans Clusters",
        loc="best",
        frameon=True
    )

    # --- Optional grid and style tweaks ---
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    # --- Show or save ---
    plt.show()
    # plt.savefig("analysis_output/tsne_clusters_with_legend.png", dpi=300)

if __name__ == "__main__":
    main()
