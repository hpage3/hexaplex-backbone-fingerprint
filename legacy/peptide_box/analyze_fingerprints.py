# ... keep your imports ...
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances, silhouette_samples
import umap

import glob
from scipy.fft import rfft

import glob
from scipy.fft import rfft

def load_all_fingerprints(fingerprint_dir, fft_len=100, weight_rms=False, ids_file=None):
    """
    Load *_fingerprint.csv files for selected PDB IDs and convert each chain's θpp sequence
    into a fixed-length FFT vector for comparison.
    """
    # --- optional ID filter ---
    # --- optional ID filter ---
    id_filter = None
    if ids_file and os.path.exists(ids_file):
        with open(ids_file) as f:
            id_filter = [line.strip().upper() for line in f if line.strip()]
        print(f"[info] Restricting to {len(id_filter)} IDs from {ids_file}")
    else:
        print("[info] No ID filter applied (loading all fingerprints)")

    # Gather all candidate fingerprint CSVs
    csv_files = sorted(glob.glob(os.path.join(fingerprint_dir, "*_fingerprint.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No *_fingerprint.csv files found in {fingerprint_dir}")

    # --- apply ID filtering ---
    if id_filter:
        selected_files = []
        for csv in csv_files:
            base = os.path.basename(csv).upper()        # e.g. 1AWP_A_FINGERPRINT.CSV
            pdb_id = base.split("_")[0]                 # e.g. 1AWP
            # keep if any ID in file matches either the exact label or PDB prefix
            if any(base.startswith(i) or pdb_id == i for i in id_filter):
                selected_files.append(csv)
        print(f"[info] Selected {len(selected_files)} of {len(csv_files)} files by ID filter")
        csv_files = selected_files

    X, labels = [], []


    for csv in csv_files:
        base = os.path.basename(csv).upper()   # e.g. 1C1U_FINGERPRINT.CSV
        pdb_prefix = base.split("_")[0]        # e.g. 1C1U

        df = pd.read_csv(csv)
        if df.empty:
            print(f"[skip] {base} is empty")
            continue
        if "theta_pp_deg" not in df.columns or "chain" not in df.columns:
            print(f"[warn] Skipping {base}: missing required columns (theta_pp_deg/chain)")
            continue

        # Determine pdb_id from column or filename
        pdb_id_guess = pdb_prefix
        pdb_id = str(df.get("pdb_id", [pdb_id_guess])[0]).upper()

        # --- build one vector PER CHAIN ---
        for ch, g in df.groupby("chain"):
            ch = str(ch)
            chain_label = f"{pdb_id}_{ch}".upper()  # e.g. 1C1U_H

            # Apply ID filter at chain or PDB level
            if id_filter:
                # keep if ids.txt has the full chain label OR the bare PDB id
                if chain_label not in id_filter and pdb_id not in id_filter:
                    continue

            # y series for this chain
            y = g["theta_pp_deg"].astype(float).to_numpy()

            if weight_rms and "box_rms" in g.columns:
                rms = g["box_rms"].fillna(0.0).to_numpy()
                weights = 1.0 / (1.0 + rms)
                y = y * weights

            # pad/trim then FFT
            if len(y) < fft_len:
                y = np.pad(y, (0, fft_len - len(y)))
            else:
                y = y[:fft_len]

            fft_vec = np.abs(rfft(y, n=fft_len))[:fft_len]
            X.append(fft_vec)
            labels.append(chain_label)


        y = df["theta_pp_deg"].astype(float).values
        if weight_rms and "box_rms" in df.columns:
            rms = df["box_rms"].fillna(0.0).values
            weights = 1.0 / (1.0 + rms)
            y = y * weights

        # Pad or truncate
        n = len(y)
        if n < fft_len:
            y = np.pad(y, (0, fft_len - n))
        else:
            y = y[:fft_len]

        fft_vec = np.abs(rfft(y, n=fft_len))
        X.append(fft_vec[:fft_len])

        pdb_id_guess = os.path.basename(csv).split("_")[0].upper()
        pdb_id = str(df.get("pdb_id", [pdb_id_guess])[0]).upper()
        chain = str(df.get("chain", ["?"])[0])
        labels.append(f"{pdb_id}_{chain}")

    if not X:
        raise RuntimeError("No valid fingerprints loaded (check ids.txt or filenames).")

    X = np.vstack(X)
    return X, labels

def compute_uniqueness(X, labels, k=10):
    """
    Compute per-fingerprint uniqueness metrics in the original vector space.
    labels like '1ABC_A' -> protein_id='1ABC', chain='A'
    """
    prot_ids = [lab.split("_")[0] for lab in labels]
    D = pairwise_distances(X, metric="cosine")  # shape (n,n); 0..2 range for cosine distance
    np.fill_diagonal(D, np.inf)  # ignore self when taking mins

    nn_other = []
    nn_same = []
    margin = []
    k_density = []

    for i in range(len(labels)):
        same_mask  = np.array(prot_ids) == prot_ids[i]
        other_mask = ~same_mask

        nn_same_i = np.min(D[i, same_mask]) if np.any(same_mask) else np.inf
        nn_other_i = np.min(D[i, other_mask]) if np.any(other_mask) else np.inf

        nn_same.append(nn_same_i)
        nn_other.append(nn_other_i)

        # margin (positive means closer to own protein than outsiders)
        if not np.isfinite(nn_same_i):  # no same-protein neighbor
            margin.append(nn_other_i)
        elif not np.isfinite(nn_other_i):
            margin.append(np.nan)
        else:
            margin.append(nn_other_i - nn_same_i)

        # local density: mean distance to k nearest neighbors overall
        # (use finite values only)
        row = D[i, :]
        finite_row = row[np.isfinite(row)]
        k_eff = min(k, len(finite_row))
        if k_eff > 0:
            k_density.append(np.partition(finite_row, k_eff-1)[:k_eff].mean())
        else:
            k_density.append(np.nan)

    # silhouette vs protein IDs (treat each protein as its own class)
    # needs at least 2 distinct labels and at least 2 samples total
    try:
        sil = silhouette_samples(X, prot_ids, metric="cosine")
    except Exception:
        sil = np.full(len(labels), np.nan)

    return pd.DataFrame({
        "label": labels,             # e.g., 1ABC_A
        "protein_id": prot_ids,      # e.g., 1ABC
        "nn_same": nn_same,          # smaller is tighter within protein
        "nn_other": nn_other,        # larger is more unique vs others
        "margin": margin,            # >0 desirable
        "kNN10_density": k_density,  # larger => more isolated
        "silhouette_cosine": sil     # -1..1
    })

def main():
    ap = argparse.ArgumentParser(description="Fingerprint → FFT → Map of Protein Space + Uniqueness metrics")
    ap.add_argument("--fingerprint-dir", default="fingerprints", help="Directory with *_fingerprint.csv files")
    ap.add_argument("--fft-len", type=int, default=100, help="Number of FFT coefficients to keep")
    ap.add_argument("--outdir", default="fingerprint_analysis", help="Output directory")
    ap.add_argument("--use-umap", action="store_true", help="Use UMAP instead of t-SNE")
    ap.add_argument("--weight-rms", action="store_true", help="Down-weight high RMS regions")
    ap.add_argument("--clusters", type=int, default=10, help="Number of clusters for k-means")
    ap.add_argument("--ids", default="ids.txt", help="Optional file listing PDB IDs to include (one per line)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) Load fingerprints → vectors (per-chain)
    X, labels = load_all_fingerprints(args.fingerprint_dir, args.fft_len, args.weight_rms, args.ids)
    print(f"[ok] Loaded {len(labels)} fingerprints with {X.shape[1]}-dim vectors")

    # 2) Scale for embedding/clustering
    Xs = StandardScaler().fit_transform(X)

    # 2b) Uniqueness metrics in ORIGINAL (unscaled) space
    uniq_df = compute_uniqueness(X, labels, k=10)
    uniq_path = os.path.join(args.outdir, "fingerprint_uniqueness.tsv")
    uniq_df.to_csv(uniq_path, sep="\t", index=False)
    print(f"[ok] wrote {uniq_path}")

    # 3) Embed with t-SNE or UMAP (for visualization only)
    if args.use_umap:
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
        coords = reducer.fit_transform(Xs)
    else:
        tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=42)
        coords = tsne.fit_transform(Xs)

    # 4) KMeans clustering (on scaled vectors)
    kmeans = KMeans(n_clusters=args.clusters, n_init="auto", random_state=42).fit(Xs)
    cluster_labels = kmeans.labels_

    # 5) Save embeddings
    df_out = pd.DataFrame({
        "label": labels,
        "protein_id": [lab.split("_")[0] for lab in labels],
        "x": coords[:,0],
        "y": coords[:,1],
        "cluster": cluster_labels
    })
    out_csv = os.path.join(args.outdir, "fingerprint_embedding.tsv")
    df_out.to_csv(out_csv, sep="\t", index=False)
    print(f"[ok] wrote {out_csv}")

    # 6) Plot (color by cluster)
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(coords[:,0], coords[:,1], c=cluster_labels, cmap="tab10", alpha=0.7)
    for i, lab in enumerate(labels):
        plt.text(coords[i,0], coords[i,1], lab, fontsize=6, alpha=0.6)
    plt.title("Protein Fingerprint Map (FFT compressed)")
    plt.tight_layout()
    out_png = os.path.join(args.outdir, "fingerprint_embedding.png")
    plt.savefig(out_png, dpi=300)
    print(f"[ok] saved {out_png}")

if __name__ == "__main__":
    main()
