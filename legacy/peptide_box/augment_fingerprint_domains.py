#!/usr/bin/env python3
"""
augment_fingerprint_domains_per_pdb.py

Reads PDB IDs from ids.txt and corresponding PDB files in ./input_data/,
adds CATH (CDF 2.0) and ECOD (2023 TSV) domain annotations to each residue,
and writes each augmented fingerprint to fingerprints/{PDB}_augmented_fingerprint.csv

Requires:
    pip install pandas biopython
"""

import os
import re
import pandas as pd
from Bio import PDB

# ---------------- CONFIG ----------------
IDS_FILE = "ids.txt"
PDB_DIR = "./input_data"
CATH_FILE = "cath-domain-boundaries.txt"      # CDF 2.0 format
ECOD_FILE = "../HHSearch/ecod.latest.domains_2023.tsv"
OUTPUT_DIR = "./fingerprints"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ----------------------------------------

# ---------- ECOD PARSER ----------
import mmap

def parse_ecod(file_path, pdb_filter=None):
    """
    Efficient ECOD parser using memory mapping and optional PDB filtering.
    pdb_filter: set of lowercase PDB IDs to include (reduces parsing time drastically)
    """
    ecod_records = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        for line in iter(mm.readline, b""):
            if not line or line.startswith(b"#"):
                continue
            parts = line.decode("utf-8", errors="ignore").strip().split("\t")
            if len(parts) < 7:
                continue

            pdb = parts[4].lower()
            if pdb_filter and pdb not in pdb_filter:
                continue  # skip irrelevant entries quickly

            chain_field = parts[5].strip()
            rng = parts[6]  # e.g. "A:1-86"
            m = re.match(r"([A-Za-z0-9]):(\d+)-(\d+)", rng)
            if not m:
                continue
            chain, start, end = m.groups()

            ecod_records.append({
                "pdb": pdb,
                "chain": chain,
                "start": int(start),
                "end": int(end),
                "ecod_domain": parts[1],
                "f_id": parts[3],
                "h_name": parts[12] if len(parts) > 12 else None,
                "f_name": parts[13] if len(parts) > 13 else None
            })
        mm.close()

    df = pd.DataFrame.from_records(ecod_records)
    if df.empty:
        print("⚠️ No ECOD entries matched your PDB list.")
    else:
        print(f"✅ Parsed {len(df)} ECOD domain records for {len(set(df['pdb']))} PDBs.")
    return df

# ---------- CATH PARSER ----------
def parse_cath(file_path):
    cath_data = []
    pattern = re.compile(
        r"^(\w{4})([A-Za-z0-9])\s+D(\d+)\s+F(\d+)\s+(\d+)\s+([A-Za-z0-9])\s+(\d+)\s*-\s+[A-Za-z0-9]\s+(\d+)"
    )
    with open(file_path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            m = pattern.match(line)
            if not m:
                continue
            pdb, chain, dnum, fnum, cath_class, chn, start, end = m.groups()
            cath_data.append({
                "pdb": pdb.lower(),
                "chain": chn,
                "start": int(start),
                "end": int(end),
                "cath_domain": f"{pdb}{chain}D{dnum}",
                "cath_class": int(cath_class)
            })
    return pd.DataFrame(cath_data)

# ---------- Assign Domain ----------
def assign_domain(row, domain_df, cols):
    subset = domain_df[
        (domain_df["pdb"] == row["pdb"]) &
        (domain_df["chain"] == row["chain"]) &
        (domain_df["start"] <= row["res_i"]) &
        (domain_df["end"] >= row["res_i"])
    ]
    if subset.empty:
        return [None] * len(cols)
    hit = subset.iloc[0]
    return [hit[col] for col in cols]

# ---------- Extract residues ----------
def extract_residues(pdb_path):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("X", pdb_path)
    records = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if not PDB.is_aa(residue):
                    continue
                res_id = residue.get_id()[1]
                records.append((os.path.basename(pdb_path)[:4].lower(), chain.id, res_id))
    return pd.DataFrame(records, columns=["pdb", "chain", "res_i"])

# ---------- MAIN ----------
def main():
    with open(IDS_FILE) as f:
        pdb_ids = [line.strip().lower() for line in f if line.strip()]

    print(f"Found {len(pdb_ids)} PDB IDs")

    print("Loading domain files...")
    cath_df = parse_cath(CATH_FILE)
    # pass the PDB list to the ECOD parser for filtering
    ecod_df = parse_ecod(ECOD_FILE, pdb_filter=set(pdb_ids))

    for pdb_id in pdb_ids:
        pdb_path = os.path.join(PDB_DIR, f"{pdb_id}.pdb")
        if not os.path.exists(pdb_path):
            print(f"⚠️ Missing file: {pdb_path}")
            continue

        print(f"Processing {pdb_id}...")
        res_df = extract_residues(pdb_path)

        cath_cols = ["cath_domain", "cath_class"]
        ecod_cols = ["ecod_domain", "f_id", "h_name", "f_name"]

        res_df[cath_cols] = res_df.apply(
            lambda r: assign_domain(r, cath_df, cath_cols), axis=1, result_type="expand"
        )
        res_df[ecod_cols] = res_df.apply(
            lambda r: assign_domain(r, ecod_df, ecod_cols), axis=1, result_type="expand"
        )

        out_file = os.path.join(OUTPUT_DIR, f"{pdb_id}_augmented_fingerprint.csv")
        res_df.to_csv(out_file, index=False)
        print(f"✅ Wrote {out_file}")

if __name__ == "__main__":
    main()
