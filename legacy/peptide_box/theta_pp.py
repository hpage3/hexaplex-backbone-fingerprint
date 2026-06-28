#!/usr/bin/env python3
"""
theta_pp_pipeline.py – now defaults to signed θpp fingerprints.

Changes: 
  - Signed angles [-180, 180] are the default.
  - Unsigned [0, 180] histograms are optional via --unsigned.
  - Clustering etc. works on signed by default.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import requests
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import subprocess

RCSB_BASE = "https://files.rcsb.org/download"

# -----------------------------
# Utilities
# -----------------------------

def read_ids_file(path: Path) -> List[str]:
    ids = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        ids.append(s.upper())
    return ids

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def download_structure(pdb_id: str, out_file: Path, fmt: str = "pdb", retries: int = 3, timeout: int = 60) -> bool:
    """
    Try downloading PDB format first. If not available, fall back to CIF and convert to PDB.
    """
    pid = pdb_id.upper()

    # 1. Try PDB first
    pdb_url = f"{RCSB_BASE}/{pid}.pdb"
    try:
        r = requests.get(pdb_url, timeout=timeout)
        if r.status_code == 200 and r.content:
            out_file.write_bytes(r.content)
            return True
    except Exception as e:
        print(f"[warn] {pid}: PDB fetch failed: {e}", file=sys.stderr)

    # 2. Fall back to CIF
    cif_url = f"{RCSB_BASE}/{pid}.cif"
    try:
        r = requests.get(cif_url, timeout=timeout)
        if r.status_code == 200 and r.content:
            cif_file = out_file.with_suffix(".cif")
            cif_file.write_bytes(r.content)
            # Quick CIF → PDB conversion using gemmi
            try:
                import gemmi
                doc = gemmi.cif.read_file(str(cif_file))
                block = doc.sole_block()
                st = gemmi.make_structure_from_block(block)
                pdb_str = st.make_pdb_string()
                out_file.write_text(pdb_str)
                return True
            except Exception as conv_err:
                print(f"[warn] {pid}: CIF conversion failed ({conv_err})", file=sys.stderr)
                return False
        else:
            print(f"[warn] {pid}: HTTP {r.status_code} for CIF", file=sys.stderr)
    except Exception as e:
        print(f"[warn] {pid}: CIF fetch failed: {e}", file=sys.stderr)

    return False


def resolve_boxed_paths(boxed_path: Path) -> Tuple[Path, Path]:
    base = boxed_path.resolve().parent
    input_dir = (base / "input_data").resolve()
    output_dir = (base / "output").resolve()
    return input_dir, output_dir

def run_boxed_on_all(boxed_path: Path, python_bin: str, parallel: int, extra_args: List[str]):
    base = boxed_path.resolve().parent
    outdir = (base / "output").resolve()

    cmd = [python_bin, str(boxed_path)]
    # ensure batch-from-folder processing
    if not any(a == "--from-input-data" for a in (extra_args or [])):
        cmd += ["--from-input-data"]
    # ensure outputs land where the pipeline expects them
    if not any(a == "--outdir" for a in (extra_args or [])):
        cmd += ["--outdir", str(outdir)]

    if extra_args:
        cmd += [a for a in extra_args if a != "--"]  # drop separator if present

    print(f"[info] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

# -----------------------------
# θpp extraction & fingerprinting
# -----------------------------

def pick_theta_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ["theta_pp", "theta_pp_deg", "theta_pp (deg)",
                  "theta_peptide_plane", "theta_adjacent_deg",
                  "adjacent_angle_deg", "angle_signed_deg"]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if 'angle' in c.lower():
            return c
    return None

def series_to_signed(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s.clip(-180, 180)

def series_to_unsigned(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.min() < 0:
        s = s.abs()
    return s.clip(0, 180)

def build_histogram(s: pd.Series, bins: int, rng: Tuple[float, float]) -> np.ndarray:
    hist, _ = np.histogram(s.values, bins=bins, range=rng)
    hist = hist.astype(float)
    if hist.sum() > 0:
        hist /= hist.sum()
    return hist

def collect_fingerprints(output_dir: Path, bins: int = 36, unsigned: bool = False) -> pd.DataFrame:
    rows = []
    for csv in sorted(output_dir.glob("*.csv")):
        if "_boxes_normals" in csv.stem:
            continue
        try:
            df = pd.read_csv(csv)
        except Exception as e:
            print(f"[warn] Could not read {csv.name}: {e}", file=sys.stderr)
            continue
        col = pick_theta_column(df)
        if not col or col not in df.columns:
            continue
        s_signed = series_to_signed(df[col])
        if s_signed.empty:
            continue
        fp_signed = build_histogram(s_signed, bins=bins, rng=(-180,180))
        row = {"protein": csv.stem, "theta_col": col}
        for i,v in enumerate(fp_signed):
            row[f"b{i:02d}"] = v
        row["frac_pos"] = float((s_signed > 0).mean()) if len(s_signed) else 0.0
        row["frac_neg"] = 1.0 - row["frac_pos"]
        if unsigned:
            s_unsigned = series_to_unsigned(df[col])
            fp_unsigned = build_histogram(s_unsigned, bins=bins, rng=(0,180))
            for i,v in enumerate(fp_unsigned):
                row[f"unsigned_b{i:02d}"] = v
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("protein").sort_index()

# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="End-to-end θpp fingerprint pipeline (signed by default)")
    g_in = ap.add_mutually_exclusive_group(required=True)
    g_in.add_argument("--ids", type=str, help="Text file with one PDB ID per line")
    g_in.add_argument("--pdbs", nargs="+", help="Local PDB/mmCIF file paths")
    ap.add_argument("--fmt", choices=["pdb","cif"], default="pdb")
    ap.add_argument("--boxed-path", type=str, required=True)
    ap.add_argument("--python-bin", type=str, default=sys.executable)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--bins", type=int, default=36)
    ap.add_argument("--cluster", type=int, default=0)
    ap.add_argument("--unsigned", action="store_true", help="Also compute unsigned [0,180] histogram")
    ap.add_argument("--extra-boxed-args", nargs=argparse.REMAINDER)
    ap.add_argument("--keep-csv", action="store_true",
                help="Keep boxer CSV outputs in the output/ folder")
    ap.add_argument("--keep-inputs", action="store_true",
                help="Keep downloaded/copied inputs in input_data/")

    args = ap.parse_args()

    boxed_path = Path(args.boxed_path).resolve()
    input_dir, output_dir = resolve_boxed_paths(boxed_path)
    ensure_dir(input_dir); ensure_dir(output_dir)

    created_structures = []
    if args.ids:
        ids = read_ids_file(Path(args.ids))
        for pid in ids:
            out_file = input_dir / f"{pid}.{args.fmt}"
            if not out_file.exists():
                if download_structure(pid, out_file, fmt=args.fmt):
                    created_structures.append(out_file)
    else:
        for p in args.pdbs:
            src = Path(p).resolve()
            if src.exists():
                dest = input_dir / src.name
                if dest.resolve() != src:
                    dest.write_bytes(src.read_bytes())
                    created_structures.append(dest)

    try:
        run_boxed_on_all(boxed_path, args.python_bin, args.parallel, args.extra_boxed_args or [])
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)

    fp = collect_fingerprints(output_dir, bins=args.bins, unsigned=args.unsigned)
    if fp.empty:
        print("[error] No fingerprints produced", file=sys.stderr)
        sys.exit(3)

    if args.cluster and args.cluster > 0 and len(fp) >= args.cluster:
        labels = KMeans(n_clusters=args.cluster, n_init="auto", random_state=42).fit_predict(fp.filter(like="b").values)
        fp.insert(0, "cluster", labels)

    out_csv = Path("theta_pp_fingerprints.csv").resolve()
    fp.to_csv(out_csv)
    print(f"[ok] Wrote {out_csv} with {len(fp)} proteins")

    # Only remove staging files if not requested to keep
    if not args.keep_inputs:
        for f in created_structures:
            f.unlink(missing_ok=True)
    if not args.keep_csv:
        for csv in output_dir.glob("*.csv"):
            csv.unlink(missing_ok=True)

if __name__ == "__main__":
    main()
