#!/usr/bin/env python3
"""
Make θpp(i) vs i fingerprint graphs for each protein and compute a scalar summary per protein.

What it does
------------
1) Reads a list of PDB IDs from --ids (one per line) OR accepts local PDB paths via --pdbs.
2) Downloads any missing structures (RCSB) into boxed.py's input_data/ and runs boxed.py (all-Python).
3) For each "*_boxes_adjacent_angles.csv" emitted by boxed.py, extracts the signed angle column
   (e.g., angle_signed_deg), plots θpp(i) vs i, and saves one PNG per protein.
4) Collapses each protein's signed angle series into a single scalar fingerprint using the
   mean resultant length (R in [0,1]) from circular statistics. (Higher R → angles cluster tightly
   near a mean direction; lower R → spread out.)
5) Writes a summary CSV with one row per protein containing R and other helpful stats.

Usage
-----
python theta_pp_fingerprint_graphs.py --ids ids.txt --boxed-path planes_from_backbone_ortho_boxes.py --outdir fingerprint_plots --extra-boxed-args "--from-input-data --csv --plot --outdir output_boxes"

Or with local PDB files:
python theta_pp_fingerprint_graphs.py --pdbs 1tim.pdb 2ypi.pdb --boxed-path ./boxed.py

Outputs
-------
- <outdir>/<PDB>_theta_pp_vs_index.png : θpp(i) vs i plot (signed degrees)
- theta_pp_scalar_summary.csv          : one-row-per-protein summary including scalar R
- Optionally deletes structures/boxed CSVs to keep footprint small (see flags)
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import subprocess
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests

RCSB_BASE = "https://files.rcsb.org/download"  # {pdb}.pdb or {pdb}.cif

# -----------------------------
# Utilities
# -----------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def read_ids_file(path: Path) -> List[str]:
    ids = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        ids.append(s.lower())
    return ids


def resolve_boxed_paths(boxed_path: Path) -> Tuple[Path, Path]:
    base = boxed_path.resolve().parent
    return (base / "input_data").resolve(), (base / "output").resolve()


def download_structure(pdb_id: str, out_file: Path, fmt: str = "pdb", retries: int = 3, timeout: int = 60) -> bool:
    url = f"{RCSB_BASE}/{pdb_id}.{fmt}"
    for attempt in range(1, retries+1):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200 and r.content:
                out_file.write_bytes(r.content)
                return True
            else:
                print(f"[warn] {pdb_id}: HTTP {r.status_code} on attempt {attempt}", file=sys.stderr)
        except Exception as e:
            print(f"[warn] {pdb_id}: {e} on attempt {attempt}", file=sys.stderr)
        time.sleep(min(2*attempt, 10))
    return False


def run_boxed_on_all(boxed_path: Path, python_bin: str, parallel: int, extra_args: List[str]):
    cmd = [python_bin, str(boxed_path)]
    if extra_args:
        cmd += [a for a in extra_args if a != "--"]
    print(f"[info] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


# -----------------------------
# θpp extraction and scalarization
# -----------------------------

def pick_angle_col(df: pd.DataFrame) -> Optional[str]:
    preferred = [
        "angle_signed_deg", "theta_pp", "theta_pp_deg", "theta_peptide_plane",
        "theta_adjacent_deg", "adjacent_angle_deg", "angle_deg", "theta", "angle"
    ]
    for c in preferred:
        if c in df.columns:
            return c
    for c in df.columns:
        if 'angle' in c.lower():
            return c
    return None


def load_signed_angles(csv_path: Path) -> Optional[np.ndarray]:
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[warn] Could not read {csv_path.name}: {e}", file=sys.stderr)
        return None
    col = pick_angle_col(df)
    if not col:
        return None
    s = pd.to_numeric(df[col], errors='coerce').dropna()
    if s.empty:
        return None
    # Signed range in degrees (clip to avoid crazy outliers)
    s = s.clip(-180, 180)
    return s.to_numpy()


def mean_resultant_length_deg(theta_deg: np.ndarray) -> float:
    """Return R in [0,1] from signed degrees using first trigonometric moment."""
    th = np.deg2rad(theta_deg)
    C = np.mean(np.cos(th))
    S = np.mean(np.sin(th))
    R = float(np.hypot(C, S))
    return R


# -----------------------------
# Plotting
# -----------------------------

def plot_theta_vs_index(theta_deg: np.ndarray, title: str, outfile: Path):
    plt.figure(figsize=(9, 3.2))
    plt.plot(np.arange(1, len(theta_deg)+1), theta_deg)
    plt.axhline(0.0, linestyle='--', linewidth=1)
    plt.xlabel("Residue index i")
    plt.ylabel("θpp(i) (degrees, signed)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Make θpp(i) vs i plots per protein and compute per-protein scalar R")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ids", type=str, help="Text file with one PDB ID per line")
    g.add_argument("--pdbs", nargs="+", help="Local PDB/mmCIF file paths")

    ap.add_argument("--fmt", choices=["pdb", "cif"], default="pdb", help="Download format")
    ap.add_argument("--boxed-path", type=str, required=True, help="Path to boxed.py")
    ap.add_argument("--python-bin", type=str, default=sys.executable, help="Python binary to run boxed.py")
    ap.add_argument("--parallel", type=int, default=4, help="Parallel workers for boxed.py --all")
    ap.add_argument("--outdir", type=str, default="fingerprint_plots", help="Where to save the θpp(i) plots")
    ap.add_argument("--summary", type=str, default="theta_pp_scalar_summary.csv", help="Output CSV with per-protein scalars")
    ap.add_argument("--no-clean-structures", action="store_true", help="Keep downloaded/copied structures (default: delete)")
    ap.add_argument("--no-clean-outputs", action="store_true", help="Keep boxed.py CSV outputs (default: delete)")
    ap.add_argument("--extra-boxed-args", nargs=argparse.REMAINDER, help="Args to pass to boxed.py after '--'")
    args = ap.parse_args()

    boxed_path = Path(args.boxed_path).resolve()
    if not boxed_path.exists():
        print(f"[error] boxed.py not found at {boxed_path}", file=sys.stderr)
        sys.exit(2)

    input_dir, output_dir = resolve_boxed_paths(boxed_path)
    ensure_dir(input_dir)
    ensure_dir(output_dir)

    outdir = Path(args.outdir).resolve()
    ensure_dir(outdir)

    # 1) Prepare inputs (download or copy)
    created_structures: List[Path] = []
    if args.ids:
        ids = read_ids_file(Path(args.ids))
        print(f"[info] Read {len(ids)} IDs from {args.ids}")
        for pid in ids:
            out_file = input_dir / f"{pid}.{args.fmt}"
            if not out_file.exists():
                ok = download_structure(pid, out_file, fmt=args.fmt)
                if not ok:
                    print(f"[warn] Failed to download {pid}.{args.fmt}", file=sys.stderr)
                    continue
            created_structures.append(out_file)
    else:
        for p in args.pdbs:
            src = Path(p).resolve()
            if not src.exists():
                print(f"[warn] Missing local file: {src}", file=sys.stderr)
                continue
            dest = input_dir / src.name
            if dest.resolve() != src:
                dest.write_bytes(src.read_bytes())
                created_structures.append(dest)

    # 2) Run boxed.py on all inputs
    try:
        run_boxed_on_all(boxed_path, args.python_bin, args.parallel, args.extra_boxed_args or [])
    except subprocess.CalledProcessError as e:
        print(f"[error] boxed.py returned {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)

    # 3) For each adjacent_angles CSV, plot θpp(i) vs i and compute scalar R
    rows = []
    csvs = sorted(output_dir.glob("*_boxes_adjacent_angles.csv"))
    if not csvs:
        print(f"[error] No *_boxes_adjacent_angles.csv found in {output_dir}", file=sys.stderr)
        sys.exit(3)

    for csv in csvs:
        pdb_stem = csv.stem.replace("_boxes_adjacent_angles", "")
        theta = load_signed_angles(csv)
        if theta is None or len(theta) == 0:
            print(f"[warn] No usable angles in {csv.name}", file=sys.stderr)
            continue

        # Plot
        png_path = outdir / f"{pdb_stem}_theta_pp_vs_index.png"
        plot_theta_vs_index(theta, f"{pdb_stem}: θpp(i) vs i (signed degrees)", png_path)

        # Scalarization (R)
        R = mean_resultant_length_deg(theta)

        # Additional helpful stats
        mu = float(np.degrees(np.angle(np.mean(np.exp(1j*np.deg2rad(theta))))) )
        frac_pos = float((theta > 0).mean())
        frac_neg = 1.0 - frac_pos
        # Entropy of signed histogram (36 bins)
        hist, _ = np.histogram(theta, bins=36, range=(-180,180))
        p = hist.astype(float) / hist.sum() if hist.sum() else np.zeros_like(hist, dtype=float)
        entropy = float(-(p[p>0] * np.log2(p[p>0])).sum())

        rows.append({
            "protein": pdb_stem,
            "R": R,                     # recommended single-number fingerprint (0..1)
            "mu_deg": mu,               # circular mean direction (degrees)
            "frac_pos": frac_pos,
            "frac_neg": frac_neg,
            "entropy": entropy,
            "n_angles": int(len(theta))
        })

    # 4) Write summary
    if not rows:
        print("[error] No results to write", file=sys.stderr)
        sys.exit(4)

    summary_path = Path(args.summary).resolve()
    pd.DataFrame(rows).set_index("protein").sort_index().to_csv(summary_path)
    print(f"[ok] Wrote per-protein scalar summary to {summary_path}")

    # 5) Cleanup to scale
    if not args.no_clean_structures:
        removed = 0
        for f in created_structures:
            try:
                f.unlink(missing_ok=True)
                removed += 1
            except Exception:
                pass
        print(f"[info] Removed {removed} structure file(s) from {input_dir}")

    if not args.no_clean_outputs:
        removed_out = 0
        for f in output_dir.glob("*.csv"):
            try:
                f.unlink(missing_ok=True)
                removed_out += 1
            except Exception:
                pass
        print(f"[info] Removed {removed_out} boxed output CSV(s) from {output_dir}")

if __name__ == "__main__":
    main()
