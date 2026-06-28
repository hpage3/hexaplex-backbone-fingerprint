#!/usr/bin/env python3
"""
Python wrapper for planes_from_backbone_ortho_boxes.py
Replacement for boxed.sh
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = Path(__file__).resolve().parent
PYFILE = SCRIPT_DIR / "planes_from_backbone_ortho_boxes.py"
INPUT_DIR = SCRIPT_DIR / "input_data"
OUTDIR = SCRIPT_DIR / "output"

# Defaults (same as boxed.sh)
DEFAULTS = [
    "--csv", "--plot", "--color-ss",
    "--force-chain", "A",
    "--as-sticks", "--stick-radius", "0.15"
]

def run_one(pdb_name: str, python_bin: str = "python3", keep_copy=True, verbose=False):
    """Process a single PDB file"""
    in_file = INPUT_DIR / pdb_name
    if not in_file.exists():
        print(f"[warn] Missing: {in_file} — skipping", file=sys.stderr)
        return

    stem = in_file.stem
    out_pdb = OUTDIR / f"{stem}_boxes.pdb"
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print(f"[info] Processing: {in_file}")

    # Copy input file for reference
    if keep_copy:
        shutil.copy(in_file, OUTDIR / f"{stem}_input.pdb")

    cmd = [python_bin, str(PYFILE), "--outdir", str(OUTDIR), "--output", str(out_pdb), str(in_file)] + DEFAULTS

    if verbose:
        print("[debug] Running:", " ".join(cmd))

    subprocess.run(cmd, check=True, text=True)



def collect_pdbs(all_mode: bool, names):
    """Return list of pdb names to process"""
    if all_mode:
        files = list(INPUT_DIR.glob("*.pdb")) + list(INPUT_DIR.glob("*.PDB"))
        to_run = [f.name for f in files if "_boxes" not in f.stem.lower()]
        if not to_run:
            print(f"[info] No plain *.pdb files found in {INPUT_DIR}/")
        return to_run
    else:
        out = []
        for n in names:
            if not (n.lower().endswith(".pdb")):
                n = f"{n}.pdb"
            out.append(n)
        return out


def main():
    parser = argparse.ArgumentParser(description="Wrapper for peptide plane boxing")
    parser.add_argument("names", nargs="*", help="Input PDB file names (without path)")
    parser.add_argument("--all", action="store_true", help="Process all *.pdb in input_data/")
    parser.add_argument("--python-bin", default="python3", help="Python executable to use")
    parser.add_argument("--no-copy", action="store_true", help="Do not copy original PDB into output dir")
    parser.add_argument("--parallel", type=int, default=1, help="Run in parallel with N workers")
    parser.add_argument("--verbose", action="store_true", help="Show full commands")
    args = parser.parse_args()

    pdbs = collect_pdbs(args.all, args.names)
    if not pdbs:
        sys.exit(0)

    if args.parallel > 1:
        with ThreadPoolExecutor(max_workers=args.parallel) as ex:
            futures = {ex.submit(run_one, n, args.python_bin, not args.no_copy, args.verbose): n for n in pdbs}
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as e:
                    print(f"[error] {futures[fut]} failed: {e}", file=sys.stderr)
    else:
        for n in pdbs:
            try:
                run_one(n, args.python_bin, not args.no_copy, args.verbose)
            except Exception as e:
                print(f"[error] {n} failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
