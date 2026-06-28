#!/usr/bin/env python3
import argparse, csv, math, sys
from pathlib import Path

ANGLE_COL_NAMES = ["angle_err_deg", "angle", "angle_deg", "err_deg"]

def parse_args():
    ap = argparse.ArgumentParser(
        description="Summarize QC from *_normals_validation.tsv (+ *_flags.tsv if present)."
    )
    ap.add_argument("qc_dir", nargs="?", default="qc",
                    help="Directory containing per-protein TSVs (default: qc)")
    ap.add_argument("--out", default=None,
                    help="Output TSV (default: <qc_dir>/batch_qc_summary.tsv)")
    ap.add_argument("--thresh", type=float, default=None,
                    help="If given, recompute flagged count as angles >= THRESH (deg)")
    return ap.parse_args()

def find_angle_index(header):
    hdr = [h.strip().lower() for h in header]
    for name in ANGLE_COL_NAMES:
        if name in hdr:
            return hdr.index(name)
    return None

def read_normals_file(path: Path):
    """Return list of angles (floats)."""
    angles = []
    with path.open("r", encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        i_ang = find_angle_index(header)
        if i_ang is None:
            return angles
        for ln in fh:
            if not ln.strip():
                continue
            parts = ln.rstrip("\n").split("\t")
            if i_ang >= len(parts):
                continue
            try:
                angles.append(float(parts[i_ang]))
            except Exception:
                continue
    return angles

def count_flags_file(path: Path):
    """Return number of data rows in flags tsv (skips header)."""
    n = 0
    with path.open("r", encoding="utf-8") as fh:
        first = True
        for _ in fh:
            if first:
                first = False
                continue
            n += 1
    return n

def main():
    args = parse_args()
    qc = Path(args.qc_dir)
    out = Path(args.out) if args.out else qc / "batch_qc_summary.tsv"

    normals_files = sorted(qc.glob("*_normals_validation.tsv"))
    if not normals_files:
        print(f"[ERROR] No *_normals_validation.tsv in {qc}", file=sys.stderr)
        sys.exit(1)

    rows = []
    tot_valid = tot_flag = 0
    for f in normals_files:
        protein = f.stem.replace("_normals_validation", "")
        angles = read_normals_file(f)
        if not angles:
            # skip silently if file empty or header missing angle column
            continue
        valid = len(angles)
        mean_err = sum(angles) / valid
        worst_err = max(angles)

        flags_path = f.with_name(protein + "_flags.tsv")
        if args.thresh is not None:
            flagged = sum(1 for a in angles if a >= args.thresh)
            thresh_used = args.thresh
        elif flags_path.exists():
            flagged = count_flags_file(flags_path)
            thresh_used = None  # unknown/varies, came from per-protein run
        else:
            flagged = 0
            thresh_used = None

        rows.append((protein, valid, flagged, mean_err, worst_err, thresh_used))
        tot_valid += valid
        tot_flag  += flagged

    # sort: highest flag rate, then worst err
    rows.sort(key=lambda r: (-(r[2]/r[1] if r[1] else 0.0), -r[4], r[0]))

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["protein","validated","flagged","flag_rate","mean_abs_err_deg","worst_err_deg","threshold_deg"])
        for protein, valid, flagged, mean_err, worst_err, thresh_used in rows:
            rate = (100.0*flagged/valid) if valid else 0.0
            w.writerow([
                protein, valid, flagged, f"{rate:.2f}%",
                f"{mean_err:.3f}", f"{worst_err:.3f}",
                ("" if thresh_used is None else f"{thresh_used:.1f}")
            ])

    batch_rate = (100.0*tot_flag/tot_valid) if tot_valid else 0.0
    print(f"TOTAL validated: {tot_valid}")
    print(f"TOTAL flagged  : {tot_flag}")
    print(f"Batch flag rate: {batch_rate:.2f}%")
    print(f"[ok] wrote {out}")

if __name__ == "__main__":
    main()
