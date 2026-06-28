#!/usr/bin/env python3
import argparse, csv, sys
from pathlib import Path

def parse_args():
    ap = argparse.ArgumentParser(
        description="Collect all per-protein *_flags.tsv into one TSV (robust parsing)."
    )
    ap.add_argument("qc_dir", nargs="?", default="qc",
                    help="Folder containing per-protein *_flags.tsv files (default: qc)")
    ap.add_argument("--out", default=None,
                    help="Output TSV path (default: <qc_dir>/all_flags.tsv)")
    ap.add_argument("--sort-by", choices=["protein", "angle", "none"],
                    default="protein",
                    help="Sort rows by protein (default), angle (desc), or keep original order")
    ap.add_argument("--pattern", default="*_flags.tsv",
                    help="Glob for input files (default: *_flags.tsv)")
    ap.add_argument("--exclude", default="all_flags.tsv,all_flags_sorted_by_angle.tsv",
                    help="Comma-separated basenames to skip (default excludes common aggregates)")
    return ap.parse_args()

def find_col(header, names):
    hdr = [h.strip().lower() for h in header]
    for n in names:
        if n in hdr:
            return hdr.index(n)
    return None

def main():
    args = parse_args()
    qc_dir = Path(args.qc_dir)
    out_path = Path(args.out) if args.out else qc_dir / "all_flags.tsv"
    excluded = {n.strip().lower() for n in args.exclude.split(",") if n.strip()}
    excluded.add(out_path.name.lower())

    rows = []  # (protein, chain, resseq, angle_err_deg)
    seen = set()

    inputs = sorted(qc_dir.glob(args.pattern))
    if not inputs:
        print(f"[warn] No files matched {qc_dir}/{args.pattern}", file=sys.stderr)

    for f in inputs:
        if f.name.lower() in excluded:
            continue
        try:
            with f.open("r", encoding="utf-8") as fh:
                header_line = fh.readline().rstrip("\n")
                if not header_line:
                    print(f"[warn] Empty file: {f}", file=sys.stderr); continue
                header = header_line.split("\t")
                i_chain = find_col(header, ["chain"])
                i_resi  = find_col(header, ["resseq", "resid", "res_seq"])
                i_ang   = find_col(header, ["angle_err_deg", "angle", "angle_deg", "err_deg"])
                # Require these columns
                if None in (i_chain, i_resi, i_ang):
                    print(f"[skip] {f.name}: missing required columns in header {header}", file=sys.stderr)
                    continue

                prot = f.stem[:-6] if f.stem.endswith("_flags") else f.stem  # strip _flags
                for ln in fh:
                    ln = ln.strip()
                    if not ln or ln.lower().startswith("chain\t"):
                        continue
                    parts = ln.split("\t")
                    if max(i_chain, i_resi, i_ang) >= len(parts):
                        continue
                    chain = parts[i_chain].strip()
                    resi  = parts[i_resi].strip()
                    ang   = parts[i_ang].strip()
                    # type checks
                    try:
                        resi_i = int(resi)
                        ang_f  = float(ang)
                    except Exception:
                        # Skip non-numeric rows (prevents 'all   all   3SN6   R' etc.)
                        continue
                    key = (prot, chain, resi_i, round(ang_f, 3))
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append((prot, chain, str(resi_i), f"{ang_f:.3f}"))
        except FileNotFoundError:
            continue

    # sort
    if args.sort_by == "angle":
        rows.sort(key=lambda r: (-float(r[3]), r[0], r[1], int(r[2])))
    elif args.sort_by == "protein":
        rows.sort(key=lambda r: (r[0], r[1], int(r[2])))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["protein", "chain", "resseq", "angle_err_deg"])
        w.writerows(rows)

    print(f"[ok] wrote {out_path} ({len(rows)} rows)")

if __name__ == "__main__":
    main()
