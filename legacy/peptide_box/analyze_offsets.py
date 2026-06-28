import os, argparse, glob, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def parse_pdb_backbone(pdb_path):
    """
    Returns dict keyed by (chain, resseq) with atoms {name: (x,y,z, altloc, occ, b, resname)}.
    """
    bb = {}
    chains = {}
    with open(pdb_path, 'r', encoding='utf-8', errors='ignore') as fh:
        for ln in fh:
            if not ln.startswith(('ATOM','HETATM')):
                continue
            name = ln[12:16].strip()
            if name not in ('N','CA','C','O'):
                continue
            altloc = ln[16].strip()
            resname = ln[17:20].strip()
            chain = (ln[21] or ' ').strip() or ' '
            try:
                resseq = int(ln[22:26])
            except:
                continue
            x = float(ln[30:38]); y = float(ln[38:46]); z = float(ln[46:54])
            try:
                occ = float(ln[54:60])
            except:
                occ = 1.0
            try:
                b = float(ln[60:66])
            except:
                b = float('nan')
            key = (chain, resseq)
            bb.setdefault(key, {})[name] = (x,y,z, altloc, occ, b, resname)
            chains.setdefault(chain, set()).add(resseq)
    for ch in list(chains.keys()):
        chains[ch] = sorted(chains[ch])
    return bb, chains

def bestfit_plane_normal(points):
    P = np.asarray(points, dtype=float)
    c = P.mean(axis=0)
    U, S, Vt = np.linalg.svd(P - c, full_matrices=False)
    n = Vt[-1, :]
    n = n / (np.linalg.norm(n) + 1e-15)
    return n, c

def point_plane_offset(p, n, c):
    return abs(np.dot(np.asarray(p) - c, n))

def compute_offsets_for_link(bb, chain, i, j):
    ai = bb.get((chain, i), {})
    aj = bb.get((chain, j), {})
    if not ai or not aj:
        return None, "MISSING_ATOMS"
    if not all(k in ai for k in ('C','O','CA')) or 'N' not in aj or 'CA' not in aj:
        return None, "MISSING_ATOMS"

    Ci = ai['C']; Oi = ai['O']; CAi = ai['CA']; Nj = aj['N']; CAj = aj['CA']
    pts = [(Ci[0],Ci[1],Ci[2]), (CAi[0],CAi[1],CAi[2]), (Nj[0],Nj[1],Nj[2]), (CAj[0],CAj[1],CAj[2])]
    n, c = bestfit_plane_normal(pts)

    out = {}
    out['O_offset']  = point_plane_offset((Oi[0],Oi[1],Oi[2]), n, c)
    out['C_offset']  = point_plane_offset((Ci[0],Ci[1],Ci[2]), n, c)
    out['N_offset']  = point_plane_offset((Nj[0],Nj[1],Nj[2]), n, c)
    out['CAi_offset']= point_plane_offset((CAi[0],CAi[1],CAi[2]), n, c)
    out['CAj_offset']= point_plane_offset((CAj[0],CAj[1],CAj[2]), n, c)

    out['altloc_flag'] = int(any(x[3] not in ('', ' ') for x in (Ci, Oi, Nj, CAi, CAj)))
    out['occ_flag']    = int(any((x[4] if x[4]==x[4] else 1.0) < 1.0 for x in (Ci, Oi, Nj, CAi, CAj)))
    out['b_any'] = max([(x[5] if x[5]==x[5] else 0.0) for x in (Ci, Oi, Nj, CAi, CAj)])
    out['resname_i'] = Ci[6]; out['resname_j'] = Nj[6]
    return out, "OK"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="input_data")
    ap.add_argument("--boxes-dir",  default="output")
    ap.add_argument("--qc-dir",     default="qc")
    ap.add_argument("--out-dir",    default="qc_features")
    ap.add_argument("--angle-thresh", type=float, default=8.0)
    ap.add_argument("--o-offset-thresh", type=float, default=0.8)
    ap.add_argument("--bfactor-mult", type=float, default=1.5)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    all_rows = []
    summaries = []

    pdb_files = sorted(glob.glob(os.path.join(args.input_dir, "*.pdb")))
    for pdb_path in pdb_files:
        pdb_id = Path(pdb_path).stem.lower()
        boxes_csv = os.path.join(args.boxes_dir, f"{pdb_id}_boxes_normals.csv")
        qc_tsv    = os.path.join(args.qc_dir,    f"{pdb_id}_normals_validation.tsv")

        if not (os.path.exists(boxes_csv) and os.path.exists(qc_tsv)):
            continue

        boxes = pd.read_csv(boxes_csv)
        qc    = pd.read_csv(qc_tsv, sep="\t")

        if not {"chain","res_i","res_j","rms"}.issubset(set(boxes.columns)):
            rename_map = {}
            for c in boxes.columns:
                cl = c.lower()
                if cl in ("chain",): rename_map[c] = "chain"
                if cl in ("res_i","resseq"): rename_map[c] = "res_i"
                if cl in ("res_j",): rename_map[c] = "res_j"
                if cl in ("rms","fit_rms","box_rms"): rename_map[c] = "rms"
            boxes = boxes.rename(columns=rename_map)
        boxes = boxes[["chain","res_i","res_j","rms"]].copy()

        if not {"chain","resseq","angle_err_deg"}.issubset(set(qc.columns)):
            rename_map = {}
            for c in qc.columns:
                cl = c.lower()
                if cl in ("chain",): rename_map[c] = "chain"
                if cl in ("resseq","resi","res_i"): rename_map[c] = "resseq"
                if cl in ("angle_err_deg","angle_err","err_deg"): rename_map[c] = "angle_err_deg"
            qc = qc.rename(columns=rename_map)
        qc = qc[["chain","resseq","angle_err_deg"]].copy()

        theta_pp = None
        theta_path = os.path.join(args.qc_dir, f"{pdb_id}_theta_pp.tsv")
        if os.path.exists(theta_path):
            try:
                theta_df = pd.read_csv(theta_path, sep="\t")
                if "theta_pp_deg" in theta_df.columns:
                    theta_df = theta_df.rename(columns={"theta_pp_deg":"theta_deg"})
                if {"chain","resseq","theta_deg"}.issubset(theta_df.columns):
                    theta_pp = theta_df[["chain","resseq","theta_deg"]]
            except Exception:
                theta_pp = None

        bb, chains = parse_pdb_backbone(pdb_path)

        b_by_chain = {}
        for ch in chains:
            vals = []
            for resi in chains[ch]:
                at = bb.get((ch, resi), {})
                for nm in ('N','CA','C','O'):
                    if nm in at and (at[nm][5] == at[nm][5]):
                        vals.append(at[nm][5])
            med = float(np.median(vals)) if vals else float('nan')
            b_by_chain[ch] = med

        df = boxes.merge(qc, left_on=["chain","res_i"], right_on=["chain","resseq"], how="inner")
        if theta_pp is not None:
            df = df.merge(theta_pp, on=["chain","resseq"], how="left")

        rows = []
        for _, row in df.iterrows():
            chain = row["chain"]; i = int(row["resseq"]); j = int(row["res_j"])
            offsets, status = compute_offsets_for_link(bb, chain, i, j)
            if offsets is None:
                reason = "MISSING_ATOMS"
                Ooff=Coff=Noff=CAioff=CAjoff=float('nan')
                altloc=occ=0; b_any=float('nan')
                resname_i=resname_j=""
            else:
                Ooff  = offsets['O_offset']; Coff  = offsets['C_offset']; Noff  = offsets['N_offset']
                CAioff= offsets['CAi_offset']; CAjoff= offsets['CAj_offset']
                altloc= offsets['altloc_flag']; occ   = offsets['occ_flag']
                b_any = offsets['b_any']
                resname_i = offsets['resname_i']; resname_j = offsets['resname_j']
                reason = "OK"

            reslist = chains.get(chain, [])
            is_terminal = int((len(reslist)>0) and (i==reslist[0] or j==reslist[-1]))

            medB = b_by_chain.get(chain, float('nan'))
            high_b = int((b_any == b_any) and (medB == medB) and (b_any > args.bfactor_mult * medB))

            angle_err = float(row["angle_err_deg"])
            box_rms   = float(row["rms"])
            dev_reason = "OK"
            if math.isfinite(angle_err) and angle_err > args.angle_thresh:
                if math.isfinite(Ooff) and Ooff > args.o_offset_thresh:
                    dev_reason = "OXYGEN_SKEW"
                else:
                    dev_reason = "OTHER"
            elif reason == "MISSING_ATOMS":
                dev_reason = "MISSING_ATOMS"

            rows.append({
                "pdb_id": pdb_id,
                "chain": chain,
                "res_i": i,
                "res_j": j,
                "resname_i": resname_i,
                "resname_j": resname_j,
                "box_rms": box_rms,
                "angle_err_deg": angle_err,
                "theta_deg": float(row["theta_deg"]) if "theta_deg" in row and pd.notna(row["theta_deg"]) else float('nan'),
                "O_offset": Ooff,
                "C_offset": Coff,
                "N_offset": Noff,
                "CAi_offset": CAioff,
                "CAj_offset": CAjoff,
                "is_terminal": is_terminal,
                "altloc_flag": altloc,
                "occ_flag": occ,
                "high_bfactor_flag": high_b,
                "deviation_reason": dev_reason
            })
        all_rows.extend(rows)

        sdf = pd.DataFrame(rows)
        if len(sdf):
            corr_rms_err = sdf["box_rms"].corr(sdf["angle_err_deg"])
            corr_O_err   = sdf["O_offset"].corr(sdf["angle_err_deg"])
            summaries.append({
                "pdb_id": pdb_id,
                "N_links": int(len(sdf)),
                "mean_box_rms": float(sdf["box_rms"].mean()),
                "q95_box_rms": float(sdf["box_rms"].quantile(0.95)),
                "mean_angle_err": float(sdf["angle_err_deg"].mean()),
                "q95_angle_err": float(sdf["angle_err_deg"].quantile(0.95)),
                "corr(box_rms,angle_err)": float(corr_rms_err) if corr_rms_err==corr_rms_err else float('nan'),
                "corr(O_offset,angle_err)": float(corr_O_err)   if corr_O_err==corr_O_err else float('nan'),
                "frac_OK": float((sdf["deviation_reason"]=="OK").mean()),
                "frac_OXYGEN_SKEW": float((sdf["deviation_reason"]=="OXYGEN_SKEW").mean()),
                "frac_OTHER": float((sdf["deviation_reason"]=="OTHER").mean()),
                "frac_MISSING_ATOMS": float((sdf["deviation_reason"]=="MISSING_ATOMS").mean()),
            })

    out_dir = Path(args.out_dir)
    all_df = pd.DataFrame(all_rows)
    all_path = out_dir / "all_features.tsv"
    all_df.to_csv(all_path.as_posix(), sep="\t", index=False)

    summ_df = pd.DataFrame(summaries).sort_values("pdb_id")
    summ_path = out_dir / "all_features_summary.tsv"
    summ_df.to_csv(summ_path.as_posix(), sep="\t", index=False)

    plt.figure(figsize=(7,5))
    m = all_df[all_df["O_offset"].notna() & all_df["angle_err_deg"].notna()]
    plt.scatter(m["O_offset"], m["angle_err_deg"], s=6)
    plt.axvline(args.o_offset_thresh, linestyle="--", alpha=0.6)
    plt.axhline(args.angle_thresh, linestyle="--", alpha=0.6)
    plt.xlabel("O_offset (Å)")
    plt.ylabel("angle_err_deg (3-pt vs SVD)")
    plt.title("All proteins: O_offset vs angle_err")
    plt.tight_layout()
    plt.savefig((out_dir / "overall_ooffset_vs_err.png").as_posix(), dpi=150)

    plt.figure(figsize=(7,5))
    m2 = all_df[all_df["box_rms"].notna() & all_df["angle_err_deg"].notna()]
    plt.scatter(m2["box_rms"], m2["angle_err_deg"], s=6)
    plt.axhline(args.angle_thresh, linestyle="--", alpha=0.6)
    plt.xlabel("box_rms (Å)")
    plt.ylabel("angle_err_deg (3-pt vs SVD)")
    plt.title("All proteins: box_rms vs angle_err")
    plt.tight_layout()
    plt.savefig((out_dir / "overall_rms_vs_err.png").as_posix(), dpi=150)

    plt.figure(figsize=(6,4.6))
    counts = all_df["deviation_reason"].value_counts(dropna=False)
    total = counts.sum() if counts.size else 1.0
    xs = list(counts.index)
    ys = [counts[k]/total for k in xs]
    plt.bar(range(len(xs)), ys)
    plt.xticks(range(len(xs)), xs, rotation=15)
    plt.ylabel("Fraction of links")
    plt.title("All proteins: deviation reasons")
    plt.tight_layout()
    plt.savefig((out_dir / "overall_reason_bars.png").as_posix(), dpi=150)

if __name__ == "__main__":
    main()