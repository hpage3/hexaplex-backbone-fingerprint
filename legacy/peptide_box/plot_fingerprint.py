import pandas as pd
import matplotlib.pyplot as plt
import argparse, os, glob
import numpy as np


def plot_fingerprint(csv_path, outdir, gap=5, rms_scale=None):
    pdb_id = os.path.basename(csv_path).replace("_fingerprint.csv", "").upper()
    df = pd.read_csv(csv_path)
    df = df.sort_values(["chain", "res_i"]).reset_index(drop=True)

    # Build serial x with chain breaks
    serial_x, offset, last_chain = [], 0, None
    for _, r in df.iterrows():
        if last_chain is not None and r["chain"] != last_chain:
            offset += gap
        serial_x.append(len(serial_x) + offset)
        last_chain = r["chain"]
    df["serial_x"] = serial_x

    # --- automatic RMS scaling ---
    auto_scale = False
    rms_column = "box_rms" if "box_rms" in df.columns else None
    if rms_scale is None and rms_column:
        auto_scale = True
        rms_vals = df[rms_column].fillna(0.0).to_numpy()
        max_rms = float(np.nanmax(rms_vals))
        theta_range = df["theta_pp_deg"].max() - df["theta_pp_deg"].min()

        if max_rms > 0:
            rms_scale = 0.15 * theta_range / max_rms
        else:
            rms_scale = 1000.0  # default if all RMS are zero

        # enforce practical limits
        rms_scale = max(500.0, min(rms_scale, 10000.0))

        print(f"[auto] {pdb_id}: θ-range={theta_range:.1f}, max RMS={max_rms:.5f}, scale={rms_scale:.1f}")

    plt.figure(figsize=(14, 5))
    if rms_scale and "box_rms" in df.columns:
        yerr = (df["box_rms"].fillna(0.0) * rms_scale).values
        plt.errorbar(df["serial_x"], df["theta_pp_deg"], yerr=yerr,
                     ecolor="red", fmt="-o", linewidth=1.0, markersize=3)
    else:
        plt.plot(df["serial_x"], df["theta_pp_deg"], "-o", linewidth=1.0, markersize=3)

    plt.xticks(df["serial_x"], df["aa_i"], rotation=90, fontsize=6)
    for i in range(len(df) - 1):
        if df["chain"].iloc[i] != df["chain"].iloc[i + 1]:
            plt.axvline(df["serial_x"].iloc[i] + gap / 2, color="red", linestyle="--", alpha=0.5)

    plt.title(f"θpp fingerprint for {pdb_id}")
    plt.xlabel("Residues (chains laid out serially)")
    plt.ylabel("θpp (deg)")
    if auto_scale:
        plt.text(0.01, 0.95, f"Auto RMS × {rms_scale:.1f}", transform=plt.gca().transAxes,
                 fontsize=8, color="red", va="top")
    plt.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"{pdb_id}_fingerprint.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[ok] saved {out} (RMS×{rms_scale:.2f})")


def main():
    p = argparse.ArgumentParser(description="Plot all θpp fingerprints in a directory")
    p.add_argument("--fingerprint-dir", default="fingerprints", help="Input directory with *_fingerprint.csv files")
    p.add_argument("--outdir", default="fingerprint_plots", help="Output directory for plots")
    p.add_argument("--gap", type=int, default=5, help="x-gap between chains")
    p.add_argument("--rms-as-yerr", type=str, default=None,
                   help="Scale factor k to draw yerr = k * box_rms (omit or 'auto' for per-file autoscale)")
    args = p.parse_args()

    # interpret argument
    if args.rms_as_yerr is None or args.rms_as_yerr.lower() == "auto":
        rms_scale = None  # triggers auto-scaling
    else:
        rms_scale = float(args.rms_as_yerr)

    csv_files = sorted(glob.glob(os.path.join(args.fingerprint_dir, "*_fingerprint.csv")))
    if not csv_files:
        print(f"No fingerprint files found in {args.fingerprint_dir}")
        return

    print(f"Found {len(csv_files)} fingerprint files.")
    for csv_path in csv_files:
        try:
            plot_fingerprint(csv_path, args.outdir, gap=args.gap, rms_scale=rms_scale)
        except Exception as e:
            print(f"[warn] failed on {csv_path}: {e}")


if __name__ == "__main__":
    main()
