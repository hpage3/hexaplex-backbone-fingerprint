#!/usr/bin/env python3
import argparse, os, sys, glob
import numpy as np, pandas as pd

# Optional plotting: script still works without matplotlib (falls back to CSVs)
try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False
    
def pick_angle_col(df: pd.DataFrame):
    for c in ["theta_pp_deg", "angle_signed_deg", "theta_pp", "adjacent_angle_deg"]:
        if c in df.columns:
            return c
    # last resort: any column containing 'angle'
    for c in df.columns:
        if "angle" in c.lower():
            return c
    return None

# ---------- Core FFT on a θpp sequence ----------
def theta_pp_fft(theta_deg, top_k=8, window='hann', detrend=True):
    x = np.asarray(theta_deg, dtype=float)

    # fill NaNs by linear interpolation
    n = np.isnan(x)
    if n.any():
        ii = np.arange(len(x))
        x[n] = np.interp(ii[n], ii[~n], x[~n])

    # unwrap angles (deg→rad unwrap→deg)
    xr = np.unwrap(np.deg2rad(x))
    x = np.rad2deg(xr)

    # detrend or mean-center
    if detrend:
        i = np.arange(len(x))
        A = np.vstack([i, np.ones_like(i)]).T
        m, b = np.linalg.lstsq(A, x, rcond=None)[0]
        x = x - (m*i + b)
    else:
        x = x - x.mean()

    N = len(x)
    if N < 4:
        return {"freqs_cyc_per_res": np.array([]), "power": np.array([]), "peaks": [], "N": N}

    # window
    if window == 'hann':
        w = np.hanning(N)
    elif window == 'hamming':
        w = np.hamming(N)
    elif window is None:
        w = np.ones(N)
    else:
        raise ValueError("window must be 'hann','hamming', or None")

    X = np.fft.rfft(x*w)
    P = (np.abs(X)**2) / (w**2).sum()
    freqs = np.fft.rfftfreq(N, d=1.0)  # cycles per residue

    mags = P.copy()
    if len(mags): mags[0] = 0.0
    order = np.argsort(mags)[::-1][:top_k]
    peaks = []
    for k in order:
        f = freqs[k]
        if f <= 0: 
            continue
        peaks.append({
            "k": int(k),
            "freq_cyc_per_res": float(f),
            "period_res_per_cycle": float(1.0/f),
            "power": float(P[k])
        })
    return {"freqs_cyc_per_res": freqs, "power": P, "peaks": peaks, "N": N}

# ---------- Sliding-window FFT (STFT) ----------
def stft_sliding(theta_deg, win_len=31, hop=5, window='hann', detrend=True):
    """
    Returns (freqs, times, power2D) where:
      freqs: 1D freq axis (cycles/res)
      times: center residue index of each window
      power2D: array [len(freqs) x len(times)]
    """
    x = np.asarray(theta_deg, float)

    # interpolate NaNs
    n = np.isnan(x)
    if n.any():
        ii = np.arange(len(x))
        x[n] = np.interp(ii[n], ii[~n], x[~n])

    # unwrap + detrend (as above)
    x = np.rad2deg(np.unwrap(np.deg2rad(x)))
    if detrend:
        i = np.arange(len(x))
        A = np.vstack([i, np.ones_like(i)]).T
        m, b = np.linalg.lstsq(A, x, rcond=None)[0]
        x = x - (m*i + b)
    else:
        x = x - x.mean()

    # window vector
    if window == 'hann':
        w = np.hanning(win_len)
    elif window == 'hamming':
        w = np.hamming(win_len)
    else:
        w = np.ones(win_len)

    freqs = np.fft.rfftfreq(win_len, d=1.0)
    times = []
    cols = []
    N = len(x)
    for start in range(0, max(1, N - win_len + 1), hop):
        seg = x[start:start+win_len]
        if len(seg) < win_len:
            break
        X = np.fft.rfft(seg*w)
        P = (np.abs(X)**2) / (w**2).sum()
        cols.append(P)
        times.append(start + win_len/2.0)

    if not cols:
        return np.array([]), np.array([]), np.zeros((0,0))
    power2D = np.stack(cols, axis=1)  # freq x time
    return freqs, np.asarray(times), power2D

# ---------- Utilities ----------
def ensure_outdir(d):
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)

def save_peaks_csv(rows, path): 
    pd.DataFrame(rows).to_csv(path, index=False)

def pick_band_peaks(freqs, power, fmin, fmax, top_k=8):
    m = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(m): return []
    idx = np.where(m)[0]
    order = np.argsort(power[m])[::-1][:top_k]
    rows = []
    for j in order:
        k = int(idx[j])
        f = float(freqs[k])
        rows.append({
            "k": k,
            "freq_cyc_per_res": f,
            "period_res_per_cycle": float(1.0/f),
            "power": float(power[k])
        })
    return rows

def plot_or_export_spectrum(freqs, power, title, out_base, band=None,
                            highpass=None, log_power=False, xlim=None):
    # Always export CSV so you can plot in Excel if no matplotlib
    spec_csv = f"{out_base}_fft_spectrum.csv"
    pd.DataFrame({"freq": freqs, "power": power}).to_csv(spec_csv, index=False)

    if not HAVE_MPL:
        return None, spec_csv

    png = f"{out_base}_fft_spectrum.png"
    plt.figure(figsize=(8,4.5))
    y = np.log10(power + 1e-12) if log_power else power
    plt.plot(freqs, y)
    plt.ylabel("log10 Power" if log_power else "Power")
    if xlim: plt.xlim(xlim)
    
    plt.xlabel("Frequency (cycles per residue)")

    plt.title(title)
    if band:
        bmin, bmax = band
        plt.axvspan(bmin, bmax, alpha=0.15)
    if highpass:
        plt.axvline(highpass, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(png, dpi=150)
    plt.close()
    return png, spec_csv

def plot_stft(freqs, times, power2D, out_base, fmax=0.4):
    # CSV fallback
    stft_csv = f"{out_base}_stft.csv"
    pd.DataFrame(power2D, index=freqs, columns=times).to_csv(stft_csv)
    if not HAVE_MPL:
        return None, stft_csv

    png = f"{out_base}_stft.png"
    plt.figure(figsize=(8,4.6))
    M = power2D[(freqs<=fmax), :]

    # take log10 to compress dynamic range
    M_log = np.log10(M + 1e-12)

    plt.imshow(M_log,
           aspect='auto', origin='lower',
           extent=[times[0], times[-1], freqs[freqs<=fmax][0], fmax],
           vmin=np.percentile(M_log, 5), vmax=np.percentile(M_log, 95))
    plt.colorbar(label="log10 Power")
    plt.colorbar(label="Power")
    plt.xlabel("Residue index")
    plt.ylabel("Frequency (cycles per residue)")
    plt.title("Sliding-window θpp spectrum")
    plt.tight_layout()
    plt.savefig(png, dpi=150)
    plt.close()
    return png, stft_csv

# ---------- Per-file pipeline ----------
def process_file(csv_path, args):
    df = pd.read_csv(csv_path)
    col = pick_angle_col(df)
    if not col:
        print(f"[WARN] {csv_path}: no θpp angle column found; skipping.", file=sys.stderr)
        return

    theta = pd.to_numeric(df[col], errors="coerce").dropna().clip(-180, 180).to_numpy(float)
    res = theta_pp_fft(theta, top_k=args.top_k, window=args.window, detrend=(not args.no_detrend))
    freqs, power, N = res["freqs_cyc_per_res"], res["power"], res["N"]
    
    # Optional high-pass for ranking/plotting
    if args.highpass is not None:
        hp = float(args.highpass)
        power_hp = power.copy()
        power_hp[freqs < hp] = 0.0
    else:
        hp = None
        power_hp = power

    base = os.path.splitext(os.path.basename(csv_path))[0]
    outdir = args.outdir or os.path.dirname(csv_path) or "."
    ensure_outdir(outdir)
    out_base = os.path.join(outdir, base)

    # Save overall peaks (on high-passed spectrum if requested)
    mags = power_hp.copy()
    if len(mags): mags[0] = 0.0
    order = np.argsort(mags)[::-1][:args.top_k]
    peaks = []
    for k in order:
        f = freqs[k]
        if f <= 0: continue
        peaks.append({"k": int(k),
                      "freq_cyc_per_res": float(f),
                      "period_res_per_cycle": float(1.0/f),
                      "power": float(power[k])})
    save_peaks_csv(peaks, f"{out_base}_fft_peaks.csv")

    # Optional band-specific tables
    for (bmin, bmax) in (args.band or []):
        rows = pick_band_peaks(freqs, power, bmin, bmax, top_k=args.top_k)
        tag = f"{bmin:.2f}-{bmax:.2f}".replace('.', 'p')
        save_peaks_csv(rows, f"{out_base}_fft_peaks_band_{tag}.csv")

    # Spectrum plot/CSV
    title = f"θpp Power Spectrum ({base})"
    png, spec_csv = plot_or_export_spectrum(
    freqs, power_hp, title, out_base,
    band=args.band[0] if args.band else None,
    highpass=hp,
    log_power=args.log_power,
    xlim=tuple(args.xlim) if args.xlim else None
)


    # Sliding-window spectrogram
    if args.stft:
        win, hop = args.stft
        f, t, M = stft_sliding(theta, win_len=win, hop=hop,
                               window=args.window, detrend=(not args.no_detrend))
        if M.size:
            stft_png, stft_csv = plot_stft(f, t, M, out_base, fmax=args.stft_fmax)
            if stft_png:
                print(f"Sliding-window plot: {stft_png}")
            print(f"Sliding-window CSV:  {stft_csv}")
        else:
            print("Sliding-window: no data generated (sequence shorter than window).")

    # Console summary
    print(f"\n{csv_path} (N={N})")
    if peaks:
        print("Top peaks (after high-pass if applied):")
        for p in peaks:
            print(f"  f={p['freq_cyc_per_res']:.3f} cyc/res, "
                  f"period={p['period_res_per_cycle']:.2f} res/cycle, "
                  f"power={p['power']:.0f}")
    if args.band:
        for (bmin, bmax) in args.band:
            rows = pick_band_peaks(freqs, power, bmin, bmax, top_k=min(args.top_k, 6))
            print(f"Band {bmin:.2f}-{bmax:.2f} cyc/res:")
            if rows:
                for r in rows:
                    print(f"  f={r['freq_cyc_per_res']:.3f} (≈{1/r['freq_cyc_per_res']:.2f} res/turn), power={r['power']:.0f}")
            else:
                print("  (no peaks)")
    if png:
        print(f"Spectrum PNG: {png}")
    print(f"Spectrum CSV: {spec_csv}")

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="FFT of θpp sequences with bands, high-pass, and sliding-window (STFT).")
    ap.add_argument("inputs", nargs="*", help="CSV files with theta_pp_deg column.")
    ap.add_argument("--glob", help="Glob pattern (e.g., data/*_ppn.csv)")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--window", choices=["hann","hamming","None"], default="hann")
    ap.add_argument("--no-detrend", action="store_true")
    ap.add_argument("--highpass", type=float, help="Zero power below this frequency (cycles/res).")
    ap.add_argument("--band", nargs=2, type=float, action="append",
                    metavar=("FMIN","FMAX"),
                    help="Add a band to report (repeatable). Example: --band 0.20 0.35")
    ap.add_argument("--stft", nargs=2, type=int, metavar=("WIN","HOP"),
                    help="Sliding-window FFT: window length and hop (in residues).")
    ap.add_argument("--stft-fmax", type=float, default=0.4,
                    help="Max frequency to display in STFT plot.")
    ap.add_argument("--log-power", action="store_true", help="Plot log10 power in spectra/plots.")
    ap.add_argument("--xlim", nargs=2, type=float, metavar=("XMIN","XMAX"),help="Limit x-axis for spectrum plot.")
    ap.add_argument("--ids", type=str,
                help="Text file with one PDB ID per line (expects *_boxes_adjacent_angles.csv for each)")
    ap.add_argument("--angles-dir", type=str, default="output",
                help="Directory where *_boxes_adjacent_angles.csv files are stored")


    args = ap.parse_args()

    files = []
    if args.ids:
        with open(args.ids) as fh:
            for line in fh:
                pid = line.strip().lower()
                if not pid or pid.startswith("#"):
                    continue
                csv_path = os.path.join(args.angles_dir, f"{pid}_boxes_adjacent_angles.csv")
                if os.path.exists(csv_path):
                    files.append(csv_path)
                else:
                    print(f"[WARN] Missing {csv_path}", file=sys.stderr)

    if args.glob:
        files.extend(sorted(glob.glob(args.glob)))

    files.extend(args.inputs)


    # normalize window option
    if args.window == "None":
        args.window = None

    # sanity check
    if not files:
        print("No inputs. Provide files via --ids/--angles-dir, --glob, or positional CSVs.", file=sys.stderr)
        sys.exit(2)

    # process each CSV
    for f in files:
        try:
            process_file(f, args)
        except Exception as e:
            print(f"[ERROR] {f}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
