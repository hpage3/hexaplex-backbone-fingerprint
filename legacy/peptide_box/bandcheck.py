import argparse
import numpy as np
import pandas as pd

# ---- Robust imports: fall back if theta_pp_fft lacks stft_band_power ----
try:
    from theta_pp_fft import stft_sliding, stft_band_power
except Exception:
    from theta_pp_fft import stft_sliding  # type: ignore
    def stft_band_power(freqs, spec, fmin, fmax):
        """Compute band power across time for STFT matrix.
        freqs: (K,) cycles/residue; spec: (T, K) complex STFT values.
        Returns (T,) band power normalized by number of bins.
        """
        freqs = np.asarray(freqs)
        spec = np.asarray(spec)
        if spec.ndim != 2:
            raise ValueError("spec must be 2D (time x freq)")
        mask = (freqs >= fmin) & (freqs <= fmax)
        if not np.any(mask):
            return np.zeros(spec.shape[0])
        band = spec[:, mask]
        power = np.abs(band) ** 2
        return power.sum(axis=1) / np.count_nonzero(mask)


def metrics(pred, true):
    tp = np.sum((pred == 1) & (true == 1))
    fp = np.sum((pred == 1) & (true == 0))
    fn = np.sum((pred == 0) & (true == 1))
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    return prec, rec


def resample_track_to_residues(centers, track, n_res):
    """Nearest-neighbor resampling of a per-window track to per-residue values."""
    centers = np.asarray(centers)
    track = np.asarray(track)
    out = np.zeros(n_res)
    for i in range(n_res):
        j = int(np.argmin(np.abs(centers - i)))
        out[i] = track[j]
    return out


# --- θpp classification helpers ---
def zero_crossing_rate(sig, tol=1e-3):
    s = np.sign(np.where(np.abs(sig) < tol, 0.0, sig))
    s_compact = s[np.insert(np.diff(s).astype(bool), 0, True)]
    if len(s_compact) < 2:
        return 0.0
    return np.mean(np.diff(s_compact) != 0)


def lag1_autocorr(sig):
    if len(sig) < 2:
        return 0.0
    x = sig - np.mean(sig)
    var = float(np.dot(x, x))
    if var < 1e-12:
        return 0.0
    return float(np.dot(x[:-1], x[1:]) / var)


def main():
    ap = argparse.ArgumentParser(description="Band-power + θpp-based SS from theta_pp CSV")
    ap.add_argument("csv", help="CSV from theta_pp.py with theta_pp_deg (and optionally phi_deg, psi_deg)")
    ap.add_argument("--alpha", nargs=2, type=float, metavar=("FMIN","FMAX"), default=[0.20, 0.35],
                    help="Helix band in cycles/residue (default 0.20 0.35)")
    ap.add_argument("--beta", nargs=2, type=float, metavar=("FMIN","FMAX"), default=[0.45, 0.55],
                    help="Sheet band in cycles/residue (default 0.45 0.55)")
    ap.add_argument("--stft", nargs=2, type=int, metavar=("WIN","HOP"), default=[31, 5],
                    help="STFT window and hop in residues (default 31 5)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if "theta_pp_deg" not in df.columns:
        raise SystemExit("Input CSV missing 'theta_pp_deg' column")

    theta = np.deg2rad(df["theta_pp_deg"].to_numpy())
    win, hop = args.stft

    # STFT
    centers, freqs, spec = stft_sliding(theta, win, hop)

    # Band powers
    a_min, a_max = args.alpha
    b_min, b_max = args.beta
    a_pow = stft_band_power(freqs, spec, a_min, a_max)
    b_pow = stft_band_power(freqs, spec, b_min, b_max)

    base = args.csv.rsplit(".", 1)[0]
    pd.DataFrame({"center": centers, "helix_band": a_pow, "sheet_band": b_pow}).to_csv(
        f"{base}_stft_band.csv", index=False)
    print(f"Wrote {base}_stft_band.csv")

    # Optional φ/ψ SS baseline
    have_phipsi = ("phi_deg" in df.columns and "psi_deg" in df.columns)
    ss = None
    if have_phipsi:
        phi = df["phi_deg"].to_numpy()
        psi = df["psi_deg"].to_numpy()
        ss = np.zeros(len(phi), dtype=int)
        # Simple heuristic regions (you can swap for DSSP later)
        ss[(phi < -40) & (phi > -80) & (psi < -20) & (psi > -60)] = 1  # helix-ish
        ss[(phi < -80) & (phi > -160) & (psi > 100)] = 2               # sheet-ish
        pd.DataFrame({"residue": np.arange(len(ss)), "ss_phipsi": ss}).to_csv(
            f"{base}_ss_from_phipsi.csv", index=False)
        print(f"Wrote {base}_ss_from_phipsi.csv")

    # --- θpp-based SS classification ---
    ahat = resample_track_to_residues(centers, a_pow, len(theta))
    bhat = resample_track_to_residues(centers, b_pow, len(theta))

    valid_a = ahat[~np.isnan(ahat)]
    valid_b = bhat[~np.isnan(bhat)]
    tauH = np.percentile(valid_a, 90) if valid_a.size and np.any(valid_a > 0) else np.inf
    tauE = np.percentile(valid_b, 90) if valid_b.size and np.any(valid_b > 0) else np.inf

    pred_ss = np.zeros(len(theta), dtype=int)
    for i in range(len(theta)):
        w = theta[max(0, i-15):min(len(theta), i+16)]
        zcr = zero_crossing_rate(w)
        r1  = lag1_autocorr(w)
        if ahat[i] >= tauH and zcr <= 0.15 and r1 >= 0.6:
            pred_ss[i] = 1
        elif bhat[i] >= tauE and zcr >= 0.35 and r1 <= 0.1:
            pred_ss[i] = 2
        else:
            pred_ss[i] = 0

    pd.DataFrame({
        "residue": np.arange(len(pred_ss)),
        "ss_theta_0loop_1helix_2sheet": pred_ss
    }).to_csv(f"{base}_ss_from_theta.csv", index=False)
    print(f"Wrote {base}_ss_from_theta.csv")

    if have_phipsi:
        prec_h, rec_h = metrics(pred_ss==1, ss==1)
        prec_e, rec_e = metrics(pred_ss==2, ss==2)
        with open(f"{base}_theta_vs_phipsi.txt", "w", encoding="utf-8") as f:
            f.write(f"Theta-based helix vs φ/ψ: Prec {prec_h:.3f}, Rec {rec_h:.3f}\n")
            f.write(f"Theta-based sheet vs φ/ψ: Prec {prec_e:.3f}, Rec {rec_e:.3f}\n")
        print(f"Wrote {base}_theta_vs_phipsi.txt")


if __name__ == "__main__":
    main()
