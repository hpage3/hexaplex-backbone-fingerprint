#!/usr/bin/env python3
import sys
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, linregress

if len(sys.argv) < 3:
    print("Usage: python plot_rms_vs_omega.py <omega_csv> <fingerprint_csv>")
    sys.exit(1)

omega_file, rms_file = sys.argv[1], sys.argv[2]
omega_df = pd.read_csv(omega_file)
rms_df = pd.read_csv(rms_file)

# --- Find RMS column ---
rms_col = next((c for c in rms_df.columns if "rms" in c.lower()), None)
if rms_col is None:
    raise ValueError("No RMS column (e.g., 'rms' or 'box_rms') found in fingerprint file.")

# --- Merge on residue index ---
merge_col = "res_i" if "res_i" in omega_df.columns and "res_i" in rms_df.columns else None
if merge_col:
    merged = pd.merge(omega_df, rms_df, on=merge_col)
else:
    print("Warning: could not find residue index to merge on; truncating to min length.")
    merged = pd.concat([omega_df.reset_index(drop=True), rms_df.reset_index(drop=True)], axis=1)

if "omega_deg" not in merged.columns:
    raise ValueError("Column 'omega_deg' not found in omega CSV.")

merged = merged.dropna(subset=["omega_deg", rms_col])

# --- Compute deviation from planarity ---
merged["omega_dev"] = abs(((merged["omega_deg"] + 180) % 360) - 180)

# --- Descriptive statistics ---
desc = merged[["omega_deg", "omega_dev", rms_col]].describe()
print("\n=== Descriptive Statistics ===")
print(desc.to_string(float_format=lambda x: f"{x:8.3f}"))

# --- Correlations ---
def corr_stats(x, y):
    pearson_r, pearson_p = pearsonr(x, y)
    spearman_r, spearman_p = spearmanr(x, y)
    slope, intercept, r_val, p_val, stderr = linregress(x, y)
    return dict(
        pearson_r=pearson_r, pearson_p=pearson_p,
        spearman_r=spearman_r, spearman_p=spearman_p,
        slope=slope, intercept=intercept, r2=r_val**2
    )

raw_stats = corr_stats(merged[rms_col], merged["omega_deg"])
dev_stats = corr_stats(merged[rms_col], merged["omega_dev"])

print("\n=== RMS vs ω Statistics ===")
for k,v in raw_stats.items():
    print(f"{k:12s}: {v:8.4f}")

print("\n=== RMS vs |ω deviation| Statistics ===")
for k,v in dev_stats.items():
    print(f"{k:12s}: {v:8.4f}")

# --- Plot raw ω ---
plt.figure(figsize=(8,6))
plt.scatter(merged[rms_col], merged["omega_deg"], s=15, alpha=0.7, color="steelblue", label="Residues")
plt.plot(merged[rms_col], raw_stats["intercept"] + raw_stats["slope"]*merged[rms_col], 
         'r--', label=f"Fit (R²={raw_stats['r2']:.3f})")
plt.xlabel("Box RMS (Å)")
plt.ylabel("ω angle (degrees)")
plt.title("ω vs Box RMS – Peptide Planarity Deviation")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# --- Plot deviation ---
plt.figure(figsize=(8,6))
plt.scatter(merged[rms_col], merged["omega_dev"], s=15, alpha=0.7, color="orange", label="Residues")
plt.plot(merged[rms_col], dev_stats["intercept"] + dev_stats["slope"]*merged[rms_col], 
         'r--', label=f"Fit (R²={dev_stats['r2']:.3f})")
plt.xlabel("Box RMS (Å)")
plt.ylabel("|ω deviation| (degrees)")
plt.title("|ω deviation| vs Box RMS – Peptide Planarity Distortion")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
