#!/usr/bin/env python3
import sys
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, linregress

if len(sys.argv) < 3:
    print("Usage: python plot_theta_vs_omega_dev.py <omega_csv> <fingerprint_csv>")
    sys.exit(1)

omega_file, theta_file = sys.argv[1], sys.argv[2]

# --- Load data ---
omega_df = pd.read_csv(omega_file)
theta_df = pd.read_csv(theta_file)

# Identify theta column (could be theta_pp_deg, theta_deg, etc.)
theta_col = next((c for c in theta_df.columns if "theta" in c.lower()), None)
if theta_col is None:
    raise ValueError("No θ column found (looked for 'theta' in headers).")

# --- Merge on residue index ---
merge_col = "res_i" if "res_i" in omega_df.columns and "res_i" in theta_df.columns else None
if merge_col:
    merged = pd.merge(omega_df, theta_df, on=merge_col)
else:
    print("Warning: could not find residue index; truncating to same length.")
    merged = pd.concat([omega_df.reset_index(drop=True), theta_df.reset_index(drop=True)], axis=1)

if "omega_deg" not in merged.columns:
    raise ValueError("Column 'omega_deg' not found in omega CSV.")

# --- Compute |ω deviation| from 180° ---
merged["omega_dev"] = abs(((merged["omega_deg"] + 180) % 360) - 180)

# --- Clean up ---
merged = merged.dropna(subset=["omega_dev", theta_col])

# --- Statistics ---
pearson_r, pearson_p = pearsonr(merged[theta_col], merged["omega_dev"])
spearman_r, spearman_p = spearmanr(merged[theta_col], merged["omega_dev"])
slope, intercept, r_val, p_val, stderr = linregress(merged[theta_col], merged["omega_dev"])

print("\n=== θ vs |ω deviation| Statistics ===")
print(f"Pearson r:  {pearson_r:.3f}  (p={pearson_p:.3e})")
print(f"Spearman r: {spearman_r:.3f}  (p={spearman_p:.3e})")
print(f"Linear Fit: y = {slope:.3f}x + {intercept:.3f}")
print(f"R² = {r_val**2:.3f}")

# --- Plot ---
plt.figure(figsize=(8,6))
plt.scatter(merged[theta_col], merged["omega_dev"], s=15, alpha=0.7, color="purple", label="Residues")
plt.plot(merged[theta_col], intercept + slope*merged[theta_col], 'r--', label=f"Fit (R²={r_val**2:.3f})")
plt.xlabel("θ angle between planes (degrees)")
plt.ylabel("|ω deviation| (degrees)")
plt.title("|ω deviation| vs θ – Coupling of torsion and plane orientation")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
