import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load the fingerprint uniqueness data ---
df = pd.read_csv("fingerprint_analysis/fingerprint_uniqueness.tsv", sep="\t")

# Clean up possible missing / infinite values
for col in ["nn_same", "nn_other", "margin", "kNN10_density", "silhouette_cosine"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
df.replace([float("inf"), -float("inf")], pd.NA, inplace=True)
df.dropna(subset=["silhouette_cosine"], inplace=True)

# --- Create the figure layout ---
fig = plt.figure(figsize=(12, 6))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.2], height_ratios=[1, 1], hspace=0.4, wspace=0.35)

# Panel A: Correlation heatmap of metrics
# Panel A: Correlation heatmap of metrics
ax1 = fig.add_subplot(gs[:, 0])

# Replace any pd.NA with np.nan and ensure numeric dtype
import numpy as np
df = df.replace(pd.NA, np.nan)
df = df.fillna(0)

corr = df[["nn_same", "nn_other", "margin", "kNN10_density", "silhouette_cosine"]].apply(
    pd.to_numeric, errors="coerce"
).corr()

sns.heatmap(corr, annot=True, cmap="coolwarm", square=True, ax=ax1)
ax1.set_title("A. Correlation among uniqueness metrics")

# Panel B1: Histogram of silhouette scores
ax2 = fig.add_subplot(gs[0, 1])
sns.histplot(df["silhouette_cosine"], bins=40, kde=True, color="steelblue", ax=ax2)
ax2.set_xlabel("Silhouette (cosine)")
ax2.set_ylabel("Count")
ax2.set_title("B. Distribution of fingerprint uniqueness")

# Panel B2: Top-10 bar chart
ax3 = fig.add_subplot(gs[1, 1])
top = df.sort_values("silhouette_cosine", ascending=False).head(10)
sns.barplot(x="silhouette_cosine", y="label", data=top, palette="viridis", ax=ax3)
ax3.set_xlabel("Silhouette (cosine)")
ax3.set_ylabel("")
ax3.set_title("Top 10 most unique fingerprints")

# --- Final layout and save ---
plt.tight_layout()
plt.savefig("fingerprint_analysis/fingerprint_uniqueness_summary.png", dpi=300)
plt.show()

print("[ok] Saved fingerprint_analysis/fingerprint_uniqueness_summary.png")
