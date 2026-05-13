"""
Step 1: MNIST Digit Recognition — Data Loading + Scaling + PCA
"""

import numpy as np
import matplotlib.pyplot as plt
import joblib, os
from sklearn.datasets import fetch_openml          # ← real MNIST (784 features, 70k samples)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ─── 1. LOAD MNIST ────────────────────────────────────────────────────────────
print("Loading MNIST (downloads ~11MB on first run)...")
mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
X, y = mnist.data.astype(float), mnist.target.astype(int)
print("Dataset Shape:", X.shape)
print("Pixel Value Range:", X.min(), "to", X.max())

# ─── 2. TRAIN / TEST SPLIT ────────────────────────────────────────────────────
# MNIST comes pre-split: first 60k = train, last 10k = test
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]
print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# ─── 3. SCALING (StandardScaler) ──────────────────────────────────────────────
# Each of the 784 pixel features gets zero mean and unit variance.
# IMPORTANT: fit ONLY on training data, then transform both sets.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit + transform
X_test_scaled  = scaler.transform(X_test)         # transform only (no data leakage)
print("Scaling done.")

# ─── 4. PCA ───────────────────────────────────────────────────────────────────
# n_components=0.95 → keep enough components to explain 95% of variance.
# For MNIST this is typically ~154 components (down from 784).
pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)   # fit + transform
X_test_pca  = pca.transform(X_test_scaled)         # transform only

print("Original number of features:", 784)
print("Number of PCA components kept:", pca.n_components_)
print("We kept 95% of the variance")
print("Compression ratio:", round(784 / pca.n_components_, 1), "times smaller")

# ─── 5. VISUALISE ─────────────────────────────────────────────────────────────
cumvar = np.cumsum(pca.explained_variance_ratio_) * 100



#   --   Its Graphs for understanding the best values

fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor="#0d0d0d")
fig.suptitle("PCA Analysis — MNIST", color="white", fontsize=16, fontweight="bold")

# --- Cumulative variance curve ---
ax = axes[0]
ax.set_facecolor("#1a1a1a")
ax.plot(range(1, len(cumvar) + 1), cumvar, color="#00e5ff", linewidth=2.5)
ax.fill_between(range(1, len(cumvar) + 1), cumvar, alpha=0.15, color="#00e5ff")
ax.axhline(95, color="#ff4081", linestyle="--", linewidth=1.5, label="95% threshold")
ax.axvline(pca.n_components_, color="#ff4081", linestyle=":", linewidth=1.5)
ax.set_xlabel("Components", color="#bbb")
ax.set_ylabel("Cumulative Variance (%)", color="#bbb")
ax.set_title("Cumulative Explained Variance", color="white")
ax.legend(facecolor="#222", labelcolor="white", fontsize=9)
ax.tick_params(colors="#999"); ax.spines[:].set_color("#333")
ax.annotate(f" {pca.n_components_} components", xy=(pca.n_components_, 95),
            color="#ff4081", fontsize=9, va="bottom")

# --- Per-component variance bar ---
ax2 = axes[1]
ax2.set_facecolor("#1a1a1a")
ax2.bar(range(1, len(pca.explained_variance_ratio_) + 1),
        pca.explained_variance_ratio_ * 100, color="#00e5ff", width=0.8)
ax2.set_xlabel("Component", color="#bbb")
ax2.set_ylabel("Variance (%)", color="#bbb")
ax2.set_title("Per-Component Variance", color="white")
ax2.tick_params(colors="#999"); ax2.spines[:].set_color("#333")

# --- 2-D scatter of first two PCs ---
ax3 = axes[2]
ax3.set_facecolor("#1a1a1a")
# Use a subset for speed
idx = np.random.choice(len(X_train_pca), 3000, replace=False)
cmap = plt.get_cmap("tab10")
for digit in range(10):
    mask = y_train[idx] == digit
    ax3.scatter(X_train_pca[idx][mask, 0], X_train_pca[idx][mask, 1],
                c=[cmap(digit)], label=str(digit), alpha=0.5, s=8, edgecolors="none")
ax3.set_xlabel("PC 1", color="#bbb"); ax3.set_ylabel("PC 2", color="#bbb")
ax3.set_title("2D PCA — Class Separation", color="white")
ax3.legend(title="Digit", facecolor="#222", labelcolor="white",
           title_fontsize=8, fontsize=7, ncol=2, loc="best")
ax3.tick_params(colors="#999"); ax3.spines[:].set_color("#333")

plt.tight_layout()
plt.savefig("pca_analysis.png", dpi=140, bbox_inches="tight", facecolor="#0d0d0d")
plt.show()
print("Plot saved as pca_analysis.png")

# ─── 6. SAVE ARTIFACTS ────────────────────────────────────────────────────────
# These will be loaded again in Step 2 (KNN training) and Step 3 (Flask).
os.makedirs("mnist_model", exist_ok=True)
joblib.dump(scaler, "mnist_model/scaler.pkl")   # needed for preprocessing new images
joblib.dump(pca,    "mnist_model/pca.pkl")       # needed for transforming new images

np.save("mnist_model/X_train_pca.npy", X_train_pca)
np.save("mnist_model/X_test_pca.npy",  X_test_pca)
np.save("mnist_model/y_train.npy", y_train)
np.save("mnist_model/y_test.npy",  y_test)

print("\n=== Saved to mnist_model/ ===")
print("  scaler.pkl       ← StandardScaler (fit on train)")
print("  pca.pkl          ← PCA (fit on train)")
print("  X_train_pca.npy  ← transformed training features")
print("  X_test_pca.npy   ← transformed test features")
print("  y_train.npy      ← training labels")
print("  y_test.npy       ← test labels")
