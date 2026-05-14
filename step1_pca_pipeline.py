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
