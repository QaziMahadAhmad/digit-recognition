"""
Step 2: KNN Training with Cross-Validation
===========================================
Run AFTER step1_pca_pipeline.py has been executed.
This script loads the PCA-transformed data, finds the best K,
trains the final KNN model, evaluates it, and saves it.

Folder structure expected:
    mnist_model/
        scaler.pkl
        pca.pkl
        X_train_pca.npy
        X_test_pca.npy
        y_train.npy
        y_test.npy

Install deps:
    pip install scikit-learn numpy matplotlib joblib
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report
)

# ─── 1. LOAD PCA-TRANSFORMED DATA ────────────────────────────────────────────
print("Loading PCA data from mnist_model/...")
X_train_pca = np.load("mnist_model/X_train_pca.npy")
X_test_pca  = np.load("mnist_model/X_test_pca.npy")
y_train     = np.load("mnist_model/y_train.npy")
y_test      = np.load("mnist_model/y_test.npy")
pca         = joblib.load("mnist_model/pca.pkl")

print(f"  Train shape : {X_train_pca.shape}")
print(f"  Test shape  : {X_test_pca.shape}")
print(f"  PCA dims    : {X_train_pca.shape[1]}")

# ─── 2. FIND BEST K VIA CROSS-VALIDATION ─────────────────────────────────────
# We try multiple K values and use 5-fold stratified cross-validation.
# Stratified = each fold has the same class distribution as the full set.
k_values  = [1, 3, 5, 7, 9, 11, 15, 19, 25]
cv_scores = []
cv_stds   = []
cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\nCross-validating K values (5-fold stratified)...")
for k in k_values:
    knn    = KNeighborsClassifier(n_neighbors=k, metric="euclidean", n_jobs=-1)
    scores = cross_val_score(knn, X_train_pca, y_train, cv=cv, scoring="accuracy")
    cv_scores.append(scores.mean())
    cv_stds.append(scores.std())
    print(f"  K={k:2d}  accuracy={scores.mean():.4f} ± {scores.std():.4f}")

best_k   = k_values[int(np.argmax(cv_scores))]
best_acc = max(cv_scores)
print(f"\n  ✅ Best K = {best_k}  (CV accuracy = {best_acc*100:.2f}%)")

# ─── 3. TRAIN FINAL KNN MODEL ────────────────────────────────────────────────
# Train on the FULL training set using the best K found above.
print(f"\nTraining final KNN with K={best_k}...")
knn_final = KNeighborsClassifier(n_neighbors=best_k, metric="euclidean", n_jobs=-1)
knn_final.fit(X_train_pca, y_train)

# ─── 4. EVALUATE ON TEST SET ─────────────────────────────────────────────────
y_pred   = knn_final.predict(X_test_pca)
test_acc = accuracy_score(y_test, y_pred)
cm       = confusion_matrix(y_test, y_pred)
per_class_acc = cm.diagonal() / cm.sum(axis=1)

print(f"\n  Test Accuracy : {test_acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

# ─── 5. VISUALISE RESULTS ────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 10), facecolor="#0d0d0d")
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
fig.suptitle("KNN Training Results", color="white", fontsize=18, fontweight="bold", y=0.98)

# --- Plot 1: K vs CV accuracy ---
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor("#1a1a1a")
ax1.plot(k_values, cv_scores, color="#00e5ff", linewidth=2.5, marker="o", markersize=7, zorder=3)
ax1.fill_between(
    k_values,
    [s - e for s, e in zip(cv_scores, cv_stds)],
    [s + e for s, e in zip(cv_scores, cv_stds)],
    alpha=0.2, color="#00e5ff"
)
ax1.axvline(best_k, color="#ff4081", linestyle="--", linewidth=1.5, label=f"Best K={best_k}")
ax1.scatter([best_k], [best_acc], color="#ff4081", s=100, zorder=5)
ax1.set_xlabel("K (neighbors)", color="#bbb")
ax1.set_ylabel("CV Accuracy", color="#bbb")
ax1.set_title("K vs Cross-Val Accuracy", color="white", fontsize=11)
ax1.legend(facecolor="#222", labelcolor="white", fontsize=9)
ax1.tick_params(colors="#999")
ax1.spines[:].set_color("#333")

# --- Plot 2: Confusion matrix ---
ax2 = fig.add_subplot(gs[0, 1:])
ax2.set_facecolor("#1a1a1a")
im = ax2.imshow(cm, cmap="Blues", interpolation="nearest")
plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
ax2.set_xticks(range(10)); ax2.set_yticks(range(10))
ax2.set_xticklabels(range(10), color="#bbb")
ax2.set_yticklabels(range(10), color="#bbb")
ax2.set_xlabel("Predicted Label", color="#bbb")
ax2.set_ylabel("True Label", color="#bbb")
ax2.set_title("Confusion Matrix", color="white", fontsize=11)
for i in range(10):
    for j in range(10):
        ax2.text(j, i, str(cm[i, j]), ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "#aaa", fontsize=9)
ax2.spines[:].set_color("#333")

# --- Plot 3: Per-class accuracy ---
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_facecolor("#1a1a1a")
bar_colors = ["#00e5ff" if v >= 0.95 else "#ff9800" if v >= 0.90 else "#ff4081"
              for v in per_class_acc]
bars = ax3.bar(range(10), per_class_acc * 100, color=bar_colors, width=0.7)
ax3.set_xticks(range(10))
ax3.set_xticklabels(range(10), color="#bbb")
ax3.set_xlabel("Digit", color="#bbb")
ax3.set_ylabel("Accuracy (%)", color="#bbb")
ax3.set_title("Per-Class Accuracy", color="white", fontsize=11)
ax3.set_ylim(0, 110)
for bar, val in zip(bars, per_class_acc):
    ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
             f"{val*100:.1f}%", ha="center", va="bottom", color="white", fontsize=8)
ax3.tick_params(colors="#999")
ax3.spines[:].set_color("#333")

# --- Plot 4: Summary stats ---
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor("#1a1a1a")
ax4.axis("off")
stats = [
    ("Best K",          f"{best_k}"),
    ("CV Accuracy",     f"{best_acc*100:.2f}%"),
    ("Test Accuracy",   f"{test_acc*100:.2f}%"),
    ("Train Samples",   f"{len(X_train_pca):,}"),
    ("Test Samples",    f"{len(X_test_pca):,}"),
    ("PCA Components",  f"{X_train_pca.shape[1]}"),
    ("Distance Metric", "Euclidean"),
]
for i, (label, val) in enumerate(stats):
    y_pos = 0.88 - i * 0.13
    ax4.text(0.05, y_pos, label, transform=ax4.transAxes,
             color="#888", fontsize=10, va="center")
    ax4.text(0.95, y_pos, val, transform=ax4.transAxes,
             color="#00e5ff", fontsize=11, fontweight="bold", va="center", ha="right")
ax4.set_title("Model Summary", color="white", fontsize=11, pad=10)

# --- Plot 5: CV std dev (stability) ---
ax5 = fig.add_subplot(gs[1, 2])
ax5.set_facecolor("#1a1a1a")
ax5.bar(k_values, [s * 100 for s in cv_stds], color="#7c4dff", width=1.5, alpha=0.8)
ax5.set_xlabel("K", color="#bbb")
ax5.set_ylabel("Std Dev (%)", color="#bbb")
ax5.set_title("CV Stability (lower = better)", color="white", fontsize=11)
ax5.tick_params(colors="#999")
ax5.spines[:].set_color("#333")

plt.savefig("knn_results.png", dpi=140, bbox_inches="tight", facecolor="#0d0d0d")
plt.show()
print("Plot saved as knn_results.png")

# ─── 6. SAVE THE TRAINED MODEL ───────────────────────────────────────────────
# Flask will load knn_model.pkl, scaler.pkl, and pca.pkl at startup.
joblib.dump(knn_final, "mnist_model/knn_model.pkl")
print("\n=== Saved to mnist_model/ ===")
print("  knn_model.pkl  ← trained KNN classifier")
print("  (scaler.pkl and pca.pkl were already saved in Step 1)")
print("\n✅ Step 2 complete. Ready for Step 3: Flask backend.")