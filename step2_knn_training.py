"""
Step 2: KNN Training with Cross-Validation         """

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

print("Training data shape after PCA:", X_train_pca.shape)
print("Testing data shape after PCA:", X_test_pca.shape)
print("Number of PCA dimensions:", X_train_pca.shape[1])

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
print("\nBest K found:", best_k)
print("Model achieved", round(best_acc * 100, 2), "% accuracy in cross-validation")

# ─── 3. TRAIN FINAL KNN MODEL ────────────────────────────────────────────────
# Train on the FULL training set using the best K found above.
print("Training final KNN model with K =", best_k)
knn_final = KNeighborsClassifier(n_neighbors=best_k, metric="euclidean", n_jobs=-1)
knn_final.fit(X_train_pca, y_train)

# ─── 4. EVALUATE ON TEST SET ─────────────────────────────────────────────────
y_pred   = knn_final.predict(X_test_pca)
test_acc = accuracy_score(y_test, y_pred)
cm       = confusion_matrix(y_test, y_pred)
per_class_acc = cm.diagonal() / cm.sum(axis=1)

print("\nTest Accuracy:", round(test_acc * 100, 2), "%")
print("Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

# ─── 6. SAVE THE TRAINED MODEL ───────────────────────────────────────────────
# Flask will load knn_model.pkl, scaler.pkl, and pca.pkl at startup.
joblib.dump(knn_final, "mnist_model/knn_model.pkl")
print("\n=== Saved to mnist_model/ ===")
print("  knn_model.pkl  ← trained KNN classifier")
print("  (scaler.pkl and pca.pkl were already saved in Step 1)")
print("\n Step 2 complete")