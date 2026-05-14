---
title: Digit Recognition PCA KNN
emoji: ✏️
colorFrom: cyan
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

<div align="center">

# ✏️ Digit.AI — Handwritten Digit Recognition

**A from-scratch machine learning pipeline that recognises handwritten digits in real time.**
Built with PCA dimensionality reduction + K-Nearest Neighbours — no deep learning, no black box.

[![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?style=flat-square&logo=scikit-learn)](https://scikit-learn.org)
[![MNIST](https://img.shields.io/badge/Dataset-MNIST-yellow?style=flat-square)](http://yann.lecun.com/exdb/mnist/)
[![Accuracy](https://img.shields.io/badge/Accuracy-95.03%25-brightgreen?style=flat-square)]()
[![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)](LICENSE)

</div>

---

## 🧠 What is this?

Digit.AI is a complete end-to-end machine learning system that:

- Trains on **70,000 MNIST handwritten digit images**
- Compresses 784 pixel features down to **331 dimensions using PCA** (keeping 95% of information)
- Classifies digits 0–9 using **K-Nearest Neighbours** with K=3
- Serves predictions through a **FastAPI backend** with a clean draw/upload UI
- Achieves **95.03% test accuracy** — with zero neural networks

> The goal of this project is to demonstrate that classical ML, done right, is powerful, interpretable, and fast.

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Algorithm | K-Nearest Neighbours (K=3) |
| Dataset | MNIST (60k train / 10k test) |
| Original features | 784 (28×28 pixels) |
| After PCA | 331 components |
| Variance retained | 95% |
| Cross-val accuracy | 94.99% |
| **Test accuracy** | **95.03%** |
| Distance metric | Euclidean |

### Per-class accuracy

| Digit | Precision | Recall | F1 |
|-------|-----------|--------|----|
| 0 | 95.19% | 98.88% | 97.00% |
| 1 | 96.32% | 99.30% | 97.79% |
| 2 | 95.71% | 95.06% | 95.38% |
| 3 | 93.31% | 95.25% | 94.27% |
| 4 | 95.39% | 94.91% | 95.15% |
| 5 | 93.99% | 92.94% | 93.46% |
| 6 | 96.67% | 96.87% | 96.77% |
| 7 | 93.84% | 93.39% | 93.61% |
| 8 | 96.08% | 90.66% | 93.29% |
| 9 | 93.67% | 92.37% | 93.01% |

---

## 🔬 How It Works

```
User draws/uploads digit
         ↓
  Convert to grayscale
         ↓
   Resize to 28×28
         ↓
  Invert if needed
  (white digit on black)
         ↓
   Flatten → 784 values
         ↓
  StandardScaler.transform()
         ↓
    PCA.transform()
    784 → 331 dims
         ↓
   KNN.predict()
   find 3 nearest neighbours
         ↓
  Return digit + confidence
```

### Why PCA before KNN?

KNN suffers from the **curse of dimensionality** — in 784 dimensions, every point looks equally distant from every other point, making nearest-neighbour meaningless. PCA compresses the features while keeping 95% of the variance, making KNN fast and accurate.

### Why StandardScaler before PCA?

Without scaling, pixels near the center of the image (which vary more) would dominate the PCA directions. Scaling gives every pixel equal weight so PCA finds truly meaningful components.

---

## 🗂️ Project Structure

```
digit-recognition/
│
├── step1_pca_pipeline.py     ← Load MNIST · Scale · Apply PCA · Save artifacts
├── step2_knn_training.py     ← Cross-validate K · Train KNN · Evaluate · Save model
├── app.py                    ← FastAPI server + embedded HTML/CSS/JS UI
│
├── mnist_model/
│   ├── scaler.pkl            ← Fitted StandardScaler
│   ├── pca.pkl               ← Fitted PCA (784 → 331 dims)
│   └── knn_model.pkl         ← Trained KNN (K=3, Euclidean)
│
├── Dockerfile                ← Container config for deployment
├── requirements.txt          ← Python dependencies
└── README.md                 ← You are here
```

---

## 🚀 Run Locally

### Prerequisites
```bash
pip install fastapi uvicorn scikit-learn numpy pillow joblib matplotlib
```

### Step 1 — Train the model
```bash
# Downloads MNIST (~11MB), applies PCA, saves artifacts
python step1_pca_pipeline.py

# Cross-validates K values, trains final KNN, saves model
python step2_knn_training.py
```

### Step 2 — Launch the app
```bash
uvicorn app:app --reload
```

Open **http://127.0.0.1:8000** in your browser.

---

## 🐳 Run with Docker

```bash
docker build -t digit-recognition .
docker run -p 7860:7860 digit-recognition
```

Open **http://localhost:7860**

---

## 🌐 API Reference

### `GET /`
Returns the full web UI (HTML/CSS/JS embedded in Python).

### `POST /predict`

**Request body:**
```json
{
  "image": "data:image/png;base64,..."
}
```

**Response:**
```json
{
  "digit": 7,
  "confidence": 100.0,
  "probabilities": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0]
}
```

Interactive API docs available at `/docs` (Swagger UI).

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10 |
| ML | scikit-learn |
| Image processing | Pillow (PIL) |
| Web framework | FastAPI |
| Server | Uvicorn |
| Serialization | joblib |
| Deployment | Docker · Hugging Face Spaces |

---

## 📁 Training Pipeline Details

### step1_pca_pipeline.py
- Loads MNIST via `fetch_openml`
- Splits into 60k train / 10k test
- Fits `StandardScaler` on training data only (no data leakage)
- Fits `PCA(n_components=0.95)` — automatically selects 331 components
- Saves `scaler.pkl` and `pca.pkl`
- Generates variance analysis plots

### step2_knn_training.py
- Loads PCA-transformed data from `mnist_model/`
- Runs 5-fold stratified cross-validation for K ∈ {1, 3, 5, 7, 9, 11, 15, 19, 25}
- Selects best K = 3 (CV accuracy = 94.99%)
- Trains final KNN on full training set
- Evaluates on held-out test set (95.03%)
- Saves `knn_model.pkl`
- Generates confusion matrix and per-class accuracy plots

---

## 👤 Author

**Mahad Ahmad**
- GitHub: [@QaziMahadAhmad](https://github.com/QaziMahadAhmad)
- Hugging Face: [@Mahad0007](https://huggingface.co/Mahad0007)

---

## 📄 License

MIT License — feel free to use, modify, and distribute.