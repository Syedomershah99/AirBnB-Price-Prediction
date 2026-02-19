# Eigenflowers — PCA on the Oxford Flowers Dataset

An end-to-end Principal Component Analysis (PCA) pipeline applied to flower images from the [Oxford 102 Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) dataset. Built for CSE 4/555 Pattern Recognition (Spring 2026, University at Buffalo).

## Overview

The notebook preprocesses flower images (grayscale, resize, flatten), builds a data matrix, and computes PCA using three independent methods. It then visualizes eigenflowers, reconstructs images from a reduced set of components, and analyzes explained variance.

### Methods Implemented

| # | Method | Approach |
|---|--------|----------|
| 1 | **Thin SVD** | `np.linalg.svd` on the centered data matrix directly |
| 2 | **Dual Trick (eigh)** | Eigendecompose the small M×M Gram matrix via `np.linalg.eigh`, then lift eigenvectors back to pixel space |
| 3 | **Dual Trick (Power Method)** | Hand-written power iteration with rank-1 deflation — no library eigensolvers |

All three produce identical eigenvalues (verified in the notebook).

## Results

- **30 training images** (6 flower classes, 5 images each) resized to 64×64 grayscale
- **Mean image** computed and subtracted for centering
- **Top-6 eigenflowers** visualized for each method
- **Reconstruction** shown at k = 5, 10, 15, 20, 25, 30 components with MSE curves for both training and test images
- **Cumulative EVR** plot — 19 components capture 90% of variance
- **T-SNE** visualization of training data colored by flower class (bonus)

## Repository Structure

```
Homework1-Eigenflowers/
├── Homework1_eigenflowers.ipynb   # Complete notebook with code, plots, and explanations
├── dataset/
│   ├── train/                     # 30 training images + train.csv (class labels)
│   └── test/                      # 18 test images + test.csv (class labels)
├── Homework1.pdf                  # Assignment specification
└── README.md
```

## How to Run

```bash
# 1. Clone the repo
git clone <repo-url> && cd Homework1-Eigenflowers

# 2. Install dependencies
pip install numpy pillow matplotlib scikit-learn

# 3. Open and run the notebook
jupyter notebook Homework1_eigenflowers.ipynb
```

> Set `TOKEN` in the first code cell to the last 4 digits of your UB person number before running.

## Key Concepts

- **PCA** — finds orthogonal directions of maximum variance in the data
- **SVD** — factors A = UΣVᵀ; eigenvalues of the covariance are σᵢ²/M
- **Dual Trick** — works with the small M×M matrix AᵀA/M instead of the large p×p covariance AAᵀ/M
- **Power Method** — iteratively multiplies a random vector by the matrix; converges to the dominant eigenvector
- **Deflation** — removes a found eigenpair (L ← L − λvvᵀ) to reveal the next one
- **Reconstruction** — approximates an image using only the top-k eigenimages plus the mean
- **EVR** — fraction of total variance captured by each principal component
- **T-SNE** — non-linear 2D embedding for visualizing high-dimensional clusters

## Dependencies

- Python ≥ 3.8
- NumPy
- Pillow (PIL)
- Matplotlib
- scikit-learn (for T-SNE bonus)

## License

Academic coursework — University at Buffalo, CSE 4/555 Pattern Recognition.
