# Pneumonia Classification Using Deep Learning
### A Comparative Study of Oversampling Techniques for Imbalanced Medical Image Data

---

## 🚀 Live Demo

**[View Interactive Report →](https://harsh-raj00.github.io/ML_PROJECT/)**

---

## Overview

This project investigates the effectiveness of different oversampling techniques for addressing class imbalance in pneumonia detection from chest X-ray images. Three training conditions are evaluated using a baseline CNN:

| Method | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Original Dataset | 97.56% | 97.42% | 99.58% | 98.49% |
| VAE + SMOTE | 96.89% | 96.38% | 99.86% | 98.09% |
| **GAN** | **98.00%** | **98.35%** | **99.17%** | **98.76%** |

---

## Dataset

**Source:** [Chest X-Ray Images (Pneumonia) — Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

| Class | Label | Samples | Percentage |
|---|---|---|---|
| 0 | Normal | 900 | 20% |
| 1 | Pneumonia | 3,600 | 80% |

- **Imbalance Ratio:** 1:4 (Normal : Pneumonia)
- **Image Size:** 224 × 224 pixels (grayscale)
- **Train / Val Split:** 80% / 20%, stratified

---

## Project Structure

```
.
├── dataset.ipynb           # Dataset download, preprocessing & H5 export
├── train_eval.py           # Main training script (CNN, VAE+SMOTE, GAN)
├── results_metrics.json    # Saved evaluation metrics (accuracy, F1, etc.)
├── index.html              # Interactive explainable AI report (deployed)
└── README.md
```

---

## Methodology

### 1. Baseline CNN

A lightweight CNN used consistently across all three conditions:

```
Conv2D(8, 3×3) → ReLU → MaxPool
Conv2D(16, 3×3) → ReLU → MaxPool
Flatten → FC(32) → ReLU → FC(1) → Sigmoid
```

- **Loss:** Binary Cross-Entropy  
- **Optimizer:** Adam (lr = 1e-3)  
- **Epochs:** 5  
- **Batch size:** 32  

---

### 2. VAE + SMOTE

Combines a Variational Autoencoder with SMOTE interpolation in latent space:

- **Encoder:** 3 × Conv2D layers → 128-dim latent vector  
- **Decoder:** 3 × ConvTranspose2D layers → reconstructed image  
- **SMOTE** applied to latent representations of the minority class  
- **1,440 synthetic** Class 0 samples generated → Final Class 0: **2,160**

---

### 3. GAN-Based Augmentation

Adversarial training to generate high-fidelity synthetic X-rays:

- **Generator:** FC(128 → 64×28×28) + 3 × ConvTranspose2D → 224×224 image  
- **Discriminator:** 2 × Conv2D + FC → real/fake probability  
- **Training:** 5 epochs on minority class images  
- **1,440 synthetic** Class 0 samples generated → Final Class 0: **2,160**

---

## Results

### Confusion Matrices

|  | Original | VAE + SMOTE | GAN |
|---|---|---|---|
| True Negative | 161 | 153 | **168** |
| False Positive | 19 | 27 | **12** |
| False Negative | 3 | **1** | 6 |
| True Positive | 717 | **719** | 714 |

### Key Findings

- **GAN** achieved the best overall accuracy (98.0%) and F1-score (98.75%), with the fewest false positives (12).
- **VAE+SMOTE** achieved the highest recall (99.86%) but at the cost of more false positives (27), lowering precision significantly.
- **Original** imbalanced data performed surprisingly well, but GAN augmentation improved accuracy by **+2.44%**.

---

## Installation

```bash
pip install torch torchvision numpy scikit-learn imbalanced-learn h5py opencv-python kagglehub
```

---

## Usage

### Step 1 — Download & prepare the dataset

Run the data preparation notebook to download from Kaggle and generate `train_pneumonia_subset_correct.h5`:

```bash
jupyter notebook dataset.ipynb
```

### Step 2 — Run training

```bash
python train_eval.py
```

This trains three models sequentially (Original → VAE+SMOTE → GAN) and saves metrics to `results_metrics.json`.

### Step 3 — View report

Open `index.html` locally or visit the **[live demo](https://harsh-raj00.github.io/ML_PROJECT/)** for the interactive explainability report.

---

## Reproducibility

All experiments use a fixed seed for deterministic results:

```python
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
```

---

## References

1. Kingma, D. P., & Welling, M. (2013). *Auto-Encoding Variational Bayes.* arXiv:1312.6114.
2. Chawla, N. V., et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique.* Journal of Artificial Intelligence Research, 16, 321–357.
3. Goodfellow, I., et al. (2014). *Generative Adversarial Nets.* Advances in Neural Information Processing Systems, 27.
4. He, H., & Garcia, E. A. (2009). *Learning from Imbalanced Data.* IEEE TKDE, 21(9), 1263–1284.

---

## License

This project is for academic and research purposes. The dataset is subject to Kaggle's terms of use.
