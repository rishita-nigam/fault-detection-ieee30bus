# ⚡ IEEE 30-Bus Power System — Explainable ML Fault Classification

> Classifying power system faults using Random Forest and SVM with physical validation and XAI on the IEEE 30-bus benchmark network.

---

## 📌 Overview

This project applies machine learning to **automatically detect and classify electrical faults** in the IEEE 30-bus power system. Given real-time bus measurements (voltage, current, power), the models identify which type of fault — if any — has occurred.

The pipeline is end-to-end: MATLAB simulation → dataset generation → ML training → XAI explainability → physical validation.

---

## 🔍 Fault Classes

| Label | Fault Type | Description |
|-------|-----------|-------------|
| 0 | **Normal** | No fault — healthy system operation |
| 1 | **LG** | Line-to-Ground fault |
| 2 | **LL** | Line-to-Line fault |
| 3 | **LLG** | Double Line-to-Ground fault |
| 4 | **LLL** | Three-Phase fault (most severe) |

---

## ⚙️ How It Works

### 1. Dataset Generation (MATLAB)
The IEEE 30-bus network is simulated in MATLAB/Simulink. For each fault type (LG, LL, LLG, LLL) across multiple fault locations and resistance values, the simulation records per-bus measurements:
- **V_mag** — Voltage magnitude (pu)
- **V_ang** — Voltage angle (degrees)
- **I_mag** — Current magnitude (pu)
- **P** — Active power (MW)
- **Q** — Reactive power (MVAR)

### 2. Python ML Pipeline (`ml_pipeline.py`)

| Stage | Details |
|-------|---------|
| **Data loading** | Merges fault + normal CSVs |
| **Reshaping** | Long → Wide (one row per simulation sample) |
| **Feature engineering** | System-level aggregates: `V_mag_mean`, `V_mag_min`, `V_mag_std`, `I_mag_max`, `I_mag_std`, `P_total`, `Q_total` |
| **Models** | Random Forest (200 trees) · SVM (RBF kernel, C=10) |
| **Evaluation** | Accuracy, Weighted F1, 5-Fold Stratified CV |
| **XAI** | Permutation Importance (model-agnostic) + Gini Importance |
| **Physical Validation** | Validates model behaviour against known power system physics |

---

## 📊 Results

| Model | Accuracy | Weighted F1 | 5-Fold CV |
|-------|----------|-------------|-----------|
| **Random Forest** | **~86.5%** | — | **~83.8% ± σ** |
| SVM (RBF) | — | — | — |

### Physical Validation Summary
The model's per-class feature means were checked against known power system behaviour:

| Check | Expected | Result |
|-------|----------|--------|
| `V_mag_min` decreases under fault | ✅ Voltage sag | ✅ Pass |
| `V_mag_std` increases under fault | ✅ Uneven sag | ✅ Pass |
| `I_mag_max` increases under fault | ✅ Fault current surge | ✅ Pass |
| LLL > LLG > LG severity ordering | ✅ Standard physics | ✅ Pass |

> **7/8 physical checks passed.** One flagged ambiguity: LG vs LLL confusion at high fault impedance (expected due to overlapping electrical signatures at Rf → ∞).

---

## 📈 Output Plots

After running the pipeline, these plots are saved to the working directory:

| File | Description |
|------|-------------|
| `confusion_matrices.png` | RF vs SVM confusion matrices |
| `model_comparison.png` | Accuracy & F1 bar chart |
| `xai_permutation_importance.png` | Top-20 features by permutation importance |
| `physical_validation_FINAL.png` | Voltage sag & current swell per fault class |
| `cross_validation.png` | 5-fold CV accuracy per fold |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- MATLAB R2021a+ (for dataset generation only)

### Installation

```bash
git clone https://github.com/yourusername/fault-classification-ieee30.git
cd fault-classification-ieee30
pip install -r requirements.txt
```

### Running the Pipeline

1. **Update dataset paths** in `ml_pipeline.py`:
```python
FAULT_CSV      = "data/fault_dataset_long.csv"
NORMAL_CSV     = "data/normal_dataset_long.csv"
FAULT_WIDE_CSV = "data/fault_dataset.csv"
```

2. **Run:**
```bash
python ml_pipeline.py
```

All output plots will be saved to the current directory.

---

## 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 🔑 Key Findings

- **Voltage angle shift** (`V_ang`) is the single most discriminative feature — faults cause a dramatic shift from ~20° (normal) to ~90° (fault)
- **Random Forest** outperforms SVM on this dataset due to the non-linear, high-dimensional feature space
- **LLL faults** are easiest to classify (largest electrical signature); **LG faults** are hardest (can mimic normal at high fault resistance)
- Physical validation confirms the model has learned genuine fault physics, not statistical artifacts

---

## 🏫 About

Developed as part of a Power Systems & Machine Learning project at **VIT** on the IEEE 30-bus benchmark network.

---

## 📄 License

This project is for academic use. Feel free to use and build upon it with attribution.
