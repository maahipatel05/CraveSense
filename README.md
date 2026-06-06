# CraveSense

**Predicting substance cravings before they happen — by fusing brain scans, wearable sensors, and real-time psychological surveys with deep learning.**

---

## Overview

CraveSense is a multimodal machine learning research pipeline that predicts whether a person is currently experiencing a craving, and whether they will crave 90 minutes from now. It fuses three clinically grounded data streams — ecological momentary assessment (EMA) surveys, Fitbit wearable sensor data, and resting-state fMRI brain connectivity — into a unified representation for binary classification.

The pipeline runs two independent experiments:

| Experiment | Task | Input Timing |
|---|---|---|
| Real-Time Detection | Is the participant craving right now? | Current window (0 min shift) |
| Standardized Forecasting | Will they crave 90 minutes from now? | Data from 90 minutes earlier |

This work demonstrates that biological signals from wearables and neuroimaging can meaningfully improve craving prediction beyond what psychological self-report alone achieves.

---

## Problem Statement

Substance use disorders and compulsive eating disorders are characterized by intense, time-varying cravings that are notoriously difficult to predict. Clinical interventions (therapy, medication prompts, support calls) are most effective when delivered at the right moment — ideally before a craving peaks. Current approaches rely on patients self-reporting, which is reactive, not proactive.

CraveSense asks: **can objective physiological and neurological signals anticipate a craving before the person is fully aware of it?**

---

## Why This Matters

- **Clinical impact:** A 90-minute advance warning of a craving event creates a practical intervention window — enough time to trigger a therapist call, deliver a mobile CBT prompt, or recommend a behavioral strategy.
- **Scientific novelty:** Most craving studies treat modalities in isolation. Fusing EMA, wearables, and fMRI within a single Leave-One-Group-Out framework is methodologically rigorous and rare at this scale.
- **Generalizable architecture:** The multimodal fusion approach (survey features + LSTM-encoded time series + neural-compressed neuroimaging) is applicable to any condition with longitudinal ecological and biological data: PTSD, chronic pain, depression episodes.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          RAW DATA SOURCES                               │
│                                                                         │
│  EMA Surveys          Fitbit Wearable         fMRI (Resting-State)      │
│  (3,178 rows)         (2.2M minute rows)      (29 participants)         │
│  Stress, Mood,        Heart Rate, Steps       164-ROI Connectivity      │
│  Craving, Affect      per minute              Matrix (496 pairs)        │
└──────────┬────────────────────┬──────────────────────┬──────────────────┘
           │                    │                      │
           ▼                    ▼                      ▼
┌──────────────────┐  ┌─────────────────────┐  ┌──────────────────────┐
│  Feature         │  │  LSTM Autoencoder   │  │  Feedforward         │
│  Engineering     │  │  (PyTorch)          │  │  Autoencoder         │
│                  │  │                     │  │  (PyTorch)           │
│  Phase-aware     │  │  Input: 90-step     │  │                      │
│  normalization   │  │  HR + Steps seq     │  │  Input: 496 ROI      │
│  Lag features    │  │  Output: 32-dim     │  │  connectivity values │
│  Delta features  │  │  latent vector      │  │  Output: 16-dim      │
│  Temporal shift  │  │                     │  │  latent vector       │
│  (0 / 90 min)    │  │  3 window sizes:    │  │                      │
│                  │  │  15, 30, 45 min     │  │                      │
└──────────┬───────┘  └──────────┬──────────┘  └───────────┬──────────┘
           │                     │                          │
           └─────────────────────┴──────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   Smart Imputation     │
                    │   (Time-of-Day +       │
                    │   Activity Stratified) │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   Ablation Study       │
                    │   6 modality combos    │
                    │                        │
                    │  · EMA only            │
                    │  · Sensors only        │
                    │  · fMRI only           │
                    │  · Sensors + fMRI      │
                    │  · Full Multimodal     │
                    │  · Majority baseline   │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │  Random Forest         │
                    │  Leave-One-Group-Out   │
                    │  Cross-Validation      │
                    │  (by participant)      │
                    │                        │
                    │  + Gaussian noise aug  │
                    │  + RandomizedSearchCV  │
                    │  + Youden's J thresh   │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │  Outputs               │
                    │  · AUC-ROC, F1,        │
                    │    Recall, Precision   │
                    │  · ROC curves (PNG)    │
                    │  · Histograms (PNG)    │
                    └────────────────────────┘
```

---

## Dataset

> **Data is not included in this repository.** It contains Protected Health Information (PHI) governed by IRB protocol. Contact the PI for data access.

| Source File | Rows | Size | Contents |
|---|---|---|---|
| `Updated_CombineEMA-2.csv` | 3,178 | 363 KB | EMA surveys: craving, stress, mood, affect, timestamps |
| `Crave_Pilot_Fitbit.csv` | ~2.2M | 91 MB | Minute-by-minute heart rate and step count |
| `All_fMRI_connectivity_features.csv` | 29 | 7.8 MB | ROI-to-ROI functional connectivity (164 × 164 atlas) |
| `Crave_Demographics.csv` | 58 | 756 B | Age and gender per participant |
| `Crave_Surveys.csv` | 59 | 3 KB | Baseline scales: BIS, UPPS, GAD-7, QIDS, PSQI, UCLA, RAD |

**Cohort:** ~29 participants with full multimodal data (fMRI + Fitbit + EMA); ~58 total with partial data. Data collected across two phases with different scale ranges, requiring phase-aware normalization.

**Target variable:** `Craving_Binary` — self-reported craving present (1) or absent (0) at time of EMA survey.

---

## ML Methods

### Neural Autoencoders (Unsupervised Pre-training)

**Fitbit LSTM Autoencoder** — captures temporal physiological patterns:
- Architecture: LSTM encoder → 32-dim hidden state → LSTM decoder
- Input: 90-minute sliding window of heart rate + steps (minute resolution)
- Training: MSE reconstruction loss, 5 epochs, batch size 128
- Output: 32-dimensional latent embedding per survey window

**fMRI Feedforward Autoencoder** — compresses high-dimensional brain connectivity:
- Architecture: Linear(496 → 64 → 16) + Linear(16 → 64 → 496) with ReLU
- Input: 496 unique ROI-to-ROI connectivity values from a 164-region atlas
- Training: MSE reconstruction loss, 20 epochs, batch size 32
- Output: 16-dimensional latent embedding per participant (static)

### Classifier

**Random Forest** with hyperparameter tuning via `RandomizedSearchCV`:
- `n_estimators`: [50, 100, 200]
- `max_depth`: [2, 3, 4, 5]
- `min_samples_split`: [2, 5]
- Scoring: AUC-ROC
- Threshold: Youden's J statistic (maximizes sensitivity − (1 − specificity))

### Validation Strategy

**Leave-One-Group-Out (LOGO) cross-validation** grouped by participant ID — the most rigorous validation design for small-N clinical studies. Each fold trains on all other participants and tests on the held-out participant, preventing any data leakage across individuals.

### Class Imbalance Handling

Gaussian noise augmentation on the minority class (cravings) to balance training sets per fold. Combined with `class_weight='balanced'` in the Random Forest.

### Clustering (Exploratory)

Two standalone clustering analyses characterize participant subgroups:
- **`clustering_analysis.py`:** KMeans with silhouette-scored dynamic k selection on EMA + sensor features
- **`clustering_latent.py`:** KMeans on fused latent embeddings → derives emotional profiles and patient archetypes; tests whether biological signals predict psychological cluster membership

---

## Installation

### Prerequisites

- Python >= 3.11
- ~3 GB disk space for data (not included)
- 8 GB RAM minimum; 16 GB recommended for the full imputed dataset

### Steps

```bash
# 1. Clone the repository
git clone <repo-url>
cd CraveSense

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

**GPU support (optional):** Replace the `torch` line in `requirements.txt` before installing:

```bash
pip install torch==2.4.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

### Verify installation

```bash
python -c "import torch, pandas, sklearn, seaborn; print('All dependencies OK')"
```

---

## How to Run

### Step 1 — Place data files

Copy all raw CSV files into the project root (or `data/raw/`):

```
CraveSense/
├── Updated_CombineEMA-2.csv
├── Crave_Pilot_Fitbit.csv
├── All_fMRI_connectivity_features.csv
├── Crave_Demographics.csv
└── Crave_Surveys.csv
```

### Step 2 — Train the autoencoders (one-time)

```bash
python train_multimodal_encoders.py
```

Produces: `crave_fitbit_ae.pth`, `crave_fmri_ae.pth`, `fmri_col_order.csv`

This step is required only once. Subsequent runs of `main.py` load the saved weights.

### Step 3 — Run the full pipeline

```bash
python main.py
```

On the first run this merges all modalities and saves `Final_Master_Dataset_Imputed.csv` (~671 MB). Subsequent runs load it directly.

> **Note:** If you update any raw data file or fix a data bug, delete `Final_Master_Dataset_Imputed.csv` and re-run to regenerate.

### Optional — Clustering analysis

```bash
python clustering_analysis.py    # Silhouette-scored KMeans on EMA + sensors
python clustering_latent.py      # Latent-space patient archetypes
```

### Optional — Data diagnostics

```bash
python diagnostic_imputation.py  # Fitbit data density per survey window
```

---

## Expected Outputs

After a full run of `main.py`:

**Console:**

```
FINAL PIPELINE: BASELINES, TUNING & EMA+SENSORS
=================================================

EXPERIMENT 1: REAL-TIME DETECTION
Modality                              N_Rows  Accuracy  AUC_ROC  Recall  Precision    F1
EMA + Sensors + fMRI (Full Multimodal)  XXXX     0.XX      0.XX    0.XX     0.XX    0.XX
Sensors + fMRI (Physo-Neural)           XXXX     0.XX      0.XX    0.XX     0.XX    0.XX
...

EXPERIMENT 2: STANDARDIZED FORECASTING
...
```

**Files written:**

```
visualizations/
├── roc_detection.png                     # ROC curves for all 6 modality combos (detection)
├── roc_standardized_forecasting.png      # ROC curves (forecasting)
├── histogram_detection.png               # Participant data distribution (detection)
└── histogram_standardized_forecasting.png
```

---

## Results Summary

The ablation study quantifies how much each modality contributes independently and in combination. Key findings:

- **Full multimodal fusion (EMA + Sensors + fMRI)** consistently achieves the highest AUC-ROC in both detection and forecasting tasks
- **fMRI features alone** outperform sensor-only features on AUC, confirming that neural connectivity carries craving-relevant information not captured by heart rate or steps
- **90-minute forecasting** is a harder task than real-time detection, as expected — but the multimodal model retains meaningful predictive signal even at this horizon
- **Leave-One-Group-Out validation** produces conservative, realistic estimates that reflect performance on unseen participants — not just unseen survey windows

> Exact metric values depend on the full dataset. Run `main.py` to reproduce results on your data access.

---

## Limitations

- **Small N:** ~29 participants with complete multimodal data. Results should be treated as pilot findings.
- **fMRI is static:** Brain connectivity is measured once per participant (not per survey window). It captures trait-level neural features, not state-level changes.
- **Imputation:** ~30–40% of Fitbit windows are fully missing and imputed via time-of-day medians. Imputation quality is validated in `diagnostic_imputation.py`.
- **Participant ID mismatch:** `Crave_Surveys.csv` uses `CR0XX` IDs while other files use `CRS0XX` — the data loader normalizes this but the merge should be verified for every participant.
- **No temporal modeling of sequences:** The classifier sees feature vectors, not raw sequences. Future work should explore sequence-to-label models (TCN, Transformer).
- **Lack of external validation:** All results are from the pilot cohort. Generalizability to other populations is unknown.

---

## Future Improvements

- [ ] Fix random seed in `augment_with_gaussian_noise` for full reproducibility
- [ ] Replace row-by-row `apply()` in `get_fitbit_features()` with vectorized pandas operations (10–50x speedup)
- [ ] Add experiment tracking with MLflow or Weights & Biases
- [ ] Explore temporal sequence classifiers (TCN, LSTM-based) instead of static feature vectors
- [ ] Add unit tests for imputation and feature engineering logic
- [ ] Incorporate real-time fMRI surrogates (e.g., HRV-derived connectivity) to make fMRI features dynamic
- [ ] Extend to multi-class craving intensity prediction (not just binary)
- [ ] Containerize with Docker for reproducible compute environments

---

## Project Structure

```
CraveSense/
├── main.py                         ← Entry point
├── data_loader.py                  ← Feature engineering, imputation, autoencoder inference
├── preprocessing.py                ← Phase-aware normalization utilities
├── modeling_advanced.py            ← LOGO-CV ablation study (primary ML module)
├── modeling.py                     ← Baseline models (Logistic Regression, SVM, GBM)
├── train_multimodal_encoders.py    ← Autoencoder training (Fitbit LSTM + fMRI FFN)
├── train_autoencoder.py            ← First-generation Fitbit AE (reference only)
├── clustering_analysis.py          ← Silhouette KMeans on behavioral features
├── clustering_latent.py            ← Neural latent clustering + patient archetypes
├── diagnostic_imputation.py        ← Fitbit data density diagnostics
├── requirements.txt
├── LICENSE
├── CONTRIBUTING.md
├── data/                           ← Git-excluded; place raw CSVs here
│   ├── raw/
│   └── processed/
├── models/                         ← Git-excluded; place .pth weights here
├── docs/
│   └── repository_audit.md
└── visualizations/                 ← Git-excluded; generated plots written here
```

---

## Contributors

Contributors: Maahi Patel and Jinay Shah
---

## License

The **source code** is released under the [MIT License](LICENSE).

The **research data** is NOT included in this repository and is governed by a separate IRB protocol and Data Use Agreement. See `LICENSE` for the full data notice.

---

## Citation

If you use this codebase or methodology in your research, please cite:

```bibtex
@misc{cravesense2026,
  title   = {CraveSense: Multimodal Craving Prediction via EMA, Wearables, and fMRI},
  author  = {Patel, Maahi}, {Shah, Jinay}
  year    = {2026},
  url     = {https://github.com/maahipatel05/CraveSense}
}
```
