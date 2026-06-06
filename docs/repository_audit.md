# CraveSense — Repository Audit Report

**Date:** 2026-06-06  
**Auditor:** Claude Code (claude-sonnet-4-6)  
**Working Directory:** `/Users/maahi/Desktop/CraveSense`  
**Git Repository:** No

---

## Table of Contents

1. [Project Purpose](#1-project-purpose)
2. [Architecture](#2-architecture)
3. [ML & Data Pipeline Explanation](#3-ml--data-pipeline-explanation)
4. [Technologies & Dependencies](#4-technologies--dependencies)
5. [Frontend Structure](#5-frontend-structure)
6. [Backend Structure](#6-backend-structure)
7. [Database Structure](#7-database-structure)
8. [APIs](#8-apis)
9. [Environment Variables](#9-environment-variables)
10. [Build Process](#10-build-process)
11. [Deployment Process](#11-deployment-process)
12. [Missing Documentation](#12-missing-documentation)
13. [Missing Tests](#13-missing-tests)
14. [Security Concerns](#14-security-concerns)
15. [Issues Found](#15-issues-found)
16. [GitHub Readiness Score](#16-github-readiness-score)

---

## 1. Project Purpose

CraveSense is an academic machine learning research pipeline designed to **predict substance/food cravings** in real time and up to 90 minutes in advance, using three fused data modalities:

| Modality | Source | What It Captures |
|---|---|---|
| Psychological (EMA) | Ecological Momentary Assessment surveys | Stress, mood, trauma, negative affect, craving intensity |
| Physiological (Sensors) | Fitbit wearable | Heart rate (mean, std), step count across time windows |
| Neurological (fMRI) | Brain connectivity scans | Resting-state functional connectivity across 164 ROIs |

The project runs two experiments:

- **Experiment 1 — Real-Time Detection:** Predict whether a participant is currently craving using data from the same time window (current EMA + current Fitbit window).
- **Experiment 2 — Standardized Forecasting:** Predict current craving using only data from 90 minutes earlier, enabling advance warning.

The study involves approximately **29 participants** with fMRI data and **~58 participants** total, across two data collection phases.

---

## 2. Architecture

CraveSense is a **flat-file, scripts-only research codebase** with no web framework, no API server, and no database engine. All computation runs locally on a single machine.

```
CraveSense/
│
├── main.py                         ← Entry point: runs both experiments
│
├── data_loader.py                  ← Core data pipeline (merge, feature engineering, imputation)
├── preprocessing.py                ← Legacy/early preprocessing (orphaned module)
│
├── modeling_advanced.py            ← Primary ML module (LOGO-CV, ablation, ROC)
├── modeling.py                     ← Early baseline modeling (superseded)
│
├── train_multimodal_encoders.py    ← Production autoencoder training (Fitbit + fMRI)
├── train_autoencoder.py            ← First-generation Fitbit autoencoder (superseded)
│
├── clustering_analysis.py          ← Silhouette-scored KMeans on EMA + sensors
├── clustering_latent.py            ← Latent-space clustering + psychological profiling
│
├── diagnostic_imputation.py        ← Diagnostic: measures Fitbit data density per window
│
├── requirements.txt                ← Partial dependency list (has omissions)
│
├── [Raw Data CSVs]                 ← Source data (not version-controlled ideally)
├── [Intermediate CSVs]             ← Derived/merged datasets (generated artifacts)
├── [*.pth files]                   ← Saved PyTorch model weights
├── [*.png files]                   ← Generated visualizations
└── visualizations/                 ← Subdirectory for modeling_advanced ROC/histogram outputs
```

### Execution Flow

```
train_multimodal_encoders.py   (one-time)
        ↓ produces crave_fitbit_ae.pth, crave_fmri_ae.pth, fmri_col_order.csv
        
main.py
    └── data_loader.generate_master_dataset()
            ├── Load: Updated_CombineEMA-2.csv, Crave_Pilot_Fitbit.csv,
            │         All_fMRI_connectivity_features.csv, Crave_Demographics.csv, Crave_Surveys.csv
            ├── Feature engineering: Fitbit time windows (3 sizes × 2 shifts)
            ├── Neural latent features: LSTM (Fitbit → 32-dim), FFN (fMRI → 16-dim)
            ├── Static feature merge (demographics + surveys + fMRI)
            └── Smart imputation (time-of-day + activity stratified)
            ↓ produces Final_Master_Dataset_Imputed.csv
    └── modeling_advanced.run_analysis() × 2 (detection + forecasting)
            ├── Leave-One-Group-Out cross-validation (by participant)
            ├── Gaussian noise augmentation (minority class oversampling)
            ├── RandomizedSearchCV (RandomForestClassifier)
            ├── Youden's J threshold selection
            └── Ablation over 6 modality combinations
```

---

## 3. ML & Data Pipeline Explanation

### 3.1 Raw Data Sources

| File | Rows | Size | Description |
|---|---|---|---|
| `Updated_CombineEMA-2.csv` | 3,178 | 363 KB | EMA surveys with craving, stress, mood, affect labels |
| `Crave_Pilot_Fitbit.csv` | ~2.2M | 91 MB | Minute-by-minute HR + steps from Fitbit devices |
| `All_fMRI_connectivity_features.csv` | 29 | 7.8 MB | ROI-to-ROI functional connectivity matrix (164×164 ROIs = 496 unique pairs) |
| `Crave_Demographics.csv` | 58 | 756 B | Age, gender per participant |
| `Crave_Surveys.csv` | 59 | 3 KB | Baseline psych scales: CAMS, QIDS, GAD-7, BIS, UPPS, UCLA, PSQI, RAD, LOTR |
| `ROIROI_Matrix.json` | 29 samples | 11 KB | ROI metadata: 164 region names + coordinates |
| `fmri_col_order.csv` | 496 | 21 KB | Column name order for fMRI AE input (generated at training time) |

### 3.2 Target Variable

- `Craving_Binary` (0/1) → renamed to `Target_Now`
- Represents whether the participant self-reported a craving at the time of the EMA survey

### 3.3 Feature Engineering

**EMA-Derived (Psychological):**
- `Stress_Norm`, `Mood_Norm` — phase-aware normalization (Phase 1: 0–6/0–5 scale; Phase 2: 0–12)
- `Delta_Stress`, `Delta_Mood` — change from previous survey
- `Stress_Prev`, `Mood_Prev`, `Craving_Prev` — lag-1 features for forecasting
- `Delta_Stress_Prev`, `Delta_Mood_Prev` — lag-2 delta features
- `Hours_Since_Prev` — time gap between consecutive surveys
- `Hour` — hour of day (circadian feature)

**Fitbit-Derived (Physiological):**  
For each of 3 window sizes (15, 30, 45 min) × 2 shifts (0 = detection, 90 = forecasting):
- `HR_Mean_{w}`, `HR_Std_{w}` — heart rate statistics
- `Steps_{w}` — cumulative step count
- `Latent_Fitbit_0..31` — 32-dimensional LSTM latent vector (only for 45-min window, using 90-min lookback)

**fMRI-Derived (Neurological):**
- `Latent_fMRI_0..15` — 16-dimensional feedforward autoencoder embedding of 496 ROI-to-ROI connectivity values
- Raw NW-prefixed columns also available but compressed by AE

**Survey-Derived (Static Baseline):**  
One value per participant (does not vary over time):  
`BIS_nonplanning`, `BIS_motor`, `BIS_attention`, `UPPS_*` (5 facets), `GAD`, `QIDS`, `CAMS`, `PSQI`, `UCLA`, `RAD`, `LOTR`

### 3.4 Neural Autoencoders

| Model | Architecture | Input | Latent | File |
|---|---|---|---|---|
| Fitbit AE | LSTM Encoder + LSTM Decoder | 90-step × 2-feature sequence (HR, Steps) | 32-dim hidden state | `crave_fitbit_ae.pth` (23 KB) |
| fMRI AE | Linear(496→64→16) + Linear(16→64→496) | 496 ROI connectivity values | 16-dim embedding | `crave_fmri_ae.pth` (262 KB) |
| Legacy AE | LSTM Encoder + LSTM Decoder | 45-step × 2-feature sequence | 5-dim | `crave_autoencoder.pth` (4.5 KB) — unused |

Autoencoders are trained **unsupervised** (reconstruction loss, MSE) and their latent vectors are used as features in the downstream classifier.

### 3.5 Classification Pipeline

```
Modality Ablation Study (6 combinations):
  1. Baseline (Majority) — DummyClassifier
  2. EMA Only (Psychological)
  3. Sensors Only (Physical)
  4. fMRI Only (Neural)
  5. Sensors + fMRI (Physio-Neural)
  6. EMA + Sensors + fMRI (Full Multimodal)

Per fold (Leave-One-Group-Out by participant):
  1. StandardScaler fit on training participants
  2. Gaussian noise augmentation to balance minority class
  3. RandomizedSearchCV: n_estimators=[50,100,200], max_depth=[2..5], min_samples_split=[2,5]
  4. Youden's J statistic for threshold selection (maximizes TPR - FPR)
  5. Evaluate: Accuracy, AUC-ROC, Recall, Precision, F1
```

### 3.6 Smart Imputation Strategy

When Fitbit readings are missing for a survey window:
1. Impute using participant's median for that **time-of-day bucket** (Morning/Afternoon/Evening/Night)
2. Fall back to participant-level median
3. Fall back to population median
4. If steps > 50 (active) but HR missing → multiply imputed HR by 1.15 (active heart rate boost)

### 3.7 Clustering

Two separate clustering analyses are provided as standalone scripts:

- `clustering_analysis.py` — KMeans with dynamic k selection via silhouette score (k=2–6) on EMA and sensor features. Identifies behavioral craving contexts.
- `clustering_latent.py` — KMeans on latent neural embeddings (Fitbit + fMRI). Derives 2 emotional profiles and 3 patient archetypes. Tests whether biological signals predict psychological cluster membership.

---

## 4. Technologies & Dependencies

### 4.1 Listed in `requirements.txt`

```
pandas
numpy
matplotlib
seaborn           ← missing from requirements.txt (see issues)
scikit-learn
xgboost           ← listed but never used in any code file
imbalanced-learn  ← listed but SMOTE not used; Gaussian augmentation used instead
```

### 4.2 Used in Code but Missing from `requirements.txt`

| Package | Where Used | Current Installed Version |
|---|---|---|
| `torch` | `data_loader.py`, `train_*.py` | 2.4.1 |
| `seaborn` | `modeling.py`, `modeling_advanced.py`, `clustering_analysis.py`, `diagnostic_imputation.py` | Not pinned |

### 4.3 Full Resolved Environment (System Python 3.11.4)

| Package | Installed Version |
|---|---|
| Python | 3.11.4 |
| torch | 2.4.1 |
| pandas | 2.2.2 |
| numpy | 1.26.4 |
| scikit-learn | 1.5.1 |
| xgboost | 3.0.5 |
| imbalanced-learn | 0.12.3 |

> Note: `__pycache__` contains `.pyc` files compiled under both Python 3.12 and Python 3.13, indicating the codebase was previously run on newer Python versions. The system Python is 3.11.4.

---

## 5. Frontend Structure

**None.** CraveSense has no frontend. There is no web UI, dashboard, mobile app, or interactive interface of any kind. All output is:
- Console/terminal print statements
- Static PNG images saved to disk
- CSV files

---

## 6. Backend Structure

**None.** There is no API server, web server, REST endpoint, or service layer. The project is a standalone local analysis pipeline executed via `python main.py`.

All "backend" logic lives in the Python scripts described in the architecture section.

---

## 7. Database Structure

**None.** There is no database (SQL or NoSQL). Data persistence is handled entirely through flat CSV files.

| File | Role | Size |
|---|---|---|
| `Updated_CombineEMA-2.csv` | Primary survey data (raw) | 363 KB |
| `Crave_Pilot_Fitbit.csv` | Fitbit time-series (raw) | 91 MB |
| `All_fMRI_connectivity_features.csv` | fMRI features (raw) | 7.8 MB |
| `Crave_Demographics.csv` | Demographics (raw) | 756 B |
| `Crave_Surveys.csv` | Psych scales (raw) | 3 KB |
| `Final_Master_Dataset_Raw.csv` | Merged, unimputed | 486 MB |
| `Final_Master_Dataset.csv` | Merged (intermediate) | 780 MB |
| `Final_Master_Dataset_Imputed.csv` | Merged + imputed (primary ML input) | 671 MB |
| `Static_Features.csv` | Per-participant static features | 21 MB |
| `Time_Series.csv` | Time-series features | 572 KB |

**Total data footprint: ~2.06 GB**

---

## 8. APIs

**None.** No internal or external APIs are used or defined. No API keys, no network calls, no HTTP requests of any kind are made at runtime.

---

## 9. Environment Variables

**None defined.** The project uses no environment variables. All configuration is hardcoded:

| Hardcoded Value | Location | Example |
|---|---|---|
| Dataset filename | `main.py:11` | `'Final_Master_Dataset_Imputed.csv'` |
| Raw CSV filenames | `data_loader.py:189-193` | `'Updated_CombineEMA-2.csv'`, etc. |
| Model weight paths | `data_loader.py:38,47` | `'crave_fitbit_ae.pth'` |
| Sequence lengths | `data_loader.py:21` | `seq_len=90` |
| Embedding dimensions | various | `embedding_dim=32`, `embedding_dim=16` |
| Window sizes | `data_loader.py:9` | `WINDOW_SIZES = [15, 30, 45]` |
| Forecast shifts | `data_loader.py:10` | `SHIFTS = [0, 90]` |
| Output directory | `modeling_advanced.py:14` | `'visualizations'` |

All paths assume the working directory is the project root. Running from any other directory will fail silently or with `FileNotFoundError`.

---

## 10. Build Process

There is no build process. The pipeline is run directly:

```bash
# Step 1 (one-time): Train the autoencoders
python train_multimodal_encoders.py

# Step 2: Run the main detection + forecasting pipeline
python main.py

# Optional: Clustering analysis
python clustering_analysis.py
python clustering_latent.py

# Optional: Data diagnostics
python diagnostic_imputation.py
```

### Missing Setup Steps

The following steps are required but nowhere documented:

1. **Install Python** — version not specified. Tested internally on 3.12 and 3.13 (from pycache). 3.11 is the system default.
2. **Install dependencies** — `pip install -r requirements.txt` installs an incomplete set (torch and seaborn missing).
3. **Obtain raw data files** — all 5 raw CSV files must be present in the project root. No instructions on where they come from or how to request access.
4. **Run autoencoder training** — must be done before `main.py` or data loading will silently degrade (autoencoders fall back to `None`, and latent features are filled with `NaN`).
5. **Delete stale intermediate CSVs** — `main.py` itself warns (line 19) that `Final_Master_Dataset_Imputed.csv` must be deleted and regenerated after a bug fix. This is a manual step with no automation.

---

## 11. Deployment Process

**None.** This is a local research pipeline with no deployment story. There is no:
- Docker or container configuration
- Cloud compute setup (AWS, GCP, Azure)
- Job scheduler or HPC configuration
- Reproducibility tool (e.g., DVC, MLflow, Weights & Biases)
- Continuous integration (GitHub Actions, CircleCI)

---

## 12. Missing Documentation

| Missing Item | Severity | Impact |
|---|---|---|
| `README.md` | Critical | No one can understand what the project does, how to run it, or what data is needed |
| `.gitignore` | Critical | Sensitive clinical data and large binary files would be committed to version control |
| Data access instructions | Critical | Raw CSVs are the starting point; there is no guidance on obtaining them |
| `requirements.txt` completeness | High | `torch` and `seaborn` are uninstalled on a fresh environment, causing immediate import errors |
| Virtual environment setup | High | No `venv`, `conda`, or `pyproject.toml` instructions |
| Data dictionary | High | Column names in EMA CSV (Q1–Q4, PVN, HVL, etc.) are not explained anywhere |
| Module-level docstrings | Medium | No file describes its purpose, inputs, or outputs |
| Experiment results log | Medium | No record of what metrics were achieved; must re-run to see |
| Architecture diagram | Medium | The multi-step fusion pipeline is non-obvious without reading all code |
| Deprecation notices | Low | `preprocessing.py`, `modeling.py`, `train_autoencoder.py` are superseded but not labeled as such |
| Autoencoder training rationale | Low | No documentation on why embedding dims (32, 16, 5) were chosen |

---

## 13. Missing Tests

There are **zero test files** in the repository. No test framework is configured.

| Test Category | Status | Risk |
|---|---|---|
| Unit tests for `data_loader.py` | Missing | Feature engineering bugs (shift logic, window boundaries) go undetected |
| Unit tests for `preprocessing.py` | Missing | Normalization logic (phase-aware stress/mood) untested |
| Unit tests for `smart_impute()` | Missing | The activity-based HR inflation (`× 1.15`) has no guard against double-application |
| Unit tests for autoencoders | Missing | Shape and forward-pass correctness unverified |
| Integration test for full pipeline | Missing | A single bad merge can silently produce all-NaN feature columns |
| Data validation checks | Missing | No schema validation on raw CSVs (wrong column names would cause silent KeyErrors) |
| Reproducibility test | Missing | `np.random.normal` in `augment_with_gaussian_noise` has no seed; results are not reproducible |
| Regression test on metrics | Missing | No baseline AUC stored; a code change could silently degrade model performance |

---

## 14. Security Concerns

### 14.1 Sensitive Health Data in Plain Files — CRITICAL

The raw CSV files contain **Protected Health Information (PHI)**:

- `Crave_Demographics.csv` — participant age and gender, linkable to ID
- `Crave_Surveys.csv` — psychiatric scale scores (QIDS depression, GAD anxiety, PSQI sleep, UCLA loneliness, trauma scores)
- `Updated_CombineEMA-2.csv` — longitudinal self-report craving, mood, and stress data per participant
- `All_fMRI_connectivity_features.csv` — brain connectivity data, uniquely identifying

There is **no encryption**, **no access control**, and **no anonymization** beyond replacing names with ID codes (CRS001, CR001, etc.). If committed to a public GitHub repository, this data would be exposed permanently.

### 14.2 No `.gitignore` — HIGH

Without a `.gitignore`, the following would be committed if `git init` + `git add .` were run:

- All clinical CSV files (~2 GB)
- All model `.pth` weight files (trained on participant data)
- `.DS_Store` macOS metadata files
- `__pycache__/` compiled bytecode

### 14.3 Inconsistent Participant ID Schemes — MEDIUM

Participants are identified with two different ID formats across files:
- `CRS001`–`CRS029` format in `Crave_Demographics.csv` and `Updated_CombineEMA-2.csv`
- `CR001`–`CR029` format in `Crave_Surveys.csv` and `Crave_Pilot_Fitbit.csv`

The merge in `data_loader.py` relies on a `clean_ids()` function that normalizes to uppercase, but does not reconcile the `CRS`/`CR` prefix difference. Joins on `User_ID` across these files will silently fail to match unless both formats resolve to the same string. This is a **potential data integrity bug** that would cause survey/fitbit features to be all-NaN for some participants.

### 14.4 Module-Level Side Effects — MEDIUM

`data_loader.py` runs PyTorch model loading at **import time** (lines 36–53). This means:
- Any `import data_loader` triggers file I/O and model loading
- Failure produces a warning but no error, making it easy to run the pipeline believing autoencoders are loaded when they are not
- The scaler in `get_fitbit_features()` is fit on a per-call basis (line 101), not once globally — this is inefficient and could produce inconsistent scaling across rows

### 14.5 No Reproducibility Guarantee — LOW

- `np.random.normal` in `augment_with_gaussian_noise` (`modeling_advanced.py:35`) has no `random_state` argument, so results differ between runs even with the same data
- The `RandomizedSearchCV` does use `random_state=42`, partially mitigating this

---

## 15. Issues Found

### Code Issues

| Issue | File | Line | Severity |
|---|---|---|---|
| `torch` missing from `requirements.txt` | `requirements.txt` | — | High |
| `seaborn` missing from `requirements.txt` | `requirements.txt` | — | High |
| `xgboost` in requirements but never imported | `requirements.txt` | — | Low |
| `imbalanced-learn` in requirements but SMOTE never called | `requirements.txt` | — | Low |
| `np.random.normal` with no seed (non-reproducible) | `modeling_advanced.py` | 35 | High |
| Module-level model loading at import time | `data_loader.py` | 36–53 | Medium |
| StandardScaler re-fit per row inside `get_fitbit_features()` | `data_loader.py` | 101 | Medium |
| `preprocessing.py` is orphaned (never imported) | `preprocessing.py` | — | Medium |
| `modeling.py` is superseded but not marked deprecated | `modeling.py` | — | Low |
| `train_autoencoder.py` is superseded but not marked deprecated | `train_autoencoder.py` | — | Low |
| `warnings.filterwarnings("ignore")` suppresses all warnings | Multiple files | — | Low |
| All file paths are hardcoded relative to CWD | Multiple files | — | Medium |
| Warning message in `main.py` instructs user to manually delete CSV | `main.py` | 19 | Medium |
| `Crave_Surveys.csv` uses `CR0XX` IDs; `Crave_Demographics.csv` uses `CRS0XX` — merge may silently fail | `data_loader.py` | 244–248 | High |
| Emoji in print statements may cause encoding issues on some systems | `clustering_latent.py` | Multiple | Low |
| No logging framework; all output via bare `print()` | All files | — | Low |

### Architecture Issues

| Issue | Severity |
|---|---|
| No reproducibility infrastructure (MLflow, DVC, W&B) | Medium |
| Intermediate 671 MB CSV regenerated every time on a fresh run (slow) | Medium |
| `get_fitbit_features()` uses `apply()` row-by-row on 3,178 EMA rows × 6 window configs = ~19K individual Fitbit lookups — very slow | High |
| No pipeline caching or checkpointing between stages | Medium |
| Two generations of autoencoders exist (`train_autoencoder.py` vs `train_multimodal_encoders.py`) with different architectures | Low |

---

## 16. GitHub Readiness Score

**Score: 2 / 10**

| Criterion | Weight | Score | Notes |
|---|---|---|---|
| README present | 10% | 0/10 | No README whatsoever |
| `.gitignore` present | 10% | 0/10 | No .gitignore; clinical data would be committed |
| Sensitive data protected | 15% | 0/10 | Raw PHI CSVs sit unguarded in the project root |
| `requirements.txt` complete & pinned | 10% | 2/10 | Incomplete (missing torch, seaborn) and unpinned |
| Tests present | 10% | 0/10 | Zero test files |
| Documentation quality | 10% | 1/10 | No README, no docstrings, no data dictionary |
| Reproducibility | 10% | 3/10 | Random seed partially applied but AE scaler and augmentation are not seeded |
| Code quality | 10% | 5/10 | Readable research code; some orphaned modules and hardcoded paths |
| License | 5% | 0/10 | No LICENSE file |
| CI/CD | 5% | 0/10 | No GitHub Actions or equivalent |
| Large file handling | 5% | 0/10 | 2+ GB of CSVs with no Git LFS setup |
| Package structure | 5% | 1/10 | Flat script layout, no `setup.py` or `pyproject.toml` |

**Weighted Total: ~1.7 / 10 → rounded to 2/10**

---

## Recommended Next Steps (Priority Order)

### Immediate (Before Any GitHub Push)

1. **Create `.gitignore`** — exclude all `*.csv`, `*.pth`, `*.png`, `__pycache__/`, `.DS_Store`
2. **Confirm IRB/data sharing policy** — do not commit any raw participant data to any repository (public or private) without explicit IRB approval and data sharing agreement
3. **Store data externally** — use OSF, Zenodo, institutional data repository, or a private S3 bucket with a download script

### Short Term

4. **Write `README.md`** — project purpose, setup steps, how to run, data access request process
5. **Fix `requirements.txt`** — add `torch`, `seaborn`; pin all versions; consider `torch` CPU vs GPU variant
6. **Fix ID prefix mismatch** — verify `CRS` vs `CR` participant ID reconciliation in `data_loader.py`
7. **Add random seed to augmentation** — `modeling_advanced.py:35`

### Medium Term

8. **Add unit tests** — at minimum for `smart_impute()`, `get_fitbit_features()`, and normalization logic
9. **Remove or clearly mark deprecated files** — `preprocessing.py`, `modeling.py`, `train_autoencoder.py`
10. **Add MLflow or W&B experiment tracking** — replace print-based metric reporting with logged artifacts
11. **Pre-compute Fitbit windows** — replace the slow row-by-row `apply()` with vectorized pandas operations

### Long Term

12. **Add data dictionary** — document every column in every CSV
13. **Add a LICENSE** — required for academic reproducibility (MIT or Apache 2.0 for code; separate for data)
14. **Add GitHub Actions CI** — lint (flake8/ruff) + test (pytest) on push
15. **Consider Git LFS** — if intermediate CSVs are ever version-controlled
