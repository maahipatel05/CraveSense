# Contributing to CraveSense

Thank you for your interest in contributing. This is a clinical research
pipeline — contributions must meet a higher bar than typical software projects
because bugs can produce invalid science and because participant privacy must
be preserved at every step.

---

## Before You Start

1. **Do not commit data.** Raw participant files (`*.csv`, `*.pth` model
   weights, generated intermediate datasets) are excluded via `.gitignore`.
   If you accidentally stage a data file, remove it with
   `git rm --cached <file>` before opening a PR.

2. **IRB compliance.** Any change that affects how participant data is
   processed, merged, or exposed must be reviewed against the active IRB
   protocol. When in doubt, ask the PI before implementing.

3. **Reproducibility.** Every change that touches the ML pipeline must
   preserve or improve result reproducibility. All random operations must
   accept a `random_state` / `seed` argument.

---

## Repository Layout

```
CraveSense/
├── src/                    ← All pipeline source code (target location)
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── modeling.py
│   ├── modeling_advanced.py
│   ├── clustering_analysis.py
│   ├── clustering_latent.py
│   ├── train_autoencoder.py
│   ├── train_multimodal_encoders.py
│   └── diagnostic_imputation.py
├── data/                   ← Raw & intermediate data (git-excluded, local only)
│   ├── raw/                    ← Original files from data provider
│   └── processed/              ← Generated datasets (Final_Master_Dataset_*.csv)
├── models/                 ← Trained model weights (git-excluded, local only)
├── docs/                   ← Documentation and audit reports (committed)
├── visualizations/         ← Generated plots (git-excluded)
├── main.py                 ← Pipeline entry point
├── requirements.txt
├── .gitignore
├── LICENSE
└── CONTRIBUTING.md         ← This file
```

> **Note:** Source files currently live in the project root. The `src/`
> directory represents the intended final layout. Do not move files without
> coordinating with the team.

---

## Development Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd CraveSense
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

For GPU support, replace the `torch` line first:

```bash
pip install torch==2.4.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 4. Obtain data access

Raw data is NOT in this repository. To run the pipeline you need:

| File | Description | How to Obtain |
|---|---|---|
| `Updated_CombineEMA-2.csv` | EMA survey responses | Request from PI |
| `Crave_Pilot_Fitbit.csv` | Fitbit minute-by-minute HR + steps | Request from PI |
| `All_fMRI_connectivity_features.csv` | fMRI ROI connectivity | Request from PI |
| `Crave_Demographics.csv` | Age, gender per participant | Request from PI |
| `Crave_Surveys.csv` | Baseline psych scales | Request from PI |

Place all raw files in `data/raw/` (or the project root if using the
current layout).

### 5. Train autoencoders (one-time)

```bash
python train_multimodal_encoders.py
```

This produces `crave_fitbit_ae.pth`, `crave_fmri_ae.pth`, and
`fmri_col_order.csv`. Store weights in `models/`.

### 6. Run the pipeline

```bash
python main.py
```

---

## Making Changes

### Branch naming

```
feature/<short-description>
fix/<short-description>
docs/<short-description>
refactor/<short-description>
```

### Commit style

Use the imperative mood in the subject line (50 chars max):

```
Add random_state to augment_with_gaussian_noise
Fix participant ID mismatch between CRS and CR prefixes
Refactor get_fitbit_features to vectorized pandas ops
```

### What needs a PR vs. what you can push directly

| Change Type | Requires PR? |
|---|---|
| Source code changes | Yes |
| `requirements.txt` changes | Yes |
| Documentation updates | Recommended |
| `.gitignore` additions | Recommended |
| `LICENSE` changes | Yes — notify PI |

---

## Code Standards

- **Python >= 3.11**
- **Style:** PEP 8. Run `ruff check .` before committing (install with
  `pip install ruff`).
- **No hardcoded paths.** Use `pathlib.Path` or pass paths as arguments.
- **Random seeds.** Every stochastic operation must accept a `random_state`
  parameter defaulting to `42`.
- **No `warnings.filterwarnings("ignore")` in new code.** Suppress only
  the specific warning category you intend to silence.
- **No bare `except` clauses.** Catch the specific exception type.
- **Comments:** Explain *why*, not *what*. One line max per comment block.

---

## Pull Request Checklist

Before requesting review, confirm:

- [ ] No data files staged (`git status` shows no `*.csv`, `*.pth`, `*.png`)
- [ ] `requirements.txt` updated if new packages were added
- [ ] Random operations use a fixed seed
- [ ] No hardcoded file paths introduced
- [ ] `ruff check .` passes with no errors
- [ ] Manually tested with the full pipeline (`python main.py`)
- [ ] PR description explains *what changed* and *why*

---

## Reporting Issues

Open a GitHub Issue with:

1. A clear title describing the problem
2. Steps to reproduce
3. Expected vs. actual output
4. Python version and OS
5. **Do NOT include any participant data or model weights in issue reports.**

---

## Questions?

Open a GitHub Discussion or contact the maintainers directly.
