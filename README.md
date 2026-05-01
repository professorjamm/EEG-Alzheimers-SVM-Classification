# EEG Biomarkers for Alzheimer's Disease Using Frequency Analysis and Machine Learning

## Setup

### 1. Clone the repo

```
git clone https://github.com/professorjamm/EEG-Alzheimers-SVM-Classification.git
cd EEG-Alzheimers-SVM-Classification
```

### 2. Install Python dependencies

```
python3 -m pip install -r requirements.txt
```

This installs everything needed for:
- Task 2/Task 3 analysis (`mne`, `pandas`, `scipy`, `matplotlib`)
- ML pipeline work (`scikit-learn`, `seaborn`)

### 3. Download the EEG dataset (only needed to regenerate `data.csv` with Task 2)

```
git clone https://github.com/OpenNeuroDatasets/ds004504.git Dataset
cd Dataset
git annex init
git annex get derivatives/
cd ..
```

Note: You need [git-annex](https://git-annex.branchable.com/install/) installed for this step. The derivatives folder is ~2GB.

### 4. Run Task 2 (frequency analysis)

```
cd Tasks
python3 task2.py
```

processes all EEG files and outputs `data.csv`.

### 5. Run Task 3 (statistical comparison)

```
python3 task3.py
```

generates boxplots and prints t-test results.

### 6. Run the full ML pipeline

From the repo root:

```
python3 ML/main.py
```

`ML/main.py` reads channel-level data from:
- `Tasks/data.csv` (default)
- `data.csv` in repo root (fallback)

## What `ML/main.py` runs

- Feature subset experiments on subject-level features:
  - `Alpha`
  - `Alpha + Delta`
  - `Delta + Theta + Alpha + Beta`
- Subject vs channel representation experiments.
- Hyperparameter tuning across `linear`, `rbf`, and `poly` kernels using stratified 5-fold CV.
- Final held-out test evaluation with:
  - accuracy
  - confusion matrix
  - classification report
  - AD sensitivity (recall for class `A`)

## Output artifacts

- Results are written under `ML/results/`.
- Final summary visualizations are in `ML/results/final_visualizations/`.
- Per-run artifacts include:
  - confusion matrix heatmaps
  - precision/recall/F1 bar charts
  - classification report text files
  - kernel comparison charts
  - feature subset comparison CSV/chart
