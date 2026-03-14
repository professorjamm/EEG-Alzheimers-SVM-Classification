# EEG Biomarkers for Alzheimer's Disease Using Frequency Analysis and Machine Learning

## Setup

### 1. Clone the repo

```
git clone https://github.com/professorjamm/EEG-Alzheimers-SVM-Classification.git
cd EEG-Alzheimers-SVM-Classification
```

### 2. Install Python dependencies

```
pip install mne pandas scipy matplotlib
```

### 3. Download the EEG dataset

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
python task2.py
```

processes all EEG files and outputs `data.csv`.

### 5. Run Task 3 (statistical comparison)

```
python task3.py
```

generates boxplots and prints t-test results.
