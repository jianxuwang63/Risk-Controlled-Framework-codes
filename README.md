# A Risk-Controlled Framework with Guaranteed Confidence for Lung Adenocarcinoma Classification

This repository contains the experimental implementation of **"A Risk-Controlled Framework with Guaranteed Confidence for Lung Adenocarcinoma Classification"** and a physician-supervised hospital pilot application for KDD 2027 ADS preparation.

> The new deployment application is a research pilot, not a standalone diagnosis or a cleared medical device. Clinical mode fails closed unless exact model weights and an independently certified deployment policy are present.

## Reviewer artifacts

### Full reviewer package: v0.10.15

The [v0.10.15 GitHub Release](https://github.com/jianxuwang63/Risk-Controlled-Framework-codes/releases/tag/v0.10.15)
contains the complete local reviewer application, the exact five frozen
Cost=5 checkpoints, the hash-bound Case 4 FNR-5% deployment policy, and
cross-platform setup and launch scripts. The package performs real five-model
inference and verifies every checkpoint before startup. It contains no
pathology image, hospital or pilot database, physician record, credential, or
access key. See
[START_HERE_FULL_REVIEWER.md](START_HERE_FULL_REVIEWER.md) for Windows, macOS,
and Linux instructions.

### Source repository: lightweight interface demo

Reviewers can inspect the complete HistoNexa-MIP interface locally without an
account, password, model checkpoint, hospital connection, or internet-facing
inference endpoint:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-app.txt
./start_reviewer_demo.sh
```

Then open <http://127.0.0.1:8000/>. macOS and Windows launchers are also
provided. See [REVIEWER_QUICKSTART.md](REVIEWER_QUICKSTART.md).

This lightweight demonstration uses deterministic placeholder scores and does
not reproduce the paper's model results. The five Cost=5 checkpoints are
provided only in the v0.10.15 Release package, not in the normal source
checkout. Hospital images, pilot databases, credentials, and retained physician
records are never included. Reviewers may use only pathology images that they
are authorized to process. The interface demo complements—but does not
replace—the controlled physician pilot and its post-launch measurements.

## Hospital Pilot Application

The `clinical_app/` package presents the five-model system under the display name **HistoNexa-MIP** and provides an English image-review web application for two clearly separated contexts: pathologist self-review and patient image information. A multi-image upload is only an operational session; every image receives an independent result and a printable, standardized interpretation report with microscopic findings, diagnostic impression, recommended diagnostic work-up, suggested reporting language, and explicit limitations. The application does not derive a patient diagnosis from the group. The pathologist workflow hides system output until every image has an independent assessment, then records objective image-level agreement, review time, and whether any image diagnosis changed. Patient-information sessions reveal per-image educational results immediately but are excluded from FNR and risk evaluation. Persistent previews, explicit inconclusive results, hash-bound models, audit events, CSV exports, local retention, and checksum-verified backups remain supported. The controlled pilot runs on a clinician-controlled host under the approved data-governance workflow; the public repository is intentionally not an open clinical inference service.

## Overview

Accurate detection of the micropapillary (MIP) subtype in lung adenocarcinoma is clinically urgent. This framework addresses two key challenges:

1. **Uncontrolled false-negative risk** — addressed using cost-sensitive learning and empirical Neyman-Pearson-style threshold calibration
2. **Unreliable prediction confidence** — addressed using selective classification with explicit deferral

The pipeline proceeds in three phases:

| Phase | Method | Purpose |
|-------|--------|---------|
| I | Cost-Sensitive Learning (CSL) | Penalize false negatives during training |
| II | Neyman-Pearson (NP) Calibration | Calibrate a classification threshold toward a target FNR; a finite-sample guarantee requires an order-statistic index satisfying the stated binomial-tail condition |
| III | Selective Classification | Abstain on uncertain cases, defer to pathologists |

## Repository Structure

```
.
├── utils.py                                    # Shared utilities (dataset, model, transforms, collate)
├── baseline_5fold.py                           # Phase 0: Baseline ViT-MIL (Table 1, Fig. 2)
├── cost_5fold.py                               # Phase I: Cost-sensitive learning (Table 2, Fig. 3)
├── new_np.py                                   # Phase II: NP threshold calibration (Table 3, Fig. 4)
├── cost+np.py                                  # Enhanced Phase II: CSL + NP calibration (Table 4, Figs. 5-6)
├── new_case1+2_5fold.py                        # Phase III Case 1 & 2: Selective classification + Risk/FNR control (Tables 5-6)
├── new_case3+4_cost5_5fold.py                  # Phase III Case 3 & 4: Risk/FNR control + CSL (w1=5) (Tables 7-8)
├── SOTA_validation_frombaseline_clam_sb_0.py   # SOTA comparison: CLAM-SB
├── SOTA_validation_frombaseline_ds_0.py        # SOTA comparison: DS-MIL
├── SOTA_validation_frombaseline_dtfd_0.py      # SOTA comparison: DTFD-MIL
├── SOTA_validation_frombaseline_transmil_0.py  # SOTA comparison: TransMIL
├── external_validation_baseline.py             # External validation: Baseline
├── external_validation_cost_np.py              # External validation: CSL + NP (Table 9)
├── external_validation_unified.py              # External validation: Selective classification (Table 10)
```

## Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA 12.0+ (recommended)
- NVIDIA GPU with ≥24 GB VRAM (all experiments run on RTX 5090 32 GB)

## Licensing and data boundary

The authors' original source code and documentation in this repository are
licensed under the [Apache License 2.0](LICENSE). That license does **not**
apply to third-party models, model checkpoints, pathology images, clinical
records, or other data.

Phikon is provided by Owkin under a separate non-commercial license. The
Phikon-derived checkpoints in the v0.10.15 Release are supplied solely for
academic reproducibility review by eligible non-profit users. They are not
covered by this repository's Apache-2.0 license and may be used only to the
extent permitted by the original Owkin license and any applicable
institutional agreement. See
[THIRD_PARTY_MODEL_NOTICE.md](THIRD_PARTY_MODEL_NOTICE.md).

No pathology image is included in the public repository. The software license
grants no right to use patient data, hospital images, institutional names, or
clinical records. This research software is not a medical device and is not
licensed for clinical-production use.

### Installation

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers albumentations opencv-python scikit-learn pandas matplotlib seaborn openpyxl tqdm
```

### Pretrained Weights

Download the Phikon backbone weights from the [official Phikon repository](https://huggingface.co/owkin/phikon) and place `phikon_base.pth` at a path referenced in the config of each script.

## Dataset Preparation

Organize your data as two folders:

```
data/
├── yes/    # MIP-positive images
└── no/     # MIP-negative images
```

Update `Config.pos_path` and `Config.neg_path` in each script to point to your data directories.

## Experiment Reproduction Guide

Each script is self-contained. Modify the path variables in the `Config` class, then run:

### 1. Baseline (Section IV-B, Table 1)

```bash
python baseline_5fold.py
```

Runs stratified 5-fold CV with ViT-MIL. Outputs per-fold metrics, mean ROC curve, and aggregate confusion matrix.

### 2. SOTA Comparison (Section IV-C, Table 2)

```bash
python SOTA_validation_frombaseline_clam_sb_0.py
python SOTA_validation_frombaseline_ds_0.py
python SOTA_validation_frombaseline_dtfd_0.py
python SOTA_validation_frombaseline_transmil_0.py
```

Compares CLAM-SB, DS-MIL, DTFD-MIL, and TransMIL aggregators under identical Phikon encodings.

### 3. Cost-Sensitive Learning (Section IV-D, Table 3)

```bash
python cost_5fold.py
```

Runs 5-fold CV with the deployed cost weight `[5]`. Outputs confusion matrices, density plots, ROC curves, and trade-off bar charts.

### 4. NP Threshold Calibration (Section IV-E, Table 4)

```bash
python new_np.py
```


Runs 10 Monte-Carlo CV repetitions on the baseline model. Calibrates thresholds at FNR targets of 1%, 5%, and 10%.

### 5. CSL + NP Calibration (Section IV-F, Table 5)

```bash
python cost+np.py
```

Combines cost-sensitive training (w1=5) with NP threshold calibration across 10 MCCV runs. Generates averaged distribution plots.

### 6. Selective Classification (Section IV-G)

**Case 1 (Risk Control) & Case 2 (FNR Control):**
```bash
python new_case1+2_5fold.py
```

**Case 3 (Risk Control + CSL) & Case 4 (FNR Control + CSL):**
```bash
python new_case3+4_cost5_5fold.py
```

Each script covers both risk control and FNR control modes via configuration switch. All cases run 5-fold CV with swap-calibration (producing 10 independent evaluations). Outputs coverage, risk, FNR, and AUC at target levels `[1%, 3%, 5%, 10%]`.

### 7. External Validation (Section IV-H, Tables 9-10)

Uses the [TRACERx 100 cohort](https://zenodo.org/records/10016027). Organize the external dataset into `yes/` and `no/` folders (MIP vs. other subtypes).

```bash
python external_validation_baseline.py
python external_validation_cost_np.py
python external_validation_unified.py
```

## Results Summary

| Method | ACC | FNR | Key Feature |
|--------|-----|-----|-------------|
| Baseline ViT-MIL | 93.41% | 16.06% | Strong overall, unsafe FNR |
| CSL (w1=5) | 91.24% | 6.25% | Reduced FNR with minimal ACC loss |
| NP Calibration (5% target) | 82.72% | 4.47% | Empirical FNR calibration; a formal guarantee depends on a valid NP index |
| Selective (Case 1, α=5%) | 94.82%* | 12.91%* | 92.51% coverage, risk-controlled |

*Accuracy and AUC are computed on the accepted subset. Case 4 system FNR uses accepted false negatives divided by all positive cases, matching Eq. (2), rather than the accepted-positive denominator.
