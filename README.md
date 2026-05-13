# Unsupervised RUL Prediction with Pseudo-Label Pipeline

Code for the M.Sc. thesis: *"Unsupervised Remaining Useful Life Prediction with Pseudo-Label Pipeline: A Cross-Detector and Cross-Dataset Investigation."*

## Overview

The pipeline combines anomaly detection (Isolation Forest or Autoencoder), threshold-based pseudo-label generation, and Ridge regression for unsupervised RUL prediction. The empirical investigation covers two datasets (NASA C-MAPSS turbofan engines, FEMTO PRONOSTIA bearings) and two detector families (IF, AE).

## Setup

Python 3.12.10

```bash
pip install -r requirements.txt
```

Dependencies: `scikit-learn`, `shap`, `pandas`, `numpy`, `scipy`, `matplotlib`, `pyspark`, `torch`.

## Project Structure

```
thesis_codes/
├── data/raw/                        # raw dataset files (place manually — not tracked)
├── results/
│   ├── cmapss/                      # analysis outputs (scripts 15-16)
│   │   └── figures/
│   ├── femto/                       # FEMTO track predictions and eval CSVs
│   ├── figures/                     # C-MAPSS pipeline figures
│   ├── predictions/                 # C-MAPSS per-engine predictions
│   └── tables/                      # C-MAPSS summary tables
├── src/
│   ├── _common/                     # shared utilities (reserved)
│   ├── cmapss/                      # C-MAPSS pipeline
│   │   ├── config.py
│   │   ├── 01_load_data.py
│   │   ├── 02_preprocess.py
│   │   ├── 03a_global_pipeline.py   # IF global (Table 4.2)
│   │   ├── 03b_cluster_pipeline.py  # IF cluster-based (Table 4.5 / Table 4.8)
│   │   ├── 04_sensitivity.py
│   │   ├── 05_robustness_10seed.py
│   │   ├── 06_shap_analysis.py      # SHAP for IF (Tables 4.6-4.7)
│   │   ├── 07_extended_metrics.py
│   │   ├── 08_bias_calibration.py
│   │   ├── 09_naive_baselines.py
│   │   ├── 10_near_failure_classification.py
│   │   ├── 12_spark_scalability.py
│   │   ├── 13_autoencoder_pipeline.py    # AE pseudo-label (Section 4.5)
│   │   ├── 13b_ae_sensitivity.py
│   │   ├── 13c_ae_robustness.py
│   │   ├── 13d_ae_sensitivity_extension.py
│   │   ├── 13e_ae_robustness_optimal.py
│   │   ├── 14_if_vs_ae_comparison.py
│   │   ├── 14b_if_vs_ae_optimal.py
│   │   ├── 15_within_engine_correlation.py  # within-engine Pearson r analysis
│   │   ├── 16_shap_analysis_ae.py           # SHAP for AE + 3-way comparison
│   │   └── figures/                          # figure generation scripts
│   └── femto/                       # FEMTO PRONOSTIA pipeline
│       ├── config.py
│       ├── 01_load_data.py
│       ├── 02_feature_extraction.py
│       ├── 02b_feature_extraction_enhanced.py
│       ├── 03_build_dataset.py
│       ├── 04_pipeline_strict.py      # Track 1: strict C-MAPSS transfer
│       ├── 04b_pipeline_adapted.py    # Track 2: RobustScaler + transductive
│       ├── 04c_pipeline_enhanced.py   # Track 3: 39 enhanced features
│       ├── 04d_pipeline_combined.py   # Track 4: T2 scaling + T3 features
│       ├── 04e_pipeline_ensemble.py   # Track 5: IF + AE naive ensemble
│       ├── 05_evaluation.py
│       ├── 06_compare_t1_t2.py
│       ├── 06b_compare_t1_t2_t3.py
│       ├── 06c_compare_all_tracks.py
│       └── 06d_compare_with_ensemble.py
└── requirements.txt
```

## Data

### C-MAPSS (NASA Turbofan Engine Degradation)

Place files manually into `data/raw/`:

```
data/raw/
├── train_FD001.txt   test_FD001.txt   RUL_FD001.txt
├── train_FD002.txt   test_FD002.txt   RUL_FD002.txt
├── train_FD003.txt   test_FD003.txt   RUL_FD003.txt
└── train_FD004.txt   test_FD004.txt   RUL_FD004.txt
```

### FEMTO PRONOSTIA (Bearing Degradation)

Place bearing folders under `data/femto/`:

```
data/femto/
├── Training_set/   (6 bearings: Bearing1_1 .. Bearing3_2)
└── Test_set/       (11 bearings: Bearing1_3 .. Bearing3_3)
```

## C-MAPSS Execution Order

```
01 -> 02 -> 03a -> 03b -> 04 -> 05 -> 06 -> 07 -> 08 -> 09 -> 12
                   |
                   +-> 13 -> 13b -> 13c -> 13d -> 13e
                         -> 14 -> 14b
                         -> 15 (within-engine correlation)
                         -> 16 (AE SHAP + 3-way comparison)
```

| Script | Purpose |
|--------|---------|
| `01_load_data.py` | Load raw C-MAPSS files |
| `02_preprocess.py` | RUL computation, sensor filtering, MinMaxScaling |
| `03a_global_pipeline.py` | IF global pipeline (Table 4.2) |
| `03b_cluster_pipeline.py` | IF cluster-based pipeline (Table 4.5 / Table 4.8) |
| `04_sensitivity.py` | n_estimators x threshold sweep |
| `05_robustness_10seed.py` | 10-seed statistical robustness |
| `06_shap_analysis.py` | SHAP attribution for IF (Tables 4.6-4.7) |
| `07_extended_metrics.py` | Extended classification + regression metrics |
| `08_bias_calibration.py` | Bias t-tests and isotonic calibration |
| `09_naive_baselines.py` | Naive baseline comparison |
| `12_spark_scalability.py` | sklearn vs PySpark benchmark |
| `13_autoencoder_pipeline.py` | AE pseudo-label pipeline (Section 4.5) |
| `13b/c/d/e` | AE sensitivity and robustness sweeps |
| `14_if_vs_ae_comparison.py` | IF vs AE summary comparison |
| `14b_if_vs_ae_optimal.py` | IF vs AE with per-subset optimal AE config |
| `15_within_engine_correlation.py` | Within-engine Pearson r by condition |
| `16_shap_analysis_ae.py` | AE SHAP + IF/AE/supervised 3-way comparison |

## FEMTO Execution Order

```
01 -> 02 -> 02b -> 03 -> 04 (Track 1)
                      -> 04b (Track 2) -> 06_compare_t1_t2
                      -> 04c (Track 3) -> 06b_compare_t1_t2_t3
                      -> 04d (Track 4) -> 06c_compare_all_tracks
                      -> 04e (Track 5) -> 06d_compare_with_ensemble
```

| Track | Script | Config |
|-------|--------|--------|
| T1 | `04_pipeline_strict.py` | MinMaxScaler, alpha=1, 18 features |
| T2 | `04b_pipeline_adapted.py` | RobustScaler transductive, alpha=100, 18 features |
| T3 | `04c_pipeline_enhanced.py` | MinMaxScaler, alpha=1, 39 features |
| T4 | `04d_pipeline_combined.py` | RobustScaler transductive, alpha=100, 39 features |
| T5 | `04e_pipeline_ensemble.py` | IF + AE naive ensemble (T2 config) |

## Canonical Reference Values

**C-MAPSS cluster pipeline (Table 4.8):**

| Subset | Pearson r | Pseudo RMSE | True RMSE |
|--------|-----------|-------------|-----------|
| FD001  | -0.5931   | 32.05       | 21.15     |
| FD002  | -0.6115   | 42.40       | 31.89     |
| FD003  | -0.6904   | 28.87       | 22.50     |
| FD004  | -0.6978   | 42.03       | 33.69     |

**FEMTO best oracle CC-RMSE (Track 2 — IF, 18 features):** 735.20

## Global Config

All pipeline constants are in `src/cmapss/config.py` (C-MAPSS) and `src/femto/config.py` (FEMTO):

- `SEED = 42` — all stochastic operations
- `RUL_CEILING = 125` — C-MAPSS RUL clip
- `IF_CONTAMINATION = 0.05`, `IF_N_ESTIMATORS_DEFAULT = 200`
- `KMEANS_K_MULTI = 6` — operating condition clusters (FD002, FD004)
