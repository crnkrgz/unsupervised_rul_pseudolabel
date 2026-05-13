# Results Mapping: Thesis Tables and Figures to Source Scripts

This document maps every table and figure in the thesis to the script(s) that produce them. All scripts are located under `src/cmapss/` or `src/femto/`. Output files are written to `results/`.

## Chapter 3 — Methodology

| Item | Source | Output |
|------|--------|--------|
| Table 3.1 — C-MAPSS Subset Characteristics | Static (no script) | — |
| Table 3.2 — FEMTO Bearing Characteristics | Static (no script) | — |
| Table 3.3 — Autoencoder Architecture by Dataset | Static (no script) | — |
| Table 3.4 — Spark vs sklearn Wall-Clock | `src/cmapss/12_spark_scalability.py` | `results/tables/spark_benchmark.csv` |
| Table 3.5 — Theoretical Spark Projections | `src/cmapss/12_spark_scalability.py` | `results/tables/spark_projections.csv` |

## Chapter 4 — Findings

### Section 4.1 — Anomaly Score and RUL Correlation

| Item | Source | Output |
|------|--------|--------|
| Table 4.1 — Pearson correlation IF score vs RUL | `src/cmapss/03a_global_pipeline.py` | `results/predictions/global_default_FD00*.csv` |
| Figure 1 — IF anomaly score vs RUL scatter | `src/cmapss/figures/fig_correlation_scatter.py` | `results/figures/correlation_scatter.png` |

### Section 4.2 — Baseline RUL Prediction (Global Pipeline)

| Item | Source | Output |
|------|--------|--------|
| Table 4.2 — Global pipeline baseline RMSE | `src/cmapss/03a_global_pipeline.py` | `results/predictions/global_default_FD00*.csv` |

### Section 4.3 — Sensitivity Analysis

| Item | Source | Output |
|------|--------|--------|
| Table 4.3 — Optimal global config | `src/cmapss/04_sensitivity.py` | `results/tables/sensitivity_*.csv` |
| Table 4.4 — Welch t-test 10-seed | `src/cmapss/05_robustness_10seed.py` | `results/tables/robustness_10seed.csv` |
| Figure 2 — Hyperparameter sensitivity sweep | `src/cmapss/figures/fig_sensitivity_sweep.py` | `results/figures/sensitivity_sweep.png` |

### Section 4.4 — Cluster-Based Scaling

| Item | Source | Output |
|------|--------|--------|
| Table 4.5 — Global vs Cluster RMSE | `src/cmapss/03b_cluster_pipeline.py` | `results/predictions/cluster_default_FD00*.csv` |
| Figure 3 — Cluster impact bar chart | `src/cmapss/figures/fig_cluster_impact.py` | `results/figures/cluster_impact.png` |

### Section 4.5 — IF vs Autoencoder Comparison

| Item | Source | Output |
|------|--------|--------|
| Table 4.15 — AE hyperparameter sensitivity | `src/cmapss/13b_ae_sensitivity.py`, `src/cmapss/13d_ae_sensitivity_extension.py` | `results/tables/ae_sensitivity.csv` |
| Table 4.16 — Subset-specific optimal AE config | `src/cmapss/13b_ae_sensitivity.py` | `results/tables/ae_sensitivity.csv` |
| Table 4.17 — 5-seed robustness AE | `src/cmapss/13e_ae_robustness_optimal.py` | `results/tables/ae_robustness_optimal.csv` |
| Table 4.18 — IF vs AE optimal performance | `src/cmapss/14b_if_vs_ae_optimal.py` | `results/tables/if_vs_ae_optimal.csv` |

### Section 4.6 — SHAP Feature Attribution

| Item | Source | Output |
|------|--------|--------|
| Table 4.6 — SHAP top-10 IF pseudo (FD001) | `src/cmapss/06_shap_analysis.py` | `results/tables/table_4_6_shap_fd001_detail.csv` |
| Table 4.7 — SHAP cross-subset summary | `src/cmapss/06_shap_analysis.py` | `results/tables/table_4_7_shap_cross_subset.csv` |
| Table 4.19 — Three-way SHAP comparison | `src/cmapss/16_shap_analysis_ae.py` | `results/cmapss/shap_three_way_comparison.csv` |
| Table 4.20 — Cross-subset AE SHAP | `src/cmapss/16_shap_analysis_ae.py` | `results/cmapss/shap_ae_pseudo_summary.csv` |
| Figure 4 — SHAP Pseudo vs True (FD001) | `src/cmapss/06_shap_analysis.py` | `results/figures/shap_top10_pseudo_vs_true.png` |
| Figure 5 — SHAP beeswarm | `src/cmapss/06_shap_analysis.py` | `results/figures/shap_beeswarm_fd001_pseudo.png` |
| Figure 6 — Temporal SHAP evolution | `src/cmapss/06_shap_analysis.py` | `results/figures/shap_temporal_evolution_fd001.png` |

### Section 4.7 — Final Performance and Diagnostics

| Item | Source | Output |
|------|--------|--------|
| Table 4.8 — Final optimized RMSE | `src/cmapss/03b_cluster_pipeline.py` | `results/predictions/cluster_optimal_FD00*_full.csv` |
| Table 4.9 — Extended regression metrics | `src/cmapss/07_extended_metrics.py` | `results/tables/extended_metrics.csv` |
| Table 4.10 — Extended classification metrics | `src/cmapss/07_extended_metrics.py` | `results/tables/extended_metrics.csv` |
| Table 4.11 — Naive baseline comparison | `src/cmapss/09_naive_baselines.py` | `results/tables/naive_baselines.csv` |
| Table 4.12 — One-sample t-test bias | `src/cmapss/08_bias_calibration.py` | `results/tables/bias_ttest.csv` |
| Table 4.13 — Isotonic calibration | `src/cmapss/08_bias_calibration.py` | `results/tables/calibration.csv` |
| Table 4.14 — Near-failure binary classification | `src/cmapss/10_near_failure_classification.py` | `results/tables/table_5_1_near_failure_accuracy_per_subset.csv` |
| Figure 7 — Final RMSE pseudo vs true | `src/cmapss/figures/fig_final_rmse.py` | `results/figures/final_rmse.png` |

### Section 4.8 — FEMTO External Validation

| Item | Source | Output |
|------|--------|--------|
| Table 4.21 — FEMTO bearing characteristics | Static (cross-ref Table 3.2) | — |
| Table 4.22 — Four-track CC-RMSE | `src/femto/06c_compare_all_tracks.py` | `results/femto/all_tracks_comparison.csv` |
| Table 4.23 — Multi-detector ensemble (T5) | `src/femto/06d_compare_with_ensemble.py` | `results/femto/t2_vs_t5_ensemble.csv` |
| Table 4.24 — Pseudo-label onset freezing | `src/femto/04b_pipeline_adapted.py` | `results/femto/track2_adapted_predictions.csv` |
| Table 4.25 — Within-bearing correlations | `src/femto/05_evaluation.py` | `results/femto/track*_eval.csv` |
| Table 4.26 — C-MAPSS within-engine cluster correlations | `src/cmapss/15_within_engine_correlation.py` | `results/cmapss/within_engine_correlation_summary.csv` |
| Table 4.27 — Detector-architecture asymmetry | `src/femto/06c_compare_all_tracks.py` | `results/femto/all_tracks_comparison.csv` |

## Reproducibility Note

All scripts use `SEED = 42` fixed at the top of each file. Re-running any script on the same input data and library versions (see `requirements.txt`) reproduces the values reported in the thesis to within floating-point tolerance.

The two C-MAPSS pipeline scripts (`03a_global_pipeline.py` and `03b_cluster_pipeline.py`) embed canonical reference values as inline constants for self-validation — when run, they print a pass/fail check if the produced RMSE matches the thesis values within 1% tolerance.
