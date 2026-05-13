"""
03b_cluster_pipeline.py
Cluster-based scaling pipeline (Section 4.4 of thesis):
  - KMeans(k=6) fitted on op_setting columns of training data (multi-condition).
  - Per-cluster Min-Max scaling: one scaler per cluster, fitted on training data.
  - Isolation Forest + pseudo-label generation + Ridge regression identical to 03a.
  - For single-condition subsets (FD001, FD003), k=1 collapses to global pipeline.
  - Comparison table: global vs cluster RMSE quantifies the scaling improvement (RQ3).
Generates: results/tables/cluster_pipeline_default_config.csv
           results/tables/global_vs_cluster_comparison.csv
           results/predictions/cluster_default_FD00X.csv (one per subset)
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from cmapss.config import (
    KMEANS_K_MULTI,
    MULTI_CONDITION,
    RESULTS_DIR,
    SEED,
    SINGLE_CONDITION,
    SUBSETS,
    THRESHOLD_MULTI,
    THRESHOLD_SINGLE,
)

_SRC = Path(__file__).parent


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, _SRC / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Clustering helpers ────────────────────────────────────────────────────────

def cluster_op_settings(
    df_train: pd.DataFrame, df_test: pd.DataFrame, k: int, seed: int = None
) -> tuple:
    op_cols = [c for c in df_train.columns if c.startswith("op_setting_")]
    km = KMeans(n_clusters=k, random_state=(seed if seed is not None else SEED), n_init=10)
    km.fit(df_train[op_cols])
    train_out = df_train.copy()
    test_out  = df_test.copy()
    train_out["cluster"] = km.labels_.astype(int)
    test_out["cluster"]  = km.predict(df_test[op_cols]).astype(int)
    return train_out, test_out, km


def per_cluster_minmax(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: list,
    k: int,
) -> tuple:
    train_out = df_train.copy()
    test_out  = df_test.copy()
    # Ensure float dtype so scaler output can be assigned without pandas warnings
    train_out[feature_cols] = train_out[feature_cols].astype(float)
    test_out[feature_cols]  = test_out[feature_cols].astype(float)
    scalers = {}
    for c in range(k):
        tr_mask = train_out["cluster"] == c
        te_mask = test_out["cluster"] == c
        sc = MinMaxScaler()
        train_out.loc[tr_mask, feature_cols] = sc.fit_transform(
            train_out.loc[tr_mask, feature_cols]
        )
        if te_mask.any():
            test_out.loc[te_mask, feature_cols] = sc.transform(
                test_out.loc[te_mask, feature_cols]
            )
        scalers[c] = sc
    return train_out, test_out, scalers


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_cluster_pipeline(
    subset_name: str,
    n_estimators: int = None,
    threshold: float = None,
    seed: int = None,
) -> dict:
    name = subset_name.upper()
    if threshold is None:
        threshold = THRESHOLD_SINGLE if name in SINGLE_CONDITION else THRESHOLD_MULTI
    k = 1 if name in SINGLE_CONDITION else KMEANS_K_MULTI

    load_mod = _load_module("load_data",  "01_load_data.py")
    prep_mod = _load_module("preprocess", "02_preprocess.py")
    pipe_mod = _load_module("global_pipe","03a_global_pipeline.py")

    raw    = load_mod.load_subset(name)
    result = prep_mod.preprocess_subset(raw, name, scale=False)

    train_df     = result["train"]
    test_df      = result["test"]
    feature_cols = result["feature_cols"]

    # Cluster on op_settings, then per-cluster scale
    train_df, test_df, _km     = cluster_op_settings(train_df, test_df, k, seed=seed)
    train_df, test_df, scalers = per_cluster_minmax(train_df, test_df, feature_cols, k)

    cluster_sizes = {
        int(c): int((train_df["cluster"] == c).sum()) for c in range(k)
    }

    X_train = train_df[feature_cols].values
    X_test  = test_df[feature_cols].values

    # IF + smoothing (training set)
    clf          = pipe_mod.fit_isolation_forest(X_train, n_estimators, seed=seed)
    raw_train    = pipe_mod.score_anomaly(clf, X_train)
    smooth_train = pipe_mod.smooth_scores_per_unit(train_df, raw_train)

    pearson_r  = pipe_mod.compute_correlation(smooth_train, train_df["true_RUL"].values)
    pseudo_rul = pipe_mod.generate_pseudo_labels(smooth_train, threshold)

    X_train_r    = np.column_stack([X_train, smooth_train])
    pseudo_model = pipe_mod.train_ridge(X_train_r, pseudo_rul)
    true_model   = pipe_mod.train_ridge(X_train_r, train_df["true_RUL"].values)

    raw_te      = pipe_mod.score_anomaly(clf, X_test)
    smooth_te   = pipe_mod.smooth_scores_per_unit(test_df, raw_te)
    X_te_full_r = np.column_stack([X_test, smooth_te])
    last_idx    = test_df.groupby("unit_id")["cycle"].idxmax()
    is_last     = test_df.index.isin(last_idx)
    X_test_r    = X_te_full_r[is_last]
    last_df     = test_df.loc[last_idx]

    y_pred_pseudo = np.clip(pseudo_model.predict(X_test_r), 0, 125)
    y_pred_true   = np.clip(true_model.predict(X_test_r),   0, 125)

    unit_ids_sorted = sorted(test_df["unit_id"].unique())
    rul_map         = dict(zip(unit_ids_sorted, raw["rul"].values))
    y_true_eng      = last_df["unit_id"].map(rul_map).values.astype(float)

    return {
        "subset":              name,
        "n_estimators":        clf.n_estimators,
        "threshold":           threshold,
        "k_clusters":          k,
        "cluster_sizes_train": cluster_sizes,
        "n_features_total":    len(feature_cols) + 1,
        "pearson_r":           pearson_r,
        "pseudo_metrics":      pipe_mod.compute_metrics(y_true_eng, y_pred_pseudo),
        "true_metrics":        pipe_mod.compute_metrics(y_true_eng, y_pred_true),
        "y_true_per_engine":   y_true_eng,
        "y_pred_pseudo":       y_pred_pseudo,
        "y_pred_true":         y_pred_true,
    }


# ── __main__ ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Canonical reference values (previously loaded from verify_baseline.py)
    CANONICAL = {
        "FD001": {"r": -0.5931, "pseudo_rmse": 32.05, "true_rmse": 21.15, "gap": 10.90},
        "FD002": {"r": -0.6115, "pseudo_rmse": 42.40, "true_rmse": 31.89, "gap": 10.51},
        "FD003": {"r": -0.6904, "pseudo_rmse": 28.87, "true_rmse": 22.50, "gap":  6.37},
        "FD004": {"r": -0.6978, "pseudo_rmse": 42.03, "true_rmse": 33.69, "gap":  8.34},
    }
    TOL_PCT = 1.0  # canonical reference tolerance (%)

    results = {}
    for subset in SUBSETS:
        print(f"Running {subset}...")
        results[subset] = run_cluster_pipeline(subset)

    # Summary table
    W = 97
    print("\n" + "=" * W)
    print("CLUSTER PIPELINE - Default Config (n=200, thr=0.20/0.15)")
    hdr = (
        f"{'Subset':<8} {'k':>2} {'Pearson_r':>10} {'Pseudo_RMSE':>12} {'Pseudo_MAE':>11} "
        f"{'Pseudo_NASA':>12} {'True_RMSE':>10} {'True_MAE':>9} {'True_NASA':>10}"
    )
    print(hdr)
    print("-" * W)
    rows = []
    for subset, r in results.items():
        pm, tm = r["pseudo_metrics"], r["true_metrics"]
        print(
            f"{subset:<8} {r['k_clusters']:>2} {r['pearson_r']:>10.4f} "
            f"{pm['rmse']:>12.2f} {pm['mae']:>11.2f} {pm['nasa_score']:>12.0f} "
            f"{tm['rmse']:>10.2f} {tm['mae']:>9.2f} {tm['nasa_score']:>10.0f}"
        )
        rows.append({
            "subset":       subset,
            "k":            r["k_clusters"],
            "n_estimators": r["n_estimators"],
            "threshold":    r["threshold"],
            "pearson_r":    round(r["pearson_r"], 4),
            "pseudo_rmse":  round(pm["rmse"], 2),
            "pseudo_mae":   round(pm["mae"], 2),
            "pseudo_nasa":  int(round(pm["nasa_score"])),
            "true_rmse":    round(tm["rmse"], 2),
            "true_mae":     round(tm["mae"], 2),
            "true_nasa":    int(round(tm["nasa_score"])),
        })

    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    cluster_df = pd.DataFrame(rows)
    cluster_df.to_csv(tables_dir / "cluster_pipeline_default_config.csv", index=False)
    print(f"\nSaved: results/tables/cluster_pipeline_default_config.csv")

    preds_dir = RESULTS_DIR / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)
    for subset, r in results.items():
        pd.DataFrame({
            "y_true":        r["y_true_per_engine"],
            "y_pred_pseudo": r["y_pred_pseudo"],
            "y_pred_true":   r["y_pred_true"],
        }).to_csv(preds_dir / f"cluster_default_{subset}.csv", index=False)
    print("Saved: results/predictions/cluster_default_FD00X.csv (x4)")

    # Sanity check
    print("\n-- Sanity Check vs canonical cluster_pipeline (tolerance +/-1%) " + "-" * 20)
    print(f"{'Subset':<8} {'Metric':<18} {'Canonical':>10} {'Got':>10} {'Err%':>7}  Status")
    print("-" * 60)
    all_pass = True
    for subset, r in results.items():
        canon = CANONICAL[subset]
        pm, tm = r["pseudo_metrics"], r["true_metrics"]
        checks = [
            ("pearson_r",   canon["r"],           r["pearson_r"]),
            ("pseudo_rmse", canon["pseudo_rmse"],  pm["rmse"]),
            ("true_rmse",   canon["true_rmse"],    tm["rmse"]),
        ]
        for metric, expected, got in checks:
            pct = abs(got - expected) / abs(expected) * 100 if abs(expected) > 1e-9 else 0.0
            status = "PASS" if pct <= TOL_PCT else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(
                f"{subset:<8} {metric:<18} {expected:>10.4f} {got:>10.4f} "
                f"{pct:>6.2f}%  {status}"
            )

    print()
    print("Overall:", "ALL PASS" if all_pass else "SOME CHECKS FAILED - see above")

    # Global vs cluster comparison table
    global_csv = tables_dir / "global_pipeline_default_config.csv"
    if global_csv.exists():
        global_df = pd.read_csv(global_csv).set_index("subset")
        comp_rows = []
        for subset, r in results.items():
            pm, tm = r["pseudo_metrics"], r["true_metrics"]
            g_pseudo = global_df.loc[subset, "pseudo_rmse"]
            g_true   = global_df.loc[subset, "true_rmse"]
            c_pseudo = round(pm["rmse"], 2)
            c_true   = round(tm["rmse"], 2)
            dp = round(c_pseudo - g_pseudo, 2)
            dt = round(c_true   - g_true,   2)
            pp = round(dp / g_pseudo * 100, 1) if g_pseudo else 0.0
            pt = round(dt / g_true   * 100, 1) if g_true   else 0.0
            comp_rows.append({
                "subset":               subset,
                "k":                    r["k_clusters"],
                "pseudo_rmse_global":   g_pseudo,
                "pseudo_rmse_cluster":  c_pseudo,
                "pseudo_rmse_delta":    dp,
                "pseudo_rmse_pct":      pp,
                "true_rmse_global":     g_true,
                "true_rmse_cluster":    c_true,
                "true_rmse_delta":      dt,
                "true_rmse_pct":        pt,
            })
        comp_df = pd.DataFrame(comp_rows)
        comp_df.to_csv(tables_dir / "global_vs_cluster_comparison.csv", index=False)
        print("\nGlobal vs Cluster comparison:")
        print(
            f"{'Subset':<8} {'k':>2} {'Pseudo_G':>10} {'Pseudo_C':>10} {'dPseudo':>8} "
            f"{'%':>6} {'True_G':>8} {'True_C':>8} {'dTrue':>7} {'%':>6}"
        )
        print("-" * 80)
        for row in comp_rows:
            print(
                f"{row['subset']:<8} {row['k']:>2} {row['pseudo_rmse_global']:>10.2f} "
                f"{row['pseudo_rmse_cluster']:>10.2f} {row['pseudo_rmse_delta']:>8.2f} "
                f"{row['pseudo_rmse_pct']:>5.1f}% {row['true_rmse_global']:>8.2f} "
                f"{row['true_rmse_cluster']:>8.2f} {row['true_rmse_delta']:>7.2f} "
                f"{row['true_rmse_pct']:>5.1f}%"
            )
        print("\nSaved: results/tables/global_vs_cluster_comparison.csv")
    else:
        print("\nWARNING: global_pipeline_default_config.csv not found - run 03a first")

