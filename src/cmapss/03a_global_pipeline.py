"""
03a_global_pipeline.py
Global pipeline (Section 4.3 of thesis):
  - Single Min-Max scaler fitted on all training data.
  - Isolation Forest (n_estimators=200, contamination=0.05, seed=42).
  - 5-cycle rolling mean on anomaly scores per unit.
  - Threshold-based pseudo-label generation: top threshold_frac → anomaly zone
    [0, 62], remainder → healthy zone [63, 125].
  - Parallel Ridge regression (alpha=10): pseudo-label model + true-label reference.
  - Evaluation on last-cycle test predictions: RMSE, MAE, NASA score, Pearson r.
Generates: results/tables/global_pipeline_default_config.csv
           results/predictions/global_default_FD00X.csv (one per subset)
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).parent.parent))
from cmapss.config import (
    IF_CONTAMINATION,
    IF_N_ESTIMATORS_DEFAULT,
    RESULTS_DIR,
    RIDGE_ALPHA,
    ROLLING_WINDOW,
    SEED,
    SINGLE_CONDITION,
    SUBSETS,
    THRESHOLD_MULTI,
    THRESHOLD_SINGLE,
    ZONE_DEGRADED,
    ZONE_HEALTHY,
)

_SRC = Path(__file__).parent


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, _SRC / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Stage 1: Isolation Forest ─────────────────────────────────────────────────

def fit_isolation_forest(
    X_train: np.ndarray, n_estimators: int = None, seed: int = None
) -> IsolationForest:
    if n_estimators is None:
        n_estimators = IF_N_ESTIMATORS_DEFAULT
    if seed is None:
        seed = SEED
    clf = IsolationForest(
        n_estimators=n_estimators,
        contamination=IF_CONTAMINATION,
        random_state=seed,
    )
    clf.fit(X_train)
    return clf


def score_anomaly(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    return -model.decision_function(X)


# ── Stage 2: Per-engine smoothing ─────────────────────────────────────────────

def smooth_scores_per_unit(
    df: pd.DataFrame, raw_scores: np.ndarray, window: int = None
) -> np.ndarray:
    if window is None:
        window = ROLLING_WINDOW
    tmp = pd.DataFrame({
        "unit_id": df["unit_id"].values,
        "cycle":   df["cycle"].values,
        "_score":  raw_scores,
        "_pos":    np.arange(len(raw_scores)),
    })
    tmp = tmp.sort_values(["unit_id", "cycle"])
    smoothed = tmp.groupby("unit_id")["_score"].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )
    result = np.empty(len(raw_scores))
    result[tmp["_pos"].values] = smoothed.values
    return result


# ── Stage 3: Pseudo-label generation (zone-split) ─────────────────────────────

def generate_pseudo_labels(scores: np.ndarray, threshold: float) -> np.ndarray:
    cutoff = np.quantile(scores, 1.0 - threshold)
    pseudo = np.empty(len(scores), dtype=float)

    anom_mask = scores >= cutoff
    norm_mask = ~anom_mask

    # Anomaly zone → [0, 62]: cutoff→62, max_score→0
    if anom_mask.any():
        s = scores[anom_mask]
        denom = s.max() - cutoff
        if denom == 0:
            pseudo[anom_mask] = (ZONE_DEGRADED[0] + ZONE_DEGRADED[1]) / 2.0
        else:
            pseudo[anom_mask] = ZONE_DEGRADED[1] - ZONE_DEGRADED[1] * (s - cutoff) / denom

    # Normal zone → [63, 125]: min_score→125, just-below-cutoff→63
    if norm_mask.any():
        s = scores[norm_mask]
        min_norm = s.min()
        denom = cutoff - min_norm
        if denom == 0:
            pseudo[norm_mask] = (ZONE_HEALTHY[0] + ZONE_HEALTHY[1]) / 2.0
        else:
            zone_span = ZONE_HEALTHY[1] - ZONE_HEALTHY[0]   # 62
            pseudo[norm_mask] = ZONE_HEALTHY[1] - zone_span * (s - min_norm) / denom

    return np.clip(pseudo, 0, 125)


# ── Stage 4: Ridge regression ─────────────────────────────────────────────────

def train_ridge(X_train: np.ndarray, y_train: np.ndarray) -> Ridge:
    return Ridge(alpha=RIDGE_ALPHA).fit(X_train, y_train)


# ── Stage 6: Metrics ──────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    diff   = y_pred - y_true
    rmse   = float(np.sqrt(np.mean(diff ** 2)))
    mae    = float(np.mean(np.abs(diff)))
    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2     = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    nasa   = float(np.sum(
        np.where(diff < 0, np.exp(-diff / 13) - 1, np.exp(diff / 10) - 1)
    ))
    return {"rmse": rmse, "mae": mae, "r2": r2, "nasa_score": nasa}


def compute_correlation(scores: np.ndarray, true_rul: np.ndarray) -> float:
    r, p = pearsonr(scores, true_rul)
    return float(r)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_global_pipeline(
    subset_name: str,
    n_estimators: int = None,
    threshold: float = None,
    seed: int = None,
) -> dict:
    name = subset_name.upper()
    if n_estimators is None:
        n_estimators = IF_N_ESTIMATORS_DEFAULT
    if threshold is None:
        threshold = THRESHOLD_SINGLE if name in SINGLE_CONDITION else THRESHOLD_MULTI

    load_mod = _load_module("load_data", "01_load_data.py")
    prep_mod = _load_module("preprocess", "02_preprocess.py")

    raw    = load_mod.load_subset(name)
    result = prep_mod.preprocess_subset(raw, name)

    train_df     = result["train"]
    test_df      = result["test"]
    feature_cols = result["feature_cols"]

    X_train = train_df[feature_cols].values
    X_test  = test_df[feature_cols].values

    # Stage 1
    clf = fit_isolation_forest(X_train, n_estimators, seed=seed)

    # Stage 2 — training: score all cycles, apply per-engine rolling mean
    raw_train    = score_anomaly(clf, X_train)
    smooth_train = smooth_scores_per_unit(train_df, raw_train)

    # Pearson r on training set
    pearson_r = compute_correlation(smooth_train, train_df["true_RUL"].values)

    # Stage 3
    pseudo_rul = generate_pseudo_labels(smooth_train, threshold)

    # Stage 4: Ridge features = scaled sensors/op_settings + smoothed anomaly score
    X_train_r = np.column_stack([X_train, smooth_train])

    pseudo_model = train_ridge(X_train_r, pseudo_rul)
    true_model   = train_ridge(X_train_r, train_df["true_RUL"].values)

    raw_test      = score_anomaly(clf, X_test)
    smooth_test   = smooth_scores_per_unit(test_df, raw_test)
    X_test_full_r = np.column_stack([X_test, smooth_test])
    last_idx      = test_df.groupby("unit_id")["cycle"].idxmax()
    is_last       = test_df.index.isin(last_idx)
    X_test_r      = X_test_full_r[is_last]
    last_df       = test_df.loc[last_idx]

    y_pred_pseudo = np.clip(pseudo_model.predict(X_test_r), 0, 125)
    y_pred_true   = np.clip(true_model.predict(X_test_r),   0, 125)

    unit_ids_sorted = sorted(test_df["unit_id"].unique())
    rul_map         = dict(zip(unit_ids_sorted, raw["rul"].values))
    y_true_eng      = last_df["unit_id"].map(rul_map).values.astype(float)

    return {
        "subset":            name,
        "n_estimators":      n_estimators,
        "threshold":         threshold,
        "n_features_total":  len(feature_cols) + 1,
        "pearson_r":         pearson_r,
        "pseudo_metrics":    compute_metrics(y_true_eng, y_pred_pseudo),
        "true_metrics":      compute_metrics(y_true_eng, y_pred_true),
        "y_true_per_engine": y_true_eng,
        "y_pred_pseudo":     y_pred_pseudo,
        "y_pred_true":       y_pred_true,
    }


# ── __main__ ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Canonical reference values (previously loaded from verify_baseline.py)
    CANONICAL = {
        "FD001": {"r": -0.5931, "pseudo_rmse_default": 32.05, "true_rmse": 21.15},
        "FD002": {"r": -0.5281, "pseudo_rmse_default": 46.08, "true_rmse": 34.18},
        "FD003": {"r": -0.6904, "pseudo_rmse_default": 28.87, "true_rmse": 22.50},
        "FD004": {"r": -0.6244, "pseudo_rmse_default": 45.62, "true_rmse": 36.07},
    }
    TOL_PCT = 1.0  # canonical reference tolerance (%)

    results = {}
    for subset in SUBSETS:
        print(f"Running {subset}...")
        results[subset] = run_global_pipeline(subset)

    # Summary table
    W = 90
    print("\n" + "=" * W)
    print("GLOBAL PIPELINE - Default Config (n=200, thr=0.20/0.15)")
    hdr = (
        f"{'Subset':<8} {'Pearson_r':>10} {'Pseudo_RMSE':>12} {'Pseudo_MAE':>11} "
        f"{'Pseudo_NASA':>12} {'True_RMSE':>10} {'True_MAE':>9} {'True_NASA':>10}"
    )
    print(hdr)
    print("-" * W)
    rows = []
    for subset, r in results.items():
        pm, tm = r["pseudo_metrics"], r["true_metrics"]
        print(
            f"{subset:<8} {r['pearson_r']:>10.4f} {pm['rmse']:>12.2f} {pm['mae']:>11.2f} "
            f"{pm['nasa_score']:>12.0f} {tm['rmse']:>10.2f} {tm['mae']:>9.2f} "
            f"{tm['nasa_score']:>10.0f}"
        )
        rows.append({
            "subset":       subset,
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

    # Save table
    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        tables_dir / "global_pipeline_default_config.csv", index=False
    )
    print(f"\nSaved: results/tables/global_pipeline_default_config.csv")

    # Save per-subset predictions
    preds_dir = RESULTS_DIR / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)
    for subset, r in results.items():
        pd.DataFrame({
            "y_true":        r["y_true_per_engine"],
            "y_pred_pseudo": r["y_pred_pseudo"],
            "y_pred_true":   r["y_pred_true"],
        }).to_csv(preds_dir / f"global_default_{subset}.csv", index=False)
    print("Saved: results/predictions/global_default_FD00X.csv (x4)")

    # Sanity check
    print("\n-- Sanity Check (tolerance +/-1%) " + "-" * 38)
    print(f"{'Subset':<8} {'Metric':<22} {'Canonical':>10} {'Got':>10} {'Err%':>7}  Status")
    print("-" * 66)
    all_pass = True
    for subset, r in results.items():
        canon = CANONICAL[subset]
        pm, tm = r["pseudo_metrics"], r["true_metrics"]
        canon_pseudo_nasa = {
            "FD001": 4018, "FD002": 56301, "FD003": 3187, "FD004": 49575,
        }
        canon_true_nasa = {
            "FD001": 1156, "FD002": 19283, "FD003": 2471, "FD004": 13898,
        }
        checks = [
            ("pearson_r",    canon["r"],                   r["pearson_r"]),
            ("pseudo_rmse",  canon["pseudo_rmse_default"], pm["rmse"]),
            ("true_rmse",    canon["true_rmse"],           tm["rmse"]),
            ("pseudo_nasa",  canon_pseudo_nasa[subset],    pm["nasa_score"]),
            ("true_nasa",    canon_true_nasa[subset],      tm["nasa_score"]),
        ]
        for metric, expected, got in checks:
            pct = abs(got - expected) / abs(expected) * 100 if abs(expected) > 1e-9 else 0.0
            status = "PASS" if pct <= TOL_PCT else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(
                f"{subset:<8} {metric:<22} {expected:>10.4f} {got:>10.4f} "
                f"{pct:>6.2f}%  {status}"
            )

    print()
    print("Overall:", "ALL PASS" if all_pass else "SOME CHECKS FAILED — see above")

