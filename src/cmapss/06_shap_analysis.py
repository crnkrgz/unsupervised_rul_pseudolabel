"""
06_shap_analysis.py
SHAP feature attribution for Ridge models (Tables 4.6, 4.7).
Also generates unified per-cycle test prediction CSVs for all 4 subsets.
"""

import importlib.util
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from cmapss.config import RESULTS_DIR, SUBSETS, SINGLE_CONDITION, KMEANS_K_MULTI

_SRC      = Path(__file__).parent
_PRED_DIR = RESULTS_DIR / "predictions"
_TAB_DIR  = RESULTS_DIR / "tables"
_FIG_DIR  = RESULTS_DIR / "figures"

OPTIMAL = {
    "FD001": {"n_estimators": 200, "threshold": 0.20},
    "FD002": {"n_estimators": 200, "threshold": 0.15},
    "FD003": {"n_estimators": 200, "threshold": 0.20},
    "FD004": {"n_estimators": 200, "threshold": 0.15},
}


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, _SRC / filename)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_pipeline_data(subset_name: str) -> dict:
    name = subset_name.upper()
    cfg  = OPTIMAL[name]

    load_mod  = _load_module("load_data",    "01_load_data.py")
    prep_mod  = _load_module("preprocess",   "02_preprocess.py")
    pipe_mod  = _load_module("global_pipe",  "03a_global_pipeline.py")
    clust_mod = _load_module("cluster_pipe", "03b_cluster_pipeline.py")

    raw    = load_mod.load_subset(name)
    result = prep_mod.preprocess_subset(raw, name, scale=False)
    train_df     = result["train"]
    test_df      = result["test"]
    feature_cols = result["feature_cols"]

    k = 1 if name in SINGLE_CONDITION else KMEANS_K_MULTI
    train_df, test_df, _km      = clust_mod.cluster_op_settings(train_df, test_df, k)
    train_df, test_df, _scalers = clust_mod.per_cluster_minmax(train_df, test_df, feature_cols, k)

    X_train = train_df[feature_cols].values
    X_test  = test_df[feature_cols].values

    clf          = pipe_mod.fit_isolation_forest(X_train, cfg["n_estimators"])
    raw_train    = pipe_mod.score_anomaly(clf, X_train)
    smooth_train = pipe_mod.smooth_scores_per_unit(train_df, raw_train)
    pseudo_rul   = pipe_mod.generate_pseudo_labels(smooth_train, cfg["threshold"])

    X_train_r    = np.column_stack([X_train, smooth_train])
    pseudo_model = pipe_mod.train_ridge(X_train_r, pseudo_rul)
    true_model   = pipe_mod.train_ridge(X_train_r, train_df["true_RUL"].values)

    raw_test    = pipe_mod.score_anomaly(clf, X_test)
    smooth_test = pipe_mod.smooth_scores_per_unit(test_df, raw_test)
    X_test_r    = np.column_stack([X_test, smooth_test])

    y_pred_pseudo_full = np.clip(pseudo_model.predict(X_test_r), 0, 125)
    y_pred_true_full   = np.clip(true_model.predict(X_test_r),   0, 125)
    pseudo_label_full  = pipe_mod.generate_pseudo_labels(smooth_test, cfg["threshold"])

    last_idx = test_df.groupby("unit_id")["cycle"].idxmax()
    is_last  = test_df.index.isin(last_idx)

    unit_ids_sorted  = sorted(test_df["unit_id"].unique())
    rul_map          = dict(zip(unit_ids_sorted, raw["rul"].values))
    y_true_per_row   = test_df["unit_id"].map(rul_map).values.astype(float)
    y_true_unclipped = np.where(is_last, y_true_per_row, np.nan)

    pred_df = pd.DataFrame({
        "unit_id":            test_df["unit_id"].values,
        "cycle":              test_df["cycle"].values,
        "true_RUL_per_cycle": test_df["true_RUL"].values,
        "anomaly_score":      smooth_test,
        "pseudo_label":       pseudo_label_full,
        "y_pred_pseudo":      y_pred_pseudo_full,
        "y_pred_true":        y_pred_true_full,
        "y_true_unclipped":   y_true_unclipped,
        "is_last_cycle":      is_last.astype(int),
    })
    _PRED_DIR.mkdir(parents=True, exist_ok=True)
    cache = _PRED_DIR / f"cluster_optimal_{name}_full.csv"
    pred_df.to_csv(cache, index=False)
    print(f"  Saved: {cache.name}")

    feature_names = feature_cols + ["anomaly_score"]
    X_test_last_r = X_test_r[is_last]

    return {
        "subset":        name,
        "train_df":      train_df,
        "X_train_r":     X_train_r,
        "X_test_last_r": X_test_last_r,
        "pseudo_model":  pseudo_model,
        "true_model":    true_model,
        "feature_names": feature_names,
    }


def compute_shap_for_subset(data: dict) -> dict:
    X_tr = data["X_train_r"]
    X_te = data["X_test_last_r"]
    fn   = data["feature_names"]

    exp_p = shap.LinearExplainer(data["pseudo_model"], X_tr)
    exp_t = shap.LinearExplainer(data["true_model"],   X_tr)

    sv_p = exp_p.shap_values(X_te)
    sv_t = exp_t.shap_values(X_te)

    ma_p = np.abs(sv_p).mean(axis=0)
    ma_t = np.abs(sv_t).mean(axis=0)

    return {
        "sv_pseudo":       sv_p,
        "sv_true":         sv_t,
        "mean_abs_pseudo": ma_p,
        "mean_abs_true":   ma_t,
        "rank_pseudo":     np.argsort(-ma_p),
        "rank_true":       np.argsort(-ma_t),
        "feature_names":   fn,
        "X_test_last_r":   X_te,
    }


if __name__ == "__main__":
    t0 = time.perf_counter()
    _TAB_DIR.mkdir(parents=True, exist_ok=True)
    _FIG_DIR.mkdir(parents=True, exist_ok=True)

    all_data = {}
    all_shap = {}

    for subset in SUBSETS:
        print(f"\n[{subset}] Building pipeline + predictions...")
        all_data[subset] = build_pipeline_data(subset)
        print(f"[{subset}] Computing SHAP...")
        all_shap[subset] = compute_shap_for_subset(all_data[subset])

    # ── Table 4.6: FD001 top-10 pseudo with true side-by-side ───────────────────
    sh1 = all_shap["FD001"]
    fn  = sh1["feature_names"]
    true_rank_map = {fn[i]: r + 1 for r, i in enumerate(sh1["rank_true"])}

    rows46 = []
    for rk, fi in enumerate(sh1["rank_pseudo"][:10]):
        feat = fn[fi]
        rows46.append({
            "rank_pseudo":          rk + 1,
            "feature":              feat,
            "mean_abs_shap_pseudo": round(float(sh1["mean_abs_pseudo"][fi]), 4),
            "rank_true":            true_rank_map[feat],
            "mean_abs_shap_true":   round(float(sh1["mean_abs_true"][fi]),   4),
        })
    pd.DataFrame(rows46).to_csv(_TAB_DIR / "table_4_6_shap_fd001_detail.csv", index=False)
    print("\nSaved: table_4_6_shap_fd001_detail.csv")

    # ── Table 4.7: top-5 per subset (pseudo) ────────────────────────────────────
    rows47 = []
    for rank in range(5):
        row = {"rank": rank + 1}
        for subset in SUBSETS:
            sh = all_shap[subset]
            fi = sh["rank_pseudo"][rank]
            row[f"{subset}_feature"] = sh["feature_names"][fi]
            row[f"{subset}_shap"]    = round(float(sh["mean_abs_pseudo"][fi]), 4)
        rows47.append(row)
    pd.DataFrame(rows47).to_csv(_TAB_DIR / "table_4_7_shap_cross_subset.csv", index=False)
    print("Saved: table_4_7_shap_cross_subset.csv")

    # ── Console: top-3 per subset ────────────────────────────────────────────────
    print("\nTop-3 Features (Pseudo Model):")
    print(f"{'Subset':<8} {'#1 Feature':<22} {'SHAP':>8}  {'#2 Feature':<22} {'SHAP':>8}  {'#3 Feature':<22} {'SHAP':>8}")
    print("-" * 96)
    for subset in SUBSETS:
        sh   = all_shap[subset]
        top3 = [
            (sh["feature_names"][sh["rank_pseudo"][i]], sh["mean_abs_pseudo"][sh["rank_pseudo"][i]])
            for i in range(3)
        ]
        print(f"{subset:<8} {top3[0][0]:<22} {top3[0][1]:>8.4f}  {top3[1][0]:<22} {top3[1][1]:>8.4f}  {top3[2][0]:<22} {top3[2][1]:>8.4f}")

    # ── Sanity check ─────────────────────────────────────────────────────────────
    print("\n-- Sanity Check --")
    fd1  = all_shap["FD001"]
    fi_p = fd1["rank_pseudo"][0]
    fi_t = fd1["rank_true"][0]
    top_p_feat = fd1["feature_names"][fi_p]
    top_p_val  = fd1["mean_abs_pseudo"][fi_p]
    top_t_feat = fd1["feature_names"][fi_t]
    top_t_val  = fd1["mean_abs_true"][fi_t]
    pass_p = top_p_feat == "anomaly_score"
    pass_t = top_t_feat in ("sensor_11", "sensor_12")
    print(f"FD001 pseudo top: {top_p_feat}  mean|SHAP|={top_p_val:.4f}  (canonical: anomaly_score ~11.582)  {'PASS' if pass_p else 'FAIL'}")
    print(f"FD001 true  top: {top_t_feat}   mean|SHAP|={top_t_val:.4f}  (canonical: sensor_11 ~4.444 or sensor_12 ~3.787)  {'PASS' if pass_t else 'WARN'}")
    all_top_anom = all(
        all_shap[s]["feature_names"][all_shap[s]["rank_pseudo"][0]] == "anomaly_score"
        for s in SUBSETS
    )
    print(f"anomaly_score #1 pseudo in all subsets: {'PASS' if all_top_anom else 'FAIL'}")

    # ── Figure 1: beeswarm FD001 pseudo ──────────────────────────────────────────
    shap.summary_plot(
        fd1["sv_pseudo"], fd1["X_test_last_r"],
        feature_names=fd1["feature_names"],
        plot_type="dot", max_display=10, show=False,
    )
    plt.title("SHAP Beeswarm — Pseudo-Label Model (FD001)", pad=14)
    plt.savefig(_FIG_DIR / "shap_beeswarm_fd001_pseudo.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("\nSaved: shap_beeswarm_fd001_pseudo.png")

    # ── Figure 2: bar chart pseudo vs true top-10 (FD001) ────────────────────────
    fig2, (axL, axR) = plt.subplots(1, 2, figsize=(14, 6))

    top10p = fd1["rank_pseudo"][:10]
    axL.barh([fn[i] for i in top10p[::-1]], [fd1["mean_abs_pseudo"][i] for i in top10p[::-1]], color="#2E86AB")
    axL.set_xlabel("Mean |SHAP Value|"); axL.set_title("Pseudo-Label Model")
    axL.grid(axis="x", linestyle="--", alpha=0.4)

    top10t = fd1["rank_true"][:10]
    axR.barh([fn[i] for i in top10t[::-1]], [fd1["mean_abs_true"][i] for i in top10t[::-1]], color="#E63946")
    axR.set_xlabel("Mean |SHAP Value|"); axR.set_title("True-Label Model")
    axR.grid(axis="x", linestyle="--", alpha=0.4)

    fig2.suptitle("SHAP Feature Importance — Pseudo vs True-Label Model (FD001)", fontsize=13)
    plt.tight_layout()
    fig2.savefig(_FIG_DIR / "shap_top10_pseudo_vs_true.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print("Saved: shap_top10_pseudo_vs_true.png")

    # ── Figure 3: SHAP temporal evolution (FD001, unit_id=1) ─────────────────────
    data1    = all_data["FD001"]
    train_df = data1["train_df"]
    X_tr_r   = data1["X_train_r"]
    fn1      = data1["feature_names"]

    u1_mask   = (train_df["unit_id"] == 1).values
    X_u1      = X_tr_r[u1_mask]
    cycles_u1 = train_df["cycle"].values[u1_mask]
    rul_u1    = train_df["true_RUL"].values[u1_mask]

    exp_u1 = shap.LinearExplainer(data1["pseudo_model"], X_tr_r)
    sv_u1  = exp_u1.shap_values(X_u1)

    top5_idx = np.argsort(-np.abs(sv_u1).mean(axis=0))[:5]
    palette  = ["#2E86AB", "#E63946", "#06A77D", "#1D3557", "#F4A261"]

    fig3, ax_m = plt.subplots(figsize=(12, 6))
    for j, fi in enumerate(top5_idx):
        ax_m.plot(cycles_u1, sv_u1[:, fi], label=fn1[fi], color=palette[j], linewidth=1.5)

    rlt25 = cycles_u1[rul_u1 < 25]
    if len(rlt25) > 0:
        ax_m.axvline(rlt25.min(), color="black", linestyle="--", linewidth=1.2, label="RUL < 25")

    ax_m.set_xlabel("Cycle"); ax_m.set_ylabel("SHAP Value")
    ax_m.grid(True, linestyle="--", alpha=0.3)

    ax_rul = ax_m.twinx()
    ax_rul.fill_between(cycles_u1, rul_u1, alpha=0.12, color="grey", label="True RUL")
    ax_rul.set_ylabel("True RUL (cycles)", color="grey")
    ax_rul.tick_params(axis="y", labelcolor="grey")
    ax_rul.set_ylim(0, float(rul_u1.max()) * 1.15)

    lines1, labs1 = ax_m.get_legend_handles_labels()
    lines2, labs2 = ax_rul.get_legend_handles_labels()
    ax_m.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc="upper left")
    ax_m.set_title("SHAP Temporal Evolution — Top 5 Features, Motor 1 (FD001)")

    plt.tight_layout()
    fig3.savefig(_FIG_DIR / "shap_temporal_evolution_fd001.png", dpi=300, bbox_inches="tight")
    plt.close(fig3)
    print("Saved: shap_temporal_evolution_fd001.png")

    print(f"\nTotal runtime: {(time.perf_counter()-t0)/60:.1f} min")

