"""
15_within_engine_correlation.py
Within-engine Pearson r: corr(IF_anomaly_score, true_RUL_linear) per engine.

For each training engine we compute:
  r = Pearson(smoothed_anomaly_score, true_RUL_linear)
where true_RUL_linear = max_cycle_per_engine - current_cycle  (UNCLIPPED).

Multi-condition subsets (FD002, FD004): engines grouped by dominant KMeans cluster
(k=6 on op_settings) to test whether within-r sign/magnitude differs by condition.

Single-condition subsets (FD001, FD003): sanity check -- all engines in condition 0,
heterogeneity should be absent.

Hypothesis: if pipeline works correctly, all within-r should be < 0
(high anomaly score when RUL is low).

Outputs:
  results/cmapss/within_engine_correlation_summary.csv
  results/cmapss/within_r_by_condition.csv
  results/cmapss/heterogeneity_test.csv
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, pearsonr
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

_SRC = Path(__file__).parent
sys.path.insert(0, str(_SRC.parent))

from cmapss.config import (
    IF_CONTAMINATION,
    IF_N_ESTIMATORS_DEFAULT,
    KMEANS_K_MULTI,
    MULTI_CONDITION,
    RESULTS_DIR,
    ROLLING_WINDOW,
    SEED,
    SINGLE_CONDITION,
    SUBSETS,
)

MIN_CYCLES = 20       # skip engines with fewer cycles (r unreliable)
MIN_CYCLES_COND = 8   # per (engine, cluster) within-condition r threshold
OUT_DIR = RESULTS_DIR / "cmapss"

ZERO_VAR_SENSORS = [
    "sensor_1", "sensor_5", "sensor_6", "sensor_10",
    "sensor_16", "sensor_18", "sensor_19",
]


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, _SRC / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_train(subset_name: str) -> tuple[pd.DataFrame, list]:
    """Load and preprocess training data; return (df, feature_cols).
    true_RUL_linear = max_cycle - cycle  (unclipped, for correlation).
    true_RUL        = clipped at 125     (used to scale IF for consistency).
    """
    ld  = _load_module("load_data", "01_load_data.py")
    raw = ld.load_subset(subset_name)
    tr  = raw["train"].copy()

    # Zero-variance sensor drop
    tr.drop(columns=[c for c in ZERO_VAR_SENSORS if c in tr.columns], inplace=True)
    # op_setting_3 drop for single-condition
    if subset_name in SINGLE_CONDITION and "op_setting_3" in tr.columns:
        tr.drop(columns=["op_setting_3"], inplace=True)

    # Linear (unclipped) RUL for correlation
    max_cyc = tr.groupby("unit_id")["cycle"].transform("max")
    tr["rul_linear"] = max_cyc - tr["cycle"]

    non_feat = {"unit_id", "cycle", "rul_linear"}
    feature_cols = [c for c in tr.columns if c not in non_feat]

    # Global MinMaxScaler (same as 03a global pipeline)
    scaler = MinMaxScaler()
    tr[feature_cols] = scaler.fit_transform(tr[feature_cols])

    return tr, feature_cols


def _fit_if(X_train: np.ndarray) -> IsolationForest:
    clf = IsolationForest(
        n_estimators=IF_N_ESTIMATORS_DEFAULT,
        contamination=IF_CONTAMINATION,
        random_state=SEED,
    )
    clf.fit(X_train)
    return clf


def _smooth_scores(df: pd.DataFrame, raw_scores: np.ndarray) -> np.ndarray:
    tmp = pd.DataFrame({
        "unit_id": df["unit_id"].values,
        "cycle":   df["cycle"].values,
        "_score":  raw_scores,
        "_pos":    np.arange(len(raw_scores)),
    })
    tmp = tmp.sort_values(["unit_id", "cycle"])
    smoothed = tmp.groupby("unit_id")["_score"].transform(
        lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).mean()
    )
    result = np.empty(len(raw_scores))
    result[tmp["_pos"].values] = smoothed.values
    return result


def _within_engine_r(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-engine within-r (all cycles, regardless of cluster).
    dominant_condition = most-frequent cluster across the engine's cycles.
    """
    rows = []
    for uid, grp in df.groupby("unit_id"):
        grp = grp.sort_values("cycle")
        n = len(grp)
        if n < MIN_CYCLES:
            continue
        s = grp["score"].values
        r = grp["rul_linear"].values
        if s.std() == 0 or r.std() == 0:
            within_r = float("nan")
        else:
            within_r = float(pearsonr(s, r)[0])
        # dominant condition: most frequent cluster label
        dom_cond = int(grp["cluster"].value_counts().idxmax()) \
            if "cluster" in grp.columns else 0
        rows.append({
            "engine_id":  uid,
            "n_cycles":   n,
            "within_r":   within_r,
            "dominant_condition": dom_cond,
        })
    return pd.DataFrame(rows)


def _within_engine_r_per_cluster(df: pd.DataFrame) -> pd.DataFrame:
    """For multi-condition: compute within-r per (engine, cluster) pair.
    Uses only the cycles belonging to that cluster for each pair.
    Aggregates across engines per cluster -> cluster-level within-r distribution.
    """
    rows = []
    for (uid, cond), grp in df.groupby(["unit_id", "cluster"]):
        grp = grp.sort_values("cycle")
        n = len(grp)
        if n < MIN_CYCLES_COND:
            continue
        s = grp["score"].values
        r = grp["rul_linear"].values
        if s.std() == 0 or r.std() == 0:
            within_r = float("nan")
        else:
            within_r = float(pearsonr(s, r)[0])
        rows.append({
            "engine_id": uid,
            "cluster":   int(cond),
            "n_cycles":  n,
            "within_r":  within_r,
        })
    return pd.DataFrame(rows)


def process_subset(subset_name: str) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print(f"Processing {subset_name}")
    print(f"{'='*60}")

    tr, feature_cols = _load_train(subset_name)
    X_tr = tr[feature_cols].values

    # KMeans clustering
    is_multi = subset_name in MULTI_CONDITION
    k = KMEANS_K_MULTI if is_multi else 1
    if is_multi:
        op_cols = [c for c in feature_cols if c.startswith("op_setting_")]
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        tr["cluster"] = km.fit_predict(tr[op_cols])
        print(f"  KMeans k={k}: cluster sizes = "
              f"{tr['cluster'].value_counts().sort_index().to_dict()}")
    else:
        tr["cluster"] = 0

    # IF + smoothed scores
    clf = _fit_if(X_tr)
    raw = -clf.decision_function(X_tr)
    tr["score"] = _smooth_scores(tr, raw)

    # Per-engine within-r (all cycles)
    per_engine = _within_engine_r(tr)
    per_engine.insert(0, "subset", subset_name)

    n_valid = per_engine["within_r"].notna().sum()
    n_pos   = (per_engine["within_r"] > 0).sum()
    n_neg   = (per_engine["within_r"] < 0).sum()
    mean_r  = per_engine["within_r"].mean()
    std_r   = per_engine["within_r"].std()

    print(f"  Engines with valid r: {n_valid}")
    print(f"  Mean within-r: {mean_r:.4f}  Std: {std_r:.4f}")
    print(f"  Positive r: {n_pos} ({100*n_pos/n_valid:.1f}%)  "
          f"Negative r: {n_neg} ({100*n_neg/n_valid:.1f}%)")

    # Per-cluster within-r (multi-condition: within-condition analysis)
    if is_multi:
        per_cluster = _within_engine_r_per_cluster(tr)
        per_cluster.insert(0, "subset", subset_name)
        clu_agg = (
            per_cluster.groupby("cluster")["within_r"]
            .agg(n_pairs="count", mean_r="mean", std_r="std",
                 pct_positive=lambda x: 100*(x>0).mean())
            .reset_index()
        )
        print(f"\n  Per-cluster within-r (engine x cluster pairs, min_cycles={MIN_CYCLES_COND}):")
        print(f"  {'cluster':>8} {'n_pairs':>8} {'mean_r':>8} {'std_r':>7} {'pct_pos':>8}")
        for _, row in clu_agg.iterrows():
            print(f"  {int(row.cluster):>8} {int(row.n_pairs):>8} "
                  f"{row.mean_r:>8.4f} {row.std_r:>7.4f} {row.pct_positive:>7.1f}%")

    return per_engine


def by_condition_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (subset, cond), grp in summary_df.groupby(["subset", "dominant_condition"]):
        r_vals = grp["within_r"].dropna()
        n = len(r_vals)
        if n == 0:
            continue
        rows.append({
            "subset":      subset,
            "condition":   cond,
            "n_engines":   n,
            "mean_r":      round(float(r_vals.mean()), 4),
            "std_r":       round(float(r_vals.std(ddof=1)) if n > 1 else 0.0, 4),
            "pct_positive": round(100 * (r_vals > 0).sum() / n, 1),
            "min_r":       round(float(r_vals.min()), 4),
            "max_r":       round(float(r_vals.max()), 4),
        })
    return pd.DataFrame(rows).sort_values(["subset", "condition"]).reset_index(drop=True)


def heterogeneity_test(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subset in SUBSETS:
        sub = summary_df[summary_df["subset"] == subset]
        groups = [
            grp["within_r"].dropna().values
            for _, grp in sub.groupby("dominant_condition")
            if len(grp["within_r"].dropna()) >= 3
        ]
        if len(groups) < 2:
            rows.append({
                "subset": subset, "test": "Kruskal-Wallis",
                "statistic": float("nan"), "p_value": float("nan"),
                "n_groups": len(groups), "conclusion": "insufficient groups",
            })
            continue
        stat, p = kruskal(*groups)
        conclusion = (
            "heterogeneous (p<0.05)" if p < 0.05
            else "homogeneous (p>=0.05)"
        )
        rows.append({
            "subset": subset, "test": "Kruskal-Wallis",
            "statistic": round(stat, 4), "p_value": round(p, 6),
            "n_groups": len(groups), "conclusion": conclusion,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_engines = []
    for subset in SUBSETS:
        df = process_subset(subset)
        all_engines.append(df)

    summary_df = pd.concat(all_engines, ignore_index=True)

    # ── By-condition table ───────────────────────────────────────────────────
    cond_df = by_condition_summary(summary_df)

    W = 100
    print("\n" + "=" * W)
    print("Within-r by Condition")
    print("=" * W)
    print(cond_df.to_string(index=False))

    # ── Sanity: FD001, FD003 (single-condition) ──────────────────────────────
    print("\n" + "-" * W)
    print("Sanity Check: FD001, FD003 (single condition, should be homogeneous)")
    print("-" * W)
    for s in SINGLE_CONDITION:
        sub = summary_df[summary_df["subset"] == s]["within_r"].dropna()
        n_pos = (sub > 0).sum()
        print(f"  {s}: n={len(sub)}  mean_r={sub.mean():.4f}  std={sub.std():.4f}  "
              f"pct_positive={100*n_pos/len(sub):.1f}%")

    # ── Per-cluster heterogeneity (multi-condition only) ────────────────────
    print("\n" + "-" * W)
    print("Per-Cluster Within-r (multi-condition: within-condition within-engine r)")
    print("Each row = one (engine, cluster) pair, aggregated across engines.")
    print("-" * W)
    clu_rows = []
    for subset in MULTI_CONDITION:
        tr_data, feat = _load_train(subset)
        op_cols = [c for c in feat if c.startswith("op_setting_")]
        km = KMeans(n_clusters=KMEANS_K_MULTI, random_state=SEED, n_init=10)
        tr_data["cluster"] = km.fit_predict(tr_data[op_cols])
        clf2 = _fit_if(tr_data[feat].values)
        raw2 = -clf2.decision_function(tr_data[feat].values)
        tr_data["score"] = _smooth_scores(tr_data, raw2)
        pc = _within_engine_r_per_cluster(tr_data)
        agg = (
            pc.groupby("cluster")["within_r"]
            .agg(n_pairs="count", mean_r="mean", std_r="std",
                 pct_positive=lambda x: 100*(x>0).mean())
            .reset_index()
        )
        agg.insert(0, "subset", subset)
        clu_rows.append(agg)
        # KW test across clusters
        groups = [
            pc[pc["cluster"]==c]["within_r"].dropna().values
            for c in pc["cluster"].unique()
            if len(pc[pc["cluster"]==c]["within_r"].dropna()) >= 3
        ]
        if len(groups) >= 2:
            stat, p = kruskal(*groups)
            print(f"  {subset} KW test: H={stat:.3f}  p={p:.4f}  "
                  f"({'heterogeneous' if p<0.05 else 'homogeneous'})")

    clu_summary = pd.concat(clu_rows, ignore_index=True)
    print(clu_summary.to_string(index=False))

    # ── Heterogeneity test ───────────────────────────────────────────────────
    print("\n" + "-" * W)
    print("Kruskal-Wallis Heterogeneity Test (within-r distributions by condition)")
    print("-" * W)
    het_df = heterogeneity_test(summary_df)
    print(het_df.to_string(index=False))

    # ── Most striking finding ────────────────────────────────────────────────
    print("\n" + "-" * W)
    print("Most striking per-condition stats (mean_r, pct_positive):")
    print("-" * W)
    for _, row in cond_df.iterrows():
        marker = " <-- positive majority" if row["pct_positive"] > 50 else ""
        print(f"  {row['subset']} cond={row['condition']}  "
              f"n={row['n_engines']:3d}  mean_r={row['mean_r']:+.4f}  "
              f"pct_pos={row['pct_positive']:5.1f}%{marker}")

    # ── Save outputs ─────────────────────────────────────────────────────────
    out1 = OUT_DIR / "within_engine_correlation_summary.csv"
    out2 = OUT_DIR / "within_r_by_condition.csv"
    out3 = OUT_DIR / "heterogeneity_test.csv"

    summary_df.to_csv(out1, index=False)
    cond_df.to_csv(out2, index=False)
    het_df.to_csv(out3, index=False)

    print(f"\nSaved: {out1}")
    print(f"Saved: {out2}")
    print(f"Saved: {out3}")

    # ── Scenario verdict ─────────────────────────────────────────────────────
    print("\n" + "=" * W)
    print("Scenario Verdict")
    print("=" * W)
    multi_het = het_df[het_df["subset"].isin(MULTI_CONDITION)]
    n_heterogeneous = (multi_het["conclusion"].str.startswith("heterogeneous")).sum()
    any_positive_multi = (
        cond_df[cond_df["subset"].isin(MULTI_CONDITION)]["pct_positive"] > 50
    ).any()

    if n_heterogeneous >= 1 and any_positive_multi:
        verdict = ("H1 -- C-MAPSS multi-condition subsets show condition-dependent "
                   "within-r with positive-majority conditions. "
                   "Cross-dataset mechanistic pattern confirmed.")
    elif n_heterogeneous >= 1:
        verdict = ("H1 partial -- Kruskal-Wallis significant but no condition "
                   "shows positive-majority within-r. Heterogeneity exists but milder.")
    else:
        verdict = ("H2 -- C-MAPSS within-r homogeneous across conditions. "
                   "FEMTO heterogeneity is dataset-specific.")

    single_pct_pos = (
        summary_df[summary_df["subset"].isin(SINGLE_CONDITION)]["within_r"] > 0
    ).mean() * 100
    if single_pct_pos > 30:
        verdict += (f"\n  WARNING: FD001/FD003 also show {single_pct_pos:.1f}% "
                    "positive within-r -- sanity check concern.")
    else:
        verdict += f"\n  Sanity OK: FD001/FD003 only {single_pct_pos:.1f}% positive within-r."

    print(f"  {verdict}")
