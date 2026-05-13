"""
04_sensitivity.py
Hyperparameter sensitivity analysis (Section 4.3 supporting analysis):
  - Grid: n_estimators in {50, 100, 200, 300} x threshold in {0.05, 0.10, 0.15, 0.20}.
  - Both global and cluster pipelines swept (128 total runs).
  - Optimal configuration selected by minimum pseudo_rmse per (subset, pipeline).
  - gap_cycles = pseudo_rmse - true_rmse (pseudo-label vs supervised gap, RQ2).
Generates: results/tables/sensitivity_full_sweep.csv
           results/tables/sensitivity_optimal_configs.csv
"""

import importlib.util
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from cmapss.config import RESULTS_DIR, SUBSETS

_SRC = Path(__file__).parent

N_ESTIMATORS_GRID = [50, 100, 200, 300]
THRESHOLD_GRID    = [0.05, 0.10, 0.15, 0.20]

CANONICAL_OPTIMAL = {
    "global": {
        "FD001": {"n_estimators": 200, "threshold": 0.20, "pseudo_rmse": 31.68},
        "FD002": {"n_estimators":  50, "threshold": 0.10, "pseudo_rmse": 47.79},
        "FD003": {"n_estimators": 200, "threshold": 0.20, "pseudo_rmse": 29.73},
        "FD004": {"n_estimators": 300, "threshold": 0.10, "pseudo_rmse": 47.12},
    },
    "cluster": {
        "FD001": {"n_estimators": 200, "threshold": 0.20, "pseudo_rmse": 31.68},
        "FD002": {"n_estimators": 200, "threshold": 0.15, "pseudo_rmse": 42.29},
        "FD003": {"n_estimators": 200, "threshold": 0.20, "pseudo_rmse": 29.73},
        "FD004": {"n_estimators": 300, "threshold": 0.10, "pseudo_rmse": 42.14},
    },
}


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, _SRC / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_sweep(pipeline: str = "global") -> pd.DataFrame:
    if pipeline == "global":
        mod = _load_module("global_pipe", "03a_global_pipeline.py")
        run_fn = mod.run_global_pipeline
    else:
        mod = _load_module("cluster_pipe", "03b_cluster_pipeline.py")
        run_fn = mod.run_cluster_pipeline

    n_total = len(SUBSETS) * len(N_ESTIMATORS_GRID) * len(THRESHOLD_GRID)
    counter = 0
    rows = []

    for subset in SUBSETS:
        for n_est in N_ESTIMATORS_GRID:
            for thr in THRESHOLD_GRID:
                counter += 1
                print(
                    f"  {counter:>3}/{n_total}: {subset} {pipeline} "
                    f"n={n_est} thr={thr:.2f}",
                    end=" ... ",
                    flush=True,
                )
                t0 = time.perf_counter()
                result = run_fn(subset, n_estimators=n_est, threshold=thr)
                elapsed = time.perf_counter() - t0
                pm = result["pseudo_metrics"]
                tm = result["true_metrics"]
                gap = pm["rmse"] - tm["rmse"]
                rows.append({
                    "subset":       subset,
                    "pipeline":     pipeline,
                    "n_estimators": n_est,
                    "threshold":    thr,
                    "pearson_r":    round(result["pearson_r"], 4),
                    "pseudo_rmse":  round(pm["rmse"], 4),
                    "pseudo_mae":   round(pm["mae"],  4),
                    "pseudo_nasa":  int(round(pm["nasa_score"])),
                    "true_rmse":    round(tm["rmse"], 4),
                    "true_mae":     round(tm["mae"],  4),
                    "true_nasa":    int(round(tm["nasa_score"])),
                    "gap_cycles":   round(gap, 4),
                    "gap_pct":      round(gap / tm["rmse"] * 100, 2) if tm["rmse"] > 0 else 0.0,
                })
                print(f"pseudo_rmse={pm['rmse']:.2f}  ({elapsed:.1f}s)")

    return pd.DataFrame(rows)


def find_optimal_config(sweep_df: pd.DataFrame, pipeline: str) -> pd.DataFrame:
    sub = sweep_df[sweep_df["pipeline"] == pipeline]
    idx = sub.groupby("subset")["pseudo_rmse"].idxmin()
    opt = sub.loc[idx, [
        "subset", "pipeline", "n_estimators", "threshold",
        "pearson_r", "pseudo_rmse", "true_rmse", "gap_cycles", "gap_pct",
    ]].reset_index(drop=True)
    return opt


if __name__ == "__main__":
    t_start = time.perf_counter()

    print("=" * 60)
    print("Running GLOBAL pipeline sweep (64 configs)...")
    print("=" * 60)
    global_df = run_sweep("global")

    print()
    print("=" * 60)
    print("Running CLUSTER pipeline sweep (64 configs)...")
    print("=" * 60)
    cluster_df = run_sweep("cluster")

    full_df = pd.concat([global_df, cluster_df], ignore_index=True)

    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(tables_dir / "sensitivity_full_sweep.csv", index=False)
    print(f"\nSaved: results/tables/sensitivity_full_sweep.csv ({len(full_df)} rows)")

    opt_global  = find_optimal_config(full_df, "global")
    opt_cluster = find_optimal_config(full_df, "cluster")
    opt_all     = pd.concat([opt_global, opt_cluster], ignore_index=True)
    opt_all.to_csv(tables_dir / "sensitivity_optimal_configs.csv", index=False)
    print("Saved: results/tables/sensitivity_optimal_configs.csv")

    # ── Print optimal tables ────────────────────────────────────────────────
    def print_opt_table(opt_df, title):
        print(f"\n{title}")
        hdr = (
            f"{'Subset':<8} {'n_est':>6} {'thr':>5}  "
            f"{'Pseudo_RMSE':>12} {'True_RMSE':>10} {'Gap_cycles':>11} {'Gap_pct':>8}"
        )
        print(hdr)
        print("-" * len(hdr))
        for _, row in opt_df.iterrows():
            print(
                f"{row['subset']:<8} {int(row['n_estimators']):>6} {row['threshold']:>5.2f}  "
                f"{row['pseudo_rmse']:>12.2f} {row['true_rmse']:>10.2f} "
                f"{row['gap_cycles']:>11.2f} {row['gap_pct']:>7.1f}%"
            )

    print_opt_table(
        opt_global,
        "GLOBAL PIPELINE - Optimal Configuration per Subset (Table 4.3 candidate)",
    )
    print_opt_table(
        opt_cluster,
        "CLUSTER PIPELINE - Optimal Configuration per Subset (Table 4.8 candidate)",
    )

    # ── Key finding ─────────────────────────────────────────────────────────
    rmse_min = opt_cluster["pseudo_rmse"].min()
    rmse_max = opt_cluster["pseudo_rmse"].max()
    gap_min  = opt_cluster["gap_cycles"].min()
    gap_max  = opt_cluster["gap_cycles"].max()
    gpct_min = opt_cluster["gap_pct"].min()
    gpct_max = opt_cluster["gap_pct"].max()
    print(
        f"\nKey finding: Cluster optimal Pseudo RMSE range: {rmse_min:.2f}-{rmse_max:.2f} cycles. "
        f"Gap range: {gap_min:.2f}-{gap_max:.2f} cycles ({gpct_min:.1f}%-{gpct_max:.1f}%)"
    )

    # ── Sanity check ─────────────────────────────────────────────────────────
    print("\n-- Sanity Check vs canonical optimal configs (tolerance +/-1%) " + "-" * 10)
    print(f"{'Pipeline':<9} {'Subset':<8} {'Metric':<18} {'Canonical':>10} {'Got':>10} {'Err%':>7}  Status")
    print("-" * 68)
    all_pass = True
    for pl_name, opt_df in [("global", opt_global), ("cluster", opt_cluster)]:
        canon_pl = CANONICAL_OPTIMAL[pl_name]
        for _, row in opt_df.iterrows():
            subset = row["subset"]
            canon  = canon_pl[subset]
            got_rmse = row["pseudo_rmse"]
            exp_rmse = canon["pseudo_rmse"]
            pct = abs(got_rmse - exp_rmse) / abs(exp_rmse) * 100
            status = "PASS" if pct <= 1.0 else "FAIL"
            if status == "FAIL":
                all_pass = False
            n_match  = int(row["n_estimators"]) == canon["n_estimators"]
            thr_match = abs(row["threshold"] - canon["threshold"]) < 1e-6
            config_note = "" if (n_match and thr_match) else (
                f"  [config: n={int(row['n_estimators'])}, thr={row['threshold']:.2f} "
                f"vs canonical n={canon['n_estimators']}, thr={canon['threshold']:.2f}]"
            )
            print(
                f"{pl_name:<9} {subset:<8} {'pseudo_rmse':<18} {exp_rmse:>10.2f} "
                f"{got_rmse:>10.2f} {pct:>6.2f}%  {status}{config_note}"
            )

    print()
    print("Overall:", "ALL PASS" if all_pass else "SOME CHECKS FAILED - see above")

    total_time = time.perf_counter() - t_start
    print(f"\nTotal runtime: {total_time/60:.1f} min")

