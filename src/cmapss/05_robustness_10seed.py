"""
05_robustness_10seed.py
Statistical robustness analysis (Section 4.3 / Table 4.4):
  - Compares optimal config vs worst config (n_est=50, thr=0.05) across
    10 independent random seeds for IsolationForest and KMeans.
  - Cluster pipeline only; 4 subsets x 10 seeds x 2 configs = 80 runs.
  - Independent samples t-test (scipy.stats.ttest_ind) per subset.
  - Reports: mean RMSE, SD, 95% CI, t-statistic, p-value.
Generates: results/tables/robustness_10seed.csv
           results/tables/table_4_4_significance.csv
"""

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

sys.path.insert(0, str(Path(__file__).parent.parent))
from cmapss.config import RESULTS_DIR, SUBSETS

_SRC = Path(__file__).parent

SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
WORST_N_ESTIMATORS = 50
WORST_THRESHOLD    = 0.05


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, _SRC / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_with_seed(
    subset_name: str,
    n_estimators: int,
    threshold: float,
    seed: int,
) -> float:
    mod = _load_module("cluster_pipe", "03b_cluster_pipeline.py")
    result = mod.run_cluster_pipeline(
        subset_name,
        n_estimators=n_estimators,
        threshold=threshold,
        seed=seed,
    )
    return result["pseudo_metrics"]["rmse"]


if __name__ == "__main__":
    t_start = time.perf_counter()

    # Load optimal configs for cluster pipeline
    opt_csv = RESULTS_DIR / "tables" / "sensitivity_optimal_configs.csv"
    if not opt_csv.exists():
        raise FileNotFoundError(
            "Run 04_sensitivity.py first to generate sensitivity_optimal_configs.csv"
        )
    opt_df = pd.read_csv(opt_csv)
    opt_df = opt_df[opt_df["pipeline"] == "cluster"].set_index("subset")

    all_rows   = []
    stat_rows  = []
    run_total  = len(SUBSETS) * len(SEEDS) * 2
    run_count  = 0

    print("ROBUSTNESS — 10-seed Optimal vs Worst Configuration (Cluster Pipeline)")
    print("=" * 72)

    for subset in SUBSETS:
        opt_n   = int(opt_df.loc[subset, "n_estimators"])
        opt_thr = float(opt_df.loc[subset, "threshold"])

        opt_rmses  = []
        wst_rmses  = []

        for seed in SEEDS:
            run_count += 1
            print(
                f"  {run_count:>2}/{run_total}: {subset} optimal  "
                f"n={opt_n} thr={opt_thr:.2f} seed={seed}",
                end=" ... ", flush=True,
            )
            t0  = time.perf_counter()
            rmse = run_with_seed(subset, opt_n, opt_thr, seed)
            opt_rmses.append(rmse)
            print(f"{rmse:.2f}  ({time.perf_counter()-t0:.1f}s)")
            all_rows.append({
                "subset": subset, "pipeline": "cluster",
                "config_type": "optimal",
                "n_estimators": opt_n, "threshold": opt_thr,
                "seed": seed, "pseudo_rmse": round(rmse, 4),
            })

            run_count += 1
            print(
                f"  {run_count:>2}/{run_total}: {subset} worst   "
                f"n={WORST_N_ESTIMATORS} thr={WORST_THRESHOLD:.2f} seed={seed}",
                end=" ... ", flush=True,
            )
            t0   = time.perf_counter()
            rmse = run_with_seed(subset, WORST_N_ESTIMATORS, WORST_THRESHOLD, seed)
            wst_rmses.append(rmse)
            print(f"{rmse:.2f}  ({time.perf_counter()-t0:.1f}s)")
            all_rows.append({
                "subset": subset, "pipeline": "cluster",
                "config_type": "worst",
                "n_estimators": WORST_N_ESTIMATORS, "threshold": WORST_THRESHOLD,
                "seed": seed, "pseudo_rmse": round(rmse, 4),
            })

        opt_arr = np.array(opt_rmses)
        wst_arr = np.array(wst_rmses)
        n       = len(SEEDS)
        se_opt  = opt_arr.std(ddof=1) / np.sqrt(n)
        se_wst  = wst_arr.std(ddof=1) / np.sqrt(n)
        t_stat, p_val = ttest_ind(opt_arr, wst_arr, equal_var=False)

        stat_rows.append({
            "subset":          subset,
            "opt_mean":        round(float(opt_arr.mean()), 4),
            "opt_sd":          round(float(opt_arr.std(ddof=1)), 4),
            "opt_ci_low":      round(float(opt_arr.mean() - 1.96 * se_opt), 4),
            "opt_ci_high":     round(float(opt_arr.mean() + 1.96 * se_opt), 4),
            "wst_mean":        round(float(wst_arr.mean()), 4),
            "wst_sd":          round(float(wst_arr.std(ddof=1)), 4),
            "wst_ci_low":      round(float(wst_arr.mean() - 1.96 * se_wst), 4),
            "wst_ci_high":     round(float(wst_arr.mean() + 1.96 * se_wst), 4),
            "t_stat":          round(float(t_stat), 4),
            "p_value":         round(float(p_val), 6),
            "significant_at_05": bool(p_val < 0.05),
        })

    # Save raw per-seed results
    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(tables_dir / "robustness_10seed.csv", index=False)
    print(f"\nSaved: results/tables/robustness_10seed.csv ({len(all_rows)} rows)")

    stat_df = pd.DataFrame(stat_rows)
    stat_df.to_csv(tables_dir / "table_4_4_significance.csv", index=False)
    print("Saved: results/tables/table_4_4_significance.csv")

    # Print summary table
    print()
    print("ROBUSTNESS - 10-seed Optimal vs Worst Configuration (Cluster Pipeline)")
    print("=" * 95)
    hdr = (
        f"{'Subset':<8} {'Opt Mean+/-SD':<18} {'95% CI':<22} "
        f"{'Wst Mean+/-SD':<18} {'95% CI':<22} {'t':>7} {'p':>8} {'Sig':>4}"
    )
    print(hdr)
    print("-" * 95)
    for r in stat_rows:
        opt_str = f"{r['opt_mean']:.2f}+/-{r['opt_sd']:.2f}"
        opt_ci  = f"[{r['opt_ci_low']:.2f},{r['opt_ci_high']:.2f}]"
        wst_str = f"{r['wst_mean']:.2f}+/-{r['wst_sd']:.2f}"
        wst_ci  = f"[{r['wst_ci_low']:.2f},{r['wst_ci_high']:.2f}]"
        sig     = "YES" if r["significant_at_05"] else "NO"
        print(
            f"{r['subset']:<8} {opt_str:<18} {opt_ci:<22} "
            f"{wst_str:<18} {wst_ci:<22} {r['t_stat']:>7.3f} {r['p_value']:>8.4f} {sig:>4}"
        )

    total = time.perf_counter() - t_start
    print(f"\nTotal runtime: {total/60:.1f} min")

