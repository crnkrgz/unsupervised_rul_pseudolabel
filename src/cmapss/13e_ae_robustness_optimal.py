"""
13e_ae_robustness_optimal.py
Seed robustness test with optimal per-subset AE configs.
Seeds: 42-46 (5 seeds) x FD001 + FD002 = 10 runs.

Optimal configs:
  FD001: latent_dim=2, n_epochs=50
  FD002: latent_dim=4, n_epochs=30

Generates: results/tables/ae_robustness_optimal.csv
           results/tables/ae_robustness_optimal_summary.csv
"""

import importlib.util
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from cmapss.config import RESULTS_DIR

_SRC = Path(__file__).parent

OPTIMAL_AE_CONFIG = {
    "FD001": {"latent_dim": 2, "n_epochs": 50},
    "FD002": {"latent_dim": 4, "n_epochs": 30},
}
INSTABILITY_THRESHOLD = 5.0


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, _SRC / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


if __name__ == "__main__":
    ae_mod = _load_module("ae_pipe", "13_autoencoder_pipeline.py")

    SUBSETS = ["FD001", "FD002"]
    SEEDS   = [42, 43, 44, 45, 46]

    rows = []
    total = len(SUBSETS) * len(SEEDS)
    done  = 0

    for subset in SUBSETS:
        cfg = OPTIMAL_AE_CONFIG[subset]
        for seed in SEEDS:
            done += 1
            print(f"[{done:2d}/{total}] {subset}  ld={cfg['latent_dim']}  ep={cfg['n_epochs']}  seed={seed} ...", flush=True)
            r  = ae_mod.run_ae_pipeline(subset, seed=seed, **cfg)
            pm = r["pseudo_metrics"]
            rows.append({
                "subset":        subset,
                "latent_dim":    cfg["latent_dim"],
                "n_epochs":      cfg["n_epochs"],
                "seed":          seed,
                "pseudo_rmse":   round(pm["rmse"], 2),
                "corr_with_rul": round(r["pearson_r"], 4),
            })
            print(f"        pseudo_rmse={pm['rmse']:.2f}  corr={r['pearson_r']:.4f}")

    df = pd.DataFrame(rows)

    summary_rows = []
    for subset in SUBSETS:
        sub  = df[df["subset"] == subset]
        rmse = sub["pseudo_rmse"]
        corr = sub["corr_with_rul"]
        summary_rows.append({
            "subset":    subset,
            "latent_dim": OPTIMAL_AE_CONFIG[subset]["latent_dim"],
            "n_epochs":   OPTIMAL_AE_CONFIG[subset]["n_epochs"],
            "mean_rmse": round(float(rmse.mean()), 2),
            "std_rmse":  round(float(rmse.std()),  2),
            "min_rmse":  round(float(rmse.min()),  2),
            "max_rmse":  round(float(rmse.max()),  2),
            "mean_corr": round(float(corr.mean()), 4),
            "std_corr":  round(float(corr.std()),  4),
        })

    df_summary = pd.DataFrame(summary_rows)

    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables_dir / "ae_robustness_optimal.csv",         index=False)
    df_summary.to_csv(tables_dir / "ae_robustness_optimal_summary.csv", index=False)

    print(f"\n{'='*65}")
    print("AE Robustness (Optimal Config) -- 5 seeds, FD001 + FD002")
    print(f"{'='*65}")
    print("\nPer-run:")
    print(df.to_string(index=False))
    print("\nSummary:")
    print(df_summary.to_string(index=False))

    print(f"\n{'-'*55}")
    for row in summary_rows:
        status = "UNSTABLE" if row["std_rmse"] > INSTABILITY_THRESHOLD else "stable"
        print(f"  {row['subset']} (ld={row['latent_dim']}, ep={row['n_epochs']}): "
              f"mean={row['mean_rmse']:.2f}  std={row['std_rmse']:.2f}  "
              f"corr={row['mean_corr']:.4f}+/-{row['std_corr']:.4f}  -> {status}")
        if row["std_rmse"] > INSTABILITY_THRESHOLD:
            print(f"    WARNING: RMSE std > {INSTABILITY_THRESHOLD} cycles")

    print(f"\nSaved: ae_robustness_optimal.csv + ae_robustness_optimal_summary.csv")
