"""
14b_if_vs_ae_optimal.py
IF vs AE comparison using per-subset optimal AE configs.

Optimal configs determined by max |Pearson r| across sensitivity grid:
  FD001: latent_dim=2, n_epochs=50  (corr=-0.469, from ae_sensitivity.csv)
  FD002: latent_dim=4, n_epochs=30  (corr=-0.537, from ae_sensitivity.csv)
  FD003: latent_dim=2, n_epochs=50  (corr=-0.694, from ae_sensitivity_extension.csv)
  FD004: latent_dim=2, n_epochs=50  (corr=-0.737, from ae_sensitivity_extension.csv)

IF baseline: run_cluster_pipeline with default config (Table 4.8 consistent).

Generates: results/tables/tablo_if_vs_ae_optimal.csv
"""

import importlib.util
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from cmapss.config import RESULTS_DIR, SEED

_SRC = Path(__file__).parent

OPTIMAL_AE_CONFIG = {
    "FD001": {"latent_dim": 2, "n_epochs": 50},
    "FD002": {"latent_dim": 4, "n_epochs": 30},
    "FD003": {"latent_dim": 2, "n_epochs": 50},
    "FD004": {"latent_dim": 2, "n_epochs": 50},
}


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, _SRC / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


if __name__ == "__main__":
    clust_mod = _load_module("cluster_pipe", "03b_cluster_pipeline.py")
    ae_mod    = _load_module("ae_pipe",      "13_autoencoder_pipeline.py")

    subsets = ["FD001", "FD002", "FD003", "FD004"]
    rows = []

    for subset in subsets:
        cfg = OPTIMAL_AE_CONFIG[subset]
        print(f"\n{'-'*55}")
        print(f"[{subset}] IF pipeline...")
        if_r  = clust_mod.run_cluster_pipeline(subset)
        if_pm = if_r["pseudo_metrics"]
        if_tm = if_r["true_metrics"]
        rows.append({
            "subset":        subset,
            "model":         "IF",
            "latent_dim":    None,
            "n_epochs":      None,
            "pseudo_rmse":   round(if_pm["rmse"], 2),
            "true_rmse":     round(if_tm["rmse"], 2),
            "gap":           round(if_pm["rmse"] - if_tm["rmse"], 2),
            "corr_with_rul": round(if_r["pearson_r"], 4),
        })
        print(f"  IF   pseudo_rmse={if_pm['rmse']:.2f}  true_rmse={if_tm['rmse']:.2f}  r={if_r['pearson_r']:.4f}")

        print(f"[{subset}] AE pipeline (latent_dim={cfg['latent_dim']}, n_epochs={cfg['n_epochs']})...")
        ae_r  = ae_mod.run_ae_pipeline(subset, seed=SEED, **cfg)
        ae_pm = ae_r["pseudo_metrics"]
        ae_tm = ae_r["true_metrics"]
        rows.append({
            "subset":        subset,
            "model":         "AE",
            "latent_dim":    cfg["latent_dim"],
            "n_epochs":      cfg["n_epochs"],
            "pseudo_rmse":   round(ae_pm["rmse"], 2),
            "true_rmse":     round(ae_tm["rmse"], 2),
            "gap":           round(ae_pm["rmse"] - ae_tm["rmse"], 2),
            "corr_with_rul": round(ae_r["pearson_r"], 4),
        })
        print(f"  AE   pseudo_rmse={ae_pm['rmse']:.2f}  true_rmse={ae_tm['rmse']:.2f}  r={ae_r['pearson_r']:.4f}")

    df = pd.DataFrame(rows)

    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    out = tables_dir / "tablo_if_vs_ae_optimal.csv"
    df.to_csv(out, index=False)

    print(f"\n{'='*65}")
    print("IF vs AE (Optimal AE Config) -- All Subsets")
    print(f"{'='*65}")

    display_cols = ["subset", "model", "pseudo_rmse", "true_rmse", "gap", "corr_with_rul"]
    print(df[display_cols].to_string(index=False))

    print(f"\nOptimal AE configs used:")
    for subset, cfg in OPTIMAL_AE_CONFIG.items():
        print(f"  {subset}: latent_dim={cfg['latent_dim']}  n_epochs={cfg['n_epochs']}")

    print(f"\nSaved: {out}")
