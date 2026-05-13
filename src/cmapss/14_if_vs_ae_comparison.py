"""
14_if_vs_ae_comparison.py
Side-by-side comparison of IF cluster pipeline (03b) vs AE pipeline (13)
for all four C-MAPSS subsets (FD001-FD004).

IF baseline: run_cluster_pipeline (k=1 for single-condition, k=6 for multi).
AE baseline: run_ae_pipeline (same cluster scaling as IF).

Generates: results/tables/tablo_if_vs_ae.csv

Output columns:
  subset | model | pseudo_rmse | true_rmse | gap | corr_with_rul

Usage:
  python src/cmapss/14_if_vs_ae_comparison.py
"""

import importlib.util
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from cmapss.config import RESULTS_DIR

_SRC = Path(__file__).parent


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
        print(f"\n{'-'*50}")
        print(f"Running IF pipeline for {subset}...")
        if_r  = clust_mod.run_cluster_pipeline(subset)
        if_pm = if_r["pseudo_metrics"]
        if_tm = if_r["true_metrics"]
        rows.append({
            "subset":        subset,
            "model":         "IF",
            "pseudo_rmse":   round(if_pm["rmse"], 2),
            "true_rmse":     round(if_tm["rmse"], 2),
            "gap":           round(if_pm["rmse"] - if_tm["rmse"], 2),
            "corr_with_rul": round(if_r["pearson_r"], 4),
        })
        print(f"  IF  pseudo_rmse={if_pm['rmse']:.2f}  true_rmse={if_tm['rmse']:.2f}  r={if_r['pearson_r']:.4f}")

        print(f"Running AE pipeline for {subset}...")
        ae_r  = ae_mod.run_ae_pipeline(subset)
        ae_pm = ae_r["pseudo_metrics"]
        ae_tm = ae_r["true_metrics"]
        rows.append({
            "subset":        subset,
            "model":         "AE",
            "pseudo_rmse":   round(ae_pm["rmse"], 2),
            "true_rmse":     round(ae_tm["rmse"], 2),
            "gap":           round(ae_pm["rmse"] - ae_tm["rmse"], 2),
            "corr_with_rul": round(ae_r["pearson_r"], 4),
        })
        print(f"  AE  pseudo_rmse={ae_pm['rmse']:.2f}  true_rmse={ae_tm['rmse']:.2f}  r={ae_r['pearson_r']:.4f}")

    df = pd.DataFrame(rows)

    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    out = tables_dir / "tablo_if_vs_ae.csv"
    df.to_csv(out, index=False)

    print(f"\n{'='*65}")
    print("IF vs AE Comparison -- All Subsets")
    print(f"{'='*65}")
    print(df.to_string(index=False))
    print(f"\nSaved: {out}")

    # -- Validation checks ----------------------------------------------------
    print(f"\n{'-'*50}")
    print("AE Validation Checks")
    print(f"{'-'*50}")
    for subset in subsets:
        ae_row = next(r for r in rows if r["subset"] == subset and r["model"] == "AE")
        corr   = ae_row["corr_with_rul"]
        rmse   = ae_row["pseudo_rmse"]
        print(f"\n{subset}:")
        print(f"  Pearson r (rec_error vs true_RUL): {corr:.4f}", end="")
        if corr > 0:
            print("  <- WARNING: positive -- flip rec_error sign or threshold direction")
        elif -0.7 <= corr <= -0.4:
            print("  <- OK (expected [-0.4, -0.7])")
        else:
            print(f"  <- outside expected [-0.4, -0.7]")

        print(f"  Pseudo RMSE: {rmse:.2f}", end="")
        if rmse > 50 and subset == "FD001":
            print("  <- WARNING: >50 on FD001 -- check AE training")
        elif rmse <= 0:
            print("  <- ERROR: RMSE <= 0")
        else:
            print("  <- OK")
