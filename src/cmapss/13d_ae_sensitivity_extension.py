"""
13d_ae_sensitivity_extension.py
Short sensitivity extension for FD003 and FD004.
Grid: latent_dim in {2, 4} x n_epochs=50, seed=42.

Complements ae_sensitivity.csv (which covered FD001+FD002).
Generates: results/tables/ae_sensitivity_extension.csv
"""

import importlib.util
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from cmapss.config import RESULTS_DIR, SEED

_SRC = Path(__file__).parent


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, _SRC / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


if __name__ == "__main__":
    ae_mod = _load_module("ae_pipe", "13_autoencoder_pipeline.py")

    CONFIGS = [
        ("FD003", 2, 50),
        ("FD003", 4, 50),
        ("FD004", 2, 50),
        ("FD004", 4, 50),
    ]

    rows = []
    for i, (subset, ld, ne) in enumerate(CONFIGS, 1):
        print(f"[{i}/{len(CONFIGS)}] {subset}  latent_dim={ld}  n_epochs={ne} ...", flush=True)
        r = ae_mod.run_ae_pipeline(subset, n_epochs=ne, latent_dim=ld, seed=SEED)
        pm = r["pseudo_metrics"]
        rows.append({
            "subset":      subset,
            "latent_dim":  ld,
            "n_epochs":    ne,
            "pseudo_rmse": round(pm["rmse"], 2),
            "true_rmse":   round(r["true_metrics"]["rmse"], 2),
            "corr":        round(r["pearson_r"], 4),
        })
        print(f"        pseudo_rmse={pm['rmse']:.2f}  corr={r['pearson_r']:.4f}")

    df = pd.DataFrame(rows)

    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    out = tables_dir / "ae_sensitivity_extension.csv"
    df.to_csv(out, index=False)

    print(f"\n{'='*55}")
    print("AE Sensitivity Extension -- FD003 + FD004")
    print(f"{'='*55}")
    print(df[["subset", "latent_dim", "n_epochs", "pseudo_rmse", "corr"]].to_string(index=False))

    print("\nOptimal config (best corr per subset):")
    for subset in ["FD003", "FD004"]:
        sub = df[df["subset"] == subset]
        best = sub.loc[sub["corr"].idxmin()]
        print(f"  {subset}: latent_dim={int(best['latent_dim'])}  "
              f"pseudo_rmse={best['pseudo_rmse']:.2f}  corr={best['corr']:.4f}")

    print(f"\nSaved: {out}")
