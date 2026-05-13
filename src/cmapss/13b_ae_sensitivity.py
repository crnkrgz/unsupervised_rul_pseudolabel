"""
13b_ae_sensitivity.py
Hyperparameter sensitivity for the AE pipeline.
Grid: latent_dim in {2, 4, 8} x n_epochs in {30, 50, 100} = 9 combinations.
Subsets: FD001 + FD002.
Baseline config: latent_dim=4, n_epochs=50 (matches run_ae_pipeline defaults).

Generates: results/tables/ae_sensitivity.csv

Usage:
  python src/cmapss/13b_ae_sensitivity.py
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

    SUBSETS = ["FD001", "FD002"]
    LATENT_DIMS = [2, 4, 8]
    EPOCHS = [30, 50, 100]
    BASELINE_LD, BASELINE_EP = 4, 50
    IMPROVEMENT_THRESHOLD = 2.0  # cycles -- flag if best beats baseline by more than this

    rows = []
    total = len(SUBSETS) * len(LATENT_DIMS) * len(EPOCHS)
    done = 0

    for subset in SUBSETS:
        for ld in LATENT_DIMS:
            for ne in EPOCHS:
                done += 1
                print(f"[{done:2d}/{total}] {subset}  latent_dim={ld}  n_epochs={ne:3d} ...", flush=True)
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
    out = tables_dir / "ae_sensitivity.csv"
    df.to_csv(out, index=False)

    print(f"\n{'='*65}")
    print("AE Sensitivity Grid -- FD001 + FD002")
    print(f"{'='*65}")

    flag_rerun = False
    for subset in SUBSETS:
        sub = df[df["subset"] == subset].copy().reset_index(drop=True)
        print(f"\n{subset}:")
        print(sub[["latent_dim", "n_epochs", "pseudo_rmse", "corr"]].to_string(index=False))

        bl = sub[(sub["latent_dim"] == BASELINE_LD) & (sub["n_epochs"] == BASELINE_EP)].iloc[0]
        best_idx = sub["pseudo_rmse"].idxmin()
        best = sub.loc[best_idx]
        diff = float(best["pseudo_rmse"]) - float(bl["pseudo_rmse"])

        print(f"\n  Baseline (ld={BASELINE_LD}, ep={BASELINE_EP}): pseudo_rmse={bl['pseudo_rmse']:.2f}")
        print(f"  Best    (ld={int(best['latent_dim'])}, ep={int(best['n_epochs'])}): "
              f"pseudo_rmse={best['pseudo_rmse']:.2f}  (diff={diff:+.2f} cycles)")
        if abs(diff) > IMPROVEMENT_THRESHOLD:
            print(f"  NOTE: best beats baseline by {abs(diff):.2f} cycles (>{IMPROVEMENT_THRESHOLD}) -- "
                  f"consider rerunning tablo_if_vs_ae with optimal config")
            flag_rerun = True
        else:
            print(f"  Baseline is adequate (diff within {IMPROVEMENT_THRESHOLD}-cycle threshold)")

    if flag_rerun:
        print("\nACTION: At least one subset has a better config -- check above and decide.")
    else:
        print("\nConclusion: Default (latent_dim=4, n_epochs=50) is adequate for all subsets.")

    print(f"\nSaved: {out}")
