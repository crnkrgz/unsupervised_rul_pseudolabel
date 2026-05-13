"""
fig_correlation_scatter.py
Figure: Isolation Forest anomaly score vs. true RUL scatter — all four subsets.
Output: results/figures/correlation_scatter.png
"""

import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
from cmapss.config import RESULTS_DIR, SUBSETS

_SRC = ROOT / "src"
_PRED_DIR = RESULTS_DIR / "predictions"
_PRED_DIR.mkdir(parents=True, exist_ok=True)


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, _SRC / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_train_scores(subset: str) -> pd.DataFrame:
    cache = _PRED_DIR / f"global_default_{subset}_train_scores.csv"
    if cache.exists():
        return pd.read_csv(cache)

    load_mod = _load_module("load_data", "01_load_data.py")
    prep_mod = _load_module("preprocess", "02_preprocess.py")
    pipe_mod = _load_module("global_pipe", "03a_global_pipeline.py")

    raw    = load_mod.load_subset(subset)
    result = prep_mod.preprocess_subset(raw, subset)
    train_df     = result["train"]
    feature_cols = result["feature_cols"]

    X_train      = train_df[feature_cols].values
    clf          = pipe_mod.fit_isolation_forest(X_train)
    raw_scores   = pipe_mod.score_anomaly(clf, X_train)
    smooth_train = pipe_mod.smooth_scores_per_unit(train_df, raw_scores)

    df = pd.DataFrame({
        "anomaly_score": smooth_train,
        "true_RUL":      train_df["true_RUL"].values,
        "cycle":         train_df["cycle"].values,
    })
    df.to_csv(cache, index=False)
    print(f"  Saved train scores: {cache.name}")
    return df


def main():
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax, subset in zip(axes.flat, SUBSETS):
        print(f"Processing {subset}...")
        df = get_train_scores(subset)

        r = np.corrcoef(df["anomaly_score"], df["true_RUL"])[0, 1]

        sc = ax.scatter(
            df["anomaly_score"], df["true_RUL"],
            c=df["cycle"], cmap="viridis",
            alpha=0.5, s=2, rasterized=True,
        )
        plt.colorbar(sc, ax=ax, label="Cycle")
        ax.set_title(f"{subset}  (r = {r:.4f})", fontsize=11)
        ax.set_xlabel("Anomaly Score (higher = more degraded)")
        ax.set_ylabel("True RUL (cycles)")

    fig.suptitle(
        "Isolation Forest Anomaly Score vs. True RUL — All Subsets",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()

    out = RESULTS_DIR / "figures" / "correlation_scatter.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()

