"""
fig_sensitivity_sweep.py
Figure: Threshold vs pseudo-RMSE for each n_estimators — global pipeline, all 4 subsets.
Output: results/figures/sensitivity_sweep.png
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
from cmapss.config import RESULTS_DIR, SUBSETS

PALETTE = {50: "#e41a1c", 100: "#ff7f00", 200: "#377eb8", 300: "#4daf4a"}
MARKERS = {50: "o", 100: "s", 200: "^", 300: "D"}


def main():
    sweep_csv = RESULTS_DIR / "tables" / "sensitivity_full_sweep.csv"
    if not sweep_csv.exists():
        raise FileNotFoundError("Run 04_sensitivity.py first to generate sensitivity_full_sweep.csv")

    df = pd.read_csv(sweep_csv)
    global_df = df[df["pipeline"] == "global"].copy()

    opt_csv = RESULTS_DIR / "tables" / "sensitivity_optimal_configs.csv"
    opt_df = pd.read_csv(opt_csv)
    opt_global = opt_df[opt_df["pipeline"] == "global"].set_index("subset")

    n_values = sorted(global_df["n_estimators"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=False)

    for ax, subset in zip(axes.flat, SUBSETS):
        sub = global_df[global_df["subset"] == subset]
        for n in n_values:
            grp = sub[sub["n_estimators"] == n].sort_values("threshold")
            ax.plot(
                grp["threshold"], grp["pseudo_rmse"],
                color=PALETTE[n], marker=MARKERS[n],
                linewidth=1.8, markersize=6, label=f"n={n}",
            )

        # Mark optimal
        opt_row = opt_global.loc[subset]
        opt_n   = int(opt_row["n_estimators"])
        opt_thr = float(opt_row["threshold"])
        opt_rmse = float(opt_row["pseudo_rmse"])
        ax.scatter(
            opt_thr, opt_rmse,
            marker="*", s=220, color="black", zorder=5,
            label=f"Optimal (n={opt_n}, thr={opt_thr:.2f})",
        )

        ax.set_title(subset, fontsize=11)
        ax.set_xlabel("Anomaly Threshold")
        ax.set_ylabel("Pseudo RMSE")
        ax.legend(fontsize=7.5, loc="upper left")
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(
        "Sensitivity Sweep — Global Pipeline: Threshold vs Pseudo RMSE",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()

    out = RESULTS_DIR / "figures" / "sensitivity_sweep.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()

