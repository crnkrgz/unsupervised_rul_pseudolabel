"""
fig_final_rmse.py
Figure: Pseudo vs True RMSE under optimal cluster config — paired bars with gap annotation.
Output: results/figures/final_rmse.png
"""

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

C_PSEUDO  = "#2E86AB"
C_TRUE    = "#E63946"
C_GAP     = "#06A77D"


def main():
    opt_csv = RESULTS_DIR / "tables" / "sensitivity_optimal_configs.csv"
    if not opt_csv.exists():
        raise FileNotFoundError("Run 04_sensitivity.py first to generate sensitivity_optimal_configs.csv")

    df = pd.read_csv(opt_csv)
    cluster_df = df[df["pipeline"] == "cluster"].set_index("subset")
    cluster_df = cluster_df.loc[SUBSETS]

    x     = np.arange(len(SUBSETS))
    width = 0.32

    fig, ax = plt.subplots(figsize=(9, 6))

    bars_p = ax.bar(x - width / 2, cluster_df["pseudo_rmse"], width,
                    label="Pseudo-label RMSE", color=C_PSEUDO, edgecolor="grey", linewidth=0.6)
    bars_t = ax.bar(x + width / 2, cluster_df["true_rmse"],   width,
                    label="True-label RMSE",   color=C_TRUE,   edgecolor="grey", linewidth=0.6)

    # Gap annotations above the taller bar of each pair
    max_val = cluster_df[["pseudo_rmse", "true_rmse"]].values.max()
    for i, subset in enumerate(SUBSETS):
        p_val = cluster_df.loc[subset, "pseudo_rmse"]
        t_val = cluster_df.loc[subset, "true_rmse"]
        gap   = p_val - t_val
        pct   = gap / t_val * 100
        top   = max(p_val, t_val)
        ax.text(
            x[i], top + max_val * 0.02,
            f"Gap: {gap:.2f}\n({pct:.1f}% of true)",
            fontsize=8, color=C_GAP, ha="center", va="bottom", fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(SUBSETS, fontsize=11)
    ax.set_ylabel("RMSE (cycles)", fontsize=11)
    ax.set_title(
        "Final RMSE — Optimal Cluster Config: Pseudo-label vs True-label Reference",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(0, max_val * 1.28)

    plt.tight_layout()

    out = RESULTS_DIR / "figures" / "final_rmse.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()

