"""
fig_cluster_impact.py
Figure: Global vs Cluster pipeline RMSE comparison — grouped bar chart.
Output: results/figures/cluster_impact.png
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

C_GLOBAL   = "#A8DADC"
C_CLUSTER  = "#1D3557"
C_TRUE     = "#E63946"
C_ANNOT    = "#06A77D"


def main():
    comp_csv = RESULTS_DIR / "tables" / "global_vs_cluster_comparison.csv"
    if not comp_csv.exists():
        raise FileNotFoundError("Run 03b_cluster_pipeline.py first to generate global_vs_cluster_comparison.csv")

    df = pd.read_csv(comp_csv).set_index("subset")
    df = df.loc[SUBSETS]

    x     = np.arange(len(SUBSETS))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    bars_g = ax.bar(x - width, df["pseudo_rmse_global"],  width, label="Global (pseudo)", color=C_GLOBAL,  edgecolor="grey", linewidth=0.6)
    bars_c = ax.bar(x,         df["pseudo_rmse_cluster"], width, label="Cluster (pseudo)", color=C_CLUSTER, edgecolor="grey", linewidth=0.6)
    bars_t = ax.bar(x + width, df["true_rmse_cluster"],   width, label="Cluster (true ref)", color=C_TRUE,  edgecolor="grey", linewidth=0.6, alpha=0.85)

    # Annotations per subset
    max_bar = max(df["pseudo_rmse_global"].max(), df["pseudo_rmse_cluster"].max(), df["true_rmse_cluster"].max())
    for i, subset in enumerate(SUBSETS):
        group_top = max(
            df.loc[subset, "pseudo_rmse_global"],
            df.loc[subset, "pseudo_rmse_cluster"],
            df.loc[subset, "true_rmse_cluster"],
        )
        if subset in ("FD002", "FD004"):
            delta = df.loc[subset, "pseudo_rmse_delta"]
            pct   = df.loc[subset, "pseudo_rmse_pct"]
            ax.text(
                x[i], group_top + max_bar * 0.03,
                f"Δ={delta:+.2f}\n({pct:+.1f}%)",
                fontsize=8.5, color=C_ANNOT, fontweight="bold",
                ha="center", va="bottom",
            )
        else:
            ax.text(
                x[i], group_top + max_bar * 0.03,
                "k=1\n(no change)",
                fontsize=8, color="grey",
                ha="center", va="bottom",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(SUBSETS, fontsize=11)
    ax.set_ylabel("RMSE (cycles)", fontsize=11)
    ax.set_title("Impact of Cluster-Based Scaling on Pseudo-Label RMSE", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(0, max_bar * 1.28)

    plt.tight_layout()

    out = RESULTS_DIR / "figures" / "cluster_impact.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()

