"""
13_autoencoder_pipeline.py
Autoencoder-based pseudo-label pipeline -- drop-in alternative to IF in 03a/03b.

Architecture: input->8->latent_dim->8->input (ReLU hidden, linear output, MSE loss)
  FD001/FD003 (16 features): 16->8->4->8->16
  FD002/FD004 (17 features): 17->8->4->8->17

Anomaly proxy: L2 reconstruction error per sample -- replaces IF anomaly score.
Remaining steps (smoothing, pseudo-labels, Ridge) are identical to 03b.
Cluster-based scaling (k=6 for FD002/FD004) is supported.

run_ae_pipeline(subset_name, ...) -> dict  (same shape as run_cluster_pipeline)

Generates (when run as __main__):
  results/tables/ae_baseline_results.csv  (FD001 + FD002)
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from cmapss.config import (
    KMEANS_K_MULTI,
    MULTI_CONDITION,
    RESULTS_DIR,
    SEED,
    SINGLE_CONDITION,
    THRESHOLD_MULTI,
    THRESHOLD_SINGLE,
)

_SRC = Path(__file__).parent


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, _SRC / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# -- Autoencoder ---------------------------------------------------------------

class _Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8), nn.ReLU(),
            nn.Linear(8, latent_dim), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8), nn.ReLU(),
            nn.Linear(8, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def _train_autoencoder(
    X: np.ndarray,
    latent_dim: int = 4,
    n_epochs: int = 50,
    lr: float = 1e-3,
    seed: int = SEED,
) -> _Autoencoder:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = _Autoencoder(input_dim=X.shape[1], latent_dim=latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    dataset = TensorDataset(torch.FloatTensor(X))
    loader = DataLoader(dataset, batch_size=256, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    model.train()
    for _ in range(n_epochs):
        for (batch,) in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def _score_ae(model: _Autoencoder, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        x_t = torch.FloatTensor(X)
        x_hat = model(x_t)
        return torch.norm(x_t - x_hat, dim=1).numpy().astype(float)


# -- Main pipeline -------------------------------------------------------------

def run_ae_pipeline(
    subset_name: str,
    n_epochs: int = 50,
    latent_dim: int = 4,
    threshold: float = None,
    seed: int = None,
) -> dict:
    name = subset_name.upper()
    if seed is None:
        seed = SEED
    if threshold is None:
        threshold = THRESHOLD_SINGLE if name in SINGLE_CONDITION else THRESHOLD_MULTI
    k = 1 if name in SINGLE_CONDITION else KMEANS_K_MULTI

    load_mod  = _load_module("load_data",    "01_load_data.py")
    prep_mod  = _load_module("preprocess",   "02_preprocess.py")
    pipe_mod  = _load_module("global_pipe",  "03a_global_pipeline.py")
    clust_mod = _load_module("cluster_pipe", "03b_cluster_pipeline.py")

    raw    = load_mod.load_subset(name)
    result = prep_mod.preprocess_subset(raw, name, scale=False)

    train_df     = result["train"]
    test_df      = result["test"]
    feature_cols = result["feature_cols"]

    # Cluster-based scaling -- same as 03b
    train_df, test_df, _km     = clust_mod.cluster_op_settings(train_df, test_df, k, seed=seed)
    train_df, test_df, scalers = clust_mod.per_cluster_minmax(train_df, test_df, feature_cols, k)

    cluster_sizes = {int(c): int((train_df["cluster"] == c).sum()) for c in range(k)}

    X_train = train_df[feature_cols].values.astype(np.float32)

    # Train autoencoder
    ae_model = _train_autoencoder(
        X_train, latent_dim=latent_dim, n_epochs=n_epochs, seed=seed
    )

    # Reconstruction error as anomaly score (high error -> more anomalous)
    raw_train    = _score_ae(ae_model, X_train)
    smooth_train = pipe_mod.smooth_scores_per_unit(train_df, raw_train)

    pearson_r  = pipe_mod.compute_correlation(smooth_train, train_df["true_RUL"].values)
    pseudo_rul = pipe_mod.generate_pseudo_labels(smooth_train, threshold)

    X_train_r    = np.column_stack([X_train, smooth_train])
    pseudo_model = pipe_mod.train_ridge(X_train_r, pseudo_rul)
    true_model   = pipe_mod.train_ridge(X_train_r, train_df["true_RUL"].values)

    X_test      = test_df[feature_cols].values.astype(np.float32)
    raw_te      = _score_ae(ae_model, X_test)
    smooth_te   = pipe_mod.smooth_scores_per_unit(test_df, raw_te)
    X_te_full_r = np.column_stack([X_test, smooth_te])
    last_idx    = test_df.groupby("unit_id")["cycle"].idxmax()
    is_last     = test_df.index.isin(last_idx)
    X_test_r    = X_te_full_r[is_last]
    last_df     = test_df.loc[last_idx]

    y_pred_pseudo = np.clip(pseudo_model.predict(X_test_r), 0, 125)
    y_pred_true   = np.clip(true_model.predict(X_test_r),   0, 125)

    unit_ids_sorted = sorted(test_df["unit_id"].unique())
    rul_map         = dict(zip(unit_ids_sorted, raw["rul"].values))
    y_true_eng      = last_df["unit_id"].map(rul_map).values.astype(float)

    return {
        "subset":              name,
        "pipeline_type":       "ae",
        "n_epochs":            n_epochs,
        "latent_dim":          latent_dim,
        "threshold":           threshold,
        "k_clusters":          k,
        "cluster_sizes_train": cluster_sizes,
        "n_features_total":    len(feature_cols) + 1,
        "pearson_r":           pearson_r,
        "pseudo_metrics":      pipe_mod.compute_metrics(y_true_eng, y_pred_pseudo),
        "true_metrics":        pipe_mod.compute_metrics(y_true_eng, y_pred_true),
        "y_true_per_engine":   y_true_eng,
        "y_pred_pseudo":       y_pred_pseudo,
        "y_pred_true":         y_pred_true,
    }


# -- __main__ ------------------------------------------------------------------

if __name__ == "__main__":
    subsets = ["FD001", "FD002"]
    rows = []

    for subset in subsets:
        print(f"\nRunning AE pipeline for {subset}...")
        r = run_ae_pipeline(subset)
        pm, tm = r["pseudo_metrics"], r["true_metrics"]

        corr = r["pearson_r"]
        print(f"  Pearson r (rec_error vs true_RUL): {corr:.4f}", end="")
        if corr > 0:
            print("  <- WARNING: positive correlation -- check AE training")
        elif -0.7 <= corr <= -0.4:
            print("  <- OK (expected range [-0.4, -0.7])")
        else:
            print(f"  <- outside expected [-0.4, -0.7]")

        print(f"  Pseudo RMSE: {pm['rmse']:.2f}", end="")
        if pm["rmse"] > 50 and subset == "FD001":
            print("  <- WARNING: >50 on FD001 -- possible AE training issue")
        else:
            print()

        rows.append({
            "subset":       subset,
            "pipeline_type": "ae",
            "n_epochs":     r["n_epochs"],
            "latent_dim":   r["latent_dim"],
            "threshold":    r["threshold"],
            "k_clusters":   r["k_clusters"],
            "pearson_r":    round(corr, 4),
            "pseudo_rmse":  round(pm["rmse"], 2),
            "pseudo_mae":   round(pm["mae"], 2),
            "pseudo_nasa":  int(round(pm["nasa_score"])),
            "true_rmse":    round(tm["rmse"], 2),
            "true_mae":     round(tm["mae"], 2),
            "true_nasa":    int(round(tm["nasa_score"])),
        })

    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    out = tables_dir / "ae_baseline_results.csv"
    pd.DataFrame(rows).to_csv(out, index=False)

    print(f"\n{'='*65}")
    print("AE PIPELINE - FD001 + FD002")
    print(f"{'Subset':<8} {'Pearson_r':>10} {'Pseudo_RMSE':>12} {'True_RMSE':>10}")
    print("-" * 45)
    for r in rows:
        print(f"{r['subset']:<8} {r['pearson_r']:>10.4f} {r['pseudo_rmse']:>12.2f} {r['true_rmse']:>10.2f}")
    print(f"\nSaved: {out}")
