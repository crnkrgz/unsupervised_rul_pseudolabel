"""
04_pipeline_strict.py
Strict C-MAPSS parameter transfer to FEMTO.

Pipeline (applied identically for IF and AE):
  1. Condition-aware MinMax scaling (one scaler per condition on train features)
  2. Anomaly scoring  (IF: -decision_function; AE: mean reconstruction error)
  3. Per-bearing rolling smoothing (window=5)
  4. Pseudo-label generation per training bearing (quantile-based degradation onset)
  5. Ridge regression on [scaled_features | smoothed_score]
  6. Predict on all test cycles

Output: results/femto/track1_strict_predictions.csv
Columns: bearing_id, cycle, true_rul,
         if_pseudo_rul, if_predicted_rul,
         ae_pseudo_rul, ae_predicted_rul
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from femto.config import (
    AE_LATENT_DIM, AE_LR, AE_N_EPOCHS,
    CONDITION_COLS, DEGRADATION_QUANTILE,
    FEATURE_COLS, FEMTO_PROCESSED, FEMTO_RESULTS,
    IF_CONTAMINATION, IF_MAX_SAMPLES, IF_N_ESTIMATORS,
    RIDGE_ALPHA, ROLLING_WINDOW, SEED, TEST_BEARINGS, TRAIN_BEARINGS,
    bearing_condition,
)


# ── Autoencoder ───────────────────────────────────────────────────────────────

class _AE(nn.Module):
    def __init__(self, input_dim: int = 21, latent_dim: int = AE_LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8), nn.ReLU(),
            nn.Linear(8, latent_dim), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8), nn.ReLU(),
            nn.Linear(8, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _smooth_per_bearing(df: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
    """Rolling mean (window=ROLLING_WINDOW) applied independently per bearing."""
    smoothed = np.zeros_like(scores)
    for bname in df["bearing_id"].unique():
        mask = (df["bearing_id"] == bname).values
        s    = pd.Series(scores[mask]).rolling(ROLLING_WINDOW, min_periods=1).mean()
        smoothed[mask] = s.values
    return smoothed


def _generate_pseudo_labels(
    bearing_id_col: pd.Series,
    cycles_col: pd.Series,
    scores: np.ndarray,
    meta: dict,
) -> np.ndarray:
    """
    Per bearing: threshold = DEGRADATION_QUANTILE of bearing's own scores.
    Healthy zone (score < threshold): pseudo_rul = total_life - cycle   (true RUL)
    Degraded zone (score >= threshold): pseudo_rul = total_life - degradation_start_cycle
    """
    pseudo = np.zeros(len(scores), dtype=float)
    for bname in bearing_id_col.unique():
        mask   = (bearing_id_col == bname).values
        s      = scores[mask]
        cyc    = cycles_col.values[mask]
        total  = int(meta[bname]["total_life"])

        thr    = np.quantile(s, DEGRADATION_QUANTILE)
        deg_at = cyc[s >= thr]
        if len(deg_at) == 0:
            deg_start = cyc[-1] + 1   # no degradation detected
        else:
            deg_start = int(deg_at[0])

        pseudo[mask] = np.where(
            cyc < deg_start,
            total - cyc,              # healthy: true RUL
            total - deg_start,        # degraded: constant ceiling at onset
        ).astype(float)

    return pseudo


def _fit_condition_scalers(
    df: pd.DataFrame, feature_cols: list
) -> tuple[dict, np.ndarray]:
    """Fit one MinMaxScaler per condition on df; return (scalers, scaled_X)."""
    X = df[feature_cols].values.astype(float).copy()
    scalers = {}
    for cond in [1, 2, 3]:
        mask = (df["condition"] == cond).values
        if not mask.any():
            continue
        sc = MinMaxScaler()
        X[mask] = sc.fit_transform(X[mask])
        scalers[cond] = sc
    return scalers, X


def _apply_condition_scalers(
    df: pd.DataFrame, feature_cols: list, scalers: dict
) -> np.ndarray:
    X = df[feature_cols].values.astype(float).copy()
    for cond, sc in scalers.items():
        mask = (df["condition"] == cond).values
        if mask.any():
            X[mask] = sc.transform(X[mask])
    return X


def _train_ae(X_ae: np.ndarray) -> tuple[_AE, np.ndarray]:
    """Train AE on X_ae (21 features), return (model, reconstruction_errors)."""
    torch.manual_seed(SEED)
    model    = _AE(input_dim=X_ae.shape[1], latent_dim=AE_LATENT_DIM)
    opt      = torch.optim.Adam(model.parameters(), lr=AE_LR)
    crit     = nn.MSELoss()
    X_tensor = torch.tensor(X_ae, dtype=torch.float32)
    loader   = DataLoader(
        TensorDataset(X_tensor), batch_size=64, shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    model.train()
    for ep in range(AE_N_EPOCHS):
        ep_loss = 0.0
        for (batch,) in loader:
            opt.zero_grad()
            loss = crit(model(batch), batch)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(batch)
        if (ep + 1) % 10 == 0:
            print(f"    AE epoch {ep+1:3d}/{AE_N_EPOCHS}  "
                  f"loss={ep_loss/len(X_ae):.6f}", flush=True)

    model.eval()
    with torch.no_grad():
        recon  = model(X_tensor)
        errors = torch.mean((recon - X_tensor) ** 2, dim=1).numpy()
    return model, errors


# ── IF pipeline ───────────────────────────────────────────────────────────────

def run_if_pipeline(train_df, test_df, meta) -> dict:
    print("\n[IF] Fitting condition scalers...")
    scalers, X_train = _fit_condition_scalers(train_df, FEATURE_COLS)
    X_test           = _apply_condition_scalers(test_df, FEATURE_COLS, scalers)

    print("[IF] Training Isolation Forest...")
    clf = IsolationForest(
        n_estimators=IF_N_ESTIMATORS,
        max_samples=IF_MAX_SAMPLES,
        contamination=IF_CONTAMINATION,
        random_state=SEED,
    )
    clf.fit(X_train)

    raw_train   = -clf.decision_function(X_train)
    smooth_train = _smooth_per_bearing(train_df, raw_train)

    print("[IF] Generating pseudo-labels...")
    pseudo_train = _generate_pseudo_labels(
        train_df["bearing_id"], train_df["cycle"], smooth_train, meta
    )

    print("[IF] Training Ridge (pseudo-label model)...")
    X_ridge_train = np.column_stack([X_train, smooth_train])
    ridge = Ridge(alpha=RIDGE_ALPHA, random_state=SEED)
    ridge.fit(X_ridge_train, pseudo_train)

    print("[IF] Predicting on test bearings...")
    raw_test    = -clf.decision_function(X_test)
    smooth_test  = _smooth_per_bearing(test_df, raw_test)
    pseudo_test  = _generate_pseudo_labels(
        test_df["bearing_id"], test_df["cycle"], smooth_test, meta
    )
    X_ridge_test  = np.column_stack([X_test, smooth_test])
    y_pred_test   = np.clip(ridge.predict(X_ridge_test), 0, None)

    return {
        "pseudo_rul":    pseudo_test,
        "predicted_rul": y_pred_test,
    }


# ── AE pipeline ───────────────────────────────────────────────────────────────

def run_ae_pipeline(train_df, test_df, meta) -> dict:
    print("\n[AE] Fitting condition scalers (shared with IF)...")
    scalers, X_train_feat = _fit_condition_scalers(train_df, FEATURE_COLS)
    X_test_feat           = _apply_condition_scalers(test_df, FEATURE_COLS, scalers)

    # AE input = 18 scaled features + 3 condition dummies
    cond_train = train_df[CONDITION_COLS].values.astype(float)
    cond_test  = test_df[CONDITION_COLS].values.astype(float)
    X_ae_train = np.column_stack([X_train_feat, cond_train])
    X_ae_test  = np.column_stack([X_test_feat,  cond_test])

    print(f"[AE] Training autoencoder ({X_ae_train.shape[1]}->8->{AE_LATENT_DIM}->8->...")
    model, ae_errors_train = _train_ae(X_ae_train)

    smooth_train = _smooth_per_bearing(train_df, ae_errors_train)

    print("[AE] Generating pseudo-labels...")
    pseudo_train = _generate_pseudo_labels(
        train_df["bearing_id"], train_df["cycle"], smooth_train, meta
    )

    print("[AE] Training Ridge (pseudo-label model)...")
    X_ridge_train = np.column_stack([X_train_feat, smooth_train])
    ridge = Ridge(alpha=RIDGE_ALPHA, random_state=SEED)
    ridge.fit(X_ridge_train, pseudo_train)

    print("[AE] Predicting on test bearings...")
    model.eval()
    X_ae_test_t = torch.tensor(X_ae_test, dtype=torch.float32)
    with torch.no_grad():
        recon_test     = model(X_ae_test_t)
        ae_errors_test = torch.mean(
            (recon_test - X_ae_test_t) ** 2, dim=1
        ).numpy()

    smooth_test  = _smooth_per_bearing(test_df, ae_errors_test)
    pseudo_test  = _generate_pseudo_labels(
        test_df["bearing_id"], test_df["cycle"], smooth_test, meta
    )
    X_ridge_test  = np.column_stack([X_test_feat, smooth_test])
    y_pred_test   = np.clip(ridge.predict(X_ridge_test), 0, None)

    return {
        "pseudo_rul":    pseudo_test,
        "predicted_rul": y_pred_test,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    FEMTO_RESULTS.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    train_df = pd.read_csv(FEMTO_PROCESSED / "femto_train.csv")
    test_df  = pd.read_csv(FEMTO_PROCESSED / "femto_test_input.csv")
    meta_df  = pd.read_csv(FEMTO_PROCESSED / "bearing_metadata.csv")

    # Derive condition column from bearing_id (not stored in test_input)
    train_df["condition"] = train_df["bearing_id"].map(bearing_condition)
    test_df["condition"]  = test_df["bearing_id"].map(bearing_condition)

    meta = meta_df.set_index("bearing_id").to_dict("index")

    print(f"Train: {len(train_df)} rows ({len(TRAIN_BEARINGS)} bearings)")
    print(f"Test:  {len(test_df)} rows ({len(TEST_BEARINGS)} bearings)")

    # True RUL for test cycles (from Validation_Set lengths)
    def _true_rul(row):
        return int(meta[row["bearing_id"]]["total_life"]) - int(row["cycle"])
    test_df["true_rul"] = test_df.apply(_true_rul, axis=1)

    # Run IF
    if_res = run_if_pipeline(train_df.copy(), test_df.copy(), meta)

    # Run AE
    ae_res = run_ae_pipeline(train_df.copy(), test_df.copy(), meta)

    # Assemble output
    out_df = test_df[["bearing_id", "cycle", "true_rul"]].copy()
    out_df["if_pseudo_rul"]    = np.round(if_res["pseudo_rul"],    2)
    out_df["if_predicted_rul"] = np.round(if_res["predicted_rul"], 2)
    out_df["ae_pseudo_rul"]    = np.round(ae_res["pseudo_rul"],    2)
    out_df["ae_predicted_rul"] = np.round(ae_res["predicted_rul"], 2)

    out_path = FEMTO_RESULTS / "track1_strict_predictions.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"Rows: {len(out_df)}  Bearings: {out_df['bearing_id'].nunique()}")

    # Quick console summary (last-cycle predictions)
    print("\nChallenge-Convention Predictions (last cycle per bearing):")
    print(f"{'bearing':<12} {'true_rul':>8} {'if_pred':>8} {'ae_pred':>8}")
    print("-" * 42)
    for bname in TEST_BEARINGS:
        sub  = out_df[out_df["bearing_id"] == bname]
        last = sub.loc[sub["cycle"].idxmax()]
        print(f"{bname:<12} {last['true_rul']:>8.0f} "
              f"{last['if_predicted_rul']:>8.1f} "
              f"{last['ae_predicted_rul']:>8.1f}")
