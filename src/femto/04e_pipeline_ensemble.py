"""
04e_pipeline_ensemble.py
Track 5: Naive ensemble of IF and AE anomaly detectors.

Feature setup: T2 (18 features, RobustScaler transductive, Ridge alpha=100).
AE: T2 config (21->8->2->8->21, seed=42) — same feature space as IF.

Ensemble strategy:
  1. Run IF -> smoothed anomaly score
  2. Run AE -> smoothed reconstruction error
  3. Normalize each score independently to [0,1] (MinMaxScaler fit on train)
  4. Variant A (Mean): (if_norm + ae_norm) / 2
  5. Variant B (Max):  max(if_norm, ae_norm)
  6. Pseudo-label generation and Ridge regression on each ensemble score

Also recomputes IF-only and AE-only as reference columns (should match T2).

Outputs:
  results/femto/track5_ensemble_predictions.csv
  results/femto/track5_ensemble_eval.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from femto.config import (
    AE_LATENT_DIM, AE_LR, AE_N_EPOCHS,
    CONDITION_COLS, DEGRADATION_QUANTILE,
    FEATURE_COLS, FEMTO_PROCESSED, FEMTO_RESULTS,
    IF_CONTAMINATION, IF_MAX_SAMPLES, IF_N_ESTIMATORS,
    ROLLING_WINDOW, TEST_BEARINGS, TRAIN_BEARINGS, TV_STEP,
    bearing_condition,
)

RIDGE_ALPHA = 100.0
AE_SEED     = 42


# ── Autoencoder (T2 config: 21->8->2->8->21) ─────────────────────────────────

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

def _fit_scalers_transductive(train_df, test_df, feature_cols):
    X_tr = train_df[feature_cols].values.astype(float).copy()
    X_te = test_df[feature_cols].values.astype(float).copy()
    scalers = {}
    for cond in [1, 2, 3]:
        mask_tr = (train_df["condition"] == cond).values
        mask_te = (test_df["condition"]  == cond).values
        if not mask_tr.any():
            continue
        combined = (
            np.concatenate([X_tr[mask_tr], X_te[mask_te]], axis=0)
            if mask_te.any() else X_tr[mask_tr]
        )
        sc = RobustScaler(quantile_range=(25.0, 75.0))
        sc.fit(combined)
        X_tr[mask_tr] = sc.transform(X_tr[mask_tr])
        if mask_te.any():
            X_te[mask_te] = sc.transform(X_te[mask_te])
        scalers[cond] = sc
    return scalers, X_tr, X_te


def _smooth_per_bearing(df, scores):
    smoothed = np.zeros_like(scores)
    for bname in df["bearing_id"].unique():
        mask = (df["bearing_id"] == bname).values
        s    = pd.Series(scores[mask]).rolling(ROLLING_WINDOW, min_periods=1).mean()
        smoothed[mask] = s.values
    return smoothed


def _generate_pseudo_labels(bearing_id_col, cycles_col, scores, meta):
    pseudo = np.zeros(len(scores), dtype=float)
    for bname in bearing_id_col.unique():
        mask      = (bearing_id_col == bname).values
        s         = scores[mask]
        cyc       = cycles_col.values[mask]
        total     = int(meta[bname]["total_life"])
        thr       = np.quantile(s, DEGRADATION_QUANTILE)
        above     = cyc[s >= thr]
        deg_start = int(above[0]) if len(above) else int(cyc[-1]) + 1
        pseudo[mask] = np.where(cyc < deg_start, total - cyc, total - deg_start)
    return pseudo.astype(float)


def _train_ridge(X, y):
    return Ridge(alpha=RIDGE_ALPHA, random_state=42).fit(X, y)


def _train_ae(X_ae, seed):
    torch.manual_seed(seed)
    model  = _AE(input_dim=X_ae.shape[1], latent_dim=AE_LATENT_DIM)
    opt    = torch.optim.Adam(model.parameters(), lr=AE_LR)
    crit   = nn.MSELoss()
    Xt     = torch.tensor(X_ae, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(Xt), batch_size=64, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    model.train()
    for ep in range(AE_N_EPOCHS):
        for (batch,) in loader:
            opt.zero_grad()
            loss = crit(model(batch), batch)
            loss.backward()
            opt.step()
        if (ep + 1) % 10 == 0:
            with torch.no_grad():
                ep_loss = crit(model(Xt), Xt).item()
            print(f"    AE epoch {ep+1:3d}/{AE_N_EPOCHS}  loss={ep_loss:.6f}", flush=True)
    model.eval()
    with torch.no_grad():
        recon  = model(Xt)
        errors = torch.mean((recon - Xt) ** 2, dim=1).numpy()
    return model, errors


def _eval(preds_df, meta, pred_col):
    tv_true, tv_pred, cc_true, cc_pred = [], [], [], []
    for bname in TEST_BEARINGS:
        trunc = int(meta[bname]["truncated_at"])
        total = int(meta[bname]["total_life"])
        sub   = preds_df[preds_df["bearing_id"] == bname].set_index("cycle")
        for cyc in range(TV_STEP, trunc + 1, TV_STEP):
            avail = sub.index[sub.index <= cyc]
            if not len(avail):
                continue
            c = avail[-1]
            tv_true.append(total - c)
            tv_pred.append(float(sub.loc[c, pred_col]))
        last = sub.index.max()
        cc_true.append(float(sub.loc[last, "true_rul"]))
        cc_pred.append(float(sub.loc[last, pred_col]))
    yt_tv = np.array(tv_true);  yp_tv = np.array(tv_pred)
    yt_cc = np.array(cc_true);  yp_cc = np.array(cc_pred)
    tv_rmse = float(np.sqrt(np.mean((yp_tv - yt_tv) ** 2)))
    cc_rmse = float(np.sqrt(np.mean((yp_cc - yt_cc) ** 2)))
    tv_r    = float(pearsonr(yp_tv, yt_tv)[0])
    cc_r    = float(pearsonr(yp_cc, yt_cc)[0])
    return {
        "tv_rmse": tv_rmse, "n_tv": len(yt_tv), "tv_r": tv_r,
        "cc_rmse": cc_rmse, "n_cc": len(yt_cc), "cc_r": cc_r,
    }


# ── Main pipeline ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    FEMTO_RESULTS.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(FEMTO_PROCESSED / "femto_train.csv")
    test_df  = pd.read_csv(FEMTO_PROCESSED / "femto_test_input.csv")
    meta_df  = pd.read_csv(FEMTO_PROCESSED / "bearing_metadata.csv")

    train_df["condition"] = train_df["bearing_id"].map(bearing_condition)
    test_df["condition"]  = test_df["bearing_id"].map(bearing_condition)
    meta = meta_df.set_index("bearing_id").to_dict("index")

    def _true_rul(row):
        return int(meta[row["bearing_id"]]["total_life"]) - int(row["cycle"])
    test_df["true_rul"] = test_df.apply(_true_rul, axis=1)

    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")

    # ── Feature scaling (T2: RobustScaler transductive) ───────────────────────
    print("\n[SETUP] Transductive RobustScaler fitting (shared by IF and AE)...")
    _, X_tr, X_te = _fit_scalers_transductive(train_df, test_df, FEATURE_COLS)

    # ── IF anomaly scores ─────────────────────────────────────────────────────
    print("\n[IF] Training Isolation Forest...")
    clf = IsolationForest(
        n_estimators=IF_N_ESTIMATORS, max_samples=IF_MAX_SAMPLES,
        contamination=IF_CONTAMINATION, random_state=42,
    )
    clf.fit(X_tr)
    raw_tr_if    = -clf.decision_function(X_tr)
    raw_te_if    = -clf.decision_function(X_te)
    smooth_tr_if = _smooth_per_bearing(train_df, raw_tr_if)
    smooth_te_if = _smooth_per_bearing(test_df,  raw_te_if)
    print(f"  IF score range (train): [{smooth_tr_if.min():.4f}, {smooth_tr_if.max():.4f}]")

    # ── AE reconstruction errors ──────────────────────────────────────────────
    print(f"\n[AE seed={AE_SEED}] Training autoencoder (21->8->{AE_LATENT_DIM}->8->21)...")
    cond_tr  = train_df[CONDITION_COLS].values.astype(float)
    cond_te  = test_df[CONDITION_COLS].values.astype(float)
    X_ae_tr  = np.column_stack([X_tr, cond_tr])
    X_ae_te  = np.column_stack([X_te, cond_te])
    ae_model, ae_err_tr = _train_ae(X_ae_tr, AE_SEED)
    X_ae_te_t = torch.tensor(X_ae_te, dtype=torch.float32)
    with torch.no_grad():
        ae_err_te = torch.mean(
            (ae_model(X_ae_te_t) - X_ae_te_t) ** 2, dim=1
        ).numpy()
    smooth_tr_ae = _smooth_per_bearing(train_df, ae_err_tr)
    smooth_te_ae = _smooth_per_bearing(test_df,  ae_err_te)
    print(f"  AE error range (train): [{smooth_tr_ae.min():.6f}, {smooth_tr_ae.max():.6f}]")

    # ── Score normalization (fit on train) ────────────────────────────────────
    print("\n[NORM] Normalizing scores to [0,1] (MinMaxScaler, fit on train)...")
    ss_if = MinMaxScaler()
    ss_ae = MinMaxScaler()
    norm_tr_if = ss_if.fit_transform(smooth_tr_if.reshape(-1, 1)).ravel()
    norm_te_if = ss_if.transform(smooth_te_if.reshape(-1, 1)).ravel()
    norm_tr_ae = ss_ae.fit_transform(smooth_tr_ae.reshape(-1, 1)).ravel()
    norm_te_ae = ss_ae.transform(smooth_te_ae.reshape(-1, 1)).ravel()
    print(f"  IF norm range (test):  [{norm_te_if.min():.3f}, {norm_te_if.max():.3f}]")
    print(f"  AE norm range (test):  [{norm_te_ae.min():.3f}, {norm_te_ae.max():.3f}]")

    # Ensemble scores
    ens_tr_mean = (norm_tr_if + norm_tr_ae) / 2.0
    ens_te_mean = (norm_te_if + norm_te_ae) / 2.0
    ens_tr_max  = np.maximum(norm_tr_if, norm_tr_ae)
    ens_te_max  = np.maximum(norm_te_if, norm_te_ae)

    # ── Ridge training and prediction per variant ──────────────────────────────
    true_tr = train_df["rul"].values.astype(float)

    out_df = test_df[["bearing_id", "cycle", "true_rul"]].copy()

    variants = {
        "if_only":   (smooth_tr_if, smooth_te_if),
        "ae_only":   (smooth_tr_ae, smooth_te_ae),
        "ens_mean":  (ens_tr_mean,  ens_te_mean),
        "ens_max":   (ens_tr_max,   ens_te_max),
    }

    for vname, (score_tr, score_te) in variants.items():
        print(f"\n[{vname}] Pseudo-labels + Ridge...")
        pseudo_tr  = _generate_pseudo_labels(
            train_df["bearing_id"], train_df["cycle"], score_tr, meta
        )
        X_ridge_tr   = np.column_stack([X_tr, score_tr])
        ridge_pseudo = _train_ridge(X_ridge_tr, pseudo_tr)
        ridge_oracle = _train_ridge(X_ridge_tr, true_tr)

        pseudo_te  = _generate_pseudo_labels(
            test_df["bearing_id"], test_df["cycle"], score_te, meta
        )
        X_ridge_te = np.column_stack([X_te, score_te])
        out_df[f"{vname}_pseudo_rul"]  = np.round(pseudo_te, 2)
        out_df[f"{vname}_pseudo_pred"] = np.round(
            np.clip(ridge_pseudo.predict(X_ridge_te), 0, None), 2
        )
        out_df[f"{vname}_oracle_pred"] = np.round(
            np.clip(ridge_oracle.predict(X_ridge_te), 0, None), 2
        )

    pred_out = FEMTO_RESULTS / "track5_ensemble_predictions.csv"
    out_df.to_csv(pred_out, index=False)
    print(f"\nSaved: {pred_out}  ({len(out_df)} rows)")

    # ── Evaluation table ──────────────────────────────────────────────────────
    eval_rows = []

    variant_meta = [
        ("IF",       "reference", "if_only_pseudo_rul",  "if_only_pseudo_pred",  "if_only_oracle_pred"),
        ("AE",       "reference", "ae_only_pseudo_rul",  "ae_only_pseudo_pred",  "ae_only_oracle_pred"),
        ("Ensemble", "Mean",      "ens_mean_pseudo_rul", "ens_mean_pseudo_pred", "ens_mean_oracle_pred"),
        ("Ensemble", "Max",       "ens_max_pseudo_rul",  "ens_max_pseudo_pred",  "ens_max_oracle_pred"),
    ]

    for det, variant, pseudo_col, pred_col, oracle_col in variant_meta:
        m_pseudo = _eval(out_df, meta, pseudo_col)
        m_pred   = _eval(out_df, meta, pred_col)
        m_oracle = _eval(out_df, meta, oracle_col)
        for eval_type, key in [("TV", "tv"), ("CC", "cc")]:
            eval_rows.append({
                "detector":    det,
                "variant":     variant,
                "eval_type":   eval_type,
                "pseudo_rmse": round(m_pseudo[f"{key}_rmse"], 2),
                "true_rmse":   round(m_pred[f"{key}_rmse"],   2),
                "oracle_rmse": round(m_oracle[f"{key}_rmse"], 2),
                "gap":         round(m_pred[f"{key}_rmse"] - m_oracle[f"{key}_rmse"], 2),
                "pearson_r":   round(m_pred[f"{key}_r"],      4),
                "n":           m_pred[f"n_{key}"],
            })

    eval_df  = pd.DataFrame(eval_rows)
    col_ord  = ["detector", "variant", "eval_type", "pseudo_rmse",
                "true_rmse", "oracle_rmse", "gap", "pearson_r", "n"]
    eval_df  = eval_df[col_ord]
    eval_out = FEMTO_RESULTS / "track5_ensemble_eval.csv"
    eval_df.to_csv(eval_out, index=False)
    print(f"Saved: {eval_out}")

    # ── Console summary ───────────────────────────────────────────────────────
    W = 100
    print("\n" + "=" * W)
    print("Track 5 Ensemble - Evaluation")
    print("=" * W)
    print(eval_df.to_string(index=False))

    print("\nCC per-bearing (pseudo_pred):")
    print(f"{'bearing':<12} {'true':>6} {'IF_ref':>9} {'AE_ref':>9} {'Mean':>9} {'Max':>9}")
    print("-" * 58)
    for bname in TEST_BEARINGS:
        sub  = out_df[out_df["bearing_id"] == bname]
        last = sub.loc[sub["cycle"].idxmax()]
        print(f"{bname:<12} {last['true_rul']:>6.0f} "
              f"{last['if_only_pseudo_pred']:>9.1f} "
              f"{last['ae_only_pseudo_pred']:>9.1f} "
              f"{last['ens_mean_pseudo_pred']:>9.1f} "
              f"{last['ens_max_pseudo_pred']:>9.1f}")

    # ── T2 comparison ─────────────────────────────────────────────────────────
    t2_if_oracle = 735.20
    t5_if_oracle  = float(
        eval_df[(eval_df["detector"]=="IF")&(eval_df["eval_type"]=="CC")]["oracle_rmse"].values[0]
    )
    t5_mean_oracle = float(
        eval_df[(eval_df["detector"]=="Ensemble")&(eval_df["variant"]=="Mean")&
                (eval_df["eval_type"]=="CC")]["oracle_rmse"].values[0]
    )
    t5_max_oracle  = float(
        eval_df[(eval_df["detector"]=="Ensemble")&(eval_df["variant"]=="Max")&
                (eval_df["eval_type"]=="CC")]["oracle_rmse"].values[0]
    )

    print(f"\n{'-'*60}")
    print(f"  Oracle CC-RMSE comparison:")
    print(f"    T2 IF (reference):          {t2_if_oracle:.2f}")
    print(f"    T5 IF  (re-computed):       {t5_if_oracle:.2f}")
    print(f"    T5 Ensemble Mean:           {t5_mean_oracle:.2f}  ({(t5_mean_oracle-t2_if_oracle)/t2_if_oracle*100:+.1f}%)")
    print(f"    T5 Ensemble Max:            {t5_max_oracle:.2f}  ({(t5_max_oracle-t2_if_oracle)/t2_if_oracle*100:+.1f}%)")

    best_ens = min(t5_mean_oracle, t5_max_oracle)
    if best_ens < t2_if_oracle - 100:
        print("  -> Scenario E1: Ensemble substantially improves oracle.")
    elif best_ens <= t2_if_oracle + 100:
        print("  -> Scenario E2: Ensemble close to T2 IF — marginal effect.")
    else:
        print("  -> Scenario E3: Ensemble worse — AE noise pulls IF down.")
    print(f"{'-'*60}")
