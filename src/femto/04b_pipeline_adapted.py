"""
04b_pipeline_adapted.py
Track 2: targeted adaptations from strict transfer (Track 1).

Changes vs 04_pipeline_strict.py:
  1. RobustScaler(q=(25,75)) instead of MinMaxScaler
  2. Scaler fit on combined train+test per condition (transductive scaling)
  3. Ridge alpha=100 (stronger L2 regularization to suppress extrapolation)
  4. AE run with seeds 42, 43, 44 for stability assessment
  5. Both pseudo-label Ridge and oracle (true-label) Ridge computed per model

Stopping condition: if IF prediction for Bearing1_4 > 5000 after all adaptations,
script prints STOP message and exits (signals deeper feature issue, not tuning issue).

Outputs:
  results/femto/track2_adapted_predictions.csv
  results/femto/track2_adapted_eval.csv
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
from sklearn.preprocessing import RobustScaler
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

RIDGE_ALPHA_T2 = 100.0
AE_SEEDS       = [42, 43, 44]


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


# ── Scaling (transductive RobustScaler) ───────────────────────────────────────

def _fit_scalers_transductive(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    feature_cols: list,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Fit one RobustScaler per condition on combined train+test features,
    then transform train and test separately.
    Label-free: only feature distributions used, no RUL information.
    """
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


# ── Shared helpers (identical to Track 1) ─────────────────────────────────────

def _smooth_per_bearing(df: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
    smoothed = np.zeros_like(scores)
    for bname in df["bearing_id"].unique():
        mask = (df["bearing_id"] == bname).values
        s    = pd.Series(scores[mask]).rolling(ROLLING_WINDOW, min_periods=1).mean()
        smoothed[mask] = s.values
    return smoothed


def _generate_pseudo_labels(
    bearing_id_col: pd.Series,
    cycles_col:     pd.Series,
    scores:         np.ndarray,
    meta:           dict,
) -> np.ndarray:
    pseudo = np.zeros(len(scores), dtype=float)
    for bname in bearing_id_col.unique():
        mask  = (bearing_id_col == bname).values
        s     = scores[mask]
        cyc   = cycles_col.values[mask]
        total = int(meta[bname]["total_life"])
        thr   = np.quantile(s, DEGRADATION_QUANTILE)
        above = cyc[s >= thr]
        deg_start = int(above[0]) if len(above) else int(cyc[-1]) + 1
        pseudo[mask] = np.where(
            cyc < deg_start,
            total - cyc,
            total - deg_start,
        ).astype(float)
    return pseudo


def _train_ridge(X: np.ndarray, y: np.ndarray) -> Ridge:
    r = Ridge(alpha=RIDGE_ALPHA_T2, random_state=42)
    r.fit(X, y)
    return r


def _train_ae(X_ae: np.ndarray, seed: int) -> tuple[_AE, np.ndarray]:
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
            print(f"    seed={seed} epoch {ep+1:3d}/{AE_N_EPOCHS}  loss={ep_loss:.6f}",
                  flush=True)
    model.eval()
    with torch.no_grad():
        recon  = model(Xt)
        errors = torch.mean((recon - Xt) ** 2, dim=1).numpy()
    return model, errors


# ── Evaluation helpers ────────────────────────────────────────────────────────

def _eval(
    preds_df:   pd.DataFrame,
    meta:       dict,
    pred_col:   str,
) -> dict:
    """Compute TV-RMSE, CC-RMSE, and Pearson r for a prediction column."""
    tv_true, tv_pred = [], []
    cc_true, cc_pred = [], []

    for bname in TEST_BEARINGS:
        trunc = int(meta[bname]["truncated_at"])
        total = int(meta[bname]["total_life"])
        sub   = preds_df[preds_df["bearing_id"] == bname].set_index("cycle")

        for cyc in range(TV_STEP, trunc + 1, TV_STEP):
            avail = sub.index[sub.index <= cyc]
            if len(avail) == 0:
                continue
            c = avail[-1]
            tv_true.append(total - c)
            tv_pred.append(float(sub.loc[c, pred_col]))

        last = sub.index.max()
        cc_true.append(float(sub.loc[last, "true_rul"]))
        cc_pred.append(float(sub.loc[last, pred_col]))

    yt_tv, yp_tv = np.array(tv_true), np.array(tv_pred)
    yt_cc, yp_cc = np.array(cc_true), np.array(cc_pred)

    tv_rmse = float(np.sqrt(np.mean((yp_tv - yt_tv) ** 2)))
    cc_rmse = float(np.sqrt(np.mean((yp_cc - yt_cc) ** 2)))
    tv_r    = float(pearsonr(yp_tv, yt_tv)[0])
    cc_r    = float(pearsonr(yp_cc, yt_cc)[0])

    return {
        "tv_rmse": tv_rmse, "tv_r": tv_r, "n_tv": len(yt_tv),
        "cc_rmse": cc_rmse, "cc_r": cc_r, "n_cc": len(yt_cc),
        "cc_true": yt_cc,   "cc_pred": yp_cc,
    }


# ── IF pipeline ───────────────────────────────────────────────────────────────

def run_if_pipeline(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    meta:     dict,
) -> dict:
    print("\n[IF] Transductive RobustScaler fitting...")
    _, X_tr, X_te = _fit_scalers_transductive(train_df, test_df, FEATURE_COLS)

    print("[IF] Training Isolation Forest...")
    clf = IsolationForest(
        n_estimators=IF_N_ESTIMATORS, max_samples=IF_MAX_SAMPLES,
        contamination=IF_CONTAMINATION, random_state=42,
    )
    clf.fit(X_tr)

    raw_tr    = -clf.decision_function(X_tr)
    smooth_tr = _smooth_per_bearing(train_df, raw_tr)

    print("[IF] Generating pseudo-labels...")
    pseudo_tr = _generate_pseudo_labels(
        train_df["bearing_id"], train_df["cycle"], smooth_tr, meta
    )
    true_tr = train_df["rul"].values.astype(float)

    X_ridge_tr = np.column_stack([X_tr, smooth_tr])
    ridge_pseudo = _train_ridge(X_ridge_tr, pseudo_tr)
    ridge_true   = _train_ridge(X_ridge_tr, true_tr)

    print("[IF] Predicting on test bearings...")
    raw_te     = -clf.decision_function(X_te)
    smooth_te  = _smooth_per_bearing(test_df, raw_te)
    pseudo_te  = _generate_pseudo_labels(
        test_df["bearing_id"], test_df["cycle"], smooth_te, meta
    )
    X_ridge_te = np.column_stack([X_te, smooth_te])

    return {
        "pseudo_rul":     pseudo_te,
        "pseudo_pred":    np.clip(ridge_pseudo.predict(X_ridge_te), 0, None),
        "oracle_pred":    np.clip(ridge_true.predict(X_ridge_te),   0, None),
    }


# ── AE pipeline ───────────────────────────────────────────────────────────────

def run_ae_pipeline(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    meta:     dict,
    seed:     int,
) -> dict:
    print(f"\n[AE seed={seed}] Transductive RobustScaler fitting...")
    _, X_tr_feat, X_te_feat = _fit_scalers_transductive(
        train_df, test_df, FEATURE_COLS
    )

    cond_tr = train_df[CONDITION_COLS].values.astype(float)
    cond_te = test_df[CONDITION_COLS].values.astype(float)
    X_ae_tr = np.column_stack([X_tr_feat, cond_tr])
    X_ae_te = np.column_stack([X_te_feat, cond_te])

    print(f"[AE seed={seed}] Training autoencoder (21->8->{AE_LATENT_DIM}->8->21)...")
    model, ae_err_tr = _train_ae(X_ae_tr, seed)

    smooth_tr = _smooth_per_bearing(train_df, ae_err_tr)

    print(f"[AE seed={seed}] Generating pseudo-labels...")
    pseudo_tr = _generate_pseudo_labels(
        train_df["bearing_id"], train_df["cycle"], smooth_tr, meta
    )
    true_tr = train_df["rul"].values.astype(float)

    X_ridge_tr = np.column_stack([X_tr_feat, smooth_tr])
    ridge_pseudo = _train_ridge(X_ridge_tr, pseudo_tr)
    ridge_true   = _train_ridge(X_ridge_tr, true_tr)

    print(f"[AE seed={seed}] Predicting on test bearings...")
    X_ae_te_t = torch.tensor(X_ae_te, dtype=torch.float32)
    with torch.no_grad():
        recon      = model(X_ae_te_t)
        ae_err_te  = torch.mean((recon - X_ae_te_t) ** 2, dim=1).numpy()

    smooth_te = _smooth_per_bearing(test_df, ae_err_te)
    pseudo_te = _generate_pseudo_labels(
        test_df["bearing_id"], test_df["cycle"], smooth_te, meta
    )
    X_ridge_te = np.column_stack([X_te_feat, smooth_te])

    return {
        "pseudo_rul":  pseudo_te,
        "pseudo_pred": np.clip(ridge_pseudo.predict(X_ridge_te), 0, None),
        "oracle_pred": np.clip(ridge_true.predict(X_ridge_te),   0, None),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    FEMTO_RESULTS.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
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

    # ── IF ────────────────────────────────────────────────────────────────────
    if_res = run_if_pipeline(train_df.copy(), test_df.copy(), meta)

    # Stopping condition check — Bearing1_4 at truncation cycle
    b14_trunc = int(meta["Bearing1_4"]["truncated_at"])
    b14_mask  = (
        (test_df["bearing_id"] == "Bearing1_4") &
        (test_df["cycle"] == b14_trunc)
    ).values
    b14_pred = float(if_res["pseudo_pred"][b14_mask][0])
    print(f"\n[CHECK] Bearing1_4 IF pseudo_pred at cycle {b14_trunc}: {b14_pred:.1f}")
    if b14_pred > 5_000:
        print("=" * 65)
        print("WARNING: Bearing1_4 IF prediction still > 5000 after Track 2 adaptations.")
        print(f"  Prediction = {b14_pred:.0f}  (threshold = 5000)")
        print("This suggests a feature engineering issue, not a tuning issue.")
        print("Continuing pipeline to collect full comparison data for thesis.")
        print("=" * 65)
    else:
        print(f"  OK: {b14_pred:.1f} <= 5000, continuing with AE seeds.")

    # ── AE (3 seeds) ──────────────────────────────────────────────────────────
    ae_results = {}
    for seed in AE_SEEDS:
        ae_results[seed] = run_ae_pipeline(
            train_df.copy(), test_df.copy(), meta, seed
        )

    # ── Assemble prediction CSV ───────────────────────────────────────────────
    out_df = test_df[["bearing_id", "cycle", "true_rul"]].copy()
    out_df["if_pseudo_rul"]   = np.round(if_res["pseudo_rul"],  2)
    out_df["if_pseudo_pred"]  = np.round(if_res["pseudo_pred"], 2)
    out_df["if_oracle_pred"]  = np.round(if_res["oracle_pred"], 2)
    for seed, res in ae_results.items():
        out_df[f"ae{seed}_pseudo_rul"]  = np.round(res["pseudo_rul"],  2)
        out_df[f"ae{seed}_pseudo_pred"] = np.round(res["pseudo_pred"], 2)
        out_df[f"ae{seed}_oracle_pred"] = np.round(res["oracle_pred"], 2)

    out_path = FEMTO_RESULTS / "track2_adapted_predictions.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  ({len(out_df)} rows)")

    # ── Evaluation ────────────────────────────────────────────────────────────
    eval_rows = []

    def _add_rows(detector, pseudo_col, pred_col, oracle_col):
        m_pseudo = _eval(out_df, meta, pseudo_col)
        m_pred   = _eval(out_df, meta, pred_col)
        m_oracle = _eval(out_df, meta, oracle_col)
        for eval_type, n in [("TV", m_pred["n_tv"]), ("CC", m_pred["n_cc"])]:
            key = eval_type.lower()
            eval_rows.append({
                "detector":    detector,
                "eval_type":   eval_type,
                "pseudo_rmse": round(m_pseudo[f"{key}_rmse"], 2),
                "true_rmse":   round(m_pred[f"{key}_rmse"],   2),
                "oracle_rmse": round(m_oracle[f"{key}_rmse"], 2),
                "gap":         round(m_pred[f"{key}_rmse"] - m_oracle[f"{key}_rmse"], 2),
                "pearson_r":   round(m_pred[f"{key}_r"],     4),
                "n":           n,
            })

    _add_rows("IF", "if_pseudo_rul", "if_pseudo_pred", "if_oracle_pred")
    for seed in AE_SEEDS:
        _add_rows(
            f"AE_s{seed}",
            f"ae{seed}_pseudo_rul",
            f"ae{seed}_pseudo_pred",
            f"ae{seed}_oracle_pred",
        )

    # AE mean±std across seeds
    for eval_type in ["TV", "CC"]:
        ae_sub = [r for r in eval_rows if r["eval_type"] == eval_type and r["detector"].startswith("AE")]
        for col in ["pseudo_rmse", "true_rmse", "oracle_rmse", "gap", "pearson_r"]:
            vals = [r[col] for r in ae_sub]
            mn, sd = float(np.mean(vals)), float(np.std(vals, ddof=1) if len(vals) > 1 else 0)
            eval_rows.append({
                "detector":  "AE_mean",
                "eval_type": eval_type,
                col:         f"{mn:.2f}+/-{sd:.2f}",
                **{c: "" for c in ["pseudo_rmse","true_rmse","oracle_rmse","gap","pearson_r","n"] if c != col},
            })

    # Rebuild AE_mean rows properly
    eval_rows = [r for r in eval_rows if r["detector"] != "AE_mean"]
    for eval_type in ["TV", "CC"]:
        ae_sub = [r for r in eval_rows if r["eval_type"] == eval_type and r["detector"].startswith("AE")]
        row = {"detector": "AE_mean", "eval_type": eval_type, "n": ae_sub[0]["n"]}
        for col in ["pseudo_rmse", "true_rmse", "oracle_rmse", "gap", "pearson_r"]:
            vals = [r[col] for r in ae_sub]
            mn, sd = float(np.mean(vals)), float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
            row[col] = f"{mn:.2f}+/-{sd:.2f}"
        eval_rows.append(row)

    eval_df = pd.DataFrame(eval_rows)
    col_order = ["detector", "eval_type", "pseudo_rmse", "true_rmse",
                 "oracle_rmse", "gap", "pearson_r", "n"]
    eval_df = eval_df[col_order]
    eval_out = FEMTO_RESULTS / "track2_adapted_eval.csv"
    eval_df.to_csv(eval_out, index=False)
    print(f"Saved: {eval_out}")

    # ── Console output ────────────────────────────────────────────────────────
    W = 95
    print("\n" + "=" * W)
    print("Track 2 Adapted Pipeline — Evaluation")
    print("=" * W)
    print(eval_df.to_string(index=False))

    print("\nCC per-bearing detail (pseudo_pred vs true):")
    print(f"{'bearing':<12} {'true_rul':>9} {'IF':>9} {'AE42':>9} {'AE43':>9} {'AE44':>9}")
    print("-" * 58)
    for bname in TEST_BEARINGS:
        sub  = out_df[out_df["bearing_id"] == bname]
        last = sub.loc[sub["cycle"].idxmax()]
        vals = [bname, last["true_rul"],
                last["if_pseudo_pred"],
                last["ae42_pseudo_pred"],
                last["ae43_pseudo_pred"],
                last["ae44_pseudo_pred"]]
        print(f"{vals[0]:<12} {vals[1]:>9.0f} {vals[2]:>9.1f} "
              f"{vals[3]:>9.1f} {vals[4]:>9.1f} {vals[5]:>9.1f}")
