"""
04c_pipeline_enhanced.py
Track 3: Enhanced feature engineering (39 features from 02b).

Pipeline parameters match Track 1 exactly (MinMaxScaler condition-aware,
Ridge alpha=1.0, contamination=0.20, rolling=5, seed=42) — only feature
set and AE architecture change.

AE architecture: 42->16->4->16->42  (39 enhanced features + 3 condition dummies)

Also computes:
  - Oracle Ridge (true-label supervision) for feature-space diagnostics
  - T1 Oracle Baseline (original 18 features, same T1 settings) for comparison

Outputs:
  results/femto/track3_enhanced_predictions.csv
  results/femto/track3_enhanced_eval.csv
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
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from femto.config import (
    AE_LR, AE_N_EPOCHS,
    CONDITION_COLS, DEGRADATION_QUANTILE,
    FEATURE_COLS, FEMTO_PROCESSED, FEMTO_RESULTS,
    IF_CONTAMINATION, IF_MAX_SAMPLES, IF_N_ESTIMATORS,
    RIDGE_ALPHA, ROLLING_WINDOW, SEED, TEST_BEARINGS, TRAIN_BEARINGS,
    TV_STEP, bearing_condition,
)

# ── Enhanced feature columns (must match 02b output) ─────────────────────────

_HOT = ["horiz_skewness", "horiz_shape_factor", "vert_skewness", "vert_shape_factor"]
_WPD = (
    [f"horiz_wpd_{n}" for n in ["aaaa", "aaad", "aada", "addd"]] +
    [f"vert_wpd_{n}"  for n in ["aaaa", "aaad", "aada", "addd"]]
)
_FAULT = (
    [f"horiz_{f}" for f in ["BPFO", "BPFI", "BSF", "FTF"]] +
    [f"vert_{f}"  for f in ["BPFO", "BPFI", "BSF", "FTF"]]
)
ENHANCED_COLS = FEATURE_COLS + _HOT + _WPD + _FAULT + ["pca_hi"]  # 39 features

AE_LATENT_DIM_T3 = 4
AE_HIDDEN_T3     = 16


# ── Autoencoder (42-16-4-16-42) ──────────────────────────────────────────────

class _AE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = AE_LATENT_DIM_T3,
                 hidden: int = AE_HIDDEN_T3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, latent_dim), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ── Shared helpers ────────────────────────────────────────────────────────────

def _smooth_per_bearing(df: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
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
        deg_at    = cyc[s >= thr]
        deg_start = int(deg_at[0]) if len(deg_at) else int(cyc[-1]) + 1
        pseudo[mask] = np.where(cyc < deg_start, total - cyc, total - deg_start)
    return pseudo.astype(float)


def _fit_condition_scalers(df, feature_cols):
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


def _apply_condition_scalers(df, feature_cols, scalers):
    X = df[feature_cols].values.astype(float).copy()
    for cond, sc in scalers.items():
        mask = (df["condition"] == cond).values
        if mask.any():
            X[mask] = sc.transform(X[mask])
    return X


def _train_ae(X_ae: np.ndarray) -> tuple:
    torch.manual_seed(SEED)
    input_dim = X_ae.shape[1]
    model     = _AE(input_dim=input_dim)
    opt       = torch.optim.Adam(model.parameters(), lr=AE_LR)
    crit      = nn.MSELoss()
    Xt        = torch.tensor(X_ae, dtype=torch.float32)
    loader    = DataLoader(
        TensorDataset(Xt), batch_size=64, shuffle=True,
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
        "tv_rmse": tv_rmse, "tv_r": tv_r, "n_tv": len(yt_tv),
        "cc_rmse": cc_rmse, "cc_r": cc_r, "n_cc": len(yt_cc),
        "cc_true": yt_cc,   "cc_pred": yp_cc,
    }


# ── T1 Oracle Baseline (original 18 features, MinMaxScaler, alpha=1.0) ────────

def compute_t1_oracle_baseline(train_df_orig, test_df_orig, meta):
    """Ridge on true labels with Track 1 setup — gives proper T1 oracle."""
    print("\n[BASELINE] T1 Oracle: original 18 features, MinMaxScaler, alpha=1.0")
    scalers, X_tr = _fit_condition_scalers(train_df_orig, FEATURE_COLS)
    X_te          = _apply_condition_scalers(test_df_orig, FEATURE_COLS, scalers)

    clf = IsolationForest(
        n_estimators=IF_N_ESTIMATORS, max_samples=IF_MAX_SAMPLES,
        contamination=IF_CONTAMINATION, random_state=SEED,
    )
    clf.fit(X_tr)
    raw_tr    = -clf.decision_function(X_tr)
    smooth_tr = _smooth_per_bearing(train_df_orig, raw_tr)
    true_tr   = train_df_orig["rul"].values.astype(float)

    X_ridge_tr = np.column_stack([X_tr, smooth_tr])
    ridge_oracle = Ridge(alpha=RIDGE_ALPHA, random_state=SEED).fit(X_ridge_tr, true_tr)

    raw_te    = -clf.decision_function(X_te)
    smooth_te = _smooth_per_bearing(test_df_orig, raw_te)
    X_ridge_te = np.column_stack([X_te, smooth_te])
    oracle_pred = np.clip(ridge_oracle.predict(X_ridge_te), 0, None)

    # CC-RMSE
    cc_true, cc_pred = [], []
    for bname in TEST_BEARINGS:
        trunc = int(meta[bname]["truncated_at"])
        sub   = test_df_orig[(test_df_orig["bearing_id"] == bname) &
                              (test_df_orig["cycle"] == trunc)]
        if sub.empty:
            sub = test_df_orig[test_df_orig["bearing_id"] == bname]
            sub = sub.loc[[sub["cycle"].idxmax()]]
        idx = sub.index[0]
        cc_true.append(float(sub.loc[idx, "true_rul"]))
        cc_pred.append(float(oracle_pred[idx]))

    cc_rmse = float(np.sqrt(np.mean((np.array(cc_pred) - np.array(cc_true)) ** 2)))
    print(f"  T1 Oracle IF CC-RMSE (MinMaxScaler, alpha=1.0): {cc_rmse:.2f}")
    return cc_rmse


# ── IF pipeline (enhanced features) ──────────────────────────────────────────

def run_if_pipeline(train_df, test_df, meta):
    print("\n[IF] Fitting condition scalers (MinMaxScaler, train-only)...")
    scalers, X_tr = _fit_condition_scalers(train_df, ENHANCED_COLS)
    X_te          = _apply_condition_scalers(test_df, ENHANCED_COLS, scalers)

    print("[IF] Training Isolation Forest...")
    clf = IsolationForest(
        n_estimators=IF_N_ESTIMATORS, max_samples=IF_MAX_SAMPLES,
        contamination=IF_CONTAMINATION, random_state=SEED,
    )
    clf.fit(X_tr)

    raw_tr    = -clf.decision_function(X_tr)
    smooth_tr = _smooth_per_bearing(train_df, raw_tr)

    print("[IF] Generating pseudo-labels...")
    pseudo_tr = _generate_pseudo_labels(
        train_df["bearing_id"], train_df["cycle"], smooth_tr, meta
    )
    true_tr = train_df["rul"].values.astype(float)

    X_ridge_tr   = np.column_stack([X_tr, smooth_tr])
    ridge_pseudo = Ridge(alpha=RIDGE_ALPHA, random_state=SEED).fit(X_ridge_tr, pseudo_tr)
    ridge_oracle = Ridge(alpha=RIDGE_ALPHA, random_state=SEED).fit(X_ridge_tr, true_tr)

    print("[IF] Predicting on test bearings...")
    raw_te    = -clf.decision_function(X_te)
    smooth_te = _smooth_per_bearing(test_df, raw_te)
    pseudo_te = _generate_pseudo_labels(
        test_df["bearing_id"], test_df["cycle"], smooth_te, meta
    )
    X_ridge_te = np.column_stack([X_te, smooth_te])

    return {
        "pseudo_rul":  pseudo_te,
        "pseudo_pred": np.clip(ridge_pseudo.predict(X_ridge_te), 0, None),
        "oracle_pred": np.clip(ridge_oracle.predict(X_ridge_te), 0, None),
    }


# ── AE pipeline (enhanced features) ──────────────────────────────────────────

def run_ae_pipeline(train_df, test_df, meta):
    print("\n[AE] Fitting condition scalers...")
    scalers, X_tr_feat = _fit_condition_scalers(train_df, ENHANCED_COLS)
    X_te_feat          = _apply_condition_scalers(test_df, ENHANCED_COLS, scalers)

    cond_tr = train_df[CONDITION_COLS].values.astype(float)
    cond_te = test_df[CONDITION_COLS].values.astype(float)
    X_ae_tr = np.column_stack([X_tr_feat, cond_tr])
    X_ae_te = np.column_stack([X_te_feat, cond_te])

    print(f"[AE] Training autoencoder ({X_ae_tr.shape[1]}->{AE_HIDDEN_T3}->"
          f"{AE_LATENT_DIM_T3}->{AE_HIDDEN_T3}->{X_ae_tr.shape[1]})...")
    model, ae_err_tr = _train_ae(X_ae_tr)

    smooth_tr = _smooth_per_bearing(train_df, ae_err_tr)

    print("[AE] Generating pseudo-labels...")
    pseudo_tr = _generate_pseudo_labels(
        train_df["bearing_id"], train_df["cycle"], smooth_tr, meta
    )
    true_tr = train_df["rul"].values.astype(float)

    X_ridge_tr   = np.column_stack([X_tr_feat, smooth_tr])
    ridge_pseudo = Ridge(alpha=RIDGE_ALPHA, random_state=SEED).fit(X_ridge_tr, pseudo_tr)
    ridge_oracle = Ridge(alpha=RIDGE_ALPHA, random_state=SEED).fit(X_ridge_tr, true_tr)

    print("[AE] Predicting on test bearings...")
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
        "oracle_pred": np.clip(ridge_oracle.predict(X_ridge_te), 0, None),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    FEMTO_RESULTS.mkdir(parents=True, exist_ok=True)

    meta_df = pd.read_csv(FEMTO_PROCESSED / "bearing_metadata.csv")
    meta    = meta_df.set_index("bearing_id").to_dict("index")

    # Check enhanced feature files exist
    missing = [
        f"{prefix}_{b}_enhanced_features.csv"
        for prefix, bearings in [("train", TRAIN_BEARINGS), ("test", TEST_BEARINGS)]
        for b in bearings
        if not (FEMTO_PROCESSED / f"{prefix}_{b}_enhanced_features.csv").exists()
    ]
    if missing:
        print("ERROR: Run 02b_feature_extraction_enhanced.py first. Missing:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)

    # ── Load enhanced datasets ────────────────────────────────────────────────
    print("Loading enhanced datasets...")
    train_parts = []
    for bname in TRAIN_BEARINGS:
        df = pd.read_csv(FEMTO_PROCESSED / f"train_{bname}_enhanced_features.csv")
        df["bearing_id"] = bname
        df["condition"]  = bearing_condition(bname)
        total            = int(meta[bname]["total_life"])
        df["rul"]        = total - df["cycle"]
        for col in ["cond_1", "cond_2", "cond_3"]:
            df[col] = 0
        df[f"cond_{bearing_condition(bname)}"] = 1
        train_parts.append(df)
    train_df = pd.concat(train_parts, ignore_index=True)

    test_parts = []
    for bname in TEST_BEARINGS:
        df = pd.read_csv(FEMTO_PROCESSED / f"test_{bname}_enhanced_features.csv")
        df["bearing_id"] = bname
        df["condition"]  = bearing_condition(bname)
        total            = int(meta[bname]["total_life"])
        df["true_rul"]   = total - df["cycle"]
        for col in ["cond_1", "cond_2", "cond_3"]:
            df[col] = 0
        df[f"cond_{bearing_condition(bname)}"] = 1
        test_parts.append(df)
    test_df = pd.concat(test_parts, ignore_index=True)

    print(f"Train: {len(train_df)} rows ({len(TRAIN_BEARINGS)} bearings)")
    print(f"Test:  {len(test_df)} rows ({len(TEST_BEARINGS)} bearings)")
    print(f"Enhanced feature count: {len(ENHANCED_COLS)}")

    # ── T1 Oracle Baseline ────────────────────────────────────────────────────
    # Load original datasets for apples-to-apples oracle comparison
    orig_train = pd.read_csv(FEMTO_PROCESSED / "femto_train.csv")
    orig_test  = pd.read_csv(FEMTO_PROCESSED / "femto_test_input.csv")
    orig_train["condition"] = orig_train["bearing_id"].map(bearing_condition)
    orig_test["condition"]  = orig_test["bearing_id"].map(bearing_condition)
    def _true_rul_orig(row):
        return int(meta[row["bearing_id"]]["total_life"]) - int(row["cycle"])
    orig_test["true_rul"] = orig_test.apply(_true_rul_orig, axis=1)

    t1_oracle_cc = compute_t1_oracle_baseline(orig_train, orig_test, meta)

    # ── Track 3 pipelines ────────────────────────────────────────────────────
    if_res = run_if_pipeline(train_df.copy(), test_df.copy(), meta)
    ae_res = run_ae_pipeline(train_df.copy(), test_df.copy(), meta)

    # ── Assemble predictions CSV ──────────────────────────────────────────────
    out_df = test_df[["bearing_id", "cycle", "true_rul"]].copy()
    out_df["if_pseudo_rul"]   = np.round(if_res["pseudo_rul"],   2)
    out_df["if_pseudo_pred"]  = np.round(if_res["pseudo_pred"],  2)
    out_df["if_oracle_pred"]  = np.round(if_res["oracle_pred"],  2)
    out_df["ae_pseudo_rul"]   = np.round(ae_res["pseudo_rul"],   2)
    out_df["ae_pseudo_pred"]  = np.round(ae_res["pseudo_pred"],  2)
    out_df["ae_oracle_pred"]  = np.round(ae_res["oracle_pred"],  2)

    pred_out = FEMTO_RESULTS / "track3_enhanced_predictions.csv"
    out_df.to_csv(pred_out, index=False)
    print(f"\nSaved: {pred_out}  ({len(out_df)} rows)")

    # ── Evaluation ────────────────────────────────────────────────────────────
    eval_rows = []
    for det, pseudo_col, pred_col, oracle_col in [
        ("IF", "if_pseudo_rul", "if_pseudo_pred", "if_oracle_pred"),
        ("AE", "ae_pseudo_rul", "ae_pseudo_pred", "ae_oracle_pred"),
    ]:
        m_pseudo = _eval(out_df, meta, pseudo_col)
        m_pred   = _eval(out_df, meta, pred_col)
        m_oracle = _eval(out_df, meta, oracle_col)
        for eval_type, key in [("TV", "tv"), ("CC", "cc")]:
            eval_rows.append({
                "detector":    det,
                "eval_type":   eval_type,
                "pseudo_rmse": round(m_pseudo[f"{key}_rmse"], 2),
                "true_rmse":   round(m_pred[f"{key}_rmse"],   2),
                "oracle_rmse": round(m_oracle[f"{key}_rmse"], 2),
                "gap":         round(m_pred[f"{key}_rmse"] - m_oracle[f"{key}_rmse"], 2),
                "pearson_r":   round(m_pred[f"{key}_r"],      4),
                "n":           m_pred[f"n_{key}"],
            })

    eval_df = pd.DataFrame(eval_rows)
    eval_out = FEMTO_RESULTS / "track3_enhanced_eval.csv"
    eval_df.to_csv(eval_out, index=False)
    print(f"Saved: {eval_out}")

    # ── Console summary ───────────────────────────────────────────────────────
    W = 90
    print("\n" + "=" * W)
    print("Track 3 Enhanced Features - Evaluation")
    print("=" * W)
    print(eval_df.to_string(index=False))

    # Key comparison
    t3_if_oracle_cc = eval_df[
        (eval_df["detector"] == "IF") & (eval_df["eval_type"] == "CC")
    ]["oracle_rmse"].values[0]
    print(f"\n{'-'*50}")
    print(f"  T1 Oracle CC-RMSE (IF, MinMaxScaler, alpha=1.0): {t1_oracle_cc:.2f}")
    print(f"  T3 Oracle CC-RMSE (IF, enhanced features):       {t3_if_oracle_cc:.2f}")
    delta = t3_if_oracle_cc - t1_oracle_cc
    pct   = delta / t1_oracle_cc * 100
    print(f"  Change: {delta:+.2f}  ({pct:+.1f}%)")
    if delta < -50:
        print("  -> Enhanced features IMPROVE the feature space (oracle dropped).")
    elif delta > 50:
        print("  -> Enhanced features WORSEN the feature space (oracle rose).")
    else:
        print("  -> No meaningful change in oracle. Feature space essentially equivalent.")
    print(f"{'-'*50}")

    print("\nCC per-bearing (pseudo_pred vs true_rul):")
    print(f"{'bearing':<12} {'true_rul':>9} {'IF_pseudo':>10} {'IF_oracle':>10} {'AE_pseudo':>10} {'AE_oracle':>10}")
    print("-" * 63)
    for bname in TEST_BEARINGS:
        sub  = out_df[out_df["bearing_id"] == bname]
        last = sub.loc[sub["cycle"].idxmax()]
        print(f"{bname:<12} {last['true_rul']:>9.0f} "
              f"{last['if_pseudo_pred']:>10.1f} {last['if_oracle_pred']:>10.1f} "
              f"{last['ae_pseudo_pred']:>10.1f} {last['ae_oracle_pred']:>10.1f}")
