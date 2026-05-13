"""
04d_pipeline_combined.py
Track 4: T2 scaling regime + T3 enhanced feature set combined.

Changes vs T2 (04b_pipeline_adapted.py):
  1. 39 enhanced features from 02b_feature_extraction_enhanced.py
     (18 original + 4 HOT + 8 WPD + 8 fault-freq + 1 PCA HI)
  2. AE architecture: 42->16->4->16->42  (was 21->8->2->8->21)

Unchanged from T2:
  - RobustScaler(q=25-75) per condition, transductive (train+test combined)
  - Ridge alpha=100
  - AE seeds [42, 43, 44]
  - Smoothing window=5, contamination=0.20
  - Pseudo-label generation: zone-based (top-20% anomaly score per bearing)
  - Both pseudo Ridge and oracle Ridge

Outputs:
  results/femto/track4_combined_predictions.csv
  results/femto/track4_combined_eval.csv
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
    AE_LR, AE_N_EPOCHS,
    CONDITION_COLS, DEGRADATION_QUANTILE,
    FEATURE_COLS, FEMTO_PROCESSED, FEMTO_RESULTS,
    IF_CONTAMINATION, IF_MAX_SAMPLES, IF_N_ESTIMATORS,
    ROLLING_WINDOW, TEST_BEARINGS, TRAIN_BEARINGS, TV_STEP,
    bearing_condition,
)

RIDGE_ALPHA_T4 = 100.0
AE_SEEDS       = [42, 43, 44]
AE_LATENT_T4   = 4
AE_HIDDEN_T4   = 16

# ── Enhanced feature columns (must match 02b output) ─────────────────────────

_HOT   = ["horiz_skewness", "horiz_shape_factor", "vert_skewness", "vert_shape_factor"]
_WPD   = (
    [f"horiz_wpd_{n}" for n in ["aaaa", "aaad", "aada", "addd"]] +
    [f"vert_wpd_{n}"  for n in ["aaaa", "aaad", "aada", "addd"]]
)
_FAULT = (
    [f"horiz_{f}" for f in ["BPFO", "BPFI", "BSF", "FTF"]] +
    [f"vert_{f}"  for f in ["BPFO", "BPFI", "BSF", "FTF"]]
)
ENHANCED_COLS = FEATURE_COLS + _HOT + _WPD + _FAULT + ["pca_hi"]  # 39


# ── Autoencoder (42->16->4->16->42) ──────────────────────────────────────────

class _AE(nn.Module):
    def __init__(self, input_dim: int = 42):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, AE_HIDDEN_T4), nn.ReLU(),
            nn.Linear(AE_HIDDEN_T4, AE_LATENT_T4), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(AE_LATENT_T4, AE_HIDDEN_T4), nn.ReLU(),
            nn.Linear(AE_HIDDEN_T4, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ── Scaling (transductive RobustScaler, identical to T2) ─────────────────────

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


# ── Shared helpers ────────────────────────────────────────────────────────────

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
    return Ridge(alpha=RIDGE_ALPHA_T4, random_state=42).fit(X, y)


def _train_ae(X_ae, seed):
    torch.manual_seed(seed)
    model  = _AE(input_dim=X_ae.shape[1])
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
    }


# ── IF pipeline ───────────────────────────────────────────────────────────────

def run_if_pipeline(train_df, test_df, meta):
    print("\n[IF] Transductive RobustScaler fitting (39 enhanced features)...")
    _, X_tr, X_te = _fit_scalers_transductive(train_df, test_df, ENHANCED_COLS)

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

    X_ridge_tr   = np.column_stack([X_tr, smooth_tr])
    ridge_pseudo = _train_ridge(X_ridge_tr, pseudo_tr)
    ridge_oracle = _train_ridge(X_ridge_tr, true_tr)

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


# ── AE pipeline ───────────────────────────────────────────────────────────────

def run_ae_pipeline(train_df, test_df, meta, seed):
    print(f"\n[AE seed={seed}] Transductive RobustScaler fitting...")
    _, X_tr_feat, X_te_feat = _fit_scalers_transductive(
        train_df, test_df, ENHANCED_COLS
    )

    cond_tr = train_df[CONDITION_COLS].values.astype(float)
    cond_te = test_df[CONDITION_COLS].values.astype(float)
    X_ae_tr = np.column_stack([X_tr_feat, cond_tr])   # 42-dim
    X_ae_te = np.column_stack([X_te_feat, cond_te])

    print(f"[AE seed={seed}] Training autoencoder "
          f"({X_ae_tr.shape[1]}->{AE_HIDDEN_T4}->{AE_LATENT_T4}->"
          f"{AE_HIDDEN_T4}->{X_ae_tr.shape[1]})...")
    model, ae_err_tr = _train_ae(X_ae_tr, seed)

    smooth_tr = _smooth_per_bearing(train_df, ae_err_tr)

    print(f"[AE seed={seed}] Generating pseudo-labels...")
    pseudo_tr = _generate_pseudo_labels(
        train_df["bearing_id"], train_df["cycle"], smooth_tr, meta
    )
    true_tr = train_df["rul"].values.astype(float)

    X_ridge_tr   = np.column_stack([X_tr_feat, smooth_tr])
    ridge_pseudo = _train_ridge(X_ridge_tr, pseudo_tr)
    ridge_oracle = _train_ridge(X_ridge_tr, true_tr)

    print(f"[AE seed={seed}] Predicting on test bearings...")
    X_ae_te_t = torch.tensor(X_ae_te, dtype=torch.float32)
    with torch.no_grad():
        recon     = model(X_ae_te_t)
        ae_err_te = torch.mean((recon - X_ae_te_t) ** 2, dim=1).numpy()

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
        for col in CONDITION_COLS:
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
        for col in CONDITION_COLS:
            df[col] = 0
        df[f"cond_{bearing_condition(bname)}"] = 1
        test_parts.append(df)
    test_df = pd.concat(test_parts, ignore_index=True)

    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")
    print(f"Enhanced features: {len(ENHANCED_COLS)}  AE input: {len(ENHANCED_COLS)+3}")

    # ── IF ────────────────────────────────────────────────────────────────────
    if_res = run_if_pipeline(train_df.copy(), test_df.copy(), meta)

    # ── AE (3 seeds) ──────────────────────────────────────────────────────────
    ae_results = {}
    for seed in AE_SEEDS:
        ae_results[seed] = run_ae_pipeline(
            train_df.copy(), test_df.copy(), meta, seed
        )

    # ── Assemble predictions CSV ──────────────────────────────────────────────
    out_df = test_df[["bearing_id", "cycle", "true_rul"]].copy()
    out_df["if_pseudo_rul"]   = np.round(if_res["pseudo_rul"],   2)
    out_df["if_pseudo_pred"]  = np.round(if_res["pseudo_pred"],  2)
    out_df["if_oracle_pred"]  = np.round(if_res["oracle_pred"],  2)
    for seed, res in ae_results.items():
        out_df[f"ae{seed}_pseudo_rul"]  = np.round(res["pseudo_rul"],  2)
        out_df[f"ae{seed}_pseudo_pred"] = np.round(res["pseudo_pred"], 2)
        out_df[f"ae{seed}_oracle_pred"] = np.round(res["oracle_pred"], 2)

    pred_out = FEMTO_RESULTS / "track4_combined_predictions.csv"
    out_df.to_csv(pred_out, index=False)
    print(f"\nSaved: {pred_out}  ({len(out_df)} rows)")

    # ── Evaluation ────────────────────────────────────────────────────────────
    eval_rows = []

    def _add_rows(detector, pseudo_col, pred_col, oracle_col):
        m_pseudo = _eval(out_df, meta, pseudo_col)
        m_pred   = _eval(out_df, meta, pred_col)
        m_oracle = _eval(out_df, meta, oracle_col)
        for eval_type, key in [("TV", "tv"), ("CC", "cc")]:
            eval_rows.append({
                "detector":    detector,
                "eval_type":   eval_type,
                "pseudo_rmse": round(m_pseudo[f"{key}_rmse"], 2),
                "true_rmse":   round(m_pred[f"{key}_rmse"],   2),
                "oracle_rmse": round(m_oracle[f"{key}_rmse"], 2),
                "gap":         round(m_pred[f"{key}_rmse"] - m_oracle[f"{key}_rmse"], 2),
                "pearson_r":   round(m_pred[f"{key}_r"],      4),
                "n":           m_pred[f"n_{key}"],
            })

    _add_rows("IF", "if_pseudo_rul", "if_pseudo_pred", "if_oracle_pred")
    for seed in AE_SEEDS:
        _add_rows(
            f"AE_s{seed}",
            f"ae{seed}_pseudo_rul", f"ae{seed}_pseudo_pred", f"ae{seed}_oracle_pred",
        )

    # AE mean+/-std across seeds
    eval_rows_base = [r for r in eval_rows if not r["detector"].startswith("AE_s")]
    for eval_type in ["TV", "CC"]:
        ae_sub = [r for r in eval_rows if r["eval_type"] == eval_type
                  and r["detector"].startswith("AE_s")]
        row = {"detector": "AE_mean", "eval_type": eval_type, "n": ae_sub[0]["n"]}
        for col in ["pseudo_rmse", "true_rmse", "oracle_rmse", "gap", "pearson_r"]:
            vals = [r[col] for r in ae_sub]
            mn   = float(np.mean(vals))
            sd   = float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
            row[col] = f"{mn:.2f}+/-{sd:.2f}"
        eval_rows.append(row)

    eval_df  = pd.DataFrame(eval_rows)
    col_order = ["detector", "eval_type", "pseudo_rmse", "true_rmse",
                 "oracle_rmse", "gap", "pearson_r", "n"]
    eval_df  = eval_df[col_order]
    eval_out = FEMTO_RESULTS / "track4_combined_eval.csv"
    eval_df.to_csv(eval_out, index=False)
    print(f"Saved: {eval_out}")

    # ── Console summary ───────────────────────────────────────────────────────
    W = 95
    print("\n" + "=" * W)
    print("Track 4 Combined (T2 scaling + T3 enhanced features) - Evaluation")
    print("=" * W)
    print(eval_df.to_string(index=False))

    print("\nCC per-bearing detail (pseudo_pred vs true_rul):")
    print(f"{'bearing':<12} {'true_rul':>9} {'IF':>9} {'AE42':>9} {'AE43':>9} {'AE44':>9}")
    print("-" * 58)
    for bname in TEST_BEARINGS:
        sub  = out_df[out_df["bearing_id"] == bname]
        last = sub.loc[sub["cycle"].idxmax()]
        print(f"{bname:<12} {last['true_rul']:>9.0f} "
              f"{last['if_pseudo_pred']:>9.1f} "
              f"{last['ae42_pseudo_pred']:>9.1f} "
              f"{last['ae43_pseudo_pred']:>9.1f} "
              f"{last['ae44_pseudo_pred']:>9.1f}")

    # ── Key oracle comparison ─────────────────────────────────────────────────
    t2_if_oracle_cc = 735.20   # from track2_adapted_eval.csv
    t4_if_oracle_cc = float(
        eval_df[(eval_df["detector"] == "IF") & (eval_df["eval_type"] == "CC")
               ]["oracle_rmse"].values[0]
    )
    delta = t4_if_oracle_cc - t2_if_oracle_cc
    pct   = delta / t2_if_oracle_cc * 100
    print(f"\n{'-'*55}")
    print(f"  T2 Oracle CC-RMSE (IF, 18 feat, RobustScaler): {t2_if_oracle_cc:.2f}")
    print(f"  T4 Oracle CC-RMSE (IF, 39 feat, RobustScaler): {t4_if_oracle_cc:.2f}")
    print(f"  Change: {delta:+.2f}  ({pct:+.1f}%)")
    if delta < -100:
        print("  -> Scenario P: feature engineering + scaling together improve oracle.")
    elif abs(delta) <= 100:
        print("  -> Scenario Q: scaling absorbs feature engineering benefit.")
    else:
        print("  -> Scenario R: 39 features worsen performance under RobustScaler.")
    print(f"{'-'*55}")
