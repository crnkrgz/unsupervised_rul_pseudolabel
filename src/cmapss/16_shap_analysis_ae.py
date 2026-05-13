"""
16_shap_analysis_ae.py
SHAP feature attribution for AE-based pseudo-label Ridge models.

For each subset, runs:
  - AE pipeline (optimal config: FD001/FD003/FD004 ld=2 ep=50; FD002 ld=4 ep=30)
  - IF pipeline  (same cluster-scaling as 06_shap_analysis.py)
  - Supervised Ridge baseline (true RUL, same features as AE model)

Computes LinearExplainer SHAP for all three models and generates:
  - results/cmapss/shap_ae_pseudo_summary.csv
  - results/cmapss/shap_three_way_comparison.csv
  - results/cmapss/figures/shap_ae_top10_{subset}.png  (4 figures)
"""

import importlib.util
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from cmapss.config import (
    KMEANS_K_MULTI,
    RESULTS_DIR,
    RIDGE_ALPHA,
    ROLLING_WINDOW,
    SEED,
    SINGLE_CONDITION,
    SUBSETS,
    THRESHOLD_MULTI,
    THRESHOLD_SINGLE,
)

_SRC     = Path(__file__).parent
_OUT_DIR = RESULTS_DIR / "cmapss"
_FIG_DIR = _OUT_DIR / "figures"

# ── Subset-specific optimal AE configs ────────────────────────────────────────
AE_OPTIMAL = {
    "FD001": {"latent_dim": 2, "n_epochs": 50},
    "FD002": {"latent_dim": 4, "n_epochs": 30},
    "FD003": {"latent_dim": 2, "n_epochs": 50},
    "FD004": {"latent_dim": 2, "n_epochs": 50},
}

# IF optimal (from 06_shap_analysis.py)
IF_OPTIMAL = {
    "FD001": {"n_estimators": 200, "threshold": 0.20},
    "FD002": {"n_estimators": 200, "threshold": 0.15},
    "FD003": {"n_estimators": 200, "threshold": 0.20},
    "FD004": {"n_estimators": 200, "threshold": 0.15},
}


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, _SRC / filename)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Autoencoder ────────────────────────────────────────────────────────────────

class _AE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 8), nn.ReLU(),
                                     nn.Linear(8, latent_dim), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 8), nn.ReLU(),
                                     nn.Linear(8, input_dim))

    def forward(self, x):
        return self.decoder(self.encoder(x))


def _train_ae(X: np.ndarray, latent_dim: int, n_epochs: int, seed: int) -> _AE:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = _AE(X.shape[1], latent_dim)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit  = nn.MSELoss()
    dl    = DataLoader(TensorDataset(torch.FloatTensor(X)),
                       batch_size=256, shuffle=True,
                       generator=torch.Generator().manual_seed(seed))
    model.train()
    for _ in range(n_epochs):
        for (b,) in dl:
            opt.zero_grad(); crit(model(b), b).backward(); opt.step()
    model.eval()
    return model


def _ae_score(model: _AE, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        t = torch.FloatTensor(X)
        return torch.norm(t - model(t), dim=1).numpy().astype(float)


def _smooth(df: pd.DataFrame, raw: np.ndarray) -> np.ndarray:
    tmp = pd.DataFrame({"uid": df["unit_id"].values, "cyc": df["cycle"].values,
                        "s": raw, "p": np.arange(len(raw))})
    tmp = tmp.sort_values(["uid", "cyc"])
    sm  = tmp.groupby("uid")["s"].transform(
        lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).mean())
    out = np.empty(len(raw))
    out[tmp["p"].values] = sm.values
    return out


# ── Shared data loading (clustering + scaling) ────────────────────────────────

def _load_scaled(subset_name: str):
    """Return (train_df, test_df, raw, feature_cols, k, pipe_mod, clust_mod)."""
    name      = subset_name.upper()
    load_mod  = _load_module("load_data",    "01_load_data.py")
    prep_mod  = _load_module("preprocess",   "02_preprocess.py")
    pipe_mod  = _load_module("global_pipe",  "03a_global_pipeline.py")
    clust_mod = _load_module("cluster_pipe", "03b_cluster_pipeline.py")

    raw    = load_mod.load_subset(name)
    result = prep_mod.preprocess_subset(raw, name, scale=False)
    tr, te = result["train"], result["test"]
    fc     = result["feature_cols"]

    k = 1 if name in SINGLE_CONDITION else KMEANS_K_MULTI
    tr, te, _ = clust_mod.cluster_op_settings(tr, te, k)
    tr, te, _ = clust_mod.per_cluster_minmax(tr, te, fc, k)

    return tr, te, raw, fc, k, pipe_mod, clust_mod


# ── AE pipeline data builder ──────────────────────────────────────────────────

def build_ae_pipeline_data(subset_name: str) -> dict:
    name = subset_name.upper()
    cfg  = AE_OPTIMAL[name]
    thr  = THRESHOLD_SINGLE if name in SINGLE_CONDITION else THRESHOLD_MULTI

    tr, te, raw, fc, k, pipe_mod, _ = _load_scaled(name)

    X_tr  = tr[fc].values.astype(np.float32)
    ae    = _train_ae(X_tr, cfg["latent_dim"], cfg["n_epochs"], SEED)

    raw_tr  = _ae_score(ae, X_tr)
    sm_tr   = _smooth(tr, raw_tr)
    prl     = pipe_mod.generate_pseudo_labels(sm_tr, thr)

    X_tr_r        = np.column_stack([X_tr, sm_tr])
    pseudo_model  = pipe_mod.train_ridge(X_tr_r, prl)
    true_model    = pipe_mod.train_ridge(X_tr_r, tr["true_RUL"].values)

    X_te        = te[fc].values.astype(np.float32)
    raw_te      = _ae_score(ae, X_te)
    sm_te       = _smooth(te, raw_te)
    X_te_r_full = np.column_stack([X_te, sm_te])
    last_idx    = te.groupby("unit_id")["cycle"].idxmax()
    is_last     = te.index.isin(last_idx)
    X_te_r      = X_te_r_full[is_last]

    feature_names = fc + ["reconstruction_error"]

    return {
        "subset":        name,
        "train_df":      tr,
        "X_train_r":     X_tr_r,
        "X_test_last_r": X_te_r,
        "pseudo_model":  pseudo_model,
        "true_model":    true_model,
        "feature_names": feature_names,
    }


# ── IF pipeline data builder (mirrors 06_shap_analysis.py) ───────────────────

def build_if_pipeline_data(subset_name: str) -> dict:
    name = subset_name.upper()
    cfg  = IF_OPTIMAL[name]
    thr  = cfg["threshold"]

    tr, te, raw, fc, k, pipe_mod, _ = _load_scaled(name)

    X_tr = tr[fc].values
    clf  = pipe_mod.fit_isolation_forest(X_tr, cfg["n_estimators"])

    raw_tr = pipe_mod.score_anomaly(clf, X_tr)
    sm_tr  = pipe_mod.smooth_scores_per_unit(tr, raw_tr)
    prl    = pipe_mod.generate_pseudo_labels(sm_tr, thr)

    X_tr_r       = np.column_stack([X_tr, sm_tr])
    pseudo_model = pipe_mod.train_ridge(X_tr_r, prl)
    true_model   = pipe_mod.train_ridge(X_tr_r, tr["true_RUL"].values)

    X_te        = te[fc].values
    raw_te      = pipe_mod.score_anomaly(clf, X_te)
    sm_te       = pipe_mod.smooth_scores_per_unit(te, raw_te)
    X_te_r_full = np.column_stack([X_te, sm_te])
    last_idx    = te.groupby("unit_id")["cycle"].idxmax()
    is_last     = te.index.isin(last_idx)
    X_te_r      = X_te_r_full[is_last]

    return {
        "subset":        name,
        "X_train_r":     X_tr_r,
        "X_test_last_r": X_te_r,
        "pseudo_model":  pseudo_model,
        "true_model":    true_model,
        "feature_names": fc + ["anomaly_score"],
    }


# ── SHAP computation ──────────────────────────────────────────────────────────

def compute_shap(data: dict) -> dict:
    X_tr = data["X_train_r"]
    X_te = data["X_test_last_r"]
    fn   = data["feature_names"]

    exp_p = shap.LinearExplainer(data["pseudo_model"], X_tr)
    exp_t = shap.LinearExplainer(data["true_model"],   X_tr)

    sv_p = exp_p.shap_values(X_te)
    sv_t = exp_t.shap_values(X_te)

    ma_p = np.abs(sv_p).mean(axis=0)
    ma_t = np.abs(sv_t).mean(axis=0)

    return {
        "sv_pseudo":       sv_p,
        "sv_true":         sv_t,
        "mean_abs_pseudo": ma_p,
        "mean_abs_true":   ma_t,
        "rank_pseudo":     np.argsort(-ma_p),
        "rank_true":       np.argsort(-ma_t),
        "feature_names":   fn,
        "X_test_last_r":   X_te,
    }


# ── Table builders ────────────────────────────────────────────────────────────

def make_ae_pseudo_summary(all_ae_shap: dict) -> pd.DataFrame:
    """Per-feature mean |SHAP| for AE pseudo model, all subsets."""
    rows = []
    for subset, sh in all_ae_shap.items():
        fn  = sh["feature_names"]
        ma  = sh["mean_abs_pseudo"]
        rks = sh["rank_pseudo"]
        for rank_idx, fi in enumerate(rks):
            rows.append({
                "subset":              subset,
                "feature":             fn[fi],
                "mean_abs_shap_pseudo": round(float(ma[fi]), 6),
                "rank":                rank_idx + 1,
            })
    return pd.DataFrame(rows)


def make_three_way_comparison(
    all_if_shap: dict,
    all_ae_shap: dict,
) -> pd.DataFrame:
    """
    For each subset, merge IF pseudo / AE pseudo / supervised (AE basis) SHAP.
    Uses the union of all feature names (note: IF has 'anomaly_score',
    AE has 'reconstruction_error' -- both kept as separate rows).
    """
    rows = []
    for subset in SUBSETS:
        sh_if  = all_if_shap[subset]
        sh_ae  = all_ae_shap[subset]

        fn_if  = sh_if["feature_names"]
        fn_ae  = sh_ae["feature_names"]

        # Build lookup dicts: feature -> (mean_abs_shap, rank)
        def _lookup(fn, ma, rk_arr):
            rank_map = {fn[i]: r+1 for r, i in enumerate(rk_arr)}
            return {fn[i]: (float(ma[i]), rank_map[fn[i]]) for i in range(len(fn))}

        lif = _lookup(fn_if, sh_if["mean_abs_pseudo"], sh_if["rank_pseudo"])
        lae = _lookup(fn_ae, sh_ae["mean_abs_pseudo"], sh_ae["rank_pseudo"])
        lsup = _lookup(fn_ae, sh_ae["mean_abs_true"],   sh_ae["rank_true"])

        # Shared sensor features (exclude the score columns for union)
        sensor_feats = [f for f in fn_ae if f != "reconstruction_error"]

        for feat in sensor_feats:
            if_shap, if_rank   = lif.get(feat, (float("nan"), float("nan")))
            ae_shap, ae_rank   = lae.get(feat, (float("nan"), float("nan")))
            sup_shap, sup_rank = lsup.get(feat, (float("nan"), float("nan")))
            rows.append({
                "subset":         subset,
                "feature":        feat,
                "shap_if_pseudo": round(if_shap,  6),
                "rank_if":        if_rank,
                "shap_ae_pseudo": round(ae_shap,  6),
                "rank_ae":        ae_rank,
                "shap_supervised": round(sup_shap, 6),
                "rank_sup":       sup_rank,
            })

        # Score features separately
        for score_feat, lsrc, label in [
            ("anomaly_score",       lif,  "if_score"),
            ("reconstruction_error", lae, "ae_score"),
        ]:
            val, rnk = lsrc.get(score_feat, (float("nan"), float("nan")))
            rows.append({
                "subset":          subset,
                "feature":         score_feat,
                "shap_if_pseudo":  round(val, 6) if label == "if_score" else float("nan"),
                "rank_if":         rnk           if label == "if_score" else float("nan"),
                "shap_ae_pseudo":  round(val, 6) if label == "ae_score" else float("nan"),
                "rank_ae":         rnk           if label == "ae_score" else float("nan"),
                "shap_supervised": float("nan"),
                "rank_sup":        float("nan"),
            })

    return pd.DataFrame(rows)


# ── Bar-chart figure ──────────────────────────────────────────────────────────

def plot_ae_top10(sh: dict, subset: str, out_path: Path) -> None:
    fn = sh["feature_names"]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 6))

    top10p = sh["rank_pseudo"][:10]
    axL.barh([fn[i] for i in top10p[::-1]],
             [sh["mean_abs_pseudo"][i] for i in top10p[::-1]], color="#2E86AB")
    axL.set_xlabel("Mean |SHAP Value|")
    axL.set_title("AE Pseudo-Label Model")
    axL.grid(axis="x", linestyle="--", alpha=0.4)

    top10t = sh["rank_true"][:10]
    axR.barh([fn[i] for i in top10t[::-1]],
             [sh["mean_abs_true"][i] for i in top10t[::-1]], color="#E63946")
    axR.set_xlabel("Mean |SHAP Value|")
    axR.set_title("Supervised Model (true RUL)")
    axR.grid(axis="x", linestyle="--", alpha=0.4)

    fig.suptitle(f"SHAP Feature Importance — AE Pseudo vs Supervised ({subset})",
                 fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.perf_counter()
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    _FIG_DIR.mkdir(parents=True, exist_ok=True)

    all_ae_data = {}
    all_ae_shap = {}
    all_if_data = {}
    all_if_shap = {}

    for subset in SUBSETS:
        cfg = AE_OPTIMAL[subset]
        print(f"\n[{subset}] AE pipeline  (ld={cfg['latent_dim']}, ep={cfg['n_epochs']})...")
        all_ae_data[subset] = build_ae_pipeline_data(subset)
        print(f"[{subset}] AE SHAP...")
        all_ae_shap[subset] = compute_shap(all_ae_data[subset])

        print(f"[{subset}] IF pipeline...")
        all_if_data[subset] = build_if_pipeline_data(subset)
        print(f"[{subset}] IF SHAP...")
        all_if_shap[subset] = compute_shap(all_if_data[subset])

    # ── Console: top-5 AE pseudo per subset ──────────────────────────────────
    W = 100
    print("\n" + "=" * W)
    print("Top-5 Features — AE Pseudo Model")
    print("=" * W)
    hdr = f"{'Subset':<8}"
    for i in range(1, 6):
        hdr += f"  {'#'+str(i)+' Feature':<22} {'SHAP':>7}"
    print(hdr)
    print("-" * W)
    for subset in SUBSETS:
        sh  = all_ae_shap[subset]
        fn  = sh["feature_names"]
        ma  = sh["mean_abs_pseudo"]
        rk  = sh["rank_pseudo"]
        row = f"{subset:<8}"
        for i in range(5):
            fi = rk[i]
            row += f"  {fn[fi]:<22} {ma[fi]:>7.4f}"
        print(row)

    # ── Console: 3-way comparison top-5 per subset ───────────────────────────
    print("\n" + "=" * W)
    print("Three-Way Comparison — Top-5 (IF Pseudo / AE Pseudo / Supervised)")
    print("=" * W)
    for subset in SUBSETS:
        sh_if = all_if_shap[subset]
        sh_ae = all_ae_shap[subset]
        print(f"\n  {subset}")
        print(f"  {'Rank':<6} {'IF Pseudo':<26} {'AE Pseudo':<26} {'Supervised (AE basis)'}")
        print(f"  {'-'*85}")
        for i in range(5):
            fn_if_feat = sh_if["feature_names"][sh_if["rank_pseudo"][i]]
            fn_ae_feat = sh_ae["feature_names"][sh_ae["rank_pseudo"][i]]
            fn_su_feat = sh_ae["feature_names"][sh_ae["rank_true"][i]]
            v_if = sh_if["mean_abs_pseudo"][sh_if["rank_pseudo"][i]]
            v_ae = sh_ae["mean_abs_pseudo"][sh_ae["rank_pseudo"][i]]
            v_su = sh_ae["mean_abs_true"][sh_ae["rank_true"][i]]
            print(f"  {i+1:<6} {fn_if_feat:<18} {v_if:>6.4f}   "
                  f"{fn_ae_feat:<18} {v_ae:>6.4f}   "
                  f"{fn_su_feat:<18} {v_su:>6.4f}")

    # ── Scenario check: reconstruction_error dominance ───────────────────────
    print("\n" + "=" * W)
    print("Scenario Check: reconstruction_error dominance in AE pseudo model")
    print("=" * W)
    for subset in SUBSETS:
        sh  = all_ae_shap[subset]
        fn  = sh["feature_names"]
        top = fn[sh["rank_pseudo"][0]]
        val = sh["mean_abs_pseudo"][sh["rank_pseudo"][0]]
        dominant = top == "reconstruction_error"
        # Fraction of total SHAP held by reconstruction_error
        re_idx = fn.index("reconstruction_error")
        total  = sh["mean_abs_pseudo"].sum()
        frac   = sh["mean_abs_pseudo"][re_idx] / total * 100
        verdict = "S1 (dominant)" if dominant else "S2 (spread)"
        print(f"  {subset}: top={top:<26} val={val:.4f}  "
              f"re_frac={frac:.1f}%  -> {verdict}")

    # ── Save tables ──────────────────────────────────────────────────────────
    ae_summary = make_ae_pseudo_summary(all_ae_shap)
    three_way  = make_three_way_comparison(all_if_shap, all_ae_shap)

    out1 = _OUT_DIR / "shap_ae_pseudo_summary.csv"
    out2 = _OUT_DIR / "shap_three_way_comparison.csv"
    ae_summary.to_csv(out1, index=False)
    three_way.to_csv(out2, index=False)
    print(f"\nSaved: {out1}")
    print(f"Saved: {out2}")

    # ── Figures ──────────────────────────────────────────────────────────────
    for subset in SUBSETS:
        fig_path = _FIG_DIR / f"shap_ae_top10_{subset}.png"
        plot_ae_top10(all_ae_shap[subset], subset, fig_path)
        print(f"Saved: {fig_path.name}")

    # ── Three-way comparison table (console, sensor features only, FD001) ────
    print("\n" + "=" * W)
    print("Three-way SHAP table (FD001, top-10 by AE pseudo rank):")
    print("=" * W)
    fd1_3w = three_way[three_way["subset"] == "FD001"].copy()
    fd1_3w = fd1_3w[fd1_3w["shap_ae_pseudo"].notna()].sort_values("rank_ae")
    print(fd1_3w[["feature", "shap_if_pseudo", "rank_if",
                   "shap_ae_pseudo", "rank_ae",
                   "shap_supervised", "rank_sup"]].head(12).to_string(index=False))

    print(f"\nTotal runtime: {(time.perf_counter()-t0)/60:.1f} min")
