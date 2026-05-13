"""
06d_compare_with_ensemble.py
T2 (best single-detector) vs T5 (naive ensemble) comparison.

Reads:
  results/femto/track2_adapted_eval.csv
  results/femto/track5_ensemble_eval.csv
  results/femto/track2_adapted_predictions.csv
  results/femto/track5_ensemble_predictions.csv

Output:
  results/femto/t2_vs_t5_ensemble.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from femto.config import FEMTO_RESULTS, TEST_BEARINGS


def _get(df, detector, variant, eval_type, col, default=None):
    mask = (df["eval_type"] == eval_type)
    if "detector" in df.columns:
        mask &= (df["detector"] == detector)
    if "variant" in df.columns and variant is not None:
        mask &= (df["variant"] == variant)
    row = df[mask]
    if row.empty:
        return default
    val = row.iloc[0].get(col, default)
    try:
        return float(val)
    except (ValueError, TypeError):
        return val


def _pct(new_val, old_val):
    try:
        return round((float(new_val) - float(old_val)) / float(old_val) * 100, 1)
    except (ValueError, TypeError, ZeroDivisionError):
        return None


if __name__ == "__main__":
    t2_path = FEMTO_RESULTS / "track2_adapted_eval.csv"
    t5_path = FEMTO_RESULTS / "track5_ensemble_eval.csv"
    for p in [t2_path, t5_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    t2 = pd.read_csv(t2_path)
    t5 = pd.read_csv(t5_path)

    W = 105
    print("=" * W)
    print("T2 (Best Single Detector) vs T5 (Naive Ensemble) Comparison")
    print("=" * W)

    rows = []
    specs = [
        ("IF",       "T2 IF",       t2, "IF",       None),
        ("AE",       "T2 AE",       t2, "AE_s42",   None),
        ("Mean",     "T5 Mean",     t5, "Ensemble",  "Mean"),
        ("Max",      "T5 Max",      t5, "Ensemble",  "Max"),
        ("IF-ref",   "T5 IF ref",   t5, "IF",        "reference"),
        ("AE-ref",   "T5 AE ref",   t5, "AE",        "reference"),
    ]

    for short, label, df_src, det, var in specs:
        for eval_type in ["TV", "CC"]:
            pseudo = _get(df_src, det, var, eval_type, "true_rmse")
            oracle = _get(df_src, det, var, eval_type, "oracle_rmse")
            pearsr = _get(df_src, det, var, eval_type, "pearson_r")
            rows.append({
                "track":      label,
                "eval_type":  eval_type,
                "pseudo_rmse": pseudo,
                "oracle_rmse": oracle,
                "pearson_r":   pearsr,
            })

    comp_df = pd.DataFrame(rows)
    print(comp_df.to_string(index=False))

    out = FEMTO_RESULTS / "t2_vs_t5_ensemble.csv"
    comp_df.to_csv(out, index=False)
    print(f"\nSaved: {out}")

    # ── Key metrics table ─────────────────────────────────────────────────────
    print("\n" + "-" * W)
    print("Key metrics (CC only):")
    print(f"{'Track':<16} {'Pseudo CC':>10} {'Oracle CC':>10} {'Pearson r CC':>13} {'vs T2-IF pseudo':>16}")
    print("-" * 60)
    t2_if_pseudo = _get(t2, "IF", None, "CC", "true_rmse")
    t2_if_oracle = _get(t2, "IF", None, "CC", "oracle_rmse")
    for short, label, df_src, det, var in specs:
        pseudo = _get(df_src, det, var, "CC", "true_rmse")
        oracle = _get(df_src, det, var, "CC", "oracle_rmse")
        r_cc   = _get(df_src, det, var, "CC", "pearson_r")
        pct    = _pct(pseudo, t2_if_pseudo)
        pct_s  = f"{pct:+.1f}%" if pct is not None else "N/A"
        p_s    = f"{pseudo:.1f}" if pseudo is not None else "N/A"
        o_s    = f"{oracle:.1f}" if oracle is not None else "N/A"
        r_s    = f"{r_cc:.4f}"   if r_cc   is not None else "N/A"
        print(f"{label:<16} {p_s:>10} {o_s:>10} {r_s:>13} {pct_s:>16}")

    # ── Scenario classification ───────────────────────────────────────────────
    print("\n" + "-" * W)
    t5_mean_cc = _get(t5, "Ensemble", "Mean", "CC", "true_rmse")
    t5_max_cc  = _get(t5, "Ensemble", "Max",  "CC", "true_rmse")
    best_cc    = min(v for v in [t5_mean_cc, t5_max_cc] if v is not None)
    t2_if_cc   = t2_if_pseudo

    print(f"T2 IF Pseudo CC-RMSE:     {t2_if_cc:.1f}")
    print(f"T5 Mean Pseudo CC-RMSE:   {t5_mean_cc:.1f}")
    print(f"T5 Max  Pseudo CC-RMSE:   {t5_max_cc:.1f}")
    print(f"Best ensemble:            {best_cc:.1f}  ({_pct(best_cc, t2_if_cc):+.1f}% vs T2 IF)")

    if best_cc < 1500:
        print("-> Scenario E1: Ensemble substantially outperforms best single detector.")
    elif best_cc <= t2_if_cc * 1.15:
        print("-> Scenario E2: Ensemble marginal — T2 IF already near optimum.")
    else:
        print("-> Scenario E3: Ensemble worse — AE noise pulls down IF's performance.")

    if t5_mean_cc is not None and t5_max_cc is not None:
        diff = abs(t5_mean_cc - t5_max_cc)
        if diff > 500:
            print(f"-> Scenario E4 also: Mean vs Max differ by {diff:.0f} — strategy choice is critical.")

    # ── Per-bearing CC detail ─────────────────────────────────────────────────
    p2_path = FEMTO_RESULTS / "track2_adapted_predictions.csv"
    p5_path = FEMTO_RESULTS / "track5_ensemble_predictions.csv"
    if p2_path.exists() and p5_path.exists():
        p2 = pd.read_csv(p2_path)
        p5 = pd.read_csv(p5_path)

        print("\n" + "-" * W)
        print("Per-Bearing CC: T2 IF vs T5 Ensemble variants")
        print(f"{'bearing':<12} {'true':>6} {'T2_IF':>8} {'T5_Mean':>8} {'T5_Max':>8} {'T2_AE42':>8}")
        print("-" * 55)
        for bname in TEST_BEARINGS:
            def _last(df, col):
                sub = df[df["bearing_id"] == bname]
                if sub.empty:
                    return float("nan")
                return float(sub.loc[sub["cycle"].idxmax(), col])

            tr     = _last(p2, "true_rul")
            t2_if  = _last(p2, "if_pseudo_pred")
            t2_ae  = _last(p2, "ae42_pseudo_pred")
            t5_mn  = _last(p5, "ens_mean_pseudo_pred")
            t5_mx  = _last(p5, "ens_max_pseudo_pred")
            print(f"{bname:<12} {tr:>6.0f} {t2_if:>8.1f} {t5_mn:>8.1f} {t5_mx:>8.1f} {t2_ae:>8.1f}")
