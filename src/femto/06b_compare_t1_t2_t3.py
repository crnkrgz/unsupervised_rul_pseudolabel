"""
06b_compare_t1_t2_t3.py
Three-track comparison: T1 (strict) vs T2 (adapted) vs T3 (enhanced features).

Reads:
  results/femto/track1_strict_eval.csv       -- pseudo only (T1)
  results/femto/track2_adapted_eval.csv      -- pseudo + oracle (T2)
  results/femto/track3_enhanced_eval.csv     -- pseudo + oracle (T3)

Also reads prediction CSVs for per-bearing CC detail.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from femto.config import FEMTO_RESULTS, TEST_BEARINGS


def _get(df, detector, eval_type, col, default=None):
    row = df[(df["detector"] == detector) & (df["eval_type"] == eval_type)]
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
    t1_path = FEMTO_RESULTS / "track1_strict_eval.csv"
    t2_path = FEMTO_RESULTS / "track2_adapted_eval.csv"
    t3_path = FEMTO_RESULTS / "track3_enhanced_eval.csv"

    for p in [t1_path, t2_path, t3_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    # T1: columns = model, tv_rmse, n_tv, cc_rmse, n_cc  (pseudo only)
    # Map T1 to detector names used in T2/T3
    t1_raw = pd.read_csv(t1_path)
    t1 = {}
    for _, row in t1_raw.iterrows():
        det = row["model"]
        t1[(det, "TV")] = row["tv_rmse"]
        t1[(det, "CC")] = row["cc_rmse"]

    t2 = pd.read_csv(t2_path)
    t3 = pd.read_csv(t3_path)

    W = 115
    print("=" * W)
    print("FEMTO Track 1 / Track 2 / Track 3 Comparison")
    print("T1: MinMaxScaler+alpha=1.0+18feat | "
          "T2: RobustScaler+transductive+alpha=100+18feat | "
          "T3: MinMaxScaler+alpha=1.0+39feat")
    print("=" * W)

    rows = []
    for det in ["IF", "AE"]:
        for eval_type in ["TV", "CC"]:
            t1_pseudo = t1.get((det, eval_type))
            t2_pseudo  = _get(t2, det if det == "IF" else "AE_s42", eval_type, "true_rmse")
            t2_oracle  = _get(t2, det if det == "IF" else "AE_s42", eval_type, "oracle_rmse")
            t3_pseudo  = _get(t3, det, eval_type, "true_rmse")
            t3_oracle  = _get(t3, det, eval_type, "oracle_rmse")

            rows.append({
                "detector":      det,
                "eval_type":     eval_type,
                "T1_pseudo":     round(t1_pseudo, 2) if t1_pseudo is not None else None,
                "T2_pseudo":     round(t2_pseudo, 2) if t2_pseudo is not None else None,
                "T3_pseudo":     round(t3_pseudo, 2) if t3_pseudo is not None else None,
                "T3vsT1_%":      _pct(t3_pseudo, t1_pseudo),
                "T2_oracle":     round(t2_oracle, 2) if t2_oracle is not None else None,
                "T3_oracle":     round(t3_oracle, 2) if t3_oracle is not None else None,
                "Oracle_T3vsT2_%": _pct(t3_oracle, t2_oracle),
            })

    comp_df = pd.DataFrame(rows)
    print(comp_df.to_string(index=False))

    out = FEMTO_RESULTS / "t1_t2_t3_comparison.csv"
    comp_df.to_csv(out, index=False)
    print(f"\nSaved: {out}")

    # -- Oracle diagnostic -----------------------------------------------------
    print("\n" + "-" * W)
    print("Oracle Analysis (Ridge trained on true RUL -- measures feature-space quality)")
    print("-" * W)

    for det in ["IF", "AE"]:
        t2_det = det if det == "IF" else "AE_s42"
        t2_occ = _get(t2, t2_det, "CC", "oracle_rmse")
        t3_occ = _get(t3, det,    "CC", "oracle_rmse")
        if t2_occ is not None and t3_occ is not None:
            delta = t3_occ - t2_occ
            pct   = _pct(t3_occ, t2_occ)
            arrow = "improved improved" if delta < -50 else ("worsened worsened" if delta > 50 else "~ no change")
            print(f"  {det}: T2 oracle={t2_occ:.1f}  T3 oracle={t3_occ:.1f}  "
                  f"delta={delta:+.1f} ({pct:+.1f}%)  {arrow}")
        else:
            print(f"  {det}: T2 oracle={t2_occ}  T3 oracle={t3_occ}")

    # -- Per-bearing CC detail (T1 vs T3, IF) ---------------------------------
    t1_pred_path = FEMTO_RESULTS / "track1_strict_predictions.csv"
    t3_pred_path = FEMTO_RESULTS / "track3_enhanced_predictions.csv"
    if t1_pred_path.exists() and t3_pred_path.exists():
        p1 = pd.read_csv(t1_pred_path)
        p3 = pd.read_csv(t3_pred_path)
        print("\n" + "-" * W)
        print("Per-Bearing CC: IF Predictions T1 vs T3")
        print(f"{'bearing':<12} {'true_rul':>9} {'T1_IF':>9} {'T3_IF':>9} {'diff':>8} "
              f"{'T1_AE':>9} {'T3_AE':>9} {'diff':>8}")
        print("-" * 80)
        for bname in TEST_BEARINGS:
            s1  = p1[p1["bearing_id"] == bname]
            s3  = p3[p3["bearing_id"] == bname]
            l1  = s1.loc[s1["cycle"].idxmax()]
            l3  = s3.loc[s3["cycle"].idxmax()]
            tr  = l1["true_rul"]
            t1_if = l1.get("if_predicted_rul", float("nan"))
            t3_if = l3.get("if_pseudo_pred",   float("nan"))
            t1_ae = l1.get("ae_predicted_rul", float("nan"))
            t3_ae = l3.get("ae_pseudo_pred",   float("nan"))
            print(f"{bname:<12} {tr:>9.0f} {t1_if:>9.1f} {t3_if:>9.1f} "
                  f"{t3_if-t1_if:>+8.1f} {t1_ae:>9.1f} {t3_ae:>9.1f} {t3_ae-t1_ae:>+8.1f}")

    # -- Scenario classification -----------------------------------------------
    print("\n" + "-" * W)
    print("Scenario Classification (T3 CC pseudo_rmse):")
    for det in ["IF", "AE"]:
        rmse = _get(t3, det, "CC", "true_rmse")
        if rmse is None:
            continue
        if rmse < 200:
            scenario = "A -- Major improvement. Feature engineering confirms paradigm."
        elif rmse < 500:
            scenario = "B -- Moderate. Limited transfer despite richer features."
        elif rmse < 2000:
            scenario = "C -- Minor/structural limits. Feature space not sufficient."
        else:
            scenario = "D -- No meaningful improvement. Pipeline is domain-specific."
        print(f"  {det}: CC-RMSE={rmse:.1f} -> Scenario {scenario}")
