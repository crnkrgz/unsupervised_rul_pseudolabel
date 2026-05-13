"""
06_compare_t1_t2.py
Side-by-side comparison of Track 1 (strict transfer) vs Track 2 (targeted adaptation).

Reads:
  results/femto/track1_strict_eval.csv
  results/femto/track2_adapted_eval.csv

Output:
  results/femto/t1_vs_t2_comparison.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from femto.config import FEMTO_RESULTS, TEST_BEARINGS


def _pct(new_val, old_val):
    """Percent change (negative = improvement)."""
    try:
        v_new = float(new_val)
        v_old = float(old_val)
        return round((v_new - v_old) / v_old * 100, 1)
    except (ValueError, TypeError, ZeroDivisionError):
        return None


if __name__ == "__main__":
    t1_path = FEMTO_RESULTS / "track1_strict_eval.csv"
    t2_path = FEMTO_RESULTS / "track2_adapted_eval.csv"

    if not t1_path.exists():
        raise FileNotFoundError(f"Run 05_evaluation.py first: {t1_path}")
    if not t2_path.exists():
        raise FileNotFoundError(f"Run 04b_pipeline_adapted.py first: {t2_path}")

    t1 = pd.read_csv(t1_path)
    t2 = pd.read_csv(t2_path)

    # T1 columns: model, tv_rmse, n_tv, cc_rmse, n_cc
    # T2 columns: detector, eval_type, pseudo_rmse, true_rmse, oracle_rmse, gap, pearson_r, n
    # Map T1 model names to T2 detector names
    t1_map = {"IF": "IF", "AE": "AE_s42"}  # T1 used SEED=42 for AE

    rows = []

    def _get_t1(model, metric):
        row = t1[t1["model"] == model]
        if row.empty:
            return None
        return float(row.iloc[0][metric])

    def _get_t2(detector, eval_type, metric):
        row = t2[(t2["detector"] == detector) & (t2["eval_type"] == eval_type)]
        if row.empty:
            return None
        val = row.iloc[0][metric]
        try:
            return float(val)
        except (ValueError, TypeError):
            return str(val)

    for model_label, t1_name, t2_name in [
        ("IF",    "IF",  "IF"),
        ("AE",    "AE",  "AE_s42"),
        ("AE_mean", None, "AE_mean"),
    ]:
        for eval_type, t1_tv_key, t1_cc_key in [("TV", "tv_rmse", None), ("CC", None, "cc_rmse")]:
            t1_key = t1_tv_key if eval_type == "TV" else t1_cc_key

            t1_rmse = _get_t1(t1_name, t1_key) if t1_name else None
            t2_rmse = _get_t2(t2_name, eval_type, "true_rmse")
            t2_oracle = _get_t2(t2_name, eval_type, "oracle_rmse")
            t2_pseudo = _get_t2(t2_name, eval_type, "pseudo_rmse")
            t2_r      = _get_t2(t2_name, eval_type, "pearson_r")
            t2_n      = _get_t2(t2_name, eval_type, "n")

            pct = _pct(t2_rmse, t1_rmse) if t1_rmse is not None else None

            rows.append({
                "detector":     model_label,
                "eval_type":    eval_type,
                "t1_rmse":      t1_rmse,
                "t2_pseudo_rmse": t2_pseudo,
                "t2_true_rmse": t2_rmse,
                "t2_oracle_rmse": t2_oracle,
                "pct_change":   pct,
                "pearson_r":    t2_r,
                "n":            t2_n,
            })

    comp_df = pd.DataFrame(rows)
    out = FEMTO_RESULTS / "t1_vs_t2_comparison.csv"
    comp_df.to_csv(out, index=False)

    # ── Console ───────────────────────────────────────────────────────────────
    W = 100
    print("=" * W)
    print("FEMTO Track 1 vs Track 2 Comparison")
    print("Adaptations: RobustScaler + transductive fit + Ridge alpha=100 + AE 3 seeds")
    print("=" * W)
    print(comp_df.to_string(index=False))
    print(f"\nSaved: {out}")

    # ── Per-bearing CC comparison ─────────────────────────────────────────────
    t1_preds = FEMTO_RESULTS / "track1_strict_predictions.csv"
    t2_preds = FEMTO_RESULTS / "track2_adapted_predictions.csv"
    if t1_preds.exists() and t2_preds.exists():
        p1 = pd.read_csv(t1_preds)
        p2 = pd.read_csv(t2_preds)

        print("\nPer-Bearing CC: IF predictions T1 vs T2")
        print(f"{'bearing':<12} {'true_rul':>9} {'T1_IF':>9} {'T2_IF':>9} {'diff':>8} {'T1_AE':>9} {'T2_AE42':>9} {'diff':>8}")
        print("-" * 80)
        for bname in TEST_BEARINGS:
            s1 = p1[p1["bearing_id"] == bname]
            s2 = p2[p2["bearing_id"] == bname]
            l1 = s1.loc[s1["cycle"].idxmax()]
            l2 = s2.loc[s2["cycle"].idxmax()]
            tr = l1["true_rul"]
            t1_if  = l1.get("if_predicted_rul", float("nan"))
            t2_if  = l2.get("if_pseudo_pred",   float("nan"))
            t1_ae  = l1.get("ae_predicted_rul", float("nan"))
            t2_ae  = l2.get("ae42_pseudo_pred", float("nan"))
            print(f"{bname:<12} {tr:>9.0f} {t1_if:>9.1f} {t2_if:>9.1f} "
                  f"{t2_if-t1_if:>+8.1f} {t1_ae:>9.1f} {t2_ae:>9.1f} {t2_ae-t1_ae:>+8.1f}")

    # ── Scenario classification ───────────────────────────────────────────────
    print("\nScenario Assessment (based on T2 CC-RMSE, pseudo model):")
    for _, row in comp_df[comp_df["eval_type"] == "CC"].iterrows():
        det = row["detector"]
        if det == "AE_mean":
            continue
        rmse = row["t2_true_rmse"]
        try:
            rmse_f = float(rmse)
        except (ValueError, TypeError):
            continue
        if rmse_f < 200:
            scenario = "A -- Major improvement. Paradigm is transferable with adaptation."
        elif rmse_f < 500:
            scenario = "B -- Moderate improvement. Limited transferability."
        elif rmse_f < 2000:
            scenario = "C -- Minor improvement. Structural limits of pipeline reached."
        else:
            scenario = "D -- No meaningful improvement. Pipeline is domain-specific."
        print(f"  {det}: CC-RMSE={rmse_f:.1f} -> Scenario {scenario}")
