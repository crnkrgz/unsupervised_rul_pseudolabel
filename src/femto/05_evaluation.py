"""
05_evaluation.py
Compute time-varying (TV) and challenge-convention (CC) RMSE
for both IF and AE strict-transfer predictions.

TV-RMSE: all (bearing, eval_cycle) pairs where cycle in {50, 100, ..., truncated_at}
CC-RMSE: one prediction per bearing (at truncated_at)

Inputs:
  results/femto/track1_strict_predictions.csv
  data/processed/femto/bearing_metadata.csv

Output: results/femto/track1_strict_eval.csv (summary table)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from femto.config import FEMTO_PROCESSED, FEMTO_RESULTS, TEST_BEARINGS, TV_STEP


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def compute_tv_rmse(preds: pd.DataFrame, meta: dict, model_col: str) -> tuple[float, int]:
    """
    Evaluate at every TV_STEP cycles from TV_STEP up to truncated_at.
    Returns (rmse, n_eval_points).
    """
    records = []
    for bname in TEST_BEARINGS:
        trunc      = int(meta[bname]["truncated_at"])
        total_life = int(meta[bname]["total_life"])
        sub        = preds[preds["bearing_id"] == bname].set_index("cycle")

        for cyc in range(TV_STEP, trunc + 1, TV_STEP):
            if cyc not in sub.index:
                # find nearest available cycle
                available = sub.index[sub.index <= cyc]
                if len(available) == 0:
                    continue
                cyc = available[-1]
            true_rul = total_life - cyc
            pred_rul = float(sub.loc[cyc, model_col])
            records.append((true_rul, pred_rul))

    y_true = np.array([r[0] for r in records])
    y_pred = np.array([r[1] for r in records])
    return _rmse(y_true, y_pred), len(records)


def compute_cc_rmse(preds: pd.DataFrame, meta: dict, model_col: str) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Evaluate at the truncated_at cycle only (challenge convention).
    Returns (rmse, y_true_array, y_pred_array).
    """
    y_true, y_pred = [], []
    for bname in TEST_BEARINGS:
        trunc = int(meta[bname]["truncated_at"])
        sub   = preds[(preds["bearing_id"] == bname) & (preds["cycle"] == trunc)]
        if sub.empty:
            # fallback: last available cycle
            sub = preds[preds["bearing_id"] == bname]
            sub = sub.loc[[sub["cycle"].idxmax()]]
        y_true.append(float(sub["true_rul"].values[0]))
        y_pred.append(float(sub[model_col].values[0]))

    yt = np.array(y_true)
    yp = np.array(y_pred)
    return _rmse(yt, yp), yt, yp


if __name__ == "__main__":
    preds_path = FEMTO_RESULTS / "track1_strict_predictions.csv"
    meta_path  = FEMTO_PROCESSED / "bearing_metadata.csv"

    if not preds_path.exists():
        raise FileNotFoundError(
            f"{preds_path} not found. Run 04_pipeline_strict.py first."
        )

    preds = pd.read_csv(preds_path)
    meta  = pd.read_csv(meta_path).set_index("bearing_id").to_dict("index")

    models = {
        "IF": ("if_predicted_rul",  "if_pseudo_rul"),
        "AE": ("ae_predicted_rul",  "ae_pseudo_rul"),
    }

    W = 75
    print("=" * W)
    print("FEMTO Strict Transfer — Evaluation Results")
    print("=" * W)

    summary_rows = []
    for model_name, (pred_col, pseudo_col) in models.items():
        tv_rmse, n_tv  = compute_tv_rmse(preds, meta, pred_col)
        cc_rmse, yt, yp = compute_cc_rmse(preds, meta, pred_col)
        summary_rows.append({
            "model":    model_name,
            "tv_rmse":  round(tv_rmse, 2),
            "n_tv":     n_tv,
            "cc_rmse":  round(cc_rmse, 2),
            "n_cc":     len(yt),
        })
        print(f"\n{model_name} ({pred_col})")
        print(f"  TV-RMSE : {tv_rmse:.2f} cycles  ({n_tv} eval points)")
        print(f"  CC-RMSE : {cc_rmse:.2f} cycles  ({len(yt)} bearings)")

        # Per-bearing CC detail
        print(f"\n  {'bearing':<12} {'true_rul':>9} {'predicted':>10} {'error':>8}")
        print(f"  {'-'*42}")
        for bname, tr, pr in zip(TEST_BEARINGS, yt, yp):
            print(f"  {bname:<12} {tr:>9.0f} {pr:>10.1f} {pr-tr:>+8.1f}")

    print()

    df_summary = pd.DataFrame(summary_rows)
    out = FEMTO_RESULTS / "track1_strict_eval.csv"
    df_summary.to_csv(out, index=False)

    print("=" * W)
    print("Summary")
    print("=" * W)
    print(df_summary.to_string(index=False))
    print(f"\nSaved: {out}")

    # Pseudo-RUL vs predicted RMSE comparison (shows Ridge improvement)
    print("\nPseudo-label quality (before Ridge):")
    for model_name, (pred_col, pseudo_col) in models.items():
        _, yt, _ = compute_cc_rmse(preds, meta, pred_col)
        pseudo_cc_rmse, _, yp_pseudo = compute_cc_rmse(preds, meta, pseudo_col)
        print(f"  {model_name} pseudo CC-RMSE: {pseudo_cc_rmse:.2f}  "
              f"(predicted: {[r for r in summary_rows if r['model']==model_name][0]['cc_rmse']:.2f})")
