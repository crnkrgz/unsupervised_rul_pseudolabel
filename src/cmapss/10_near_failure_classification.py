"""
10_near_failure_classification.py
Near-failure binary classification accuracy for pseudo-label and true-label models.

Positive class: y_true_unclipped < threshold  (engine is near failure)
Thresholds tested: 30, 40, 50, 60 cycles

Comparison baselines:
  Chopra et al.  (CMAPSS, supervised classifier) : 97.3 % accuracy
  Juliet et al.  (CMAPSS, supervised classifier) : 92.86 % accuracy

Data source: results/predictions/cluster_optimal_{subset}_full.csv
             filtered to is_last_cycle == 1 (one row per test engine).

Generates:
  results/tables/table_near_failure_accuracy.csv
  results/tables/table_near_failure_by_subset.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

PRED_DIR   = ROOT / "results" / "predictions"
TAB_DIR    = ROOT / "results" / "tables"
SUBSETS    = ["FD001", "FD002", "FD003", "FD004"]
THRESHOLDS = [30, 40, 50, 60]

LITERATURE = {
    "Chopra et al. (supervised)": 97.3,
    "Juliet et al. (supervised)": 92.86,
}


def _metrics(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> dict:
    """Return accuracy, balanced_accuracy, precision, recall, f1 for a binary pair."""
    zero_div = 0
    return {
        "accuracy":          round(accuracy_score(y_true_bin, y_pred_bin) * 100, 2),
        "balanced_accuracy": round(balanced_accuracy_score(y_true_bin, y_pred_bin) * 100, 2),
        "precision":         round(precision_score(y_true_bin, y_pred_bin,
                                                    zero_division=zero_div) * 100, 2),
        "recall":            round(recall_score(y_true_bin, y_pred_bin,
                                                zero_division=zero_div) * 100, 2),
        "f1":                round(f1_score(y_true_bin, y_pred_bin,
                                            zero_division=zero_div) * 100, 2),
    }


def analyse_subset(subset: str) -> list:
    path = PRED_DIR / f"cluster_optimal_{subset}_full.csv"
    df   = pd.read_csv(path)
    last = df[df["is_last_cycle"] == 1].copy()

    y_true = last["y_true_unclipped"].values
    y_ps   = last["y_pred_pseudo"].values
    y_tr   = last["y_pred_true"].values

    rows = []
    for thr in THRESHOLDS:
        pos     = (y_true < thr)
        pred_ps = (y_ps   < thr)
        pred_tr = (y_tr   < thr)
        n_pos   = int(pos.sum())
        n_total = len(pos)

        m_ps = _metrics(pos, pred_ps)
        m_tr = _metrics(pos, pred_tr)

        rows.append({
            "subset":                   subset,
            "threshold":                thr,
            "n_engines":                n_total,
            "n_near_failure":           n_pos,
            "pct_near_failure":         round(n_pos / n_total * 100, 1),
            "pseudo_accuracy":          m_ps["accuracy"],
            "pseudo_balanced_accuracy": m_ps["balanced_accuracy"],
            "pseudo_precision":         m_ps["precision"],
            "pseudo_recall":            m_ps["recall"],
            "pseudo_f1":                m_ps["f1"],
            "true_accuracy":            m_tr["accuracy"],
            "true_balanced_accuracy":   m_tr["balanced_accuracy"],
            "true_precision":           m_tr["precision"],
            "true_recall":              m_tr["recall"],
            "true_f1":                  m_tr["f1"],
        })
    return rows


def _pooled_rows() -> list:
    """Aggregate all subsets pooled (all engines concatenated)."""
    # Pre-load last-cycle data for all subsets
    data = {}
    for subset in SUBSETS:
        path = PRED_DIR / f"cluster_optimal_{subset}_full.csv"
        df   = pd.read_csv(path)
        last = df[df["is_last_cycle"] == 1]
        data[subset] = {
            "y_true": last["y_true_unclipped"].values,
            "y_ps":   last["y_pred_pseudo"].values,
            "y_tr":   last["y_pred_true"].values,
        }

    out = []
    for thr in THRESHOLDS:
        y_true_all = np.concatenate([data[s]["y_true"] for s in SUBSETS])
        y_ps_all   = np.concatenate([data[s]["y_ps"]   for s in SUBSETS])
        y_tr_all   = np.concatenate([data[s]["y_tr"]   for s in SUBSETS])

        pos     = (y_true_all < thr)
        pred_ps = (y_ps_all   < thr)
        pred_tr = (y_tr_all   < thr)
        n_pos   = int(pos.sum())
        n_total = len(pos)

        m_ps = _metrics(pos, pred_ps)
        m_tr = _metrics(pos, pred_tr)

        out.append({
            "subset":                   "ALL",
            "threshold":                thr,
            "n_engines":                n_total,
            "n_near_failure":           n_pos,
            "pct_near_failure":         round(n_pos / n_total * 100, 1),
            "pseudo_accuracy":          m_ps["accuracy"],
            "pseudo_balanced_accuracy": m_ps["balanced_accuracy"],
            "pseudo_precision":         m_ps["precision"],
            "pseudo_recall":            m_ps["recall"],
            "pseudo_f1":                m_ps["f1"],
            "true_accuracy":            m_tr["accuracy"],
            "true_balanced_accuracy":   m_tr["balanced_accuracy"],
            "true_precision":           m_tr["precision"],
            "true_recall":              m_tr["recall"],
            "true_f1":                  m_tr["f1"],
        })
    return out


if __name__ == "__main__":
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    # -- Per-subset analysis --------------------------------------------------
    all_rows = []
    for subset in SUBSETS:
        rows = analyse_subset(subset)
        all_rows.extend(rows)
        print(f"\n{subset}:")
        for r in rows:
            print(f"  thr={r['threshold']:2d}  "
                  f"near-fail={r['n_near_failure']:3d}/{r['n_engines']} "
                  f"({r['pct_near_failure']:4.1f}%)  "
                  f"pseudo acc={r['pseudo_accuracy']:5.2f}%  "
                  f"true acc={r['true_accuracy']:5.2f}%")

    df_sub = pd.DataFrame(all_rows)

    # -- Pooled across all subsets --------------------------------------------
    print("\nPooled (all subsets):")
    pooled = _pooled_rows()
    for r in pooled:
        print(f"  thr={r['threshold']:2d}  "
              f"near-fail={r['n_near_failure']:3d}/{r['n_engines']} "
              f"({r['pct_near_failure']:4.1f}%)  "
              f"pseudo acc={r['pseudo_accuracy']:5.2f}%  "
              f"true acc={r['true_accuracy']:5.2f}%  "
              f"pseudo BA={r['pseudo_balanced_accuracy']:5.2f}%  "
              f"true BA={r['true_balanced_accuracy']:5.2f}%")

    df_all = pd.concat([df_sub, pd.DataFrame(pooled)], ignore_index=True)
    df_all.to_csv(TAB_DIR / "table_near_failure_by_subset.csv", index=False)
    print("\nSaved: table_near_failure_by_subset.csv")

    # -- Summary table (pooled rows only) -------------------------------------
    summary_cols = [
        "threshold", "n_engines", "n_near_failure", "pct_near_failure",
        "pseudo_accuracy", "pseudo_balanced_accuracy", "pseudo_precision",
        "pseudo_recall", "pseudo_f1",
        "true_accuracy",  "true_balanced_accuracy",  "true_precision",
        "true_recall",    "true_f1",
    ]
    df_summary = pd.DataFrame(pooled)[summary_cols]
    df_summary.to_csv(TAB_DIR / "table_near_failure_accuracy.csv", index=False)
    print("Saved: table_near_failure_accuracy.csv")

    # -- Literature comparison -------------------------------------------------
    print("\n-- Literature comparison (pooled, accuracy %) -----------------------")
    print(f"  {'Source':<35} {'thr=30':>7} {'thr=40':>7} {'thr=50':>7} {'thr=60':>7}")
    print("  " + "-" * 64)
    for name, val in LITERATURE.items():
        print(f"  {name:<35} {val:>7.2f} {val:>7.2f} {val:>7.2f} {val:>7.2f}")

    ps_accs = {r["threshold"]: r["pseudo_accuracy"] for r in pooled}
    tr_accs = {r["threshold"]: r["true_accuracy"]   for r in pooled}
    ps_bas  = {r["threshold"]: r["pseudo_balanced_accuracy"] for r in pooled}
    tr_bas  = {r["threshold"]: r["true_balanced_accuracy"]   for r in pooled}

    print(f"  {'Pseudo-label model (ours)':<35} "
          + "  ".join(f"{ps_accs[t]:>6.2f}%" for t in THRESHOLDS))
    print(f"  {'True-label model (ours)':<35} "
          + "  ".join(f"{tr_accs[t]:>6.2f}%" for t in THRESHOLDS))
    print()
    print(f"  {'Source':<35} {'thr=30':>7} {'thr=40':>7} {'thr=50':>7} {'thr=60':>7}")
    print(f"  {'(balanced accuracy)':<35}")
    print(f"  {'Pseudo-label model (ours)':<35} "
          + "  ".join(f"{ps_bas[t]:>6.2f}%" for t in THRESHOLDS))
    print(f"  {'True-label model (ours)':<35} "
          + "  ".join(f"{tr_bas[t]:>6.2f}%" for t in THRESHOLDS))

    # -- Sanity checks --------------------------------------------------------
    print("\n-- Sanity checks ----------------------------------------------------")
    ok = True
    for r in pooled:
        t = r["threshold"]
        if not (50 <= r["pseudo_accuracy"] <= 100):
            print(f"  WARN thr={t}: pseudo_accuracy={r['pseudo_accuracy']} out of range")
            ok = False
        if not (50 <= r["true_accuracy"] <= 100):
            print(f"  WARN thr={t}: true_accuracy={r['true_accuracy']} out of range")
            ok = False
    if ok:
        print("  All accuracy values in [50, 100] -- PASS")

    for r in pooled:
        if r["pct_near_failure"] < 10 or r["pct_near_failure"] > 90:
            print(f"  NOTE thr={r['threshold']}: class imbalance "
                  f"({r['pct_near_failure']:.1f}% positive) -- "
                  f"balanced_accuracy is more informative than accuracy")
