"""
06c_compare_all_tracks.py
Four-track comparison: T1 / T2 / T3 / T4.

  T1 (strict):   MinMaxScaler + alpha=1.0 + 18 feat
  T2 (adapted):  RobustScaler transductive + alpha=100 + 18 feat
  T3 (enhanced): MinMaxScaler + alpha=1.0 + 39 feat
  T4 (combined): RobustScaler transductive + alpha=100 + 39 feat

Oracle baseline note:
  T1 oracle (MinMaxScaler+alpha=1.0+18feat) = 2455.59 -- computed by 04c, not in any CSV.
  All other oracle values come from their respective eval CSVs.

Reads:
  results/femto/track1_strict_eval.csv
  results/femto/track2_adapted_eval.csv
  results/femto/track3_enhanced_eval.csv
  results/femto/track4_combined_eval.csv

Outputs:
  results/femto/all_tracks_comparison.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from femto.config import FEMTO_RESULTS, TEST_BEARINGS

# T1 oracle computed by 04c_pipeline_enhanced.py (not stored in CSV)
T1_IF_ORACLE_CC = 2455.59


def _get(df, detector, eval_type, col, default=None):
    if df is None:
        return default
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


def _fmt(v, decimals=2):
    if v is None:
        return "N/A"
    try:
        return f"{float(v):.{decimals}f}"
    except (ValueError, TypeError):
        return str(v)


if __name__ == "__main__":
    paths = {
        "T1": FEMTO_RESULTS / "track1_strict_eval.csv",
        "T2": FEMTO_RESULTS / "track2_adapted_eval.csv",
        "T3": FEMTO_RESULTS / "track3_enhanced_eval.csv",
        "T4": FEMTO_RESULTS / "track4_combined_eval.csv",
    }
    for label, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing {label}: {p}")

    # Load T1 (pseudo only: model, tv_rmse, n_tv, cc_rmse, n_cc)
    t1_raw = pd.read_csv(paths["T1"])
    t1_pseudo = {}
    for _, row in t1_raw.iterrows():
        det = row["model"]
        t1_pseudo[(det, "TV")] = row["tv_rmse"]
        t1_pseudo[(det, "CC")] = row["cc_rmse"]

    t2 = pd.read_csv(paths["T2"])
    t3 = pd.read_csv(paths["T3"])
    t4 = pd.read_csv(paths["T4"])

    W = 120
    print("=" * W)
    print("FEMTO All-Track Comparison")
    print("T1: MinMax+a=1+18f  |  T2: Robust+transduct+a=100+18f  |  "
          "T3: MinMax+a=1+39f  |  T4: Robust+transduct+a=100+39f")
    print("=" * W)

    rows = []
    for det, t2_det in [("IF", "IF"), ("AE", "AE_s42")]:
        for eval_type in ["TV", "CC"]:
            key = eval_type.lower()

            t1_p = t1_pseudo.get((det, eval_type))
            t2_p = _get(t2, t2_det, eval_type, "true_rmse")
            t3_p = _get(t3, det,    eval_type, "true_rmse")
            t4_p = _get(t4, t2_det, eval_type, "true_rmse")   # T4 uses AE_s42 like T2

            # Oracle values
            t1_o = T1_IF_ORACLE_CC if det == "IF" and eval_type == "CC" else None
            t2_o = _get(t2, t2_det, eval_type, "oracle_rmse")
            t3_o = _get(t3, det,    eval_type, "oracle_rmse")
            t4_o = _get(t4, t2_det, eval_type, "oracle_rmse")

            rows.append({
                "detector":  det,
                "eval_type": eval_type,
                "T1_pseudo": _fmt(t1_p),
                "T2_pseudo": _fmt(t2_p),
                "T3_pseudo": _fmt(t3_p),
                "T4_pseudo": _fmt(t4_p),
                "T4vsT2_%":  _pct(t4_p, t2_p),
                "T1_oracle": _fmt(t1_o),
                "T2_oracle": _fmt(t2_o),
                "T3_oracle": _fmt(t3_o),
                "T4_oracle": _fmt(t4_o),
                "T4vsT2_oracle_%": _pct(t4_o, t2_o),
            })

    comp_df = pd.DataFrame(rows)
    print(comp_df.to_string(index=False))

    out = FEMTO_RESULTS / "all_tracks_comparison.csv"
    comp_df.to_csv(out, index=False)
    print(f"\nSaved: {out}")

    # ── Oracle summary (key diagnostic) ──────────────────────────────────────
    print("\n" + "-" * W)
    print("Oracle Summary -- IF CC-RMSE across tracks")
    print("-" * W)
    oracles = [
        ("T1 (MinMax+a=1+18f)",          T1_IF_ORACLE_CC),
        ("T2 (Robust+transduct+a=100+18f)", _get(t2, "IF", "CC", "oracle_rmse")),
        ("T3 (MinMax+a=1+39f)",           _get(t3, "IF", "CC", "oracle_rmse")),
        ("T4 (Robust+transduct+a=100+39f)", _get(t4, "IF", "CC", "oracle_rmse")),
    ]
    best = min(v for _, v in oracles if v is not None)
    for label, val in oracles:
        marker = " <-- best" if val == best else ""
        print(f"  {label:<45} Oracle CC-RMSE = {_fmt(val)}{marker}")

    # ── AE seed stability (T4) ────────────────────────────────────────────────
    print("\n" + "-" * W)
    print("AE Seed Stability (CC-RMSE: pseudo / oracle)")
    print("-" * W)
    for track_label, df_eval in [("T2", t2), ("T4", t4)]:
        vals_p, vals_o = [], []
        for seed in [42, 43, 44]:
            det_name = f"AE_s{seed}"
            vp = _get(df_eval, det_name, "CC", "true_rmse")
            vo = _get(df_eval, det_name, "CC", "oracle_rmse")
            if vp is not None:
                vals_p.append(vp)
            if vo is not None:
                vals_o.append(vo)
        if vals_p:
            mp = np.mean(vals_p); sp = np.std(vals_p, ddof=1) if len(vals_p) > 1 else 0
            mo = np.mean(vals_o); so = np.std(vals_o, ddof=1) if len(vals_o) > 1 else 0
            print(f"  {track_label} AE CC pseudo: {mp:.1f} +/- {sp:.1f}  "
                  f"|  oracle: {mo:.1f} +/- {so:.1f}")
            for seed, vp, vo in zip([42, 43, 44], vals_p, vals_o):
                print(f"      seed={seed}  pseudo={vp:.1f}  oracle={vo:.1f}")

    # ── Per-bearing CC detail ─────────────────────────────────────────────────
    p1_path = FEMTO_RESULTS / "track1_strict_predictions.csv"
    p2_path = FEMTO_RESULTS / "track2_adapted_predictions.csv"
    p3_path = FEMTO_RESULTS / "track3_enhanced_predictions.csv"
    p4_path = FEMTO_RESULTS / "track4_combined_predictions.csv"

    if all(p.exists() for p in [p1_path, p2_path, p3_path, p4_path]):
        p1 = pd.read_csv(p1_path)
        p2 = pd.read_csv(p2_path)
        p3 = pd.read_csv(p3_path)
        p4 = pd.read_csv(p4_path)

        print("\n" + "-" * W)
        print("Per-Bearing CC: IF pseudo predictions, all tracks")
        print(f"{'bearing':<12} {'true':>6} {'T1':>8} {'T2':>8} {'T3':>8} {'T4':>8} "
              f"  {'best_abs_err':>12}")
        print("-" * 70)
        for bname in TEST_BEARINGS:
            def _last(df, col):
                sub = df[df["bearing_id"] == bname]
                if sub.empty:
                    return float("nan")
                return float(sub.loc[sub["cycle"].idxmax(), col])

            tr   = _last(p1, "true_rul")
            t1_v = _last(p1, "if_predicted_rul")
            t2_v = _last(p2, "if_pseudo_pred")
            t3_v = _last(p3, "if_pseudo_pred")
            t4_v = _last(p4, "if_pseudo_pred")

            errs = {k: abs(v - tr) for k, v in
                    [("T1", t1_v), ("T2", t2_v), ("T3", t3_v), ("T4", t4_v)]
                    if not np.isnan(v)}
            best_track = min(errs, key=errs.get) if errs else "?"
            best_err   = min(errs.values()) if errs else float("nan")

            print(f"{bname:<12} {tr:>6.0f} {t1_v:>8.1f} {t2_v:>8.1f} "
                  f"{t3_v:>8.1f} {t4_v:>8.1f}   {best_track}={best_err:>7.1f}")

    # ── Scenario classification (T4) ─────────────────────────────────────────
    print("\n" + "-" * W)
    print("Scenario Classification (T4 CC pseudo_rmse):")
    for det in ["IF", "AE"]:
        rmse = _get(t4, det, "CC", "true_rmse")
        if rmse is None:
            continue
        if rmse < 200:
            scenario = "P -- Oracle + pseudo both strong. Feature+scaling synergy confirmed."
        elif rmse < 500:
            scenario = "B -- Moderate improvement. Partial synergy."
        elif rmse < 2000:
            scenario = "C -- Minor improvement. Structural limits remain."
        else:
            scenario = "D -- No meaningful improvement. Pipeline domain-specific."
        print(f"  {det}: CC-RMSE={rmse:.1f} -> Scenario {scenario}")
