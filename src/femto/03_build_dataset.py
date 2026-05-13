"""
03_build_dataset.py
Assemble pipeline-ready CSV files from per-bearing feature CSVs.

Outputs:
  data/processed/femto/femto_train.csv           (6 training bearings, full life)
  data/processed/femto/femto_test_input.csv      (11 test bearings, truncated)
  data/processed/femto/femto_test_groundtruth_cc.csv  (11 rows, CC evaluation)
  data/processed/femto/femto_test_full.csv       (11 validation bearings, full)
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from femto.config import (
    CONDITION_COLS, FEATURE_COLS,
    FEMTO_PROCESSED, TEST_BEARINGS, TRAIN_BEARINGS, bearing_condition,
)


def _load_features(prefix: str, bname: str) -> pd.DataFrame:
    path = FEMTO_PROCESSED / f"{prefix}_{bname}_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}. Run 02_feature_extraction.py first.")
    return pd.read_csv(path)


def _add_condition_and_dummies(df: pd.DataFrame, bname: str) -> pd.DataFrame:
    cond = bearing_condition(bname)
    df["bearing_id"] = bname
    df["condition"]  = cond
    for c in [1, 2, 3]:
        df[f"cond_{c}"] = int(cond == c)
    return df


def build_all() -> None:
    meta = pd.read_csv(FEMTO_PROCESSED / "bearing_metadata.csv").set_index("bearing_id")

    BASE_COLS  = ["bearing_id", "condition", "cycle"]
    FEAT_BLOCK = FEATURE_COLS + CONDITION_COLS

    # ── Training set ──────────────────────────────────────────────────────────
    train_dfs = []
    for bname in TRAIN_BEARINGS:
        df = _load_features("train", bname)
        df = _add_condition_and_dummies(df, bname)
        df["rul"] = df["cycle"].max() - df["cycle"]
        train_dfs.append(df[BASE_COLS + FEAT_BLOCK + ["rul"]])

    train_df = pd.concat(train_dfs, ignore_index=True)
    out = FEMTO_PROCESSED / "femto_train.csv"
    train_df.to_csv(out, index=False)
    print(f"femto_train.csv          : {len(train_df):6d} rows  "
          f"({len(TRAIN_BEARINGS)} bearings)")

    # ── Test input (truncated trajectories) ───────────────────────────────────
    test_dfs = []
    for bname in TEST_BEARINGS:
        df = _load_features("test", bname)
        df = _add_condition_and_dummies(df, bname)
        test_dfs.append(df[BASE_COLS + FEAT_BLOCK])

    test_df = pd.concat(test_dfs, ignore_index=True)
    out = FEMTO_PROCESSED / "femto_test_input.csv"
    test_df.to_csv(out, index=False)
    print(f"femto_test_input.csv     : {len(test_df):6d} rows  "
          f"({len(TEST_BEARINGS)} bearings)")

    # ── Challenge-convention ground truth (one row per test bearing) ──────────
    cc_rows = []
    for bname in TEST_BEARINGS:
        m = meta.loc[bname]
        cc_rows.append({
            "bearing_id":        bname,
            "condition":         bearing_condition(bname),
            "truncated_at":      int(m["truncated_at"]),
            "total_life":        int(m["total_life"]),
            "rul_at_truncation": int(m["rul_at_truncation"]),
        })
    cc_df = pd.DataFrame(cc_rows)
    out = FEMTO_PROCESSED / "femto_test_groundtruth_cc.csv"
    cc_df.to_csv(out, index=False)
    print(f"femto_test_groundtruth_cc: {len(cc_df):6d} rows  (challenge convention)")

    # ── Full validation trajectories (time-varying evaluation) ────────────────
    val_dfs = []
    for bname in TEST_BEARINGS:
        df         = _load_features("val", bname)
        df         = _add_condition_and_dummies(df, bname)
        total_life = int(meta.loc[bname, "total_life"])
        df["rul"]  = total_life - df["cycle"]
        val_dfs.append(df[BASE_COLS + FEAT_BLOCK + ["rul"]])

    val_df = pd.concat(val_dfs, ignore_index=True)
    out = FEMTO_PROCESSED / "femto_test_full.csv"
    val_df.to_csv(out, index=False)
    print(f"femto_test_full.csv      : {len(val_df):6d} rows  "
          f"(Validation_Set, time-varying eval)")


if __name__ == "__main__":
    build_all()
