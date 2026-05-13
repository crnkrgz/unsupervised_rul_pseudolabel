"""
01_load_data.py
Scan FEMTO bearing folders and build bearing_metadata.csv.

Outputs: data/processed/femto/bearing_metadata.csv
Columns: bearing_id, split, condition, n_cycles, total_life,
         truncated_at, rul_at_truncation
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from femto.config import (
    FEMTO_PROCESSED, FEMTO_RAW_ROOT,
    TEST_BEARINGS, TRAIN_BEARINGS, bearing_condition,
)


def load_bearing_metadata() -> pd.DataFrame:
    rows = []

    train_path = FEMTO_RAW_ROOT / "Training_set"
    for bname in TRAIN_BEARINGS:
        n = len(list((train_path / bname).glob("acc_*.csv")))
        rows.append({
            "bearing_id":        bname,
            "split":             "train",
            "condition":         bearing_condition(bname),
            "n_cycles":          n,
            "total_life":        n,
            "truncated_at":      None,
            "rul_at_truncation": None,
        })

    test_path = FEMTO_RAW_ROOT / "Test_set"
    val_path  = FEMTO_RAW_ROOT / "Validation_Set"
    for bname in TEST_BEARINGS:
        trunc  = len(list((test_path / bname).glob("acc_*.csv")))
        val_n  = len(list((val_path  / bname).glob("acc_*.csv")))
        rows.append({
            "bearing_id":        bname,
            "split":             "test",
            "condition":         bearing_condition(bname),
            "n_cycles":          trunc,
            "total_life":        val_n,
            "truncated_at":      trunc,
            "rul_at_truncation": val_n - trunc,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    FEMTO_PROCESSED.mkdir(parents=True, exist_ok=True)
    df = load_bearing_metadata()
    out = FEMTO_PROCESSED / "bearing_metadata.csv"
    df.to_csv(out, index=False)

    print(f"Saved: {out}")
    print()

    W = 95
    print("=" * W)
    print("FEMTO Bearing Metadata")
    print("=" * W)
    print(df.to_string(index=False))
    print()
    print(f"Train bearings: {len(df[df['split']=='train'])}")
    print(f"Test bearings:  {len(df[df['split']=='test'])}")
    print()
    test_df = df[df["split"] == "test"].copy()
    print(f"RUL range at truncation: [{test_df['rul_at_truncation'].min()}, "
          f"{test_df['rul_at_truncation'].max()}]")
    print(f"Total life range (test):  [{test_df['total_life'].min()}, "
          f"{test_df['total_life'].max()}]")
