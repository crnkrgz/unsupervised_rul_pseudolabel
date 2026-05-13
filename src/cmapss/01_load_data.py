"""
01_load_data.py
Loads NASA C-MAPSS train, test, and RUL files for all four subsets
(FD001, FD002, FD003, FD004) from data/raw/.
Column layout: unit_id, cycle, op_setting_1-3, sensor_1-21.
Outputs: dict of DataFrames keyed by subset name, each containing
         {'train': DataFrame, 'test': DataFrame, 'rul': Series}.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from cmapss.config import DATA_DIR, SUBSETS

COLUMNS = (
    ["unit_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

EXPECTED = {
    "FD001": dict(train_rows=20631, train_eng=100, test_rows=13096, test_eng=100, rul=100),
    "FD002": dict(train_rows=53759, train_eng=260, test_rows=33991, test_eng=259, rul=259),
    "FD003": dict(train_rows=24720, train_eng=100, test_rows=16596, test_eng=100, rul=100),
    "FD004": dict(train_rows=61249, train_eng=249, test_rows=41214, test_eng=248, rul=248),
}


def load_subset(subset_name: str) -> dict:
    name = subset_name.upper()
    train = pd.read_csv(
        DATA_DIR / f"train_{name}.txt",
        sep=r"\s+",
        header=None,
        names=COLUMNS,
        engine="python",
    )
    test = pd.read_csv(
        DATA_DIR / f"test_{name}.txt",
        sep=r"\s+",
        header=None,
        names=COLUMNS,
        engine="python",
    )
    rul = pd.read_csv(
        DATA_DIR / f"RUL_{name}.txt",
        sep=r"\s+",
        header=None,
        names=["RUL"],
        engine="python",
    )["RUL"]

    for df in (train, test):
        df["unit_id"] = df["unit_id"].astype(int)
        df["cycle"] = df["cycle"].astype(int)

    print(
        f"Loading {name}: train={len(train)} rows, "
        f"test={len(test)} rows, rul={len(rul)} engines"
    )

    exp = EXPECTED.get(name)
    if exp:
        checks = [
            (len(train),                       exp["train_rows"], "train rows"),
            (train["unit_id"].nunique(),       exp["train_eng"],  "train engines"),
            (len(test),                        exp["test_rows"],  "test rows"),
            (test["unit_id"].nunique(),        exp["test_eng"],   "test engines"),
            (len(rul),                         exp["rul"],        "RUL values"),
        ]
        for actual, expected, label in checks:
            if actual != expected:
                print(f"  WARNING {name} {label}: expected {expected}, got {actual}")

    return {"train": train, "test": test, "rul": rul}


def load_all_subsets() -> dict:
    return {s: load_subset(s) for s in SUBSETS}


if __name__ == "__main__":
    data = load_all_subsets()
    print()
    header = f"{'Subset':<8} {'n_train_rows':>13} {'n_train_engines':>16} {'n_test_rows':>12} {'n_test_engines':>15} {'n_rul':>6}"
    print(header)
    print("-" * len(header))
    for subset, d in data.items():
        train, test, rul = d["train"], d["test"], d["rul"]
        print(
            f"{subset:<8} {len(train):>13,} {train['unit_id'].nunique():>16,} "
            f"{len(test):>12,} {test['unit_id'].nunique():>15,} {len(rul):>6,}"
        )

