"""
02_preprocess.py
Preprocessing pipeline applied to raw C-MAPSS DataFrames:
  - RUL computation: max_cycle_per_unit - current_cycle, clipped at RUL_CEILING=125.
  - Zero-variance sensor filter: drop sensors with constant values across training set.
  - op_setting_3 removal for single-condition subsets (FD001, FD003).
  - Min-Max scaling fitted on training data only; same scaler applied to test data.
Outputs: cleaned and scaled train/test DataFrames, list of retained feature columns.
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from cmapss.config import RUL_CEILING, SINGLE_CONDITION, SUBSETS

ZERO_VAR_SENSORS = ["sensor_1", "sensor_5", "sensor_6", "sensor_10",
                    "sensor_16", "sensor_18", "sensor_19"]

EXPECTED_N_FEATURES = {"FD001": 16, "FD002": 17, "FD003": 16, "FD004": 17}


def _add_train_rul(train: pd.DataFrame) -> pd.DataFrame:
    train = train.copy()
    max_cycle = train.groupby("unit_id")["cycle"].transform("max")
    train["true_RUL"] = (max_cycle - train["cycle"]).clip(upper=RUL_CEILING)
    return train


def _add_test_rul(test: pd.DataFrame, rul_series: pd.Series) -> pd.DataFrame:
    test = test.copy()
    last_cycle = test.groupby("unit_id")["cycle"].transform("max")
    # rul_series is indexed 0..N-1; unit_ids run 1..N in the same order
    unit_ids = test["unit_id"].unique()
    unit_ids.sort()
    rul_map = dict(zip(unit_ids, rul_series.values))
    rul_at_last = test["unit_id"].map(rul_map)
    test["true_RUL"] = ((last_cycle - test["cycle"]) + rul_at_last).clip(upper=RUL_CEILING)
    return test


def _drop_zero_var(train: pd.DataFrame, test: pd.DataFrame):
    cols_to_drop = [c for c in ZERO_VAR_SENSORS if c in train.columns]
    return train.drop(columns=cols_to_drop), test.drop(columns=cols_to_drop)


def _drop_op3_if_single(
    train: pd.DataFrame, test: pd.DataFrame, subset_name: str
):
    if subset_name in SINGLE_CONDITION and "op_setting_3" in train.columns:
        return train.drop(columns=["op_setting_3"]), test.drop(columns=["op_setting_3"])
    return train, test


def preprocess_subset(subset_data: dict, subset_name: str, scale: bool = True) -> dict:
    name = subset_name.upper()
    train = _add_train_rul(subset_data["train"])
    test  = _add_test_rul(subset_data["test"], subset_data["rul"])

    train, test = _drop_zero_var(train, test)
    train, test = _drop_op3_if_single(train, test, name)

    non_feature = {"unit_id", "cycle", "true_RUL"}
    feature_cols = [c for c in train.columns if c not in non_feature]

    if scale:
        scaler = MinMaxScaler()
        train[feature_cols] = scaler.fit_transform(train[feature_cols])
        test[feature_cols]  = scaler.transform(test[feature_cols])
    else:
        scaler = None

    col_order = ["unit_id", "cycle"] + feature_cols + ["true_RUL"]
    train = train[col_order]
    test  = test[col_order]

    return {
        "train": train,
        "test": test,
        "feature_cols": feature_cols,
        "scaler": scaler,
        "n_features": len(feature_cols),
    }


def preprocess_all() -> dict:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "load_data",
        Path(__file__).parent / "01_load_data.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    raw = mod.load_all_subsets()
    return {s: preprocess_subset(raw[s], s) for s in SUBSETS}


if __name__ == "__main__":
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "load_data", Path(__file__).parent / "01_load_data.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    raw = mod.load_all_subsets()

    print()
    for subset in SUBSETS:
        result = preprocess_subset(raw[subset], subset)
        train, test = result["train"], result["test"]
        n_feat = result["n_features"]
        feat_cols = result["feature_cols"]
        max_rul_train = train["true_RUL"].max()
        max_rul_test  = test["true_RUL"].max()
        exp = EXPECTED_N_FEATURES[subset]
        feat_ok = "OK" if n_feat == exp else f"MISMATCH (expected {exp})"
        rul_ok  = "OK" if max_rul_train <= RUL_CEILING and max_rul_test <= RUL_CEILING else "FAIL"
        print(f"{subset}:")
        print(f"  n_features : {n_feat} [{feat_ok}]")
        print(f"  features   : {feat_cols}")
        print(f"  train rows : {len(train):,}   test rows: {len(test):,}")
        print(f"  max true_RUL (train/test): {max_rul_train} / {max_rul_test}  [{rul_ok}]")
        print()

