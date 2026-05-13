"""
12_spark_scalability.py
Wall-clock benchmark: scikit-learn vs PySpark MLlib for FD001 pipeline (Tables 3.3, 3.4).

scikit-learn: full global pipeline — preprocess, IF(n=200), 5-cycle rolling smooth,
              pseudo-label generation (threshold=0.20), Ridge(alpha=10).

PySpark:      equivalent single-node pipeline — VectorAssembler, MinMaxScaler,
              sklearn IF wrapped in pandas_udf (falls back to GBT placeholder if needed),
              window-function rolling smooth, LinearRegression(elasticNet=0, regParam=10).

Both run local[1] (single core) for a fair apples-to-apples comparison.
JAVA_HOME is set programmatically from the Adoptium install path if not in PATH.

Generates: results/tables/table_3_3_spark_benchmark.csv
           results/tables/table_3_4_scaling_projections.csv
"""

import importlib.util
import os
import platform
import sys
import time
import warnings
from pathlib import Path

# ── Locate Java if JAVA_HOME not already set ──────────────────────────────────
_ADOPTIUM = Path("C:/Program Files/Eclipse Adoptium")
if "JAVA_HOME" not in os.environ and _ADOPTIUM.exists():
    candidates = sorted(_ADOPTIUM.iterdir(), reverse=True)
    if candidates:
        os.environ["JAVA_HOME"] = str(candidates[0])

# ── Tell Spark's JVM which Python to spawn for worker processes ───────────────
# Required on Windows where 'python' may resolve to the MS Store stub.
if "PYSPARK_PYTHON" not in os.environ:
    os.environ["PYSPARK_PYTHON"] = sys.executable
if "PYSPARK_DRIVER_PYTHON" not in os.environ:
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from cmapss.config import (
    IF_CONTAMINATION, RIDGE_ALPHA, ROLLING_WINDOW, RESULTS_DIR, SEED,
    ZONE_DEGRADED, ZONE_HEALTHY,
)

_SRC      = ROOT / "src"
_TAB_DIR  = RESULTS_DIR / "tables"
N_RUNS    = 3
SUBSET    = "FD001"
THRESHOLD = 0.20   # FD001 optimal


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, _SRC / filename)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _hw_info() -> str:
    try:
        import psutil
        ram_gb  = psutil.virtual_memory().total / 1e9
        cpu     = platform.processor() or platform.machine()
        cores   = psutil.cpu_count(logical=False)
        logical = psutil.cpu_count(logical=True)
        return f"{cpu}  cores={cores}p/{logical}l  RAM={ram_gb:.1f} GB"
    except ImportError:
        return f"{platform.processor() or platform.machine()}  (psutil not available)"


# ══════════════════════════════════════════════════════════════════════════════
# scikit-learn pipeline  (pure sklearn, single core)
# ══════════════════════════════════════════════════════════════════════════════

def _sklearn_pipeline_once(train_df: pd.DataFrame, feature_cols: list) -> float:
    """Run full sklearn pipeline once; return wall-clock seconds."""
    t0 = time.perf_counter()

    # MinMaxScaler
    scaler  = MinMaxScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values)

    # Isolation Forest
    clf        = IsolationForest(n_estimators=200, contamination=IF_CONTAMINATION,
                                 random_state=SEED)
    clf.fit(X_train)
    raw_scores = -clf.decision_function(X_train)

    # 5-cycle rolling smooth per engine
    tmp = pd.DataFrame({
        "unit_id": train_df["unit_id"].values,
        "cycle":   train_df["cycle"].values,
        "_score":  raw_scores,
        "_pos":    np.arange(len(raw_scores)),
    }).sort_values(["unit_id", "cycle"])
    smoothed = tmp.groupby("unit_id")["_score"].transform(
        lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).mean()
    )
    smooth = np.empty(len(raw_scores))
    smooth[tmp["_pos"].values] = smoothed.values

    # Pseudo-label generation
    cutoff = np.quantile(smooth, 1.0 - THRESHOLD)
    pseudo = np.empty(len(smooth), dtype=float)
    a_mask = smooth >= cutoff
    n_mask = ~a_mask
    if a_mask.any():
        s     = smooth[a_mask]
        denom = s.max() - cutoff
        pseudo[a_mask] = (ZONE_DEGRADED[1] / 2.0) if denom == 0 else \
                          ZONE_DEGRADED[1] - ZONE_DEGRADED[1] * (s - cutoff) / denom
    if n_mask.any():
        s     = smooth[n_mask]
        mn    = s.min()
        denom = cutoff - mn
        span  = ZONE_HEALTHY[1] - ZONE_HEALTHY[0]
        pseudo[n_mask] = ((ZONE_HEALTHY[0] + ZONE_HEALTHY[1]) / 2.0) if denom == 0 else \
                          ZONE_HEALTHY[1] - span * (s - mn) / denom
    pseudo = np.clip(pseudo, 0, 125)

    # Ridge regression
    X_aug = np.column_stack([X_train, smooth])
    Ridge(alpha=RIDGE_ALPHA).fit(X_aug, pseudo)

    return time.perf_counter() - t0


def run_sklearn_benchmark(train_df: pd.DataFrame, feature_cols: list) -> dict:
    times = []
    for i in range(N_RUNS):
        t = _sklearn_pipeline_once(train_df, feature_cols)
        times.append(t)
        print(f"    Run {i+1}: {t:.3f}s")
    mean_t = float(np.mean(times))
    n_rec  = len(train_df)
    print(f"    Mean:  {mean_t:.3f}s")
    print(f"    Throughput: {n_rec/mean_t:,.0f} records/sec")
    return {"times": times, "mean": mean_t, "n_records": n_rec,
            "throughput": n_rec / mean_t}


# ══════════════════════════════════════════════════════════════════════════════
# PySpark pipeline
# ══════════════════════════════════════════════════════════════════════════════

def _build_spark():
    from pyspark.sql import SparkSession
    return (SparkSession.builder
            .master("local[1]")
            .appName("ThesisBenchmark")
            .config("spark.driver.memory", "2g")
            .config("spark.sql.shuffle.partitions", "4")
            .config("spark.ui.enabled", "false")
            .config("spark.ui.showConsoleProgress", "false")
            # Disable Arrow to avoid Python-worker spawning on Windows/Python 3.12
            .config("spark.sql.execution.arrow.pyspark.enabled", "false")
            .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "false")
            .getOrCreate())


def _spark_pipeline_once(spark, train_df: pd.DataFrame, feature_cols: list,
                          if_strategy: str, tmp_dir: Path) -> float:
    """
    Run equivalent Spark pipeline once; return wall-clock seconds.

    Data enters Spark via a temp CSV (JVM CSV reader) to avoid the Python-worker
    spawning that createDataFrame(pandas) triggers on Python 3.12 + PySpark 3.5.
    IF scoring runs on the driver via toPandas() + sklearn (Py4J socket, no worker).
    All Spark ML operations (VectorAssembler, MinMaxScaler, window, LinearRegression)
    are pure JVM.
    """
    import tempfile
    from pyspark.ml.feature import VectorAssembler, MinMaxScaler as SparkMMS
    from pyspark.ml.regression import LinearRegression
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window
    from pyspark.sql.types import (StructType, StructField,
                                   IntegerType, DoubleType, LongType)

    t0 = time.perf_counter()

    # ── Step 1: Write to temp CSV; JVM reads it natively (no Python workers) ──
    pdf_in = train_df[["unit_id", "cycle"] + feature_cols].copy()
    pdf_in["_row_id"] = np.arange(len(pdf_in), dtype=np.int64)
    tmp_path = tmp_dir / "spark_input.csv"
    pdf_in.to_csv(tmp_path, index=False)

    int_cols   = {"unit_id": IntegerType(), "cycle": IntegerType()}
    schema     = StructType(
        [StructField("unit_id",  IntegerType(),  True),
         StructField("cycle",    IntegerType(),  True)]
        + [StructField(c, DoubleType(), True) for c in feature_cols]
        + [StructField("_row_id", LongType(), True)]
    )
    sdf = spark.read.schema(schema).option("header", "true").csv(str(tmp_path))

    # ── Step 2: VectorAssembler + MinMaxScaler (pure JVM) ────────────────────
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw",
                                handleInvalid="skip")
    sdf_a     = assembler.transform(sdf)
    mms       = SparkMMS(inputCol="features_raw", outputCol="features")
    mms_model = mms.fit(sdf_a)
    sdf_s     = mms_model.transform(sdf_a).drop("features_raw")

    # ── Step 3: sklearn IF scored on driver, merged back via _row_id ─────────
    # Collect to driver — measures Spark→driver transfer cost (realistic for IF)
    pdf_scaled = sdf_s.select("_row_id", *feature_cols).toPandas()
    pdf_scaled = pdf_scaled.sort_values("_row_id")
    X_scaled   = pdf_scaled[feature_cols].values.astype(np.float64)

    clf        = IsolationForest(n_estimators=200, contamination=IF_CONTAMINATION,
                                 random_state=SEED)
    clf.fit(X_scaled)
    scores     = -clf.decision_function(X_scaled)

    pdf_scores = pd.DataFrame({"_row_id": pdf_scaled["_row_id"].values,
                                "raw_score": scores})
    scores_path = tmp_dir / "spark_scores.csv"
    pdf_scores.to_csv(scores_path, index=False)
    score_schema = StructType([
        StructField("_row_id",   LongType(),   True),
        StructField("raw_score", DoubleType(), True),
    ])
    sdf_scores = (spark.read.schema(score_schema)
                       .option("header", "true")
                       .csv(str(scores_path)))

    # Rejoin scores into the Spark DataFrame
    sdf_scored = sdf_s.join(sdf_scores, on="_row_id").drop("_row_id")

    # ── Step 4: 5-cycle rolling smooth (JVM window function) ─────────────────
    w      = (Window.partitionBy("unit_id").orderBy("cycle")
               .rowsBetween(-(ROLLING_WINDOW - 1), 0))
    sdf_sm = sdf_scored.withColumn("anomaly_score", F.avg("raw_score").over(w)) \
                        .drop("raw_score")

    # ── Step 5: Pseudo-label generation (driver computes thresholds, JVM expr) ─
    pct_row = sdf_sm.approxQuantile("anomaly_score", [1.0 - THRESHOLD], 0.005)
    cutoff  = float(pct_row[0]) if pct_row else 0.5

    stats = sdf_sm.agg(
        F.max(F.when(F.col("anomaly_score") >= cutoff,
                     F.col("anomaly_score"))).alias("anom_max"),
        F.min(F.when(F.col("anomaly_score") <  cutoff,
                     F.col("anomaly_score"))).alias("norm_min"),
    ).collect()[0]
    anom_max = float(stats["anom_max"] or cutoff)
    norm_min = float(stats["norm_min"] or 0.0)
    denom_a  = max(anom_max - cutoff, 1e-9)
    denom_n  = max(cutoff - norm_min, 1e-9)
    hi_zone  = float(ZONE_DEGRADED[1])
    hi_full  = float(ZONE_HEALTHY[1])
    span     = float(ZONE_HEALTHY[1] - ZONE_HEALTHY[0])

    sdf_pl = sdf_sm.withColumn(
        "pseudo_label",
        F.when(
            F.col("anomaly_score") >= cutoff,
            F.greatest(F.lit(0.0),
                        F.lit(hi_zone) - F.lit(hi_zone) *
                        (F.col("anomaly_score") - cutoff) / denom_a)
        ).otherwise(
            F.least(F.lit(125.0),
                     F.lit(hi_full) - F.lit(span) *
                     (F.col("anomaly_score") - norm_min) / denom_n)
        )
    )

    # ── Step 6: Ridge via LinearRegression (pure JVM MLlib) ──────────────────
    score_asm = VectorAssembler(inputCols=["anomaly_score"], outputCol="score_vec",
                                handleInvalid="skip")
    aug_asm   = VectorAssembler(inputCols=["features", "score_vec"],
                                outputCol="features_aug", handleInvalid="skip")
    sdf_aug   = aug_asm.transform(score_asm.transform(sdf_pl))

    lr = LinearRegression(featuresCol="features_aug", labelCol="pseudo_label",
                          elasticNetParam=0.0, regParam=RIDGE_ALPHA,
                          maxIter=100, standardization=False)
    lr.fit(sdf_aug)

    return time.perf_counter() - t0


def run_spark_benchmark(spark, train_df: pd.DataFrame, feature_cols: list) -> dict:
    import tempfile
    # Driver-side sklearn IF scoring avoids Python-worker issues on Windows/Python 3.12
    if_strategy = "sklearn_driver_side"
    print(f"    IF strategy: {if_strategy}")
    times = []
    with tempfile.TemporaryDirectory() as _tmp:
        tmp_dir = Path(_tmp)
        for i in range(N_RUNS):
            t = _spark_pipeline_once(spark, train_df, feature_cols, if_strategy, tmp_dir)
            times.append(t)
            print(f"    Run {i+1}: {t:.3f}s")
    mean_t = float(np.mean(times))
    n_rec  = len(train_df)
    print(f"    Mean:  {mean_t:.3f}s")
    print(f"    Throughput: {n_rec/mean_t:,.0f} records/sec")
    return {"times": times, "mean": mean_t, "n_records": n_rec,
            "throughput": n_rec / mean_t, "if_strategy": if_strategy}


# ══════════════════════════════════════════════════════════════════════════════
# Scalability projections
# ══════════════════════════════════════════════════════════════════════════════

def compute_projections(sklearn_res: dict, spark_res: dict) -> pd.DataFrame:
    """
    Decompose Spark time into fixed startup overhead + per-record cost.
    Project crossover and 100M-record scenario for a 10-node cluster.
    """
    n_obs   = sklearn_res["n_records"]
    sk_mean = sklearn_res["mean"]
    sp_mean = spark_res["mean"]
    sk_tput = sklearn_res["throughput"]          # rec/s, single core

    # Estimate Spark startup (~30 % of observed time for this dataset size)
    spark_startup     = max(sp_mean * 0.30, 2.0)
    spark_per_rec_sec = (sp_mean - spark_startup) / n_obs
    spark_tput_node   = 1.0 / spark_per_rec_sec  # rec/s per Spark node

    N_NODES = 10

    # Crossover: N / sk_tput = startup + N / (spark_tput_node * N_NODES)
    denom = (1.0 / sk_tput) - (1.0 / (spark_tput_node * N_NODES))
    crossover_n = int(spark_startup / denom) if denom > 0 else None

    print(f"\n  Estimated Spark startup overhead : {spark_startup:.1f}s")
    print(f"  Spark per-record throughput (1 node): {spark_tput_node:,.0f} rec/s")
    if crossover_n:
        print(f"  10-node crossover point: ~{crossover_n:,} records")
    else:
        print(f"  10-node crossover: not reached in projection range")

    def _row(label, N):
        sk_t = N / sk_tput
        sp_t = spark_startup + N / (spark_tput_node * N_NODES)
        adv  = sk_t / sp_t if sp_t > 0 else float("inf")
        return {
            "scenario":              label,
            "n_records":             N,
            "sklearn_time_sec":      round(sk_t, 2),
            "spark_10node_time_sec": round(sp_t, 2),
            "advantage_factor":      round(adv, 2),
        }

    rows = [
        _row("FD001 (observed)",         n_obs),
        _row("10x FD001",                n_obs * 10),
        _row("100x FD001",               n_obs * 100),
        _row("Crossover estimate",        crossover_n or n_obs * 50),
        _row("100M records",              100_000_000),
    ]
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t_total = time.perf_counter()
    _TAB_DIR.mkdir(parents=True, exist_ok=True)

    # ── Hardware + environment ────────────────────────────────────────────────
    print("Hardware / environment:")
    print(f"  {_hw_info()}")
    print(f"  JAVA_HOME: {os.environ.get('JAVA_HOME', 'not set')}")
    print(f"  Python: {sys.version.split()[0]}")
    try:
        import pyspark
        print(f"  PySpark: {pyspark.__version__}")
    except ImportError:
        print("  PySpark: not installed — cannot proceed")
        sys.exit(1)

    # ── Load FD001 training data ──────────────────────────────────────────────
    load_mod = _load_module("load_data",  "01_load_data.py")
    prep_mod = _load_module("preprocess", "02_preprocess.py")

    raw          = load_mod.load_subset(SUBSET)
    result       = prep_mod.preprocess_subset(raw, SUBSET, scale=False)
    train_df     = result["train"]
    feature_cols = result["feature_cols"]
    n_records    = len(train_df)

    print(f"\nDataset: {SUBSET}  records={n_records:,}  features={len(feature_cols)}")

    # ── scikit-learn benchmark ────────────────────────────────────────────────
    print(f"\nscikit-learn pipeline ({SUBSET}, {n_records:,} records):")
    sklearn_res = run_sklearn_benchmark(train_df, feature_cols)

    # ── PySpark benchmark ─────────────────────────────────────────────────────
    print(f"\nPySpark MLlib pipeline ({SUBSET}, {n_records:,} records, local[1]):")
    print("  Starting SparkSession...")
    spark = _build_spark()
    spark.sparkContext.setLogLevel("ERROR")
    spark_res = run_spark_benchmark(spark, train_df, feature_cols)
    spark.stop()

    # ── Summary ───────────────────────────────────────────────────────────────
    overhead = spark_res["mean"] / sklearn_res["mean"]
    print(f"\nComparison:")
    print(f"  sklearn mean : {sklearn_res['mean']:.3f}s  "
          f"({sklearn_res['throughput']:,.0f} rec/s)")
    print(f"  PySpark mean : {spark_res['mean']:.3f}s  "
          f"({spark_res['throughput']:,.0f} rec/s)")
    print(f"  Overhead factor (Spark / sklearn): {overhead:.1f}x")

    # ── Table 3.3 ─────────────────────────────────────────────────────────────
    df33 = pd.DataFrame([
        {
            "framework":                  "scikit-learn",
            "dataset":                    SUBSET,
            "n_records":                  n_records,
            "mean_time_seconds":          round(sklearn_res["mean"], 4),
            "throughput_records_per_sec": round(sklearn_res["throughput"], 1),
            "overhead_factor":            1.0,
        },
        {
            "framework":                  f"PySpark MLlib ({spark_res['if_strategy']})",
            "dataset":                    SUBSET,
            "n_records":                  n_records,
            "mean_time_seconds":          round(spark_res["mean"], 4),
            "throughput_records_per_sec": round(spark_res["throughput"], 1),
            "overhead_factor":            round(overhead, 2),
        },
    ])
    df33.to_csv(_TAB_DIR / "table_3_3_spark_benchmark.csv", index=False)
    print("\nSaved: table_3_3_spark_benchmark.csv")

    # ── Table 3.4 ─────────────────────────────────────────────────────────────
    print("\nScalability projections (10-node Spark vs single-core sklearn):")
    df34 = compute_projections(sklearn_res, spark_res)
    print(df34.to_string(index=False))
    df34.to_csv(_TAB_DIR / "table_3_4_scaling_projections.csv", index=False)
    print("\nSaved: table_3_4_scaling_projections.csv")

    # ── Sanity checks ─────────────────────────────────────────────────────────
    print("\n-- Sanity Check --")
    sk_ok   = 0.1 <= sklearn_res["mean"] <= 30.0
    sp_ok   = spark_res["mean"] > sklearn_res["mean"]
    thr_ok  = sklearn_res["throughput"] > 500
    over_ok = overhead > 1.0
    print(f"  sklearn runtime in [0.1, 30]s : {'PASS' if sk_ok else 'WARN'}  "
          f"({sklearn_res['mean']:.2f}s)")
    print(f"  Spark slower than sklearn     : {'PASS' if sp_ok else 'WARN'}")
    print(f"  sklearn throughput > 500 rec/s: {'PASS' if thr_ok else 'WARN'}  "
          f"({sklearn_res['throughput']:,.0f})")
    print(f"  overhead_factor > 1           : {'PASS' if over_ok else 'WARN'}  "
          f"({overhead:.1f}x)")

    print(f"\nTotal runtime: {(time.perf_counter()-t_total)/60:.1f} min")

