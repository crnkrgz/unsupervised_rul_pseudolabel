"""
02_feature_extraction.py
Extract 18 vibration features per acc file for all bearings.

Time-domain (8): RMS, peak-to-peak, kurtosis, crest factor (x2 axes)
Frequency-domain (10): FFT energy in 5 bands (x2 axes)

Sample rate = 25.6 kHz, FFT N = 2560, one feature row per acc file.
Uses ProcessPoolExecutor for parallel file I/O.

Outputs: data/processed/femto/{prefix}_{bearing}_features.csv
         prefix in {train, test, val}
"""

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as scipy_kurtosis

sys.path.insert(0, str(Path(__file__).parent.parent))
from femto.config import (
    FEATURE_COLS, FEMTO_PROCESSED, FEMTO_RAW_ROOT,
    FFT_N, FREQ_BANDS, SAMPLE_RATE,
    TEST_BEARINGS, TRAIN_BEARINGS,
)

# Pre-compute frequency axis and band masks at module level (used by workers)
_FREQS      = np.fft.rfftfreq(FFT_N, d=1.0 / SAMPLE_RATE)
_BAND_MASKS = [((_FREQS >= lo) & (_FREQS < hi)) for (lo, hi) in FREQ_BANDS]


def _extract_one(args: tuple):
    """
    Worker: extract 18 features from a single acc CSV file.
    Returns [cycle_idx, f1, ..., f18] or None on error.
    """
    filepath_str, cycle_idx = args
    try:
        raw = pd.read_csv(filepath_str, header=None, usecols=[4, 5])
        horiz = raw.iloc[:, 0].values.astype(np.float64)
        vert  = raw.iloc[:, 1].values.astype(np.float64)
        return [cycle_idx] + _compute_features(horiz, vert)
    except Exception:
        return None


def _compute_features(horiz: np.ndarray, vert: np.ndarray) -> list:
    """Compute 8 time-domain + 10 frequency-domain features (18 total)."""
    feats = []

    # Time-domain: RMS, p2p, kurtosis, crest factor per axis
    for x in (horiz, vert):
        rms   = float(np.sqrt(np.mean(x ** 2)))
        p2p   = float(np.max(x) - np.min(x))
        kurt  = float(scipy_kurtosis(x, fisher=True))
        crest = float(np.max(np.abs(x))) / (rms + 1e-12)
        feats.extend([rms, p2p, kurt, crest])

    # Frequency-domain: FFT energy per band per axis
    for x in (horiz, vert):
        mag    = np.abs(np.fft.rfft(x)) / FFT_N
        energy = mag ** 2
        for mask in _BAND_MASKS:
            feats.append(float(np.sum(energy[mask])))

    return feats


def extract_bearing(acc_dir: Path, out_path: Path, n_workers: int = 4) -> pd.DataFrame:
    """
    Extract features for all acc files in acc_dir, save CSV, return DataFrame.
    Files processed in parallel; output is sorted by cycle index.
    """
    acc_files = sorted(acc_dir.glob("acc_*.csv"))
    if not acc_files:
        raise FileNotFoundError(f"No acc_*.csv files found in {acc_dir}")

    args_list = [(str(f), i + 1) for i, f in enumerate(acc_files)]
    results   = [None] * len(args_list)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_idx = {executor.submit(_extract_one, a): i
                         for i, a in enumerate(args_list)}
        done = 0
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            results[idx] = fut.result()
            done += 1
            if done % 200 == 0 or done == len(args_list):
                print(f"    {done}/{len(args_list)} files", end="\r", flush=True)

    print()
    valid = [r for r in results if r is not None]
    valid.sort(key=lambda r: r[0])
    df = pd.DataFrame(valid, columns=["cycle"] + FEATURE_COLS)
    df.to_csv(out_path, index=False)
    return df


if __name__ == "__main__":
    FEMTO_PROCESSED.mkdir(parents=True, exist_ok=True)

    splits = [
        ("Training_set",   TRAIN_BEARINGS, "train"),
        ("Test_set",       TEST_BEARINGS,  "test"),
        ("Validation_Set", TEST_BEARINGS,  "val"),
    ]

    total_bearings = len(TRAIN_BEARINGS) + 2 * len(TEST_BEARINGS)
    done_count = 0

    for set_name, bearings, prefix in splits:
        set_path = FEMTO_RAW_ROOT / set_name
        for bname in bearings:
            done_count += 1
            out_path = FEMTO_PROCESSED / f"{prefix}_{bname}_features.csv"
            if out_path.exists():
                print(f"[{done_count:2d}/{total_bearings}] SKIP {out_path.name}")
                continue
            acc_dir = set_path / bname
            n_files = len(list(acc_dir.glob("acc_*.csv")))
            print(f"[{done_count:2d}/{total_bearings}] {set_name}/{bname} "
                  f"({n_files} files) -> {out_path.name}")
            df = extract_bearing(acc_dir, out_path)
            print(f"  Saved {len(df)} rows, {len(df.columns)} cols")

    print("\nDone. Feature CSVs in:", FEMTO_PROCESSED)
