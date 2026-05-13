"""
02b_feature_extraction_enhanced.py
Extract 39 enhanced vibration features per acc file.

Feature breakdown (39 total):
  Original 18:  RMS, p2p, kurtosis, crest factor (x2 axes) + 5 FFT bands (x2 axes)
  HOT 4:        skewness, shape factor (x2 axes)
  WPD 8:        wavelet packet energy at 4 nodes (x2 axes), db4 level-4
  Fault 8:      BPFO, BPFI, BSF, FTF envelope spectral energy (x2 axes)
  PCA HI 1:     PC1 of 38 raw features, fit on train healthy zone (first 50% cycles)

Note on frequency resolution:
  Sample rate = 25600 Hz, N = 2560 samples → 10 Hz/bin.
  Fault freq window = ±25 Hz (≈2.5 bins each side) for reliable capture.
  FTF (cage, ~13 Hz for cond-1) is only 1.3 bins from DC; treat with caution.

NSK 6804 geometry: n=13 balls, d=3.5 mm ball dia, D=25.6 mm pitch dia.
Shaft speed: cond-1=1800 rpm (30 Hz), cond-2=1650 rpm (27.5 Hz), cond-3=1500 rpm (25 Hz).

Two-pass approach:
  Pass 1 (parallel):  38 raw features → {prefix}_{bearing}_enhanced38.csv
  Pass 2 (sequential): PCA HI added  → {prefix}_{bearing}_enhanced_features.csv
"""

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
from scipy.signal import hilbert
from scipy.stats import kurtosis as scipy_kurtosis
from scipy.stats import skew as scipy_skew
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from femto.config import (
    FEATURE_COLS, FEMTO_PROCESSED, FEMTO_RAW_ROOT,
    FFT_N, FREQ_BANDS, SAMPLE_RATE,
    TEST_BEARINGS, TRAIN_BEARINGS,
    bearing_condition,
)

# ── Enhanced feature name lists ───────────────────────────────────────────────

HOT_FEATURES = [
    "horiz_skewness", "horiz_shape_factor",
    "vert_skewness",  "vert_shape_factor",
]

_WPD_NODES = ["aaaa", "aaad", "aada", "addd"]
WPD_FEATURES = (
    [f"horiz_wpd_{n}" for n in _WPD_NODES] +
    [f"vert_wpd_{n}"  for n in _WPD_NODES]
)

_FAULT_NAMES = ["BPFO", "BPFI", "BSF", "FTF"]
FAULT_FEATURES = (
    [f"horiz_{f}" for f in _FAULT_NAMES] +
    [f"vert_{f}"  for f in _FAULT_NAMES]
)

ENHANCED_FEATURE_COLS = FEATURE_COLS + HOT_FEATURES + WPD_FEATURES + FAULT_FEATURES  # 38
ALL_ENHANCED_COLS     = ENHANCED_FEATURE_COLS + ["pca_hi"]                           # 39

# ── NSK 6804 bearing geometry and fault frequencies ───────────────────────────

_N_BALLS   = 13
_BALL_DIA  = 3.5     # mm
_PITCH_DIA = 25.6    # mm
_RATIO     = _BALL_DIA / _PITCH_DIA   # 0.13672

_SHAFT_HZ = {1: 30.0, 2: 27.5, 3: 25.0}   # shaft frequency (Hz) per condition

def _fault_freqs(fr: float) -> dict:
    bpfo = (13 / 2) * fr * (1 - _RATIO)
    bpfi = (13 / 2) * fr * (1 + _RATIO)
    bsf  = (_PITCH_DIA / (2 * _BALL_DIA)) * fr * (1 - _RATIO ** 2)
    ftf  = 0.5 * fr * (1 - _RATIO)
    return {"BPFO": bpfo, "BPFI": bpfi, "BSF": bsf, "FTF": ftf}

_FAULT_FREQS_BY_COND = {c: _fault_freqs(_SHAFT_HZ[c]) for c in [1, 2, 3]}

# ── Pre-compute frequency axes and masks ─────────────────────────────────────

_FREQS      = np.fft.rfftfreq(FFT_N, d=1.0 / SAMPLE_RATE)
_BAND_MASKS = [(_FREQS >= lo) & (_FREQS < hi) for (lo, hi) in FREQ_BANDS]

_ENV_WIN_HZ = 25.0   # ±25 Hz window; 10 Hz/bin → ~5 bins total per fault freq

_FAULT_MASKS_BY_COND = {
    cond: {
        name: (_FREQS >= ff - _ENV_WIN_HZ) & (_FREQS <= ff + _ENV_WIN_HZ)
        for name, ff in freqs.items()
    }
    for cond, freqs in _FAULT_FREQS_BY_COND.items()
}


# ── Per-file worker (module-level for ProcessPoolExecutor) ────────────────────

def _extract_one_enhanced(args: tuple):
    """Worker: extract 38 raw enhanced features from one acc file."""
    filepath_str, cycle_idx, condition = args
    try:
        raw   = pd.read_csv(filepath_str, header=None, usecols=[4, 5])
        horiz = raw.iloc[:, 0].values.astype(np.float64)
        vert  = raw.iloc[:, 1].values.astype(np.float64)
        return [cycle_idx] + _compute_enhanced_features(horiz, vert, condition)
    except Exception:
        return None


def _compute_enhanced_features(horiz: np.ndarray, vert: np.ndarray,
                                condition: int) -> list:
    """Return list of 38 feature values (no PCA HI)."""
    feats = []

    # ── Original 18 ──────────────────────────────────────────────────────────
    for x in (horiz, vert):
        rms   = float(np.sqrt(np.mean(x ** 2)))
        p2p   = float(np.max(x) - np.min(x))
        kurt  = float(scipy_kurtosis(x, fisher=True))
        crest = float(np.max(np.abs(x))) / (rms + 1e-12)
        feats.extend([rms, p2p, kurt, crest])

    for x in (horiz, vert):
        mag    = np.abs(np.fft.rfft(x)) / FFT_N
        energy = mag ** 2
        for mask in _BAND_MASKS:
            feats.append(float(np.sum(energy[mask])))

    # ── Higher-order time (4) ────────────────────────────────────────────────
    for x in (horiz, vert):
        skewness = float(scipy_skew(x))
        rms      = float(np.sqrt(np.mean(x ** 2)))
        shape    = rms / (float(np.mean(np.abs(x))) + 1e-12)
        feats.extend([skewness, shape])

    # ── Wavelet packet energies (8) ──────────────────────────────────────────
    for x in (horiz, vert):
        wp = pywt.WaveletPacket(data=x, wavelet="db4", maxlevel=4)
        for node in _WPD_NODES:
            feats.append(float(np.sum(wp[node].data ** 2)))

    # ── Bearing fault frequencies via envelope spectrum (8) ──────────────────
    fault_masks = _FAULT_MASKS_BY_COND[condition]
    for x in (horiz, vert):
        env      = np.abs(hilbert(x))
        env_spec = np.abs(np.fft.rfft(env)) ** 2    # power spectrum of envelope
        for fname in _FAULT_NAMES:
            feats.append(float(np.sum(env_spec[fault_masks[fname]])))

    return feats   # 38 values


# ── Per-bearing extraction ────────────────────────────────────────────────────

def extract_bearing_enhanced(acc_dir: Path, out_path: Path,
                             condition: int, n_workers: int = 4) -> pd.DataFrame:
    acc_files = sorted(acc_dir.glob("acc_*.csv"))
    if not acc_files:
        raise FileNotFoundError(f"No acc_*.csv files in {acc_dir}")

    args_list = [(str(f), i + 1, condition) for i, f in enumerate(acc_files)]
    results   = [None] * len(args_list)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_idx = {executor.submit(_extract_one_enhanced, a): i
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
    cols = ["cycle"] + ENHANCED_FEATURE_COLS
    df = pd.DataFrame(valid, columns=cols)
    df.to_csv(out_path, index=False)
    return df


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    FEMTO_PROCESSED.mkdir(parents=True, exist_ok=True)

    splits = [
        ("Training_set",   TRAIN_BEARINGS, "train"),
        ("Test_set",       TEST_BEARINGS,  "test"),
        ("Validation_Set", TEST_BEARINGS,  "val"),
    ]
    total = len(TRAIN_BEARINGS) + 2 * len(TEST_BEARINGS)

    # ── Pass 1: parallel per-file feature extraction ──────────────────────────
    print("=" * 65)
    print("Pass 1: Extracting 38 raw enhanced features (parallel)")
    print("=" * 65)
    done_count = 0
    for set_name, bearings, prefix in splits:
        set_path = FEMTO_RAW_ROOT / set_name
        for bname in bearings:
            done_count += 1
            out38 = FEMTO_PROCESSED / f"{prefix}_{bname}_enhanced38.csv"
            if out38.exists():
                print(f"[{done_count:2d}/{total}] SKIP {out38.name}")
                continue
            acc_dir = set_path / bname
            cond    = bearing_condition(bname)
            n_files = len(list(acc_dir.glob("acc_*.csv")))
            print(f"[{done_count:2d}/{total}] {set_name}/{bname} "
                  f"({n_files} files, cond={cond}) -> {out38.name}")
            df = extract_bearing_enhanced(acc_dir, out38, condition=cond)
            print(f"  Saved {len(df)} rows, {len(df.columns)} cols")

    # ── Pass 2: PCA health indicator ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Pass 2: Computing PCA Health Indicator")
    print("=" * 65)

    # Load all train enhanced38 CSVs; build healthy-zone matrix (first 50% cycles)
    healthy_rows = []
    for bname in TRAIN_BEARINGS:
        df       = pd.read_csv(FEMTO_PROCESSED / f"train_{bname}_enhanced38.csv")
        cutoff   = df["cycle"].max() // 2
        sub      = df[df["cycle"] <= cutoff]
        healthy_rows.append(sub[ENHANCED_FEATURE_COLS].values)
        print(f"  {bname}: {len(sub)}/{len(df)} rows in healthy zone (cycle <= {cutoff})")

    X_healthy = np.vstack(healthy_rows)
    print(f"\n  Total healthy-zone rows for PCA fit: {len(X_healthy)}")

    ss  = StandardScaler()
    X_h = ss.fit_transform(X_healthy)
    pca = PCA(n_components=1, random_state=42)
    pca.fit(X_h)
    print(f"  PCA PC1 explained variance ratio: {pca.explained_variance_ratio_[0]:.4f}")

    # Apply to all splits → save final enhanced_features.csv
    print("\n  Writing final enhanced_features.csv for all bearings:")
    for set_name, bearings, prefix in splits:
        for bname in bearings:
            in_path  = FEMTO_PROCESSED / f"{prefix}_{bname}_enhanced38.csv"
            out_path = FEMTO_PROCESSED / f"{prefix}_{bname}_enhanced_features.csv"
            if not in_path.exists():
                print(f"  MISSING {in_path.name} — skipping")
                continue
            df = pd.read_csv(in_path)
            if df.empty:
                df["pca_hi"] = pd.Series(dtype=float)
                df.to_csv(out_path, index=False)
                print(f"  {out_path.name}  (0 rows — skipped PCA transform)")
                continue
            X_scaled     = ss.transform(df[ENHANCED_FEATURE_COLS].values)
            df["pca_hi"] = pca.transform(X_scaled).ravel()
            df.to_csv(out_path, index=False)
            print(f"  {out_path.name}  ({len(df)} rows, {len(df.columns)} cols)")

    print(f"\nDone. Enhanced feature CSVs in: {FEMTO_PROCESSED}")
    print(f"Feature count per file: {len(ALL_ENHANCED_COLS)} (+ cycle column)")
