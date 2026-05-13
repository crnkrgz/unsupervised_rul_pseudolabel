from pathlib import Path

_HERE = Path(__file__).parent       # src/femto/
_ROOT = _HERE.parent.parent         # project root

FEMTO_RAW_ROOT  = _ROOT / "data" / "raw" / "10.+FEMTO+Bearing" / "FEMTO"
FEMTO_PROCESSED = _ROOT / "data" / "processed" / "femto"
FEMTO_RESULTS   = _ROOT / "results" / "femto"

SAMPLE_RATE = 25_600   # Hz
FFT_N       = 2_560    # samples per acc file

TRAIN_BEARINGS = [
    "Bearing1_1", "Bearing1_2",
    "Bearing2_1", "Bearing2_2",
    "Bearing3_1", "Bearing3_2",
]
TEST_BEARINGS = [
    "Bearing1_3", "Bearing1_4", "Bearing1_5", "Bearing1_6", "Bearing1_7",
    "Bearing2_3", "Bearing2_4", "Bearing2_5", "Bearing2_6", "Bearing2_7",
    "Bearing3_3",
]


def bearing_condition(name: str) -> int:
    """BearingX_Y -> condition integer (1, 2, or 3)."""
    return int(name[7])


TIME_FEATURES = [
    "horiz_rms", "horiz_p2p", "horiz_kurtosis", "horiz_crest",
    "vert_rms",  "vert_p2p",  "vert_kurtosis",  "vert_crest",
]
FREQ_FEATURES = [
    "horiz_band_0_1kHz",  "horiz_band_1_3kHz",  "horiz_band_3_6kHz",
    "horiz_band_6_10kHz", "horiz_band_10_12kHz",
    "vert_band_0_1kHz",   "vert_band_1_3kHz",   "vert_band_3_6kHz",
    "vert_band_6_10kHz",  "vert_band_10_12kHz",
]
FEATURE_COLS   = TIME_FEATURES + FREQ_FEATURES   # 18 vibration features
CONDITION_COLS = ["cond_1", "cond_2", "cond_3"]  # 3 condition dummies
AE_INPUT_COLS  = FEATURE_COLS + CONDITION_COLS   # 21 AE input features

FREQ_BANDS = [          # (lo_Hz, hi_Hz) — hi is exclusive except last
    (0,      1_000),
    (1_000,  3_000),
    (3_000,  6_000),
    (6_000,  10_000),
    (10_000, 99_999),   # 10 kHz to Nyquist (12.8 kHz)
]

# Strict C-MAPSS transfer parameters
ROLLING_WINDOW       = 5
IF_N_ESTIMATORS      = 200
IF_MAX_SAMPLES       = 256
IF_CONTAMINATION     = 0.20
DEGRADATION_QUANTILE = 0.80
RIDGE_ALPHA          = 1.0
AE_LATENT_DIM        = 2
AE_N_EPOCHS          = 50
AE_LR                = 1e-3
SEED                 = 42

TV_STEP = 50   # cycles between time-varying evaluation points
