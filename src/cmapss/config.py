from pathlib import Path

SEED = 42
RUL_CEILING = 125
ROLLING_WINDOW = 5
IF_N_ESTIMATORS_DEFAULT = 200
IF_CONTAMINATION = 0.05
RIDGE_ALPHA = 10.0
THRESHOLD_SINGLE = 0.20      # FD001, FD003
THRESHOLD_MULTI = 0.15       # FD002, FD004
KMEANS_K_MULTI = 6
ZONE_HEALTHY = (63, 125)     # normal zone RUL range
ZONE_DEGRADED = (0, 62)      # anomaly zone RUL range

# File is at src/cmapss/config.py → go 3 levels up to reach project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
RESULTS_DIR = PROJECT_ROOT / 'results'
TABLES_DIR = RESULTS_DIR / 'tables'
FIGURES_DIR = RESULTS_DIR / 'figures'

SUBSETS = ['FD001', 'FD002', 'FD003', 'FD004']
SINGLE_CONDITION = ['FD001', 'FD003']
MULTI_CONDITION = ['FD002', 'FD004']
