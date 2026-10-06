"""
Configuration file for Thyroid Disease Classification System
Author: Prasad Kanade
"""

from pathlib import Path

# Directory structure
MODELS_DIR = Path('models')
OUTPUTS_DIR = Path('outputs')
PLOTS_DIR = OUTPUTS_DIR / 'plots'

# Dataset parameters
N_SAMPLES = 150000
N_FEATURES_ORIGINAL = 19
N_FEATURES_SELECTED = 12
RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.20

# Model parameters
XGBOOST_PARAMS = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'eval_metric': 'mlogloss'
}

RANDOMFOREST_PARAMS = {
    'n_estimators': 200,
    'max_depth': 12,
    'min_samples_split': 5,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# Target classes
TARGET_CLASSES = ['negative', 'hypothyroid', 'hyperthyroid']

# Visualization settings
PLOT_STYLE = 'seaborn-v0_8'
PLOT_PALETTE = 'husl'
DPI = 300