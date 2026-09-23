"""
Module: config.py

Description:
    Centralized configuration for the ORBITA framework.

    All mission parameters, physical bounds, training hyperparameters,
    and pipeline constants are defined here to ensure consistency across
    every module and eliminate scattered magic numbers.
"""

import numpy as np

from physics.oracle import R_EQ

# =============================================================================
# PHYSICAL DOMAIN BOUNDS
# =============================================================================

# Semi-Major Axis operational range [m] (300 km to 2000 km altitude)
TOTAL_SMA_BOUNDS = (R_EQ + 300e3, R_EQ + 2000e3)

# Eccentricity operational range [-] (circular to low-elliptical)
TOTAL_ECC_BOUNDS = (0.0, 0.1)

# Inclination operational range [rad] (equatorial to polar)
TOTAL_INC_BOUNDS = (0.0, np.radians(90.0))

# Angular parameters always span the full 360-degree orbit
RAAN_BOUNDS = (0.0, 2.0 * np.pi)
AOP_BOUNDS = (0.0, 2.0 * np.pi)
TA_BOUNDS = (0.0, 2.0 * np.pi)

# Minimum safe perigee to prevent atmospheric decay or gravity singularities
MIN_SAFE_PERIGEE = R_EQ + 200e3  # 200 km minimum altitude [m]


# =============================================================================
# TIME OF FLIGHT PARAMETERS
# =============================================================================

# Maximum propagation window for single-step dataset generation [s]
MAX_TOF_SECONDS = 30.0 * 60.0  # 30 minutes

# Time step interval for iterative propagation loops [s]
PROPAGATION_STEP_SECONDS = 15.0 * 60.0  # 15 minutes

# Maximum simulation duration for stress tests [s]
MAX_SIMULATION_TOF = 4.0 * 3600.0  # 4 hours


# =============================================================================
# ACTIVE LEARNING & UNCERTAINTY
# =============================================================================

# Epistemic uncertainty threshold to trigger GPS Reset [m]
UNCERTAINTY_THRESHOLD_METERS = 100.0

# Number of MC-Dropout forward passes for uncertainty estimation
MC_DROPOUT_SAMPLES = 50


# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================

# Base training configuration
BASE_EPOCHS = 150
BASE_BATCH_SIZE = 512
BASE_LEARNING_RATE = 1e-3

# Fine-tuning configuration
FINETUNE_EPOCHS = 50
FINETUNE_BATCH_SIZE = 256
FINETUNE_LEARNING_RATE = 1e-5

# Train/validation split ratio
TRAIN_SPLIT = 0.8

# Reproducibility seed for dataset splitting
SPLIT_SEED = 42


# =============================================================================
# DATASET GENERATION
# =============================================================================

# Default number of Monte Carlo samples per expert cell
SAMPLES_PER_EXPERT = 100_000

# Active learning pool and selection sizes
AL_POOL_SIZE = 100_000
AL_HARD_CASES = 5_000
AL_REPLAY_CASES = 15_000


# =============================================================================
# BENCHMARK CONFIGURATION
# =============================================================================

# Number of randomized orbits for time-domain secular degradation tests
TIME_DOMAIN_TEST_CASES = 10_000

# Number of randomized orbits for space-domain Monte Carlo generalization tests
SPACE_DOMAIN_SAMPLES = 100_000


# =============================================================================
# HELPER CONVENTIONS
# =============================================================================

def format_domain_suffix(sma_bounds, ecc_bounds, inc_bounds):
    """
    Formats orbital domain bounds into the standardized string representation:
    - SMA: integer altitude in km, e.g., '300-2000'
    - ECC: 4 decimal places, e.g., '0.0000-0.1000'
    - INC: degrees with 2 decimal places, e.g., '0.00-90.00'

    Args:
        sma_bounds (tuple): (min_sma, max_sma) in meters.
        ecc_bounds (tuple): (min_ecc, max_ecc).
        inc_bounds (tuple): (min_inc, max_inc) in radians or degrees.

    Returns:
        str: Standardized domain string suffix.
    """
    alt_min_km = int((sma_bounds[0] - R_EQ) / 1e3)
    alt_max_km = int((sma_bounds[1] - R_EQ) / 1e3)
    sma_str = f"{alt_min_km}-{alt_max_km}"
    ecc_str = f"{ecc_bounds[0]:.4f}-{ecc_bounds[1]:.4f}"

    inc_min = inc_bounds[0]
    inc_max = inc_bounds[1]
    # Convert to degrees if provided in radians (values <= 2*pi)
    deg_min = np.degrees(inc_min) if inc_min <= 2 * np.pi else inc_min
    deg_max = np.degrees(inc_max) if inc_max <= 2 * np.pi else inc_max
    inc_str = f"{deg_min:.2f}-{deg_max:.2f}"

    return f"{sma_str}_{ecc_str}_{inc_str}"


def get_global_dataset_filename():
    """Returns canonical file path for the global dataset."""
    suffix = format_domain_suffix(TOTAL_SMA_BOUNDS, TOTAL_ECC_BOUNDS, TOTAL_INC_BOUNDS)
    return f"data/datasets/training/orbita_dataset_{suffix}.csv"


def get_global_model_filename(architecture):
    """Returns canonical file path for the global model of a given architecture."""
    suffix = format_domain_suffix(
        TOTAL_SMA_BOUNDS, TOTAL_ECC_BOUNDS, TOTAL_INC_BOUNDS
    )
    if architecture == "tree":
        return f"models/tree/orbita_predictor_tree_{suffix}.joblib"
    return (
        f"models/{architecture}/base/orbita_predictor_"
        f"{architecture}_{suffix}.pth"
    )
