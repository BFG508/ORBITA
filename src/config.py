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
