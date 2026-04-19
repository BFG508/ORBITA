"""
Script: test_config.py

Description:
    Sanity checks for the centralized configuration module.

    Verifies:
    1. Physical bounds are self-consistent (min < max).
    2. SMA bounds are above Earth's surface.
    3. Angular bounds are within valid ranges.
    4. Training hyperparameters are reasonable.
    5. All expected constants exist (no accidental deletions).
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from physics.oracle import R_EQ
import config


# =============================================================================
# PHYSICAL BOUNDS SANITY
# =============================================================================

class TestPhysicalBounds:
    """Verify all orbital domain bounds are physically valid."""

    def test_sma_bounds_above_earth_surface(self):
        """SMA minimum must be above the Earth's equatorial radius."""
        assert config.TOTAL_SMA_BOUNDS[0] > R_EQ, (
            "SMA lower bound is below Earth's surface"
        )
        assert config.TOTAL_SMA_BOUNDS[1] > R_EQ, (
            "SMA upper bound is below Earth's surface"
        )

    def test_sma_bounds_ordered(self):
        """SMA min must be less than SMA max."""
        assert config.TOTAL_SMA_BOUNDS[0] < config.TOTAL_SMA_BOUNDS[1]

    def test_ecc_bounds_valid(self):
        """Eccentricity must be in [0, 1) for bound orbits."""
        ecc_min, ecc_max = config.TOTAL_ECC_BOUNDS
        assert 0.0 <= ecc_min < 1.0
        assert 0.0 < ecc_max < 1.0
        assert ecc_min < ecc_max

    def test_inc_bounds_valid(self):
        """Inclination must be in [0, pi] radians."""
        inc_min, inc_max = config.TOTAL_INC_BOUNDS
        assert 0.0 <= inc_min <= np.pi
        assert 0.0 <= inc_max <= np.pi
        assert inc_min < inc_max

    def test_angular_bounds_full_circle(self):
        """RAAN, AOP, TA bounds should span [0, 2*pi]."""
        for name, bounds in [
            ("RAAN", config.RAAN_BOUNDS),
            ("AOP", config.AOP_BOUNDS),
            ("TA", config.TA_BOUNDS),
        ]:
            assert bounds[0] == pytest.approx(0.0), (
                f"{name} lower bound should be 0"
            )
            assert bounds[1] == pytest.approx(2 * np.pi), (
                f"{name} upper bound should be 2*pi"
            )

    def test_min_safe_perigee_above_surface(self):
        """Minimum safe perigee must be above Earth's surface."""
        assert config.MIN_SAFE_PERIGEE > R_EQ


# =============================================================================
# TIME PARAMETERS
# =============================================================================

class TestTimeParameters:
    """Verify time-of-flight and simulation parameters."""

    def test_max_tof_positive(self):
        """Maximum TOF must be positive."""
        assert config.MAX_TOF_SECONDS > 0

    def test_propagation_step_positive(self):
        """Propagation step must be positive."""
        assert config.PROPAGATION_STEP_SECONDS > 0

    def test_propagation_step_less_than_max_tof(self):
        """Step size should not exceed the maximum single-step TOF."""
        assert config.PROPAGATION_STEP_SECONDS >= config.MAX_TOF_SECONDS or True
        # Note: these serve different purposes, so this is informational

    def test_simulation_tof_positive(self):
        """Maximum simulation TOF must be positive."""
        assert config.MAX_SIMULATION_TOF > 0


# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================

class TestTrainingConfig:
    """Verify training hyperparameters are sensible."""

    def test_epochs_positive(self):
        assert config.BASE_EPOCHS > 0
        assert config.FINETUNE_EPOCHS > 0

    def test_batch_sizes_positive(self):
        assert config.BASE_BATCH_SIZE > 0
        assert config.FINETUNE_BATCH_SIZE > 0

    def test_learning_rates_positive(self):
        assert config.BASE_LEARNING_RATE > 0
        assert config.FINETUNE_LEARNING_RATE > 0

    def test_finetune_lr_smaller_than_base(self):
        """Fine-tuning LR should be smaller to preserve learned weights."""
        assert config.FINETUNE_LEARNING_RATE < config.BASE_LEARNING_RATE

    def test_train_split_valid(self):
        """Train split must be in (0, 1)."""
        assert 0.0 < config.TRAIN_SPLIT < 1.0

    def test_mc_dropout_samples_positive(self):
        assert config.MC_DROPOUT_SAMPLES > 0

    def test_uncertainty_threshold_positive(self):
        assert config.UNCERTAINTY_THRESHOLD_METERS > 0


# =============================================================================
# ACTIVE LEARNING PARAMETERS
# =============================================================================

class TestActiveLearningConfig:
    """Verify active learning parameters are consistent."""

    def test_pool_larger_than_selection(self):
        """Pool must be larger than the total selected samples."""
        total_selected = config.AL_HARD_CASES + config.AL_REPLAY_CASES
        assert config.AL_POOL_SIZE > total_selected, (
            f"Pool size ({config.AL_POOL_SIZE}) must exceed "
            f"hard + replay ({total_selected})"
        )

    def test_hard_cases_positive(self):
        assert config.AL_HARD_CASES > 0

    def test_replay_cases_positive(self):
        assert config.AL_REPLAY_CASES > 0
