"""
Script: test_residuals.py

Description:
    Tests for the shared residuals computation module.

    Verifies:
    1. Output shapes and types are correct.
    2. Residuals are finite (no NaN/Inf from integration).
    3. Analytical state propagation is self-consistent.
    4. Zero TOF produces near-zero residuals.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from physics.oracle import R_EQ
from physics.residuals import compute_analytical_state, compute_mee_residuals

# =============================================================================
# FIXTURES
# =============================================================================

# Standard ISS-like orbit for integration tests
ISS_ORBIT = dict(
    sma=R_EQ + 420e3,
    ecc=0.0007,
    inc=np.radians(51.64),
    raan=np.radians(30),
    aop=np.radians(60),
    ta=np.radians(0),
)

# Higher eccentricity test case
ECCENTRIC_ORBIT = dict(
    sma=R_EQ + 800e3,
    ecc=0.07,
    inc=np.radians(45),
    raan=np.radians(120),
    aop=np.radians(200),
    ta=np.radians(90),
)


# =============================================================================
# RESIDUAL COMPUTATION TESTS
# =============================================================================


class TestComputeMeeResiduals:
    """Verify the core Grey-Box residual pipeline."""

    def test_output_shapes(self):
        """Inputs and residuals must both be 6-element arrays."""
        mee_in, mee_res = compute_mee_residuals(**ISS_ORBIT, tof=300.0)

        assert mee_in.shape == (
            6,
        ), f"MEE inputs shape: {mee_in.shape}, expected (6,)"
        assert mee_res.shape == (
            6,
        ), f"MEE residuals shape: {mee_res.shape}, expected (6,)"

    def test_outputs_are_finite(self):
        """No NaN or Inf must appear in the residual computation."""
        mee_in, mee_res = compute_mee_residuals(**ISS_ORBIT, tof=600.0)

        assert np.isfinite(mee_in).all(), "MEE inputs contain NaN/Inf"
        assert np.isfinite(mee_res).all(), "MEE residuals contain NaN/Inf"

    def test_residuals_near_zero_for_short_tof(self):
        """
        For very short propagation times, the analytical model should be
        very close to the numerical oracle, producing small residuals.
        """
        _, mee_res = compute_mee_residuals(
            **ISS_ORBIT, tof=1.0  # 1 second propagation
        )

        # Residuals should be very small for dt=1s
        max_residual = np.max(np.abs(mee_res))
        assert max_residual < 1.0, (
            f"Residuals too large for 1-second propagation: "
            f"max={max_residual:.6f}"
        )

    def test_residuals_grow_with_tof(self):
        """
        Residuals should generally be larger for longer propagation
        times, since the analytical model accumulates more error.
        """
        _, res_short = compute_mee_residuals(**ISS_ORBIT, tof=60.0)
        _, res_long = compute_mee_residuals(**ISS_ORBIT, tof=1800.0)

        norm_short = np.linalg.norm(res_short)
        norm_long = np.linalg.norm(res_long)

        assert norm_long > norm_short, (
            f"Residuals did not grow with TOF: "
            f"short={norm_short:.6f}, long={norm_long:.6f}"
        )

    def test_true_longitude_residual_is_wrapped(self):
        """The True Longitude residual (index 5) must be in [-pi, pi]."""
        _, mee_res = compute_mee_residuals(**ECCENTRIC_ORBIT, tof=900.0)

        L_residual = mee_res[5]
        assert (
            -np.pi <= L_residual <= np.pi
        ), f"True Longitude residual not wrapped: {L_residual:.6f}"


# =============================================================================
# ANALYTICAL STATE TESTS
# =============================================================================


class TestComputeAnalyticalState:
    """Verify the analytical-only propagation function."""

    def test_output_shapes(self):
        """Position (3,), velocity (3,), and MEE (6,) shapes."""
        pos, vel, mee = compute_analytical_state(**ISS_ORBIT, dt=300.0)

        assert pos.shape == (3,), f"Position shape: {pos.shape}"
        assert vel.shape == (3,), f"Velocity shape: {vel.shape}"
        assert mee.shape == (6,), f"MEE shape: {mee.shape}"

    def test_outputs_are_finite(self):
        """All outputs must be finite."""
        pos, vel, mee = compute_analytical_state(**ISS_ORBIT, dt=600.0)

        assert np.isfinite(pos).all(), "Position contains NaN/Inf"
        assert np.isfinite(vel).all(), "Velocity contains NaN/Inf"
        assert np.isfinite(mee).all(), "MEE contains NaN/Inf"

    def test_position_magnitude_is_reasonable(self):
        """
        For LEO propagation, the position vector norm should be
        roughly Earth radius + altitude (6700-8400 km).
        """
        pos, _, _ = compute_analytical_state(**ISS_ORBIT, dt=300.0)

        r_mag = np.linalg.norm(pos)
        assert 6.3e6 < r_mag < 8.5e6, (
            f"Position magnitude {r_mag / 1e3:.0f} km is outside"
            " expected LEO range"
        )

    def test_velocity_magnitude_is_reasonable(self):
        """
        LEO orbital velocity should be roughly 7.0-8.0 km/s.
        """
        _, vel, _ = compute_analytical_state(**ISS_ORBIT, dt=300.0)

        v_mag = np.linalg.norm(vel)
        assert 6.5e3 < v_mag < 8.5e3, (
            f"Velocity magnitude {v_mag:.0f} m/s is outside"
            " expected LEO range"
        )

    def test_mee_semilatus_rectum_positive(self):
        """The semi-latus rectum (p, MEE index 0) must be positive."""
        _, _, mee = compute_analytical_state(**ISS_ORBIT, dt=300.0)

        assert mee[0] > 0, f"Semi-latus rectum p={mee[0]} is not positive"
