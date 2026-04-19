"""
Script: test_coordinate_transforms.py

Description:
    Roundtrip and self-consistency tests for the ORBITA coordinate
    transformation pipeline (COE ↔ ECI, COE → MEE → COE) and the
    Kepler solver.

    These tests verify that:
    1. COE → ECI → COE is a lossless roundtrip within numerical precision.
    2. COE → MEE → COE is a lossless roundtrip for non-singular orbits.
    3. The unperturbed Kepler propagator returns the initial state at dt=0.
    4. The residuals module produces correctly shaped outputs.
"""

import sys
import os
import numpy as np
import pytest

# Add the src directory to PYTHONPATH so imports work as expected
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from physics.oracle import (
    MU, R_EQ,
    coe_to_eci, eci_to_coe, coe_to_mee, mee_to_coe
)
from physics.kepler import get_keplerian
from physics.residuals import wrap_to_pi


# =============================================================================
# TEST FIXTURES
# =============================================================================

# Non-degenerate LEO orbit for robust testing (avoids i=0, e=0 edge cases)
LEO_ORBIT = {
    "sma": R_EQ + 500e3,     # 500 km altitude
    "ecc": 0.01,             # Nearly circular
    "inc": np.radians(51.6), # ISS-like inclination
    "raan": np.radians(45),
    "aop": np.radians(90),
    "ta": np.radians(30),
}

# High eccentricity test orbit
ELLIPTICAL_ORBIT = {
    "sma": R_EQ + 1000e3,
    "ecc": 0.08,
    "inc": np.radians(75),
    "raan": np.radians(120),
    "aop": np.radians(200),
    "ta": np.radians(270),
}

# Near-equatorial orbit (tests inclination near zero)
EQUATORIAL_ORBIT = {
    "sma": R_EQ + 400e3,
    "ecc": 0.005,
    "inc": np.radians(1.0),  # Near-equatorial (not exactly zero)
    "raan": np.radians(0),
    "aop": np.radians(0),
    "ta": np.radians(180),
}


ALL_ORBITS = [LEO_ORBIT, ELLIPTICAL_ORBIT, EQUATORIAL_ORBIT]
ORBIT_IDS = ["LEO_ISS", "ELLIPTICAL_75deg", "EQUATORIAL_1deg"]


# =============================================================================
# ROUNDTRIP TESTS: COE → ECI → COE
# =============================================================================

class TestCoeEciRoundtrip:
    """Verify that COE → ECI → COE is a lossless transformation."""

    @pytest.mark.parametrize("orbit", ALL_ORBITS, ids=ORBIT_IDS)
    def test_roundtrip_preserves_elements(self, orbit):
        """Each COE element must survive the ECI roundtrip."""
        sma, ecc, inc = orbit["sma"], orbit["ecc"], orbit["inc"]
        raan, aop, ta = orbit["raan"], orbit["aop"], orbit["ta"]

        # Forward: COE → ECI
        pos, vel = coe_to_eci(MU, sma, ecc, inc, raan, aop, ta)

        # Inverse: ECI → COE
        recovered = eci_to_coe(MU, pos, vel)
        r_sma, r_ecc, r_inc, r_raan, r_aop, r_ta = recovered

        # Validate scalar elements (SMA, ECC)
        assert r_sma == pytest.approx(sma, rel=1e-10), \
            f"SMA mismatch: {r_sma} vs {sma}"
        assert r_ecc == pytest.approx(ecc, rel=1e-8), \
            f"ECC mismatch: {r_ecc} vs {ecc}"

        # Validate angular elements with wrapping tolerance
        assert r_inc == pytest.approx(inc, abs=1e-10), \
            f"INC mismatch: {r_inc} vs {inc}"

        # Angular residuals need wrapping before comparison
        assert abs(wrap_to_pi(r_raan - raan)) < 1e-10, \
            f"RAAN mismatch: {r_raan} vs {raan}"
        assert abs(wrap_to_pi(r_aop - aop)) < 1e-10, \
            f"AOP mismatch: {r_aop} vs {aop}"
        assert abs(wrap_to_pi(r_ta - ta)) < 1e-10, \
            f"TA mismatch: {r_ta} vs {ta}"


# =============================================================================
# ROUNDTRIP TESTS: COE → MEE → COE
# =============================================================================

class TestCoeMeeRoundtrip:
    """Verify that COE → MEE → COE preserves orbital elements."""

    @pytest.mark.parametrize("orbit", ALL_ORBITS, ids=ORBIT_IDS)
    def test_roundtrip_preserves_elements(self, orbit):
        """MEE roundtrip must recover the original COE to machine precision."""
        sma, ecc, inc = orbit["sma"], orbit["ecc"], orbit["inc"]
        raan, aop, ta = orbit["raan"], orbit["aop"], orbit["ta"]

        # Forward: COE → MEE
        mee = coe_to_mee(sma, ecc, inc, raan, aop, ta)

        # Inverse: MEE → COE
        recovered = mee_to_coe(*mee)
        r_sma, r_ecc, r_inc, r_raan, r_aop, r_ta = recovered

        assert r_sma == pytest.approx(sma, rel=1e-10), \
            f"SMA mismatch: {r_sma} vs {sma}"
        assert r_ecc == pytest.approx(ecc, rel=1e-8), \
            f"ECC mismatch: {r_ecc} vs {ecc}"
        assert r_inc == pytest.approx(inc, abs=1e-10), \
            f"INC mismatch: {r_inc} vs {inc}"
        assert abs(wrap_to_pi(r_raan - raan)) < 1e-10, \
            f"RAAN mismatch: {r_raan} vs {raan}"
        assert abs(wrap_to_pi(r_aop - aop)) < 1e-10, \
            f"AOP mismatch: {r_aop} vs {aop}"
        assert abs(wrap_to_pi(r_ta - ta)) < 1e-10, \
            f"TA mismatch: {r_ta} vs {ta}"


# =============================================================================
# KEPLER SOLVER TESTS
# =============================================================================

class TestKeplerSolver:
    """Verify the unperturbed Kepler propagator's boundary conditions."""

    @pytest.mark.parametrize("orbit", ALL_ORBITS, ids=ORBIT_IDS)
    def test_zero_tof_returns_initial_state(self, orbit):
        """At dt=0, the propagated state must equal the initial state."""
        sma, ecc, inc = orbit["sma"], orbit["ecc"], orbit["inc"]
        raan, aop, ta = orbit["raan"], orbit["aop"], orbit["ta"]

        r0, v0 = coe_to_eci(MU, sma, ecc, inc, raan, aop, ta)
        r_prop, v_prop = get_keplerian(MU, r0, v0, dt=0.0)

        np.testing.assert_allclose(
            r_prop, r0, atol=1e-8,
            err_msg="Position changed at dt=0"
        )
        np.testing.assert_allclose(
            v_prop, v0, atol=1e-8,
            err_msg="Velocity changed at dt=0"
        )

    @pytest.mark.parametrize("orbit", ALL_ORBITS, ids=ORBIT_IDS)
    def test_full_orbit_period_returns_to_start(self, orbit):
        """After one full orbital period, state must return to start."""
        sma = orbit["sma"]
        ecc, inc = orbit["ecc"], orbit["inc"]
        raan, aop, ta = orbit["raan"], orbit["aop"], orbit["ta"]

        T = 2.0 * np.pi * np.sqrt(sma ** 3 / MU)  # Orbital period [s]

        r0, v0 = coe_to_eci(MU, sma, ecc, inc, raan, aop, ta)
        r_prop, v_prop = get_keplerian(MU, r0, v0, dt=T)

        # Tolerance is larger here due to accumulated numerical error
        np.testing.assert_allclose(
            r_prop, r0, atol=1.0,
            err_msg="Position drift after one full period exceeds 1 m"
        )
        np.testing.assert_allclose(
            v_prop, v0, atol=1e-3,
            err_msg="Velocity drift after one full period exceeds 1 mm/s"
        )


# =============================================================================
# WRAP_TO_PI TESTS
# =============================================================================

class TestWrapToPi:
    """Verify the angular wrapping utility."""

    def test_within_range_unchanged(self):
        """Angles already in [-pi, pi] should be unchanged."""
        assert wrap_to_pi(0.0) == pytest.approx(0.0)
        assert wrap_to_pi(1.0) == pytest.approx(1.0)
        assert wrap_to_pi(-1.0) == pytest.approx(-1.0)

    def test_wrap_positive_overflow(self):
        """2*pi should wrap to approximately 0."""
        assert abs(wrap_to_pi(2 * np.pi)) < 1e-14

    def test_wrap_negative_overflow(self):
        """-2*pi should wrap to approximately 0."""
        assert abs(wrap_to_pi(-2 * np.pi)) < 1e-14

    def test_wrap_large_angle(self):
        # 5*pi wraps to -pi (equivalent to pi on the unit circle)
        assert abs(wrap_to_pi(5 * np.pi)) == pytest.approx(np.pi, abs=1e-14)

    def test_vectorized(self):
        """Must work with numpy arrays."""
        angles = np.array([0.0, np.pi, 2 * np.pi, -np.pi])
        wrapped = wrap_to_pi(angles)
        assert wrapped.shape == angles.shape


# =============================================================================
# NORMALIZATION IDENTITY TEST
# =============================================================================

class TestNormalization:
    """Verify that normalize → unnormalize yields the original data."""

    def test_z_score_roundtrip(self):
        """Z-score normalization must be perfectly invertible."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((100, 6))

        mean = data.mean(axis=0)
        std = data.std(axis=0)

        normalized = (data - mean) / std
        recovered = normalized * std + mean

        np.testing.assert_allclose(
            recovered, data, atol=1e-12,
            err_msg="Z-score roundtrip failed"
        )
