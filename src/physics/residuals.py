"""
Module: residuals.py

Description:
    Shared residual computation logic for the ORBITA framework.

    Encapsulates the core Grey-Box pipeline: propagates a given orbital
    state using both the Numerical Oracle (ground truth) and the
    Analytical Baseline (ESTHER + Kepler), then computes the residual
    error in Modified Equinoctial Elements (MEE).

    This module eliminates code duplication across dataset generation,
    active learning, and benchmarking scripts.
"""

import numpy as np

from physics.analytical import compute_general_solution
from physics.kepler import get_keplerian
from physics.oracle import (J2, J3, MU, R_EQ, coe_to_eci, coe_to_mee,
                            eci_to_coe, get_ground_truth)


def wrap_to_pi(angle):
    """
    Wraps an angle or array of angles to the interval [-pi, pi].

    Crucial for computing accurate angular errors without
    360-degree discontinuity jumps in the True Longitude residual.

    Args:
        angle (float or np.ndarray): Angle in radians.

    Returns:
        float or np.ndarray: Wrapped angle in radians.
    """
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def compute_mee_residuals(sma, ecc, inc, raan, aop, ta, tof):
    """
    Computes the MEE residual error between the Numerical Oracle and
    the Analytical Baseline for a single orbital state propagation.

    This is the core computation shared by dataset generation, active
    learning mining, and benchmark evaluation.

    Args:
        sma (float): Semi-major axis [m].
        ecc (float): Eccentricity [-].
        inc (float): Inclination [rad].
        raan (float): Right Ascension of the Ascending Node [rad].
        aop (float): Argument of periapsis [rad].
        ta (float): True anomaly [rad].
        tof (float): Time of Flight for the propagation [s].

    Returns:
        tuple: A tuple containing:
            - mee_inputs (np.ndarray): Initial state in MEE (6 elements).
            - mee_residuals (np.ndarray): Residual errors in MEE (6 elements).
    """
    # 1. Convert initial COE to MEE (singularity-free AI input)
    mee_inputs = coe_to_mee(sma, ecc, inc, raan, aop, ta)

    # 2. Numerical Oracle execution (ground truth via Cowell's formulation)
    pos_num, vel_num = get_ground_truth(sma, ecc, inc, raan, aop, ta, tof)
    coe_num = eci_to_coe(MU, pos_num, vel_num)
    mee_num = coe_to_mee(*coe_num)

    # 3. Analytical Baseline execution (ESTHER + unperturbed Kepler)
    r0, v0 = coe_to_eci(MU, sma, ecc, inc, raan, aop, ta)
    x0_pert = np.zeros(6)
    n_mean = np.sqrt(MU / (sma**3))

    delta_pos_esther, delta_vel_esther = compute_general_solution(
        J2, J3, R_EQ, sma, ecc, inc, raan, aop, ta, x0_pert, n_mean, tof
    )
    pos_kep, vel_kep = get_keplerian(MU, r0, v0, tof)

    pos_esther = pos_kep + delta_pos_esther
    vel_esther = vel_kep + delta_vel_esther

    coe_esther = eci_to_coe(MU, pos_esther, vel_esther)
    mee_esther = coe_to_mee(*coe_esther)

    # 4. Compute residuals (Oracle - Analytical) with angular wrapping
    mee_residuals = mee_num - mee_esther
    mee_residuals[5] = wrap_to_pi(mee_residuals[5])  # Wrap True Longitude

    return mee_inputs, mee_residuals


def compute_analytical_state(sma, ecc, inc, raan, aop, ta, dt):
    """
    Propagates an orbital state using the Analytical Baseline only
    (ESTHER + Kepler) and returns the result in both ECI and MEE.

    Used by the benchmark and simulation modules where only the
    analytical prediction is needed (without Oracle comparison).

    Args:
        sma (float): Semi-major axis [m].
        ecc (float): Eccentricity [-].
        inc (float): Inclination [rad].
        raan (float): Right Ascension of the Ascending Node [rad].
        aop (float): Argument of periapsis [rad].
        ta (float): True anomaly [rad].
        dt (float): Time of Flight for the propagation [s].

    Returns:
        tuple: A tuple containing:
            - pos_esther (np.ndarray): Position vector in ECI [m].
            - vel_esther (np.ndarray): Velocity vector in ECI [m/s].
            - mee_esther (np.ndarray): Final state in MEE (6 elements).
    """
    r0, v0 = coe_to_eci(MU, sma, ecc, inc, raan, aop, ta)
    x0_pert = np.zeros(6)
    n_mean = np.sqrt(MU / (sma**3))

    delta_pos, delta_vel = compute_general_solution(
        J2, J3, R_EQ, sma, ecc, inc, raan, aop, ta, x0_pert, n_mean, dt
    )
    pos_kep, vel_kep = get_keplerian(MU, r0, v0, dt)

    pos_esther = pos_kep + delta_pos
    vel_esther = vel_kep + delta_vel

    coe_esther = eci_to_coe(MU, pos_esther, vel_esther)
    mee_esther = coe_to_mee(*coe_esther)

    return pos_esther, vel_esther, mee_esther
