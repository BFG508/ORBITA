"""
Module: oracle.py

Description:
    This module serves as the 'Numerical Oracle' for the ORBITA system. 
    It generates high-precision ground truth data using numerical 
    integration (Cowell's method).
    
    Built completely from scratch using SciPy's LSODA integrator to 
    ensure maximum stability, full transparency in the physics (grey-box 
    modeling), and zero library-version dependencies.
"""

import numpy as np
from scipy.integrate import solve_ivp

# =============================================================================
# GLOBAL ASTRODYNAMIC CONSTANTS (WGS84)
# =============================================================================
MU = 3.986004418e14      # Gravitational parameter [m^3/s^2]
J2 = 1.082635854e-3      # J2 zonal harmonic coefficient [-]
J3 = -2.532435346e-6     # J3 zonal harmonic coefficient [-]
R_EQ = 6378.137e3        # Equatorial radius [m]

# Pre-compute perturbation constants to save computation time inside ODE solvers
C_J2_PRE = -3.0 / 2.0 * J2 * MU * (R_EQ**2)
C_J3_PRE = -1.0 / 2.0 * J3 * MU * (R_EQ**3)


def eci_to_coe(mu, r_eci, v_eci):
    """
    Converts Cartesian ECI (Earth-Centered Inertial) vectors to 
    Classical Orbital Elements (COE).
    
    Args:
        mu (float): Gravitational parameter of the central body [m^3/s^2].
        r_eci (np.ndarray): Position vector in ECI frame [m].
        v_eci (np.ndarray): Velocity vector in ECI frame [m/s].
        
    Returns:
        np.ndarray: Array containing the 6 Classical Orbital Elements:
            - sma (float): Semi-major axis [m].
            - ecc (float): Eccentricity [-].
            - inc (float): Inclination [rad].
            - raan (float): Right Ascension of the Ascending Node [rad].
            - aop (float): Argument of periapsis [rad].
            - ta (float): True anomaly [rad].
    """
    eps = 1e-10  # Tolerance for singularities
    
    r = np.linalg.norm(r_eci)
    v = np.linalg.norm(v_eci)
    
    # 1. Angular Momentum
    h_vec = np.cross(r_eci, v_eci)
    h = np.linalg.norm(h_vec)
    
    # 2. Semi-Major Axis (SMA)
    energy = (v**2 / 2.0) - (mu / r)
    sma = -mu / (2.0 * energy)
    
    # 3. Inclination (INC)
    inc = np.arccos(np.clip(h_vec[2] / h, -1.0, 1.0))
    
    # 4. Node Vector and RAAN
    n_vec = np.array([-h_vec[1], h_vec[0], 0.0])
    n = np.linalg.norm(n_vec)
    
    if n < eps:
        raan = 0.0  # Equatorial orbit singularity
    else:
        raan = np.arccos(np.clip(n_vec[0] / n, -1.0, 1.0))
        if n_vec[1] < 0:
            raan = 2.0 * np.pi - raan
            
    # 5. Eccentricity Vector and ECC
    e_vec = (1.0 / mu) * ((v**2 - mu / r) * r_eci - np.dot(r_eci, v_eci) * v_eci)
    ecc = np.linalg.norm(e_vec)
    
    # 6. Argument of Periapsis (AOP)
    if n < eps or ecc < eps:
        aop = 0.0  # Circular or equatorial singularity
    else:
        aop = np.arccos(np.clip(np.dot(n_vec, e_vec) / (n * ecc), -1.0, 1.0))
        if e_vec[2] < 0:
            aop = 2.0 * np.pi - aop
            
    # 7. True Anomaly (TA)
    if ecc < eps:
        ta = 0.0  # Circular orbit singularity
    else:
        ta = np.arccos(np.clip(np.dot(e_vec, r_eci) / (ecc * r), -1.0, 1.0))
        if np.dot(r_eci, v_eci) < 0:
            ta = 2.0 * np.pi - ta
            
    return np.array([sma, ecc, inc, raan, aop, ta])


def coe_to_eci(mu, sma, ecc, inc, raan, aop, ta):
    """
    Converts Classical Orbital Elements (COE) to 
    Cartesian ECI (Earth-Centered Inertial) vectors.
    
    Args:
        mu (float): Gravitational parameter of the central body [m^3/s^2].
        sma (float): Semi-major axis [m].
        ecc (float): Eccentricity [-].
        inc (float): Inclination [rad].
        raan (float): Right Ascension of the Ascending Node [rad].
        aop (float): Argument of periapsis [rad].
        ta (float): True anomaly [rad].
    
    Returns:
        tuple: A tuple containing:
            - r_eci (np.ndarray): Position vector in ECI frame [m].
            - v_eci (np.ndarray): Velocity vector in ECI frame [m/s].
    """
    # Semi-latus rectum
    p = sma * (1.0 - ecc**2)
    
    # Position and velocity in the perifocal coordinate system (PQW)
    denom = 1.0 + ecc * np.cos(ta)
    r_pqw = np.array([p * np.cos(ta) / denom, p * np.sin(ta) / denom, 0.0])
    v_pqw = np.array([-np.sqrt(mu / p) * np.sin(ta), np.sqrt(mu / p) * (ecc + np.cos(ta)), 0.0])
    
    # Trigonometric functions for the rotation matrix
    c_o, s_o = np.cos(raan), np.sin(raan)
    c_w, s_w = np.cos(aop), np.sin(aop)
    c_i, s_i = np.cos(inc), np.sin(inc)
    
    # Transformation matrix from PQW to ECI
    rotation_matrix = np.array([
        [c_o*c_w - s_o*s_w*c_i, -c_o*s_w - s_o*c_w*c_i,  s_o*s_i],
        [s_o*c_w + c_o*s_w*c_i, -s_o*s_w + c_o*c_w*c_i, -c_o*s_i],
        [              s_w*s_i,                c_w*s_i,      c_i]
    ])
    
    return rotation_matrix @ r_pqw, rotation_matrix @ v_pqw


def coe_to_mee(sma, ecc, inc, raan, aop, ta):
    """
    Converts Classical Orbital Elements (COE) to 
    Modified Equinoctial Elements (MEE).
    
    MEE mathematically eliminate the singularities associated with 
    circular (ecc = 0) and equatorial (inc = 0) orbits.
    
    Args:
        sma (float): Semi-major axis [m].
        ecc (float): Eccentricity [-].
        inc (float): Inclination [rad].
        raan (float): Right Ascension of the Ascending Node [rad].
        aop (float): Argument of periapsis [rad].
        ta (float): True anomaly [rad].
        
    Returns:
        np.ndarray: Array containing the 6 Modified Equinoctial Elements:
            - p (float): Semi-latus rectum [m].
            - f (float): Equinoctial eccentricity component x [-].
            - g (float): Equinoctial eccentricity component y [-].
            - h (float): Equinoctial inclination component x [-].
            - k (float): Equinoctial inclination component y [-].
            - l (float): True longitude [rad].
    """
    # 1. Semi-latus Rectum
    p = sma * (1.0 - ecc**2)
    
    # 2. Longitude of periapsis
    lon_per = raan + aop
    
    # 3. Equinoctial eccentricity vector components
    f = ecc * np.cos(lon_per)
    g = ecc * np.sin(lon_per)
    
    # 4. Equinoctial inclination vector components
    h = np.tan(inc / 2.0) * np.cos(raan)
    k = np.tan(inc / 2.0) * np.sin(raan)
    
    # 5. True Longitude
    l_true = raan + aop + ta
    l_true = l_true % (2.0 * np.pi)  # Wrap to [0, 2*pi]
    
    return np.array([p, f, g, h, k, l_true])


def mee_to_coe(p, f, g, h, k, l_true):
    """
    Converts Modified Equinoctial Elements (MEE) back to 
    Classical Orbital Elements (COE).
    
    Args:
        p (float): Semi-latus rectum [m].
        f (float): Equinoctial eccentricity component x [-].
        g (float): Equinoctial eccentricity component y [-].
        h (float): Equinoctial inclination component x [-].
        k (float): Equinoctial inclination component y [-].
        l_true (float): True longitude [rad].
        
    Returns:
        np.ndarray: Array containing the 6 Classical Orbital Elements:
            - sma (float): Semi-major axis [m].
            - ecc (float): Eccentricity [-].
            - inc (float): Inclination [rad].
            - raan (float): Right Ascension of the Ascending Node [rad].
            - aop (float): Argument of periapsis [rad].
            - ta (float): True anomaly [rad].
    """
    # 1. Eccentricity and Semi-Major Axis
    ecc = np.sqrt(f**2 + g**2)
    
    # Safety check: avoid division by zero if orbit is perfectly parabolic (e = 1)
    sma = p / (1.0 - ecc**2) 
    
    # 2. Inclination
    inc = 2.0 * np.arctan(np.sqrt(h**2 + k**2))
    
    # 3. Right Ascension of the Ascending Node (RAAN)
    raan = np.arctan2(k, h)
    raan = raan % (2.0 * np.pi)
    
    # 4. Argument of Periapsis (AOP)
    lon_per = np.arctan2(g, f)
    aop = (lon_per - raan) % (2.0 * np.pi)
    
    # 5. True Anomaly (TA)
    ta = (l_true - lon_per) % (2.0 * np.pi)
    
    return np.array([sma, ecc, inc, raan, aop, ta])


def get_ground_truth(sma, ecc, inc, raan, aop, ta, tof):
    """
    Computes a high-precision orbital state by numerically integrating 
    the equations of motion (Newtonian Gravity + J2 & J3 perturbations).
    
    Args:
        sma (float): Semi-major axis [m].
        ecc (float): Eccentricity [-].
        inc (float): Inclination [rad].
        raan (float): Right Ascension of the Ascending Node [rad].
        aop (float): Argument of periapsis [rad].
        ta (float): True anomaly [rad].
        tof (float): Time of Flight for the integration [s].
        
    Returns:
        tuple: A tuple containing:
            - r_final (np.ndarray): Final position vector in ECI frame [m].
            - v_final (np.ndarray): Final velocity vector in ECI frame [m/s].
    """
    
    # 1. Initial State Definition
    r0, v0 = coe_to_eci(MU, sma, ecc, inc, raan, aop, ta)
    y0 = np.concatenate((r0, v0))
    
    # 2. Equations of Motion (State Derivatives)
    def orbit_dynamics(t, y):
        r_vec = y[0:3]
        v_vec = y[3:6]
        
        x_pos, y_pos, z_pos = r_vec
        r = np.linalg.norm(r_vec)
        
        # Pure Newtonian central gravity acceleration
        a_central = -MU / (r**3) * r_vec
        
        # Zonal harmonic perturbation accelerations (J2 & J3)
        z2_r2 = (z_pos / r)**2
        
        # J2 Acceleration
        c_j2_dyn = C_J2_PRE / (r**5)
        ax_j2 = c_j2_dyn * x_pos * (1.0 - 5.0 * z2_r2)
        ay_j2 = c_j2_dyn * y_pos * (1.0 - 5.0 * z2_r2)
        az_j2 = c_j2_dyn * z_pos * (3.0 - 5.0 * z2_r2)
        a_j2 = np.array([ax_j2, ay_j2, az_j2])
        
        # J3 Acceleration
        c_j3_dyn = C_J3_PRE / (r**7)
        ax_j3 = c_j3_dyn * x_pos * (15.0 * z_pos - 35.0 * z_pos * z2_r2)
        ay_j3 = c_j3_dyn * y_pos * (15.0 * z_pos - 35.0 * z_pos * z2_r2)
        az_j3 = c_j3_dyn         * (30.0 * z_pos**2 - 35.0 * z_pos**2 * z2_r2 - 3.0 * r**2)
        a_j3 = np.array([ax_j3, ay_j3, az_j3])
        
        # Total acceleration
        a_total = a_central + a_j2 + a_j3
        
        return np.concatenate((v_vec, a_total))

    # 3. Numerical Integration (Cowell's formulation)
    res = solve_ivp(
        fun=orbit_dynamics, 
        t_span=[0, tof], 
        y0=y0, 
        method='LSODA',
        rtol=1e-12, 
        atol=1e-5
    )
    
    # 4. Extract the state at the final integration step
    r_final = res.y[0:3, -1]
    v_final = res.y[3:6, -1]
    
    return r_final, v_final