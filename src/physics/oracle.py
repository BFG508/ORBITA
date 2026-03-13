"""
Module: oracle.py

Description:
    This module serves as the 'Numerical Oracle'. It generates high-precision 
    ground truth data using numerical integration (Cowell's method).
    
    NOTE: Built completely from scratch using SciPy's LSODA integrator to 
    ensure maximum stability, full transparency in the physics (Grey-Box), 
    and zero library-version dependencies.
"""

import numpy as np
from scipy.integrate import solve_ivp

def ECI2COE(mu, rECI, vECI):
    """
    Converts Cartesian ECI (Earth-Centered Inertial) vectors to 
    COE (Classical Orbit Elements).
    
    Inputs:
        mu  : Gravitational parameter of the central body [m^3/s^2]
        rECI: Position vector in ECI frame                      [m]
        vECI: Velocity vector in ECI frame                    [m/s]
        
    Outputs:
        SMA : Semi-Major Axis                                   [m]
        ECC : Eccentricity                                      [-]
        INC : Inclination                                     [rad]
        RAAN: Right Ascension of the Ascending Node           [rad]
        AOP : Argument of Periapsis                           [rad]
        TA  : True Anomaly                                    [rad]
    """
    eps = 1e-10  # Tolerance for singularities
    
    r = np.linalg.norm(rECI)
    v = np.linalg.norm(vECI)
    
    # 1. Angular Momentum
    h_vec = np.cross(rECI, vECI)
    h = np.linalg.norm(h_vec)
    
    # 2. Semi-Major Axis (SMA)
    energy = (v**2 / 2.0) - (mu / r)
    SMA = -mu / (2.0 * energy)
    
    # 3. Inclination (INC)
    INC = np.arccos(np.clip(h_vec[2] / h, -1.0, 1.0))
    
    # 4. Node Vector and RAAN
    n_vec = np.array([-h_vec[1], h_vec[0], 0.0])
    n = np.linalg.norm(n_vec)
    
    if n < eps:
        RAAN = 0.0 # Equatorial orbit singularity
    else:
        RAAN = np.arccos(np.clip(n_vec[0] / n, -1.0, 1.0))
        if n_vec[1] < 0:
            RAAN = 2.0 * np.pi - RAAN
            
    # 5. Eccentricity Vector and ECC
    e_vec = (1.0 / mu) * ((v**2 - mu / r) * rECI - np.dot(rECI, vECI) * vECI)
    ECC = np.linalg.norm(e_vec)
    
    # 6. Argument of Periapsis (AOP)
    if n < eps or ECC < eps:
        AOP = 0.0 # Circular or equatorial singularity
    else:
        AOP = np.arccos(np.clip(np.dot(n_vec, e_vec) / (n * ECC), -1.0, 1.0))
        if e_vec[2] < 0:
            AOP = 2.0 * np.pi - AOP
            
    # 7. True Anomaly (TA)
    if ECC < eps:
        TA = 0.0 # Circular orbit singularity
    else:
        TA = np.arccos(np.clip(np.dot(e_vec, rECI) / (ECC * r), -1.0, 1.0))
        if np.dot(rECI, vECI) < 0:
            TA = 2.0 * np.pi - TA
            
    return np.array([SMA, ECC, INC, RAAN, AOP, TA])


def COE2ECI(mu, SMA, ECC, INC, RAAN, AOP, TA):
    """
    Converts COE (Classical Orbit Elements) to 
    Cartesian ECI (Earth-Centered Inertial) vectors.
    
    Inputs:
        mu  : Gravitational parameter of the central body [m^3/s^2]
        SMA : Semi-Major Axis                                   [m]
        ECC : Eccentricity                                      [-]
        INC : Inclination                                     [rad]
        RAAN: Right Ascension of the Ascending Node           [rad]
        AOP : Argument of Periapsis                           [rad]
        TA  : True Anomaly                                    [rad]
    
    Outputs:
        rECI: Position vector in ECI frame                      [m]
        vECI: Velocity vector in ECI frame                    [m/s]
    """
    # Semi-latus rectum
    p = SMA * (1.0 - ECC**2)
    
    # Position and velocity in the perifocal coordinate system (PQW)
    denom = 1.0 + ECC * np.cos(TA)
    rPQW = np.array([p * np.cos(TA) / denom, p * np.sin(TA) / denom, 0.0])
    vPQW = np.array([-np.sqrt(mu / p) * np.sin(TA), np.sqrt(mu / p) * (ECC + np.cos(TA)), 0.0])
    
    # Trigonometric functions for the rotation matrix
    c_O, s_O = np.cos(RAAN), np.sin(RAAN)
    c_o, s_o = np.cos(AOP), np.sin(AOP)
    c_i, s_i = np.cos(INC), np.sin(INC)
    
    # Transformation matrix from PQW to ECI
    R = np.array([
        [c_O*c_o - s_O*s_o*c_i, -c_O*s_o - s_O*c_o*c_i,  s_O*s_i],
        [s_O*c_o + c_O*s_o*c_i, -s_O*s_o + c_O*c_o*c_i, -c_O*s_i],
        [              s_o*s_i,                c_o*s_i,      c_i]
    ])
    
    return R @ rPQW, R @ vPQW


def getGroundTruth(SMA, ECC, INC, RAAN, AOP, TA, TOF):
    """
    Computes a high-precision orbital state by numerically integrating 
    the equations of motion (Newtonian Gravity + J2 & J3 perturbations).
    
    Inputs:
        SMA : Semi-Major Axis                                   [m]
        ECC : Eccentricity                                      [-]
        INC : Inclination                                     [rad]
        RAAN: Right Ascension of the Ascending Node           [rad]
        AOP : Argument of Periapsis                           [rad]
        TA  : True Anomaly                                    [rad]
        TOF : Time of Flight for the integration                [s]
        
    Outputs:
        r_final: Final position vector in ECI frame             [m]
        v_final: Final velocity vector in ECI frame           [m/s]
    """
    
    # Earth physical constants
    MU   = 3.986004418e14    # Gravitational parameter       [m^3/s^2]
    J2   =   1.082635854e-3  # J2 zonal harmonic coefficient       [-]
    J3   = - 2.532435346e-6  # J3 zonal harmonic coefficient       [-]
    R_EQ = 6378.137e3        # Equatorial radius                   [m]
    
    # Pre-compute perturbation constants to save computation time inside the ODE solver
    C_J2_PRE = -3/2 * J2 * MU * (R_EQ**2)
    C_J3_PRE = -1/2 * J3 * MU * (R_EQ**3)
    
    # 1. Initial State Definition
    r0, v0 = COE2ECI(MU, SMA, ECC, INC, RAAN, AOP, TA)
    y0 = np.concatenate((r0, v0))
    
    # 2. Equations of Motion (State Derivatives)
    def orbit_dynamics(t, y):
        # Unpack position and velocity
        r_vec = y[0:3]
        v_vec = y[3:6]
        
        x_pos, y_pos, z_pos = r_vec
        r = np.linalg.norm(r_vec)
        
        # Pure Newtonian central gravity acceleration
        a_central = -MU / (r**3) * r_vec
        
        # Zonal harmonic perturbation accelerations (J2 & J3)
        z2_r2 = (z_pos / r)**2
        
        # J2 Acceleration
        C_J2_dyn = C_J2_PRE / (r**5)
        ax_J2 = C_J2_dyn * x_pos * (1.0 - 5.0 * z2_r2)
        ay_J2 = C_J2_dyn * y_pos * (1.0 - 5.0 * z2_r2)
        az_J2 = C_J2_dyn * z_pos * (3.0 - 5.0 * z2_r2)
        a_J2 = np.array([ax_J2, ay_J2, az_J2])
        
        # J3 Acceleration
        C_J3_dyn = C_J3_PRE / (r**7)
        ax_J3 = C_J3_dyn * x_pos * (15.0 * z_pos - 35.0 * z_pos * z2_r2)
        ay_J3 = C_J3_dyn * y_pos * (15.0 * z_pos - 35.0 * z_pos * z2_r2)
        az_J3 = C_J3_dyn         * (30.0 * z_pos**2 - 35.0 * z_pos**2 * z2_r2 - 3.0 * r**2)
        a_J3 = np.array([ax_J3, ay_J3, az_J3])
        
        # Total acceleration
        a_total = a_central + a_J2 + a_J3
        
        # Return state derivative: [velocity, acceleration]
        return np.concatenate((v_vec, a_total))

    # 3. Numerical Integration (Cowell's formulation)
    res = solve_ivp(
        fun    = orbit_dynamics, 
        t_span = [0, TOF], 
        y0     = y0, 
        method = 'LSODA', # An Adams/BDF method
        rtol   = 1e-12,   # Strict relative tolerance
        atol   = 1e-5     # Strict absolute tolerance
    )
    
    # 4. Extract the state at the final integration step
    r_final = res.y[0:3, -1]
    v_final = res.y[3:6, -1]
    
    return r_final, v_final