"""
Script: compare.py

Description:
    Puts the Analytical baseline (ESTHER) and the Numerical Oracle to the test.
    Simulates a given Time of Flight (TOF) and computes the residual error
    (Euclidean distance) between both models.
"""

import numpy as np
from physics.analytical import computeGeneralSolution
from physics.kepler     import getKeplerianNumerical
from physics.oracle     import getGroundTruth, COE2ECI

def compareModels():
    """
    Executes both the numerical and analytical propagation models for a given 
    initial state and time of flight, comparing their final states.
    """
    print("--- STARTING ORBITA MODEL COMPARISON ---")

    # Earth physical constants
    MU   = 3.986004418e14   # Gravitational parameter       [m^3/s^2]
    J2   =   1.082635854e-3 # J2 zonal harmonic coefficient       [-]
    J3   = - 2.532435346e-6 # J3 zonal harmonic coefficient       [-]
    R_EQ = 6378.137e3       # Equatorial radius                   [m]

    # 1. Shared Initial Conditions
    SMA  = 7778.137e3        # Semi-Major Axis                         [m]
    ECC  = 0.001             # Eccentricity                            [-]
    INC  = np.deg2rad(90.0)  # Inclination                           [rad]
    RAAN = np.deg2rad(0.0)   # Right Ascension of the Ascending Node [rad]
    AOP  = np.deg2rad(0.0)   # Argument of Periapsis                 [rad]
    TA   = np.deg2rad(0.0)   # True Anomaly                          [rad]
    TOF  = 3600.0            # Time of Flight                          [s]

    # Mean motion
    n = np.sqrt(MU / (SMA**3))
    
    # 2. Execute Numerical Oracle (Ground Truth)
    print("\n[1/2] Computing Ground Truth (Numerical Oracle)...")
    pos_num, vel_num = getGroundTruth(
        SMA  = SMA, 
        ECC  = ECC, 
        INC  = INC, 
        RAAN = RAAN, 
        AOP  = AOP, 
        TA   = TA, 
        TOF  = TOF
    )
    
    # 3. Execute Analytical Baseline (TFG Model)
    print("[2/2] Computing Analytical Baseline (TFG Model)...")
    r0, v0 = COE2ECI(MU, SMA, ECC, INC, RAAN, AOP, TA)
    
    # This returns the variation of position and velocity, not the absolute state
    deltaPos_ESTHER, deltaVel_ESHTER = computeGeneralSolution(
        J2, J3, R_EQ, SMA, ECC, INC, 
        RAAN, AOP, TA, np.zeros(6), n, TOF
    )
    
    # Compute the pure unperturbed Keplerian orbit at the final instant
    pos_kep, vel_kep = getKeplerianNumerical(MU, r0, v0, TOF)
    
    # The final analytical state is the sum of the Keplerian baseline + perturbations
    pos_ESTHER = pos_kep + deltaPos_ESTHER
    vel_ESTHER = vel_kep + deltaVel_ESHTER
    
    # 4. Compute Residual Error (Euclidean Distance)
    error_pos = np.linalg.norm(pos_num - pos_ESTHER)
    error_vel = np.linalg.norm(vel_num - vel_ESTHER)
    
    print("\n--- RESULTS ---")
    print(f"Oracle [X, Y, Z] (m):     [{pos_num[0]:.2f}, {pos_num[1]:.2f}, {pos_num[2]:.2f}]")
    print(f"Analytical [X, Y, Z] (m): [{pos_ESTHER[0]:.2f}, {pos_ESTHER[1]:.2f}, {pos_ESTHER[2]:.2f}]")
    print("-" * 50)
    print(f"❌ POSITION RESIDUAL ERROR: {error_pos:.2f} m")
    print(f"❌ VELOCITY RESIDUAL ERROR: {error_vel:.4f} m/s")
    print("-" * 50)
    print("This is the error that the AI must learn to correct!")


# =============================================================================
# Execution Block
# =============================================================================
if __name__ == "__main__":
    compareModels()