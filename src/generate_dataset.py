"""
Script: generate_dataset.py

Description:
    Generates the dataset required to train the ORBITA Neural Network.
    It uses Monte Carlo uniform random sampling across user-defined orbital 
    parameter boundaries and Time of Flight (TOF). 
    
    The script propagates the initial states using both the Analytical Baseline 
    and the Numerical Oracle, computing the residual errors in terms of 
    Classical Orbital Elements (COE) to facilitate secular variation learning.
"""

import os
import csv
import numpy as np
from physics.analytical import computeGeneralSolution
from physics.oracle     import getGroundTruth, COE2ECI, ECI2COE
from physics.kepler     import getKeplerianNumerical

# =============================================================================
# GLOBAL ASTRODYNAMIC CONSTANTS (WGS84)
# =============================================================================
MU   = 3.986004418e14    # Gravitational parameter       [m^3/s^2]
J2   =   1.082635854e-3  # J2 zonal harmonic coefficient       [-]
J3   = - 2.532435346e-6  # J3 zonal harmonic coefficient       [-]
R_EQ = 6378.137e3        # Equatorial radius                   [m]

def wrap2pi(angle):
    """
    Wraps an angle or an array of angles to the interval [-pi, pi].
    Crucial for computing accurate angular errors without 360-degree jumps.
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi

def genTrainingData(
    num_samples, 
    output_file, 
    SMA_bounds, 
    ECC_bounds, 
    INC_bounds,
    RAAN_bounds,
    AOP_bounds,
    TA_bounds,
    TOF_bounds,
):
    """
    Samples the parameter space, runs both orbital models, and saves the 
    residual COE errors to a CSV file.

    Inputs:
        num_samples: Number of Monte Carlo samples to generate
        output_file: Path to the output CSV file
        SMA_bounds : (min, max) for Semi-Major Axis                     [m]
        ECC_bounds : (min, max) for Eccentricity                        [-]
        INC_bounds : (min, max) for Inclination                       [rad]
        RAAN_bounds: (min, max) for Right Ascension of Ascending Node [rad]
        AOP_bounds : (min, max) for Argument of Periapsis             [rad]
        TA_bounds  : (min, max) for True Anomaly                      [rad]
        TOF_bounds : (min, max) for Time of Flight                      [s]
    """
    print("=" * 60)
    print(f"🚀 INITIATING ORBITA DATASET GENERATION")
    print(f"   Target Samples : {num_samples}")
    print(f"   Output File    : {output_file}")
    print("=" * 60)
    
    # Ensure the target directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Open CSV file and write the header row
    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "SMA", "ECC", "INC", "RAAN", "AOP", "TA", "TOF",
            "err_SMA", "err_ECC", "err_INC", "err_RAAN", "err_AOP", "err_TA"
        ])
        
        valid_samples = 0
        
        # Use a while loop to keep searching until we have the desired number of valid samples
        while valid_samples < num_samples:
            # -----------------------------------------------------------------
            # 1. SAMPLE THE FEATURE SPACE
            # -----------------------------------------------------------------
            SMA  = np.random.uniform(SMA_bounds[0],  SMA_bounds[1])
            ECC  = np.random.uniform(ECC_bounds[0],  ECC_bounds[1])
            
            # Check if perigee is inside the Earth's atmosphere (150 km limit)
            r_perigee = SMA * (1.0 - ECC)
            if r_perigee < R_EQ + 150e3:
                continue
                
            INC  = np.random.uniform(INC_bounds[0],  INC_bounds[1])
            RAAN = np.random.uniform(RAAN_bounds[0], RAAN_bounds[1])
            AOP  = np.random.uniform(AOP_bounds[0],  AOP_bounds[1])
            TA   = np.random.uniform(TA_bounds[0],   TA_bounds[1])
            TOF  = np.random.uniform(TOF_bounds[0],  TOF_bounds[1])
            
            # -----------------------------------------------------------------
            # 2. NUMERICAL ORACLE EXECUTION (Ground Truth)
            # -----------------------------------------------------------------
            pos_num, vel_num = getGroundTruth(SMA, ECC, INC, RAAN, AOP, TA, TOF)
            
            # -----------------------------------------------------------------
            # 3. ANALYTICAL BASELINE EXECUTION
            # -----------------------------------------------------------------
            # Translate initial COE to ECI for Keplerian propagation
            r0, v0 = COE2ECI(MU, SMA, ECC, INC, RAAN, AOP, TA)
            
            # Compute Analytical Perturbations (J2 & J3)
            X0_pert = np.zeros(6)
            n = np.sqrt(MU / (SMA**3))
            deltaPos_ESTHER, deltaVel_ESTHER = computeGeneralSolution(
                J2, J3, R_EQ, SMA, ECC, INC, RAAN, AOP, TA, X0_pert, n, TOF
            )
            
            # Compute Pure Keplerian state and add perturbations
            pos_kep, vel_kep = getKeplerianNumerical(MU, r0, v0, TOF)
            pos_ESTHER = pos_kep + deltaPos_ESTHER
            vel_ESTHER = vel_kep + deltaVel_ESTHER 
            
            # -----------------------------------------------------------------
            # 4. STATE TRANSFORMATION (ECI -> COE)
            # -----------------------------------------------------------------
            coe_num = ECI2COE(MU, pos_num, vel_num)
            coe_ESTHER = ECI2COE(MU, pos_ESTHER, vel_ESTHER) 
            
            # -----------------------------------------------------------------
            # 5. RESIDUAL ERROR CALCULATION (Y = Oracle - Analytical)
            # -----------------------------------------------------------------
            err_SMA = coe_num[0] - coe_ESTHER[0]
            err_ECC = coe_num[1] - coe_ESTHER[1]
            
            # Angular differences must be wrapped to [-pi, pi] to avoid artificial spikes
            err_INC  = wrap2pi(coe_num[2] - coe_ESTHER[2])
            err_RAAN = wrap2pi(coe_num[3] - coe_ESTHER[3])
            err_AOP  = wrap2pi(coe_num[4] - coe_ESTHER[4])
            err_TA   = wrap2pi(coe_num[5] - coe_ESTHER[5])
            
            # -----------------------------------------------------------------
            # 6. DATA RECORDING
            # -----------------------------------------------------------------
            writer.writerow([
                SMA, ECC, INC, RAAN, AOP, TA, TOF,
                err_SMA, err_ECC, err_INC, err_RAAN, err_AOP, err_TA
            ])
            
            valid_samples += 1
            
            # Console progress tracker
            if valid_samples % 1000 == 0:
                print(f" [INFO] Processed {valid_samples}/{num_samples} samples...")

    print(f"\n ✅ Dataset successfully generated and saved to: {output_file}\n")


# =============================================================================
# EXECUTION BLOCK
# =============================================================================
if __name__ == "__main__":
    
    # 1. Define the Mission-Specific Bounds (The "Expert" domain)
    mission_SMA_bounds  = (R_EQ + 300e3, R_EQ + 2000e3)
    mission_ECC_bounds  = (0, 0.1)
    mission_INC_bounds  = (np.deg2rad(0), np.deg2rad(90))
    
    mission_RAAN_bounds = (0.0, 2 * np.pi)
    mission_AOP_bounds  = (0.0, 2 * np.pi)
    mission_TA_bounds   = (0.0, 2 * np.pi)
    mission_TOF_bounds  = (0.0, 4.0 * 3600.0) # Up to 4 hours of flight time
    
    # 2. Dynamic Filename Generation
    # We convert SMA to km and INC to degrees for a cleaner, human-readable string
    SMA_str = f"{int( (mission_SMA_bounds[0] - R_EQ)/1e3 )}-{int( (mission_SMA_bounds[1] - R_EQ)/1e3 )}"
    ECC_str = f"{mission_ECC_bounds[0]:.3f}-{mission_ECC_bounds[1]:.3f}"
    INC_str = f"{int(np.rad2deg(mission_INC_bounds[0]))}-{int(np.rad2deg(mission_INC_bounds[1]))}"
    
    filename = f"data/orbita_dataset_{SMA_str}_{ECC_str}_{INC_str}.csv"
    
    # 3. Execute the generation
    genTrainingData(
        num_samples = 10000,
        output_file = filename,
        SMA_bounds  = mission_SMA_bounds,
        ECC_bounds  = mission_ECC_bounds,
        INC_bounds  = mission_INC_bounds,
        RAAN_bounds = mission_RAAN_bounds,
        AOP_bounds  = mission_AOP_bounds,
        TA_bounds   = mission_TA_bounds,
        TOF_bounds  = mission_TOF_bounds
    )