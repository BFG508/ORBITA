"""
Script: generate_base_dataset.py

Description:
    Generates the dataset required to train the ORBITA neural network.
    It uses Monte Carlo uniform random sampling across user-defined orbital 
    parameter boundaries and Time of Flight (tof). 
    
    The script propagates the initial states using both the analytical baseline 
    (ESTHER) and the numerical oracle. Crucially, it saves both the inputs 
    and the residual errors in Modified Equinoctial Elements (MEE) to grant 
    the AI absolute mathematical immunity against classical singularities.
"""

import os
import csv
import numpy as np

from physics.analytical import compute_general_solution
from physics.oracle import get_ground_truth, coe_to_eci, eci_to_coe, coe_to_mee
from physics.kepler import get_keplerian

from physics.oracle import MU, J2, J3, R_EQ

# =============================================================================
# CONSTANTS
# =============================================================================
# A satellite in LEO requires a minimum altitude to prevent immediate 
# atmospheric decay or mathematical singularities in the gravity model.
MIN_SAFE_PERIGEE = R_EQ + 200e3  # 200 km minimum altitude


def wrap_to_pi(angle):
    """
    Wraps an angle or an array of angles to the interval [-pi, pi].
    Crucial for computing accurate angular errors without 360-degree jumps.
    
    Args:
        angle (float or np.ndarray): Angle in radians.
        
    Returns:
        float or np.ndarray: Wrapped angle in radians.
    """
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def generate_training_data(
    num_samples, 
    output_file, 
    sma_bounds, 
    ecc_bounds, 
    inc_bounds,
    raan_bounds,
    aop_bounds,
    ta_bounds,
    tof_bounds
):
    """
    Samples the parameter space, runs both orbital models, and saves the 
    input states and residual errors (both in MEE) to a CSV file.

    Args:
        num_samples (float): Number of Monte Carlo samples to generate.
        output_file (str): Path to the output CSV file.
        sma_bounds (tuple): (min, max) for semi-major axis [m].
        ecc_bounds (tuple): (min, max) for eccentricity [-].
        inc_bounds (tuple): (min, max) for inclination [rad].
        raan_bounds (tuple): (min, max) for right ascension of ascending node [rad].
        aop_bounds (tuple): (min, max) for argument of periapsis [rad].
        ta_bounds (tuple): (min, max) for true anomaly [rad].
        tof_bounds (tuple): (min, max) for time of flight [s].
    """
    print("-" * 70)
    print(" INITIATING ORBITA DATASET GENERATION")
    print(f" Target samples : {int(num_samples)}")
    print(f" Output file    : {output_file}")
    print("-" * 70)
    
    # Ensure the target directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Open CSV file and write the header row
    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow([
            "p", "f", "g", "h", "k", "L", "TOF",
            "err_p", "err_f", "err_g", "err_h", "err_k", "err_L"
        ])
        
        valid_samples = 0
        
        # Use a while loop to filter out physically invalid orbits (underground perigees)
        while valid_samples < num_samples:
            # -----------------------------------------------------------------
            # 1. SAMPLE THE FEATURE SPACE (In COE for human intuition)
            # -----------------------------------------------------------------
            sma = np.random.uniform(sma_bounds[0], sma_bounds[1])
            ecc = np.random.uniform(ecc_bounds[0], ecc_bounds[1])
            
            # Check if perigee is inside the Earth's atmosphere
            r_periapsis = sma * (1.0 - ecc)
            if r_periapsis < MIN_SAFE_PERIGEE:
                continue
                
            inc = np.random.uniform(inc_bounds[0], inc_bounds[1])
            raan = np.random.uniform(raan_bounds[0], raan_bounds[1])
            aop = np.random.uniform(aop_bounds[0], aop_bounds[1])
            ta = np.random.uniform(ta_bounds[0], ta_bounds[1])
            tof = np.random.uniform(tof_bounds[0], tof_bounds[1])
            
            # -----------------------------------------------------------------
            # 2. INPUT CONVERSION (COE -> MEE for the Neural Network)
            # -----------------------------------------------------------------
            mee_inputs = coe_to_mee(sma, ecc, inc, raan, aop, ta)
            p_in, f_in, g_in, h_in, k_in, l_in = mee_inputs
            
            # -----------------------------------------------------------------
            # 3. NUMERICAL ORACLE EXECUTION (Ground Truth)
            # -----------------------------------------------------------------
            pos_num, vel_num = get_ground_truth(sma, ecc, inc, raan, aop, ta, tof)
            
            # -----------------------------------------------------------------
            # 4. ANALYTICAL BASELINE EXECUTION
            # -----------------------------------------------------------------
            r0, v0 = coe_to_eci(MU, sma, ecc, inc, raan, aop, ta)
            
            x0_pert = np.zeros(6)
            n_mean = np.sqrt(MU / (sma**3))
            
            # compute_general_solution retains original signature from analytical.py
            delta_pos_esther, delta_vel_esther = compute_general_solution(
                J2, J3, R_EQ, sma, ecc, inc, raan, aop, ta, x0_pert, n_mean, tof
            )
            
            pos_kep, vel_kep = get_keplerian(MU, r0, v0, tof)
            
            pos_esther = pos_kep + delta_pos_esther
            vel_esther = vel_kep + delta_vel_esther
            
            # -----------------------------------------------------------------
            # 5. STATE TRANSFORMATION & ERROR CALCULATION
            # -----------------------------------------------------------------
            coe_num = eci_to_coe(MU, pos_num, vel_num)
            coe_esther = eci_to_coe(MU, pos_esther, vel_esther) 
            
            mee_num = coe_to_mee(*coe_num)
            mee_esther = coe_to_mee(*coe_esther)

            # Residual Error (Y = Oracle - Analytical)
            err_p = mee_num[0] - mee_esther[0]
            err_f = mee_num[1] - mee_esther[1]
            err_g = mee_num[2] - mee_esther[2]
            err_h = mee_num[3] - mee_esther[3]
            err_k = mee_num[4] - mee_esther[4]
            err_l = wrap_to_pi(mee_num[5] - mee_esther[5])
            
            # -----------------------------------------------------------------
            # 6. DATA RECORDING
            # -----------------------------------------------------------------
            writer.writerow([
                p_in, f_in, g_in, h_in, k_in, l_in, tof,
                err_p, err_f, err_g, err_h, err_k, err_l
            ])

            valid_samples += 1
            
            # Console progress tracker
            if valid_samples % 1000 == 0:
                print(f" [info] Processed {valid_samples}/{int(num_samples)} samples...")

    print(f"\n Dataset successfully generated and saved to: {output_file}\n")


# =============================================================================
# EXECUTION BLOCK
# =============================================================================
if __name__ == "__main__":
    # 1. Define the Mission-Specific Bounds (The "Expert" domain)
    mission_sma_bounds = (R_EQ + 300e3, R_EQ + 2000e3)
    mission_ecc_bounds = (0, 0.1)
    mission_inc_bounds = (np.deg2rad(0), np.deg2rad(90))
    
    mission_raan_bounds = (0.0, 2.0 * np.pi)
    mission_aop_bounds = (0.0, 2.0 * np.pi)
    mission_ta_bounds = (0.0, 2.0 * np.pi)
    mission_tof_bounds = (0.0, 30.0 * 60.0) # Up to 30 minutes of flight time
    
    # 2. Dynamic Filename Generation
    sma_str = f"{int((mission_sma_bounds[0] - R_EQ) / 1e3)}-{int((mission_sma_bounds[1] - R_EQ) / 1e3)}"
    ecc_str = f"{mission_ecc_bounds[0]:.4f}-{mission_ecc_bounds[1]:.4f}"
    inc_str = f"{int(np.rad2deg(mission_inc_bounds[0]))}-{int(np.rad2deg(mission_inc_bounds[1]))}"
    
    filename = f"data/orbita_dataset_{sma_str}_{ecc_str}_{inc_str}.csv"
    
    # 3. Execute the generation
    generate_training_data(
        num_samples=100000,
        output_file=filename,
        sma_bounds=mission_sma_bounds,
        ecc_bounds=mission_ecc_bounds,
        inc_bounds=mission_inc_bounds,
        raan_bounds=mission_raan_bounds,
        aop_bounds=mission_aop_bounds,
        ta_bounds=mission_ta_bounds,
        tof_bounds=mission_tof_bounds
    )