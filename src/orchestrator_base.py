"""
Script: orchestrator_base.py

Description:
    Automates the creation of a 3D Mixture of Experts (MoE) architecture.
    It takes a continuous orbital domain and subdivides it into a 3D grid 
    based on Semi-Major Axis (SMA), Eccentricity (ECC), and Inclination (INC).
    
    This hyper-specialization completely isolates scale effects (altitude), 
    denominator singularities (eccentricity), and trigonometric non-linearities 
    (inclination). For each 3D cell, it generates the dataset and trains a 
    highly specialized neural network.
"""

import numpy as np

from generate_base_dataset import generate_training_data
from train_base import train_model

from physics.oracle import R_EQ


def build_expert_grid(
    total_sma_bounds, 
    total_ecc_bounds, 
    total_inc_bounds,
    sma_splits, 
    ecc_splits,
    inc_splits, 
    samples_per_expert=5000
):
    """
    Executes the automated pipeline to generate and train a 3D grid of expert models.
    
    Args:
        total_sma_bounds (tuple): (min, max) for the entire Semi-Major Axis space [m].
        total_ecc_bounds (tuple): (min, max) for the entire Eccentricity space [-].
        total_inc_bounds (tuple): (min, max) for the entire Inclination space [rad].
        sma_splits (int): Number of subdivisions for the altitude domain.
        ecc_splits (int): Number of subdivisions for the eccentricity domain.
        inc_splits (int): Number of subdivisions for the inclination domain.
        samples_per_expert (int): Number of Monte Carlo samples per grid cell.
    """
    total_models = sma_splits * ecc_splits * inc_splits
    
    print("-" * 80)
    print(f" ORBITA FACTORY: BUILDING 3D GRID OF {total_models} EXPERT MODELS")
    print(f"    SMA Splits (Altitude)  : {sma_splits}")
    print(f"    ECC Splits (Shape)     : {ecc_splits}")
    print(f"    INC Splits (Angles)    : {inc_splits}")
    print(f"    Samples per Cell       : {int(samples_per_expert)}")
    print("-" * 80)
    
    # Calculate step sizes for the 3D grid dimensions
    sma_min, sma_max = total_sma_bounds
    sma_step = (sma_max - sma_min) / sma_splits
    
    ecc_min, ecc_max = total_ecc_bounds
    ecc_step = (ecc_max - ecc_min) / ecc_splits
    
    inc_min, inc_max = total_inc_bounds
    inc_step = (inc_max - inc_min) / inc_splits
    
    # Fixed angular bounds for all experts (covering a full 360-degree orbit)
    raan_bounds = (0.0, 2.0 * np.pi)
    aop_bounds  = (0.0, 2.0 * np.pi)
    ta_bounds   = (0.0, 2.0 * np.pi)
    tof_bounds  = (0.0, 30.0 * 60.0)  # Up to 30 minutes of flight time
    
    model_counter = 1
    
    # --- OUTER LOOP: Traverse the Semi-Major Axis (Altitude) ---
    for i in range(sma_splits):
        expert_sma_min = sma_min + (i * sma_step)
        expert_sma_max = expert_sma_min + sma_step
        expert_sma_bounds = (expert_sma_min, expert_sma_max)
        
        # Convert to km for clean file naming
        alt_min_km = int((expert_sma_min - R_EQ) / 1e3)
        alt_max_km = int((expert_sma_max - R_EQ) / 1e3)
        sma_str = f"{alt_min_km}-{alt_max_km}"
        
        # --- MIDDLE LOOP: Traverse the Eccentricity (Shape) ---
        for j in range(ecc_splits):
            expert_ecc_min = ecc_min + (j * ecc_step)
            expert_ecc_max = expert_ecc_min + ecc_step
            expert_ecc_bounds = (expert_ecc_min, expert_ecc_max)
            
            # Format to 2 decimal places to avoid messy float names
            ecc_str = f"{expert_ecc_min:.4f}-{expert_ecc_max:.4f}"
            
            # --- INNER LOOP: Traverse the Inclination (Tilt) ---
            for k in range(inc_splits):
                expert_inc_min = inc_min + (k * inc_step)
                expert_inc_max = expert_inc_min + inc_step
                expert_inc_bounds = (expert_inc_min, expert_inc_max)
                
                # Convert to degrees for clean file naming
                deg_min = int(np.degrees(expert_inc_min))
                deg_max = int(np.degrees(expert_inc_max))
                inc_str = f"{deg_min}-{deg_max}"
                
                csv_filename = f"data/orbita_dataset_{sma_str}_{ecc_str}_{inc_str}.csv"
                
                print("\n" + "=" * 75)
                print(f" PROCESSING EXPERT {model_counter}/{total_models}")
                print(f" Domain: Alt [{alt_min_km}-{alt_max_km} km] | Ecc [{ecc_str}] | Inc [{deg_min}-{deg_max} deg]")
                print("=" * 75)
                
                # --- PIPELINE STEP 1: DATA GENERATION ---
                print(f" [step 1] Generating {int(samples_per_expert)} dataset samples...")
                generate_training_data(
                    num_samples=int(samples_per_expert),
                    output_file=csv_filename,
                    sma_bounds=expert_sma_bounds,
                    ecc_bounds=expert_ecc_bounds,
                    inc_bounds=expert_inc_bounds,
                    raan_bounds=raan_bounds,
                    aop_bounds=aop_bounds,
                    ta_bounds=ta_bounds,
                    tof_bounds=tof_bounds
                )
                
                # --- PIPELINE STEP 2: MODEL TRAINING ---
                print(f"\n [step 2] Training neural network...")
                train_model(
                    csv_file=csv_filename,
                )
                
                print(f"\n [info] EXPERT {model_counter} COMPLETE.")
                model_counter += 1
            
    print("-" * 80)
    print(" 3D GRID FLEET GENERATION FINISHED SUCCESSFULLY.")
    print("-" * 80)


# =============================================================================
# Execution Block
# =============================================================================
if __name__ == "__main__":
    # Define the total operational domain
    total_sma = (R_EQ + 300e3, R_EQ + 2000e3)  # 300 km to 2000 km
    total_ecc = (0, 0.1)                       # Circular to low-elliptical
    total_inc = (0.0, np.radians(90.0))        # Equatorial to polar
    
    # Build a sma_splits x ecc_splits x inc_splits grid
    build_expert_grid(
        total_sma_bounds=total_sma,
        total_ecc_bounds=total_ecc,
        total_inc_bounds=total_inc,
        sma_splits=17,
        ecc_splits=10,
        inc_splits=9,
        samples_per_expert=100000
    )