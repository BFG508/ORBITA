"""
Script: orchestrator.py

Description:
    Automates the creation of a 3D Mixture of Experts (MoE) architecture.
    It takes a continuous orbital domain and subdivides it into a 3D grid 
    based on Semi-Major Axis (SMA), Eccentricity (ECC), and Inclination (INC).
    
    This hyper-specialization completely isolates scale effects (altitude), 
    denominator singularities (eccentricity), and trigonometric non-linearities 
    (inclination). For each 3D cell, it generates the dataset and trains a 
    highly specialized Neural Network.
"""

import os
import numpy as np
from generate_dataset import genTrainingData
from train            import trainModel

# =============================================================================
# GLOBAL ASTRODYNAMIC CONSTANTS
# =============================================================================
R_EQ = 6378.137e3  # Earth Equatorial Radius [m]

def buildExpertGrid(
    total_SMA_bounds, 
    total_ECC_bounds, 
    total_INC_bounds,
    SMA_splits, 
    ECC_splits,
    INC_splits, 
    samples_per_expert = 10000
):
    """
    Executes the automated pipeline to generate and train a 3D grid of expert models.
    
    Inputs:
        total_SMA_bounds  : (min, max) for the entire Semi-Major Axis space      [m]
        total_ECC_bounds  : (min, max) for the entire Eccentricity space         [-]
        total_INC_bounds  : (min, max) for the entire Inclination space        [rad]
        SMA_splits        : Number of subdivisions for the altitude domain
        ECC_splits        : Number of subdivisions for the eccentricity domain
        INC_splits        : Number of subdivisions for the inclination domain
        samples_per_expert: Number of Monte Carlo samples per grid cell
    """
    total_models = SMA_splits * ECC_splits * INC_splits
    
    print("=" * 80)
    print(f" 🏭 ORBITA FACTORY: BUILDING 3D GRID OF {total_models} EXPERT MODELS")
    print(f"    SMA Splits (Altitude)  : {SMA_splits}")
    print(f"    ECC Splits (Shape)     : {ECC_splits}")
    print(f"    INC Splits (Angles)    : {INC_splits}")
    print(f"    Samples per Cell       : {samples_per_expert}")
    print("=" * 80)
    
    # Calculate step sizes for the 3D grid dimensions
    SMA_min, SMA_max = total_SMA_bounds
    SMA_step = (SMA_max - SMA_min) / SMA_splits
    
    ECC_min, ECC_max = total_ECC_bounds
    ECC_step = (ECC_max - ECC_min) / ECC_splits
    
    INC_min, INC_max = total_INC_bounds
    INC_step = (INC_max - INC_min) / INC_splits
    
    # Fixed angular bounds for all experts (covering a full 360-degree orbit)
    RAAN_bounds = (0.0, 2 * np.pi)
    AOP_bounds  = (0.0, 2 * np.pi)
    TA_bounds   = (0.0, 2 * np.pi)
    TOF_bounds  = (0.0, 4.0 * 3600.0) # Up to 4 hours of flight time
    
    model_counter = 1
    
    # --- OUTER LOOP: Traverse the Semi-Major Axis (Altitude) ---
    for i in range(SMA_splits):
        expert_SMA_min = SMA_min + (i * SMA_step)
        expert_SMA_max = expert_SMA_min + SMA_step
        expert_SMA_bounds = (expert_SMA_min, expert_SMA_max)
        
        # Convert to km for clean file naming
        alt_min_km = int((expert_SMA_min - R_EQ) / 1e3)
        alt_max_km = int((expert_SMA_max - R_EQ) / 1e3)
        SMA_str = f"{alt_min_km}-{alt_max_km}"
        
        # --- MIDDLE LOOP: Traverse the Eccentricity (Shape) ---
        for j in range(ECC_splits):
            expert_ECC_min = ECC_min + (j * ECC_step)
            expert_ECC_max = expert_ECC_min + ECC_step
            expert_ECC_bounds = (expert_ECC_min, expert_ECC_max)
            
            # Format to 3 decimal places to avoid messy float names
            ECC_str = f"{expert_ECC_min:.3f}-{expert_ECC_max:.3f}"
            
            # --- INNER LOOP: Traverse the Inclination (Tilt) ---
            for k in range(INC_splits):
                expert_INC_min = INC_min + (k * INC_step)
                expert_INC_max = expert_INC_min + INC_step
                expert_INC_bounds = (expert_INC_min, expert_INC_max)
                
                # Convert to degrees for clean file naming
                deg_min = int(np.degrees(expert_INC_min))
                deg_max = int(np.degrees(expert_INC_max))
                INC_str = f"{deg_min}-{deg_max}"
                
                csv_filename = f"data/orbita_dataset_{SMA_str}_{ECC_str}_{INC_str}.csv"
                
                print("\n" + "*" * 75)
                print(f" ⚙️  PROCESSING EXPERT {model_counter}/{total_models}")
                print(f"    Domain: Alt [{alt_min_km}-{alt_max_km} km] | Ecc [{ECC_str}] | Inc [{deg_min}-{deg_max} deg]")
                print("*" * 75)
                
                # --- PIPELINE STEP 1: DATA GENERATION ---
                print(f" [STEP 1] Generating {samples_per_expert} Dataset Samples...")
                genTrainingData(
                    num_samples = samples_per_expert,
                    output_file = csv_filename,
                    SMA_bounds  = expert_SMA_bounds,
                    ECC_bounds  = expert_ECC_bounds,
                    INC_bounds  = expert_INC_bounds,
                    RAAN_bounds = RAAN_bounds,
                    AOP_bounds  = AOP_bounds,
                    TA_bounds   = TA_bounds,
                    TOF_bounds  = TOF_bounds
                )
                
                # --- PIPELINE STEP 2: MODEL TRAINING ---
                print(f"\n [STEP 2] Training Neural Network...")
                trainModel(
                    csv_file   = csv_filename,
                )
                
                print(f"\n ✅ EXPERT {model_counter} COMPLETE.")
                model_counter += 1
            
    print("=" * 80)
    print(" 🚀 3D GRID FLEET GENERATION FINISHED SUCESSFULLY.")
    print("=" * 80)


# =============================================================================
# Execution Block
# =============================================================================
if __name__ == "__main__":
    
    # Define the total operational domain
    total_SMA = (R_EQ + 300e3, R_EQ + 2000e3) # 300 km to 2000 km
    total_ECC = (0.0, 0.1)                    # Circular to low-elliptical
    total_INC = (0.0, np.radians(90.0))       # Equatorial to polar
    
    # Build a SMA_splits x ECC_splits x INC_splits grid
    buildExpertGrid(
        total_SMA_bounds   = total_SMA,
        total_ECC_bounds   = total_ECC,
        total_INC_bounds   = total_INC,
        SMA_splits         = 17,
        ECC_splits         = 10,
        INC_splits         = 18,      
        samples_per_expert = 5000
    )