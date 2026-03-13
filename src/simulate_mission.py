"""
Script: simulate_mission.py

Description:
    Final flight simulator implementing the 3D Mixture of Experts (MoE) routing.
    Dynamically selects the appropriate specialized Neural Network based on 
    the mission's initial orbital parameters. It propagates the orbit, queries 
    the AI for COE secular corrections, and uses MC-Dropout uncertainty to 
    safely fall back to the Numerical Oracle when the 15-meter threshold is breached.
"""

import os
import glob
import torch
import numpy as np
from physics.analytical import computeGeneralSolution
from physics.oracle     import COE2ECI, ECI2COE, getGroundTruth
from physics.kepler     import getKeplerianNumerical
from ml.active_learning import ResidualPredictor
from ml.dataset         import OrbitalDataset

# =============================================================================
# GLOBAL ASTRODYNAMIC CONSTANTS
# =============================================================================
MU   = 3.986004418e14    # Gravitational parameter       [m^3/s^2]
J2   =   1.082635854e-3  # J2 zonal harmonic coefficient       [-]
J3   = - 2.532435346e-6  # J3 zonal harmonic coefficient       [-]
R_EQ = 6378.137e3        # Equatorial radius                   [m]

def findExpertSystem(SMA, ECC, INC):
    """
    Acts as the MoE Router. Scans the 'models' directory and dynamically 
    selects the correct expert Neural Network based on the initial state.
    
    Returns:
        model_path   : Path to the expert's .pth weights.
        dataset_path : Path to the expert's .csv to load normalization stats.
    """
    alt_km = (SMA - R_EQ) / 1000.0
    INC_deg = np.degrees(INC)
    
    print(" [ROUTER] Searching for specialized expert model...")
    
    # Scan all trained models in the directory
    for model_path in glob.glob("models/orbita_predictor_*.pth"):
        basename = os.path.basename(model_path)
        
        # Extract the hyperparameter strings: "AltMin-AltMax_EccMin-EccMax_IncMin-IncMax"
        params_str = basename.replace("orbita_predictor_", "").replace(".pth", "")
        parts = params_str.split("_")
        
        # Skip old or improperly formatted models
        if len(parts) != 3: 
            continue
            
        alt_str, ECC_str, INC_str = parts
        
        # Parse the mathematical boundaries
        alt_min, alt_max = map(float, alt_str.split("-"))
        ECC_min, ECC_max = map(float, ECC_str.split("-"))
        INC_min, INC_max = map(float, INC_str.split("-"))
        
        # Check if the mission orbit falls within this expert's domain
        if (alt_min <= alt_km <= alt_max) and \
           (ECC_min <= ECC <= ECC_max) and \
           (INC_min <= INC_deg <= INC_max):
            
            dataset_path = f"data/orbita_dataset_{params_str}.csv"
            print(f" [ROUTER] Expert found: {basename}")
            return model_path, dataset_path
            
    raise ValueError(f"No expert model covers this orbit (Alt: {alt_km}km, Ecc: {ECC}, Inc: {INC_deg}deg)")

def run_stress_test_simulation():
    print("=" * 80)
    print(" 🚀 INITIATING ORBITA FLIGHT SIMULATION (MoE ARCHITECTURE)")
    print("=" * 80)
    
    # 1. Define Mission Parameters
    # Adjusted to 900 km altitude to fit cleanly into a mid-LEO expert block
    SMA = R_EQ + 1400e3 
    ECC = 0
    INC = np.deg2rad(90.0) 
    RAAN, AOP, TA = 0.0, 0.0, 0.0
    
    time_steps = np.arange(0, 12001, 15 * 60) # 3 hours, 15 min steps
    
    # Extreme precision safety threshold (15 meters)
    UNCERTAINTY_THRESHOLD = 15.0 
    
    # 2. MoE Routing: Find and Load the Expert Brain
    try:
        model_path, dataset_path = findExpertSystem(SMA = SMA, ECC = ECC, INC = INC)
    except ValueError as e:
        print(f"🚨 ROUTING ERROR: {e}")
        return

    # Load Normalization Stats
    dataset = OrbitalDataset(dataset_path)
    X_mean, X_std = dataset.X_mean, dataset.X_std
    
    # Initialize and load weights
    model = ResidualPredictor()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    oracle_calls = 0
    ai_calls = 0
    
    print(f"\n Simulating orbit over {len(time_steps) - 1} evaluation steps...")
    print(f" Safety Uncertainty Threshold: {UNCERTAINTY_THRESHOLD} meters\n")
    print(f"{'Time (s)':<10} | {'Decision':<10} | {'Uncertainty':<17} | {'Action'}")
    print("-" * 75)

    # 3. Flight Loop
    for tof in time_steps:
        if tof == 0:
            continue 
            
        # Prepare network inputs with trigonometric expansion
        raw_input = np.array([
            SMA, ECC, INC, 
            np.sin(RAAN), np.cos(RAAN),
            np.sin(AOP), np.cos(AOP),
            np.sin(TA), np.cos(TA),
            tof
        ])
        norm_input = (raw_input - X_mean) / X_std
        tensor_input = torch.tensor(norm_input, dtype=torch.float32).unsqueeze(0)
        
        # Predict with Uncertainty (MC-Dropout)
        mean_pred, std_pred = model.predict_with_uncertainty(tensor_input, num_samples = 15)
        
        mean_pred_np = mean_pred.numpy()[0]
        std_pred_np = std_pred.numpy()[0]
        
        # De-normalize uncertainty to physical units
        std_pred_coe = std_pred_np * dataset.Y_std
        
        # Scalar magnitude (dominated by SMA variation in meters)
        scalar_uncertainty = np.linalg.norm(std_pred_coe)
        
        # 4. The Active Learning Decision Gateway
        if scalar_uncertainty > UNCERTAINTY_THRESHOLD:
            decision = "🚨 ORACLE"
            oracle_calls += 1
            action = "Numerical Integration Executed (AI unsure!)"
            
            # Ground truth integration fallback
            pos_final, vel_final = getGroundTruth(SMA, ECC, INC, RAAN, AOP, TA, tof)

        else:
            decision = "✅ AI MODEL"
            ai_calls += 1
            
            # A) Cartesian Analytical Propagation
            r0, v0 = COE2ECI(MU, SMA, ECC, INC, RAAN, AOP, TA)
            X0_pert = np.zeros(6)
            n = np.sqrt(MU / (SMA**3))
            
            deltaPos_ESTHER, deltaVel_ESHTER = computeGeneralSolution(
                J2, J3, R_EQ, SMA, ECC, INC, RAAN, AOP, TA, X0_pert, n, tof
            )
            pos_kep, vel_kep = getKeplerianNumerical(MU, r0, v0, tof)
            
            pos_ESTHER = pos_kep + deltaPos_ESTHER
            vel_ESTHER = vel_kep + deltaVel_ESHTER
            
            # B) Convert Analytical to COE
            coe_ESTHER = ECI2COE(MU, pos_ESTHER, vel_ESTHER)
            
            # C) Apply AI residual correction to COEs
            ai_correction_coe = (mean_pred_np * dataset.Y_std) + dataset.Y_mean
            coe_corrected = coe_ESTHER + ai_correction_coe
            
            # D) Rebuild Cartesian State Vector
            final_pos, final_vel = COE2ECI(
                MU, 
                coe_corrected[0], coe_corrected[1], coe_corrected[2], 
                coe_corrected[3], coe_corrected[4], coe_corrected[5]
            )
            
            action = f"Analyt -> COE corrected -> ECI Rebuilt"
            
        print(f"{tof:<10.1f} | {decision:<10} | {scalar_uncertainty:<17.4f} | {action}")

    print("-" * 75)
    print("--- MISSION SUMMARY ---")
    total_steps = len(time_steps) - 1
    print(f"Total Evaluation Steps : {total_steps}")
    print(f"Oracle Invocations     : {oracle_calls} ({(oracle_calls/total_steps)*100:.1f}%)")
    print(f"AI Fast Predictions    : {ai_calls} ({(ai_calls/total_steps)*100:.1f}%)")
    print("Simulation Complete.")


# =============================================================================
# Execution Block
# =============================================================================
if __name__ == "__main__":
    run_stress_test_simulation()