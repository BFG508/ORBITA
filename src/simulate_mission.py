"""
Script: simulate_mission.py

Description:
    Final flight simulator implementing the 3D Mixture of Experts (MoE) routing.
    
    Features iterative propagation with GPS reset: The script propagates the orbit 
    in short time steps (dt). The AI predicts the secular corrections for these 
    intervals using stable Modified Equinoctial Elements (MEE) inputs. 
    If the epistemic uncertainty (MC-Dropout Euclidean spread) breaches the safety 
    threshold, the system triggers a reset, calling the numerical oracle to compute 
    the absolute true state from T0, wiping out any accumulated error.
"""

import os
import glob
import torch
import numpy as np
from physics.analytical import compute_general_solution
from physics.oracle import coe_to_eci, eci_to_coe, coe_to_mee, mee_to_coe, get_ground_truth
from physics.kepler import get_keplerian_numerical
from ml.active_learning import ResidualPredictor
from ml.dataset import OrbitalDataset

# =============================================================================
# GLOBAL ASTRODYNAMIC CONSTANTS
# =============================================================================
MU = 3.986004418e14      # Gravitational parameter [m^3/s^2]
J2 = 1.082635854e-3      # J2 zonal harmonic coefficient [-]
J3 = -2.532435346e-6     # J3 zonal harmonic coefficient [-]
R_EQ = 6378.137e3        # Equatorial radius [m]

def find_expert_system(sma, ecc, inc):
    """
    Acts as the MoE Router. Scans the 'models' directory and dynamically 
    selects the correct expert neural network based on the initial state.
    
    Args:
        sma (float): Semi-major axis [m].
        ecc (float): Eccentricity [-].
        inc (float): Inclination [rad].
        
    Returns:
        tuple: (model_path, dataset_path) for the selected expert.
    """
    alt_km = (sma - R_EQ) / 1000.0
    inc_deg = np.degrees(inc)
    
    print(" [ROUTER] Searching for specialized expert model...")
    for model_path in glob.glob("models/orbita_predictor_*.pth"):
        basename = os.path.basename(model_path)
        params_str = basename.replace("orbita_predictor_", "").replace(".pth", "")
        parts = params_str.split("_")
        
        if len(parts) != 3: 
            continue
            
        alt_str, ecc_str, inc_str = parts
        alt_min, alt_max = map(float, alt_str.split("-"))
        ecc_min, ecc_max = map(float, ecc_str.split("-"))
        inc_min, inc_max = map(float, inc_str.split("-"))
        
        # Check domain applicability
        if (alt_min <= alt_km <= alt_max) and (ecc_min <= ecc <= ecc_max) and (inc_min <= inc_deg <= inc_max):
            dataset_path = f"data/orbita_dataset_{params_str}.csv"
            print(f" [ROUTER] Expert found: {basename}")
            return model_path, dataset_path
            
    raise ValueError(f"No expert model covers this orbit (Alt: {alt_km}km, Ecc: {ecc}, Inc: {inc_deg}deg)")


def run_stress_test_simulation():
    """
    Executes the closed-loop flight simulation, alternating between 
    the analytical-AI hybrid model and the numerical oracle based on 
    epistemic uncertainty bounds.
    """
    print("-" * 80)
    print(" INITIATING ORBITA FLIGHT SIMULATION (MoE ARCHITECTURE)")
    print("-" * 80)
    
    # 1. Define absolute initial mission parameters (T0)
    sma_0 = R_EQ + 300e3 
    ecc_0 = 0.01
    inc_0 = np.deg2rad(0) 
    raan_0, aop_0, ta_0 = 0.0, 0.0, 0.0
    
    # Define evaluation steps (4 hours, 15-minute steps)
    time_steps = np.arange(0, 4 * 3600 + 1, 15 * 60) 
    
    # Safety margin established at 100 meters
    uncertainty_threshold = 100.0 
    
    try:
        model_path, dataset_path = find_expert_system(sma=sma_0, ecc=ecc_0, inc=inc_0)
    except ValueError as e:
        print(f" [ERROR] Routing failure: {e}")
        return

    # Load dataset normalization variables
    dataset = OrbitalDataset(dataset_path)
    
    # Load AI Brain (configured for 8 MEE inputs)
    model = ResidualPredictor(input_size=8)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    oracle_calls = 0
    ai_calls = 0
    
    print(f"\n Simulating orbit over {len(time_steps) - 1} evaluation steps...")
    print(f" Safety threshold: {uncertainty_threshold} meters\n")
    print(f"{'Time (s)':<10} | {'Decision':<12} | {'Uncertainty [m]':<12} | {'Error [m]'}")
    print("-" * 80)

    # =========================================================================
    # ITERATIVE PROPAGATION INITIALIZATION
    # We maintain a 'current' state that updates after every dt interval
    # =========================================================================
    curr_sma, curr_ecc, curr_inc = sma_0, ecc_0, inc_0
    curr_raan, curr_aop, curr_ta = raan_0, aop_0, ta_0
    prev_tof = 0.0

    # 3. Flight Loop
    for tof in time_steps:
        if tof == 0:
            continue 
            
        # Dynamic time step calculation (dt)
        dt = tof - prev_tof
            
        # ---------------------------------------------------------------------
        # NETWORK INPUT PREPARATION (MEE Formulated)
        # ---------------------------------------------------------------------
        # Convert current classical state to MEE for singularity-free AI input
        mee_current = coe_to_mee(curr_sma, curr_ecc, curr_inc, curr_raan, curr_aop, curr_ta)
        p_in, f_in, g_in, h_in, k_in, l_in = mee_current
        
        # Build the 8-variable input vector matching the dataset structure
        raw_input = np.array([
            p_in, f_in, g_in, h_in, k_in, 
            np.sin(l_in), np.cos(l_in), 
            dt
        ])
        
        # Apply dataset standardization
        norm_input = (raw_input - dataset.x_mean) / dataset.x_std
        tensor_input = torch.tensor(norm_input, dtype=torch.float32).unsqueeze(0)
        
        # ---------------------------------------------------------------------
        # EUCLIDEAN MONTE CARLO DROPOUT (Uncertainty Cloud)
        # ---------------------------------------------------------------------
        model.train() # Force Dropout activation for statistical variance
        
        # Propagate analytical baseline for dt
        r0, v0 = coe_to_eci(MU, curr_sma, curr_ecc, curr_inc, curr_raan, curr_aop, curr_ta)
        x0_pert = np.zeros(6)
        n_mean = np.sqrt(MU / (curr_sma**3))
        
        delta_pos_esther, delta_vel_esther = compute_general_solution(
            J2, J3, R_EQ, curr_sma, curr_ecc, curr_inc, curr_raan, curr_aop, curr_ta, x0_pert, n_mean, dt
        )
        pos_kep, vel_kep = get_keplerian_numerical(MU, r0, v0, dt)
        
        pos_esther = pos_kep + delta_pos_esther
        vel_esther = vel_kep + delta_vel_esther
        
        coe_esther = eci_to_coe(MU, pos_esther, vel_esther)
        mee_esther = coe_to_mee(*coe_esther)
        
        cloud_eci_positions = []
        raw_mee_predictions = []
        num_samples = 50
        
        with torch.no_grad():
            for _ in range(num_samples):
                pred_norm = model(tensor_input)
                pred_mee = (pred_norm.numpy()[0] * dataset.y_std) + dataset.y_mean
                raw_mee_predictions.append(pred_mee)
                
                mee_sample = mee_esther + pred_mee
                coe_sample = mee_to_coe(*mee_sample)
                pos_sample, _ = coe_to_eci(MU, *coe_sample)
                
                cloud_eci_positions.append(pos_sample)
                
        cloud_eci_positions = np.array(cloud_eci_positions)
        std_3d = np.std(cloud_eci_positions, axis=0) 
        scalar_uncertainty = np.linalg.norm(std_3d) 
        
        mean_pred_mee = np.mean(raw_mee_predictions, axis=0)
        model.eval() # Disable Dropout for actual prediction
        
        # ---------------------------------------------------------------------
        # ACTIVE LEARNING DECISION GATEWAY
        # ---------------------------------------------------------------------
        if scalar_uncertainty > uncertainty_threshold:
            decision = "[ORACLE]"
            oracle_calls += 1
            
            # The "GPS Reset": Wipes out accumulated error by propagating from T0
            pos_final, vel_final = get_ground_truth(
                sma_0, ecc_0, inc_0, raan_0, aop_0, ta_0, tof
            )
            
        else:
            decision = "[AI MODEL]"
            ai_calls += 1
            
            mee_corrected = mee_esther + mean_pred_mee
            coe_corrected = mee_to_coe(*mee_corrected)
            pos_final, vel_final = coe_to_eci(MU, *coe_corrected)
            
        # =====================================================================
        # TRUTH DETECTOR (Absolute Accumulated Error from T0)
        # =====================================================================
        pos_truth, _ = get_ground_truth(sma_0, ecc_0, inc_0, raan_0, aop_0, ta_0, tof)
        absolute_error_meters = np.linalg.norm(pos_final - pos_truth)
        
        action = f"Absolute: {absolute_error_meters:6.2f}  &  Spread: {scalar_uncertainty:6.2f}"
        print(f"{tof:<10.1f} | {decision:<12} | {scalar_uncertainty:<15.4f} | {action}")

        # =====================================================================
        # STATE UPDATE (Closing the loop)
        # =====================================================================
        nuevos_coe = eci_to_coe(MU, pos_final, vel_final)
        curr_sma, curr_ecc, curr_inc, curr_raan, curr_aop, curr_ta = nuevos_coe
        prev_tof = tof

    print("-" * 80)
    print(" MISSION SUMMARY")
    total_steps = len(time_steps) - 1
    print(f" Total evaluation steps : {total_steps}")
    print(f" Oracle invocations     : {oracle_calls} ({(oracle_calls/total_steps)*100:.1f}%)")
    print(f" AI fast predictions    : {ai_calls} ({(ai_calls/total_steps)*100:.1f}%)")
    print(" Simulation complete.")


# =============================================================================
# Execution Block
# =============================================================================
if __name__ == "__main__":
    run_stress_test_simulation()