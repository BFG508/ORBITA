"""
Module: simulate_mission.py

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
import numpy as np
import torch

from physics.analytical import compute_general_solution
from physics.oracle import coe_to_eci, eci_to_coe, coe_to_mee, mee_to_coe, get_ground_truth
from physics.kepler import get_keplerian
from ml.architecture import ResidualPredictor
from ml.dataset import OrbitalDataset

from physics.oracle import MU, J2, J3, R_EQ


def find_expert_system(sma, ecc, inc):
    """
    Acts as the MoE Router. Scans the 'models' directory and dynamically 
    selects the correct expert neural network based on the initial state.
    Prioritizes fine-tuned (upgraded) models if they are available.
    
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
    
    # 1. Gather only the base models to parse the domain bounds correctly
    base_models = [m for m in glob.glob("models/orbita_predictor_*.pth") 
                   if "_finetuned" not in m]
    
    for base_model_path in base_models:
        # Normalize paths for cross-platform compatibility
        base_model_path = base_model_path.replace('\\', '/')
        basename = os.path.basename(base_model_path)
        params_str = basename.replace("orbita_predictor_", "").replace(".pth", "")
        parts = params_str.split("_")
        
        if len(parts) != 3: 
            continue
            
        alt_str, ecc_str, inc_str = parts
        alt_min, alt_max = map(float, alt_str.split("-"))
        ecc_min, ecc_max = map(float, ecc_str.split("-"))
        inc_min, inc_max = map(float, inc_str.split("-"))
        
        # 2. Check if the orbit falls within this expert's domain
        if (alt_min <= alt_km <= alt_max) and (ecc_min <= ecc <= ecc_max) and \
           (inc_min <= inc_deg <= inc_max):
            
            # 3. Domain matched! Check if an upgraded fine-tuned model exists
            finetuned_path = f"models/orbita_predictor_{params_str}_finetuned.pth"
            dataset_path = f"data/orbita_dataset_{params_str}.csv"
            
            if os.path.exists(finetuned_path):
                print(f" [ROUTER] Upgraded Expert found: {os.path.basename(finetuned_path)}")
                return finetuned_path, dataset_path
            else:
                print(f" [ROUTER] Base Expert found: {basename}")
                return base_model_path, dataset_path
                
    raise ValueError(f"No expert model covers this orbit "
                     f"(Alt: {alt_km:.2f}km, Ecc: {ecc:.4f}, Inc: {inc_deg:.2f}deg)")


def run_stress_test_simulation(sma_0, ecc_0, inc_0, raan_0, aop_0, ta_0, 
                               max_tof, step_size=900, uncertainty_threshold=100.0):
    """
    Executes the closed-loop flight simulation, alternating between 
    the analytical-AI hybrid model and the numerical oracle based on 
    epistemic uncertainty bounds.
    
    Args:
        sma_0 (float): Initial semi-major axis [m].
        ecc_0 (float): Initial eccentricity [-].
        inc_0 (float): Initial inclination [rad].
        raan_0 (float): Initial RAAN [rad].
        aop_0 (float): Initial argument of perigee [rad].
        ta_0 (float): Initial true anomaly [rad].
        max_tof (float): Maximum Time of Flight for the simulation [s].
        step_size (int): Time step interval for iterative propagation [s].
        uncertainty_threshold (float): Spatial threshold to trigger GPS Reset [m].
    """
    print("-" * 80)
    print(" INITIATING ORBITA FLIGHT SIMULATION (MoE ARCHITECTURE)")
    print("-" * 80)
    
    # Define evaluation steps dynamically
    time_steps = np.arange(0, max_tof + 1, step_size) 
    
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
    print(f"{'Time (s)':<9} | {'Decision':<12} | {'Uncertainty [m]':<15} | {'Absolute Error [m]':<10}")
    print("-" * 80)

    # =========================================================================
    # ITERATIVE PROPAGATION INITIALIZATION
    # We maintain a 'current' state that updates after every dt interval
    # =========================================================================
    curr_sma, curr_ecc, curr_inc = sma_0, ecc_0, inc_0
    curr_raan, curr_aop, curr_ta = raan_0, aop_0, ta_0
    prev_tof = 0.0

    # Flight Loop
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
        model.train()  # Force Dropout activation for statistical variance
        
        # Propagate analytical baseline for dt
        r0, v0 = coe_to_eci(MU, curr_sma, curr_ecc, curr_inc, curr_raan, curr_aop, curr_ta)
        x0_pert = np.zeros(6)
        n_mean = np.sqrt(MU / (curr_sma**3))
        
        delta_pos_esther, delta_vel_esther = compute_general_solution(
            J2, J3, R_EQ, curr_sma, curr_ecc, curr_inc, curr_raan, curr_aop, 
            curr_ta, x0_pert, n_mean, dt
        )
        pos_kep, vel_kep = get_keplerian(MU, r0, v0, dt)
        
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
        model.eval()  # Disable Dropout for actual deterministic prediction
        
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
        
        print(f"{tof:<9.1f} | {decision:<12} | {scalar_uncertainty:<15.4f} | {absolute_error_meters:<16.4f}")

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
    # 1. Mission Configuration (Orbit Parameters at T0)
    TARGET_SMA = R_EQ + 300e3 
    TARGET_ECC = 0.02
    TARGET_INC = np.deg2rad(0) 
    TARGET_RAAN = 0.0
    TARGET_AOP = 0.0
    TARGET_TA = 0.0
    
    # 2. Simulation Constraints
    MAX_TOF = 4 * 3600             # 4 hours of total flight time
    PROPAGATION_STEP = 15 * 60     # 15-minute intervals
    SAFETY_THRESHOLD = 100.0       # 100 meters allowable epistemic uncertainty
    
    # 3. Execution
    run_stress_test_simulation(
        sma_0=TARGET_SMA, 
        ecc_0=TARGET_ECC, 
        inc_0=TARGET_INC, 
        raan_0=TARGET_RAAN, 
        aop_0=TARGET_AOP, 
        ta_0=TARGET_TA, 
        max_tof=MAX_TOF,
        step_size=PROPAGATION_STEP,
        uncertainty_threshold=SAFETY_THRESHOLD
    )