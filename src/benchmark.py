"""
Module: benchmark.py

Description:
    Flight-grade aerospace benchmark suite for the ORBITA framework.
    Validates the 3D Mixture of Experts (MoE) Grey-Box AI architecture against 
    the high-precision Numerical Oracle (Cowell's formulation via LSODA).

    This suite provides a comprehensive statistical and temporal analysis of 
    the network's predictive capabilities across the Low Earth Orbit (LEO) regime,
    isolating error components into the physically meaningful RIC coordinate system.

Validation Modes:
    1. Time-Domain (Secular Degradation): Evaluates the long-term numerical 
       stability of the hybrid model by propagating a target initial state 
       over continuous hours. Designed to stress-test error accumulation.
    2. Space-Domain (Global LEO Monte Carlo): Evaluates the generalization 
       capacity of the MoE grid by generating thousands of randomized 
       orbital states and propagating them for a single time step.
"""

import os
import time
import csv
import torch
import numpy as np

from physics.analytical import compute_general_solution
from physics.oracle import coe_to_eci, eci_to_coe, coe_to_mee, mee_to_coe, get_ground_truth
from physics.kepler import get_keplerian
from ml.architecture import ResidualPredictor
from ml.dataset import OrbitalDataset
from simulate_mission import find_expert_system

from physics.oracle import MU, J2, J3, R_EQ
from generate_base_dataset import MIN_SAFE_PERIGEE


def eci_to_ric_error(r_true, v_true, r_est):
    """
    Transforms the position error vector from the ECI (Earth-Centered Inertial) 
    frame to the local RIC (Radial, In-Track, Cross-Track) orbital frame.
    
    The RIC coordinate system rotates with the satellite, allowing for a physically 
    meaningful interpretation of propagation errors:
    - Radial (R): Aligned with the position vector.
    - Cross-Track (C): Aligned with the orbital angular momentum vector.
    - In-Track (I): Completes the right-handed orthogonal triad (C x R).
    
    Args:
        r_true (np.ndarray): True position vector in ECI [m].
        v_true (np.ndarray): True velocity vector in ECI [m/s].
        r_est (np.ndarray): Estimated (AI) position vector in ECI [m].
        
    Returns:
        np.ndarray: Error vector in the RIC frame [m] (Radial, In-Track, Cross-Track).
    """
    # 1. Define the Radial unit vector (aligned with the position vector)
    r_unit = r_true / np.linalg.norm(r_true)
    
    # 2. Define the Cross-Track unit vector (aligned with the angular momentum vector)
    h_vec = np.cross(r_true, v_true)
    c_unit = h_vec / np.linalg.norm(h_vec)
    
    # 3. Define the In-Track unit vector (completes the orthogonal triad)
    i_unit = np.cross(c_unit, r_unit)
    
    # Construct the rotation matrix mapping ECI to the local RIC frame
    rotation_matrix = np.vstack((r_unit, i_unit, c_unit))
    
    # Compute the absolute error vector in the ECI frame
    delta_r_eci = r_est - r_true
    
    # Project the ECI error into the local RIC frame
    delta_r_ric = rotation_matrix @ delta_r_eci
    
    return delta_r_ric


def execute_ai_step(sma, ecc, inc, raan, aop, ta, dt, model, dataset):
    """
    Executes a single Grey-Box propagation step using the AI Hybrid architecture.
    
    This function acts as the core of the ORBITA framework. It integrates the 
    deterministic analytical baseline (ESTHER) with the Deep Neural Network 
    (ResNet) to predict and correct high-frequency orbital perturbations over 
    a given Time of Flight (dt).
    
    Args:
        sma (float): Semi-major axis [m].
        ecc (float): Eccentricity [-].
        inc (float): Inclination [rad].
        raan (float): Right Ascension of the Ascending Node [rad].
        aop (float): Argument of Perigee [rad].
        ta (float): True Anomaly [rad].
        dt (float): Time step / Time of Flight for the propagation [s].
        model (ResidualPredictor): Loaded PyTorch expert model for the current grid.
        dataset (OrbitalDataset): Dataset object containing normalization statistics.
        
    Returns:
        tuple: A tuple containing:
            - pos_est (np.ndarray): Final corrected position vector in ECI [m].
            - vel_est (np.ndarray): Final corrected velocity vector in ECI [m/s].
    """
    # =========================================================================
    # 1. AI PREDICTION: Estimate secular and periodic MEE residuals
    # =========================================================================
    # Convert Classical Orbital Elements (COE) to Modified Equinoctial Elements 
    # (MEE) to provide the AI with a singularity-free mathematical space.
    mee_current = coe_to_mee(sma, ecc, inc, raan, aop, ta)
    p_in, f_in, g_in, h_in, k_in, l_in = mee_current
    
    # Build the raw feature vector and normalize it using the expert's statistics
    raw_input = np.array([p_in, f_in, g_in, h_in, k_in, np.sin(l_in), np.cos(l_in), dt])
    norm_input = (raw_input - dataset.x_mean) / dataset.x_std
    tensor_input = torch.tensor(norm_input, dtype=torch.float32).unsqueeze(0)
    
    # Execute a forward pass to predict normalized residuals, then denormalize
    with torch.no_grad():
        pred_norm = model(tensor_input)
        pred_mee = (pred_norm.numpy()[0] * dataset.y_std) + dataset.y_mean
        
    # =========================================================================
    # 2. ANALYTICAL BASELINE: ESTHER (J2/J3) + Unperturbed Keplerian
    # =========================================================================
    r0, v0 = coe_to_eci(MU, sma, ecc, inc, raan, aop, ta)
    x0_pert = np.zeros(6)
    n_mean = np.sqrt(MU / (sma**3))
    
    # Compute the analytical perturbation displacements (ESTHER formulation)
    delta_pos_esther, delta_vel_esther = compute_general_solution(
        J2, J3, R_EQ, sma, ecc, inc, raan, aop, ta, x0_pert, n_mean, dt
    )
    
    # Compute the unperturbed two-body orbital state
    pos_kep, vel_kep = get_keplerian(MU, r0, v0, dt)
    
    # Superimpose the analytical perturbations onto the Keplerian baseline
    pos_esther = pos_kep + delta_pos_esther
    vel_esther = vel_kep + delta_vel_esther
    
    # =========================================================================
    # 3. CORRECTION: Superimpose AI residuals onto the analytical state
    # =========================================================================
    # Convert the ESTHER ECI state back to MEE to safely apply the corrections
    coe_esther = eci_to_coe(MU, pos_esther, vel_esther)
    mee_esther = coe_to_mee(*coe_esther)
    
    # Apply the ResNet-predicted residuals (The actual Grey-Box fusion)
    mee_corrected = mee_esther + pred_mee
    
    # Transform the final corrected state back to ECI for downstream use
    coe_corrected = mee_to_coe(*mee_corrected)
    pos_est, vel_est = coe_to_eci(MU, *coe_corrected)
    
    return pos_est, vel_est


def run_time_domain_benchmark(sma, ecc, inc, raan, aop, ta, max_tof):
    """
    Executes the secular degradation test for a specific target orbit.
    
    Propagates a single initial state over a designated maximum Time of Flight 
    to evaluate the accumulated secular error of the AI Hybrid model against 
    the Numerical Oracle. Includes an absolute safety check to prevent evaluating 
    subterranean or re-entering orbits.
    
    Args:
        sma (float): Semi-major axis [m].
        ecc (float): Eccentricity [-].
        inc (float): Inclination [rad].
        raan (float): Right Ascension of the Ascending Node [rad].
        aop (float): Argument of Perigee [rad].
        ta (float): True Anomaly [rad].
        max_tof (float): Maximum Time of Flight for the simulation [s].
    """
    print("\n" + "=" * 80)
    print(" ⏱️ TIME-DOMAIN BENCHMARK: SECULAR DEGRADATION")
    print("=" * 80)
    
    # =========================================================================
    # 0. SAFETY CHECK: Safe Perigee Filter
    # =========================================================================
    if sma * (1.0 - ecc) < MIN_SAFE_PERIGEE:
        print(f" [error] Aborting benchmark. Target orbit is unsafe.")
        print(f"         Perigee altitude is below the 200 km limit.")
        return

    # Generate time steps dynamically up to max_tof (every 15 minutes)
    time_steps = np.arange(0, max_tof + 1, 15 * 60)
    
    # Attempt to route the initial state to the corresponding MoE expert
    try:
        model_path, dataset_path = find_expert_system(sma=sma, ecc=ecc, inc=inc)
    except ValueError as e:
        print(f" [error] Routing failure: {e}")
        return

    # Initialize the expert model and dataset statistics
    dataset = OrbitalDataset(dataset_path)
    model = ResidualPredictor(input_size=8)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    total_oracle_time = 0.0
    total_ai_time = 0.0
    results_log = []
    
    # State tracking variables
    curr_sma, curr_ecc, curr_inc = sma, ecc, inc
    curr_raan, curr_aop, curr_ta = raan, aop, ta
    prev_tof = 0.0
    
    print(f" {'Time (s)':<10} | {'Radial (m)':<12} | {'In-Track (m)':<14} | {'Cross-Track (m)':<15}")
    print("-" * 80)

    # =========================================================================
    # 1. PROPAGATION LOOP
    # =========================================================================
    for tof in time_steps:
        if tof == 0:
            continue 
            
        dt = tof - prev_tof
        
        # --- Oracle Ground Truth ---
        start_time = time.perf_counter()
        pos_true, v_true = get_ground_truth(sma, ecc, inc, raan, aop, ta, tof)
        total_oracle_time += (time.perf_counter() - start_time)
        
        # --- AI Hybrid Estimate ---
        start_time = time.perf_counter()
        pos_est, vel_est = execute_ai_step(
            curr_sma, curr_ecc, curr_inc, curr_raan, curr_aop, curr_ta, dt, model, dataset
        )
        total_ai_time += (time.perf_counter() - start_time)
        
        # --- Error Calculation ---
        ric_error = eci_to_ric_error(pos_true, v_true, pos_est)
        err_r, err_i, err_c = ric_error
        
        print(f" {tof:<10.1f} | {err_r:>10.2f}   | {err_i:>12.2f}   | {err_c:>13.2f}")
        results_log.append([tof, err_r, err_i, err_c])
        
        # Update current state for the next step iteration
        curr_sma, curr_ecc, curr_inc, curr_raan, curr_aop, curr_ta = eci_to_coe(MU, pos_est, vel_est)
        prev_tof = tof

    # =========================================================================
    # 2. LOGGING AND PERFORMANCE METRICS
    # =========================================================================
    os.makedirs("data", exist_ok=True)
    out_file = "data/benchmark_time_domain.csv"
    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Time_s", "Radial_m", "InTrack_m", "CrossTrack_m"])
        writer.writerows(results_log)

    print("-" * 80)
    print(" COMPUTATIONAL PERFORMANCE REPORT")
    print(f" Total steps propagated : {len(time_steps) - 1}")
    print(f" Oracle Time            : {total_oracle_time * 1000:.2f} ms")
    print(f" AI Model Time          : {total_ai_time * 1000:.2f} ms")
    print(f" Speedup Factor         : {total_oracle_time / total_ai_time:.2f}x faster")
    print(f" Data saved to          : {out_file}")


def run_space_domain_benchmark(num_samples=1000, dt=900):
    """
    Executes the global Monte Carlo test across the entire LEO parameter space.
    
    Evaluates the generalization capabilities of the Mixture of Experts (MoE) 
    by generating thousands of random initial states, ensuring they meet the 
    safe perigee criteria, and propagating them for a single time step.
    
    Args:
        num_samples (int): Total number of random orbital states to evaluate.
        dt (float): Time of Flight for the single-step propagation [s].
    """
    print("\n" + "=" * 80)
    print(" 🌐 SPACE-DOMAIN BENCHMARK: GLOBAL LEO COVERAGE")
    print("=" * 80)
    print(f" Generating {num_samples} random orbits...")
    
    sma_pool, ecc_pool, inc_pool = [], [], []
    raan_pool, aop_pool, ta_pool = [], [], []
    
    # =========================================================================
    # 1. MONTE CARLO POOL GENERATION
    # =========================================================================
    # We use a while loop to dynamically reject orbits that violate the 
    # subterranean perigee boundary until we meet the required sample size.
    while len(sma_pool) < num_samples:
        sma_cand = np.random.uniform(R_EQ + 300e3, R_EQ + 2000e3)
        ecc_cand = np.random.uniform(0.0, 0.1)
        
        # Apply the safe perigee filter
        if sma_cand * (1.0 - ecc_cand) >= MIN_SAFE_PERIGEE:
            sma_pool.append(sma_cand)
            ecc_pool.append(ecc_cand)
            inc_pool.append(np.random.uniform(0.0, np.radians(90.0)))
            raan_pool.append(np.random.uniform(0.0, 2 * np.pi))
            aop_pool.append(np.random.uniform(0.0, 2 * np.pi))
            ta_pool.append(np.random.uniform(0.0, 2 * np.pi))

    # Vectorize pools for efficient memory access
    sma_pool = np.array(sma_pool)
    ecc_pool = np.array(ecc_pool)
    inc_pool = np.array(inc_pool)
    raan_pool = np.array(raan_pool)
    aop_pool = np.array(aop_pool)
    ta_pool = np.array(ta_pool)
    
    # Cache dictionaries to prevent continuous I/O read operations of .pth files
    loaded_models = {}
    loaded_datasets = {}
    
    results_log = []
    r_errors, i_errors, c_errors = [], [], []
    
    print(f" Propagating all samples by {dt} seconds (dt)...\n")
    
    # =========================================================================
    # 2. GLOBAL VALIDATION LOOP
    # =========================================================================
    for i in range(num_samples):
        sma, ecc, inc = sma_pool[i], ecc_pool[i], inc_pool[i]
        raan, aop, ta = raan_pool[i], aop_pool[i], ta_pool[i]
        
        # Route state to its designated MoE cell
        try:
            model_path, dataset_path = find_expert_system(sma, ecc, inc)
        except ValueError:
            continue  # Skip gracefully if the random state falls out of the total grid bounds
            
        # Load expert lazily
        if model_path not in loaded_models:
            ds = OrbitalDataset(dataset_path)
            mdl = ResidualPredictor(input_size=8)
            mdl.load_state_dict(torch.load(model_path, weights_only=True))
            mdl.eval()
            loaded_models[model_path] = mdl
            loaded_datasets[model_path] = ds
            
        model = loaded_models[model_path]
        dataset = loaded_datasets[model_path]
        
        # Fetch the True state and AI state
        pos_true, v_true = get_ground_truth(sma, ecc, inc, raan, aop, ta, dt)
        pos_est, _ = execute_ai_step(sma, ecc, inc, raan, aop, ta, dt, model, dataset)
        
        # Compute exact discrepancies in the RIC frame
        err_r, err_i, err_c = eci_to_ric_error(pos_true, v_true, pos_est)
        
        r_errors.append(abs(err_r))
        i_errors.append(abs(err_i))
        c_errors.append(abs(err_c))
        
        results_log.append([sma, ecc, inc, err_r, err_i, err_c])
        
        # Progress indicator
        if (i + 1) % (num_samples // 10) == 0:
            print(f" Processed {i + 1}/{num_samples} orbits...")

    # =========================================================================
    # 3. STATISTICS AND LOGGING
    # =========================================================================
    os.makedirs("data", exist_ok=True)
    out_file = "data/benchmark_space_domain.csv"
    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["SMA_m", "ECC", "INC_rad", "Radial_m", "InTrack_m", "CrossTrack_m"])
        writer.writerows(results_log)

    print("-" * 80)
    print(" GLOBAL ERROR STATISTICS (Absolute values)")
    print(f" Radial      -> Mean: {np.mean(r_errors):.3f} m | Max: {np.max(r_errors):.3f} m")
    print(f" In-Track    -> Mean: {np.mean(i_errors):.3f} m | Max: {np.max(i_errors):.3f} m")
    print(f" Cross-Track -> Mean: {np.mean(c_errors):.3f} m | Max: {np.max(c_errors):.3f} m")
    print("-" * 80)
    print(f" Data saved to : {out_file}")


# =============================================================================
# Execution Block
# =============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print(" ORBITA BENCHMARK SUITE")
    print("=" * 80)
    print(" Select the validation mode:")
    print(" 1. Time-Domain (Secular Degradation over time)")
    print(" 2. Space-Domain (Global LEO Monte Carlo)")
    print(" 3. Run Both")
    
    choice = input("\n Enter choice (1/2/3): ").strip()
    
    # Time-Domain Configuration
    test_sma = R_EQ + 600e3
    test_ecc = 0.05
    test_inc = np.radians(0) 
    test_raan = 0.0
    test_aop = 0.0
    test_ta = 0.0
    test_max_tof = 4 * 3600  # 4 hours of propagation
    
    # Space-Domain Configuration
    monte_carlo_samples = 100000
    propagation_dt = 15 * 60  # 15 minutes step
    
    if choice == '1':
        run_time_domain_benchmark(
            sma=test_sma, ecc=test_ecc, inc=test_inc, 
            raan=test_raan, aop=test_aop, ta=test_ta, 
            max_tof=test_max_tof
        )
    elif choice == '2':
        run_space_domain_benchmark(
            num_samples=monte_carlo_samples, 
            dt=propagation_dt
        )
    elif choice == '3':
        run_time_domain_benchmark(
            sma=test_sma, ecc=test_ecc, inc=test_inc, 
            raan=test_raan, aop=test_aop, ta=test_ta, 
            max_tof=test_max_tof
        )
        run_space_domain_benchmark(
            num_samples=monte_carlo_samples, 
            dt=propagation_dt
        )
    else:
        print(" Invalid choice. Exiting.")