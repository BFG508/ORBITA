"""
Script: benchmark.py

Description:
    Flight-grade aerospace benchmark suite for the ORBITA framework.
    Validates the 3D Mixture of Experts (MoE) Grey-Box AI architecture against
    the high-precision Numerical Oracle (Cowell's formulation via LSODA).

    This suite provides a comprehensive statistical and temporal analysis of
    the network's predictive capabilities across the Low Earth Orbit (LEO) regime,
    isolating error components into the physically meaningful RIC coordinate system.

    Upgraded to support Ablation Studies: Can load and benchmark different
    baseline models (ResNet, Linear, MLP, LSTM, Tree) to empirically
    demonstrate the superiority of the Grey-Box approach.

Validation Modes:
    1. Time-Domain (Secular Degradation): Evaluates the long-term numerical
       stability of the hybrid model by propagating a target initial state
       over continuous hours. Designed to stress-test error accumulation.
    2. Space-Domain (Global LEO Monte Carlo): Evaluates the generalization
       capacity of the MoE grid by generating thousands of randomized
       orbital states and propagating them for a single time step.
"""

import argparse
import csv
import logging
import os
import time

import joblib
import numpy as np
import torch

logging.getLogger("codecarbon").disabled = True
from codecarbon import EmissionsTracker

from generate_base_dataset import MIN_SAFE_PERIGEE
from ml.architecture import (
    LinearBaseline,
    LSTMPredictor,
    MLPPredictor,
    ResidualPredictor,
    TreeBaseline,
)
from ml.dataset import OrbitalDataset
from physics.analytical import compute_general_solution
from physics.kepler import get_keplerian
from physics.oracle import (
    J2,
    J3,
    MU,
    R_EQ,
    coe_to_eci,
    coe_to_mee,
    eci_to_coe,
    get_ground_truth,
    mee_to_coe,
)
from simulate_mission import find_expert_system


def _log_benchmark_metrics(model_type, mode, wall_time, num_samples):
    """
    Appends a row to data/metrics_benchmark.csv with
    the wall-clock inference time collected during benchmark.

    Args:
        model_type (str): Architecture identifier.
        mode (str): 'time_domain' or 'space_domain'.
        wall_time (float): Total wall-clock time [s].
        num_samples (int): Number of samples evaluated.
    """
    os.makedirs("data", exist_ok=True)
    metrics_file = "data/metrics_benchmark.csv"
    header = [
        "architecture",
        "mode",
        "wall_time_s",
        "num_samples",
    ]

    write_header = not os.path.exists(metrics_file)
    with open(metrics_file, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow([model_type, mode, f"{wall_time:.2f}", num_samples])

    print(
        f" [metrics] Logged to {metrics_file}: "
        f"{model_type} | {mode} | {wall_time:.2f}s | "
        f"{num_samples} samples"
    )


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
    Executes a single Grey-Box propagation step using the selected AI architecture.

    This function acts as the core of the ORBITA framework. It integrates the
    deterministic analytical baseline (ESTHER) with the neural network to predict
    and correct high-frequency orbital perturbations over a given Time of Flight (dt).

    Args:
        sma (float): Semi-major axis [m].
        ecc (float): Eccentricity [-].
        inc (float): Inclination [rad].
        raan (float): Right Ascension of the Ascending Node [rad].
        aop (float): Argument of Perigee [rad].
        ta (float): True Anomaly [rad].
        dt (float): Time step / Time of Flight for the propagation [s].
        model (nn.Module): Loaded PyTorch expert model for the current grid.
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
    raw_input = np.array(
        [p_in, f_in, g_in, h_in, k_in, np.sin(l_in), np.cos(l_in), dt]
    )
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

    # Apply the AI-predicted residuals (The actual Grey-Box fusion)
    mee_corrected = mee_esther + pred_mee

    # Transform the final corrected state back to ECI for downstream use
    coe_corrected = mee_to_coe(*mee_corrected)
    pos_est, vel_est = coe_to_eci(MU, *coe_corrected)

    return pos_est, vel_est


def get_model_instance(model_type):
    """
    Instantiates the selected neural network architecture.

    Args:
        model_type (str): The architecture variant requested.

    Returns:
        nn.Module: The uninitialized PyTorch model.
    """
    model_type = model_type.lower()
    if model_type == "resnet":
        return ResidualPredictor(input_size=8)
    elif model_type == "linear":
        return LinearBaseline(input_size=8)
    elif model_type == "mlp":
        return MLPPredictor(input_size=8)
    elif model_type == "lstm":
        return LSTMPredictor(input_size=8)
    elif model_type == "tree":
        return TreeBaseline(input_size=8)
    else:
        raise ValueError(f"Unsupported model_type: '{model_type}'")


def run_time_domain_benchmark(
    sma, ecc, inc, raan, aop, ta, max_tof, case_name="default", model_type=None
):
    """
    Execute the secular degradation test for a specific target orbit.

    This function propagates a single initial orbital state over a designated
    maximum Time of Flight (TOF) to evaluate the accumulated secular error of
    the selected AI model against the Numerical Oracle.

    Args:
        sma (float): Semi-major axis [m].
        ecc (float): Eccentricity [-].
        inc (float): Inclination [rad].
        raan (float): Right Ascension of the Ascending Node [rad].
        aop (float): Argument of Perigee [rad].
        ta (float): True Anomaly [rad].
        max_tof (float): Maximum Time of Flight for the simulation [s].
        case_name (str): Identifier for the output logs. Defaults to "default".
        model_type (str): The architecture variant to test.

    Returns:
        tuple: A tuple containing:
            - results_log (list[list] or None): The logged simulation data.
            - total_oracle_time (float): Computation time used by the Oracle [s].
            - total_ai_time (float): Computation time used by the AI model [s].
    """
    model_str = model_type.upper() if model_type else "BEST AVAILABLE"
    print(f"\n{'=' * 80}")
    print(
        f" TIME-DOMAIN BENCHMARK: SECULAR DEGRADATION ({case_name.upper()}) | ARCH: {model_str}"
    )
    print("=" * 80)

    # 0. Safety Check: Filter out subterranean or immediate re-entry orbits
    if sma * (1.0 - ecc) < MIN_SAFE_PERIGEE:
        print(" [error] Aborting benchmark. Target orbit is unsafe.")
        return None, 0.0, 0.0

    # Generate the time grid (15-minute intervals)
    time_steps = np.arange(0, max_tof + 1, 15 * 60)

    # Locate the correct Mixture of Experts (MoE) cell for the current state
    try:
        expert_model_path, dataset_path = find_expert_system(
            sma=sma, ecc=ecc, inc=inc, target_model_type=model_type
        )
    except ValueError as e:
        print(f" [error] Routing failure: {e}")
        return None, 0.0, 0.0

    # Deduce the exact model file based on the selected architecture type
    if model_type is not None:
        dataset_filename = os.path.basename(dataset_path)
        if model_type == "tree":
            model_filename = dataset_filename.replace(
                "dataset", f"predictor_{model_type}"
            ).replace(".csv", ".joblib")
        else:
            model_filename = dataset_filename.replace(
                "dataset", f"predictor_{model_type}"
            ).replace(".csv", ".pth")
        model_path = os.path.join("models", model_filename)
        current_model_type = model_type
    else:
        model_path = expert_model_path
        basename = os.path.basename(expert_model_path)
        params_str = (
            basename.replace("orbita_predictor_", "")
            .replace(".pth", "")
            .replace(".joblib", "")
            .replace("_finetuned", "")
        )
        parts = params_str.split("_")
        if len(parts) == 4:
            current_model_type = parts[0]
        else:
            current_model_type = "resnet"

    if not os.path.exists(model_path):
        print(
            f" [error] Model weights not found for this domain: {model_path}"
        )
        return None, 0.0, 0.0

    # Load the expert model and its corresponding normalization
    dataset = OrbitalDataset(dataset_path)
    if current_model_type == "tree":
        model = joblib.load(model_path)
    else:
        model = get_model_instance(current_model_type)
        model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    total_oracle_time = 0.0
    total_ai_time = 0.0
    results_log = []

    # Initialize the tracking variables for the numerical integration step
    curr_sma, curr_ecc, curr_inc = sma, ecc, inc
    curr_raan, curr_aop, curr_ta = raan, aop, ta
    prev_tof = 0.0

    # Print the table header for real-time console monitoring
    print(
        f" {'Time (s)':<10} | {'Radial (m)':<12} | "
        f"{'In-Track (m)':<14} | {'Cross-Track (m)':<15}"
    )
    print("-" * 80)

    # =========================================================================
    # 1. PROPAGATION LOOP
    # =========================================================================
    for tof in time_steps:
        # Skip the initial condition step (dt = 0)
        if tof == 0:
            continue

        dt = tof - prev_tof

        # --- Oracle Ground Truth ---
        start_time = time.perf_counter()
        pos_true, v_true = get_ground_truth(sma, ecc, inc, raan, aop, ta, tof)
        total_oracle_time += time.perf_counter() - start_time

        # --- AI Estimate ---
        start_time = time.perf_counter()
        pos_est, vel_est = execute_ai_step(
            curr_sma,
            curr_ecc,
            curr_inc,
            curr_raan,
            curr_aop,
            curr_ta,
            dt,
            model,
            dataset,
        )
        total_ai_time += time.perf_counter() - start_time

        # --- Error Calculation ---
        # Transform the ECI error vector into the local RIC coordinate frame
        ric_error = eci_to_ric_error(pos_true, v_true, pos_est)
        err_r, err_i, err_c = ric_error

        print(
            f" {tof:<10.1f} | {err_r:>10.2f}   | "
            f"{err_i:>12.2f}   | {err_c:>13.2f}"
        )

        # Prepend the Case ID to the row for downstream batch sorting
        results_log.append([case_name, tof, err_r, err_i, err_c])

        # Update current orbital elements for the next recursive step
        curr_sma, curr_ecc, curr_inc, curr_raan, curr_aop, curr_ta = (
            eci_to_coe(MU, pos_est, vel_est)
        )
        prev_tof = tof

    # =========================================================================
    # 2. PERFORMANCE SUMMARY
    # =========================================================================
    print("-" * 80)
    print(
        f" Oracle Time: {total_oracle_time * 1000:.2f} ms | "
        f"AI Time: {total_ai_time * 1000:.2f} ms"
    )

    return results_log, total_oracle_time, total_ai_time


def run_space_domain_benchmark(num_samples=1000, dt=900, model_type=None):
    """
    Executes the global Monte Carlo test across the entire LEO parameter space.

    Evaluates the generalization capabilities of the selected model architecture
    by generating thousands of random initial states, ensuring they meet the
    safe perigee criteria, and propagating them for a single time step.

    Args:
        num_samples (int): Total number of random orbital states to evaluate.
        dt (float): Time of Flight for the single-step propagation [s].
        model_type (str): The architecture variant to benchmark.
    """
    model_str = model_type.upper() if model_type else "BEST AVAILABLE"
    print("\n" + "=" * 80)
    print(
        f" 🌐 SPACE-DOMAIN BENCHMARK: GLOBAL LEO COVERAGE | ARCH: {model_str}"
    )
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
            expert_model_path, dataset_path = find_expert_system(
                sma, ecc, inc, target_model_type=model_type
            )
        except ValueError:
            continue  # Skip gracefully if the random state falls out of the total grid bounds

        # Deduce the exact model file based on the architecture
        if model_type is not None:
            dataset_filename = os.path.basename(dataset_path)
            if model_type == "tree":
                model_filename = dataset_filename.replace(
                    "dataset", f"predictor_{model_type}"
                ).replace(".csv", ".joblib")
            else:
                model_filename = dataset_filename.replace(
                    "dataset", f"predictor_{model_type}"
                ).replace(".csv", ".pth")
            model_path = os.path.join("models", model_filename)
            current_model_type = model_type
        else:
            model_path = expert_model_path
            basename = os.path.basename(expert_model_path)
            params_str = (
                basename.replace("orbita_predictor_", "")
                .replace(".pth", "")
                .replace(".joblib", "")
                .replace("_finetuned", "")
            )
            parts = params_str.split("_")
            if len(parts) == 4:
                current_model_type = parts[0]
            else:
                current_model_type = "resnet"

        if not os.path.exists(model_path):
            continue

        # Load expert lazily
        if model_path not in loaded_models:
            ds = OrbitalDataset(dataset_path)
            if current_model_type == "tree":
                mdl = joblib.load(model_path)
            else:
                mdl = get_model_instance(current_model_type)
                mdl.load_state_dict(torch.load(model_path, weights_only=True))
            mdl.eval()
            loaded_models[model_path] = mdl
            loaded_datasets[model_path] = ds

        model = loaded_models[model_path]
        dataset = loaded_datasets[model_path]

        # Fetch the True state and AI state
        pos_true, v_true = get_ground_truth(sma, ecc, inc, raan, aop, ta, dt)
        pos_est, _ = execute_ai_step(
            sma, ecc, inc, raan, aop, ta, dt, model, dataset
        )

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
    out_file = f"data/benchmark_space_domain_{model_type if model_type else 'best'}.csv"

    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "SMA_m",
                "ECC",
                "INC_rad",
                "Radial_m",
                "InTrack_m",
                "CrossTrack_m",
            ]
        )
        writer.writerows(results_log)

    print("-" * 80)
    print(" GLOBAL ERROR STATISTICS (Absolute values)")
    print(
        f" Radial      -> Mean: {np.mean(r_errors):.3f} m | Max: {np.max(r_errors):.3f} m"
    )
    print(
        f" In-Track    -> Mean: {np.mean(i_errors):.3f} m | Max: {np.max(i_errors):.3f} m"
    )
    print(
        f" Cross-Track -> Mean: {np.mean(c_errors):.3f} m | Max: {np.max(c_errors):.3f} m"
    )
    print("-" * 80)
    print(f" Data saved to : {out_file}")

    return len(results_log)


def generate_random_test_cases(num_cases, max_tof=4 * 3600):
    """
    Generates a list of randomized, safe LEO orbital states for
    automated time-domain secular degradation testing.

    Args:
        num_cases (int): Number of random orbital states to generate.
        max_tof (float): Maximum Time of Flight for each case [s].

    Returns:
        list of dict: A list containing configuration dictionaries for each test case.
    """
    print(f" Generating {num_cases} randomized Time-Domain test cases...")
    random_cases = []

    # We use a while loop to reject subterranean orbits dynamically
    while len(random_cases) < num_cases:
        sma_cand = np.random.uniform(R_EQ + 300e3, R_EQ + 2000e3)
        ecc_cand = np.random.uniform(0.0, 0.1)

        # Apply the safe perigee filter
        if sma_cand * (1.0 - ecc_cand) >= MIN_SAFE_PERIGEE:
            case = {
                "name": f"random_LEO_{len(random_cases) + 1:02d}",
                "sma": sma_cand,
                "ecc": ecc_cand,
                "inc": np.random.uniform(0.0, np.radians(90.0)),
                "raan": np.random.uniform(0.0, 2 * np.pi),
                "aop": np.random.uniform(0.0, 2 * np.pi),
                "ta": np.random.uniform(0.0, 2 * np.pi),
                "max_tof": max_tof,
            }
            random_cases.append(case)

    return random_cases


def run_time_domain_batch(test_cases, output_filename=None, model_type=None):
    """
    Manage the execution of multiple time-domain benchmark cases.

    This function iterates through a list of generated orbital test cases,
    executes the secular degradation benchmark for each, and streams all
    results into a single consolidated CSV file.

    Args:
        test_cases (list[dict]): A list of dictionaries containing the orbital parameters.
        output_filename (str, optional): The target file path for the unified CSV output.
        model_type (str): The architecture variant to test.
    """
    if output_filename is None:
        model_str = model_type if model_type else "best"
        output_filename = f"data/benchmark_time_domain_{model_str}.csv"

    # Ensure the output directory exists before attempting to write
    output_dir = os.path.dirname(output_filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    total_batch_oracle_time = 0.0
    total_batch_ai_time = 0.0
    successful_cases = 0

    # Open the unified CSV file in write mode to stream data continuously
    with open(output_filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        # Write the master header including the Case_ID for downstream sorting
        writer.writerow(
            ["Case_ID", "Time_s", "Radial_m", "InTrack_m", "CrossTrack_m"]
        )

        # Process and stream results sequentially to maintain low memory usage
        for case in test_cases:
            results, t_oracle, t_ai = run_time_domain_benchmark(
                sma=case["sma"],
                ecc=case["ecc"],
                inc=case["inc"],
                raan=case["raan"],
                aop=case["aop"],
                ta=case["ta"],
                max_tof=case["max_tof"],
                case_name=case["name"],
                model_type=model_type,
            )

            # Only write to the file if the orbit was safe and processed correctly
            if results is not None:
                writer.writerows(results)
                successful_cases += 1
                total_batch_oracle_time += t_oracle
                total_batch_ai_time += t_ai

    # Output the final performance summary to the console
    model_str = model_type.upper() if model_type else "BEST"
    print(f"\n{'=' * 80}")
    print(f" UNIFIED TIME-DOMAIN BENCHMARK COMPLETE ({model_str})")
    print("=" * 80)
    print(f" Total Orbits Simulated : {successful_cases}/{len(test_cases)}")
    print(f" Total Oracle Time      : {total_batch_oracle_time:.2f} s")
    print(f" Total AI Time          : {total_batch_ai_time:.2f} s")

    if total_batch_ai_time > 0:
        speedup = total_batch_oracle_time / total_batch_ai_time
        print(f" Global Speedup         : {speedup:.2f}x faster")

    print(f" Data stream saved to   : {output_filename}")

    return successful_cases, total_batch_ai_time


# =============================================================================
# EXECUTION BLOCK
# =============================================================================
if __name__ == "__main__":
    # Setup argparse for Ablation Study execution
    parser = argparse.ArgumentParser(description="ORBITA Benchmark Suite")

    parser.add_argument(
        "--model_type",
        type=str,
        choices=["resnet", "linear", "mlp", "lstm", "tree"],
        default="resnet",
        help="Select the neural architecture to benchmark.",
    )

    # Argument to bypass the interactive terminal prompt
    parser.add_argument(
        "--mode_choice",
        type=str,
        choices=["1", "2", "3"],
        default=None,
        help="Validation mode (1: Time, 2: Space, 3: Both). Bypasses input().",
    )

    args = parser.parse_args()

    model_str = (
        args.model_type.upper() if args.model_type else "BEST AVAILABLE"
    )
    print("=" * 80)
    print(f" ORBITA BENCHMARK SUITE | ARCHITECTURE: {model_str}")
    print("=" * 80)

    # Bypass interactive prompt if the argument was passed via CLI
    if args.mode_choice:
        choice = args.mode_choice
        print(f" Automated mode active. Selected validation mode: {choice}")
    else:
        # Fallback to interactive mode if no argument is provided
        print(" Select the validation mode:")
        print(" 1. Time-Domain (Automated Secular Degradation Battery)")
        print(" 2. Space-Domain (Global LEO Monte Carlo)")
        print(" 3. Run Both")
        choice = input("\n Enter choice (1/2/3): ").strip()

    # Time-Domain Configuration
    n_cases = 10000
    max_tof = 4 * 3600

    # Space-Domain Configuration
    monte_carlo_samples = 100000
    propagation_dt = 15 * 60  # 15 minutes step

    if choice in ["1", "3"]:
        test_cases = generate_random_test_cases(
            num_cases=n_cases, max_tof=max_tof
        )
        tracker = EmissionsTracker(
            project_name=(f"benchmark_time_domain_{args.model_type}"),
            log_level="error",
            output_dir="data",
        )
        tracker.start()
        t0 = time.time()
        n_ok, ai_time = run_time_domain_batch(
            test_cases=test_cases, model_type=args.model_type
        )
        wall = time.time() - t0
        tracker.stop()
        _log_benchmark_metrics(
            args.model_type or "best",
            "time_domain",
            wall,
            n_ok,
        )

    if choice in ["2", "3"]:
        tracker = EmissionsTracker(
            project_name=(f"benchmark_space_domain_{args.model_type}"),
            log_level="error",
            output_dir="data",
        )
        tracker.start()
        t0 = time.time()
        n_evaluated = run_space_domain_benchmark(
            num_samples=monte_carlo_samples,
            dt=propagation_dt,
            model_type=args.model_type,
        )
        wall = time.time() - t0
        tracker.stop()
        _log_benchmark_metrics(
            args.model_type or "best",
            "space_domain",
            wall,
            n_evaluated,
        )

    if choice not in ["1", "2", "3"]:
        print(" Invalid choice. Exiting.")
