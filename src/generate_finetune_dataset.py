"""
Module: generate_finetune_dataset.py

Description:
    Implements Pool-based Active Learning via Uncertainty Sampling for the 
    ORBITA framework.
    
    This script generates a massive, unlabelled pool of random orbital states 
    within a specific expert's domain. It evaluates these states using the 
    pre-trained MoE expert's MC-Dropout configuration to extract epistemic 
    uncertainty. It then selects the most uncertain cases ('Hard Cases') 
    alongside a random subset of well-understood cases ('Replay Buffer') 
    to prevent catastrophic forgetting.
    
    Finally, the Numerical Oracle is invoked to compute the exact ground truth 
    residuals exclusively for the selected samples, yielding a highly optimized 
    dataset for Continual Learning.
"""

import os
import csv
import time

import numpy as np
import torch

from physics.analytical import compute_general_solution
from physics.oracle import coe_to_eci, eci_to_coe, coe_to_mee, get_ground_truth
from physics.kepler import get_keplerian
from ml.architecture import ResidualPredictor
from ml.dataset import OrbitalDataset

from physics.oracle import MU, J2, J3, R_EQ
from generate_base_dataset import MIN_SAFE_PERIGEE


def generate_active_learning_dataset(
    model_path, 
    original_csv_path, 
    output_csv_path, 
    sma_bounds, 
    ecc_bounds, 
    inc_bounds, 
    pool_size=100000, 
    hard_cases=10000, 
    replay_cases=5000
):
    """
    Executes the uncertainty sampling algorithm to mine critical error regions 
    and construct a targeted fine-tuning dataset.
    
    Args:
        model_path (str): Path to the baseline PyTorch model (.pth).
        original_csv_path (str): Path to the original dataset (for normalization stats).
        output_csv_path (str): Destination path for the generated active learning dataset.
        sma_bounds (tuple): Semi-major axis domain limits [m].
        ecc_bounds (tuple): Eccentricity domain limits [-].
        inc_bounds (tuple): Inclination domain limits [rad].
        pool_size (int): Number of unlabelled candidate states to generate.
        hard_cases (int): Number of maximum-uncertainty samples to select.
        replay_cases (int): Number of low-uncertainty samples to retain.
    """
    print("=" * 80)
    print(" ORBITA ACTIVE LEARNING: HARD CASE MINING")
    print("=" * 80)
    
    # =========================================================================
    # 1. INITIALIZATION & NORMALIZATION LOAD
    # =========================================================================
    print(" [1/5] Loading AI Model and normalization statistics...")
    dataset = OrbitalDataset(original_csv_path)
    model = ResidualPredictor(input_size=8)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    # =========================================================================
    # 2. VECTORIZED SAFE POOL GENERATION
    # =========================================================================
    print(f" [2/5] Generating unlabelled candidate pool ({pool_size} samples)...")
    
    valid_samples = 0
    
    sma_list, ecc_list = [], []
    
    # Generate parameters ensuring the physical perigee safety constraint
    while valid_samples < pool_size:
        sma_cand = np.random.uniform(sma_bounds[0], sma_bounds[1], pool_size)
        ecc_cand = np.random.uniform(ecc_bounds[0], ecc_bounds[1], pool_size)
        
        valid_mask = (sma_cand * (1.0 - ecc_cand)) >= MIN_SAFE_PERIGEE
        sma_valid = sma_cand[valid_mask]
        ecc_valid = ecc_cand[valid_mask]
        
        needed = pool_size - valid_samples
        sma_list.extend(sma_valid[:needed])
        ecc_list.extend(ecc_valid[:needed])
        valid_samples += len(sma_valid[:needed])

    sma_pool = np.array(sma_list)
    ecc_pool = np.array(ecc_list)
    inc_pool = np.random.uniform(inc_bounds[0], inc_bounds[1], pool_size)
    raan_pool = np.random.uniform(0.0, 2 * np.pi, pool_size)
    aop_pool = np.random.uniform(0.0, 2 * np.pi, pool_size)
    ta_pool = np.random.uniform(0.0, 2 * np.pi, pool_size)
    tof_pool = np.random.uniform(0.0, 30.0 * 60.0, pool_size)

    # =========================================================================
    # 3. UNCERTAINTY ESTIMATION (MC-DROPOUT)
    # =========================================================================
    print(" [3/5] AI is evaluating epistemic uncertainty (MC-Dropout)...")
    uncertainty_scores = np.zeros(pool_size)
    
    batch_size = 2000
    for i in range(0, pool_size, batch_size):
        end_idx = min(i + batch_size, pool_size)
        
        batch_mee = np.array([
            coe_to_mee(
                sma_pool[j], ecc_pool[j], inc_pool[j], 
                raan_pool[j], aop_pool[j], ta_pool[j]
            )
            for j in range(i, end_idx)
        ])
        
        p_in, f_in, g_in, h_in, k_in, l_in = batch_mee.T
        batch_tof = tof_pool[i:end_idx]
        
        x_raw = np.column_stack([
            p_in, f_in, g_in, h_in, k_in, np.sin(l_in), np.cos(l_in), batch_tof
        ])
        
        x_norm = (x_raw - dataset.x_mean) / dataset.x_std
        x_tensor = torch.tensor(x_norm, dtype=torch.float32)
        
        # Extract purely the standard deviation (spread) across stochastic passes
        _, std_pred = model.predict_with_uncertainty(x_tensor, num_samples=30)
        
        # Collapse uncertainty tensor into a scalar score via Euclidean norm
        batch_uncertainty = torch.norm(std_pred, dim=1).numpy()
        uncertainty_scores[i:end_idx] = batch_uncertainty

    # =========================================================================
    # 4. SAMPLING: HARD CASES + REPLAY BUFFER
    # =========================================================================
    print(f" [4/5] Selecting {hard_cases} Hard Cases and {replay_cases} Replay Buffer cases...")
    
    sorted_indices = np.argsort(uncertainty_scores)[::-1]  # Descending order
    hard_indices = sorted_indices[:hard_cases]
    
    # Replay buffer: random sub-selection from the remaining highly confident pool
    safe_indices = sorted_indices[hard_cases:]
    replay_indices = np.random.choice(safe_indices, replay_cases, replace=False)
    
    final_indices = np.concatenate([hard_indices, replay_indices])
    np.random.shuffle(final_indices)  # Shuffle to prevent temporal training bias
    
    # =========================================================================
    # 5. GROUND TRUTH COMPUTATION (NUMERICAL ORACLE)
    # =========================================================================
    total_selected = len(final_indices)
    print(f" [5/5] Calling Numerical Oracle to compute physical truth for {total_selected} cases...")
    
    results = []
    for idx, orig_idx in enumerate(final_indices):
        sma = sma_pool[orig_idx]
        ecc = ecc_pool[orig_idx]
        inc = inc_pool[orig_idx]
        raan = raan_pool[orig_idx]
        aop = aop_pool[orig_idx]
        ta = ta_pool[orig_idx]
        tof = tof_pool[orig_idx]
        
        # A. Numerical Truth (Cowell's Formulation)
        pos_true, vel_true = get_ground_truth(sma, ecc, inc, raan, aop, ta, tof)
        coe_true = eci_to_coe(MU, pos_true, vel_true)
        mee_true = coe_to_mee(*coe_true)
        
        # B. Analytical Baseline (ESTHER + Kepler)
        r0, v0 = coe_to_eci(MU, sma, ecc, inc, raan, aop, ta)
        x0_pert = np.zeros(6)
        n_mean = np.sqrt(MU / (sma**3))
        
        delta_pos_esther, delta_vel_esther = compute_general_solution(
            J2, J3, R_EQ, sma, ecc, inc, raan, aop, ta, x0_pert, n_mean, tof
        )
        pos_kep, vel_kep = get_keplerian(MU, r0, v0, tof)
        
        pos_esther = pos_kep + delta_pos_esther
        vel_esther = vel_kep + delta_vel_esther
        
        coe_esther = eci_to_coe(MU, pos_esther, vel_esther)
        mee_esther = coe_to_mee(*coe_esther)
        
        # C. Calculate Target Residuals (Error = Truth - Analytical)
        y_residual = np.array(mee_true) - np.array(mee_esther)
        
        # D. State Inputs (Singularity-Free MEE)
        mee_0 = coe_to_mee(sma, ecc, inc, raan, aop, ta)
        row = list(mee_0) + [tof] + list(y_residual)
        results.append(row)
        
        if (idx + 1) % 1000 == 0:
            print(f"       Processed {idx + 1}/{total_selected}")

    # =========================================================================
    # 6. EXPORT
    # =========================================================================
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'p', 'f', 'g', 'h', 'k', 'L', 'TOF', 
            'err_p', 'err_f', 'err_g', 'err_h', 'err_k', 'err_L'
        ])
        writer.writerows(results)
        
    print("=" * 80)
    print(f" ACTIVE LEARNING DATASET SAVED TO: {output_csv_path}")
    print("=" * 80)


# =============================================================================
# Execution Block
# =============================================================================
if __name__ == "__main__":
    # 1. Target Expert Domain Bounds
    SMA_BOUNDS = (R_EQ + 300e3, R_EQ + 725e3)
    ECC_BOUNDS = (0.075, 0.1)
    INC_BOUNDS = (0.0, np.radians(90.0))
    
    # 2. Dynamic File Routing
    sma_str = f"{int((SMA_BOUNDS[0] - R_EQ) / 1e3)}-{int((SMA_BOUNDS[1] - R_EQ) / 1e3)}"
    ecc_str = f"{ECC_BOUNDS[0]:.4f}-{ECC_BOUNDS[1]:.4f}"
    inc_str = f"{int(np.degrees(INC_BOUNDS[0]))}-{int(np.degrees(INC_BOUNDS[1]))}"
    
    domain_identifier = f"{sma_str}_{ecc_str}_{inc_str}"
    
    base_model = f"models/orbita_predictor_{domain_identifier}.pth"
    base_csv   = f"data/orbita_dataset_{domain_identifier}.csv"
    output_csv = f"data/orbita_finetune_{domain_identifier}.csv"
    
    # 3. Execution Execution
    generate_active_learning_dataset(
        model_path=base_model,
        original_csv_path=base_csv,
        output_csv_path=output_csv,
        sma_bounds=SMA_BOUNDS,
        ecc_bounds=ECC_BOUNDS,
        inc_bounds=INC_BOUNDS,
        pool_size=100000,    # Total candidate pool evaluation
        hard_cases=5000,     # Extraction of highest uncertainty vectors
        replay_cases=15000   # Retention of nominal vectors to mitigate amnesia
    )