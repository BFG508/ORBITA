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

    Upgraded to support Ablation Studies: Dynamically forwards the selected
    baseline architecture to the training module across the entire grid.
"""

import argparse

import numpy as np

from config import (
    AOP_BOUNDS,
    MAX_TOF_SECONDS,
    MIN_SAFE_PERIGEE,
    RAAN_BOUNDS,
    SAMPLES_PER_EXPERT,
    TA_BOUNDS,
    TOTAL_ECC_BOUNDS,
    TOTAL_INC_BOUNDS,
    TOTAL_SMA_BOUNDS,
)
from generate_base_dataset import generate_training_data
from physics.oracle import R_EQ
from train_base import train_model


def domain_can_contain_safe_orbit(sma_bounds, ecc_bounds):
    """
    Checks whether a grid cell contains at least one orbit above the minimum
    safe perigee. If the best-case corner is invalid, random generation would
    otherwise loop until its retry guard fails.
    """
    max_sma = sma_bounds[1]
    min_ecc = ecc_bounds[0]
    return max_sma * (1.0 - min_ecc) >= MIN_SAFE_PERIGEE


def build_expert_grid(
    total_sma_bounds,
    total_ecc_bounds,
    total_inc_bounds,
    sma_splits,
    ecc_splits,
    inc_splits,
    samples_per_expert=5000,
    model_type="resnet",
):
    """
    Executes the automated pipeline to generate and train a 3D grid
    of expert models.

    Args:
        total_sma_bounds (tuple): (min, max) for the entire SMA space [m].
        total_ecc_bounds (tuple): (min, max) for the entire ECC space [-].
        total_inc_bounds (tuple): (min, max) for the entire INC space [rad].
        sma_splits (int): Number of subdivisions for the altitude domain.
        ecc_splits (int): Number of subdivisions for the eccentricity domain.
        inc_splits (int): Number of subdivisions for the inclination domain.
        samples_per_expert (int): Monte Carlo samples per grid cell.
        model_type (str): Architecture variant to train.
    """
    total_models = sma_splits * ecc_splits * inc_splits

    print("-" * 80)
    print(f" ORBITA FACTORY: BUILDING 3D GRID OF {total_models} EXPERT MODELS")
    print(f"    Architecture Variant   : {model_type.upper()}")
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

    # Fixed angular and TOF bounds for all experts
    tof_bounds = (0.0, MAX_TOF_SECONDS)

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

            ecc_str = f"{expert_ecc_min:.4f}-{expert_ecc_max:.4f}"

            # --- INNER LOOP: Traverse the Inclination (Tilt) ---
            for k in range(inc_splits):
                expert_inc_min = inc_min + (k * inc_step)
                expert_inc_max = expert_inc_min + inc_step
                expert_inc_bounds = (expert_inc_min, expert_inc_max)

                deg_min = np.degrees(expert_inc_min)
                deg_max = np.degrees(expert_inc_max)
                inc_str = f"{deg_min:.2f}-{deg_max:.2f}"

                csv_filename = (
                    f"data/orbita_dataset_{sma_str}_{ecc_str}_{inc_str}.csv"
                )

                import os

                if model_type == "tree":
                    model_filename = (
                        f"models/tree/orbita_predictor_tree_"
                        f"{sma_str}_{ecc_str}_{inc_str}.joblib"
                    )
                else:
                    model_filename = (
                        f"models/{model_type}/base/orbita_predictor_{model_type}"
                        f"_{sma_str}_{ecc_str}_{inc_str}.pth"
                    )

                print("\n" + "=" * 75)
                print(f" PROCESSING EXPERT {model_counter}/{total_models}")
                print(
                    f" Domain: Alt [{alt_min_km}-{alt_max_km} km] | "
                    f"Ecc [{ecc_str}] | Inc [{deg_min}-{deg_max} deg]"
                )
                print("=" * 75)

                try:
                    if not domain_can_contain_safe_orbit(
                        expert_sma_bounds, expert_ecc_bounds
                    ):
                        print(
                            " [info] Skipping physically invalid cell:"
                            " no orbit in this domain satisfies the minimum"
                            " safe perigee."
                        )
                        model_counter += 1
                        continue

                    if os.path.exists(csv_filename) and os.path.exists(
                        model_filename
                    ):
                        print(
                            " [info] Dataset and model already exist. Skipping."
                        )
                        model_counter += 1
                        continue

                    if os.path.exists(csv_filename):
                        print(
                            f" [step 1] Skipping data generation, {csv_filename} already exists."
                        )
                    else:
                        # --- PIPELINE STEP 1: DATA GENERATION ---
                        print(
                            f" [step 1] Generating {int(samples_per_expert)}"
                            " dataset samples..."
                        )
                        generate_training_data(
                            num_samples=int(samples_per_expert),
                            output_file=csv_filename,
                            sma_bounds=expert_sma_bounds,
                            ecc_bounds=expert_ecc_bounds,
                            inc_bounds=expert_inc_bounds,
                            raan_bounds=RAAN_BOUNDS,
                            aop_bounds=AOP_BOUNDS,
                            ta_bounds=TA_BOUNDS,
                            tof_bounds=tof_bounds,
                        )

                    if os.path.exists(model_filename):
                        print(
                            f"\n [step 2] Skipping training, {model_filename} already exists."
                        )
                    else:
                        # --- PIPELINE STEP 2: MODEL TRAINING ---
                        print(
                            f"\n [step 2] Training neural network"
                            f" (Architecture: {model_type.upper()})..."
                        )
                        target_model_dir = (
                            "models/tree"
                            if model_type == "tree"
                            else f"models/{model_type}/base"
                        )
                        train_model(
                            csv_file=csv_filename,
                            model_type=model_type,
                            model_dir=target_model_dir,
                        )

                    print(f"\n [info] EXPERT {model_counter} COMPLETE.")
                    model_counter += 1
                except RuntimeError as e:
                    print(f"\n [warn] Skipping EXPERT {model_counter}: {e}")
                    model_counter += 1
                    continue

    print("-" * 80)
    print(" 3D GRID FLEET GENERATION FINISHED SUCCESSFULLY.")
    print("-" * 80)


# =============================================================================
# EXECUTION BLOCK
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ORBITA 3D MoE Grid Orchestrator"
    )

    parser.add_argument(
        "--model_type",
        type=str,
        choices=["resnet", "linear", "mlp", "lstm", "tree"],
        default="resnet",
        help="Select the neural architecture to train across the grid.",
    )
    parser.add_argument(
        "--sma_splits",
        type=int,
        default=5,
        help="Number of altitude-domain splits (default: 5).",
    )
    parser.add_argument(
        "--ecc_splits",
        type=int,
        default=4,
        help="Number of eccentricity-domain splits (default: 4).",
    )
    parser.add_argument(
        "--inc_splits",
        type=int,
        default=2,
        help="Number of inclination-domain splits (default: 2).",
    )
    parser.add_argument(
        "--samples_per_expert",
        type=int,
        default=SAMPLES_PER_EXPERT,
        help="Monte Carlo samples per valid expert cell.",
    )

    args = parser.parse_args()

    # Build a sma_splits x ecc_splits x inc_splits grid
    build_expert_grid(
        total_sma_bounds=TOTAL_SMA_BOUNDS,
        total_ecc_bounds=TOTAL_ECC_BOUNDS,
        total_inc_bounds=TOTAL_INC_BOUNDS,
        sma_splits=args.sma_splits,
        ecc_splits=args.ecc_splits,
        inc_splits=args.inc_splits,
        samples_per_expert=args.samples_per_expert,
        model_type=args.model_type,
    )
