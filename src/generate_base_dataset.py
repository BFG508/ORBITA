"""
Script: generate_base_dataset.py

Description:
    Generates the dataset required to train the ORBITA neural network.
    It uses Monte Carlo uniform random sampling across user-defined orbital
    parameter boundaries and Time of Flight (TOF).

    The script propagates the initial states using both the analytical baseline
    (ESTHER) and the numerical oracle. Crucially, it saves both the inputs
    and the residual errors in Modified Equinoctial Elements (MEE) to grant
    the AI absolute mathematical immunity against classical singularities.

    Supports multiprocessing for significant speedup on multi-core CPUs.
    Each integration is independent, making this embarrassingly parallel.
"""

import csv
import os
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np

from config import (AOP_BOUNDS, MAX_TOF_SECONDS, MIN_SAFE_PERIGEE, RAAN_BOUNDS,
                    SAMPLES_PER_EXPERT, TA_BOUNDS, TOTAL_ECC_BOUNDS,
                    TOTAL_INC_BOUNDS, TOTAL_SMA_BOUNDS)
from physics.oracle import R_EQ
from physics.residuals import compute_mee_residuals


def _generate_single_sample(
    _,
    sma_bounds,
    ecc_bounds,
    inc_bounds,
    raan_bounds,
    aop_bounds,
    ta_bounds,
    tof_bounds,
):
    """
    Generates a single valid sample: random orbital state + MEE residuals.

    This function is designed for parallel execution. Each call is fully
    independent (no shared state) and uses process-local RNG seeding.

    Args:
        _ : Unused index argument (required by Pool.map).
        sma_bounds (tuple): (min, max) for semi-major axis [m].
        ecc_bounds (tuple): (min, max) for eccentricity [-].
        inc_bounds (tuple): (min, max) for inclination [rad].
        raan_bounds (tuple): (min, max) for RAAN [rad].
        aop_bounds (tuple): (min, max) for argument of periapsis [rad].
        ta_bounds (tuple): (min, max) for true anomaly [rad].
        tof_bounds (tuple): (min, max) for time of flight [s].

    Returns:
        list or None: Row of data [mee_inputs, tof, mee_residuals],
                      or None if perigee safety check fails.
    """
    sma = np.random.uniform(sma_bounds[0], sma_bounds[1])
    ecc = np.random.uniform(ecc_bounds[0], ecc_bounds[1])

    # Physical safety check: reject orbits with perigee below atmosphere
    r_periapsis = sma * (1.0 - ecc)
    if r_periapsis < MIN_SAFE_PERIGEE:
        return None

    inc = np.random.uniform(inc_bounds[0], inc_bounds[1])
    raan = np.random.uniform(raan_bounds[0], raan_bounds[1])
    aop = np.random.uniform(aop_bounds[0], aop_bounds[1])
    ta = np.random.uniform(ta_bounds[0], ta_bounds[1])
    tof = np.random.uniform(tof_bounds[0], tof_bounds[1])

    mee_inputs, mee_residuals = compute_mee_residuals(
        sma, ecc, inc, raan, aop, ta, tof
    )

    return list(mee_inputs) + [tof] + list(mee_residuals)


def generate_training_data(
    num_samples,
    output_file,
    sma_bounds,
    ecc_bounds,
    inc_bounds,
    raan_bounds,
    aop_bounds,
    ta_bounds,
    tof_bounds,
    num_workers=None,
):
    """
    Samples the parameter space, runs both orbital models, and saves the
    input states and residual errors (both in MEE) to a CSV file.

    Uses multiprocessing to distribute the independent numerical
    integrations across all available CPU cores.

    Args:
        num_samples (int): Number of Monte Carlo samples to generate.
        output_file (str): Path to the output CSV file.
        sma_bounds (tuple): (min, max) for semi-major axis [m].
        ecc_bounds (tuple): (min, max) for eccentricity [-].
        inc_bounds (tuple): (min, max) for inclination [rad].
        raan_bounds (tuple): (min, max) for RAAN [rad].
        aop_bounds (tuple): (min, max) for argument of periapsis [rad].
        ta_bounds (tuple): (min, max) for true anomaly [rad].
        tof_bounds (tuple): (min, max) for time of flight [s].
        num_workers (int): Number of parallel workers. Defaults to
                           cpu_count - 1 (leaves one core for the OS).
    """
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    if os.path.exists(output_file):
        print(
            f" [info] Skipping data generation: {output_file} already exists."
        )
        return

    print("-" * 70)
    print(" INITIATING ORBITA DATASET GENERATION")
    print(f" Target samples : {int(num_samples)}")
    print(f" Output file    : {output_file}")
    print(f" CPU workers    : {num_workers}")
    print("-" * 70)

    # Ensure the target directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Build the worker function with fixed bounds
    worker_fn = partial(
        _generate_single_sample,
        sma_bounds=sma_bounds,
        ecc_bounds=ecc_bounds,
        inc_bounds=inc_bounds,
        raan_bounds=raan_bounds,
        aop_bounds=aop_bounds,
        ta_bounds=ta_bounds,
        tof_bounds=tof_bounds,
    )

    # Over-sample to account for rejected orbits (perigee safety filter)
    oversample_factor = 1.15
    results = []
    consecutive_empty_batches = 0
    max_consecutive_empty = 1000

    with Pool(processes=num_workers) as pool:
        while len(results) < num_samples:
            remaining = int(num_samples) - len(results)
            pool_size = int(remaining * oversample_factor) + 100

            print(
                f" [info] Dispatching {pool_size} candidates"
                f" to {num_workers} workers..."
            )

            batch = pool.map(worker_fn, range(pool_size))

            # Filter out None results (rejected by perigee check)
            valid = [r for r in batch if r is not None]

            if len(valid) == 0:
                consecutive_empty_batches += 1
                if consecutive_empty_batches >= max_consecutive_empty:
                    raise RuntimeError(
                        f"Failed to generate valid orbits {max_consecutive_empty} times. Range might be physically invalid."
                    )
            else:
                consecutive_empty_batches = 0
                results.extend(valid)

            print(
                f" [info] Progress: {min(len(results), int(num_samples))}"
                f"/{int(num_samples)} valid samples collected."
            )

    # Trim to exact requested count
    results = results[: int(num_samples)]

    # Write results to CSV
    with open(output_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "p",
                "f",
                "g",
                "h",
                "k",
                "L",
                "TOF",
                "err_p",
                "err_f",
                "err_g",
                "err_h",
                "err_k",
                "err_L",
            ]
        )
        writer.writerows(results)

    print(f"\n Dataset successfully generated and saved to: {output_file}\n")


# =============================================================================
# EXECUTION BLOCK
# =============================================================================
if __name__ == "__main__":
    # 1. Define the mission-specific bounds (the "Expert" domain)
    mission_sma_bounds = TOTAL_SMA_BOUNDS
    mission_ecc_bounds = TOTAL_ECC_BOUNDS
    mission_inc_bounds = TOTAL_INC_BOUNDS

    mission_raan_bounds = RAAN_BOUNDS
    mission_aop_bounds = AOP_BOUNDS
    mission_ta_bounds = TA_BOUNDS
    mission_tof_bounds = (0.0, MAX_TOF_SECONDS)

    # 2. Dynamic filename generation
    sma_str = (
        f"{int((mission_sma_bounds[0] - R_EQ) / 1e3)}"
        f"-{int((mission_sma_bounds[1] - R_EQ) / 1e3)}"
    )
    ecc_str = f"{mission_ecc_bounds[0]:.4f}-{mission_ecc_bounds[1]:.4f}"
    inc_str = (
        f"{int(np.rad2deg(mission_inc_bounds[0]))}"
        f"-{int(np.rad2deg(mission_inc_bounds[1]))}"
    )

    filename = f"data/orbita_dataset_{sma_str}_{ecc_str}_{inc_str}.csv"

    # 3. Execute the generation
    generate_training_data(
        num_samples=SAMPLES_PER_EXPERT,
        output_file=filename,
        sma_bounds=mission_sma_bounds,
        ecc_bounds=mission_ecc_bounds,
        inc_bounds=mission_inc_bounds,
        raan_bounds=mission_raan_bounds,
        aop_bounds=mission_aop_bounds,
        ta_bounds=mission_ta_bounds,
        tof_bounds=mission_tof_bounds,
    )
