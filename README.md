# O.R.B.I.T.A. 🛰️
**O**racle-guided **R**esidual-**B**ased **I**ntelligent **T**rajectory **A**lgorithm

A hybrid orbital mechanics framework developed to overcome the secular degradation of analytical orbital propagation models. Building upon the foundational analytical solutions derived in **[ESTHER](https://github.com/BFG508/ESTHER)**, ORBITA introduces a Machine Learning Active Learning loop to compensate for residual errors over time. This project aims to achieve high-fidelity orbit propagation (perturbed by Earth's $J_2$ and $J_3$ zonal harmonics) with the drastically reduced computational cost required for Onboard Computers (OBCs).

## 🚀 Features
* **Grey-Box Modeling Approach**: Synergizes the execution speed of a deterministic analytical model (white-box) with the adaptive error-correction capabilities of deep learning (black-box).
* **Singularity-Free Formulation**: Operates internally using Modified Equinoctial Elements (MEE), granting the AI absolute mathematical immunity against classical orbital singularities (e.g., perfectly circular or equatorial orbits).
* **Deep Residual Architecture (ResNet)**: Utilizes a deep network with skip-connections to learn the extreme, high-frequency non-linearities of orbital perturbations at centimetric precision without suffering from vanishing gradients.
* **Ablation Study with Multi-Architecture Support**: Supports comparative analysis across five architectures — ResNet, MLP, LSTM, Linear, and Decision Tree (scikit-learn) — to empirically validate architectural choices.
* **3D Mixture of Experts (MoE)**: Divides the vast orbital parameter space (Altitude, Eccentricity, Inclination) into a hyper-specialized 3D grid (38 valid cells). The framework automatically trains and dynamically routes queries to local "expert" neural networks.
* **Active Learning with "GPS Reset"**: Implements epistemic uncertainty estimation via Monte Carlo Dropout. If the neural network's uncertainty breaches a strict safety threshold, it triggers a "GPS Reset", calling the computationally expensive Numerical Oracle to wipe out any accumulated along-track error before handing control back to the AI.
* **Autonomous Continual Learning (Fine-Tuning)**: Features a dedicated active-learning pipeline that automatically mines "hard cases" using uncertainty sampling. It seamlessly upgrades weak expert models using a microscopic learning rate and a replay buffer, preventing catastrophic forgetting.
* **K-Fold Cross-Validation**: A dedicated script (`train_cv.py`) performs statistically rigorous K-Fold validation across all architectures, exporting results to `data/metrics/metrics_cv.csv`.
* **Computational Resource Monitoring**: Automated tracking of wall-clock training time, inference latency, and model file size (MB) for transparent architecture comparison.
* **Green AI / Carbon Footprint Estimation**: Integrates CodeCarbon to automatically measure the energy consumed (kWh) and CO₂ emissions (gCO₂eq) of every training and benchmarking campaign.
* **Flight-Grade & OBC-Optimized**: Written in strict PEP-8 compliant Python. Designed specifically to minimize continuous CPU load and memory footprint, validated by a built-in millisecond-level RIC (Radial, In-Track, Cross-Track) benchmarking suite.

## 🛠️ Technology Stack
Unlike its predecessor ESTHER, which was developed in MATLAB, ORBITA is built entirely in **Python** to leverage the modern Machine Learning ecosystem.
* **Core Mathematics & Physics**: `NumPy` and `SciPy` (for high-precision `LSODA` ODE solving).
* **Machine Learning**: `PyTorch` (Deep Residual Networks, MC-Dropout inference, ReduceLROnPlateau scheduling, and dataset management).
* **Classical ML Baselines**: `scikit-learn` (DecisionTreeRegressor for ablation study comparisons).
* **Green AI**: `CodeCarbon` (automated energy consumption and CO₂ emissions tracking per architecture).
* **Training Diagnostics**: `TensorBoard` (real-time loss/LR tracking and ablation study comparison).
* **Visualization & Analytics**: `Matplotlib` and `Seaborn` (CDF analysis, RIC error heatmaps, and comparative barplots for training time, model size, inference latency, energy, and cross-validation).
* **Testing**: `pytest` (88 automated tests covering coordinate transforms, architectures, residuals, configuration, visualization, audit, and naming standards).

## 📂 Repository Structure
* `/data` - Standardized workspace datasets, benchmark results, and metrics:
  * `/datasets/training/` - Monte Carlo base datasets (`orbita_dataset_*.csv`).
  * `/datasets/finetuning/{arch}/` - Active-learning fine-tuning datasets per architecture.
  * `/benchmarks/ablation_global/` - Global multi-architecture spatial and temporal benchmark CSVs.
  * `/benchmarks/ablation_resnet/` - ResNet spatial, temporal, and regional ablation CSVs.
  * `/metrics/` - Unified metrics logs (`metrics_train.csv`, `metrics_benchmark.csv`, `metrics_cv.csv`, `emissions.csv`).
* `/models` - Saved model weights organized by architecture:
  * `/{architecture}/base/` - Initial PyTorch (`.pth`) or Decision Tree (`.joblib`) models.
  * `/{architecture}/finetuned/` - Fine-tuned model checkpoints.
* `/figures` - Exported analytical plots (`.png` and `.svg`), organized in subdirectories:
  * `/benchmarks/{architecture}/` - Per-model time-domain envelopes, CDFs, heatmaps, histograms, scatter, and violin plots.
  * `/ablation_global/` - Comparative multi-architecture temporal degradation and CDF plots.
  * `/ablation_resnet/` - ResNet temporal ablation and regional improvement heatmaps.
  * `/metrics/` - Barplots for training time, model size, inference latency, energy, CO₂ emissions, and cross-validation.
* `RESULTS.md` - Local manifest for trained models, valid MoE grid cells, metrics, benchmarks, and final figures.
* `/logs` - TensorBoard training logs for interactive loss and learning rate diagnostics.
* `/tests` - Automated test suite:
  * `test_coordinate_transforms.py` - Roundtrip tests for COE↔ECI, COE↔MEE, Kepler, and angular wrapping.
  * `test_architecture.py` - Forward pass shapes, MC-Dropout correctness, gradient flow for all NN architectures, and TreeBaseline wrapper validation.
  * `test_residuals.py` - Grey-Box residual pipeline output validation and physical sanity checks.
  * `test_config.py` - Configuration bounds and hyperparameter consistency verification.
  * `test_visualization.py` - Unit tests for Spanish grammar titling and preposition logic ("de" vs "del").
  * `test_audit.py` - Structure validation and audit checks for grid cells and metrics files.
  * `test_naming.py` - Unit tests for inclination two-decimal formatting conventions (`0.00-90.00`).
* `/src` - Core framework source code:
  * `config.py` - Centralized mission parameters, physical bounds, and training hyperparameters.
  * `/ml` - Machine learning module:
    * `dataset.py` - Custom dataset loader, Z-score standardization, and angular feature engineering.
    * `architecture.py` - Deep Residual Architecture (ResidualPredictor), ablation baselines (Linear, MLP, LSTM), and TreeBaseline wrapper for scikit-learn DecisionTreeRegressor.
  * `/physics` - Astrodynamic module:
    * `residuals.py` - Shared Grey-Box residual computation pipeline (Oracle vs. Analytical in MEE).
    * `analytical.py` - The ESTHER closed-form CW equations translated and optimized.
    * `kepler.py` - Unperturbed two-body Keplerian baseline analytical Newton-Raphson propagator.
    * `oracle.py` - High-precision numerical propagation and ECI/COE/MEE transformations (Ground Truth).
  * `generate_base_dataset.py` - Parallelized Monte Carlo MEE dataset generator for initial expert training.
  * `train_base.py` - PyTorch and scikit-learn training loop with Early Stopping, TensorBoard logging, CodeCarbon emissions tracking, and metrics CSV export.
  * `train_cv.py` - K-Fold Cross-Validation suite for all architectures. Exports `data/metrics/metrics_cv.csv` with mean ± std validation MSE.
  * `orchestrator_base.py` - Automated ML Pipeline to build and train the initial 3D grid of MoE models.
  * `generate_finetune_dataset.py` - Uncertainty-based active learning sampler to mine hard cases.
  * `train_finetune.py` - Continual learning script with Early Stopping and TensorBoard logging.
  * `orchestrator_finetune.py` - Automated QA pipeline to mine hard cases and fine-tune the fleet.
  * `simulate_mission.py` - Final OBC flight simulator implementing the MoE Router and the Active Learning loop.
  * `benchmark.py` - Aerospace-grade evaluation script measuring CPU time, decomposing absolute ECI error into the local RIC frame, and exporting inference timing metrics.
  * `visualize_benchmark.py` - Telemetry and statistical visualization engine with three operating modes (`single`, `ablation`, `metrics`).
  * `plot_resnet_ablation_time.py` - Dedicated ResNet temporal ablation study runner with interactive zoom capabilities.
  * `plot_expert_mesh.py` - Expert mesh grid discrimination generator (`orbita_expert_mesh`) and regional improvement heatmaps.
* `/scripts` - Automation scripts for reproducing TFM artifacts:
  * `technical_closure.sh` / `technical_closure.bat` - Full pipeline with JSON state persistence (`data/.technical_closure_state.json`), safe SIGINT interruption handling, and quiet/verbose mode selector.
* `.flake8` - Flake8 linting configuration (PEP-8 compliance).
* `requirements.txt` - Pinned Python environment dependencies for reproducibility.

## ⚙️ Installation & Usage
### Setup
Initialize the environment, install dependencies, and run the pipeline automatically:
* **macOS/Linux**: `chmod +x setup_unix.sh && ./setup_unix.sh`
* **Windows**: `.\setup_windows.bat`

### Execution Scripts & CLI Options

1. **Build the MoE Fleet (`orchestrator_base.py`)**:
   ```bash
   python src/orchestrator_base.py --model_type resnet
   ```
   * `--model_type`: Target architecture (`resnet`, `mlp`, `lstm`, `linear`, `tree`).
   * `--samples_per_expert`: Samples to generate per expert grid cell (default: 50,000).

2. **Train Ablation Baselines**:
   ```bash
   python src/orchestrator_base.py --model_type mlp
   python src/orchestrator_base.py --model_type lstm
   python src/orchestrator_base.py --model_type linear
   python src/orchestrator_base.py --model_type tree
   ```

3. **Continual Fine-Tuning Pipeline (`orchestrator_finetune.py`)**:
   ```bash
   python src/orchestrator_finetune.py
   ```

4. **K-Fold Cross-Validation (`train_cv.py`)**:
   ```bash
   python src/train_cv.py --dataset data/datasets/training/orbita_dataset_300-2000_0.0000-0.1000_0.00-90.00.csv --model_type resnet --folds 5
   ```
   * `--dataset`: Path to input dataset CSV.
   * `--model_type`: Target architecture (`resnet`, `mlp`, `lstm`, `linear`, `tree`).
   * `--folds`: Number of validation folds (default: 5).

5. **Performance Benchmarking (`benchmark.py`)**:
   ```bash
   python src/benchmark.py --model_type resnet --mode_choice 3 --seed 42
   ```
   * `--model_type`: Target architecture (`resnet`, `mlp`, `lstm`, `linear`, `tree`).
   * `--mode_choice`: `1` (time domain), `2` (space domain), `3` (both).
   * `--space_samples`: Number of space-domain samples (default: 100,000).
   * `--time_samples`: Number of time-domain trajectories (default: 10,000).
   * `--quiet`: Suppress detailed console logs.
   * `--no_tracking`: Disable CodeCarbon and metrics CSV updates.

6. **Visualization Engine (`visualize_benchmark.py`)**:
   ```bash
   # Per-model detailed analysis report
   python src/visualize_benchmark.py --mode single --model_type resnet
   # Multi-architecture error comparison
   python src/visualize_benchmark.py --mode ablation
   # Comparative metrics barplots (training time, model size, inference latency, energy, CO2 emissions, CV)
   python src/visualize_benchmark.py --mode metrics
   ```
   * `--mode`: Visualization mode (`single`, `ablation`, `metrics`).
   * `--architecture` / `--model_type`: Specific architecture(s) to process.
   * `--data_dir`: Base data directory (default: `data`).
   * `--output_dir`: Export directory (default: `figures`).
   * `--exclude`: Component plots to skip (e.g. `--exclude training_time`).

7. **ResNet Ablation Visualization (`plot_resnet_ablation_time.py`)**:
   ```bash
   python src/plot_resnet_ablation_time.py --only-figures
   ```
   * `--only-figures`: Generate figures directly without interactive GUI.
   * `--interactive`: Launch interactive Matplotlib zoom window.

8. **Expert Mesh Grid Visualization (`plot_expert_mesh.py`)**:
   ```bash
   python src/plot_expert_mesh.py
   ```

9. **Repository Audit (`audit_results.py`)**:
   ```bash
   python src/audit_results.py
   ```

10. **Technical Closure Pipeline (`scripts/technical_closure.sh`)**:
    ```bash
    # Interactive mode (prompts for quiet/verbose execution and checkpoint resume)
    ./scripts/technical_closure.sh

    # CLI Flags
    ./scripts/technical_closure.sh --quiet    # Execute cleanly with minimal console output
    ./scripts/technical_closure.sh --verbose  # Stream execution output to terminal
    ./scripts/technical_closure.sh --reset    # Reset checkpoint state and restart from step 1
    ```

### Testing
Run the full automated test suite:
```bash
python -m pytest tests/ -v
```

### Training Diagnostics
Monitor training loss and learning rate progress interactively using TensorBoard:
```bash
tensorboard --logdir=logs/
```

## 🎓 Academic Context
This repository contains the source code and mathematical tools developed for the Master's Thesis (*Trabajo de Fin de Máster*, TFM) in Space Science and Technology at Universidad de Alcalá (UAH), Spain.
* Author: Benito Fernández González
* Tutor: David Fernández Barrero
