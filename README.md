# O.R.B.I.T.A. 🛰️
**O**racle-guided **R**esidual-**B**ased **I**ntelligent **T**rajectory **A**lgorithm

A hybrid orbital mechanics framework developed to overcome the secular degradation of analytical orbital propagation models. Building upon the foundational analytical solutions derived in **[ESTHER](https://github.com/BFG508/ESTHER)**, ORBITA introduces a Machine Learning Active Learning loop to compensate for residual errors over time. This project aims to achieve high-fidelity orbit propagation (perturbed by Earth's $J_2$ and $J_3$ zonal harmonics) with the drastically reduced computational cost required for Onboard Computers (OBCs).

## 🚀 Features
* **Grey-Box Modeling Approach**: Synergizes the execution speed of a deterministic analytical model (white-box) with the adaptive error-correction capabilities of deep learning (black-box).
* **Singularity-Free Formulation**: Operates internally using Modified Equinoctial Elements (MEE), granting the AI absolute mathematical immunity against classical orbital singularities (e.g., perfectly circular or equatorial orbits).
* **Deep Residual Architecture (ResNet)**: Utilizes a deep network with skip-connections to learn the extreme, high-frequency non-linearities of orbital perturbations at centimetric precision without suffering from vanishing gradients.
* **Ablation Study with Multi-Architecture Support**: Supports comparative analysis across five architectures — ResNet, MLP, LSTM, Linear, and Decision Tree (scikit-learn) — to empirically validate architectural choices.
* **3D Mixture of Experts (MoE)**: Divides the vast orbital parameter space (Altitude, Eccentricity, Inclination) into a hyper-specialized 3D grid. The framework automatically trains and dynamically routes queries to local "expert" neural networks.
* **Active Learning with "GPS Reset"**: Implements epistemic uncertainty estimation via Monte Carlo Dropout. If the neural network's uncertainty breaches a strict safety threshold, it triggers a "GPS Reset", calling the computationally expensive Numerical Oracle to wipe out any accumulated along-track error before handing control back to the AI.
* **Autonomous Continual Learning (Fine-Tuning)**: Features a dedicated active-learning pipeline that automatically mines "hard cases" using uncertainty sampling. It seamlessly upgrades weak expert models using a microscopic learning rate and a replay buffer, preventing catastrophic forgetting.
* **K-Fold Cross-Validation**: A dedicated script (`train_cv.py`) performs statistically rigorous K-Fold validation across all architectures, producing mean ± standard deviation MSE to ensure results are not artifacts of a single split.
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
* **Testing**: `pytest` (80 automated tests covering coordinate transforms, architectures, residuals, and configuration).

## 📂 Repository Structure
* `/data` - Dynamically generated MEE orbital datasets (`.csv`), benchmark results, and metrics logs (`metrics_train.csv`, `metrics_benchmark.csv`, `metrics_cv.csv`, `emissions.csv`).
* `/models` - Saved model weights: PyTorch (`.pth`) and scikit-learn (`.joblib`) for the MoE expert fleet.
* `/figures` - Exported analytical plots (`.png` and `.svg`).
* `/logs` - TensorBoard training logs for interactive loss and learning rate diagnostics.
* `/tests` - Automated test suite:
  * `test_coordinate_transforms.py` - Roundtrip tests for COE↔ECI, COE↔MEE, Kepler, and angular wrapping.
  * `test_architecture.py` - Forward pass shapes, MC-Dropout correctness, gradient flow for all NN architectures, and TreeBaseline wrapper validation.
  * `test_residuals.py` - Grey-Box residual pipeline output validation and physical sanity checks.
  * `test_config.py` - Configuration bounds and hyperparameter consistency verification.
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
  * `train_cv.py` - K-Fold Cross-Validation suite for all architectures. Exports `data/metrics_cv.csv` with mean ± std validation MSE.
  * `orchestrator_base.py` - Automated ML Pipeline to build and train the initial 3D grid of MoE models.
  * `generate_finetune_dataset.py` - Uncertainty-based active learning sampler to mine hard cases.
  * `train_finetune.py` - Continual learning script with Early Stopping and TensorBoard logging.
  * `orchestrator_finetune.py` - Automated QA pipeline to mine hard cases and fine-tune the entire fleet.
  * `simulate_mission.py` - Final OBC flight simulator implementing the MoE Router and the Active Learning loop.
  * `benchmark.py` - Aerospace-grade evaluation script measuring CPU time, decomposing absolute ECI error into the local RIC frame, and exporting inference timing metrics.
  * `visualize_benchmark.py` - Telemetry and statistical visualization engine with three operating modes: `single` (per-model report), `ablation` (architecture error comparison), and `metrics` (training time, model size, inference latency, cross-validation, and CodeCarbon barplots).
* `.flake8` - Flake8 linting configuration (PEP-8 compliance).
* `requirements.txt` - Pinned Python environment dependencies for reproducibility.
* `setup_windows.bat` / `setup_unix.sh` - Automated environment setup.

## ⚙️ Installation & Usage
### Setup
You can use the provided automation scripts to initialize the environment, install dependencies, and run the pipeline automatically.
* **Windows**: Double-click `setup_windows.bat` or run `.\setup_windows.bat` in your terminal.
* **macOS/Linux**: Run `chmod +x setup_unix.sh` followed by `./setup_unix.sh`.

### Run ORBITA
1. **Build the MoE Fleet**: Generate datasets and train the base specialized neural networks.
   ```bash
   python src/orchestrator_base.py --model_type resnet
   ```
2. **Train Ablation Baselines** (Optional): Train Decision Tree and other baselines for comparative analysis.
   ```bash
   python src/orchestrator_base.py --model_type tree
   python src/orchestrator_base.py --model_type mlp
   python src/orchestrator_base.py --model_type lstm
   python src/orchestrator_base.py --model_type linear
   ```
3. **Continual Learning** (Optional but recommended): Auto-mine hard cases and fine-tune the fleet:
   ```bash
   python src/orchestrator_finetune.py
   ```
4. **Execute a Mission Stress Test**: Route the initial state to the correct expert and propagate.
   ```bash
   python src/simulate_mission.py
   ```
5. **Run the Performance Benchmark**: Validate the computational speedup and evaluate RIC coordinate error degradation.
   ```bash
   python src/benchmark.py --model_type resnet
   ```
   For a non-destructive smoke run, write suffixed CSVs and disable tracking/log updates:
   ```bash
   python src/benchmark.py --model_type resnet --mode_choice 2 --space_samples 5 --output_suffix _smoke --no_tracking
   ```
6. **K-Fold Cross-Validation**: Validate model robustness across data splits.
   ```bash
   python src/train_cv.py --dataset data/orbita_dataset_300-2000_0.0000-0.1000_0-90.csv --model_type resnet --folds 5
   ```
7. **Data Visualization & Telemetry Analysis**: Generate publication-ready figures from the collected data.
   ```bash
   # Per-model error analysis
   python src/visualize_benchmark.py --mode single --model_type resnet
   # Multi-architecture error comparison
   python src/visualize_benchmark.py --mode ablation
   # Comparative metrics: training time, model size, inference latency, CodeCarbon, cross-validation
   python src/visualize_benchmark.py --mode metrics
   ```

### Testing
Run the full automated test suite to verify the integrity of coordinate transforms, neural architectures, residual computations, and configuration:
   ```bash
   python -m pytest tests/ -v
   ```

### Training Diagnostics
Monitor training progress interactively using TensorBoard:
   ```bash
   tensorboard --logdir=logs/
   ```

## 🎓 Academic Context
This repository contains the source code and mathematical tools developed for the Master's Thesis (*Trabajo de Fin de Máster*, TFM) in Space Science and Technology at Universidad de Alcalá (UAH), Spain.
* Author: Benito Fernández González
* Tutor: David Fernández Barrero
* Academic Year: 2025/2026
* Previous Work: This project directly expands upon the Bachelor's Thesis analytical derivations available at [ESTHER](https://github.com/BFG508/ESTHER).
