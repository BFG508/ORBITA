# O.R.B.I.T.A. 🛰️
**O**racle-guided **R**esidual-**B**ased **I**ntelligent **T**rajectory **A**lgorithm

A hybrid orbital mechanics framework developed to overcome the secular degradation of analytical orbital propagation models. Building upon the foundational analytical solutions derived in **[ESTHER](https://github.com/BFG508/ESTHER)**, ORBITA introduces a Machine Learning Active Learning loop to compensate for residual errors over time. This project aims to achieve high-fidelity orbit propagation (perturbed by Earth's $J_2$ and $J_3$ zonal harmonics) with the drastically reduced computational cost required for Onboard Computers (OBCs).

## 🚀 Features
* **Grey-Box Modeling Approach:**: Synergizes the execution speed of a deterministic analytical model (white-box) with the adaptive error-correction capabilities of deep learning (black-box).
* **Singularity-Free Formulation**: Operates internally using Modified Equinoctial Elements (MEE), granting the AI absolute mathematical immunity against classical orbital singularities (e.g., perfectly circular or equatorial orbits).
* **Deep Residual Architecture (ResNet)**: Utilizes a deep network with skip-connections to learn the extreme, high-frequency non-linearities of orbital perturbations at centimetric precision without suffering from vanishing gradients.
* **3D Mixture of Experts (MoE)**: Divides the vast orbital parameter space (Altitude, Eccentricity, Inclination) into a hyper-specialized 3D grid. The framework automatically trains and dynamically routes queries to local "expert" neural networks.
* **Active Learning with "GPS Reset"**: Implements epistemic uncertainty estimation via Monte Carlo Dropout. If the neural network's uncertainty breaches a strict 100-meter safety threshold, it triggers a "GPS Reset", calling the computationally expensive Numerical Oracle to wipe out any accumulated along-track error before handing control back to the AI.
* **Autonomous Continual Learning (Fine-Tuning)**: Features a dedicated active-learning pipeline that automatically mines "hard cases" using uncertainty sampling. It seamlessly upgrades weak expert models using a microscopic learning rate and a replay buffer, preventing catastrophic forgetting.
* **Flight-Grade & OBC-Optimized**: Written in strict PEP-8 compliant Python. Designed specifically to minimize continuous CPU load and memory footprint, validated by a built-in millisecond-level RIC (Radial, In-Track, Cross-Track) benchmarking suite.

## 🛠️ Technology Stack
Unlike its predecessor ESTHER, which was developed in MATLAB, ORBITA is built entirely in **Python** to leverage the modern Machine Learning ecosystem. 
* **Core Mathematics & Physics**: `NumPy`, `SciPy` (for high-precision `LSODA` ODE solving).
* **Machine Learning**: `PyTorch` (Deep Residual Networks, MC-Dropout inference, ReduceLROnPlateau scheduling, and dataset management).
* **Visualization & Analytics**: `Matplotlib`, `Seaborn` (CDF analysis and RIC error heatmaps).

## 📂 Repository Structure
* `/data` - Dynamically generated MEE orbital datasets (`.csv`) and benchmark results (ignored by git).
* `/models` - Saved PyTorch Neural Network weights (`.pth`) for the MoE expert fleet.
* `/figures` - Exported analytical plots (`.png` and `.svg`).
* `/src` - Core framework source code:
  * `/ml` - Machine learning module:
    * `dataset.py` - Custom dataset loader, Z-score standardization, and angular feature engineering.
    * `architecture.py` - Deep Residual Architecture (ResidualPredictor) and MC-Dropout inference.
  * `/physics` - Astrodynamic module:
    * `analytical.py` - The ESTHER closed-form CW equations translated and optimized.
    * `kepler.py` - Unperturbed two-body Keplerian baseline analytical Newton-Raphson propagator.
    * `oracle.py` - High-precision numerical propagation and ECI/COE/MEE transformations (Ground Truth).
  * `generate_base_dataset.py` - Monte Carlo MEE dataset generator for initial expert training.
  * `train_base.py` - PyTorch training loop featuring dynamic weight saving and aggressive LR scheduling.
  * `orchestrator_base.py` - Automated ML Pipeline to build and train the initial 3D grid of MoE models.
  * `generate_finetune_dataset.py` - Uncertainty-based active learning sampler to mine hard cases.
  * `train_finetune.py` - Continual learning script to upgrade existing models using a replay buffer.
  * `orchestrator_finetune.py` - Automated QA pipeline to mine hard cases and fine-tune the entire fleet.
  * `simulate_mission.py` - Final OBC flight simulator implementing the MoE Router and the Active Learning loop.
  * `benchmark.py` - Aerospace-grade evaluation script measuring CPU time and decomposing absolute ECI error into the local RIC frame.
  * `visualize_benchmark.py` - Telemetry and statistical visualization engine.
* `setup_windows.bat` / `setup_unix.sh` - Automated environment setup.
* `requirements.txt` - Python environment dependencies.

## ⚙️ Installation & Usage
### Setup
You can use the provided automation scripts to initialize the environment, install dependencies, and run the pipeline automatically.
* **Windows**: Double-click `setup_windows.bat` or run `.\setup_windows.bat` in your terminal.
* **macOS/Linux**: Run `chmod +x setup_unix.sh` followed by `./setup_unix.sh`.

### Run ORBITA
1. **Build the MoE Fleet**: Generate datasets and train the base specialized neural networks.
   ```bash
   python src/orchestrator_base.py
   ```
2. **Continual Learning** (Optional but recommended): Auto-mine hard cases and fine-tune the fleet:
   ```bash
   python src/orchestrator_finetune.py
   ```
3. **Execute a Mission Stress Test**: Route the initial state to the correct expert and propagate.
   ```bash
   python src/simulate_mission.py
   ```
4. **Run the Performance Benchmark**: Validate the computational speedup and evaluate RIC coordinate error degradation.
   ```bash
   python src/benchmark.py
   ```
5. **Data Visualization & Telemetry Analysis**: Once the benchmark suite has generated the raw telemetry, use the visualization engine to transform these results into figures.
   ```bash
   python src/visualize_benchmark.py
   ```

## 🎓 Academic Context
This repository contains the source code and mathematical tools developed for the Master's Thesis (*Trabajo de Fin de Máster*, TFM) in Space Science and Technology at Universidad de Alcalá (UAH), Spain.
* Author: Benito Fernández González
* Tutor: David Fernández Barrero
* Academic Year: 2025/2026
* Previous Work: This project directly expands upon the Bachelor's Thesis analytical derivations available at [ESTHER](https://github.com/BFG508/ESTHER).