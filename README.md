# ORBITA 🛰️
**O**racle-guided **R**esidual-**B**ased **I**ntelligent **T**rajectory **A**lgorithm

A hybrid orbital mechanics framework developed to overcome the secular degradation of analytical orbital propagation models. Building upon the foundational analytical solutions derived in **[ESTHER](https://github.com/BFG508/ESTHER)**, ORBITA introduces a Machine Learning Active Learning loop to compensate for residual errors over time. This project aims to achieve high-fidelity orbit propagation (perturbed by Earth's $J_2$ and $J_3$ zonal harmonics) with the drastically reduced computational cost required for Onboard Computers (OBCs).

## 🚀 Features
* **Grey-Box Modeling Approach**: Synergizes the high execution speed of a purely analytical mathematical model (white-box) with the adaptive error-correction.
* **Singularity-Free MEE Formulation**: The framework has been fully migrated to Modified Equinoctial Elements (MEE) for both inputs and outputs. This ensures 100% mathematical autonomy and stability in circular ($e=0$) and equatorial ($i=0$) orbits where classical elements fail.
* **3D Mixture of Experts (MoE) Architecture**: Divides the vast orbital parameter space (Altitude, Eccentricity, Inclination) into a hyper-specialized 3D grid. The system trains and dynamically routes queries to specialized "expert" neural networks, achieving extreme precision by neutralizing local mathematical singularities.
* **Active Learning with GPS Reset**: Implements an intelligent uncertainty-sampling mechanism via MC-Dropout. If the 3D Euclidean spread (epistemic uncertainty) breaches a 100-meter threshold, the system triggers an "Oracle Reset," computing the absolute truth from $T_0$ to eliminate accumulated drift.
* **Analytical Baseline Integration**: Utilizes the explicit closed-form Cartesian expressions developed in ESTHER to provide instantaneous, computationally inexpensive baseline orbital states.
* **Pure Python Numerical Oracle**: Features a custom-built, highly rigorous numerical integrator using Cowell's formulation and SciPy's high-order solvers (`LSODA`) to generate exact ground truth data.
* **Flight-Grade Training**: Features a deep ResNet-style architecture with Skip Connections and a ReduceLROnPlateau scheduler to squeeze maximum float32 precision for centimetric accuracy.
* **OBC-Optimized Architecture**: Designed specifically to minimize continuous CPU load, memory footprint, and iterative calculations, making it an ideal candidate for real-time trajectory planning on resource-constrained satellite hardware.

## 🛠️ Technology Stack
Unlike its predecessor ESTHER, which was developed in MATLAB, ORBITA is built entirely in **Python** to leverage the modern Machine Learning ecosystem. 
* **Core Mathematics & Physics:** `NumPy`, `SciPy` (for high-precision ODE solving).
* **Machine Learning:** `PyTorch` (Multi-Layer Perceptrons, MC-Dropout inference, and dataset management).

## 📂 Repository Structure
* `/data` - Dynamically generated orbital datasets (`.csv`) for training expert models (ignored by git).
* `/models` - Saved PyTorch Neural Network weights (`.pth`) for the expert fleet (ignored by git).
* `/src` - Core framework source code:
  * `/ml` - Machine learning module:
    * `dataset.py` - PyTorch custom dataset loader, standardization, and feature engineering.
    * `active_learning.py` - Neural network architecture (ResidualPredictor) and MC-Dropout inference.
  * `/physics` - Astrodynamic module:
    * `analytical.py` - The ESTHER closed-form CW equations translated and optimized.
    * `kepler.py` - Unperturbed two-body Keplerian baseline propagator.
    * `oracle.py` - High-precision numerical propagation and coordinate transformations (Ground Truth).
  * `generate_dataset.py` - Monte Carlo dataset generator for target orbital regimes.
  * `train.py` - Standard PyTorch training loop with dynamic saving.
  * `orchestrator.py` - Automated ML Pipeline to build and train the 3D grid of MoE models.
  * `simulate_mission.py` - Final OBC flight simulator implementing the MoE Router and Active Learning loop.
* `requirements.txt` - Python environment dependencies.

## ⚙️ Installation & Usage
1. **Clone the repository:**
   ```bash
   git clone https://github.com/BFG508/ORBITA.git
2. Set up the virtual environment:
   ```bash
   cd ORBITA
   py -m venv .venv
   ```
   * On Windows: 
      ```bash 
      .venv\Scripts\activate
      ```
    * On macOs/Linux: 
      ```bash 
      source .venv/bin/activate
      ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Generate the datasets and train the specialized neural networks using the orchestrator:
   ```bash
   python src/orchestrator.py
   ```
5. Execute a mission stress test. The system will automatically route the initial state to the correct expert model and begin propagation:
   ```bash
   python src/simulate_mission.py
   ```

## 🎓 Academic Context
This repository contains the source code and mathematical tools developed for the Master's Thesis (*Trabajo de Fin de Máster*, TFM) in Space Science and Technology at Universidad de Alcalá (UAH), Spain.
* Author: Benito Fernández González
* Tutor: David Fernández Barrero
* Academic Year: 2025/2026
* Previous Work: This project directly expands upon the Bachelor's Thesis analytical derivations available at [ESTHER](https://github.com/BFG508/ESTHER).