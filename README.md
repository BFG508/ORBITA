# ORBITA 🛰️
**O**racle-guided **R**esidual-**B**ased **I**ntelligent **T**rajectory **A**lgorithm

A hybrid orbital mechanics framework developed to overcome the secular degradation of analytical orbital propagation models. Building upon the foundational analytical solutions derived in [ESTHER](https://github.com/BFG508/ESTHER), ORBITA introduces a Machine Learning Active Learning loop to compensate for residual errors over time. This project aims to achieve high-fidelity orbit propagation (perturbed by Earth's J$_2$ and J$_3$ zonal harmonics) with the drastically reduced computational cost required for Onboard Computers (OBCs).

## 🚀 Features
* **Grey-Box Modeling Approach:** Synergizes the high execution speed of a purely analytical mathematical model (white-box) with the adaptive error-correction capabilities of deep learning (black-box).
* **Analytical Baseline Integration:** Utilizes the explicit closed-form Cartesian expressions developed in ESTHER to provide instantaneous, computationally inexpensive baseline orbital states.
* **Pure Python Numerical Oracle:** Features a custom-built, highly rigorous numerical integrator using Cowell's formulation and SciPy's high-order solvers (`LSODA`) to generate exact ground truth data without relying on heavy external astrodynamics libraries.
* **Active Learning Loop:** Implements an intelligent uncertainty-sampling mechanism. The Neural Network dynamically queries the computationally expensive Numerical Oracle only when its confidence in predicting the analytical residual error drops below a specific threshold.
* **OBC-Optimized Architecture:** Designed specifically to minimize continuous CPU load and iterative calculations, making it an ideal candidate for real-time trajectory planning on resource-constrained satellite hardware.

## 🛠️ Technology Stack
Unlike its predecessor ESTHER, which was developed in MATLAB, ORBITA is built entirely in **Python** to leverage the modern Machine Learning ecosystem. 
* **Core Mathematics & Physics:** `NumPy`, `SciPy` (for high-precision ODE solving).
* **Machine Learning:** Designed for integration with modern deep learning frameworks (`TensorFlow`) for the residual prediction network.

## 📂 Repository Structure
* `/data` - Generated orbital datasets for neural network training (ignored by git).
* `/models` - Saved Neural Network weights and uncertainty evaluation criteria.
* `/src` - Core framework source code:
  * `analytical.py` - The ESTHER closed-form CW equations translated and optimized for Python.
  * `kepler.py` - Unperturbed two-body Keplerian baseline propagator.
  * `oracle.py` - High-precision numerical propagation (Ground Truth).
  * `compare.py` - Evaluation script to compute the Euclidean residual error between models.
  * `active_learning.py` - Neural network architecture and the uncertainty sampling loop.
* `requirements.txt` - Python environment dependencies.

## ⚙️ Installation & Usage
1. **Clone the repository:**
   ```bash
   git clone https://github.com/BFG508/ORBITA.git
2. Set up the virtual environment:
   ```bash
   cd ORBITA
   py -m venv venv
   ```
   * On Windows: 
      ```bash 
      venv\Scripts\activate 
      ```
    * On macOs/Linux: 
      ```bash 
      source venv/bin/activate
      ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4.

5.

## 🎓 Academic Context
This repository contains the source code and mathematical tools developed for the Master's Thesis (*Trabajo de Fin de Máster*, TFM) in Space Science and Technology at Universidad de Alcalá (UAH), Spain.
* Author: Benito Fernández González
* Tutor: David Fernández Barrero
* Academic Year: 2025/2026
* Previous Work: This project directly expands upon the Bachelor's Thesis analytical derivations available at [ESTHER](https://github.com/BFG508/ESTHER).