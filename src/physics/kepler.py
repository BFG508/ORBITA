"""
Module: kepler.py

Description:
    This module handles the unperturbed Keplerian two-body propagation.
    It serves as the base orbit upon which the analytical J2/J3 relative 
    perturbations (from the ESTHER) are superimposed.
    
    Currently, it uses a numerical integrator (solve_ivp) to establish the 
    baseline. For a true analytical deployment on an On-Board Computer, 
    this numerical block should be replaced with an analytical Keplerian 
    propagator (e.g., using Newton-Raphson to solve Kepler's Equation 
    from Mean Anomaly to True Anomaly).
"""

import numpy as np
from scipy.integrate import solve_ivp

def getKeplerianNumerical(mu, r0, v0, TOF):
    """
    Computes the unperturbed Keplerian orbital state by numerically 
    integrating the pure two-body problem equations of motion.
    
    Inputs:
        mu : Gravitational parameter of the central body [m^3/s^2]
        r0 : Initial position vector in ECI frame              [m]
        v0 : Initial velocity vector in ECI frame            [m/s]
        TOF: Time of Flight for the propagation                [s]
        
    Outputs:
        r_final: Final position vector                         [m]
        v_final: Final velocity vector                       [m/s]
    """
    
    def kepler_dynamics(t, y):
        # State derivative for the ideal two-body problem
        r_vec = y[0:3]
        v_vec = y[3:6]
        
        r = np.linalg.norm(r_vec)
        
        # Pure Newtonian central gravity acceleration
        a_central = -mu / (r**3) * r_vec
        
        return np.concatenate((v_vec, a_central))

    # Numerical integration
    res = solve_ivp(
        fun    = kepler_dynamics, 
        t_span = [0, TOF], 
        y0     = np.concatenate((r0, v0)), 
        method = 'LSODA', # An Adams/BDF method
        rtol   = 1e-12,   # Strict relative tolerance
        atol   = 1e-5     # Strict absolute tolerance
    )
    
    # Extract the state at the final integration step
    r_final = res.y[0:3, -1]
    v_final = res.y[3:6, -1]
    
    return r_final, v_final