"""
Module: kepler.py

Description:
    Fully analytical Keplerian two-body propagation using Newton-Raphson 
    convergence. 
    
    Serves as the ultra-fast deterministic baseline upon which the analytical 
    J2/J3 relative perturbations (ESTHER) and the Deep Learning residuals (MoE) 
    are superimposed. By replacing the previous ODE numerical integrator with 
    this analytical root-finding approach, the framework achieves the execution 
    speed required for deployment on low-power On-Board Computers (OBC).
"""

import numpy as np
from physics.oracle import eci_to_coe, coe_to_eci


def get_keplerian(mu, r0, v0, dt):
    """
    Computes the unperturbed Keplerian orbital state analytically.
    
    Args:
        mu (float): Gravitational parameter of the central body [m^3/s^2].
        r0 (np.ndarray): Initial position vector in the ECI frame [m].
        v0 (np.ndarray): Initial velocity vector in the ECI frame [m/s].
        dt (float): Time of Flight for the propagation [s].
        
    Returns:
        tuple: A tuple containing:
            - r_final (np.ndarray): Final propagated position vector [m].
            - v_final (np.ndarray): Final propagated velocity vector [m/s].
    """
    # Convert Cartesian ECI coordinates to Classical Orbital Elements (COE)
    sma, ecc, inc, raan, aop, ta_0 = eci_to_coe(mu, r0, v0)
    
    # Compute the Mean Motion (n)
    n = np.sqrt(mu / (sma**3))
    
    # Convert initial True Anomaly (ta_0) to Initial Eccentric Anomaly (E0)
    e0_term = np.sqrt((1.0 - ecc) / (1.0 + ecc)) * np.tan(ta_0 / 2.0)
    E0 = 2.0 * np.arctan(e0_term)
    
    # Compute Initial Mean Anomaly (M0) via Kepler's Equation
    M0 = E0 - ecc * np.sin(E0)
    
    # Linearly advance the Mean Anomaly by the Time of Flight (dt)
    M_final = M0 + n * dt
    
    # Solve Kepler's Equation (M = E - e*sin(E)) for the final Eccentric Anomaly
    E_curr = M_final
    tolerance = 1e-12
    max_iter = 100
    
    for _ in range(max_iter):
        # f(E) = E - e*sin(E) - M
        # f'(E) = 1 - e*cos(E)
        delta_E = (E_curr - ecc * np.sin(E_curr) - M_final) / (1.0 - ecc * np.cos(E_curr))
        E_curr -= delta_E
        
        if abs(delta_E) < tolerance:
            break
            
    # Convert the solved Final Eccentric Anomaly back to Final True Anomaly
    ta_final_term = np.sqrt((1.0 + ecc) / (1.0 - ecc)) * np.tan(E_curr / 2.0)
    ta_final = 2.0 * np.arctan(ta_final_term)
    
    # Transform the updated Classical Orbital Elements back to Cartesian ECI
    r_final, v_final = coe_to_eci(mu, sma, ecc, inc, raan, aop, ta_final)
    
    return r_final, v_final