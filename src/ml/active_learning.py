"""
Module: active_learning.py

Description:
    Defines the Neural Network architecture for predicting residual errors.
    Implements Monte Carlo Dropout (MC-Dropout) to quantify prediction 
    epistemic uncertainty, enabling the Active Learning loop to dynamically 
    query the Numerical Oracle only when the AI is unsure of its prediction.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

class ResidualPredictor(nn.Module):
    """
    Multi-Layer Perceptron (MLP) designed to predict the 6 Classical 
    Orbital Element (COE) residual errors between the Analytical Baseline 
    and the Numerical Oracle.
    
    Target outputs: [err_SMA, err_ECC, err_INC, err_RAAN, err_AOP, err_TA]
    """
    
    def __init__(
        self, 
        input_size = 10, 
        hidden_size = 512, 
        output_size = 6, 
        dropout_rate = 0.2
    ):
        """
        Initializes the network architecture.
        Note: input_size is 10 due to the trigonometric expansion (sines and cosines)
        of the angular orbital elements to prevent 360-degree boundary issues.
        """
        super(ResidualPredictor, self).__init__()
        
        # Architecture definition using GELU for smoother physical regression
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass through the network.
        During standard evaluation, dropout layers are normally bypassed.
        """
        return self.net(x)

    def predict_with_uncertainty(
        self, 
        x: torch.Tensor, 
        num_samples = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs Monte Carlo Dropout inference to estimate both the 
        prediction value and its epistemic uncertainty.
        
        Inputs:
            x           : Input tensor [SMA, ECC, INC, sin(RAAN), cos(RAAN), 
                          sin(AOP), cos(AOP), sin(TA), cos(TA), TOF]
            num_samples : Number of stochastic forward passes to execute
            
        Outputs:
            mean_pred   : Mean of the stochastic predictions (The actual correction)
            uncertainty : Standard deviation of the predictions across samples
        """
        # Force the network to keep Dropout layers active during inference
        self.train()
        
        predictions = []
        with torch.no_grad(): # Disable gradient tracking for performance
            for _ in range(num_samples):
                predictions.append(self.forward(x))
                
        # Stack predictions: shape becomes (num_samples, batch_size, output_size)
        predictions = torch.stack(predictions)
        
        # Compute mean and standard deviation across the num_samples dimension (dim = 0)
        mean_pred = predictions.mean(dim = 0)
        uncertainty = predictions.std(dim = 0)
        
        # Revert the network back to standard evaluation mode
        self.eval()
        
        return mean_pred, uncertainty