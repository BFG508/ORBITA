"""
Module: architecture.py

Description:
    Contains the Deep Neural Network architecture (ResidualPredictor) used as 
    the expert models in the ORBITA framework.
    
    Upgraded to a Deep Residual Architecture (ResNet-style) to handle the 
    extreme non-linearities of the J2/J3 orbital perturbations at centimetric 
    precision. Features MC-Dropout for epistemic uncertainty estimation.
"""

import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    Deep building block with skip connections.
    Prevents the vanishing gradient problem, allowing for much deeper networks.
    """
    
    def __init__(self, hidden_size, dropout_rate):
        """
        Initializes the residual block layers.
        
        Args:
            hidden_size (int): Number of neurons in the hidden layers.
            dropout_rate (float): Probability of an element to be zeroed.
        """
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.act1    = nn.GELU()
        self.drop1   = nn.Dropout(dropout_rate)
        
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.act2    = nn.GELU()
        self.drop2   = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Forward pass with the residual skip connection.
        """
        identity = x  # Save original input for the skip connection
        
        out = self.linear1(x)
        out = self.act1(out)
        out = self.drop1(out)
        
        out = self.linear2(out)
        out = self.act2(out)
        out = self.drop2(out)
        
        # Residual Addition: Allows learning high-frequency non-linear features
        return out + identity 


class ResidualPredictor(nn.Module):
    """
    Main AI brain for the ORBITA MoE framework.
    
    Inputs: 8 variables (Initial MEE parameters + True Longitude components + TOF).
    Outputs: 6 variables (Residual errors in MEE).
    """
    
    def __init__(self, input_size=8, hidden_size=512, output_size=6, dropout_rate=0.001):
        """
        Initializes the Deep Residual architecture.
        
        Args:
            input_size (int): Number of input features (8 for stable MEE formulation).
            hidden_size (int): Number of neurons per hidden layer.
            output_size (int): Number of prediction targets.
            dropout_rate (float): Dropout probability for regularization and MC-Dropout.
        """
        super(ResidualPredictor, self).__init__()
        
        # Input layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU()
        )
        
        # The "Deep Core": 4 residual blocks (8 hidden layers total)
        self.res_blocks = nn.Sequential(
            ResidualBlock(hidden_size, dropout_rate),
            ResidualBlock(hidden_size, dropout_rate),
            ResidualBlock(hidden_size, dropout_rate),
            ResidualBlock(hidden_size, dropout_rate)
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Standard forward pass through the network.
        """
        x = self.input_layer(x)
        x = self.res_blocks(x)
        return self.output_layer(x)

    def predict_with_uncertainty(self, x, num_samples=50):
        """
        Executes the model multiple times with Dropout active to 
        extract the epistemic uncertainty cloud (MC-Dropout).
        
        Args:
            x (torch.Tensor): Input tensor.
            num_samples (int): Number of stochastic forward passes.
            
        Returns:
            tuple: A tuple containing:
                - mean_pred (torch.Tensor): Mean of the predictions.
                - std_pred (torch.Tensor): Standard deviation (uncertainty spread).
        """
        self.train()  # Force Dropout activation
        with torch.no_grad():
            preds = torch.stack([self.forward(x) for _ in range(num_samples)])
            
        mean_pred = preds.mean(dim=0)
        std_pred = preds.std(dim=0)
        
        return mean_pred, std_pred