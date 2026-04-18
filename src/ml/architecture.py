"""
Module: architecture.py

Description:
    Contains the Deep Neural Network architecture (ResidualPredictor) used as
    the expert models in the ORBITA framework, along with baseline models
    for rigorous ablation studies.

    The flagship model is upgraded to a Deep Residual Architecture (ResNet-style)
    to handle the extreme non-linearities of the J2/J3 orbital perturbations
    at centimetric precision. Features MC-Dropout for epistemic uncertainty estimation.

    Ablation baselines (Linear, MLP, LSTM) are included to empirically validate
    the superiority of the Grey-Box residual approach.
"""

import torch
import torch.nn as nn


# =============================================================================
# FLAGSHIP ARCHITECTURE (GREY-BOX RESNET)
# =============================================================================

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

    Default parameters have been scaled down (hidden_size=128, num_blocks=2)
    to strictly comply with SWaP (Size, Weight, and Power) limitations
    for deployment on On-Board Computers (OBC).

    Inputs: 8 variables (Initial MEE parameters + True Longitude components + TOF).
    Outputs: 6 variables (Residual errors in MEE).
    """

    def __init__(self, input_size=8, hidden_size=128, output_size=6, dropout_rate=0.001, num_blocks=2):
        """
        Initializes the Deep Residual architecture.

        Args:
            input_size (int): Number of input features (8 for stable MEE formulation).
            hidden_size (int): Number of neurons per hidden layer.
            output_size (int): Number of prediction targets.
            dropout_rate (float): Dropout probability for regularization and MC-Dropout.
            num_blocks (int): Number of sequential residual blocks.
        """
        super(ResidualPredictor, self).__init__()

        # Input layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU()
        )

        # The "Deep Core": Dynamically generated residual blocks
        blocks = [ResidualBlock(hidden_size, dropout_rate) for _ in range(num_blocks)]
        self.res_blocks = nn.Sequential(*blocks)

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

        self.eval()  # Restore deterministic mode after MC-Dropout sampling

        return mean_pred, std_pred


# =============================================================================
# ABLATION STUDY & BASELINE MODELS
# =============================================================================

class LinearBaseline(nn.Module):
    """
    A strict Linear Regression baseline.
    Used to mathematically prove that the J2/J3 perturbation mapping
    is highly non-linear and cannot be solved by simple matrix multiplication.
    """

    def __init__(self, input_size=8, output_size=6):
        """
        Initializes the linear baseline.

        Args:
            input_size (int): Number of input features.
            output_size (int): Number of prediction targets.
        """
        super(LinearBaseline, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        """
        Forward pass executing a single linear transformation.
        """
        return self.linear(x)


class MLPPredictor(nn.Module):
    """
    A standard Multi-Layer Perceptron (MLP) baseline without skip connections.
    Used in the ablation study to demonstrate the vanishing gradient problem
    in deep standard networks when approximating orbital mechanics.
    """

    def __init__(self, input_size=8, hidden_size=128, output_size=6, dropout_rate=0.001, num_layers=3):
        """
        Initializes the standard MLP baseline.

        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of neurons per hidden layer.
            output_size (int): Number of prediction targets.
            dropout_rate (float): Dropout probability.
            num_layers (int): Total number of hidden linear layers.
        """
        super(MLPPredictor, self).__init__()

        layers = [nn.Linear(input_size, hidden_size), nn.GELU()]

        # Build hidden layers without any residual shortcuts
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(hidden_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Standard forward pass through sequential dense layers.
        """
        return self.network(x)


class LSTMPredictor(nn.Module):
    """
    A Long Short-Term Memory (LSTM) baseline.
    Adapts the single-shot continuous time input to a recurrent architecture
    to compare inference speed and temporal sequence modeling.
    """

    def __init__(self, input_size=8, hidden_size=128, output_size=6, num_layers=2, dropout_rate=0.001):
        """
        Initializes the recurrent LSTM baseline.

        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of hidden features in the LSTM state.
            output_size (int): Number of prediction targets.
            num_layers (int): Number of recurrent layers.
            dropout_rate (float): Dropout probability between LSTM layers.
        """
        super(LSTMPredictor, self).__init__()

        # Setting batch_first=True assumes input shape (batch, seq_len, features)
        lstm_dropout = dropout_rate if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass processing the input as a temporal sequence.
        """
        # LSTM expects 3D input. If 2D (batch, features), add a dummy sequence dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)

        lstm_out, _ = self.lstm(x)

        # Extract the prediction corresponding to the final time step
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out)
