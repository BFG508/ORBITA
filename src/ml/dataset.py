"""
Module: dataset.py

Description:
    Handles data loading, train/validation splitting, and feature normalization 
    for the PyTorch Neural Network. Configured to process Classical Orbital 
    Elements (COE) errors for specific orbital regimes (Expert Domains).
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class OrbitalDataset(Dataset):
    """
    Custom PyTorch Dataset for the ORBITA framework.
    Loads orbital parameters (X) and residual COE errors (Y) from a dynamically 
    named CSV file.
    """
    def __init__(self, csv_file):
        # Load data using NumPy (skipping the header row)
        data = np.genfromtxt(csv_file, delimiter=',', skip_header=1)
        
        # 1. Extract raw input features (X)
        SMA, ECC, INC = data[:, 0], data[:, 1], data[:, 2]
        RAAN, AOP, TA = data[:, 3], data[:, 4], data[:, 5]
        TOF = data[:, 6]
        
        # 2. Convert angles to continuous components
        # This prevents the neural network from struggling with 360 to 0 degree jumps.
        # Expands the 7 base features into 10 neural inputs.
        X_raw = np.column_stack([
            SMA, ECC, INC, 
            np.sin(RAAN), np.cos(RAAN),
            np.sin(AOP),  np.cos(AOP),
            np.sin(TA),   np.cos(TA),
            TOF
        ])
        
        # 3. Extract target residuals (Y)
        # Columns 7 to 12 correspond to: [err_SMA, err_ECC, err_INC, err_RAAN, err_AOP, err_TA]
        Y_raw = data[:, 7:13]
        
        # 4. Calculate normalization statistics (Z-score standardization)
        # Epsilon (1e-8) added to standard deviation to prevent division by zero
        self.X_mean = np.mean(X_raw, axis=0)
        self.X_std  = np.std(X_raw, axis=0) + 1e-8  
        
        self.Y_mean = np.mean(Y_raw, axis=0)
        self.Y_std  = np.std(Y_raw, axis=0) + 1e-8
        
        # 5. Apply normalization
        # Crucial for COE parameters since SMA is in millions of meters while ECC is ~0.01
        X_norm = (X_raw - self.X_mean) / self.X_std
        Y_norm = (Y_raw - self.Y_mean) / self.Y_std
        
        # 6. Convert to PyTorch tensors
        self.X = torch.tensor(X_norm, dtype=torch.float32)
        self.Y = torch.tensor(Y_norm, dtype=torch.float32)

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.X)

    def __getitem__(self, idx):
        """Returns a single normalized (Feature, Target) tensor pair."""
        return self.X[idx], self.Y[idx]

    def unnormalize_y(self, y_tensor: torch.Tensor):
        """
        Reverts the neural network's normalized prediction back into 
        actual physical units (meters, radians, etc.).
        """
        y_np = y_tensor.detach().cpu().numpy()
        return (y_np * self.Y_std) + self.Y_mean


def getDataloaders(csv_file, batch_size = 32, train_split = 0.8):
    """
    Creates and returns the training and validation DataLoaders.
    
    Inputs:
        csv_file   : Path to the specific regime CSV dataset.
        batch_size : Number of samples processed before updating the model's weights. 
                     Default is 32 (standard baseline for memory efficiency), though 
                     higher values (e.g., 256, 512) are usually passed during execution 
                     for smoother gradient descent in large continuous datasets.
        train_split: Fraction of the dataset allocated to the training phase. 
                     Default is 0.8 (80% training, 20% validation), adhering to the 
                     industry-standard Pareto principle to ensure a robust evaluation 
                     of the model's generalization capabilities without overfitting.

    Outputs:
        train_loader: PyTorch DataLoader for the training loop
        val_loader  : PyTorch DataLoader for the validation loop
        full_dataset: The initialized OrbitalDataset object (useful for retrieving norm stats)
    """
    full_dataset = OrbitalDataset(csv_file)
    
    # Calculate dataset split sizes
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # Split the dataset randomly to ensure unbiased training/validation
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create the data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle = False)
    
    return train_loader, val_loader, full_dataset