"""
Module: dataset.py

Description:
    Handles data loading, train/validation splitting, and feature normalization 
    for the PyTorch neural network. Configured to process Modified Equinoctial 
    Elements (MEE) inputs and residual errors for specific orbital regimes.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class OrbitalDataset(Dataset):
    """
    Custom PyTorch Dataset for the ORBITA framework.
    Loads Modified Equinoctial Elements (MEE) inputs (x) and residual 
    MEE errors (y) from a dynamically named CSV file.
    """
    def __init__(self, csv_file, base_stats=None):
        """
        Initializes the dataset, extracting features and targets, 
        and computing or applying normalization statistics.
        
        Args:
            csv_file (str): Path to the dataset CSV file.
            base_stats (dict, optional): Pre-computed normalization statistics 
                                         from a base dataset. If None, stats 
                                         are computed from the current file.
        """
        # Load data using NumPy (skipping the header row)
        data = np.genfromtxt(csv_file, delimiter=',', skip_header=1)
        
        # 1. Extract raw input features (X)
        # Columns 0 to 6 correspond to: [p, f, g, h, k, L, TOF]
        p = data[:, 0]
        f = data[:, 1]
        g = data[:, 2]
        h = data[:, 3]
        k = data[:, 4]
        l_true = data[:, 5]
        tof = data[:, 6]
        
        # 2. Continuous angle transformation
        # To prevent the neural network from struggling with the 360-degree 
        # jump of the true longitude (L), it is decomposed into sine and cosine.
        # This results in a stable 8-feature input tensor.
        x_raw = np.column_stack([
            p, f, g, h, k, np.sin(l_true), np.cos(l_true), tof
        ])
        
        # 3. Extract target residuals (Y)
        # Columns 7 to 12 correspond to: [err_p, err_f, err_g, err_h, err_k, err_L]
        y_raw = data[:, 7:13]
        
        # 4. Calculate normalization statistics (Z-score standardization)
        # Epsilon (1e-8) added to standard deviation to prevent division by zero
        if base_stats is None:
            # Base training mode: compute statistics dynamically
            self.x_mean = np.mean(x_raw, axis=0)
            self.x_std = np.std(x_raw, axis=0)
            self.x_std[self.x_std == 0] = 1e-8  # Prevent division by zero
            
            self.y_mean = np.mean(y_raw, axis=0)
            self.y_std = np.std(y_raw, axis=0)
            self.y_std[self.y_std == 0] = 1e-8
        else:
            # Fine-tuning mode: freeze statistics using the provided dictionary
            self.x_mean = base_stats['x_mean']
            self.x_std = base_stats['x_std']
            self.y_mean = base_stats['y_mean']
            self.y_std = base_stats['y_std']
        
        # 5. Apply normalization
        x_norm = (x_raw - self.x_mean) / self.x_std
        y_norm = (y_raw - self.y_mean) / self.y_std
        
        # 6. Convert to PyTorch tensors
        self.x = torch.tensor(x_norm, dtype=torch.float32)
        self.y = torch.tensor(y_norm, dtype=torch.float32)


    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        
        Returns:
            int: Total number of samples.
        """
        return len(self.x)


    def __getitem__(self, idx):
        """
        Returns a single normalized (feature, target) tensor pair.
        
        Args:
            idx (int): Index of the sample to retrieve.
            
        Returns:
            tuple: (x_tensor, y_tensor) corresponding to the index.
        """
        return self.x[idx], self.y[idx]


    def unnormalize_y(self, y_tensor):
        """
        Reverts the neural network's normalized prediction back into 
        actual physical units.
        
        Args:
            y_tensor (torch.Tensor): Normalized prediction tensor.
            
        Returns:
            np.ndarray: Unnormalized physical predictions.
        """
        y_np = y_tensor.detach().cpu().numpy()
        return (y_np * self.y_std) + self.y_mean


def get_dataloaders(csv_file, batch_size=32, train_split=0.8, base_stats=None):
    """
    Creates and returns the training and validation DataLoaders.
    
    Args:
        csv_file (str): Path to the specific regime CSV dataset.
        batch_size (int): Number of samples processed before updating weights.
        train_split (float): Fraction of the dataset allocated to training.
        base_stats (dict, optional): Frozen normalization parameters to prevent 
                                     covariate shift during fine-tuning.

    Returns:
        tuple: A tuple containing:
            - train_loader (DataLoader): PyTorch DataLoader for training.
            - val_loader (DataLoader): PyTorch DataLoader for validation.
            - full_dataset (OrbitalDataset): The initialized dataset object.
    """
    full_dataset = OrbitalDataset(csv_file, base_stats=base_stats)
    
    # Calculate dataset split sizes
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # Split the dataset randomly to ensure unbiased training/validation
    # A fixed seed guarantees reproducible splits across executions
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=generator
    )
    
    # Create the data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, full_dataset