"""
Script: train.py

Description:
    Standard PyTorch training loop for the ORBITA ResidualPredictor.
    
    Features flight-grade training dynamics:
    Loads a dynamically generated regime-specific dataset, trains the 
    deep neural network to predict the 6 Modified Equinoctial Elements (MEE) 
    residual errors, and uses an aggressive learning rate scheduler 
    (ReduceLROnPlateau) to squeeze the maximum float32 precision 
    out of the network before saving the best model weights.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from ml.dataset import get_dataloaders
from ml.active_learning import ResidualPredictor

def train_model(csv_file, epochs=150, batch_size=512, lr=1e-3):
    """
    Executes the training and validation loop for the neural network.
    Automatically generates the output .pth filename based on the input dataset.
    
    Args:
        csv_file (str): Path to the training dataset CSV.
        epochs (int): Maximum number of training epochs.
        batch_size (int): Number of samples per batch (optimized at 512 for stability).
        lr (float): Initial learning rate for the Adam optimizer.
    """
    
    # Dynamically determine the model save path based on the CSV filename
    base_name = os.path.basename(csv_file)
    name_without_ext = os.path.splitext(base_name)[0]

    # Replace 'dataset' with 'predictor' for the saved model weights
    model_filename = name_without_ext.replace("dataset", "predictor") + ".pth"
    model_save_path = f"models/{model_filename}"

    print("-" * 80)
    print(" Starting neural network training (expert regime)")
    print(f" Input dataset : {csv_file}")
    print(f" Output model  : {model_save_path}")
    print("-" * 80)
    
    # 1. Load data
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Dataset not found at {csv_file}. Run generate_dataset.py first.")
        
    train_loader, val_loader, dataset = get_dataloaders(csv_file, batch_size=batch_size)
    print(f" [info] Training samples: {len(train_loader.dataset)} | Validation samples: {len(val_loader.dataset)}\n")
    
    # 2. Initialize network, loss function, and optimizer
    # 🚨 Explicitly setting input_size=8 for the new MEE 8-variable array
    model = ResidualPredictor(input_size=8)
    criterion = nn.MSELoss()  

    # L2 regularization (weight_decay) kept extremely low (1e-6) to allow 
    # the network to fit the exact deterministic physics without artificial smoothing
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    
    # Learning rate scheduler:
    # Reduces LR by 50% if validation loss plateaus for 10 epochs.
    # Stops dropping at 1e-7 to avoid float32 precision breakdown.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=10, 
        min_lr=1e-7
    )
    
    best_val_loss = float('inf')
    os.makedirs("models", exist_ok=True)

    # Console output header
    print(f"{'Epoch':<7} | {'Train Loss':<12} | {'Validation Loss':<12} | {'Learning Rate':<10} | {'Status'}")
    print("-" * 80)

    # 3. Epoch loop
    for epoch in range(epochs):

        # --- Training phase ---
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()                  # Clear old gradients
            predictions = model(batch_x)           # Forward pass
            loss = criterion(predictions, batch_y) # Compute error
            loss.backward()                        # Backward pass
            optimizer.step()                       # Update weights
            
            train_loss += loss.item() * batch_x.size(0)
            
        train_loss /= len(train_loader.dataset)
        

        # --- Validation phase ---
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad(): # Disable gradient calculation for validation
            for batch_x, batch_y in val_loader:
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        # Capture current learning rate before scheduler updates it
        current_lr = optimizer.param_groups[0]['lr']
        
        # Trigger scheduler
        # The scheduler monitors the validation loss. If it stops improving, it cuts the LR.
        scheduler.step(val_loss)
        
        # 4. Save the best model dynamically
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            status_mark = "[saved]" 
            
        # 5. Print progress every 10 epochs or if a new best is found in late stages
        if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch > 100 and status_mark == "[saved]"):
            print(f"{(epoch+1):03d}/{epochs} | {train_loss:.7f}    | {val_loss:.7f}       | {current_lr:.2e}      | {status_mark}")
            status_mark = "[not saved]"
            
    print("-" * 80)
    print(f" Training complete. Best weights saved to '{model_save_path}'")
    print(f" Ultimate validation loss achieved: {best_val_loss:.8f}")


# =============================================================================
# Execution Block
# =============================================================================
if __name__ == "__main__":
    # Define the specific target dataset generated by generate_dataset.py
    target_csv = "data/orbita_dataset_300-400_0.00-0.10_0-90.csv"
    
    train_model(csv_file=target_csv)