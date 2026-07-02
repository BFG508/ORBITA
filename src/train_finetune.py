"""
Script: train_finetune.py

Description:
    Implements Continual Learning (Fine-Tuning) for the ORBITA framework.

    This module loads a pre-trained Mixture of Experts (MoE) neural network
    and subjects it to a highly controlled retraining loop. By using a newly
    mined dataset (comprising high-uncertainty 'hard cases' and a low-uncertainty
    'replay buffer'), the model adjusts its boundary approximations without
    inducing catastrophic forgetting of the foundational astrodynamics.

    Includes Early Stopping and TensorBoard logging for training diagnostics.
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from benchmark import get_model_instance
from ml.dataset import OrbitalDataset, get_dataloaders


def fine_tune_model(
    base_model_path,
    fine_tune_csv,
    epochs=50,
    batch_size=256,
    lr=1e-5,
    early_stopping_patience=10,
    model_type="resnet",
):
    """
    Executes the fine-tuning sequence for a pre-trained expert network.

    Args:
        base_model_path (str): File path to the baseline model weights (.pth).
        fine_tune_csv (str): File path to the active learning dataset.
        epochs (int): Maximum number of training iterations.
        batch_size (int): Number of propagated states processed simultaneously.
        lr (float): Severely restricted initial learning rate to preserve
                    previously established internal representations.
        early_stopping_patience (int): Epochs without improvement before
                    terminating. Set to `epochs` to disable.
    """
    # 1. Pre-Flight Checks & File Routing
    if not os.path.exists(base_model_path):
        raise FileNotFoundError(f"Base model not found at {base_model_path}")
    if not os.path.exists(fine_tune_csv):
        raise FileNotFoundError(
            f"Fine-tuning dataset not found at {fine_tune_csv}"
        )

    # Normalize paths for cross-platform consistency
    base_model_path = base_model_path.replace("\\", "/")
    fine_tune_csv = fine_tune_csv.replace("\\", "/")

    # Safely construct the upgraded model's filename
    base_name = os.path.basename(base_model_path)
    name_without_ext = os.path.splitext(base_name)[0]

    if not name_without_ext.endswith("_finetuned"):
        new_model_name = name_without_ext + "_finetuned.pth"
    else:
        new_model_name = name_without_ext + ".pth"

    model_save_path = os.path.join(os.path.dirname(base_model_path), new_model_name)

    if os.path.exists(model_save_path):
        print(
            f" [info] Skipping continual learning: {model_save_path} already exists."
        )
        return

    print("-" * 80)
    print(" INITIATING CONTINUAL LEARNING (FINE-TUNING)")
    print(f" Base model     : {base_model_path}")
    print(f" Update dataset : {fine_tune_csv}")
    print(f" Output model   : {model_save_path}")
    print(f" Early stopping : patience={early_stopping_patience} epochs")
    print("-" * 80)

    # 2. Data Ingestion & Architecture Initialization
    # Infer the base dataset path to extract original normalization bounds
    params_str = name_without_ext.replace("orbita_predictor_", "").replace(
        "_finetuned", ""
    )

    parts = params_str.split("_")
    if len(parts) == 4:
        domain_str = f"{parts[1]}_{parts[2]}_{parts[3]}"
    else:
        domain_str = params_str

    base_csv_path = f"data/orbita_dataset_{domain_str}.csv"

    if not os.path.exists(base_csv_path):
        raise FileNotFoundError(
            "Original dataset required for normalization missing: "
            f"{base_csv_path}"
        )

    print(
        " [info] Extracting frozen normalization statistics"
        " from base dataset..."
    )
    base_dataset = OrbitalDataset(base_csv_path)
    frozen_stats = {
        "x_mean": base_dataset.x_mean,
        "x_std": base_dataset.x_std,
        "y_mean": base_dataset.y_mean,
        "y_std": base_dataset.y_std,
    }

    # Load the fine-tuning dataset with enforced base scaling
    train_loader, val_loader, dataset = get_dataloaders(
        fine_tune_csv, batch_size=batch_size, base_stats=frozen_stats
    )
    print(
        f" [info] Fine-tuning samples: {len(train_loader.dataset)} "
        f"| Validation samples: {len(val_loader.dataset)}\n"
    )

    # Instantiate the architecture and load the baseline knowledge
    model = get_model_instance(model_type)
    model.load_state_dict(torch.load(base_model_path, weights_only=True))

    # MSE evaluates the residual dispersion in the MEE domain
    criterion = nn.MSELoss()

    # 3. Optimizer & Learning Rate Scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-8
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    # 4. TensorBoard logging
    log_dir = f"logs/finetune_{params_str}"
    writer = SummaryWriter(log_dir=log_dir)

    # Console output header for telemetry tracking
    print(
        f"{'Epoch':<6} | {'Train Loss':<12} | {'Validation Loss':<12}"
        f" | {'Learning Rate':<10} | {'Status'}"
    )
    print("-" * 80)

    # 5. Fine-Tuning Loop
    for epoch in range(epochs):

        # --- Forward / Backward Pass (Training Phase) ---
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)

        train_loss /= len(train_loader.dataset)

        # --- Model Evaluation (Validation Phase) ---
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item() * batch_x.size(0)

        val_loss /= len(val_loader.dataset)

        # Advance the scheduler based on validation performance
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        # Log metrics to TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("LearningRate", current_lr, epoch)

        # 6. Save the best model dynamically
        status_mark = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            status_mark = "[saved]"
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # 7. Print progress
        show_progress = (
            (epoch + 1) % 5 == 0
            or epoch == 0
            or (epoch > 30 and status_mark == "[saved]")
        )
        if show_progress:
            print(
                f"{(epoch + 1):03d}/{epochs} | {train_loss:.7f}    "
                f"| {val_loss:.7f}       | {current_lr:.2e}  "
                f"    | {status_mark}"
            )

        # 8. Early stopping check
        if epochs_without_improvement >= early_stopping_patience:
            print(
                f"\n [early stop] No improvement for "
                f"{early_stopping_patience} epochs. "
                f"Stopping at epoch {epoch + 1}."
            )
            break

    writer.close()

    print("-" * 80)
    print(
        f" Fine-tuning complete. Updated weights saved to"
        f" '{model_save_path}'"
    )
    print(f" Ultimate validation loss achieved: {best_val_loss:.8f}")


# =============================================================================
# EXECUTION BLOCK
# =============================================================================
if __name__ == "__main__":
    target_base_model = (
        "models/orbita_predictor_resnet_300-2000_0.0000-0.1000_0-90.pth"
    )
    target_finetune_data = (
        "data/orbita_finetune_resnet_300-2000_0.0000-0.1000_0-90.csv"
    )

    fine_tune_model(
        base_model_path=target_base_model,
        fine_tune_csv=target_finetune_data,
        epochs=50,  # Accelerated cycle limit
        lr=1e-5,  # Restricted learning threshold
    )
