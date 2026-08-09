"""
Script: train_base.py

Description:
    Standard PyTorch training loop for the ORBITA framework.

    Features flight-grade training dynamics:
    Loads a dynamically generated regime-specific dataset, trains the
    selected neural network architecture to predict the 6 Modified Equinoctial
    Elements (MEE) residual errors, and uses an aggressive learning rate
    scheduler (ReduceLROnPlateau) to squeeze the maximum float32 precision
    out of the network before saving the best model weights.

    Includes Early Stopping to avoid wasted computation when the model
    converges, and TensorBoard logging for interactive training diagnostics.

    Upgraded to support Ablation Studies: Dynamically instantiates different
    baseline architectures (ResNet, Linear, MLP, LSTM, Tree) via
    command-line arguments.
"""

import argparse
import csv
import logging
import os
import time

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logging.getLogger("codecarbon").disabled = True  # noqa: E402
from codecarbon import EmissionsTracker  # noqa: E402
from torch.utils.tensorboard import SummaryWriter

from ml.architecture import (
    LinearBaseline,
    LSTMPredictor,
    MLPPredictor,
    ResidualPredictor,
    TreeBaseline,
)
from ml.dataset import get_dataloaders


def _select_training_device(requested_device="auto"):
    """
    Selects the PyTorch device used by neural architectures.
    """
    requested_device = requested_device.lower()
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but PyTorch cannot initialize a CUDA device."
        )

    return torch.device(requested_device)


def _log_training_metrics(model_type, training_time, val_loss, model_path):
    """
    Appends a row to data/metrics_train.csv with the
    performance metrics collected during training.

    Args:
        model_type (str): Architecture identifier.
        training_time (float): Wall-clock training time [s].
        val_loss (float): Best validation MSE loss achieved.
        model_path (str): Path to saved model file.
    """
    metrics_dir = os.path.join("data", "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_file = os.path.join(metrics_dir, "metrics_train.csv")
    header = [
        "architecture",
        "training_time_s",
        "val_loss",
        "model_size_mb",
    ]

    write_header = not os.path.exists(metrics_file)
    size_mb = os.path.getsize(model_path) / (1024 * 1024)

    with open(metrics_file, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(
            [
                model_type,
                f"{training_time:.2f}",
                f"{val_loss:.8f}",
                f"{size_mb:.4f}",
            ]
        )

    print(
        f" [metrics] Logged to {metrics_file}: "
        f"{model_type} | {training_time:.2f}s | "
        f"val_loss={val_loss:.8f} | {size_mb:.4f} MB"
    )


def train_model(
    csv_file,
    model_type="resnet",
    epochs=150,
    batch_size=512,
    lr=1e-3,
    early_stopping_patience=20,
    model_dir="models",
    device="auto",
):
    """
    Executes the training and validation loop for the selected model.
    Automatically generates the output filename based on the input
    dataset and the chosen architecture.

    Args:
        csv_file (str): Path to the training dataset CSV.
        model_type (str): Architecture to train
            ('resnet', 'linear', 'mlp', 'lstm', 'tree').
        epochs (int): Maximum number of training epochs.
        batch_size (int): Number of samples per batch.
        lr (float): Initial learning rate for the Adam optimizer.
        early_stopping_patience (int): Number of epochs without
            val_loss improvement before terminating training
            early. Set to `epochs` to effectively disable it.
        model_dir (str): Directory where model weights are saved.
        device (str): 'auto', 'cpu', or 'cuda' for neural architectures.
    """

    # 1. Dynamically determine the model save path
    base_name = os.path.basename(csv_file)
    name_without_ext = os.path.splitext(base_name)[0]

    if model_dir == "models" or model_dir == f"models/{model_type}":
        if model_type == "tree":
            model_dir = os.path.join("models", "tree")
        else:
            model_dir = os.path.join("models", model_type, "base")
    os.makedirs(model_dir, exist_ok=True)

    # Differentiate saved weights by architecture type to avoid
    # overwriting. Tree models use .joblib instead of .pth.
    target_prefix = f"predictor_{model_type}"
    if model_type == "tree":
        model_filename = (
            name_without_ext.replace("dataset", target_prefix) + ".joblib"
        )
    else:
        model_filename = (
            name_without_ext.replace("dataset", target_prefix) + ".pth"
        )
    model_save_path = os.path.join(model_dir, model_filename)

    if os.path.exists(model_save_path):
        # Even though training is skipped, ensure that this
        # model's metrics are present in the CSV for downstream
        # analytics.  If the row already exists, _log_training_metrics
        # will simply append a duplicate that the visualiser
        # de-duplicates automatically (keep="last").
        metrics_file = os.path.join("data", "metrics", "metrics_train.csv")
        already_logged = False
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as mf:
                already_logged = model_type in mf.read()

        if not already_logged:
            print(
                f" [info] Model exists ({model_save_path})."
                f" Logging missing metrics..."
            )
            _log_training_metrics(model_type, 0.0, 0.0, model_save_path)
        else:
            print(
                f" [info] Skipping training: "
                f"{model_save_path} already exists."
            )
        return

    print("-" * 80)
    print(f" Starting training (Architecture: {model_type.upper()})")
    print(f" Input dataset : {csv_file}")
    print(f" Output model  : {model_save_path}")
    print(f" Early stopping: patience={early_stopping_patience} epochs")
    print("-" * 80)

    # 2. Load data
    if not os.path.exists(csv_file):
        raise FileNotFoundError(
            f"Dataset not found at {csv_file}. "
            "Run generate_dataset.py first."
        )

    train_loader, val_loader, dataset = get_dataloaders(
        csv_file, batch_size=batch_size
    )
    print(
        f" [info] Training samples: {len(train_loader.dataset)}"
        f" | Validation samples: {len(val_loader.dataset)}\n"
    )

    # 3. Instantiate the requested model architecture
    model_type = model_type.lower()
    if model_type == "resnet":
        model = ResidualPredictor(input_size=8)
    elif model_type == "linear":
        model = LinearBaseline(input_size=8)
    elif model_type == "mlp":
        model = MLPPredictor(input_size=8)
    elif model_type == "lstm":
        model = LSTMPredictor(input_size=8)
    elif model_type == "tree":
        model = TreeBaseline(input_size=8)
    else:
        raise ValueError(
            f"Unsupported model_type: '{model_type}'. "
            "Choose from: resnet, linear, mlp, lstm, tree."
        )

    training_device = None
    if model_type != "tree":
        training_device = _select_training_device(device)
        model = model.to(training_device)
        print(f" [info] Training device: {training_device}")

    # =========================================================
    # TREE BRANCH: scikit-learn training path
    # =========================================================
    if model_type == "tree":
        tracker = EmissionsTracker(
            project_name=(f"train_{model_type}_{name_without_ext}"),
            log_level="error",
            output_dir="data",
        )
        tracker.start()
        train_start = time.time()

        # Extract all samples as NumPy arrays
        x_all = dataset.x.numpy()
        y_all = dataset.y.numpy()
        model.fit(x_all, y_all)

        # Evaluate on validation set
        val_x = np.vstack(
            [dataset.x[i].numpy() for i in val_loader.dataset.indices]
        )
        val_y = np.vstack(
            [dataset.y[i].numpy() for i in val_loader.dataset.indices]
        )
        val_pred = model.tree.predict(val_x)
        val_mse = float(np.mean((val_pred - val_y) ** 2))

        elapsed = time.time() - train_start
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(model, model_save_path)

        log_dir = f"logs/training/tree/{name_without_ext}_tree"
        writer = SummaryWriter(log_dir=log_dir)
        writer.add_scalar("Loss/val", val_mse, 1)
        writer.close()

        print("-" * 80)
        print(f" Tree training complete in {elapsed:.2f}s.")
        print(f" Validation MSE : {val_mse:.8f}")
        print(f" Model saved to : {model_save_path}")
        print(f" TensorBoard logs: {log_dir}")
        print("-" * 80)
        tracker.stop()
        _log_training_metrics(model_type, elapsed, val_mse, model_save_path)
        return

    criterion = nn.MSELoss()

    # L2 regularization (weight_decay) kept extremely low (1e-6) to allow
    # the network to fit the exact deterministic physics without smoothing
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

    # Learning rate scheduler:
    # Reduces LR by 50% if validation loss plateaus for 10 epochs.
    # Stops dropping at 1e-7 to avoid float32 precision breakdown.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-7
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    os.makedirs(model_dir, exist_ok=True)

    # 4. TensorBoard logging
    log_dir = f"logs/training/{model_type}/{name_without_ext}_{model_type}"
    writer = SummaryWriter(log_dir=log_dir)
    print(f" [info] TensorBoard logs: {log_dir}\n")

    # Console output header
    print(
        f"{'Epoch':<7} | {'Train Loss':<12} | {'Validation Loss':<12}"
        f" | {'Learning Rate':<10} | {'Status'}"
    )
    print("-" * 80)

    # 5. Epoch loop
    tracker = EmissionsTracker(
        project_name=(f"train_{model_type}_{name_without_ext}"),
        log_level="error",
        output_dir="data",
    )
    tracker.start()
    train_start = time.time()

    for epoch in range(epochs):

        # --- Training phase ---
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(training_device)
            batch_y = batch_y.to(training_device)
            optimizer.zero_grad()  # Clear old gradients
            predictions = model(batch_x)  # Forward pass
            loss = criterion(predictions, batch_y)  # Compute error
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            train_loss += loss.item() * batch_x.size(0)

        train_loss /= len(train_loader.dataset)

        # --- Validation phase ---
        model.eval()
        val_loss = 0.0

        with torch.no_grad():  # Disable gradient calc for validation
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(training_device)
                batch_y = batch_y.to(training_device)
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item() * batch_x.size(0)

        val_loss /= len(val_loader.dataset)

        # Capture current learning rate before scheduler updates it
        current_lr = optimizer.param_groups[0]["lr"]

        # Trigger scheduler
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
            (epoch + 1) % 10 == 0
            or epoch == 0
            or (epoch > 100 and status_mark == "[saved]")
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

    elapsed = time.time() - train_start

    print("-" * 80)
    print(f" Training complete. Best weights saved to '{model_save_path}'")
    print(f" Ultimate validation loss achieved: {best_val_loss:.8f}")
    print(f" Total training time: {elapsed:.2f}s")

    tracker.stop()
    _log_training_metrics(model_type, elapsed, best_val_loss, model_save_path)


# =============================================================================
# EXECUTION BLOCK
# =============================================================================
if __name__ == "__main__":
    # Setup argparse for Ablation Study execution
    parser = argparse.ArgumentParser(description="ORBITA Model Training Suite")

    parser.add_argument(
        "--dataset",
        type=str,
        default="data/orbita_dataset_300-2000_0.0000-0.1000_0-90.csv",
        help="Path to the generated training dataset.",
    )

    parser.add_argument(
        "--model_type",
        type=str,
        choices=["resnet", "linear", "mlp", "lstm", "tree"],
        default="resnet",
        help="Select the neural architecture to train.",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Select the PyTorch training device for neural models.",
    )

    args = parser.parse_args()

    # Execute training with the selected parameters
    train_model(
        csv_file=args.dataset,
        model_type=args.model_type,
        device=args.device,
    )
