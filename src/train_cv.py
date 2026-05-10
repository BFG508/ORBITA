"""
Script: train_cv.py

Description:
    K-Fold Cross-Validation suite for the ORBITA framework.

    This script provides statistically rigorous validation of each
    neural network architecture by splitting the training dataset
    into K folds, training K temporary models, and aggregating
    their validation losses to produce mean and standard deviation
    metrics. This ensures that reported performance is not an
    artifact of a single lucky train/validation split.

    Results are exported to data/metrics_cv.csv for downstream
    visualization by the ORBITA Analytics Engine.

    Supports all architectures: ResNet, MLP, LSTM, Linear, Tree.
"""

import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold

from ml.architecture import (
    LinearBaseline,
    LSTMPredictor,
    MLPPredictor,
    ResidualPredictor,
    TreeBaseline,
)
from ml.dataset import OrbitalDataset


def _get_model(model_type):
    """
    Instantiates a fresh model of the requested architecture.

    Args:
        model_type (str): One of 'resnet', 'mlp', 'lstm',
            'linear', 'tree'.

    Returns:
        Model instance (nn.Module or TreeBaseline).
    """
    model_type = model_type.lower()
    if model_type == "resnet":
        return ResidualPredictor(input_size=8)
    elif model_type == "linear":
        return LinearBaseline(input_size=8)
    elif model_type == "mlp":
        return MLPPredictor(input_size=8)
    elif model_type == "lstm":
        return LSTMPredictor(input_size=8)
    elif model_type == "tree":
        return TreeBaseline(input_size=8)
    else:
        raise ValueError(
            f"Unsupported model_type: '{model_type}'. "
            "Choose from: resnet, linear, mlp, lstm, tree."
        )


def _train_pytorch_fold(
    model,
    x_train,
    y_train,
    x_val,
    y_val,
    epochs=100,
    lr=1e-3,
    patience=15,
):
    """
    Trains a PyTorch model on one fold and returns the
    best validation MSE achieved.

    Args:
        model (nn.Module): Fresh model instance.
        x_train (torch.Tensor): Training features.
        y_train (torch.Tensor): Training targets.
        x_val (torch.Tensor): Validation features.
        y_val (torch.Tensor): Validation targets.
        epochs (int): Maximum training epochs.
        lr (float): Initial learning rate.
        patience (int): Early stopping patience.

    Returns:
        float: Best validation MSE loss for this fold.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
        min_lr=1e-7,
    )

    best_val_loss = float("inf")
    no_improve = 0

    train_ds = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=512, shuffle=True
    )

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

        # --- Validate ---
        model.eval()
        with torch.no_grad():
            val_pred = model(x_val)
            val_loss = criterion(val_pred, y_val).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    return best_val_loss


def _train_tree_fold(x_train, y_train, x_val, y_val):
    """
    Fits a DecisionTreeRegressor on one fold and returns
    the validation MSE.

    Args:
        x_train (np.ndarray): Training features.
        y_train (np.ndarray): Training targets.
        x_val (np.ndarray): Validation features.
        y_val (np.ndarray): Validation targets.

    Returns:
        float: Validation MSE for this fold.
    """
    tree = TreeBaseline(input_size=8)
    tree.fit(x_train, y_train)
    pred = tree.tree.predict(x_val)
    mse = float(np.mean((pred - y_val) ** 2))
    return mse


def run_cross_validation(
    csv_file,
    model_type="resnet",
    k_folds=5,
    epochs=100,
    lr=1e-3,
    patience=15,
):
    """
    Executes K-Fold Cross-Validation for the selected
    architecture on the given dataset.

    Args:
        csv_file (str): Path to the training dataset CSV.
        model_type (str): Architecture to evaluate.
        k_folds (int): Number of cross-validation folds.
        epochs (int): Max epochs per fold (PyTorch only).
        lr (float): Learning rate (PyTorch only).
        patience (int): Early stopping patience per fold.

    Returns:
        tuple: (mean_loss, std_loss) across all K folds.
    """
    print("=" * 80)
    print(
        f" ORBITA CROSS-VALIDATION | ARCH: {model_type.upper()}"
        f" | K={k_folds}"
    )
    print("=" * 80)

    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Dataset not found: {csv_file}")

    # Load and normalize the full dataset
    dataset = OrbitalDataset(csv_file)
    x_all = dataset.x.numpy()
    y_all = dataset.y.numpy()

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_losses = []
    total_start = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(x_all), start=1):
        x_tr = x_all[train_idx]
        y_tr = y_all[train_idx]
        x_va = x_all[val_idx]
        y_va = y_all[val_idx]

        print(
            f"\n Fold {fold_idx}/{k_folds} "
            f"(train={len(train_idx)}, val={len(val_idx)})"
        )

        fold_start = time.time()

        if model_type == "tree":
            val_loss = _train_tree_fold(x_tr, y_tr, x_va, y_va)
        else:
            model = _get_model(model_type)
            x_tr_t = torch.tensor(x_tr, dtype=torch.float32)
            y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
            x_va_t = torch.tensor(x_va, dtype=torch.float32)
            y_va_t = torch.tensor(y_va, dtype=torch.float32)
            val_loss = _train_pytorch_fold(
                model,
                x_tr_t,
                y_tr_t,
                x_va_t,
                y_va_t,
                epochs=epochs,
                lr=lr,
                patience=patience,
            )

        fold_time = time.time() - fold_start
        fold_losses.append(val_loss)
        print(f"   -> Val MSE: {val_loss:.8f} " f"({fold_time:.1f}s)")

    total_time = time.time() - total_start
    mean_loss = float(np.mean(fold_losses))
    std_loss = float(np.std(fold_losses))

    print("\n" + "-" * 80)
    print(f" CROSS-VALIDATION RESULTS ({model_type.upper()})")
    print("-" * 80)
    print(f" Folds          : {k_folds}")
    print(f" Mean Val MSE   : {mean_loss:.8f}")
    print(f" Std Val MSE    : {std_loss:.8f}")
    print(f" Total Time     : {total_time:.2f}s")
    print(f" Per-fold losses: " f"{[f'{v:.8f}' for v in fold_losses]}")
    print("-" * 80)

    # Log results to CSV
    _log_cv_metrics(
        model_type,
        k_folds,
        mean_loss,
        std_loss,
        fold_losses,
        total_time,
    )

    return mean_loss, std_loss


def _log_cv_metrics(
    model_type,
    k_folds,
    mean_loss,
    std_loss,
    fold_losses,
    total_time,
):
    """
    Appends cross-validation results to
    data/metrics_cv.csv.

    Args:
        model_type (str): Architecture identifier.
        k_folds (int): Number of folds used.
        mean_loss (float): Mean validation MSE.
        std_loss (float): Standard deviation of MSE.
        fold_losses (list[float]): Individual fold losses.
        total_time (float): Wall-clock time for all folds.
    """
    os.makedirs("data", exist_ok=True)
    metrics_file = "data/metrics_cv.csv"
    header = [
        "architecture",
        "k_folds",
        "mean_val_mse",
        "std_val_mse",
        "fold_losses",
        "total_time_s",
    ]

    write_header = not os.path.exists(metrics_file)
    fold_str = ";".join(f"{v:.8f}" for v in fold_losses)

    with open(metrics_file, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(
            [
                model_type,
                k_folds,
                f"{mean_loss:.8f}",
                f"{std_loss:.8f}",
                fold_str,
                f"{total_time:.2f}",
            ]
        )

    print(
        f" [metrics] Logged to {metrics_file}: "
        f"{model_type} | K={k_folds} | "
        f"mean={mean_loss:.8f} +/- {std_loss:.8f}"
    )


# =============================================================================
# EXECUTION BLOCK
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ORBITA K-Fold Cross-Validation Suite"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=("data/orbita_dataset" "_300-2000_0.0000-0.1000_0-90.csv"),
        help="Path to the training dataset CSV.",
    )

    parser.add_argument(
        "--model_type",
        type=str,
        choices=[
            "resnet",
            "linear",
            "mlp",
            "lstm",
            "tree",
        ],
        default="resnet",
        help="Architecture to cross-validate.",
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of K-Fold partitions (default: 5).",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Max epochs per fold (PyTorch models only).",
    )

    args = parser.parse_args()

    run_cross_validation(
        csv_file=args.dataset,
        model_type=args.model_type,
        k_folds=args.folds,
        epochs=args.epochs,
    )
