"""
Script: test_architecture.py

Description:
    Tests for the ORBITA neural network architectures.

    Verifies:
    1. Forward pass produces correct output shapes for all architectures.
    2. MC-Dropout uncertainty estimation returns valid mean and std.
    3. The predict_with_uncertainty method correctly restores eval() mode.
    4. All architectures handle batched and single-sample inputs.
"""

import sys
import os
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ml.architecture import (
    ResidualPredictor,
    LinearBaseline,
    MLPPredictor,
    LSTMPredictor,
)


# =============================================================================
# FIXTURES
# =============================================================================

INPUT_SIZE = 8   # 6 MEE elements + sin(L)/cos(L) replacement + TOF
OUTPUT_SIZE = 6  # 6 MEE residual errors
BATCH_SIZES = [1, 16, 512]


ALL_ARCHITECTURES = [
    ("ResNet", ResidualPredictor),
    ("Linear", LinearBaseline),
    ("MLP", MLPPredictor),
    ("LSTM", LSTMPredictor),
]


# =============================================================================
# FORWARD PASS SHAPE TESTS
# =============================================================================

class TestForwardPassShapes:
    """Verify every architecture produces correctly shaped output tensors."""

    @pytest.mark.parametrize(
        "name, model_cls", ALL_ARCHITECTURES,
        ids=[a[0] for a in ALL_ARCHITECTURES]
    )
    @pytest.mark.parametrize("batch_size", BATCH_SIZES, ids=str)
    def test_output_shape(self, name, model_cls, batch_size):
        """Output tensor must be (batch_size, 6) for all architectures."""
        model = model_cls(input_size=INPUT_SIZE)
        model.eval()

        x = torch.randn(batch_size, INPUT_SIZE)
        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, OUTPUT_SIZE), (
            f"{name}: Expected shape ({batch_size}, {OUTPUT_SIZE}), "
            f"got {output.shape}"
        )

    @pytest.mark.parametrize(
        "name, model_cls", ALL_ARCHITECTURES,
        ids=[a[0] for a in ALL_ARCHITECTURES]
    )
    def test_output_is_finite(self, name, model_cls):
        """Output must contain no NaN or Inf values on random input."""
        model = model_cls(input_size=INPUT_SIZE)
        model.eval()

        x = torch.randn(32, INPUT_SIZE)
        with torch.no_grad():
            output = model(x)

        assert torch.isfinite(output).all(), (
            f"{name}: Output contains NaN or Inf"
        )


# =============================================================================
# MC-DROPOUT UNCERTAINTY TESTS
# =============================================================================

class TestMCDropoutUncertainty:
    """Verify MC-Dropout uncertainty estimation on ResidualPredictor."""

    def test_mean_shape(self):
        """Mean prediction must have shape (batch, 6)."""
        model = ResidualPredictor(input_size=INPUT_SIZE)
        x = torch.randn(16, INPUT_SIZE)

        mean, std = model.predict_with_uncertainty(x, num_samples=10)

        assert mean.shape == (16, OUTPUT_SIZE)

    def test_std_shape(self):
        """Uncertainty (std) must have shape (batch, 6)."""
        model = ResidualPredictor(input_size=INPUT_SIZE)
        x = torch.randn(16, INPUT_SIZE)

        mean, std = model.predict_with_uncertainty(x, num_samples=10)

        assert std.shape == (16, OUTPUT_SIZE)

    def test_std_is_non_negative(self):
        """Standard deviation must be >= 0 everywhere."""
        model = ResidualPredictor(input_size=INPUT_SIZE)
        x = torch.randn(32, INPUT_SIZE)

        _, std = model.predict_with_uncertainty(x, num_samples=20)

        assert (std >= 0).all(), "Uncertainty contains negative values"

    def test_eval_mode_restored_after_uncertainty(self):
        """
        After calling predict_with_uncertainty, the model must be
        back in eval() mode (training=False). This was a P0 bug fix.
        """
        model = ResidualPredictor(input_size=INPUT_SIZE)
        model.eval()

        x = torch.randn(8, INPUT_SIZE)
        model.predict_with_uncertainty(x, num_samples=5)

        assert not model.training, (
            "Model is still in training mode after predict_with_uncertainty. "
            "The P0 fix for restoring eval() may have regressed."
        )

    def test_deterministic_in_eval(self):
        """
        In eval mode (no MC-Dropout), two forward passes on the same
        input must produce identical results.
        """
        model = ResidualPredictor(input_size=INPUT_SIZE)
        model.eval()

        x = torch.randn(8, INPUT_SIZE)

        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)

        torch.testing.assert_close(
            out1, out2,
            msg="Eval mode predictions are not deterministic"
        )

    def test_stochastic_in_train(self):
        """
        In train mode (MC-Dropout active), two forward passes on the
        same input should generally produce different results due to
        dropout randomness.
        """
        model = ResidualPredictor(input_size=INPUT_SIZE)
        model.train()

        x = torch.randn(32, INPUT_SIZE)

        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)

        # It's theoretically possible for them to be equal, but
        # with dropout on a 32-sample batch, practically impossible
        assert not torch.equal(out1, out2), (
            "Train mode predictions are identical — dropout may not "
            "be functioning"
        )


# =============================================================================
# GRADIENT FLOW TESTS
# =============================================================================

class TestGradientFlow:
    """Verify that gradients flow through all architectures."""

    @pytest.mark.parametrize(
        "name, model_cls", ALL_ARCHITECTURES,
        ids=[a[0] for a in ALL_ARCHITECTURES]
    )
    def test_backward_pass_produces_gradients(self, name, model_cls):
        """A backward pass must populate gradients on all parameters."""
        model = model_cls(input_size=INPUT_SIZE)
        model.train()

        x = torch.randn(16, INPUT_SIZE)
        target = torch.randn(16, OUTPUT_SIZE)

        output = model(x)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()

        for param_name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, (
                    f"{name}: No gradient for parameter '{param_name}'"
                )
                assert torch.isfinite(param.grad).all(), (
                    f"{name}: Non-finite gradient in '{param_name}'"
                )
