"""
Tests for BasePINN abstract class.

This module tests the base PINN interface and train_step functionality.
"""

import pytest
import torch
import torch.nn as nn

from src.models.base import BasePINN


class SimplePINN(BasePINN):
    """Simple concrete implementation of BasePINN for testing."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 10):
        super().__init__(input_dim, output_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def compute_pde_residual(self, x: torch.Tensor, pde_fn=None) -> torch.Tensor:
        """Simple residual: just return zeros for testing."""
        u = self.forward(x)
        return torch.zeros_like(u)


class TestBasePINN:
    """Test suite for BasePINN abstract class."""

    def test_initialization(self):
        """Test BasePINN initialization."""
        model = SimplePINN(input_dim=2, output_dim=1)
        assert model.input_dim == 2
        assert model.output_dim == 1
        assert isinstance(model, BasePINN)
        assert isinstance(model, nn.Module)

    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        model = SimplePINN(input_dim=2, output_dim=1)
        x = torch.randn(100, 2)
        u = model(x)

        assert u.shape == (100, 1)
        assert not torch.isnan(u).any()
        assert not torch.isinf(u).any()

    def test_compute_pde_residual(self):
        """Test PDE residual computation."""
        model = SimplePINN(input_dim=2, output_dim=1)
        x = torch.randn(50, 2, requires_grad=True)
        residual = model.compute_pde_residual(x)

        assert residual.shape == (50, 1)
        assert not torch.isnan(residual).any()

    def test_train_step_basic(self):
        """Test basic train_step functionality."""
        model = SimplePINN(input_dim=2, output_dim=1)

        # Create dummy data
        x_interior = torch.randn(100, 2, requires_grad=True)
        x_boundary = torch.randn(40, 2)
        u_boundary = torch.randn(40, 1)

        # Perform train step
        losses = model.train_step(x_interior, x_boundary, u_boundary)

        # Check loss dictionary structure
        assert "loss_total" in losses
        assert "loss_pde" in losses
        assert "loss_bc" in losses
        assert "loss_ic" in losses

        # Check loss values are valid
        for key, value in losses.items():
            assert isinstance(value, torch.Tensor)
            assert value.ndim == 0  # Scalar
            assert not torch.isnan(value)
            assert value >= 0  # Losses should be non-negative

    def test_train_step_with_initial_conditions(self):
        """Test train_step with initial conditions."""
        model = SimplePINN(input_dim=2, output_dim=1)

        x_interior = torch.randn(100, 2, requires_grad=True)
        x_boundary = torch.randn(40, 2)
        u_boundary = torch.randn(40, 1)
        x_initial = torch.randn(50, 2)
        u_initial = torch.randn(50, 1)

        losses = model.train_step(
            x_interior, x_boundary, u_boundary, x_initial=x_initial, u_initial=u_initial
        )

        # IC loss should be non-zero
        assert losses["loss_ic"] > 0

    def test_train_step_custom_weights(self):
        """Test train_step with custom loss weights."""
        model = SimplePINN(input_dim=2, output_dim=1)

        x_interior = torch.randn(100, 2, requires_grad=True)
        x_boundary = torch.randn(40, 2)
        u_boundary = torch.randn(40, 1)

        weights = {"pde": 2.0, "bc": 0.5, "ic": 0.0}
        losses = model.train_step(x_interior, x_boundary, u_boundary, weights=weights)

        # Total loss should reflect custom weighting
        expected_total = (
            weights["pde"] * losses["loss_pde"] + weights["bc"] * losses["loss_bc"]
        )
        assert torch.allclose(losses["loss_total"], expected_total)

    def test_get_parameters_count(self):
        """Test parameter counting."""
        model = SimplePINN(input_dim=2, output_dim=1, hidden_dim=10)
        param_count = model.get_parameters_count()

        # Manual count: (2*10 + 10) + (10*1 + 1) = 30 + 11 = 41
        assert param_count == 41

    def test_gradient_flow(self):
        """Test that gradients flow through the network."""
        model = SimplePINN(input_dim=2, output_dim=1)

        x_interior = torch.randn(50, 2, requires_grad=True)
        x_boundary = torch.randn(20, 2)
        u_boundary = torch.randn(20, 1)

        losses = model.train_step(x_interior, x_boundary, u_boundary)
        loss = losses["loss_total"]

        # Backpropagate
        loss.backward()

        # Check that gradients exist
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()

    def test_repr(self):
        """Test string representation."""
        model = SimplePINN(input_dim=2, output_dim=1)
        repr_str = repr(model)

        assert "SimplePINN" in repr_str
        assert "input_dim=2" in repr_str
        assert "output_dim=1" in repr_str
        assert "parameters" in repr_str


class TestBasePINNAbstract:
    """Test that BasePINN cannot be instantiated directly."""

    def test_cannot_instantiate_abstract(self):
        """Test that BasePINN cannot be instantiated directly."""
        with pytest.raises(TypeError):
            # This should fail because forward() and compute_pde_residual()
            # are abstract methods
            BasePINN(input_dim=2, output_dim=1)


class TestMultipleDimensions:
    """Test BasePINN with different input/output dimensions."""

    def test_3d_input(self):
        """Test with 3D input (x, y, t)."""
        model = SimplePINN(input_dim=3, output_dim=1)
        x = torch.randn(100, 3)
        u = model(x)
        assert u.shape == (100, 1)

    def test_vector_output(self):
        """Test with vector output (e.g., velocity field)."""
        model = SimplePINN(input_dim=2, output_dim=3)
        x = torch.randn(100, 2)
        u = model(x)
        assert u.shape == (100, 3)

    def test_1d_problem(self):
        """Test 1D problem."""
        model = SimplePINN(input_dim=1, output_dim=1)
        x = torch.randn(100, 1)
        u = model(x)
        assert u.shape == (100, 1)
