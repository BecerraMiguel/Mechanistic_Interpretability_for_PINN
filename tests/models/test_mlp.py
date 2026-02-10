"""
Tests for MLP PINN architecture.

This module tests the MLP implementation including forward pass,
activation extraction, gradient flow, and multiple activation functions.
"""

import pytest
import torch

from src.models.mlp import MLP


class TestMLPInitialization:
    """Test MLP initialization and configuration."""

    def test_basic_initialization(self):
        """Test basic MLP initialization."""
        model = MLP(input_dim=2, hidden_dims=[50, 50], output_dim=1)

        assert model.input_dim == 2
        assert model.output_dim == 1
        assert model.hidden_dims == [50, 50]
        assert model.activation_name == "tanh"
        assert len(model.layers) == 3  # 2 hidden + 1 output

    def test_single_hidden_layer(self):
        """Test MLP with single hidden layer."""
        model = MLP(input_dim=2, hidden_dims=[100], output_dim=1)
        assert len(model.layers) == 2

    def test_many_hidden_layers(self):
        """Test MLP with many hidden layers."""
        model = MLP(input_dim=2, hidden_dims=[64, 64, 64, 64, 64], output_dim=1)
        assert len(model.layers) == 6

    def test_varying_hidden_dims(self):
        """Test MLP with varying hidden dimensions."""
        model = MLP(input_dim=2, hidden_dims=[100, 50, 25], output_dim=1)
        assert model.get_layer_dimensions() == [2, 100, 50, 25, 1]

    def test_custom_activation(self):
        """Test MLP with custom activation function."""
        for activation in ["tanh", "relu", "gelu", "sin"]:
            model = MLP(input_dim=2, hidden_dims=[50], output_dim=1, activation=activation)
            assert model.activation_name == activation

    def test_invalid_activation(self):
        """Test that invalid activation raises error."""
        with pytest.raises(ValueError, match="not supported"):
            MLP(input_dim=2, hidden_dims=[50], output_dim=1, activation="invalid")

    def test_parameter_count(self):
        """Test parameter counting."""
        model = MLP(input_dim=2, hidden_dims=[50, 50], output_dim=1)

        # Manual calculation:
        # Layer 1: 2*50 + 50 = 150
        # Layer 2: 50*50 + 50 = 2550
        # Layer 3: 50*1 + 1 = 51
        # Total: 150 + 2550 + 51 = 2751
        assert model.get_parameters_count() == 2751


class TestMLPForwardPass:
    """Test MLP forward pass functionality."""

    def test_forward_output_shape(self):
        """Test forward pass produces correct output shape."""
        model = MLP(input_dim=2, hidden_dims=[50, 50], output_dim=1)
        x = torch.randn(100, 2)
        u = model(x)

        assert u.shape == (100, 1)
        assert not torch.isnan(u).any()
        assert not torch.isinf(u).any()

    def test_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        model = MLP(input_dim=2, hidden_dims=[50], output_dim=1)

        for batch_size in [1, 10, 100, 1000]:
            x = torch.randn(batch_size, 2)
            u = model(x)
            assert u.shape == (batch_size, 1)

    def test_different_input_dims(self):
        """Test MLP with different input dimensions."""
        for input_dim in [1, 2, 3, 4]:
            model = MLP(input_dim=input_dim, hidden_dims=[50], output_dim=1)
            x = torch.randn(100, input_dim)
            u = model(x)
            assert u.shape == (100, 1)

    def test_vector_output(self):
        """Test MLP with vector output."""
        model = MLP(input_dim=2, hidden_dims=[50, 50], output_dim=3)
        x = torch.randn(100, 2)
        u = model(x)
        assert u.shape == (100, 3)

    def test_deterministic_forward(self):
        """Test that forward pass is deterministic."""
        model = MLP(input_dim=2, hidden_dims=[50], output_dim=1)
        model.eval()

        x = torch.randn(100, 2)
        u1 = model(x)
        u2 = model(x)

        assert torch.allclose(u1, u2)


class TestMLPActivations:
    """Test activation extraction functionality."""

    def test_activation_extraction(self):
        """Test that activations are extracted correctly."""
        model = MLP(input_dim=2, hidden_dims=[50, 30], output_dim=1)
        x = torch.randn(100, 2)
        u = model(x)

        activations = model.get_activations()

        assert "layer_0" in activations
        assert "layer_1" in activations
        assert activations["layer_0"].shape == (100, 50)
        assert activations["layer_1"].shape == (100, 30)

    def test_activation_cleared_between_passes(self):
        """Test that activations are cleared between forward passes."""
        model = MLP(input_dim=2, hidden_dims=[50], output_dim=1)

        # First forward pass
        x1 = torch.randn(100, 2)
        model(x1)
        act1 = model.get_activations()

        # Second forward pass with different input
        x2 = torch.randn(100, 2)
        model(x2)
        act2 = model.get_activations()

        # Activations should be different
        assert not torch.allclose(act1["layer_0"], act2["layer_0"])

    def test_activation_storage_independent(self):
        """Test that get_activations returns a copy."""
        model = MLP(input_dim=2, hidden_dims=[50], output_dim=1)
        x = torch.randn(100, 2)
        model(x)

        act1 = model.get_activations()
        act2 = model.get_activations()

        # Should be equal but not the same object
        assert torch.allclose(act1["layer_0"], act2["layer_0"])
        assert act1 is not act2  # Different dictionary objects

    def test_no_activation_for_output_layer(self):
        """Test that output layer activation is not stored."""
        model = MLP(input_dim=2, hidden_dims=[50, 30], output_dim=1)
        x = torch.randn(100, 2)
        model(x)

        activations = model.get_activations()

        # Should only have layer_0 and layer_1, not layer_2 (output)
        assert len(activations) == 2
        assert "layer_2" not in activations


class TestMLPActivationFunctions:
    """Test different activation functions."""

    def test_tanh_activation(self):
        """Test tanh activation function."""
        model = MLP(input_dim=2, hidden_dims=[50], output_dim=1, activation="tanh")
        x = torch.randn(100, 2)
        u = model(x)

        # tanh output should be bounded
        activations = model.get_activations()
        assert (activations["layer_0"].abs() <= 1).all()

    def test_relu_activation(self):
        """Test ReLU activation function."""
        model = MLP(input_dim=2, hidden_dims=[50], output_dim=1, activation="relu")
        x = torch.randn(100, 2)
        u = model(x)

        # ReLU output should be non-negative
        activations = model.get_activations()
        assert (activations["layer_0"] >= 0).all()

    def test_gelu_activation(self):
        """Test GELU activation function."""
        model = MLP(input_dim=2, hidden_dims=[50], output_dim=1, activation="gelu")
        x = torch.randn(100, 2)
        u = model(x)

        assert u.shape == (100, 1)
        assert not torch.isnan(u).any()

    def test_sin_activation(self):
        """Test sin activation function."""
        model = MLP(input_dim=2, hidden_dims=[50], output_dim=1, activation="sin")
        x = torch.randn(100, 2)
        u = model(x)

        # sin output should be bounded
        activations = model.get_activations()
        assert (activations["layer_0"].abs() <= 1).all()


class TestMLPGradients:
    """Test gradient flow through MLP."""

    def test_gradient_flow(self):
        """Test that gradients flow through the network."""
        model = MLP(input_dim=2, hidden_dims=[50, 50], output_dim=1)
        x = torch.randn(100, 2, requires_grad=True)
        u = model(x)
        loss = u.mean()

        loss.backward()

        # Check gradients exist for all parameters
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()

        # Check gradient exists for input
        assert x.grad is not None

    def test_gradient_computation_for_pde(self):
        """Test that we can compute derivatives for PDE residuals."""
        model = MLP(input_dim=2, hidden_dims=[50], output_dim=1)
        x = torch.randn(100, 2, requires_grad=True)
        u = model(x)

        # Compute du/dx
        grad_outputs = torch.ones_like(u)
        du_dx = torch.autograd.grad(
            outputs=u, inputs=x, grad_outputs=grad_outputs, create_graph=True, retain_graph=True
        )[0]

        assert du_dx.shape == (100, 2)
        assert not torch.isnan(du_dx).any()

    def test_second_order_derivatives(self):
        """Test computation of second-order derivatives."""
        model = MLP(input_dim=2, hidden_dims=[50], output_dim=1)
        x = torch.randn(100, 2, requires_grad=True)
        u = model(x)

        # First derivative
        du_dx = torch.autograd.grad(
            outputs=u,
            inputs=x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Second derivative (just first component)
        d2u_dx2 = torch.autograd.grad(
            outputs=du_dx[:, 0:1],
            inputs=x,
            grad_outputs=torch.ones_like(du_dx[:, 0:1]),
            create_graph=True,
            retain_graph=True,
        )[0]

        assert d2u_dx2.shape == (100, 2)
        assert not torch.isnan(d2u_dx2).any()


class TestMLPPDEResidual:
    """Test PDE residual computation."""

    def test_pde_residual_without_function(self):
        """Test PDE residual returns zeros when no function provided."""
        model = MLP(input_dim=2, hidden_dims=[50], output_dim=1)
        x = torch.randn(100, 2, requires_grad=True)

        residual = model.compute_pde_residual(x)

        assert residual.shape == (100, 1)
        assert torch.allclose(residual, torch.zeros_like(residual))

    def test_pde_residual_with_function(self):
        """Test PDE residual with custom PDE function."""
        model = MLP(input_dim=2, hidden_dims=[50], output_dim=1)
        x = torch.randn(100, 2, requires_grad=True)

        def simple_pde(u, x, du_dx, d2u_dx2):
            # Simple PDE: ∇²u = 0 (Laplace equation)
            return d2u_dx2

        residual = model.compute_pde_residual(x, pde_fn=simple_pde)

        assert residual.shape == (100, 1)
        assert not torch.isnan(residual).any()

    def test_pde_residual_requires_grad(self):
        """Test that PDE residual requires grad on input."""
        model = MLP(input_dim=2, hidden_dims=[50], output_dim=1)
        x = torch.randn(100, 2, requires_grad=False)

        with pytest.raises(ValueError, match="requires_grad"):
            model.compute_pde_residual(x)

    def test_pde_residual_poisson(self):
        """Test PDE residual for Poisson equation."""
        model = MLP(input_dim=2, hidden_dims=[50, 50], output_dim=1)
        x = torch.randn(100, 2, requires_grad=True)

        def poisson_pde(u, x, du_dx, d2u_dx2):
            # Poisson: ∇²u = f(x), with f(x) = -2π²sin(πx)
            f = -2 * (torch.pi**2) * torch.sin(torch.pi * x[:, 0:1])
            return d2u_dx2 - f

        residual = model.compute_pde_residual(x, pde_fn=poisson_pde)

        assert residual.shape == (100, 1)
        assert not torch.isnan(residual).any()


class TestMLPUtilityMethods:
    """Test utility methods."""

    def test_count_layers(self):
        """Test layer counting."""
        model = MLP(input_dim=2, hidden_dims=[50, 30, 20], output_dim=1)
        assert model.count_layers() == 4  # 3 hidden + 1 output

    def test_get_layer_dimensions(self):
        """Test getting layer dimensions."""
        model = MLP(input_dim=2, hidden_dims=[50, 30], output_dim=1)
        dims = model.get_layer_dimensions()
        assert dims == [2, 50, 30, 1]

    def test_repr(self):
        """Test string representation."""
        model = MLP(input_dim=2, hidden_dims=[50, 50], output_dim=1, activation="tanh")
        repr_str = repr(model)

        assert "MLP" in repr_str
        assert "input_dim=2" in repr_str
        assert "hidden_dims=[50, 50]" in repr_str
        assert "output_dim=1" in repr_str
        assert "activation=tanh" in repr_str
        assert "parameters" in repr_str


class TestMLPIntegration:
    """Integration tests for MLP."""

    def test_full_training_step(self):
        """Test full training step with MLP."""
        model = MLP(input_dim=2, hidden_dims=[50, 50], output_dim=1)

        x_interior = torch.randn(100, 2, requires_grad=True)
        x_boundary = torch.randn(40, 2)
        u_boundary = torch.zeros(40, 1)  # Zero Dirichlet BC

        losses = model.train_step(x_interior, x_boundary, u_boundary)

        assert losses["loss_total"] >= 0
        assert losses["loss_pde"] >= 0
        assert losses["loss_bc"] >= 0

    def test_training_reduces_loss(self):
        """Test that training actually reduces loss."""
        model = MLP(input_dim=2, hidden_dims=[50, 50], output_dim=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        x_interior = torch.randn(100, 2, requires_grad=True)
        x_boundary = torch.randn(40, 2)
        u_boundary = torch.zeros(40, 1)

        # Initial loss
        losses_initial = model.train_step(x_interior, x_boundary, u_boundary)
        loss_initial = losses_initial["loss_total"].item()

        # Train for a few steps
        for _ in range(10):
            optimizer.zero_grad()
            losses = model.train_step(x_interior, x_boundary, u_boundary)
            losses["loss_total"].backward()
            optimizer.step()

        # Final loss
        losses_final = model.train_step(x_interior, x_boundary, u_boundary)
        loss_final = losses_final["loss_total"].item()

        # Loss should decrease (at least a little bit)
        assert loss_final < loss_initial
