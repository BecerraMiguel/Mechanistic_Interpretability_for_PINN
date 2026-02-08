"""
Tests for derivative computation utilities.

This module tests automatic differentiation functions used for computing
PDE residuals.
"""

import pytest
import torch
import math

from src.utils.derivatives import (
    compute_derivatives,
    compute_gradient_components,
    compute_hessian_diagonal,
    compute_mixed_derivative
)


class TestComputeDerivatives:
    """Test compute_derivatives function."""

    def test_first_order_gradient(self):
        """Test first-order gradient computation."""
        # Create input
        x = torch.randn(100, 2, requires_grad=True)

        # Simple function: u = x^2 + y^2
        u = (x[:, 0:1]**2 + x[:, 1:2]**2)

        # Compute gradient
        du_dx = compute_derivatives(u, x, order=1)

        assert du_dx.shape == (100, 2)
        assert not torch.isnan(du_dx).any()

        # Check approximate correctness: du/dx = 2x, du/dy = 2y
        expected = 2 * x
        assert torch.allclose(du_dx, expected, rtol=1e-5)

    def test_second_order_laplacian(self):
        """Test Laplacian computation."""
        x = torch.randn(100, 2, requires_grad=True)

        # Function: u = x^2 + y^2
        # Laplacian: ∇²u = 2 + 2 = 4
        u = (x[:, 0:1]**2 + x[:, 1:2]**2)

        laplacian = compute_derivatives(u, x, order=2)

        assert laplacian.shape == (100, 1)
        assert not torch.isnan(laplacian).any()

        # Laplacian should be 4 everywhere
        expected = torch.full_like(laplacian, 4.0)
        assert torch.allclose(laplacian, expected, rtol=1e-4)

    def test_requires_grad_error(self):
        """Test that error is raised when x doesn't require grad."""
        x = torch.randn(100, 2, requires_grad=False)
        u = x[:, 0:1]**2

        with pytest.raises(ValueError, match="requires_grad"):
            compute_derivatives(u, x, order=1)

    def test_invalid_order(self):
        """Test that invalid order raises error."""
        x = torch.randn(100, 2, requires_grad=True)
        u = x[:, 0:1]**2

        with pytest.raises(ValueError, match="Only order 1 and 2"):
            compute_derivatives(u, x, order=3)

    def test_1d_gradient(self):
        """Test gradient in 1D."""
        x = torch.randn(100, 1, requires_grad=True)
        u = x**2

        du_dx = compute_derivatives(u, x, order=1)

        assert du_dx.shape == (100, 1)
        expected = 2 * x
        assert torch.allclose(du_dx, expected, rtol=1e-5)

    def test_3d_gradient(self):
        """Test gradient in 3D."""
        x = torch.randn(100, 3, requires_grad=True)
        u = (x[:, 0:1]**2 + x[:, 1:2]**2 + x[:, 2:3]**2)

        du_dx = compute_derivatives(u, x, order=1)

        assert du_dx.shape == (100, 3)
        expected = 2 * x
        assert torch.allclose(du_dx, expected, rtol=1e-5)

    def test_3d_laplacian(self):
        """Test Laplacian in 3D."""
        x = torch.randn(100, 3, requires_grad=True)
        u = (x[:, 0:1]**2 + x[:, 1:2]**2 + x[:, 2:3]**2)

        laplacian = compute_derivatives(u, x, order=2)

        assert laplacian.shape == (100, 1)
        # Laplacian should be 6 (2 + 2 + 2)
        expected = torch.full_like(laplacian, 6.0)
        assert torch.allclose(laplacian, expected, rtol=1e-4)


class TestComputeGradientComponents:
    """Test compute_gradient_components function."""

    def test_2d_gradient_components(self):
        """Test extracting gradient components in 2D."""
        x = torch.randn(100, 2, requires_grad=True)
        u = x[:, 0:1]**2 + x[:, 1:2]**2

        du_dx, du_dy = compute_gradient_components(u, x)

        assert du_dx.shape == (100, 1)
        assert du_dy.shape == (100, 1)

        # Check correctness
        assert torch.allclose(du_dx, 2 * x[:, 0:1], rtol=1e-5)
        assert torch.allclose(du_dy, 2 * x[:, 1:2], rtol=1e-5)

    def test_3d_gradient_components(self):
        """Test extracting gradient components in 3D."""
        x = torch.randn(100, 3, requires_grad=True)
        u = x[:, 0:1]**3

        du_dx, du_dy, du_dz = compute_gradient_components(u, x)

        assert du_dx.shape == (100, 1)
        assert du_dy.shape == (100, 1)
        assert du_dz.shape == (100, 1)

        # Only x component should be non-zero
        assert torch.allclose(du_dx, 3 * x[:, 0:1]**2, rtol=1e-5)
        assert torch.allclose(du_dy, torch.zeros_like(du_dy), atol=1e-6)
        assert torch.allclose(du_dz, torch.zeros_like(du_dz), atol=1e-6)

    def test_requires_grad_error(self):
        """Test error when x doesn't require grad."""
        x = torch.randn(100, 2, requires_grad=False)
        u = x[:, 0:1]**2

        with pytest.raises(ValueError, match="requires_grad"):
            compute_gradient_components(u, x)


class TestComputeHessianDiagonal:
    """Test compute_hessian_diagonal function."""

    def test_2d_hessian_diagonal(self):
        """Test Hessian diagonal in 2D."""
        x = torch.randn(100, 2, requires_grad=True)
        # u = x^2 + y^3
        # d2u/dx2 = 2, d2u/dy2 = 6y
        u = x[:, 0:1]**2 + x[:, 1:2]**3

        d2u_dx2, d2u_dy2 = compute_hessian_diagonal(u, x)

        assert d2u_dx2.shape == (100, 1)
        assert d2u_dy2.shape == (100, 1)

        # Check correctness
        assert torch.allclose(d2u_dx2, torch.full_like(d2u_dx2, 2.0), rtol=1e-4)
        assert torch.allclose(d2u_dy2, 6 * x[:, 1:2], rtol=1e-4)

    def test_3d_hessian_diagonal(self):
        """Test Hessian diagonal in 3D."""
        x = torch.randn(100, 3, requires_grad=True)
        u = x[:, 0:1]**2 + x[:, 1:2]**2 + x[:, 2:3]**2

        d2u_dx2, d2u_dy2, d2u_dz2 = compute_hessian_diagonal(u, x)

        # All should be 2
        assert torch.allclose(d2u_dx2, torch.full_like(d2u_dx2, 2.0), rtol=1e-4)
        assert torch.allclose(d2u_dy2, torch.full_like(d2u_dy2, 2.0), rtol=1e-4)
        assert torch.allclose(d2u_dz2, torch.full_like(d2u_dz2, 2.0), rtol=1e-4)

    def test_laplacian_from_hessian_diagonal(self):
        """Test that sum of Hessian diagonal equals Laplacian."""
        x = torch.randn(100, 2, requires_grad=True)
        u = x[:, 0:1]**3 + x[:, 1:2]**2

        # Compute via Hessian diagonal
        d2u_dx2, d2u_dy2 = compute_hessian_diagonal(u, x)
        laplacian_from_hessian = d2u_dx2 + d2u_dy2

        # Compute via compute_derivatives
        laplacian_direct = compute_derivatives(u, x, order=2)

        assert torch.allclose(laplacian_from_hessian, laplacian_direct, rtol=1e-5)


class TestComputeMixedDerivative:
    """Test compute_mixed_derivative function."""

    def test_mixed_derivative_2d(self):
        """Test mixed derivative in 2D."""
        x = torch.randn(100, 2, requires_grad=True)
        # u = x*y
        # d2u/dxdy = 1
        u = x[:, 0:1] * x[:, 1:2]

        d2u_dxdy = compute_mixed_derivative(u, x, i=0, j=1)

        assert d2u_dxdy.shape == (100, 1)
        # Mixed derivative should be 1
        assert torch.allclose(d2u_dxdy, torch.ones_like(d2u_dxdy), rtol=1e-4)

    def test_mixed_derivative_symmetry(self):
        """Test that mixed derivatives are symmetric."""
        x = torch.randn(100, 2, requires_grad=True)
        u = x[:, 0:1]**2 * x[:, 1:2]**3

        d2u_dxdy = compute_mixed_derivative(u, x, i=0, j=1)
        d2u_dydx = compute_mixed_derivative(u, x, i=1, j=0)

        # Schwarz's theorem: mixed derivatives should be equal
        assert torch.allclose(d2u_dxdy, d2u_dydx, rtol=1e-5)

    def test_mixed_derivative_zero(self):
        """Test mixed derivative of separable function."""
        x = torch.randn(100, 2, requires_grad=True)
        # u = x^2 + y^2 (separable)
        # d2u/dxdy = 0
        u = x[:, 0:1]**2 + x[:, 1:2]**2

        d2u_dxdy = compute_mixed_derivative(u, x, i=0, j=1)

        assert torch.allclose(d2u_dxdy, torch.zeros_like(d2u_dxdy), atol=1e-6)

    def test_index_out_of_bounds(self):
        """Test error for out-of-bounds indices."""
        x = torch.randn(100, 2, requires_grad=True)
        u = x[:, 0:1]**2

        with pytest.raises(ValueError, match="out of bounds"):
            compute_mixed_derivative(u, x, i=0, j=2)

        with pytest.raises(ValueError, match="out of bounds"):
            compute_mixed_derivative(u, x, i=-1, j=0)

    def test_same_index(self):
        """Test mixed derivative with same index (should be second derivative)."""
        x = torch.randn(100, 2, requires_grad=True)
        u = x[:, 0:1]**2

        # d2u/dxdx should equal d2u/dx2
        d2u_dxdx = compute_mixed_derivative(u, x, i=0, j=0)

        # Compare with Hessian diagonal
        d2u_dx2, _ = compute_hessian_diagonal(u, x)
        assert torch.allclose(d2u_dxdx, d2u_dx2, rtol=1e-5)


class TestDerivativesIntegration:
    """Integration tests for derivative computations."""

    def test_poisson_equation_residual(self):
        """Test computing Poisson equation residual."""
        # Manufactured solution: u = sin(πx)sin(πy)
        # Laplacian: ∇²u = -2π²sin(πx)sin(πy)
        # Source: f = -∇²u = 2π²sin(πx)sin(πy)

        x = torch.rand(100, 2, requires_grad=True) * 2 - 1  # [-1, 1]

        u = torch.sin(math.pi * x[:, 0:1]) * torch.sin(math.pi * x[:, 1:2])
        laplacian = compute_derivatives(u, x, order=2)

        # Expected Laplacian
        expected_laplacian = -2 * (math.pi**2) * torch.sin(math.pi * x[:, 0:1]) * torch.sin(math.pi * x[:, 1:2])

        assert torch.allclose(laplacian, expected_laplacian, rtol=1e-3)

    def test_heat_equation_components(self):
        """Test computing components for heat equation."""
        # Heat equation: du/dt = α∇²u
        x = torch.rand(100, 3, requires_grad=True)  # (x, y, t)

        # Manufactured solution
        alpha = 0.1
        u = torch.exp(-alpha * x[:, 2:3]) * torch.sin(x[:, 0:1]) * torch.sin(x[:, 1:2])

        # Time derivative (via gradient)
        du_dt = compute_gradient_components(u, x)[2]

        # Spatial Laplacian (using only x, y)
        x_spatial = x[:, :2].detach().requires_grad_(True)
        u_spatial = torch.exp(-alpha * x[:, 2:3]) * torch.sin(x_spatial[:, 0:1]) * torch.sin(x_spatial[:, 1:2])
        laplacian = compute_derivatives(u_spatial, x_spatial, order=2)

        # Residual: du/dt - α∇²u
        residual = du_dt - alpha * laplacian

        # Should be close to zero for exact solution
        # (Note: won't be exactly zero due to numerical errors)
        assert residual.abs().mean() < 0.1
