"""
Tests for analytical derivative computation in Poisson problem.
"""

import math

import pytest
import torch

from src.problems.poisson import PoissonProblem


class TestPoissonAnalyticalDerivativesDuDx:
    """Tests for analytical first derivative ∂u/∂x."""

    def test_du_dx_shape(self):
        """Test that ∂u/∂x has correct shape."""
        problem = PoissonProblem()
        x = torch.tensor([[0.5, 0.5], [0.25, 0.75], [0.1, 0.9]])

        du_dx = problem.analytical_derivative_du_dx(x)

        assert du_dx.shape == (3, 1)

    def test_du_dx_known_values(self):
        """Test ∂u/∂x at known points."""
        problem = PoissonProblem()

        # At (0.5, 0.5): ∂u/∂x = π·cos(π·0.5)·sin(π·0.5) = π·0·1 = 0
        x1 = torch.tensor([[0.5, 0.5]])
        du_dx1 = problem.analytical_derivative_du_dx(x1)
        assert torch.abs(du_dx1).item() < 1e-6, "Should be ~0 at x=0.5"

        # At (0, 0.5): ∂u/∂x = π·cos(0)·sin(π·0.5) = π·1·1 = π
        x2 = torch.tensor([[0.0, 0.5]])
        du_dx2 = problem.analytical_derivative_du_dx(x2)
        assert torch.abs(du_dx2 - math.pi).item() < 1e-6

        # At (1, 0.5): ∂u/∂x = π·cos(π)·sin(π·0.5) = π·(-1)·1 = -π
        x3 = torch.tensor([[1.0, 0.5]])
        du_dx3 = problem.analytical_derivative_du_dx(x3)
        assert torch.abs(du_dx3 + math.pi).item() < 1e-6

    def test_du_dx_zero_at_boundaries_y(self):
        """Test that ∂u/∂x = 0 at y boundaries (where sin(πy) = 0)."""
        problem = PoissonProblem()

        # At y=0 or y=1, sin(πy) = 0, so ∂u/∂x should be 0
        x_bottom = torch.tensor([[0.3, 0.0], [0.7, 0.0]])
        x_top = torch.tensor([[0.3, 1.0], [0.7, 1.0]])

        du_dx_bottom = problem.analytical_derivative_du_dx(x_bottom)
        du_dx_top = problem.analytical_derivative_du_dx(x_top)

        assert torch.all(torch.abs(du_dx_bottom) < 1e-6)
        assert torch.all(torch.abs(du_dx_top) < 1e-6)

    def test_du_dx_dimension_mismatch(self):
        """Test error for wrong input dimensions."""
        problem = PoissonProblem()

        x_wrong = torch.randn(10, 3)  # Wrong shape

        with pytest.raises(ValueError, match="Expected input of shape"):
            problem.analytical_derivative_du_dx(x_wrong)


class TestPoissonAnalyticalDerivativesDuDy:
    """Tests for analytical first derivative ∂u/∂y."""

    def test_du_dy_shape(self):
        """Test that ∂u/∂y has correct shape."""
        problem = PoissonProblem()
        x = torch.tensor([[0.5, 0.5], [0.25, 0.75], [0.1, 0.9]])

        du_dy = problem.analytical_derivative_du_dy(x)

        assert du_dy.shape == (3, 1)

    def test_du_dy_known_values(self):
        """Test ∂u/∂y at known points."""
        problem = PoissonProblem()

        # At (0.5, 0.5): ∂u/∂y = π·sin(π·0.5)·cos(π·0.5) = π·1·0 = 0
        x1 = torch.tensor([[0.5, 0.5]])
        du_dy1 = problem.analytical_derivative_du_dy(x1)
        assert torch.abs(du_dy1).item() < 1e-6

        # At (0.5, 0): ∂u/∂y = π·sin(π·0.5)·cos(0) = π·1·1 = π
        x2 = torch.tensor([[0.5, 0.0]])
        du_dy2 = problem.analytical_derivative_du_dy(x2)
        assert torch.abs(du_dy2 - math.pi).item() < 1e-6

        # At (0.5, 1): ∂u/∂y = π·sin(π·0.5)·cos(π) = π·1·(-1) = -π
        x3 = torch.tensor([[0.5, 1.0]])
        du_dy3 = problem.analytical_derivative_du_dy(x3)
        assert torch.abs(du_dy3 + math.pi).item() < 1e-6

    def test_du_dy_zero_at_boundaries_x(self):
        """Test that ∂u/∂y = 0 at x boundaries (where sin(πx) = 0)."""
        problem = PoissonProblem()

        # At x=0 or x=1, sin(πx) = 0, so ∂u/∂y should be 0
        x_left = torch.tensor([[0.0, 0.3], [0.0, 0.7]])
        x_right = torch.tensor([[1.0, 0.3], [1.0, 0.7]])

        du_dy_left = problem.analytical_derivative_du_dy(x_left)
        du_dy_right = problem.analytical_derivative_du_dy(x_right)

        assert torch.all(torch.abs(du_dy_left) < 1e-6)
        assert torch.all(torch.abs(du_dy_right) < 1e-6)

    def test_du_dy_dimension_mismatch(self):
        """Test error for wrong input dimensions."""
        problem = PoissonProblem()

        x_wrong = torch.randn(10, 3)

        with pytest.raises(ValueError, match="Expected input of shape"):
            problem.analytical_derivative_du_dy(x_wrong)


class TestPoissonAnalyticalSecondDerivatives:
    """Tests for analytical second derivatives."""

    def test_d2u_dx2_shape(self):
        """Test that ∂²u/∂x² has correct shape."""
        problem = PoissonProblem()
        x = torch.tensor([[0.5, 0.5], [0.25, 0.75]])

        d2u_dx2 = problem.analytical_derivative_d2u_dx2(x)

        assert d2u_dx2.shape == (2, 1)

    def test_d2u_dy2_shape(self):
        """Test that ∂²u/∂y² has correct shape."""
        problem = PoissonProblem()
        x = torch.tensor([[0.5, 0.5], [0.25, 0.75]])

        d2u_dy2 = problem.analytical_derivative_d2u_dy2(x)

        assert d2u_dy2.shape == (2, 1)

    def test_d2u_dx2_known_values(self):
        """Test ∂²u/∂x² at known points."""
        problem = PoissonProblem()

        # At (0.5, 0.5): ∂²u/∂x² = -π²·sin(π·0.5)·sin(π·0.5) = -π²·1·1 = -π²
        x = torch.tensor([[0.5, 0.5]])
        d2u_dx2 = problem.analytical_derivative_d2u_dx2(x)
        expected = -(math.pi**2)
        assert torch.abs(d2u_dx2 - expected).item() < 1e-5

    def test_d2u_dy2_known_values(self):
        """Test ∂²u/∂y² at known points."""
        problem = PoissonProblem()

        # At (0.5, 0.5): ∂²u/∂y² = -π²·sin(π·0.5)·sin(π·0.5) = -π²
        x = torch.tensor([[0.5, 0.5]])
        d2u_dy2 = problem.analytical_derivative_d2u_dy2(x)
        expected = -(math.pi**2)
        assert torch.abs(d2u_dy2 - expected).item() < 1e-5

    def test_second_derivatives_equal(self):
        """Test that ∂²u/∂x² = ∂²u/∂y² due to symmetry."""
        problem = PoissonProblem()

        # For u = sin(πx)sin(πy), second derivatives are equal
        x = torch.randn(50, 2)
        d2u_dx2 = problem.analytical_derivative_d2u_dx2(x)
        d2u_dy2 = problem.analytical_derivative_d2u_dy2(x)

        torch.testing.assert_close(d2u_dx2, d2u_dy2, atol=1e-6, rtol=1e-6)

    def test_d2u_dx2_dimension_mismatch(self):
        """Test error for wrong input dimensions."""
        problem = PoissonProblem()

        x_wrong = torch.randn(10, 3)

        with pytest.raises(ValueError, match="Expected input of shape"):
            problem.analytical_derivative_d2u_dx2(x_wrong)

    def test_d2u_dy2_dimension_mismatch(self):
        """Test error for wrong input dimensions."""
        problem = PoissonProblem()

        x_wrong = torch.randn(10, 3)

        with pytest.raises(ValueError, match="Expected input of shape"):
            problem.analytical_derivative_d2u_dy2(x_wrong)


class TestPoissonAnalyticalLaplacian:
    """Tests for analytical Laplacian."""

    def test_laplacian_shape(self):
        """Test that Laplacian has correct shape."""
        problem = PoissonProblem()
        x = torch.tensor([[0.5, 0.5], [0.25, 0.75]])

        laplacian = problem.analytical_laplacian(x)

        assert laplacian.shape == (2, 1)

    def test_laplacian_known_values(self):
        """Test Laplacian at known points."""
        problem = PoissonProblem()

        # At (0.5, 0.5): ∇²u = -2π²·sin(π·0.5)·sin(π·0.5) = -2π²
        x = torch.tensor([[0.5, 0.5]])
        laplacian = problem.analytical_laplacian(x)
        expected = -2.0 * (math.pi**2)
        assert torch.abs(laplacian - expected).item() < 1e-5

    def test_laplacian_equals_sum_of_second_derivatives(self):
        """Test that ∇²u = ∂²u/∂x² + ∂²u/∂y²."""
        problem = PoissonProblem()

        x = torch.randn(100, 2)
        d2u_dx2 = problem.analytical_derivative_d2u_dx2(x)
        d2u_dy2 = problem.analytical_derivative_d2u_dy2(x)
        laplacian = problem.analytical_laplacian(x)

        sum_of_second_derivs = d2u_dx2 + d2u_dy2

        torch.testing.assert_close(
            laplacian, sum_of_second_derivs, atol=1e-6, rtol=1e-6
        )

    def test_laplacian_equals_source_term(self):
        """Test that ∇²u = source_term(x) for Poisson equation."""
        problem = PoissonProblem()

        x = torch.randn(100, 2)
        laplacian = problem.analytical_laplacian(x)
        source = problem.source_term(x)

        # For Poisson: ∇²u = f, so they should be equal
        torch.testing.assert_close(laplacian, source, atol=1e-6, rtol=1e-6)

    def test_laplacian_dimension_mismatch(self):
        """Test error for wrong input dimensions."""
        problem = PoissonProblem()

        x_wrong = torch.randn(10, 3)

        with pytest.raises(ValueError, match="Expected input of shape"):
            problem.analytical_laplacian(x_wrong)


class TestPoissonAnalyticalGradient:
    """Tests for analytical gradient (convenience method)."""

    def test_gradient_shape(self):
        """Test that gradient has correct shape."""
        problem = PoissonProblem()
        x = torch.tensor([[0.5, 0.5], [0.25, 0.75], [0.1, 0.9]])

        gradient = problem.analytical_gradient(x)

        assert gradient.shape == (3, 2)

    def test_gradient_components(self):
        """Test that gradient returns correct (∂u/∂x, ∂u/∂y)."""
        problem = PoissonProblem()

        x = torch.randn(50, 2)
        gradient = problem.analytical_gradient(x)
        du_dx = problem.analytical_derivative_du_dx(x)
        du_dy = problem.analytical_derivative_du_dy(x)

        # Check that columns match individual derivatives
        torch.testing.assert_close(gradient[:, 0:1], du_dx, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(gradient[:, 1:2], du_dy, atol=1e-6, rtol=1e-6)

    def test_gradient_known_values(self):
        """Test gradient at known point."""
        problem = PoissonProblem()

        # At (0.25, 0.25):
        x = torch.tensor([[0.25, 0.25]])
        gradient = problem.analytical_gradient(x)

        # ∂u/∂x = π·cos(π·0.25)·sin(π·0.25)
        du_dx_expected = math.pi * math.cos(math.pi * 0.25) * math.sin(math.pi * 0.25)
        # ∂u/∂y = π·sin(π·0.25)·cos(π·0.25) (same due to symmetry)
        du_dy_expected = math.pi * math.sin(math.pi * 0.25) * math.cos(math.pi * 0.25)

        assert torch.abs(gradient[0, 0] - du_dx_expected).item() < 1e-6
        assert torch.abs(gradient[0, 1] - du_dy_expected).item() < 1e-6


class TestPoissonDerivativesIntegration:
    """Integration tests for all derivatives together."""

    def test_all_derivatives_on_grid(self):
        """Test computing all derivatives on a grid of points."""
        problem = PoissonProblem()

        # Create a 10x10 grid
        x_vals = torch.linspace(0, 1, 10)
        y_vals = torch.linspace(0, 1, 10)
        xx, yy = torch.meshgrid(x_vals, y_vals, indexing="ij")
        x = torch.stack([xx.flatten(), yy.flatten()], dim=1)  # (100, 2)

        # Compute all derivatives
        u = problem.analytical_solution(x)
        du_dx = problem.analytical_derivative_du_dx(x)
        du_dy = problem.analytical_derivative_du_dy(x)
        d2u_dx2 = problem.analytical_derivative_d2u_dx2(x)
        d2u_dy2 = problem.analytical_derivative_d2u_dy2(x)
        laplacian = problem.analytical_laplacian(x)
        gradient = problem.analytical_gradient(x)

        # All should have correct shapes
        assert u.shape == (100, 1)
        assert du_dx.shape == (100, 1)
        assert du_dy.shape == (100, 1)
        assert d2u_dx2.shape == (100, 1)
        assert d2u_dy2.shape == (100, 1)
        assert laplacian.shape == (100, 1)
        assert gradient.shape == (100, 2)

        # Relationships should hold
        assert torch.all(torch.isfinite(u))
        assert torch.all(torch.isfinite(gradient))
        assert torch.all(torch.isfinite(laplacian))

    def test_derivatives_at_boundary_zero(self):
        """Test that solution and derivatives are zero/well-behaved at boundaries."""
        problem = PoissonProblem()

        # Boundary points
        x_boundary = torch.tensor(
            [
                [0.0, 0.5],  # Left
                [1.0, 0.5],  # Right
                [0.5, 0.0],  # Bottom
                [0.5, 1.0],  # Top
            ]
        )

        u = problem.analytical_solution(x_boundary)

        # Solution should be zero at boundaries
        assert torch.all(torch.abs(u) < 1e-6)

    def test_derivatives_consistency_with_analytical_solution(self):
        """Test that derivatives are consistent with the analytical solution."""
        problem = PoissonProblem()

        # Test at multiple random points
        torch.manual_seed(42)
        x = torch.rand(100, 2)

        # Compute solution and Laplacian
        laplacian = problem.analytical_laplacian(x)
        source = problem.source_term(x)

        # For Poisson equation: ∇²u = f
        # So laplacian should equal source_term
        torch.testing.assert_close(laplacian, source, atol=1e-6, rtol=1e-6)

    def test_all_derivative_methods_work_on_large_batch(self):
        """Test that all methods work on large batches."""
        problem = PoissonProblem()

        # Large batch
        x = torch.randn(10000, 2)

        # All should work without errors
        u = problem.analytical_solution(x)
        du_dx = problem.analytical_derivative_du_dx(x)
        du_dy = problem.analytical_derivative_du_dy(x)
        d2u_dx2 = problem.analytical_derivative_d2u_dx2(x)
        d2u_dy2 = problem.analytical_derivative_d2u_dy2(x)
        laplacian = problem.analytical_laplacian(x)
        gradient = problem.analytical_gradient(x)

        # All should have correct shapes
        assert u.shape == (10000, 1)
        assert du_dx.shape == (10000, 1)
        assert du_dy.shape == (10000, 1)
        assert d2u_dx2.shape == (10000, 1)
        assert d2u_dy2.shape == (10000, 1)
        assert laplacian.shape == (10000, 1)
        assert gradient.shape == (10000, 2)
