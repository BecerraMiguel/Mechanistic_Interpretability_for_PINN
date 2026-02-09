"""
Tests for Poisson equation problem class.

Tests verify:
- Analytical solution correctness
- Source term computation
- Boundary condition enforcement
- Collocation point sampling (distributions and coverage)
- PDE residual computation with analytical solution
- Boundary points lie exactly on domain edges
"""

import math

import pytest
import torch

from src.problems import PoissonProblem
from src.utils.derivatives import compute_derivatives


class TestPoissonProblemInitialization:
    """Test PoissonProblem initialization."""

    def test_default_initialization(self):
        """Test default initialization creates unit square domain."""
        problem = PoissonProblem()

        assert problem.spatial_dim == 2
        assert problem.domain == ((0.0, 1.0), (0.0, 1.0))
        assert problem.x_min == 0.0
        assert problem.x_max == 1.0
        assert problem.y_min == 0.0
        assert problem.y_max == 1.0

    def test_custom_domain(self):
        """Test initialization with custom domain."""
        domain = ((0.0, 2.0), (0.0, 3.0))
        problem = PoissonProblem(domain=domain)

        assert problem.spatial_dim == 2
        assert problem.domain == domain
        assert problem.x_min == 0.0
        assert problem.x_max == 2.0
        assert problem.y_min == 0.0
        assert problem.y_max == 3.0

    def test_invalid_dimension_raises_error(self):
        """Test that non-2D domain raises ValueError."""
        with pytest.raises(ValueError, match="only supports 2D domains"):
            PoissonProblem(domain=((0.0, 1.0),))  # 1D domain

        with pytest.raises(ValueError, match="only supports 2D domains"):
            PoissonProblem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)))  # 3D domain

    def test_repr(self):
        """Test string representation."""
        problem = PoissonProblem()
        repr_str = repr(problem)

        assert "PoissonProblem" in repr_str
        assert "domain" in repr_str
        assert "∇²u = f" in repr_str
        assert "sin(πx)sin(πy)" in repr_str


class TestPoissonAnalyticalSolution:
    """Test analytical solution computation."""

    def test_analytical_solution_shape(self):
        """Test output shape is correct."""
        problem = PoissonProblem()
        x = torch.randn(100, 2)

        u = problem.analytical_solution(x)

        assert u.shape == (100, 1)

    def test_analytical_solution_known_values(self):
        """Test analytical solution at known points."""
        problem = PoissonProblem()

        # Test at (0.5, 0.5) -> u = sin(π/2)sin(π/2) = 1 * 1 = 1
        x = torch.tensor([[0.5, 0.5]])
        u = problem.analytical_solution(x)
        assert torch.isclose(u, torch.tensor([[1.0]]), atol=1e-6)

        # Test at (0, 0) -> u = sin(0)sin(0) = 0
        x = torch.tensor([[0.0, 0.0]])
        u = problem.analytical_solution(x)
        assert torch.isclose(u, torch.tensor([[0.0]]), atol=1e-6)

        # Test at (1, 1) -> u = sin(π)sin(π) = 0
        x = torch.tensor([[1.0, 1.0]])
        u = problem.analytical_solution(x)
        assert torch.isclose(u, torch.tensor([[0.0]]), atol=1e-6)

        # Test at (0.25, 0.5) -> u = sin(π/4)sin(π/2) = √2/2 * 1
        x = torch.tensor([[0.25, 0.5]])
        u = problem.analytical_solution(x)
        expected = math.sin(math.pi / 4) * math.sin(math.pi / 2)
        assert torch.isclose(u, torch.tensor([[expected]]), atol=1e-6)

    def test_analytical_solution_boundary_zeros(self):
        """Test that analytical solution is zero on boundaries."""
        problem = PoissonProblem()

        # Test boundary points
        boundary_points = torch.tensor([
            [0.0, 0.5],  # Left edge
            [1.0, 0.5],  # Right edge
            [0.5, 0.0],  # Bottom edge
            [0.5, 1.0],  # Top edge
        ])

        u = problem.analytical_solution(boundary_points)
        assert torch.allclose(u, torch.zeros(4, 1), atol=1e-6)

    def test_analytical_solution_invalid_shape_raises_error(self):
        """Test that invalid input shape raises ValueError."""
        problem = PoissonProblem()

        with pytest.raises(ValueError, match="Expected input of shape"):
            problem.analytical_solution(torch.randn(10, 3))  # Wrong spatial dim


class TestPoissonSourceTerm:
    """Test source term computation."""

    def test_source_term_shape(self):
        """Test output shape is correct."""
        problem = PoissonProblem()
        x = torch.randn(100, 2)

        f = problem.source_term(x)

        assert f.shape == (100, 1)

    def test_source_term_known_values(self):
        """Test source term at known points."""
        problem = PoissonProblem()

        # Test at (0.5, 0.5) -> f = -2π²sin(π/2)sin(π/2) = -2π²
        x = torch.tensor([[0.5, 0.5]])
        f = problem.source_term(x)
        expected = -2 * (math.pi ** 2)
        assert torch.isclose(f, torch.tensor([[expected]]), atol=1e-5)

        # Test at (0, 0) -> f = -2π²sin(0)sin(0) = 0
        x = torch.tensor([[0.0, 0.0]])
        f = problem.source_term(x)
        assert torch.isclose(f, torch.tensor([[0.0]]), atol=1e-6)

    def test_source_term_matches_laplacian(self):
        """Test that source term matches Laplacian of analytical solution."""
        problem = PoissonProblem()

        # Sample random points
        x = torch.randn(50, 2, requires_grad=True)

        # Compute analytical solution and its Laplacian
        u = problem.analytical_solution(x)
        laplacian_u = compute_derivatives(u, x, order=2)

        # Compute source term
        f = problem.source_term(x)

        # They should match: ∇²u = f
        assert torch.allclose(laplacian_u, f, atol=1e-4)

    def test_source_term_invalid_shape_raises_error(self):
        """Test that invalid input shape raises ValueError."""
        problem = PoissonProblem()

        with pytest.raises(ValueError, match="Expected input of shape"):
            problem.source_term(torch.randn(10, 3))


class TestPoissonBoundaryCondition:
    """Test boundary condition computation."""

    def test_boundary_condition_shape(self):
        """Test output shape is correct."""
        problem = PoissonProblem()
        x = torch.randn(100, 2)

        bc = problem.boundary_condition(x)

        assert bc.shape == (100, 1)

    def test_boundary_condition_is_zero(self):
        """Test that boundary condition is zero (Dirichlet BC: u = 0)."""
        problem = PoissonProblem()
        x = torch.randn(50, 2)

        bc = problem.boundary_condition(x)

        assert torch.allclose(bc, torch.zeros(50, 1))

    def test_boundary_condition_invalid_shape_raises_error(self):
        """Test that invalid input shape raises ValueError."""
        problem = PoissonProblem()

        with pytest.raises(ValueError, match="Expected input of shape"):
            problem.boundary_condition(torch.randn(10, 1))


class TestPoissonSampleInteriorPoints:
    """Test interior point sampling with Latin Hypercube."""

    def test_sample_interior_points_shape(self):
        """Test that sampled points have correct shape."""
        problem = PoissonProblem()

        points = problem.sample_interior_points(n=100)

        assert points.shape == (100, 2)

    def test_sample_interior_points_within_domain(self):
        """Test that all sampled points are within domain bounds."""
        problem = PoissonProblem()

        points = problem.sample_interior_points(n=1000)

        # Check x coordinates
        assert torch.all(points[:, 0] >= problem.x_min)
        assert torch.all(points[:, 0] <= problem.x_max)

        # Check y coordinates
        assert torch.all(points[:, 1] >= problem.y_min)
        assert torch.all(points[:, 1] <= problem.y_max)

    def test_sample_interior_points_reproducibility(self):
        """Test that sampling with same seed produces same points."""
        problem = PoissonProblem()

        points1 = problem.sample_interior_points(n=100, random_seed=42)
        points2 = problem.sample_interior_points(n=100, random_seed=42)

        assert torch.allclose(points1, points2)

    def test_sample_interior_points_different_seeds(self):
        """Test that different seeds produce different points."""
        problem = PoissonProblem()

        points1 = problem.sample_interior_points(n=100, random_seed=42)
        points2 = problem.sample_interior_points(n=100, random_seed=43)

        assert not torch.allclose(points1, points2)

    def test_sample_interior_points_coverage(self):
        """Test that Latin Hypercube provides good coverage."""
        problem = PoissonProblem()

        # Sample many points
        points = problem.sample_interior_points(n=1000, random_seed=42)

        # Check that points are well-distributed in each dimension
        # Divide domain into 4 quadrants and check coverage
        x_mid = (problem.x_min + problem.x_max) / 2
        y_mid = (problem.y_min + problem.y_max) / 2

        q1 = ((points[:, 0] < x_mid) & (points[:, 1] < y_mid)).sum()  # Bottom-left
        q2 = ((points[:, 0] >= x_mid) & (points[:, 1] < y_mid)).sum()  # Bottom-right
        q3 = ((points[:, 0] < x_mid) & (points[:, 1] >= y_mid)).sum()  # Top-left
        q4 = ((points[:, 0] >= x_mid) & (points[:, 1] >= y_mid)).sum()  # Top-right

        # Each quadrant should have roughly 25% of points (allow 15-35% range)
        for count in [q1, q2, q3, q4]:
            assert 150 <= count <= 350, f"Quadrant has {count} points, expected ~250"

    def test_sample_interior_points_custom_domain(self):
        """Test sampling in custom domain."""
        domain = ((1.0, 3.0), (2.0, 5.0))
        problem = PoissonProblem(domain=domain)

        points = problem.sample_interior_points(n=500)

        # Check bounds
        assert torch.all(points[:, 0] >= 1.0)
        assert torch.all(points[:, 0] <= 3.0)
        assert torch.all(points[:, 1] >= 2.0)
        assert torch.all(points[:, 1] <= 5.0)

    def test_sample_interior_points_invalid_n_raises_error(self):
        """Test that invalid n raises ValueError."""
        problem = PoissonProblem()

        with pytest.raises(ValueError, match="must be positive"):
            problem.sample_interior_points(n=0)

        with pytest.raises(ValueError, match="must be positive"):
            problem.sample_interior_points(n=-10)


class TestPoissonSampleBoundaryPoints:
    """Test boundary point sampling."""

    def test_sample_boundary_points_shape(self):
        """Test that sampled points have correct shape."""
        problem = PoissonProblem()

        points = problem.sample_boundary_points(n_per_edge=25)

        # 4 edges * 25 points per edge = 100 total points
        assert points.shape == (100, 2)

    def test_sample_boundary_points_on_edges(self):
        """Test that all points lie exactly on domain edges."""
        problem = PoissonProblem()

        points = problem.sample_boundary_points(n_per_edge=50)

        # Each point should be on at least one edge
        tolerance = 1e-6

        on_left = torch.abs(points[:, 0] - problem.x_min) < tolerance
        on_right = torch.abs(points[:, 0] - problem.x_max) < tolerance
        on_bottom = torch.abs(points[:, 1] - problem.y_min) < tolerance
        on_top = torch.abs(points[:, 1] - problem.y_max) < tolerance

        # Each point should be on at least one edge
        on_any_edge = on_left | on_right | on_bottom | on_top
        assert torch.all(on_any_edge)

    def test_sample_boundary_points_all_edges_covered(self):
        """Test that all 4 edges have points."""
        problem = PoissonProblem()

        points = problem.sample_boundary_points(n_per_edge=25)

        tolerance = 1e-6

        # Check each edge has points (corners will be on multiple edges)
        on_left = torch.abs(points[:, 0] - problem.x_min) < tolerance
        on_right = torch.abs(points[:, 0] - problem.x_max) < tolerance
        on_bottom = torch.abs(points[:, 1] - problem.y_min) < tolerance
        on_top = torch.abs(points[:, 1] - problem.y_max) < tolerance

        # Each edge should have at least n_per_edge points (corners counted on adjacent edges)
        assert on_left.sum() >= 25
        assert on_right.sum() >= 25
        assert on_bottom.sum() >= 25
        assert on_top.sum() >= 25

        # Verify all edges have coverage
        assert on_left.sum() > 0
        assert on_right.sum() > 0
        assert on_bottom.sum() > 0
        assert on_top.sum() > 0

    def test_sample_boundary_points_within_domain(self):
        """Test that all boundary points are within domain bounds."""
        problem = PoissonProblem()

        points = problem.sample_boundary_points(n_per_edge=30)

        # All points should be within (or on) domain bounds
        assert torch.all(points[:, 0] >= problem.x_min)
        assert torch.all(points[:, 0] <= problem.x_max)
        assert torch.all(points[:, 1] >= problem.y_min)
        assert torch.all(points[:, 1] <= problem.y_max)

    def test_sample_boundary_points_custom_domain(self):
        """Test boundary sampling in custom domain."""
        domain = ((0.5, 2.5), (1.0, 4.0))
        problem = PoissonProblem(domain=domain)

        points = problem.sample_boundary_points(n_per_edge=20)

        tolerance = 1e-6

        on_left = torch.abs(points[:, 0] - 0.5) < tolerance
        on_right = torch.abs(points[:, 0] - 2.5) < tolerance
        on_bottom = torch.abs(points[:, 1] - 1.0) < tolerance
        on_top = torch.abs(points[:, 1] - 4.0) < tolerance

        # Each point on an edge
        on_any_edge = on_left | on_right | on_bottom | on_top
        assert torch.all(on_any_edge)

    def test_sample_boundary_points_reproducibility(self):
        """Test that sampling with same seed produces same points."""
        problem = PoissonProblem()

        points1 = problem.sample_boundary_points(n_per_edge=30, random_seed=42)
        points2 = problem.sample_boundary_points(n_per_edge=30, random_seed=42)

        assert torch.allclose(points1, points2)

    def test_sample_boundary_points_invalid_n_raises_error(self):
        """Test that invalid n_per_edge raises ValueError."""
        problem = PoissonProblem()

        with pytest.raises(ValueError, match="must be positive"):
            problem.sample_boundary_points(n_per_edge=0)

        with pytest.raises(ValueError, match="must be positive"):
            problem.sample_boundary_points(n_per_edge=-5)


class TestPoissonPDEResidual:
    """Test PDE residual computation."""

    def test_pde_residual_shape(self):
        """Test that residual has correct shape."""
        problem = PoissonProblem()

        x = torch.randn(50, 2, requires_grad=True)
        u = torch.randn(50, 1)
        du_dx = torch.randn(50, 2)
        d2u_dx2 = torch.randn(50, 1)

        residual = problem.pde_residual(u, x, du_dx, d2u_dx2)

        assert residual.shape == (50, 1)

    def test_pde_residual_zero_for_analytical_solution(self):
        """Test that PDE residual is zero for the analytical solution."""
        problem = PoissonProblem()

        # Sample interior points
        x = torch.randn(100, 2, requires_grad=True)

        # Compute analytical solution
        u = problem.analytical_solution(x)

        # Compute derivatives
        du_dx = compute_derivatives(u, x, order=1)
        d2u_dx2 = compute_derivatives(u, x, order=2)

        # Compute PDE residual
        residual = problem.pde_residual(u, x, du_dx, d2u_dx2)

        # Residual should be zero (within numerical precision)
        assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-4)

    def test_pde_residual_computation(self):
        """Test PDE residual computation: residual = ∇²u - f."""
        problem = PoissonProblem()

        x = torch.tensor([[0.3, 0.4]], requires_grad=True)
        u = torch.randn(1, 1)
        du_dx = torch.randn(1, 2)
        d2u_dx2 = torch.tensor([[5.0]])  # Laplacian

        residual = problem.pde_residual(u, x, du_dx, d2u_dx2)

        # Compute expected residual
        f = problem.source_term(x)
        expected_residual = d2u_dx2 - f

        assert torch.allclose(residual, expected_residual)


class TestPoissonComputeRelativeL2Error:
    """Test relative L2 error computation."""

    def test_compute_relative_l2_error_zero_for_perfect_model(self):
        """Test that error is near zero when model matches analytical solution."""
        problem = PoissonProblem()

        # Create a "perfect" model that returns analytical solution
        class PerfectModel(torch.nn.Module):
            def __init__(self, problem):
                super().__init__()
                self.problem = problem

            def forward(self, x):
                return self.problem.analytical_solution(x)

        model = PerfectModel(problem)

        error = problem.compute_relative_l2_error(model, n_test_points=500, random_seed=42)

        # Error should be very close to 0
        assert error < 0.01  # Less than 0.01%

    def test_compute_relative_l2_error_nonzero_for_bad_model(self):
        """Test that error is nonzero for incorrect model."""
        problem = PoissonProblem()

        # Create a bad model that returns zeros
        class BadModel(torch.nn.Module):
            def forward(self, x):
                return torch.zeros(x.shape[0], 1)

        model = BadModel()

        error = problem.compute_relative_l2_error(model, n_test_points=500, random_seed=42)

        # Error should be significant (analytical solution is non-zero in interior)
        assert error > 10  # More than 10%

    def test_compute_relative_l2_error_reproducibility(self):
        """Test that error computation is reproducible with same seed."""
        problem = PoissonProblem()

        # Use a deterministic model (not random) for reproducibility testing
        class DeterministicModel(torch.nn.Module):
            def forward(self, x):
                # Simple deterministic function: sum of coordinates
                return (x[:, 0:1] + x[:, 1:2]) / 2

        model = DeterministicModel()

        error1 = problem.compute_relative_l2_error(model, n_test_points=500, random_seed=42)
        error2 = problem.compute_relative_l2_error(model, n_test_points=500, random_seed=42)

        # Should get same error with same seed
        assert abs(error1 - error2) < 1e-5


class TestPoissonIntegration:
    """Integration tests combining multiple components."""

    def test_full_problem_setup(self):
        """Test complete problem setup and data generation."""
        problem = PoissonProblem()

        # Sample collocation points
        x_interior = problem.sample_interior_points(n=1000, random_seed=42)
        x_boundary = problem.sample_boundary_points(n_per_edge=25, random_seed=42)

        # Compute values
        u_interior = problem.analytical_solution(x_interior)
        u_boundary = problem.boundary_condition(x_boundary)
        f_interior = problem.source_term(x_interior)

        # Check shapes
        assert x_interior.shape == (1000, 2)
        assert x_boundary.shape == (100, 2)
        assert u_interior.shape == (1000, 1)
        assert u_boundary.shape == (100, 1)
        assert f_interior.shape == (1000, 1)

        # Check boundary conditions are satisfied
        assert torch.allclose(u_boundary, torch.zeros_like(u_boundary))

    def test_pde_satisfied_by_analytical_solution_everywhere(self):
        """Test that analytical solution satisfies PDE everywhere in domain."""
        problem = PoissonProblem()

        # Sample many random points
        x = problem.sample_interior_points(n=500, random_seed=123)
        x.requires_grad = True

        # Compute analytical solution and derivatives
        u = problem.analytical_solution(x)
        du_dx = compute_derivatives(u, x, order=1)
        d2u_dx2 = compute_derivatives(u, x, order=2)

        # Compute residual
        residual = problem.pde_residual(u, x, du_dx, d2u_dx2)

        # Should be zero everywhere
        assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-4)
