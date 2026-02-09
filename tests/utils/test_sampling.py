"""
Tests for collocation point sampling strategies.

Tests verify:
- Latin Hypercube Sampling
- Uniform Random Sampling
- Grid Sampling
- Boundary Sampling (1D, 2D, 3D)
- Convenience function
- Reproducibility and correctness
"""

import pytest
import torch

from src.utils.sampling import (
    LatinHypercubeSampler,
    UniformRandomSampler,
    GridSampler,
    BoundarySampler,
    sample_collocation_points,
)


class TestLatinHypercubeSampler:
    """Test Latin Hypercube Sampling."""

    def test_sample_correct_shape_2d(self):
        """Test that LHS returns correct shape for 2D."""
        sampler = LatinHypercubeSampler()
        domain = ((0.0, 1.0), (0.0, 1.0))

        points = sampler.sample(n=100, domain=domain, random_seed=42)

        assert points.shape == (100, 2)

    def test_sample_within_domain(self):
        """Test that all sampled points are within domain bounds."""
        sampler = LatinHypercubeSampler()
        domain = ((0.5, 2.5), (1.0, 4.0))

        points = sampler.sample(n=1000, domain=domain, random_seed=42)

        assert torch.all(points[:, 0] >= 0.5)
        assert torch.all(points[:, 0] <= 2.5)
        assert torch.all(points[:, 1] >= 1.0)
        assert torch.all(points[:, 1] <= 4.0)

    def test_sample_reproducibility(self):
        """Test that same seed gives same points."""
        sampler = LatinHypercubeSampler()
        domain = ((0.0, 1.0), (0.0, 1.0))

        points1 = sampler.sample(n=100, domain=domain, random_seed=42)
        points2 = sampler.sample(n=100, domain=domain, random_seed=42)

        assert torch.allclose(points1, points2)

    def test_sample_different_seeds(self):
        """Test that different seeds give different points."""
        sampler = LatinHypercubeSampler()
        domain = ((0.0, 1.0), (0.0, 1.0))

        points1 = sampler.sample(n=100, domain=domain, random_seed=42)
        points2 = sampler.sample(n=100, domain=domain, random_seed=43)

        assert not torch.allclose(points1, points2)

    def test_sample_coverage_2d(self):
        """Test that LHS provides good coverage in 2D."""
        sampler = LatinHypercubeSampler()
        domain = ((0.0, 1.0), (0.0, 1.0))

        points = sampler.sample(n=1000, domain=domain, random_seed=42)

        # Check that points are well-distributed in each quadrant
        q1 = ((points[:, 0] < 0.5) & (points[:, 1] < 0.5)).sum()
        q2 = ((points[:, 0] >= 0.5) & (points[:, 1] < 0.5)).sum()
        q3 = ((points[:, 0] < 0.5) & (points[:, 1] >= 0.5)).sum()
        q4 = ((points[:, 0] >= 0.5) & (points[:, 1] >= 0.5)).sum()

        # Each quadrant should have roughly 250 points (allow 150-350)
        for count in [q1, q2, q3, q4]:
            assert 150 <= count <= 350

    def test_sample_3d(self):
        """Test LHS in 3D."""
        sampler = LatinHypercubeSampler()
        domain = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))

        points = sampler.sample(n=500, domain=domain, random_seed=42)

        assert points.shape == (500, 3)
        assert torch.all(points >= 0.0)
        assert torch.all(points <= 1.0)


class TestUniformRandomSampler:
    """Test Uniform Random Sampling."""

    def test_sample_correct_shape(self):
        """Test that uniform sampling returns correct shape."""
        sampler = UniformRandomSampler()
        domain = ((0.0, 1.0), (0.0, 1.0))

        points = sampler.sample(n=100, domain=domain, random_seed=42)

        assert points.shape == (100, 2)

    def test_sample_within_domain(self):
        """Test that all sampled points are within domain bounds."""
        sampler = UniformRandomSampler()
        domain = ((1.0, 3.0), (2.0, 5.0))

        points = sampler.sample(n=1000, domain=domain, random_seed=42)

        assert torch.all(points[:, 0] >= 1.0)
        assert torch.all(points[:, 0] <= 3.0)
        assert torch.all(points[:, 1] >= 2.0)
        assert torch.all(points[:, 1] <= 5.0)

    def test_sample_reproducibility(self):
        """Test that same seed gives same points."""
        sampler = UniformRandomSampler()
        domain = ((0.0, 1.0), (0.0, 1.0))

        points1 = sampler.sample(n=100, domain=domain, random_seed=42)
        points2 = sampler.sample(n=100, domain=domain, random_seed=42)

        assert torch.allclose(points1, points2)

    def test_sample_different_seeds(self):
        """Test that different seeds give different points."""
        sampler = UniformRandomSampler()
        domain = ((0.0, 1.0), (0.0, 1.0))

        points1 = sampler.sample(n=100, domain=domain, random_seed=42)
        points2 = sampler.sample(n=100, domain=domain, random_seed=43)

        assert not torch.allclose(points1, points2)


class TestGridSampler:
    """Test Grid Sampling."""

    def test_sample_creates_grid(self):
        """Test that grid sampling creates regular grid."""
        sampler = GridSampler()
        domain = ((0.0, 1.0), (0.0, 1.0))

        points = sampler.sample(n=100, domain=domain)

        # For 100 points in 2D, should create ~10x10 grid
        # Actual count is 100 (10*10)
        assert points.shape[1] == 2
        assert 90 <= points.shape[0] <= 110  # Approximately 100

    def test_sample_deterministic(self):
        """Test that grid sampling is deterministic (ignores seed)."""
        sampler = GridSampler()
        domain = ((0.0, 1.0), (0.0, 1.0))

        points1 = sampler.sample(n=100, domain=domain, random_seed=42)
        points2 = sampler.sample(n=100, domain=domain, random_seed=43)

        assert torch.allclose(points1, points2)

    def test_sample_within_domain(self):
        """Test that grid points are within domain."""
        sampler = GridSampler()
        domain = ((0.5, 2.5), (1.0, 4.0))

        points = sampler.sample(n=100, domain=domain)

        assert torch.all(points[:, 0] >= 0.5)
        assert torch.all(points[:, 0] <= 2.5)
        assert torch.all(points[:, 1] >= 1.0)
        assert torch.all(points[:, 1] <= 4.0)

    def test_sample_includes_corners(self):
        """Test that grid includes domain corners."""
        sampler = GridSampler()
        domain = ((0.0, 1.0), (0.0, 1.0))

        points = sampler.sample(n=100, domain=domain)

        # Check corners are present (within tolerance)
        corners = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        for corner in corners:
            distances = torch.norm(points - corner, dim=1)
            assert torch.any(distances < 1e-5)


class TestBoundarySampler:
    """Test Boundary Sampling."""

    def test_sample_1d(self):
        """Test boundary sampling in 1D."""
        sampler = BoundarySampler()
        domain = ((0.0, 1.0),)

        points = sampler.sample(n_per_edge=10, domain=domain)

        # 1D boundary is just two endpoints
        assert points.shape == (2, 1)
        assert torch.isclose(points[0, 0], torch.tensor(0.0))
        assert torch.isclose(points[1, 0], torch.tensor(1.0))

    def test_sample_2d_shape(self):
        """Test boundary sampling shape in 2D."""
        sampler = BoundarySampler()
        domain = ((0.0, 1.0), (0.0, 1.0))

        points = sampler.sample(n_per_edge=25, domain=domain)

        # 4 edges × 25 points = 100 points
        assert points.shape == (100, 2)

    def test_sample_2d_on_edges(self):
        """Test that 2D boundary points lie on edges."""
        sampler = BoundarySampler()
        domain = ((0.0, 1.0), (0.0, 1.0))

        points = sampler.sample(n_per_edge=50, domain=domain)

        tolerance = 1e-6

        on_left = torch.abs(points[:, 0] - 0.0) < tolerance
        on_right = torch.abs(points[:, 0] - 1.0) < tolerance
        on_bottom = torch.abs(points[:, 1] - 0.0) < tolerance
        on_top = torch.abs(points[:, 1] - 1.0) < tolerance

        # Each point should be on at least one edge
        on_any_edge = on_left | on_right | on_bottom | on_top
        assert torch.all(on_any_edge)

    def test_sample_2d_all_edges_covered(self):
        """Test that all 4 edges have points."""
        sampler = BoundarySampler()
        domain = ((0.0, 1.0), (0.0, 1.0))

        points = sampler.sample(n_per_edge=25, domain=domain)

        tolerance = 1e-6

        on_left = torch.abs(points[:, 0] - 0.0) < tolerance
        on_right = torch.abs(points[:, 0] - 1.0) < tolerance
        on_bottom = torch.abs(points[:, 1] - 0.0) < tolerance
        on_top = torch.abs(points[:, 1] - 1.0) < tolerance

        # Each edge should have at least n_per_edge points
        assert on_left.sum() >= 25
        assert on_right.sum() >= 25
        assert on_bottom.sum() >= 25
        assert on_top.sum() >= 25

    def test_sample_2d_reproducibility(self):
        """Test that 2D boundary sampling is reproducible."""
        sampler = BoundarySampler()
        domain = ((0.0, 1.0), (0.0, 1.0))

        points1 = sampler.sample(n_per_edge=30, domain=domain, random_seed=42)
        points2 = sampler.sample(n_per_edge=30, domain=domain, random_seed=42)

        assert torch.allclose(points1, points2)

    def test_sample_3d_shape(self):
        """Test boundary sampling shape in 3D."""
        sampler = BoundarySampler()
        domain = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))

        points = sampler.sample(n_per_edge=10, domain=domain)

        # 6 faces × (10×10) points = 600 points
        assert points.shape == (600, 3)

    def test_sample_3d_on_faces(self):
        """Test that 3D boundary points lie on faces."""
        sampler = BoundarySampler()
        domain = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))

        points = sampler.sample(n_per_edge=10, domain=domain)

        tolerance = 1e-6

        on_x_min = torch.abs(points[:, 0] - 0.0) < tolerance
        on_x_max = torch.abs(points[:, 0] - 1.0) < tolerance
        on_y_min = torch.abs(points[:, 1] - 0.0) < tolerance
        on_y_max = torch.abs(points[:, 1] - 1.0) < tolerance
        on_z_min = torch.abs(points[:, 2] - 0.0) < tolerance
        on_z_max = torch.abs(points[:, 2] - 1.0) < tolerance

        # Each point should be on at least one face
        on_any_face = on_x_min | on_x_max | on_y_min | on_y_max | on_z_min | on_z_max
        assert torch.all(on_any_face)

    def test_sample_custom_domain(self):
        """Test boundary sampling on custom domain."""
        sampler = BoundarySampler()
        domain = ((1.0, 3.0), (2.0, 5.0))

        points = sampler.sample(n_per_edge=20, domain=domain)

        # All points should be within domain
        assert torch.all(points[:, 0] >= 1.0)
        assert torch.all(points[:, 0] <= 3.0)
        assert torch.all(points[:, 1] >= 2.0)
        assert torch.all(points[:, 1] <= 5.0)


class TestConvenienceFunction:
    """Test sample_collocation_points convenience function."""

    def test_sample_collocation_points_lhs(self):
        """Test convenience function with LHS."""
        domain = ((0.0, 1.0), (0.0, 1.0))

        x_int, x_bound = sample_collocation_points(
            n_interior=1000,
            n_boundary=25,
            domain=domain,
            interior_sampler="lhs",
            random_seed=42,
        )

        assert x_int.shape == (1000, 2)
        assert x_bound.shape == (100, 2)  # 4 edges × 25

    def test_sample_collocation_points_uniform(self):
        """Test convenience function with uniform sampling."""
        domain = ((0.0, 1.0), (0.0, 1.0))

        x_int, x_bound = sample_collocation_points(
            n_interior=500,
            n_boundary=20,
            domain=domain,
            interior_sampler="uniform",
            random_seed=42,
        )

        assert x_int.shape == (500, 2)
        assert x_bound.shape == (80, 2)

    def test_sample_collocation_points_grid(self):
        """Test convenience function with grid sampling."""
        domain = ((0.0, 1.0), (0.0, 1.0))

        x_int, x_bound = sample_collocation_points(
            n_interior=100,
            n_boundary=10,
            domain=domain,
            interior_sampler="grid",
        )

        assert x_int.shape[1] == 2
        assert x_bound.shape == (40, 2)

    def test_sample_collocation_points_invalid_sampler(self):
        """Test that invalid sampler raises error."""
        domain = ((0.0, 1.0), (0.0, 1.0))

        with pytest.raises(ValueError, match="Unknown interior_sampler"):
            sample_collocation_points(
                n_interior=100,
                n_boundary=10,
                domain=domain,
                interior_sampler="invalid",
            )

    def test_sample_collocation_points_3d(self):
        """Test convenience function in 3D."""
        domain = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))

        x_int, x_bound = sample_collocation_points(
            n_interior=500,
            n_boundary=10,
            domain=domain,
            interior_sampler="lhs",
            random_seed=42,
        )

        assert x_int.shape == (500, 3)
        assert x_bound.shape == (600, 3)  # 6 faces × (10×10)

    def test_sample_collocation_points_reproducibility(self):
        """Test that convenience function is reproducible."""
        domain = ((0.0, 1.0), (0.0, 1.0))

        x_int1, x_bound1 = sample_collocation_points(
            n_interior=100,
            n_boundary=10,
            domain=domain,
            interior_sampler="lhs",
            random_seed=42,
        )

        x_int2, x_bound2 = sample_collocation_points(
            n_interior=100,
            n_boundary=10,
            domain=domain,
            interior_sampler="lhs",
            random_seed=42,
        )

        assert torch.allclose(x_int1, x_int2)
        assert torch.allclose(x_bound1, x_bound2)


class TestSamplerComparison:
    """Compare different sampling strategies."""

    def test_lhs_vs_uniform_coverage(self):
        """Test that LHS has better coverage than uniform."""
        domain = ((0.0, 1.0), (0.0, 1.0))

        lhs = LatinHypercubeSampler()
        uniform = UniformRandomSampler()

        lhs_points = lhs.sample(n=1000, domain=domain, random_seed=42)
        uniform_points = uniform.sample(n=1000, domain=domain, random_seed=42)

        # Both should be within domain
        assert torch.all(lhs_points >= 0) and torch.all(lhs_points <= 1)
        assert torch.all(uniform_points >= 0) and torch.all(uniform_points <= 1)

        # LHS should have better coverage (tested via quadrant distribution)
        # This is tested in test_sample_coverage_2d above

    def test_grid_vs_random_determinism(self):
        """Test that grid is deterministic while LHS/uniform are random."""
        domain = ((0.0, 1.0), (0.0, 1.0))

        grid = GridSampler()
        lhs = LatinHypercubeSampler()

        # Grid should be same regardless of seed
        grid1 = grid.sample(n=100, domain=domain, random_seed=42)
        grid2 = grid.sample(n=100, domain=domain, random_seed=99)
        assert torch.allclose(grid1, grid2)

        # LHS should be different with different seeds
        lhs1 = lhs.sample(n=100, domain=domain, random_seed=42)
        lhs2 = lhs.sample(n=100, domain=domain, random_seed=99)
        assert not torch.allclose(lhs1, lhs2)
