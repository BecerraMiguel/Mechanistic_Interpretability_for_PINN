"""
Collocation point sampling strategies for PINNs.

This module provides various sampling strategies for generating collocation points:
- Latin Hypercube Sampling (better coverage than uniform random)
- Uniform Random Sampling
- Grid Sampling (for visualization/testing)
- Boundary Sampling (uniform on edges)

These samplers can be used by any PDE problem for training and evaluation.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import torch
from scipy.stats import qmc


class CollocationSampler(ABC):
    """
    Abstract base class for collocation point samplers.

    All sampling strategies should inherit from this class and implement
    the sample() method.
    """

    @abstractmethod
    def sample(
        self,
        n: int,
        domain: Tuple[Tuple[float, float], ...],
        random_seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Sample n points from the given domain.

        Parameters
        ----------
        n : int
            Number of points to sample.
        domain : Tuple[Tuple[float, float], ...]
            Domain bounds as ((x_min, x_max), (y_min, y_max), ...).
        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        torch.Tensor
            Sampled points of shape (n, spatial_dim).
        """
        pass


class LatinHypercubeSampler(CollocationSampler):
    """
    Latin Hypercube Sampling for interior collocation points.

    LHS provides better space-filling properties than uniform random sampling
    by ensuring even distribution across all dimensions. Each dimension is
    divided into n equal intervals with exactly one sample per interval.

    This is the recommended sampler for PINN interior points.
    """

    def sample(
        self,
        n: int,
        domain: Tuple[Tuple[float, float], ...],
        random_seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Sample n points using Latin Hypercube Sampling.

        Parameters
        ----------
        n : int
            Number of points to sample.
        domain : Tuple[Tuple[float, float], ...]
            Domain bounds.
        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        torch.Tensor
            Sampled points of shape (n, spatial_dim).
        """
        spatial_dim = len(domain)

        # Create Latin Hypercube sampler
        sampler = qmc.LatinHypercube(d=spatial_dim, seed=random_seed)

        # Sample in [0, 1]^d
        points = sampler.random(n=n)

        # Scale to domain bounds
        for i, (low, high) in enumerate(domain):
            points[:, i] = points[:, i] * (high - low) + low

        return torch.tensor(points, dtype=torch.float32)


class UniformRandomSampler(CollocationSampler):
    """
    Uniform random sampling for interior collocation points.

    Samples points uniformly at random from the domain. Simpler than Latin
    Hypercube but may have worse coverage (clustering/gaps possible).

    Use this if you want truly random sampling or for comparison with LHS.
    """

    def sample(
        self,
        n: int,
        domain: Tuple[Tuple[float, float], ...],
        random_seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Sample n points using uniform random sampling.

        Parameters
        ----------
        n : int
            Number of points to sample.
        domain : Tuple[Tuple[float, float], ...]
            Domain bounds.
        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        torch.Tensor
            Sampled points of shape (n, spatial_dim).
        """
        if random_seed is not None:
            torch.manual_seed(random_seed)

        spatial_dim = len(domain)
        points = torch.rand(n, spatial_dim)

        # Scale to domain bounds
        for i, (low, high) in enumerate(domain):
            points[:, i] = points[:, i] * (high - low) + low

        return points


class GridSampler(CollocationSampler):
    """
    Uniform grid sampling for visualization and testing.

    Creates a regular grid of points across the domain. Useful for:
    - Generating test/validation grids
    - Visualization
    - Debugging

    Not recommended for training (no randomness, poor for stochastic optimization).
    """

    def sample(
        self,
        n: int,
        domain: Tuple[Tuple[float, float], ...],
        random_seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Sample n points on a uniform grid.

        For 2D domain, creates an approximately sqrt(n) × sqrt(n) grid.
        For higher dimensions, uses n^(1/d) points per dimension.

        Parameters
        ----------
        n : int
            Approximate total number of points (actual count may differ).
        domain : Tuple[Tuple[float, float], ...]
            Domain bounds.
        random_seed : int, optional
            Not used (grid is deterministic).

        Returns
        -------
        torch.Tensor
            Grid points of shape (actual_n, spatial_dim).
        """
        spatial_dim = len(domain)

        # Points per dimension
        n_per_dim = int(n ** (1 / spatial_dim))

        # Create 1D grids for each dimension
        grids_1d = []
        for low, high in domain:
            grids_1d.append(torch.linspace(low, high, n_per_dim))

        # Create meshgrid
        meshgrids = torch.meshgrid(*grids_1d, indexing="ij")

        # Flatten and stack
        points = torch.stack([g.flatten() for g in meshgrids], dim=1)

        return points


class BoundarySampler:
    """
    Sampler for boundary points on rectangular domains.

    For a 2D rectangular domain, samples points uniformly on all 4 edges.
    For 3D, samples on all 6 faces, etc.

    Points lie exactly on domain boundaries (not approximate).
    """

    def sample(
        self,
        n_per_edge: int,
        domain: Tuple[Tuple[float, float], ...],
        random_seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Sample boundary points on all edges/faces of a rectangular domain.

        For 2D: Samples n_per_edge points on each of the 4 edges.
        Total points = 4 * n_per_edge (corners may be duplicated).

        Parameters
        ----------
        n_per_edge : int
            Number of points to sample per boundary edge/face.
        domain : Tuple[Tuple[float, float], ...]
            Domain bounds.
        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        torch.Tensor
            Boundary points of shape (total_boundary_points, spatial_dim).
        """
        spatial_dim = len(domain)

        if spatial_dim == 1:
            return self._sample_1d(n_per_edge, domain)
        elif spatial_dim == 2:
            return self._sample_2d(n_per_edge, domain, random_seed)
        elif spatial_dim == 3:
            return self._sample_3d(n_per_edge, domain, random_seed)
        else:
            raise NotImplementedError(f"Boundary sampling not implemented for {spatial_dim}D")

    def _sample_1d(self, n_per_edge: int, domain: Tuple[Tuple[float, float], ...]) -> torch.Tensor:
        """Sample boundary points for 1D domain (just the two endpoints)."""
        x_min, x_max = domain[0]
        return torch.tensor([[x_min], [x_max]], dtype=torch.float32)

    def _sample_2d(
        self,
        n_per_edge: int,
        domain: Tuple[Tuple[float, float], ...],
        random_seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Sample boundary points for 2D rectangular domain."""
        if random_seed is not None:
            torch.manual_seed(random_seed)

        (x_min, x_max), (y_min, y_max) = domain

        # Create uniform points along each dimension
        x_points = torch.linspace(x_min, x_max, n_per_edge)
        y_points = torch.linspace(y_min, y_max, n_per_edge)

        boundary_points = []

        # Bottom edge: y = y_min
        bottom = torch.stack([x_points, torch.full_like(x_points, y_min)], dim=1)
        boundary_points.append(bottom)

        # Right edge: x = x_max
        right = torch.stack([torch.full_like(y_points, x_max), y_points], dim=1)
        boundary_points.append(right)

        # Top edge: y = y_max
        top = torch.stack([x_points, torch.full_like(x_points, y_max)], dim=1)
        boundary_points.append(top)

        # Left edge: x = x_min
        left = torch.stack([torch.full_like(y_points, x_min), y_points], dim=1)
        boundary_points.append(left)

        return torch.cat(boundary_points, dim=0)

    def _sample_3d(
        self,
        n_per_edge: int,
        domain: Tuple[Tuple[float, float], ...],
        random_seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Sample boundary points for 3D rectangular domain."""
        if random_seed is not None:
            torch.manual_seed(random_seed)

        (x_min, x_max), (y_min, y_max), (z_min, z_max) = domain

        # For 3D, we sample on 6 faces
        # Each face gets n_per_edge × n_per_edge points
        n_per_face = n_per_edge

        x_points = torch.linspace(x_min, x_max, n_per_face)
        y_points = torch.linspace(y_min, y_max, n_per_face)
        z_points = torch.linspace(z_min, z_max, n_per_face)

        boundary_points = []

        # Face 1: z = z_min
        X, Y = torch.meshgrid(x_points, y_points, indexing="ij")
        face1 = torch.stack(
            [X.flatten(), Y.flatten(), torch.full((n_per_face * n_per_face,), z_min)], dim=1
        )
        boundary_points.append(face1)

        # Face 2: z = z_max
        face2 = torch.stack(
            [X.flatten(), Y.flatten(), torch.full((n_per_face * n_per_face,), z_max)], dim=1
        )
        boundary_points.append(face2)

        # Face 3: y = y_min
        X, Z = torch.meshgrid(x_points, z_points, indexing="ij")
        face3 = torch.stack(
            [X.flatten(), torch.full((n_per_face * n_per_face,), y_min), Z.flatten()], dim=1
        )
        boundary_points.append(face3)

        # Face 4: y = y_max
        face4 = torch.stack(
            [X.flatten(), torch.full((n_per_face * n_per_face,), y_max), Z.flatten()], dim=1
        )
        boundary_points.append(face4)

        # Face 5: x = x_min
        Y, Z = torch.meshgrid(y_points, z_points, indexing="ij")
        face5 = torch.stack(
            [torch.full((n_per_face * n_per_face,), x_min), Y.flatten(), Z.flatten()], dim=1
        )
        boundary_points.append(face5)

        # Face 6: x = x_max
        face6 = torch.stack(
            [torch.full((n_per_face * n_per_face,), x_max), Y.flatten(), Z.flatten()], dim=1
        )
        boundary_points.append(face6)

        return torch.cat(boundary_points, dim=0)


def sample_collocation_points(
    n_interior: int,
    n_boundary: int,
    domain: Tuple[Tuple[float, float], ...],
    interior_sampler: str = "lhs",
    random_seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convenience function to sample both interior and boundary points.

    Parameters
    ----------
    n_interior : int
        Number of interior collocation points.
    n_boundary : int
        Number of boundary points per edge.
    domain : Tuple[Tuple[float, float], ...]
        Domain bounds.
    interior_sampler : str, optional
        Interior sampling strategy: 'lhs', 'uniform', or 'grid'. Default: 'lhs'.
    random_seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    x_interior : torch.Tensor
        Interior points of shape (n_interior, spatial_dim).
    x_boundary : torch.Tensor
        Boundary points of shape (n_boundary_total, spatial_dim).

    Examples
    --------
    >>> domain = ((0, 1), (0, 1))  # Unit square
    >>> x_int, x_bound = sample_collocation_points(
    ...     n_interior=10000,
    ...     n_boundary=100,
    ...     domain=domain,
    ...     interior_sampler='lhs',
    ...     random_seed=42
    ... )
    >>> x_int.shape
    torch.Size([10000, 2])
    >>> x_bound.shape
    torch.Size([400, 2])  # 4 edges × 100 points
    """
    # Select interior sampler
    if interior_sampler == "lhs":
        sampler = LatinHypercubeSampler()
    elif interior_sampler == "uniform":
        sampler = UniformRandomSampler()
    elif interior_sampler == "grid":
        sampler = GridSampler()
    else:
        raise ValueError(
            f"Unknown interior_sampler: {interior_sampler}. "
            f"Choose from 'lhs', 'uniform', or 'grid'."
        )

    # Sample interior points
    x_interior = sampler.sample(n_interior, domain, random_seed=random_seed)

    # Sample boundary points
    boundary_sampler = BoundarySampler()
    x_boundary = boundary_sampler.sample(n_boundary, domain, random_seed=random_seed)

    return x_interior, x_boundary
