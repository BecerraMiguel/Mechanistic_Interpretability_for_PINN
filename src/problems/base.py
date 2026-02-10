"""
Base class for PDE problem definitions.

This module provides the abstract base class that all PDE problems should inherit from,
defining the standard interface for analytical solutions, boundary conditions, and
collocation point sampling.
"""

from abc import ABC, abstractmethod
from typing import Tuple

import torch


class BaseProblem(ABC):
    """
    Abstract base class for PDE problem definitions.

    All PDE problems (Poisson, Heat, Burgers, Helmholtz) should inherit from this class
    and implement the abstract methods.

    Parameters
    ----------
    domain : Tuple[Tuple[float, float], ...]
        Domain boundaries for each spatial dimension as ((x_min, x_max), (y_min, y_max), ...).
        For 2D: ((0, 1), (0, 1)) represents [0,1] x [0,1].
    """

    def __init__(self, domain: Tuple[Tuple[float, float], ...]):
        """
        Initialize the base problem.

        Parameters
        ----------
        domain : Tuple[Tuple[float, float], ...]
            Domain boundaries for each spatial dimension.
        """
        self.domain = domain
        self.spatial_dim = len(domain)

    @abstractmethod
    def analytical_solution(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the analytical solution at given points.

        Parameters
        ----------
        x : torch.Tensor
            Input coordinates of shape (N, spatial_dim).

        Returns
        -------
        torch.Tensor
            Analytical solution values of shape (N, 1).
        """
        pass

    @abstractmethod
    def source_term(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the source term f(x) in the PDE.

        For Poisson equation: ∇²u = f

        Parameters
        ----------
        x : torch.Tensor
            Input coordinates of shape (N, spatial_dim).

        Returns
        -------
        torch.Tensor
            Source term values of shape (N, 1).
        """
        pass

    @abstractmethod
    def boundary_condition(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the boundary condition values at given boundary points.

        Parameters
        ----------
        x : torch.Tensor
            Boundary point coordinates of shape (N, spatial_dim).

        Returns
        -------
        torch.Tensor
            Boundary condition values of shape (N, 1).
        """
        pass

    @abstractmethod
    def sample_interior_points(self, n: int, random_seed: int = None) -> torch.Tensor:
        """
        Sample collocation points inside the domain.

        Uses Latin Hypercube Sampling for better coverage than uniform random sampling.

        Parameters
        ----------
        n : int
            Number of interior points to sample.
        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        torch.Tensor
            Interior point coordinates of shape (n, spatial_dim).
        """
        pass

    @abstractmethod
    def sample_boundary_points(self, n_per_edge: int, random_seed: int = None) -> torch.Tensor:
        """
        Sample points on the domain boundary.

        For a 2D rectangular domain, samples n_per_edge points on each of the 4 edges.
        Points should lie exactly on the boundary (not approximately).

        Parameters
        ----------
        n_per_edge : int
            Number of points to sample per boundary edge.
        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        torch.Tensor
            Boundary point coordinates of shape (total_boundary_points, spatial_dim).
        """
        pass

    def compute_relative_l2_error(
        self, model: torch.nn.Module, n_test_points: int = 10000, random_seed: int = 42
    ) -> float:
        """
        Compute the relative L2 error between model prediction and analytical solution.

        Relative L2 error = ||u_pred - u_exact||_L2 / ||u_exact||_L2

        Parameters
        ----------
        model : torch.nn.Module
            Trained PINN model.
        n_test_points : int, optional
            Number of test points for error computation. Default: 10000.
        random_seed : int, optional
            Random seed for reproducibility. Default: 42.

        Returns
        -------
        float
            Relative L2 error as a percentage.
        """
        # Sample test points from interior
        x_test = self.sample_interior_points(n_test_points, random_seed=random_seed)

        # Compute analytical solution
        u_exact = self.analytical_solution(x_test)

        # Compute model prediction
        model.eval()
        with torch.no_grad():
            u_pred = model(x_test)

        # Compute relative L2 error
        numerator = torch.norm(u_pred - u_exact)
        denominator = torch.norm(u_exact)

        relative_error = (numerator / denominator).item()

        return relative_error * 100  # Return as percentage

    def __repr__(self) -> str:
        """String representation of the problem."""
        domain_str = ", ".join([f"[{low:.2f}, {high:.2f}]" for low, high in self.domain])
        return f"{self.__class__.__name__}(domain=({domain_str}), spatial_dim={self.spatial_dim})"
