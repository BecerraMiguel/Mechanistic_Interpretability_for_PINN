"""
Poisson equation problem definition with manufactured solution.

This module implements the 2D Poisson equation with Dirichlet boundary conditions:
    ∇²u = f  on [0,1]²
    u = 0    on ∂Ω

Manufactured solution: u(x,y) = sin(πx)sin(πy)
Source term: f(x,y) = -2π²sin(πx)sin(πy)
"""

import math
from typing import Tuple

import torch
from scipy.stats import qmc

from .base import BaseProblem


class PoissonProblem(BaseProblem):
    """
    2D Poisson equation problem with manufactured solution.

    The problem is defined on the unit square [0,1]² with Dirichlet boundary conditions.
    The manufactured solution allows us to verify the correctness of the PINN by comparing
    with the known analytical solution.

    Equation:
        ∇²u = f  on Ω = [0,1]²
        u = 0    on ∂Ω

    Analytical solution:
        u(x,y) = sin(πx)sin(πy)

    Source term (forcing function):
        f(x,y) = -2π²sin(πx)sin(πy)

    This is derived from:
        ∂²u/∂x² = -π²sin(πx)sin(πy)
        ∂²u/∂y² = -π²sin(πx)sin(πy)
        ∇²u = ∂²u/∂x² + ∂²u/∂y² = -2π²sin(πx)sin(πy)

    Parameters
    ----------
    domain : Tuple[Tuple[float, float], ...], optional
        Domain boundaries. Default: ((0, 1), (0, 1)) for unit square.
    """

    def __init__(
        self, domain: Tuple[Tuple[float, float], ...] = ((0.0, 1.0), (0.0, 1.0))
    ):
        """
        Initialize the Poisson problem.

        Parameters
        ----------
        domain : Tuple[Tuple[float, float], ...], optional
            Domain boundaries. Default: ((0, 1), (0, 1)).
        """
        super().__init__(domain)

        # Verify 2D domain
        if self.spatial_dim != 2:
            raise ValueError(
                f"PoissonProblem only supports 2D domains, got {self.spatial_dim}D"
            )

        # Store domain bounds for convenience
        self.x_min, self.x_max = domain[0]
        self.y_min, self.y_max = domain[1]

    def analytical_solution(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute analytical solution u(x,y) = sin(πx)sin(πy).

        Parameters
        ----------
        x : torch.Tensor
            Input coordinates of shape (N, 2).

        Returns
        -------
        torch.Tensor
            Analytical solution values of shape (N, 1).
        """
        if x.shape[1] != 2:
            raise ValueError(f"Expected input of shape (N, 2), got {x.shape}")

        x_coord = x[:, 0:1]  # (N, 1)
        y_coord = x[:, 1:2]  # (N, 1)

        u = torch.sin(math.pi * x_coord) * torch.sin(math.pi * y_coord)
        return u

    def source_term(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute source term f(x,y) = -2π²sin(πx)sin(πy).

        Parameters
        ----------
        x : torch.Tensor
            Input coordinates of shape (N, 2).

        Returns
        -------
        torch.Tensor
            Source term values of shape (N, 1).
        """
        if x.shape[1] != 2:
            raise ValueError(f"Expected input of shape (N, 2), got {x.shape}")

        x_coord = x[:, 0:1]  # (N, 1)
        y_coord = x[:, 1:2]  # (N, 1)

        # f = -2π² sin(πx)sin(πy)
        f = (
            -2
            * (math.pi**2)
            * torch.sin(math.pi * x_coord)
            * torch.sin(math.pi * y_coord)
        )
        return f

    def boundary_condition(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute boundary condition u = 0 on all boundaries.

        For the manufactured solution sin(πx)sin(πy), the boundary condition is
        automatically satisfied since sin(0) = sin(π) = 0.

        Parameters
        ----------
        x : torch.Tensor
            Boundary point coordinates of shape (N, 2).

        Returns
        -------
        torch.Tensor
            Boundary condition values of shape (N, 1), all zeros.
        """
        if x.shape[1] != 2:
            raise ValueError(f"Expected input of shape (N, 2), got {x.shape}")

        # Dirichlet BC: u = 0 on boundary
        return torch.zeros(x.shape[0], 1, dtype=x.dtype, device=x.device)

    def sample_interior_points(self, n: int, random_seed: int = None) -> torch.Tensor:
        """
        Sample interior collocation points using Latin Hypercube Sampling.

        Latin Hypercube Sampling provides better coverage than uniform random sampling
        by ensuring that each dimension is evenly divided.

        Parameters
        ----------
        n : int
            Number of interior points to sample.
        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        torch.Tensor
            Interior point coordinates of shape (n, 2).
        """
        if n <= 0:
            raise ValueError(f"Number of points must be positive, got {n}")

        # Create Latin Hypercube sampler
        sampler = qmc.LatinHypercube(d=2, seed=random_seed)

        # Sample points in [0, 1]²
        points = sampler.random(n=n)  # Shape: (n, 2), values in [0, 1]

        # Scale to domain bounds
        x_range = self.x_max - self.x_min
        y_range = self.y_max - self.y_min

        points[:, 0] = points[:, 0] * x_range + self.x_min
        points[:, 1] = points[:, 1] * y_range + self.y_min

        return torch.tensor(points, dtype=torch.float32)

    def sample_boundary_points(
        self, n_per_edge: int, random_seed: int = None
    ) -> torch.Tensor:
        """
        Sample points uniformly on the domain boundary.

        For a 2D rectangular domain [x_min, x_max] × [y_min, y_max], samples points
        on all 4 edges:
        - Bottom edge: y = y_min
        - Right edge: x = x_max
        - Top edge: y = y_max
        - Left edge: x = x_min

        Points lie exactly on the boundary edges.

        Parameters
        ----------
        n_per_edge : int
            Number of points to sample per edge (total points = 4 * n_per_edge).
        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        torch.Tensor
            Boundary point coordinates of shape (4 * n_per_edge, 2).
        """
        if n_per_edge <= 0:
            raise ValueError(
                f"Number of points per edge must be positive, got {n_per_edge}"
            )

        # Set random seed if provided (for consistency with interior sampling)
        if random_seed is not None:
            torch.manual_seed(random_seed)

        # Sample uniformly along each edge using linspace for exact boundary points
        x_points = torch.linspace(self.x_min, self.x_max, n_per_edge)
        y_points = torch.linspace(self.y_min, self.y_max, n_per_edge)

        # Create boundary points for each edge
        boundary_points = []

        # Bottom edge: y = y_min, x varies
        bottom = torch.stack([x_points, torch.full_like(x_points, self.y_min)], dim=1)
        boundary_points.append(bottom)

        # Right edge: x = x_max, y varies
        right = torch.stack([torch.full_like(y_points, self.x_max), y_points], dim=1)
        boundary_points.append(right)

        # Top edge: y = y_max, x varies
        top = torch.stack([x_points, torch.full_like(x_points, self.y_max)], dim=1)
        boundary_points.append(top)

        # Left edge: x = x_min, y varies
        left = torch.stack([torch.full_like(y_points, self.x_min), y_points], dim=1)
        boundary_points.append(left)

        # Concatenate all boundary points
        all_boundary_points = torch.cat(
            boundary_points, dim=0
        )  # Shape: (4 * n_per_edge, 2)

        return all_boundary_points

    def pde_residual(
        self,
        u: torch.Tensor,
        x: torch.Tensor,
        du_dx: torch.Tensor,
        d2u_dx2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute PDE residual: ∇²u - f(x).

        This method is used by PINN models during training to compute the PDE loss.

        Parameters
        ----------
        u : torch.Tensor
            Network prediction at points x, shape (N, 1).
        x : torch.Tensor
            Input coordinates, shape (N, 2).
        du_dx : torch.Tensor
            First-order derivatives (gradient), shape (N, 2).
        d2u_dx2 : torch.Tensor
            Second-order derivatives (Laplacian), shape (N, 1).

        Returns
        -------
        torch.Tensor
            PDE residual values, shape (N, 1).
        """
        # Laplacian of u
        laplacian_u = d2u_dx2

        # Source term
        f = self.source_term(x)

        # PDE residual: ∇²u - f = 0
        residual = laplacian_u - f

        return residual

    def analytical_derivative_du_dx(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute analytical first derivative ∂u/∂x = π·cos(πx)·sin(πy).

        This is the ground-truth derivative computed from the analytical solution
        u(x,y) = sin(πx)sin(πy). Used for training probing classifiers.

        Parameters
        ----------
        x : torch.Tensor
            Input coordinates of shape (N, 2).

        Returns
        -------
        torch.Tensor
            First derivative ∂u/∂x values of shape (N, 1).

        Examples
        --------
        >>> problem = PoissonProblem()
        >>> x = torch.tensor([[0.5, 0.5], [0.25, 0.75]])
        >>> du_dx = problem.analytical_derivative_du_dx(x)
        """
        if x.shape[1] != 2:
            raise ValueError(f"Expected input of shape (N, 2), got {x.shape}")

        x_coord = x[:, 0:1]  # (N, 1)
        y_coord = x[:, 1:2]  # (N, 1)

        # ∂u/∂x = ∂/∂x[sin(πx)sin(πy)] = π·cos(πx)·sin(πy)
        du_dx = math.pi * torch.cos(math.pi * x_coord) * torch.sin(math.pi * y_coord)
        return du_dx

    def analytical_derivative_du_dy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute analytical first derivative ∂u/∂y = π·sin(πx)·cos(πy).

        This is the ground-truth derivative computed from the analytical solution
        u(x,y) = sin(πx)sin(πy). Used for training probing classifiers.

        Parameters
        ----------
        x : torch.Tensor
            Input coordinates of shape (N, 2).

        Returns
        -------
        torch.Tensor
            First derivative ∂u/∂y values of shape (N, 1).

        Examples
        --------
        >>> problem = PoissonProblem()
        >>> x = torch.tensor([[0.5, 0.5], [0.25, 0.75]])
        >>> du_dy = problem.analytical_derivative_du_dy(x)
        """
        if x.shape[1] != 2:
            raise ValueError(f"Expected input of shape (N, 2), got {x.shape}")

        x_coord = x[:, 0:1]  # (N, 1)
        y_coord = x[:, 1:2]  # (N, 1)

        # ∂u/∂y = ∂/∂y[sin(πx)sin(πy)] = π·sin(πx)·cos(πy)
        du_dy = math.pi * torch.sin(math.pi * x_coord) * torch.cos(math.pi * y_coord)
        return du_dy

    def analytical_derivative_d2u_dx2(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute analytical second derivative ∂²u/∂x² = -π²·sin(πx)·sin(πy).

        This is the ground-truth second derivative computed from the analytical solution
        u(x,y) = sin(πx)sin(πy). Used for training probing classifiers.

        Parameters
        ----------
        x : torch.Tensor
            Input coordinates of shape (N, 2).

        Returns
        -------
        torch.Tensor
            Second derivative ∂²u/∂x² values of shape (N, 1).

        Examples
        --------
        >>> problem = PoissonProblem()
        >>> x = torch.tensor([[0.5, 0.5], [0.25, 0.75]])
        >>> d2u_dx2 = problem.analytical_derivative_d2u_dx2(x)
        """
        if x.shape[1] != 2:
            raise ValueError(f"Expected input of shape (N, 2), got {x.shape}")

        x_coord = x[:, 0:1]  # (N, 1)
        y_coord = x[:, 1:2]  # (N, 1)

        # ∂²u/∂x² = ∂/∂x[π·cos(πx)·sin(πy)] = -π²·sin(πx)·sin(πy)
        d2u_dx2 = (
            -(math.pi**2) * torch.sin(math.pi * x_coord) * torch.sin(math.pi * y_coord)
        )
        return d2u_dx2

    def analytical_derivative_d2u_dy2(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute analytical second derivative ∂²u/∂y² = -π²·sin(πx)·sin(πy).

        This is the ground-truth second derivative computed from the analytical solution
        u(x,y) = sin(πx)sin(πy). Used for training probing classifiers.

        Parameters
        ----------
        x : torch.Tensor
            Input coordinates of shape (N, 2).

        Returns
        -------
        torch.Tensor
            Second derivative ∂²u/∂y² values of shape (N, 1).

        Examples
        --------
        >>> problem = PoissonProblem()
        >>> x = torch.tensor([[0.5, 0.5], [0.25, 0.75]])
        >>> d2u_dy2 = problem.analytical_derivative_d2u_dy2(x)
        """
        if x.shape[1] != 2:
            raise ValueError(f"Expected input of shape (N, 2), got {x.shape}")

        x_coord = x[:, 0:1]  # (N, 1)
        y_coord = x[:, 1:2]  # (N, 1)

        # ∂²u/∂y² = ∂/∂y[π·sin(πx)·cos(πy)] = -π²·sin(πx)·sin(πy)
        d2u_dy2 = (
            -(math.pi**2) * torch.sin(math.pi * x_coord) * torch.sin(math.pi * y_coord)
        )
        return d2u_dy2

    def analytical_laplacian(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute analytical Laplacian ∇²u = ∂²u/∂x² + ∂²u/∂y² = -2π²·sin(πx)·sin(πy).

        This is the ground-truth Laplacian computed from the analytical solution
        u(x,y) = sin(πx)sin(πy). Note that this equals -source_term(x).

        Parameters
        ----------
        x : torch.Tensor
            Input coordinates of shape (N, 2).

        Returns
        -------
        torch.Tensor
            Laplacian ∇²u values of shape (N, 1).

        Notes
        -----
        The Laplacian equals the sum of second derivatives:
        ∇²u = ∂²u/∂x² + ∂²u/∂y² = -π²·sin(πx)·sin(πy) + (-π²·sin(πx)·sin(πy))
            = -2π²·sin(πx)·sin(πy)

        This also equals -source_term(x) by construction.

        Examples
        --------
        >>> problem = PoissonProblem()
        >>> x = torch.tensor([[0.5, 0.5], [0.25, 0.75]])
        >>> laplacian = problem.analytical_laplacian(x)
        """
        if x.shape[1] != 2:
            raise ValueError(f"Expected input of shape (N, 2), got {x.shape}")

        x_coord = x[:, 0:1]  # (N, 1)
        y_coord = x[:, 1:2]  # (N, 1)

        # ∇²u = -2π²·sin(πx)·sin(πy)
        laplacian = (
            -2.0
            * (math.pi**2)
            * torch.sin(math.pi * x_coord)
            * torch.sin(math.pi * y_coord)
        )
        return laplacian

    def analytical_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute analytical gradient ∇u = (∂u/∂x, ∂u/∂y).

        Returns both first-order derivatives as a single tensor. Convenience method
        that calls analytical_derivative_du_dx and analytical_derivative_du_dy.

        Parameters
        ----------
        x : torch.Tensor
            Input coordinates of shape (N, 2).

        Returns
        -------
        torch.Tensor
            Gradient values of shape (N, 2), where:
            - Column 0: ∂u/∂x = π·cos(πx)·sin(πy)
            - Column 1: ∂u/∂y = π·sin(πx)·cos(πy)

        Examples
        --------
        >>> problem = PoissonProblem()
        >>> x = torch.tensor([[0.5, 0.5], [0.25, 0.75]])
        >>> gradient = problem.analytical_gradient(x)  # Shape: (2, 2)
        """
        du_dx = self.analytical_derivative_du_dx(x)
        du_dy = self.analytical_derivative_du_dy(x)
        return torch.cat([du_dx, du_dy], dim=1)

    def __repr__(self) -> str:
        """String representation of the Poisson problem."""
        return (
            f"PoissonProblem(\n"
            f"  domain=[{self.x_min}, {self.x_max}] × [{self.y_min}, {self.y_max}],\n"
            f"  equation: ∇²u = f,\n"
            f"  BC: u = 0 on ∂Ω,\n"
            f"  analytical_solution: u(x,y) = sin(πx)sin(πy)\n"
            f")"
        )
