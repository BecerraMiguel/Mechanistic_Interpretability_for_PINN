"""
Base class for Physics-Informed Neural Networks (PINNs).

This module provides the abstract base class that all PINN architectures
inherit from, defining the interface for forward passes, PDE residual
computation, and training.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn


class BasePINN(nn.Module, ABC):
    """
    Abstract base class for Physics-Informed Neural Networks.

    All PINN architectures (MLP, Modified Fourier Network, Attention-Enhanced)
    inherit from this class and implement the required abstract methods.

    The PINN training objective minimizes:
        L = w_pde * L_pde + w_bc * L_bc + w_ic * L_ic

    where:
        - L_pde: PDE residual at interior collocation points
        - L_bc: Boundary condition violations
        - L_ic: Initial condition errors

    Attributes:
        input_dim (int): Dimension of input coordinates (e.g., 2 for (x,y))
        output_dim (int): Dimension of output solution (typically 1 for scalar PDEs)
    """

    def __init__(self, input_dim: int, output_dim: int):
        """
        Initialize the BasePINN.

        Args:
            input_dim: Dimension of input coordinates (e.g., 2 for (x,y), 3 for (x,y,t))
            output_dim: Dimension of output solution (typically 1 for scalar PDEs)
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input coordinates, shape (batch_size, input_dim)

        Returns:
            Network output u(x), shape (batch_size, output_dim)
        """
        pass

    @abstractmethod
    def compute_pde_residual(
        self,
        x: torch.Tensor,
        pde_fn: Optional[callable] = None
    ) -> torch.Tensor:
        """
        Compute the PDE residual N[u](x) at given points.

        The PDE residual represents how well the network output satisfies
        the differential equation. For example, for the Poisson equation
        ∇²u = f, the residual would be: N[u] = ∇²u - f

        Derivatives are computed using automatic differentiation via
        torch.autograd.grad().

        Args:
            x: Collocation points, shape (batch_size, input_dim)
            pde_fn: Optional function defining the PDE. If None, uses
                   the default PDE associated with the problem.

        Returns:
            PDE residual values, shape (batch_size, 1)
        """
        pass

    def train_step(
        self,
        x_interior: torch.Tensor,
        x_boundary: torch.Tensor,
        u_boundary: torch.Tensor,
        x_initial: Optional[torch.Tensor] = None,
        u_initial: Optional[torch.Tensor] = None,
        pde_fn: Optional[callable] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Perform a single training step, computing all loss components.

        Args:
            x_interior: Interior collocation points for PDE residual,
                       shape (n_interior, input_dim)
            x_boundary: Boundary points, shape (n_boundary, input_dim)
            u_boundary: Boundary condition values, shape (n_boundary, output_dim)
            x_initial: Initial condition points (for time-dependent PDEs),
                      shape (n_initial, input_dim)
            u_initial: Initial condition values, shape (n_initial, output_dim)
            pde_fn: Optional PDE function
            weights: Loss weights dict with keys 'pde', 'bc', 'ic'.
                    Defaults to {'pde': 1.0, 'bc': 1.0, 'ic': 1.0}

        Returns:
            Dictionary containing:
                - 'loss_total': Total weighted loss
                - 'loss_pde': PDE residual loss
                - 'loss_bc': Boundary condition loss
                - 'loss_ic': Initial condition loss (0 if not applicable)
        """
        if weights is None:
            weights = {'pde': 1.0, 'bc': 1.0, 'ic': 1.0}

        # PDE residual loss
        residual = self.compute_pde_residual(x_interior, pde_fn)
        loss_pde = torch.mean(residual ** 2)

        # Boundary condition loss
        u_pred_bc = self.forward(x_boundary)
        loss_bc = torch.mean((u_pred_bc - u_boundary) ** 2)

        # Initial condition loss (if applicable)
        loss_ic = torch.tensor(0.0, device=x_interior.device)
        if x_initial is not None and u_initial is not None:
            u_pred_ic = self.forward(x_initial)
            loss_ic = torch.mean((u_pred_ic - u_initial) ** 2)

        # Total weighted loss
        loss_total = (
            weights['pde'] * loss_pde +
            weights['bc'] * loss_bc +
            weights['ic'] * loss_ic
        )

        return {
            'loss_total': loss_total,
            'loss_pde': loss_pde,
            'loss_bc': loss_bc,
            'loss_ic': loss_ic
        }

    def get_parameters_count(self) -> int:
        """
        Get the total number of trainable parameters.

        Returns:
            Total number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        """String representation of the model."""
        return (
            f"{self.__class__.__name__}(\n"
            f"  input_dim={self.input_dim},\n"
            f"  output_dim={self.output_dim},\n"
            f"  parameters={self.get_parameters_count():,}\n"
            f")"
        )
