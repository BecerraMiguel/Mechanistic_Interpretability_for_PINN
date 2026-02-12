"""
Multi-Layer Perceptron (MLP) architecture for Physics-Informed Neural Networks.

This module implements a standard MLP PINN with configurable architecture,
multiple activation function options, and activation extraction capabilities
for mechanistic interpretability analysis.
"""

from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn

from ..utils.derivatives import compute_derivatives
from .base import BasePINN


class MLP(BasePINN):
    """
    Standard Multi-Layer Perceptron for Physics-Informed Neural Networks.

    This is the baseline PINN architecture consisting of fully-connected layers
    with configurable hidden dimensions and activation functions. The network
    includes forward hooks for extracting intermediate activations, which is
    essential for mechanistic interpretability analysis.

    Architecture:
        Input (input_dim) → FC → Activation → ... → FC → Activation → Output (output_dim)

    Attributes:
        input_dim (int): Dimension of input coordinates
        output_dim (int): Dimension of output solution
        hidden_dims (List[int]): List of hidden layer dimensions
        activation (str): Activation function name ('tanh', 'relu', 'gelu', 'sin')
        layers (nn.ModuleList): List of linear layers
        activations (Dict[str, torch.Tensor]): Storage for extracted activations
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int = 1,
        activation: str = "tanh",
    ):
        """
        Initialize the MLP PINN.

        Args:
            input_dim: Dimension of input coordinates (e.g., 2 for (x,y))
            hidden_dims: List of hidden layer dimensions (e.g., [50, 50, 50, 50])
            output_dim: Dimension of output solution (default: 1 for scalar PDEs)
            activation: Activation function ('tanh', 'relu', 'gelu', 'sin')

        Example:
            >>> model = MLP(input_dim=2, hidden_dims=[64, 64, 64], output_dim=1)
            >>> x = torch.randn(100, 2)
            >>> u = model(x)  # shape (100, 1)
        """
        super().__init__(input_dim, output_dim)

        self.hidden_dims = hidden_dims
        self.activation_name = activation

        # Select activation function
        self.activation_fn = self._get_activation_function(activation)

        # Build layers
        self.layers = nn.ModuleList()
        dims = [input_dim] + hidden_dims + [output_dim]

        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))

        # Initialize weights
        self._initialize_weights()

        # Storage for activations (for interpretability)
        self.activations: Dict[str, torch.Tensor] = {}
        self._register_hooks()

    def _get_activation_function(self, activation: str) -> Callable:
        """
        Get activation function by name.

        Args:
            activation: Name of activation function

        Returns:
            Activation function

        Raises:
            ValueError: If activation function is not supported
        """
        activation_map = {
            "tanh": torch.tanh,
            "relu": torch.relu,
            "gelu": torch.nn.functional.gelu,
            "sin": torch.sin,
        }

        if activation.lower() not in activation_map:
            raise ValueError(
                f"Activation '{activation}' not supported. "
                f"Choose from: {list(activation_map.keys())}"
            )

        return activation_map[activation.lower()]

    def _initialize_weights(self):
        """
        Initialize network weights using Xavier initialization.

        Xavier initialization is well-suited for tanh and sigmoid activations.
        For ReLU-based activations, He initialization would be more appropriate,
        but Xavier provides a reasonable default.
        """
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def _register_hooks(self):
        """
        Register forward hooks to capture intermediate activations.

        Note: Activations are now stored manually in the forward pass
        to capture post-activation values rather than pre-activation values.
        This method is kept for potential future use but doesn't register
        any hooks in the current implementation.
        """
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP network.

        Args:
            x: Input coordinates, shape (batch_size, input_dim)

        Returns:
            Network output u(x), shape (batch_size, output_dim)
        """
        # Clear previous activations
        self.activations.clear()

        # Forward pass through hidden layers
        h = x
        for i, layer in enumerate(self.layers[:-1]):
            h = layer(h)
            h = self.activation_fn(h)
            # Store post-activation values
            self.activations[f"layer_{i}"] = h.detach()

        # Output layer (no activation)
        u = self.layers[-1](h)

        return u

    def get_activations(self) -> Dict[str, torch.Tensor]:
        """
        Get the activations captured during the last forward pass.

        Returns:
            Dictionary mapping layer names to activation tensors:
            {
                'layer_0': tensor of shape (batch_size, hidden_dims[0]),
                'layer_1': tensor of shape (batch_size, hidden_dims[1]),
                ...
            }

        Note:
            Call forward() before calling this method to populate activations.

        Example:
            >>> model = MLP(input_dim=2, hidden_dims=[50, 50], output_dim=1)
            >>> x = torch.randn(100, 2)
            >>> u = model(x)
            >>> activations = model.get_activations()
            >>> print(activations['layer_0'].shape)  # torch.Size([100, 50])
        """
        return self.activations.copy()

    def compute_pde_residual(
        self, x: torch.Tensor, pde_fn: Optional[Callable] = None
    ) -> torch.Tensor:
        """
        Compute the PDE residual at given collocation points.

        This method provides a generic interface for computing PDE residuals.
        The actual PDE is defined by the pde_fn callback, which receives
        the network output u and input coordinates x, along with computed
        derivatives.

        Args:
            x: Collocation points, shape (N, input_dim). Must have
               requires_grad=True for derivative computation.
            pde_fn: Callable that computes the PDE residual given:
                    pde_fn(u, x, du_dx, d2u_dx2) -> residual
                    If None, returns zeros (must be overridden by problem-specific
                    implementation).

        Returns:
            PDE residual, shape (N, 1)

        Example:
            >>> def poisson_pde(u, x, du_dx, d2u_dx2):
            ...     # Poisson equation: ∇²u = f(x)
            ...     f = -2 * (torch.pi**2) * torch.sin(torch.pi * x[:, 0:1])
            ...     return d2u_dx2 - f
            >>>
            >>> model = MLP(input_dim=2, hidden_dims=[50, 50], output_dim=1)
            >>> x = torch.randn(100, 2, requires_grad=True)
            >>> residual = model.compute_pde_residual(x, pde_fn=poisson_pde)
        """
        if not x.requires_grad:
            raise ValueError(
                "Input x must have requires_grad=True for PDE residual computation"
            )

        # Compute network output
        u = self.forward(x)

        if pde_fn is None:
            # No PDE function provided, return zeros
            # This should be overridden in problem-specific implementations
            return torch.zeros_like(u)

        # Compute derivatives
        du_dx = compute_derivatives(u, x, order=1)  # First derivatives
        d2u_dx2 = compute_derivatives(u, x, order=2)  # Laplacian

        # Compute PDE residual using provided function
        residual = pde_fn(u, x, du_dx, d2u_dx2)

        return residual

    def count_layers(self) -> int:
        """
        Get the number of layers (including output layer).

        Returns:
            Number of layers in the network
        """
        return len(self.layers)

    def get_layer_dimensions(self) -> List[int]:
        """
        Get the dimensions of all layers.

        Returns:
            List of layer dimensions [input_dim, hidden_dim_1, ..., output_dim]
        """
        return [self.input_dim] + self.hidden_dims + [self.output_dim]

    def __repr__(self) -> str:
        """String representation of the MLP model."""
        return (
            f"{self.__class__.__name__}(\n"
            f"  input_dim={self.input_dim},\n"
            f"  hidden_dims={self.hidden_dims},\n"
            f"  output_dim={self.output_dim},\n"
            f"  activation={self.activation_name},\n"
            f"  parameters={self.get_parameters_count():,}\n"
            f")"
        )
