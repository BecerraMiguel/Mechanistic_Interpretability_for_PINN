"""
Automatic differentiation utilities for computing derivatives in PINNs.

This module provides functions to compute spatial and temporal derivatives
of neural network outputs using PyTorch's automatic differentiation.
All derivatives are computed via torch.autograd.grad().
"""

from typing import Optional, Tuple

import torch


def compute_derivatives(u: torch.Tensor, x: torch.Tensor, order: int = 1) -> torch.Tensor:
    """
    Compute spatial derivatives of network output using automatic differentiation.

    This function computes derivatives of arbitrary order for scalar or vector
    outputs with respect to input coordinates. For order=1, it computes the
    gradient. For order=2, it computes the Laplacian.

    Args:
        u: Network output, shape (N, output_dim) or (N, 1)
        x: Input coordinates, shape (N, input_dim) where input_dim is the
           spatial dimension. Must have requires_grad=True.
        order: Derivative order:
               - 1: First-order gradient ∇u
               - 2: Second-order Laplacian ∇²u

    Returns:
        Derivatives tensor:
        - order=1: shape (N, input_dim) containing [∂u/∂x₁, ∂u/∂x₂, ...]
        - order=2: shape (N, 1) containing ∇²u = ∂²u/∂x₁² + ∂²u/∂x₂² + ...

    Raises:
        ValueError: If order is not 1 or 2, or if x doesn't require gradients

    Example:
        >>> x = torch.randn(100, 2, requires_grad=True)  # 100 points in 2D
        >>> u = model(x)  # shape (100, 1)
        >>> du_dx = compute_derivatives(u, x, order=1)  # shape (100, 2)
        >>> laplacian = compute_derivatives(u, x, order=2)  # shape (100, 1)
    """
    if not x.requires_grad:
        raise ValueError(
            "Input coordinates x must have requires_grad=True for derivative computation"
        )

    if order not in [1, 2]:
        raise ValueError(f"Only order 1 and 2 are supported, got order={order}")

    if order == 1:
        # Compute first-order derivatives: ∇u = [∂u/∂x₁, ∂u/∂x₂, ...]
        grad = torch.autograd.grad(
            outputs=u,
            inputs=x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,  # Needed for second-order derivatives
            retain_graph=True,
        )[0]
        return grad  # shape (N, input_dim)

    elif order == 2:
        # Compute Laplacian: ∇²u = ∂²u/∂x₁² + ∂²u/∂x₂² + ...
        # First compute first-order derivatives
        grad = torch.autograd.grad(
            outputs=u,
            inputs=x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Then compute second-order derivatives for each dimension
        laplacian = torch.zeros((x.shape[0], 1), device=x.device)
        for i in range(x.shape[1]):
            grad_i = grad[:, i : i + 1]  # ∂u/∂xᵢ
            grad2_ii = torch.autograd.grad(
                outputs=grad_i,
                inputs=x,
                grad_outputs=torch.ones_like(grad_i),
                create_graph=True,
                retain_graph=True,
            )[0][
                :, i : i + 1
            ]  # ∂²u/∂xᵢ²
            laplacian += grad2_ii

        return laplacian  # shape (N, 1)


def compute_gradient_components(u: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """
    Compute individual gradient components ∂u/∂xᵢ for each dimension.

    This is useful when you need access to individual derivative components
    rather than the full gradient vector.

    Args:
        u: Network output, shape (N, 1)
        x: Input coordinates, shape (N, input_dim). Must have requires_grad=True.

    Returns:
        Tuple of tensors, one for each input dimension, each with shape (N, 1):
        (∂u/∂x₁, ∂u/∂x₂, ...)

    Example:
        >>> x = torch.randn(100, 2, requires_grad=True)
        >>> u = model(x)
        >>> du_dx, du_dy = compute_gradient_components(u, x)
    """
    if not x.requires_grad:
        raise ValueError(
            "Input coordinates x must have requires_grad=True for derivative computation"
        )

    # Compute full gradient
    grad = torch.autograd.grad(
        outputs=u, inputs=x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True
    )[0]

    # Split into individual components
    components = tuple(grad[:, i : i + 1] for i in range(x.shape[1]))
    return components


def compute_hessian_diagonal(u: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """
    Compute diagonal elements of the Hessian matrix: ∂²u/∂xᵢ².

    This computes only the unmixed second derivatives (diagonal of Hessian),
    not the mixed derivatives ∂²u/∂xᵢ∂xⱼ.

    Args:
        u: Network output, shape (N, 1)
        x: Input coordinates, shape (N, input_dim). Must have requires_grad=True.

    Returns:
        Tuple of tensors, one for each input dimension, each with shape (N, 1):
        (∂²u/∂x₁², ∂²u/∂x₂², ...)

    Example:
        >>> x = torch.randn(100, 2, requires_grad=True)
        >>> u = model(x)
        >>> d2u_dx2, d2u_dy2 = compute_hessian_diagonal(u, x)
    """
    if not x.requires_grad:
        raise ValueError(
            "Input coordinates x must have requires_grad=True for derivative computation"
        )

    # First compute first-order derivatives
    grad = torch.autograd.grad(
        outputs=u, inputs=x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True
    )[0]

    # Compute second-order derivatives for each dimension
    hessian_diag = []
    for i in range(x.shape[1]):
        grad_i = grad[:, i : i + 1]
        grad2_ii = torch.autograd.grad(
            outputs=grad_i,
            inputs=x,
            grad_outputs=torch.ones_like(grad_i),
            create_graph=True,
            retain_graph=True,
        )[0][:, i : i + 1]
        hessian_diag.append(grad2_ii)

    return tuple(hessian_diag)


def compute_mixed_derivative(u: torch.Tensor, x: torch.Tensor, i: int, j: int) -> torch.Tensor:
    """
    Compute a specific mixed second derivative ∂²u/∂xᵢ∂xⱼ.

    Args:
        u: Network output, shape (N, 1)
        x: Input coordinates, shape (N, input_dim). Must have requires_grad=True.
        i: First dimension index (0-based)
        j: Second dimension index (0-based)

    Returns:
        Mixed derivative ∂²u/∂xᵢ∂xⱼ, shape (N, 1)

    Example:
        >>> x = torch.randn(100, 2, requires_grad=True)
        >>> u = model(x)
        >>> d2u_dxdy = compute_mixed_derivative(u, x, 0, 1)  # ∂²u/∂x∂y
    """
    if not x.requires_grad:
        raise ValueError(
            "Input coordinates x must have requires_grad=True for derivative computation"
        )

    if i < 0 or i >= x.shape[1] or j < 0 or j >= x.shape[1]:
        raise ValueError(f"Indices i={i}, j={j} out of bounds for input_dim={x.shape[1]}")

    # First compute ∂u/∂xᵢ
    grad = torch.autograd.grad(
        outputs=u, inputs=x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True
    )[0]

    grad_i = grad[:, i : i + 1]

    # Then compute ∂/∂xⱼ(∂u/∂xᵢ) = ∂²u/∂xᵢ∂xⱼ
    grad2_ij = torch.autograd.grad(
        outputs=grad_i,
        inputs=x,
        grad_outputs=torch.ones_like(grad_i),
        create_graph=True,
        retain_graph=True,
    )[0][:, j : j + 1]

    return grad2_ij
