"""
Utility functions for PINN training and analysis.

This module provides helper functions for:
- Derivative computation via automatic differentiation
- Visualization tools
- Data processing utilities
"""

from .derivatives import (
    compute_derivatives,
    compute_gradient_components,
    compute_hessian_diagonal,
    compute_mixed_derivative
)

__all__ = [
    'compute_derivatives',
    'compute_gradient_components',
    'compute_hessian_diagonal',
    'compute_mixed_derivative'
]
