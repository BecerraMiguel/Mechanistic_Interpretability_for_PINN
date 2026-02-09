"""
Utility functions for PINN training and analysis.

This module provides helper functions for:
- Derivative computation via automatic differentiation
- Collocation point sampling strategies
- Visualization tools
- Data processing utilities
"""

from .derivatives import (
    compute_derivatives,
    compute_gradient_components,
    compute_hessian_diagonal,
    compute_mixed_derivative
)

from .sampling import (
    LatinHypercubeSampler,
    UniformRandomSampler,
    GridSampler,
    BoundarySampler,
    sample_collocation_points
)

__all__ = [
    # Derivatives
    'compute_derivatives',
    'compute_gradient_components',
    'compute_hessian_diagonal',
    'compute_mixed_derivative',
    # Sampling
    'LatinHypercubeSampler',
    'UniformRandomSampler',
    'GridSampler',
    'BoundarySampler',
    'sample_collocation_points'
]
