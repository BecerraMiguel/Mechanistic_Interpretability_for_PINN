"""
PINN model architectures.

This module provides various neural network architectures for Physics-Informed
Neural Networks, including:
- BasePINN: Abstract base class for all PINN models
- MLP: Standard multi-layer perceptron
"""

from .base import BasePINN
from .mlp import MLP

__all__ = ['BasePINN', 'MLP']
