"""
Quick demo for training a PINN on the Poisson equation (reduced epochs for testing).
"""

import torch
from src.models import MLP
from src.problems import PoissonProblem
from src.training import train_pinn

print("=" * 80)
print("Quick Training Demo: PINN on Poisson Equation")
print("=" * 80)
print()

# Setup
model = MLP(input_dim=2, hidden_dims=[64, 64, 64], output_dim=1, activation="tanh")
problem = PoissonProblem()

print(f"Model parameters: {model.get_parameters_count():,}")
print(f"Problem: {problem.__class__.__name__}")
print()

# Quick training config (reduced for testing)
config = {
    "optimizer": "adam",
    "lr": 1e-3,
    "n_epochs": 1000,  # Reduced from 5000
    "n_interior": 1000,  # Reduced from 10000
    "n_boundary": 50,   # Reduced from 100
    "loss_weights": {"pde": 1.0, "bc": 1.0, "ic": 0.0},
    "device": "cpu",
    "resample_every": 1,
    "validate_every": 200,
    "print_every": 200,
}

print("Training configuration:")
print(f"  Epochs: {config['n_epochs']}")
print(f"  Interior points: {config['n_interior']}")
print(f"  Boundary points: {config['n_boundary'] * 4} ({config['n_boundary']} per edge)")
print()

# Train
trained_model, history = train_pinn(model, problem, config)

# Results
print()
print("=" * 80)
print("Results Summary")
print("=" * 80)
print(f"Initial loss: {history['loss_total'][0]:.6f}")
print(f"Final loss: {history['loss_total'][-1]:.6f}")
print(f"Loss reduction: {(1 - history['loss_total'][-1] / history['loss_total'][0]) * 100:.2f}%")
print(f"Final relative L2 error: {history['relative_l2_error'][-1]:.4f}%")
print()

if history['relative_l2_error'][-1] < 10.0:
    print("✓ Good progress! Error < 10% with limited training")
else:
    print("⚠ Error still high (expected with limited training)")

print("=" * 80)
