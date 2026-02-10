"""
Demo: Using LinearProbe to detect linear relationships in activations.

This script demonstrates how to use the LinearProbe class to determine
whether target information (e.g., derivatives) is linearly accessible
from network activations.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.interpretability import LinearProbe

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("=" * 70)
print("LinearProbe Demo: Detecting Linear Relationships")
print("=" * 70)

# ============================================================================
# Example 1: Perfect Linear Relationship
# ============================================================================
print("\n" + "=" * 70)
print("Example 1: Perfect Linear Relationship")
print("=" * 70)

# Create synthetic "activations" and "target derivative"
# Scenario: Activations encode the derivative perfectly via linear combination
n_samples = 1000
input_dim = 64

# Generate random activations
activations = torch.randn(n_samples, input_dim)

# Create target as linear combination of activations (perfect linear relationship)
true_weights = torch.randn(input_dim, 1)
target = activations @ true_weights

print(f"\nData: {n_samples} samples, {input_dim}-dimensional activations")
print(f"Target: Linear combination of activations (perfect relationship)")

# Train linear probe
probe = LinearProbe(input_dim=input_dim, output_dim=1)
print(f"\nProbe: {probe}")

history = probe.fit(activations, target, epochs=1000, lr=1e-2, verbose=False)
print(f"Training: {len(history['loss'])} epochs")
print(f"Initial loss: {history['loss'][0]:.6f}")
print(f"Final loss: {history['loss'][-1]:.6f}")

# Evaluate
scores = probe.score(activations, target)
print(f"\nPerformance:")
print(f"  MSE: {scores['mse']:.6f}")
print(f"  R²: {scores['r_squared']:.6f}")
print(f"  Explained Variance: {scores['explained_variance']:.6f}")

if scores["r_squared"] > 0.99:
    print(f"\n✓ Excellent! R² > 0.99 indicates target is perfectly linearly accessible")

# ============================================================================
# Example 2: Linear Relationship with Noise
# ============================================================================
print("\n" + "=" * 70)
print("Example 2: Linear Relationship with Noise")
print("=" * 70)

# Create target with some noise
target_noisy = activations @ true_weights + 0.3 * torch.randn(n_samples, 1)

print(f"\nData: Same activations, but target has added noise")
print(f"Noise level: 0.3 (moderate)")

# Train new probe
probe_noisy = LinearProbe(input_dim=input_dim, output_dim=1)
probe_noisy.fit(activations, target_noisy, epochs=1000, lr=1e-2)

# Evaluate
scores_noisy = probe_noisy.score(activations, target_noisy)
print(f"\nPerformance:")
print(f"  MSE: {scores_noisy['mse']:.6f}")
print(f"  R²: {scores_noisy['r_squared']:.6f}")
print(f"  Explained Variance: {scores_noisy['explained_variance']:.6f}")

if 0.5 < scores_noisy["r_squared"] < 0.99:
    print(f"\n✓ Good! R² ≈ {scores_noisy['r_squared']:.2f} indicates strong but imperfect linear relationship")

# ============================================================================
# Example 3: Non-Linear Relationship (Probe Fails)
# ============================================================================
print("\n" + "=" * 70)
print("Example 3: Non-Linear Relationship (Linear Probe Should Fail)")
print("=" * 70)

# Create target with non-linear relationship
# Target is quadratic function of activations (not linearly accessible)
target_nonlinear = torch.sum(activations**2, dim=1, keepdim=True)

print(f"\nData: Same activations")
print(f"Target: Sum of squared activations (quadratic, not linear)")

# Train probe
probe_nonlinear = LinearProbe(input_dim=input_dim, output_dim=1)
probe_nonlinear.fit(activations, target_nonlinear, epochs=1000, lr=1e-2)

# Evaluate
scores_nonlinear = probe_nonlinear.score(activations, target_nonlinear)
print(f"\nPerformance:")
print(f"  MSE: {scores_nonlinear['mse']:.6f}")
print(f"  R²: {scores_nonlinear['r_squared']:.6f}")
print(f"  Explained Variance: {scores_nonlinear['explained_variance']:.6f}")

if scores_nonlinear["r_squared"] < 0.5:
    print(f"\n✓ Expected! R² < 0.5 indicates target is NOT linearly accessible")
    print(f"  (This is correct - quadratic relationships can't be captured by linear probes)")

# ============================================================================
# Visualization: Training Loss Curves
# ============================================================================
print("\n" + "=" * 70)
print("Generating Visualizations...")
print("=" * 70)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: Perfect relationship - loss curve
axes[0].plot(history["loss"], label="Training Loss", color="blue", linewidth=2)
axes[0].set_xlabel("Epoch", fontsize=12)
axes[0].set_ylabel("MSE Loss", fontsize=12)
axes[0].set_title(f"Perfect Linear (R²={scores['r_squared']:.4f})", fontsize=14)
axes[0].set_yscale("log")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Plot 2: Noisy relationship - comparison
predictions_noisy = probe_noisy.predict(activations).detach().numpy()
targets_noisy_np = target_noisy.numpy()
axes[1].scatter(
    targets_noisy_np, predictions_noisy, alpha=0.3, s=10, label="Predictions"
)
axes[1].plot(
    targets_noisy_np,
    targets_noisy_np,
    "r--",
    linewidth=2,
    label="Perfect Prediction",
)
axes[1].set_xlabel("True Target", fontsize=12)
axes[1].set_ylabel("Predicted Target", fontsize=12)
axes[1].set_title(f"Noisy Linear (R²={scores_noisy['r_squared']:.4f})", fontsize=14)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Non-linear relationship - comparison
predictions_nonlinear = probe_nonlinear.predict(activations).detach().numpy()
targets_nonlinear_np = target_nonlinear.numpy()
axes[2].scatter(
    targets_nonlinear_np,
    predictions_nonlinear,
    alpha=0.3,
    s=10,
    label="Predictions",
    color="orange",
)
axes[2].plot(
    targets_nonlinear_np,
    targets_nonlinear_np,
    "r--",
    linewidth=2,
    label="Perfect Prediction",
)
axes[2].set_xlabel("True Target", fontsize=12)
axes[2].set_ylabel("Predicted Target", fontsize=12)
axes[2].set_title(
    f"Non-Linear (R²={scores_nonlinear['r_squared']:.4f})", fontsize=14
)
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/demo_linear_probe.png", dpi=150, bbox_inches="tight")
print(f"\n✓ Visualization saved: outputs/demo_linear_probe.png")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("Summary: Interpreting R² Scores")
print("=" * 70)

print(f"""
The R² score tells us how well a linear probe can predict the target:

1. Perfect Linear (R² = {scores['r_squared']:.4f}):
   → Target is perfectly linearly accessible from activations
   → All information is preserved in linear form

2. Noisy Linear (R² = {scores_noisy['r_squared']:.4f}):
   → Target is mostly linearly accessible, but with some noise
   → Most information is preserved, but not all

3. Non-Linear (R² = {scores_nonlinear['r_squared']:.4f}):
   → Target is NOT linearly accessible from activations
   → Information exists in non-linear form (requires non-linear probe)

In PINN interpretability:
- High R² (>0.9): Derivative information is linearly accessible at this layer
- Medium R² (0.5-0.9): Derivative information is partially accessible
- Low R² (<0.5): Derivative information is not linearly accessible
  (may be encoded non-linearly, or not present at all)
""")

print("=" * 70)
print("Demo Complete!")
print("=" * 70)
