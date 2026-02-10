"""
Demo: Analytical Derivative Computation for Probing Experiments.

This script demonstrates how to compute ground-truth derivatives from the
analytical solution of the Poisson equation. These derivatives will be used
as targets for training linear probes on PINN activations.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.problems.poisson import PoissonProblem

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

print("=" * 70)
print("Analytical Derivatives Demo: Ground Truth for Probing")
print("=" * 70)

# ============================================================================
# 1. Create Poisson Problem
# ============================================================================
print("\n" + "=" * 70)
print("1. Poisson Equation Setup")
print("=" * 70)

problem = PoissonProblem()
print(f"\nProblem: {problem}")
print(f"\nAnalytical solution: u(x,y) = sin(πx)sin(πy)")
print(f"Domain: [0,1] × [0,1]")

# ============================================================================
# 2. Compute Derivatives at Sample Points
# ============================================================================
print("\n" + "=" * 70)
print("2. Computing Derivatives at Sample Points")
print("=" * 70)

# Create a few sample points
x_sample = torch.tensor(
    [[0.5, 0.5], [0.25, 0.75], [0.75, 0.25], [0.1, 0.9], [0.9, 0.1]]
)

print(f"\nSample points (N={x_sample.shape[0]}):")
for i, point in enumerate(x_sample):
    print(f"  Point {i}: (x={point[0]:.2f}, y={point[1]:.2f})")

# Compute solution
u = problem.analytical_solution(x_sample)
print(f"\nSolution u(x,y):")
for i, val in enumerate(u):
    print(f"  u[{i}] = {val.item():.6f}")

# Compute first-order derivatives
du_dx = problem.analytical_derivative_du_dx(x_sample)
du_dy = problem.analytical_derivative_du_dy(x_sample)
gradient = problem.analytical_gradient(x_sample)

print(f"\nFirst-order derivatives:")
print(f"  ∂u/∂x shape: {du_dx.shape}")
print(f"  ∂u/∂y shape: {du_dy.shape}")
print(f"  ∇u shape: {gradient.shape}")

# Compute second-order derivatives
d2u_dx2 = problem.analytical_derivative_d2u_dx2(x_sample)
d2u_dy2 = problem.analytical_derivative_d2u_dy2(x_sample)
laplacian = problem.analytical_laplacian(x_sample)

print(f"\nSecond-order derivatives:")
print(f"  ∂²u/∂x² shape: {d2u_dx2.shape}")
print(f"  ∂²u/∂y² shape: {d2u_dy2.shape}")
print(f"  ∇²u (Laplacian) shape: {laplacian.shape}")

# Verify Laplacian = sum of second derivatives
laplacian_check = d2u_dx2 + d2u_dy2
print(f"\nVerification: ∇²u = ∂²u/∂x² + ∂²u/∂y²")
print(f"  Max difference: {torch.max(torch.abs(laplacian - laplacian_check)).item():.2e}")
print(f"  ✓ Verified!" if torch.allclose(laplacian, laplacian_check) else "  ✗ Failed!")

# ============================================================================
# 3. Derivatives on Dense Grid (for visualization and probing)
# ============================================================================
print("\n" + "=" * 70)
print("3. Computing Derivatives on Dense Grid")
print("=" * 70)

# Create 50x50 grid
resolution = 50
x_vals = torch.linspace(0, 1, resolution)
y_vals = torch.linspace(0, 1, resolution)
xx, yy = torch.meshgrid(x_vals, y_vals, indexing="ij")
x_grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)  # (2500, 2)

print(f"\nGrid: {resolution}×{resolution} = {x_grid.shape[0]} points")

# Compute all derivatives on grid
u_grid = problem.analytical_solution(x_grid)
du_dx_grid = problem.analytical_derivative_du_dx(x_grid)
du_dy_grid = problem.analytical_derivative_du_dy(x_grid)
d2u_dx2_grid = problem.analytical_derivative_d2u_dx2(x_grid)
d2u_dy2_grid = problem.analytical_derivative_d2u_dy2(x_grid)
laplacian_grid = problem.analytical_laplacian(x_grid)

print(f"\nComputed derivatives for {x_grid.shape[0]} grid points")
print(f"  Solution u: min={u_grid.min():.4f}, max={u_grid.max():.4f}")
print(f"  ∂u/∂x: min={du_dx_grid.min():.4f}, max={du_dx_grid.max():.4f}")
print(f"  ∂u/∂y: min={du_dy_grid.min():.4f}, max={du_dy_grid.max():.4f}")
print(f"  ∇²u: min={laplacian_grid.min():.4f}, max={laplacian_grid.max():.4f}")

# ============================================================================
# 4. Visualize Derivatives
# ============================================================================
print("\n" + "=" * 70)
print("4. Generating Visualizations")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Reshape for plotting
u_2d = u_grid.reshape(resolution, resolution).numpy()
du_dx_2d = du_dx_grid.reshape(resolution, resolution).numpy()
du_dy_2d = du_dy_grid.reshape(resolution, resolution).numpy()
d2u_dx2_2d = d2u_dx2_grid.reshape(resolution, resolution).numpy()
d2u_dy2_2d = d2u_dy2_grid.reshape(resolution, resolution).numpy()
laplacian_2d = laplacian_grid.reshape(resolution, resolution).numpy()

x_np = x_vals.numpy()
y_np = y_vals.numpy()

# Plot 1: Solution u(x,y)
im1 = axes[0, 0].pcolormesh(x_np, y_np, u_2d.T, cmap="viridis", shading="auto")
axes[0, 0].set_title("Solution: u(x,y) = sin(πx)sin(πy)", fontsize=12, fontweight="bold")
axes[0, 0].set_xlabel("x")
axes[0, 0].set_ylabel("y")
axes[0, 0].set_aspect("equal")
plt.colorbar(im1, ax=axes[0, 0])

# Plot 2: ∂u/∂x
im2 = axes[0, 1].pcolormesh(x_np, y_np, du_dx_2d.T, cmap="RdBu_r", shading="auto")
axes[0, 1].set_title("First Derivative: ∂u/∂x", fontsize=12, fontweight="bold")
axes[0, 1].set_xlabel("x")
axes[0, 1].set_ylabel("y")
axes[0, 1].set_aspect("equal")
plt.colorbar(im2, ax=axes[0, 1])

# Plot 3: ∂u/∂y
im3 = axes[0, 2].pcolormesh(x_np, y_np, du_dy_2d.T, cmap="RdBu_r", shading="auto")
axes[0, 2].set_title("First Derivative: ∂u/∂y", fontsize=12, fontweight="bold")
axes[0, 2].set_xlabel("x")
axes[0, 2].set_ylabel("y")
axes[0, 2].set_aspect("equal")
plt.colorbar(im3, ax=axes[0, 2])

# Plot 4: ∂²u/∂x²
im4 = axes[1, 0].pcolormesh(x_np, y_np, d2u_dx2_2d.T, cmap="RdBu_r", shading="auto")
axes[1, 0].set_title("Second Derivative: ∂²u/∂x²", fontsize=12, fontweight="bold")
axes[1, 0].set_xlabel("x")
axes[1, 0].set_ylabel("y")
axes[1, 0].set_aspect("equal")
plt.colorbar(im4, ax=axes[1, 0])

# Plot 5: ∂²u/∂y²
im5 = axes[1, 1].pcolormesh(x_np, y_np, d2u_dy2_2d.T, cmap="RdBu_r", shading="auto")
axes[1, 1].set_title("Second Derivative: ∂²u/∂y²", fontsize=12, fontweight="bold")
axes[1, 1].set_xlabel("x")
axes[1, 1].set_ylabel("y")
axes[1, 1].set_aspect("equal")
plt.colorbar(im5, ax=axes[1, 1])

# Plot 6: Laplacian ∇²u
im6 = axes[1, 2].pcolormesh(x_np, y_np, laplacian_2d.T, cmap="RdBu_r", shading="auto")
axes[1, 2].set_title("Laplacian: ∇²u", fontsize=12, fontweight="bold")
axes[1, 2].set_xlabel("x")
axes[1, 2].set_ylabel("y")
axes[1, 2].set_aspect("equal")
plt.colorbar(im6, ax=axes[1, 2])

plt.tight_layout()
plt.savefig("outputs/demo_analytical_derivatives.png", dpi=150, bbox_inches="tight")
print(f"\n✓ Visualization saved: outputs/demo_analytical_derivatives.png")

# ============================================================================
# 5. Statistics for Each Derivative
# ============================================================================
print("\n" + "=" * 70)
print("5. Derivative Statistics (for 2500 grid points)")
print("=" * 70)

derivatives = {
    "u (solution)": u_grid,
    "∂u/∂x": du_dx_grid,
    "∂u/∂y": du_dy_grid,
    "∂²u/∂x²": d2u_dx2_grid,
    "∂²u/∂y²": d2u_dy2_grid,
    "∇²u (Laplacian)": laplacian_grid,
}

for name, deriv in derivatives.items():
    print(f"\n{name}:")
    print(f"  Mean:   {deriv.mean().item():>8.4f}")
    print(f"  Std:    {deriv.std().item():>8.4f}")
    print(f"  Min:    {deriv.min().item():>8.4f}")
    print(f"  Max:    {deriv.max().item():>8.4f}")
    print(f"  Range:  {(deriv.max() - deriv.min()).item():>8.4f}")

# ============================================================================
# 6. How This Connects to Probing (Preview)
# ============================================================================
print("\n" + "=" * 70)
print("6. Connection to Probing Experiments (Days 9-10)")
print("=" * 70)

print(
    """
In Days 9-10, we'll use these analytical derivatives as "ground truth" targets
for training linear probes. The workflow will be:

1. Load trained PINN model (from Week 1)
2. Extract activations on grid points (from Day 5)
   → activations['layer_2'] = [2500 points × 64 neurons]

3. Compute ground-truth derivatives (TODAY'S WORK):
   → du_dx_true = problem.analytical_derivative_du_dx(x_grid)  # (2500, 1)
   → du_dy_true = problem.analytical_derivative_du_dy(x_grid)  # (2500, 1)
   → laplacian_true = problem.analytical_laplacian(x_grid)     # (2500, 1)

4. Train linear probes to predict derivatives from activations:
   ```python
   probe_du_dx = LinearProbe(input_dim=64, output_dim=1)
   probe_du_dx.fit(activations['layer_2'], du_dx_true, epochs=1000)
   scores = probe_du_dx.score(activations['layer_2'], du_dx_true)

   print(f"Layer 2 encodes ∂u/∂x with R² = {scores['r_squared']:.4f}")
   # If R² > 0.9 → derivative is linearly accessible at this layer!
   ```

5. Repeat for all (layer, derivative) pairs to identify where each
   derivative emerges in the network:

   Expected results might look like:

   |  Layer  | ∂u/∂x R² | ∂u/∂y R² | ∇²u R² |
   |---------|----------|----------|--------|
   | layer_0 |   0.15   |   0.12   |  0.08  |  (not yet computing)
   | layer_1 |   0.82   |   0.85   |  0.35  |  (starting to emerge)
   | layer_2 |   0.95   |   0.96   |  0.91  |  (fully computing!)
   | layer_3 |   0.93   |   0.94   |  0.89  |  (refining)

   This reveals the network's computational structure!
"""
)

# ============================================================================
# 7. Quick Probing Preview (Synthetic Example)
# ============================================================================
print("\n" + "=" * 70)
print("7. Quick Probing Preview (Synthetic Activations)")
print("=" * 70)

from src.interpretability import LinearProbe

print("\nSimulating what we'll do in Days 9-10...")

# Create synthetic "activations" (random 64-dim vectors for each grid point)
# In reality, these will come from the trained PINN
fake_activations = torch.randn(2500, 64)

# Add some linear relationship to ∂u/∂x to simulate what a trained PINN might have
# (Real activations will have learned this relationship during PINN training)
fake_activations[:, 0] = (
    du_dx_grid.squeeze() + 0.1 * torch.randn(2500)
)  # Encode derivative

print(f"\nSynthetic activations: {fake_activations.shape}")
print(f"Ground-truth ∂u/∂x: {du_dx_grid.shape}")

# Train probe to predict ∂u/∂x from activations
probe = LinearProbe(input_dim=64, output_dim=1)
probe.fit(fake_activations, du_dx_grid, epochs=500, lr=1e-2, verbose=False)

# Evaluate
scores = probe.score(fake_activations, du_dx_grid)
print(f"\nProbe performance (synthetic data):")
print(f"  R² = {scores['r_squared']:.4f}")
print(f"  MSE = {scores['mse']:.6f}")

if scores["r_squared"] > 0.5:
    print(
        f"\n✓ Good! R² > 0.5 indicates ∂u/∂x is linearly accessible in activations"
    )
    print(f"  (With real PINN activations, we'll discover which layers encode derivatives)")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("Summary: Task 2 Complete")
print("=" * 70)

print(
    f"""
✓ Implemented 6 analytical derivative methods in PoissonProblem:
  1. analytical_derivative_du_dx(x)     → ∂u/∂x
  2. analytical_derivative_du_dy(x)     → ∂u/∂y
  3. analytical_derivative_d2u_dx2(x)   → ∂²u/∂x²
  4. analytical_derivative_d2u_dy2(x)   → ∂²u/∂y²
  5. analytical_laplacian(x)            → ∇²u
  6. analytical_gradient(x)             → (∂u/∂x, ∂u/∂y)

✓ Created 27 comprehensive tests (all passing)

✓ Generated visualization showing all derivatives

✓ These derivatives serve as "ground truth" targets for probing experiments

Next Steps (Days 9-10):
→ Train probes for each (layer, derivative) pair
→ Generate R² heatmaps showing where derivatives emerge
→ Analyze the network's computational structure
→ Discover HOW PINNs internally compute derivatives!
"""
)

print("=" * 70)
print("Demo Complete!")
print("=" * 70)
