"""
Demo script for collocation point sampling strategies.

This script demonstrates and compares:
1. Latin Hypercube Sampling (LHS)
2. Uniform Random Sampling
3. Grid Sampling
4. Boundary Sampling (1D, 2D, 3D)
"""

import matplotlib.pyplot as plt
import torch

from src.utils.sampling import (
    LatinHypercubeSampler,
    UniformRandomSampler,
    GridSampler,
    BoundarySampler,
    sample_collocation_points,
)

print("=" * 80)
print("Collocation Point Sampling Strategies Demo")
print("=" * 80)
print()

# Domain
domain = ((0.0, 1.0), (0.0, 1.0))
n_points = 500

# 1. Latin Hypercube Sampling
print("1. Latin Hypercube Sampling (LHS)")
print("-" * 80)
lhs_sampler = LatinHypercubeSampler()
lhs_points = lhs_sampler.sample(n=n_points, domain=domain, random_seed=42)
print(f"Sampled {lhs_points.shape[0]} points using LHS")
print(f"Shape: {lhs_points.shape}")
print(f"Range: x ∈ [{lhs_points[:, 0].min():.4f}, {lhs_points[:, 0].max():.4f}]")
print(f"       y ∈ [{lhs_points[:, 1].min():.4f}, {lhs_points[:, 1].max():.4f}]")
print()

# 2. Uniform Random Sampling
print("2. Uniform Random Sampling")
print("-" * 80)
uniform_sampler = UniformRandomSampler()
uniform_points = uniform_sampler.sample(n=n_points, domain=domain, random_seed=42)
print(f"Sampled {uniform_points.shape[0]} points uniformly")
print(f"Shape: {uniform_points.shape}")
print()

# 3. Grid Sampling
print("3. Grid Sampling")
print("-" * 80)
grid_sampler = GridSampler()
grid_points = grid_sampler.sample(n=n_points, domain=domain)
print(f"Created grid with {grid_points.shape[0]} points")
print(f"Shape: {grid_points.shape}")
print()

# 4. Boundary Sampling
print("4. Boundary Sampling (2D)")
print("-" * 80)
boundary_sampler = BoundarySampler()
boundary_points = boundary_sampler.sample(n_per_edge=50, domain=domain, random_seed=42)
print(f"Sampled {boundary_points.shape[0]} boundary points")
print(f"Shape: {boundary_points.shape}")
print(f"Points per edge: 50 × 4 edges = 200 total")
print()

# 5. Convenience Function
print("5. Convenience Function")
print("-" * 80)
x_int, x_bound = sample_collocation_points(
    n_interior=1000,
    n_boundary=100,
    domain=domain,
    interior_sampler="lhs",
    random_seed=42,
)
print(f"Interior points: {x_int.shape}")
print(f"Boundary points: {x_bound.shape}")
print()

# Visualization
print("6. Creating comparison visualization")
print("-" * 80)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Latin Hypercube Sampling
ax1 = axes[0, 0]
ax1.scatter(lhs_points[:, 0], lhs_points[:, 1], s=10, alpha=0.6, c='blue')
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Latin Hypercube Sampling\n(Better Coverage)')
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)

# Plot 2: Uniform Random Sampling
ax2 = axes[0, 1]
ax2.scatter(uniform_points[:, 0], uniform_points[:, 1], s=10, alpha=0.6, c='red')
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Uniform Random Sampling\n(May Have Clusters/Gaps)')
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)

# Plot 3: Grid Sampling
ax3 = axes[0, 2]
ax3.scatter(grid_points[:, 0], grid_points[:, 1], s=10, alpha=0.6, c='green')
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_title('Grid Sampling\n(Deterministic, Regular)')
ax3.set_aspect('equal')
ax3.grid(True, alpha=0.3)

# Plot 4: Boundary Sampling
ax4 = axes[1, 0]
ax4.scatter(boundary_points[:, 0], boundary_points[:, 1], s=20, alpha=0.8, c='orange', marker='x')
ax4.set_xlim(-0.1, 1.1)
ax4.set_ylim(-0.1, 1.1)
ax4.set_xlabel('x')
ax4.set_ylabel('y')
ax4.set_title('Boundary Sampling\n(Edges Only)')
ax4.set_aspect('equal')
ax4.grid(True, alpha=0.3)

# Plot 5: Interior + Boundary (Full Collocation)
ax5 = axes[1, 1]
ax5.scatter(x_int[:, 0], x_int[:, 1], s=3, alpha=0.4, c='blue', label='Interior (LHS)')
ax5.scatter(x_bound[:, 0], x_bound[:, 1], s=15, alpha=0.8, c='red', marker='x', label='Boundary')
ax5.set_xlim(-0.1, 1.1)
ax5.set_ylim(-0.1, 1.1)
ax5.set_xlabel('x')
ax5.set_ylabel('y')
ax5.set_title('Full Collocation Setup\n(Interior + Boundary)')
ax5.set_aspect('equal')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Coverage Comparison (Histogram)
ax6 = axes[1, 2]

# Divide domain into 10×10 bins and count points
import numpy as np
n_bins = 10
lhs_hist, _, _ = np.histogram2d(lhs_points[:, 0].numpy(), lhs_points[:, 1].numpy(), bins=n_bins, range=[[0, 1], [0, 1]])
uniform_hist, _, _ = np.histogram2d(uniform_points[:, 0].numpy(), uniform_points[:, 1].numpy(), bins=n_bins, range=[[0, 1], [0, 1]])

lhs_counts = lhs_hist.flatten()
uniform_counts = uniform_hist.flatten()

bins = range(len(lhs_counts))
ax6.bar(bins, lhs_counts, alpha=0.7, label='LHS', color='blue')
ax6.bar(bins, uniform_counts, alpha=0.5, label='Uniform', color='red')
ax6.set_xlabel('Bin Index')
ax6.set_ylabel('Points per Bin')
ax6.set_title('Coverage Comparison\n(More Even = Better)')
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('outputs/sampling_comparison.png', dpi=150, bbox_inches='tight')
print("Visualization saved to: outputs/sampling_comparison.png")
print()

# Statistics
print("7. Coverage Statistics")
print("-" * 80)
lhs_std = lhs_counts.std()
uniform_std = uniform_counts.std()
print(f"LHS bin count std: {lhs_std:.2f} (lower = more even)")
print(f"Uniform bin count std: {uniform_std:.2f}")
print(f"Improvement: {((uniform_std - lhs_std) / uniform_std * 100):.1f}%")
print()

print("=" * 80)
print("Demo Complete!")
print("=" * 80)
print()
print("Summary:")
print("  • Latin Hypercube: Best for training (even coverage)")
print("  • Uniform Random: Simple but may have gaps")
print("  • Grid: Good for visualization/testing (deterministic)")
print("  • Boundary: Essential for enforcing BCs")
print()
print("Recommendation: Use LHS for interior + uniform boundary for training")
print("=" * 80)
