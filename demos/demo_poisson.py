"""
Demo script showcasing the Poisson equation problem class.

This script demonstrates:
1. Creating a Poisson problem instance
2. Sampling interior and boundary collocation points
3. Computing analytical solution and source term
4. Verifying that the analytical solution satisfies the PDE
5. Computing relative L2 error for a dummy model
"""

import math

import matplotlib.pyplot as plt
import torch

from src.problems import PoissonProblem
from src.utils.derivatives import compute_derivatives

print("=" * 80)
print("Poisson Equation Problem Demo")
print("=" * 80)
print()

# 1. Create Poisson problem
print("1. Creating Poisson problem on unit square [0,1]²")
problem = PoissonProblem()
print(problem)
print()

# 2. Sample collocation points
print("2. Sampling collocation points")
n_interior = 1000
n_per_edge = 25

x_interior = problem.sample_interior_points(n=n_interior, random_seed=42)
x_boundary = problem.sample_boundary_points(n_per_edge=n_per_edge, random_seed=42)

print(f"   Interior points: {x_interior.shape[0]} points using Latin Hypercube Sampling")
print(f"   Boundary points: {x_boundary.shape[0]} points (4 edges × {n_per_edge} points)")
print()

# 3. Compute analytical solution and source term
print("3. Computing analytical solution u(x,y) = sin(πx)sin(πy)")
u_interior = problem.analytical_solution(x_interior)
u_boundary = problem.analytical_solution(x_boundary)
f_interior = problem.source_term(x_interior)

print(f"   Interior solution range: [{u_interior.min():.4f}, {u_interior.max():.4f}]")
print(f"   Boundary solution (should be ~0): [{u_boundary.min():.6f}, {u_boundary.max():.6f}]")
print(f"   Source term range: [{f_interior.min():.4f}, {f_interior.max():.4f}]")
print()

# 4. Verify PDE is satisfied
print("4. Verifying that analytical solution satisfies PDE: ∇²u = f")
x_test = torch.randn(100, 2, requires_grad=True)
u_test = problem.analytical_solution(x_test)

# Compute derivatives
du_dx = compute_derivatives(u_test, x_test, order=1)
d2u_dx2 = compute_derivatives(u_test, x_test, order=2)

# Compute PDE residual
residual = problem.pde_residual(u_test, x_test, du_dx, d2u_dx2)
max_residual = torch.abs(residual).max().item()

print(f"   Maximum PDE residual: {max_residual:.2e} (should be ~0)")
print(f"   ✓ PDE satisfied!" if max_residual < 1e-4 else "   ✗ PDE NOT satisfied!")
print()

# 5. Test relative L2 error computation
print("5. Testing relative L2 error computation")


# Create a perfect model (returns analytical solution)
class PerfectModel(torch.nn.Module):
    def __init__(self, problem):
        super().__init__()
        self.problem = problem

    def forward(self, x):
        return self.problem.analytical_solution(x)


# Create a bad model (returns zeros)
class BadModel(torch.nn.Module):
    def forward(self, x):
        return torch.zeros(x.shape[0], 1)


perfect_model = PerfectModel(problem)
bad_model = BadModel()

error_perfect = problem.compute_relative_l2_error(perfect_model, n_test_points=5000, random_seed=42)
error_bad = problem.compute_relative_l2_error(bad_model, n_test_points=5000, random_seed=42)

print(f"   Perfect model error: {error_perfect:.4f}% (should be ~0%)")
print(f"   Bad model error: {error_bad:.2f}% (should be large)")
print()

# 6. Visualization
print("6. Creating visualization")

# Create a dense grid for plotting
n_plot = 100
x_plot = torch.linspace(0, 1, n_plot)
y_plot = torch.linspace(0, 1, n_plot)
X, Y = torch.meshgrid(x_plot, y_plot, indexing='ij')
xy = torch.stack([X.flatten(), Y.flatten()], dim=1)

# Compute solution on grid
u_plot = problem.analytical_solution(xy)
u_plot = u_plot.reshape(n_plot, n_plot)

# Compute source term on grid
f_plot = problem.source_term(xy)
f_plot = f_plot.reshape(n_plot, n_plot)

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot analytical solution
im1 = axes[0, 0].contourf(X.numpy(), Y.numpy(), u_plot.numpy(), levels=20, cmap='viridis')
axes[0, 0].scatter(x_interior[:, 0], x_interior[:, 1], c='red', s=1, alpha=0.3, label='Interior points')
axes[0, 0].scatter(x_boundary[:, 0], x_boundary[:, 1], c='white', s=10, marker='x', label='Boundary points')
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('y')
axes[0, 0].set_title('Analytical Solution: u(x,y) = sin(πx)sin(πy)')
axes[0, 0].legend(loc='upper right')
plt.colorbar(im1, ax=axes[0, 0])

# Plot source term
im2 = axes[0, 1].contourf(X.numpy(), Y.numpy(), f_plot.numpy(), levels=20, cmap='RdBu_r')
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('y')
axes[0, 1].set_title('Source Term: f(x,y) = -2π²sin(πx)sin(πy)')
plt.colorbar(im2, ax=axes[0, 1])

# Plot collocation points distribution
axes[1, 0].scatter(x_interior[:, 0], x_interior[:, 1], c='blue', s=5, alpha=0.5, label='Interior (LHS)')
axes[1, 0].scatter(x_boundary[:, 0], x_boundary[:, 1], c='red', s=20, marker='x', label='Boundary')
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('y')
axes[1, 0].set_title(f'Collocation Points Distribution (n={n_interior+x_boundary.shape[0]})')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 1D slice through center
y_center = 0.5
x_slice = torch.linspace(0, 1, 200)
xy_slice = torch.stack([x_slice, torch.full_like(x_slice, y_center)], dim=1)
u_slice = problem.analytical_solution(xy_slice)

axes[1, 1].plot(x_slice.numpy(), u_slice.numpy(), 'b-', linewidth=2, label='u(x, 0.5)')
axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[1, 1].set_xlabel('x')
axes[1, 1].set_ylabel('u')
axes[1, 1].set_title('1D Slice at y = 0.5')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/poisson_demo.png', dpi=150, bbox_inches='tight')
print("   Visualization saved to: outputs/poisson_demo.png")
print()

print("=" * 80)
print("Demo completed successfully!")
print("=" * 80)
print()
print("Summary:")
print(f"  • Problem: Poisson equation ∇²u = f on [0,1]²")
print(f"  • Analytical solution: u(x,y) = sin(πx)sin(πy)")
print(f"  • Interior points: {n_interior} (Latin Hypercube Sampling)")
print(f"  • Boundary points: {x_boundary.shape[0]} (uniform on edges)")
print(f"  • PDE residual: {max_residual:.2e} ✓")
print(f"  • Relative L2 error test: {error_perfect:.4f}% (perfect), {error_bad:.2f}% (bad)")
print()
