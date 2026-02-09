"""
Demo script for training a PINN on the Poisson equation.

This script demonstrates:
1. Setting up a PINN model and Poisson problem
2. Configuring the training loop
3. Training with loss decomposition
4. Monitoring convergence
5. Evaluating the trained model
6. Visualizing results
"""

import matplotlib.pyplot as plt
import torch

from src.models import MLP
from src.problems import PoissonProblem
from src.training import train_pinn

print("=" * 80)
print("Training PINN on Poisson Equation")
print("=" * 80)
print()

# 1. Setup
print("1. Setting up model and problem")
print("-" * 80)

# Create model
model = MLP(
    input_dim=2,
    hidden_dims=[64, 64, 64, 64],  # 4 hidden layers with 64 neurons each
    output_dim=1,
    activation="tanh",  # Standard activation for PINNs
)

print(f"Model: {model}")
print(f"Total parameters: {model.get_parameters_count():,}")
print()

# Create problem
problem = PoissonProblem()
print(f"Problem: {problem}")
print()

# 2. Training Configuration
print("2. Training configuration")
print("-" * 80)

config = {
    # Optimizer settings
    "optimizer": "adam",
    "lr": 1e-3,

    # Training settings
    "n_epochs": 5000,
    "n_interior": 10000,  # Interior collocation points
    "n_boundary": 100,    # Points per edge (400 total)

    # Loss weights
    "loss_weights": {
        "pde": 1.0,   # Weight for PDE residual loss
        "bc": 1.0,    # Weight for boundary condition loss
        "ic": 0.0,    # No initial conditions for Poisson
    },

    # Device
    "device": "cpu",  # Change to "cuda" if GPU available

    # Logging settings
    "resample_every": 1,      # Resample collocation points every epoch
    "validate_every": 500,    # Compute validation error every 500 epochs
    "print_every": 500,       # Print progress every 500 epochs
    "save_every": 5000,       # Save checkpoint every 5000 epochs
    "save_path": "outputs/checkpoints",
}

print(f"Optimizer: {config['optimizer']} (lr={config['lr']})")
print(f"Training epochs: {config['n_epochs']:,}")
print(f"Interior points: {config['n_interior']:,}")
print(f"Boundary points: {config['n_boundary'] * 4:,} ({config['n_boundary']} per edge)")
print(f"Loss weights: PDE={config['loss_weights']['pde']}, BC={config['loss_weights']['bc']}")
print(f"Device: {config['device']}")
print()

# 3. Training
print("3. Training")
print("-" * 80)
print()

trained_model, history = train_pinn(model, problem, config)

print()

# 4. Analyze Results
print("4. Analyzing training results")
print("-" * 80)

# Loss statistics
initial_loss = history["loss_total"][0]
final_loss = history["loss_total"][-1]
loss_reduction = (1 - final_loss / initial_loss) * 100

print(f"Initial loss: {initial_loss:.6f}")
print(f"Final loss: {final_loss:.6f}")
print(f"Loss reduction: {loss_reduction:.2f}%")
print()

# Error statistics
final_error = history["relative_l2_error"][-1]
print(f"Final relative L2 error: {final_error:.4f}%")

if final_error < 1.0:
    print("✓ Success! Achieved < 1% error (Day 4 target)")
elif final_error < 5.0:
    print("✓ Good! Achieved < 5% error")
else:
    print("⚠ Error still high, may need more training")
print()

# 5. Evaluate on Test Grid
print("5. Evaluating on test grid")
print("-" * 80)

# Create dense test grid
n_test = 100
x_test = torch.linspace(0, 1, n_test)
y_test = torch.linspace(0, 1, n_test)
X, Y = torch.meshgrid(x_test, y_test, indexing='ij')
xy_test = torch.stack([X.flatten(), Y.flatten()], dim=1)

# Compute predictions
trained_model.eval()
with torch.no_grad():
    u_pred = trained_model(xy_test).reshape(n_test, n_test)

# Compute analytical solution
u_exact = problem.analytical_solution(xy_test).reshape(n_test, n_test)

# Compute pointwise error
error_map = torch.abs(u_pred - u_exact)

print(f"Test grid: {n_test} × {n_test} = {n_test**2:,} points")
print(f"Max prediction: {u_pred.max():.4f}")
print(f"Min prediction: {u_pred.min():.4f}")
print(f"Max pointwise error: {error_map.max():.6f}")
print(f"Mean pointwise error: {error_map.mean():.6f}")
print()

# 6. Visualization
print("6. Creating visualizations")
print("-" * 80)

fig = plt.figure(figsize=(16, 12))

# Plot 1: Training curves
ax1 = plt.subplot(3, 3, 1)
epochs = range(1, len(history["loss_total"]) + 1)
ax1.semilogy(epochs, history["loss_total"], 'b-', linewidth=2, label='Total Loss')
ax1.semilogy(epochs, history["loss_pde"], 'r--', linewidth=1.5, label='PDE Loss')
ax1.semilogy(epochs, history["loss_bc"], 'g--', linewidth=1.5, label='BC Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss (log scale)')
ax1.set_title('Training Loss Curves')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Relative L2 error
ax2 = plt.subplot(3, 3, 2)
val_epochs = [i * config["validate_every"] for i in range(1, len(history["relative_l2_error"]))]
val_epochs.append(config["n_epochs"])  # Add final epoch
ax2.plot(val_epochs, history["relative_l2_error"], 'ko-', linewidth=2, markersize=6)
ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='1% target')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Relative L2 Error (%)')
ax2.set_title('Validation Error')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Loss components ratio
ax3 = plt.subplot(3, 3, 3)
loss_ratio = [pde / (bc + 1e-10) for pde, bc in zip(history["loss_pde"], history["loss_bc"])]
ax3.plot(epochs, loss_ratio, 'purple', linewidth=2)
ax3.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
ax3.set_xlabel('Epoch')
ax3.set_ylabel('PDE Loss / BC Loss')
ax3.set_title('Loss Components Ratio')
ax3.grid(True, alpha=0.3)

# Plot 4: Analytical solution
ax4 = plt.subplot(3, 3, 4)
im4 = ax4.contourf(X.numpy(), Y.numpy(), u_exact.numpy(), levels=20, cmap='viridis')
ax4.set_xlabel('x')
ax4.set_ylabel('y')
ax4.set_title('Analytical Solution: u(x,y) = sin(πx)sin(πy)')
ax4.set_aspect('equal')
plt.colorbar(im4, ax=ax4)

# Plot 5: PINN prediction
ax5 = plt.subplot(3, 3, 5)
im5 = ax5.contourf(X.numpy(), Y.numpy(), u_pred.numpy(), levels=20, cmap='viridis')
ax5.set_xlabel('x')
ax5.set_ylabel('y')
ax5.set_title('PINN Prediction')
ax5.set_aspect('equal')
plt.colorbar(im5, ax=ax5)

# Plot 6: Pointwise error
ax6 = plt.subplot(3, 3, 6)
im6 = ax6.contourf(X.numpy(), Y.numpy(), error_map.numpy(), levels=20, cmap='Reds')
ax6.set_xlabel('x')
ax6.set_ylabel('y')
ax6.set_title(f'Pointwise Error (max: {error_map.max():.6f})')
ax6.set_aspect('equal')
plt.colorbar(im6, ax=ax6)

# Plot 7: 1D slice comparison (y = 0.5)
ax7 = plt.subplot(3, 3, 7)
y_slice_idx = n_test // 2
x_slice = x_test.numpy()
u_exact_slice = u_exact[y_slice_idx, :].numpy()
u_pred_slice = u_pred[y_slice_idx, :].numpy()
ax7.plot(x_slice, u_exact_slice, 'b-', linewidth=2, label='Analytical')
ax7.plot(x_slice, u_pred_slice, 'r--', linewidth=2, label='PINN')
ax7.set_xlabel('x')
ax7.set_ylabel('u(x, 0.5)')
ax7.set_title('1D Slice at y = 0.5')
ax7.legend()
ax7.grid(True, alpha=0.3)

# Plot 8: 1D slice comparison (x = 0.5)
ax8 = plt.subplot(3, 3, 8)
x_slice_idx = n_test // 2
y_slice = y_test.numpy()
u_exact_slice = u_exact[:, x_slice_idx].numpy()
u_pred_slice = u_pred[:, x_slice_idx].numpy()
ax8.plot(y_slice, u_exact_slice, 'b-', linewidth=2, label='Analytical')
ax8.plot(y_slice, u_pred_slice, 'r--', linewidth=2, label='PINN')
ax8.set_xlabel('y')
ax8.set_ylabel('u(0.5, y)')
ax8.set_title('1D Slice at x = 0.5')
ax8.legend()
ax8.grid(True, alpha=0.3)

# Plot 9: Error histogram
ax9 = plt.subplot(3, 3, 9)
ax9.hist(error_map.flatten().numpy(), bins=50, color='red', alpha=0.7, edgecolor='black')
ax9.set_xlabel('Pointwise Error')
ax9.set_ylabel('Frequency')
ax9.set_title('Error Distribution')
ax9.axvline(x=error_map.mean().item(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {error_map.mean():.6f}')
ax9.legend()
ax9.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/poisson_training_results.png', dpi=150, bbox_inches='tight')
print("Visualization saved to: outputs/poisson_training_results.png")
print()

# 7. Summary
print("=" * 80)
print("Training Summary")
print("=" * 80)
print(f"Problem: Poisson equation ∇²u = f on [0,1]²")
print(f"Model: MLP with {len(model.hidden_dims)} hidden layers")
print(f"Total parameters: {model.get_parameters_count():,}")
print(f"Training epochs: {config['n_epochs']:,}")
print(f"Collocation points: {config['n_interior']:,} interior + {config['n_boundary']*4} boundary")
print()
print(f"Results:")
print(f"  • Final total loss: {final_loss:.6f}")
print(f"  • Final PDE loss: {history['loss_pde'][-1]:.6f}")
print(f"  • Final BC loss: {history['loss_bc'][-1]:.6f}")
print(f"  • Relative L2 error: {final_error:.4f}%")
print(f"  • Max pointwise error: {error_map.max():.6f}")
print(f"  • Mean pointwise error: {error_map.mean():.6f}")
print()

if final_error < 1.0:
    print("✅ SUCCESS: Achieved < 1% relative L2 error (Day 4 target reached!)")
elif final_error < 5.0:
    print("✅ GOOD: Achieved < 5% relative L2 error")
else:
    print("⚠️  WARNING: Error > 5%, consider:")
    print("    - Training for more epochs")
    print("    - Using more collocation points")
    print("    - Adjusting learning rate")
    print("    - Trying different architecture")

print("=" * 80)
