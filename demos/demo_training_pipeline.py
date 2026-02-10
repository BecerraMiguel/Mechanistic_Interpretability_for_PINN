"""
Demo: Full Training Pipeline with W&B, Early Stopping, and Visualization

This script demonstrates the complete training pipeline for PINNs:
1. Full training loop with loss decomposition
2. W&B integration for experiment tracking (optional)
3. Early stopping based on validation error
4. Checkpoint saving
5. Solution heatmap visualization

The demo trains a PINN to solve the 2D Poisson equation and achieves
convergence with relative L2 error tracking.
"""

import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from src.models import MLP
from src.problems import PoissonProblem
from src.training import PINNTrainer

# Configuration
USE_WANDB = False  # Set to True to enable W&B logging (requires API key)
SAVE_DIR = "outputs/demo_training_pipeline"
os.makedirs(SAVE_DIR, exist_ok=True)


def plot_training_history(history, save_path):
    """Plot training history with loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=150)

    # Plot 1: Loss curves
    axes[0].plot(history["loss_total"], label="Total Loss", linewidth=2)
    axes[0].plot(history["loss_pde"], label="PDE Loss", alpha=0.7)
    axes[0].plot(history["loss_bc"], label="BC Loss", alpha=0.7)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_yscale("log")
    axes[0].set_title("Training Loss Curves")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Relative L2 error
    if len(history["relative_l2_error"]) > 0:
        validation_epochs = [
            i * 100 for i in range(len(history["relative_l2_error"]))
        ]
        axes[1].plot(
            validation_epochs,
            history["relative_l2_error"],
            marker="o",
            linewidth=2,
            markersize=4,
        )
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Relative L2 Error (%)")
        axes[1].set_title("Validation Error")
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=1.0, color="r", linestyle="--", alpha=0.5, label="1% target")
        axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training history plot saved to: {save_path}")


def main():
    print("=" * 80)
    print("Demo: Full Training Pipeline with Early Stopping and Visualization")
    print("=" * 80)
    print()

    # ===== 1. Create Model and Problem =====
    print("1. Creating model and problem...")
    model = MLP(
        input_dim=2,
        hidden_dims=[64, 64, 64, 64],
        output_dim=1,
        activation="tanh",
    )
    problem = PoissonProblem()

    print(f"   Model: {model.__class__.__name__}")
    print(f"   Architecture: {model.get_layer_dimensions()}")
    print(f"   Total parameters: {model.get_parameters_count():,}")
    print(f"   Problem: {problem.__class__.__name__} (2D Poisson equation)")
    print()

    # ===== 2. Create Optimizer =====
    print("2. Creating optimizer...")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(f"   Optimizer: Adam (lr=1e-3)")
    print()

    # ===== 3. Create Trainer =====
    print("3. Creating trainer with all features...")
    trainer = PINNTrainer(
        model=model,
        problem=problem,
        optimizer=optimizer,
        n_interior=10000,  # 10k interior points
        n_boundary=100,  # 100 points per edge
        loss_weights={"pde": 1.0, "bc": 1.0, "ic": 1.0},
        device="cpu",
        use_wandb=USE_WANDB,
        wandb_project="pinn-demo",
        wandb_config={
            "architecture": "MLP",
            "hidden_layers": 4,
            "hidden_dim": 64,
            "activation": "tanh",
            "optimizer": "Adam",
            "learning_rate": 1e-3,
        },
    )
    print(f"   Interior points: 10,000")
    print(f"   Boundary points: 400 (100 per edge)")
    print(f"   W&B logging: {'Enabled' if USE_WANDB else 'Disabled'}")
    print()

    # ===== 4. Train with Early Stopping =====
    print("4. Training with early stopping...")
    print("   Configuration:")
    print("   - Max epochs: 20,000")
    print("   - Early stopping: Enabled (patience=50, min_delta=0.001)")
    print("   - Validation every 100 epochs")
    print("   - Checkpoints every 2,000 epochs")
    print()

    history = trainer.train(
        n_epochs=20000,
        resample_every=1,
        validate_every=100,
        print_every=500,
        save_every=2000,
        save_path=os.path.join(SAVE_DIR, "checkpoints"),
        early_stopping=True,
        patience=50,
        min_delta=0.001,
    )

    # ===== 5. Results Summary =====
    print()
    print("=" * 80)
    print("5. Training Results Summary")
    print("=" * 80)
    print(f"Total epochs run: {len(history['loss_total']):,}")
    print(f"Initial loss: {history['loss_total'][0]:.6f}")
    print(f"Final loss: {history['loss_total'][-1]:.6f}")
    print(f"Loss reduction: {(1 - history['loss_total'][-1]/history['loss_total'][0])*100:.2f}%")
    print()
    print(f"Final relative L2 error: {history['relative_l2_error'][-1]:.4f}%")
    if history['relative_l2_error'][-1] < 1.0:
        print("✅ SUCCESS: Achieved target error < 1%")
    else:
        print(f"⚠️  Target not reached. Current: {history['relative_l2_error'][-1]:.4f}%")
    print()
    if trainer.best_val_error < float('inf'):
        print(f"Best validation error: {trainer.best_val_error:.4f}%")
        print(f"Early stopping triggered: Yes (patience counter: {trainer.patience_counter})")
    print("=" * 80)
    print()

    # ===== 6. Generate Visualizations =====
    print("6. Generating visualizations...")

    # Training history
    history_path = os.path.join(SAVE_DIR, "training_history.png")
    plot_training_history(history, history_path)

    # Solution heatmap
    heatmap_path = os.path.join(SAVE_DIR, "solution_heatmap.png")
    trainer.generate_solution_heatmap(heatmap_path, n_points=100)

    print()
    print("=" * 80)
    print("Demo Complete!")
    print("=" * 80)
    print(f"All outputs saved to: {SAVE_DIR}/")
    print("Files created:")
    print(f"  - training_history.png: Loss curves and validation error")
    print(f"  - solution_heatmap.png: PINN solution vs analytical solution")
    print(f"  - checkpoints/: Model checkpoints saved every 2,000 epochs")
    if USE_WANDB:
        print(f"  - W&B dashboard: View at wandb.ai")
    print("=" * 80)


if __name__ == "__main__":
    main()
