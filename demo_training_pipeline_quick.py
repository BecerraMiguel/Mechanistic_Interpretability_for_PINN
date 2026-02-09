"""
Quick Demo: Training Pipeline Verification

This is a quick version of the full training pipeline demo that runs
fewer epochs to verify that all components work correctly:
- Training loop
- Early stopping
- Visualization
- Checkpointing

For full training to achieve <1% error, use demo_training_pipeline.py
"""

import os
import torch
import torch.optim as optim

from src.models import MLP
from src.problems import PoissonProblem
from src.training import PINNTrainer

SAVE_DIR = "outputs/demo_quick_test"
os.makedirs(SAVE_DIR, exist_ok=True)


def main():
    print("=" * 80)
    print("Quick Demo: Training Pipeline Verification")
    print("=" * 80)
    print()

    # Create model and problem
    print("Creating model and problem...")
    model = MLP(
        input_dim=2,
        hidden_dims=[32, 32],
        output_dim=1,
        activation="tanh",
    )
    problem = PoissonProblem()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Create trainer
    print("Creating trainer...")
    trainer = PINNTrainer(
        model=model,
        problem=problem,
        optimizer=optimizer,
        n_interior=1000,
        n_boundary=50,
        device="cpu",
    )

    # Train with early stopping
    print("\nTraining with early stopping...")
    history = trainer.train(
        n_epochs=500,
        validate_every=50,
        print_every=100,
        save_every=200,
        save_path=os.path.join(SAVE_DIR, "checkpoints"),
        early_stopping=True,
        patience=5,
        min_delta=0.01,
    )

    # Results
    print("\n" + "=" * 80)
    print("Results:")
    print("=" * 80)
    print(f"Epochs run: {len(history['loss_total'])}")
    print(f"Initial loss: {history['loss_total'][0]:.6f}")
    print(f"Final loss: {history['loss_total'][-1]:.6f}")
    print(f"Final relative L2 error: {history['relative_l2_error'][-1]:.4f}%")
    print()

    # Generate visualizations
    print("Generating visualizations...")
    heatmap_path = os.path.join(SAVE_DIR, "solution_heatmap.png")
    trainer.generate_solution_heatmap(heatmap_path, n_points=50)

    print("\n✅ Pipeline verification complete!")
    print(f"Outputs saved to: {SAVE_DIR}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
