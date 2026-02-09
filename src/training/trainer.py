"""
Training loop for Physics-Informed Neural Networks (PINNs).

This module provides a flexible trainer for PINNs with:
- Loss decomposition (PDE + BC + IC)
- Configurable loss weights
- Metric logging (total loss, component losses, relative L2 error)
- Optional W&B integration
- Checkpointing
- Validation during training
- Early stopping based on validation error
- Solution visualization (heatmaps)
"""

from typing import Dict, Optional, Callable, Tuple
import time
import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from src.utils.derivatives import compute_derivatives


class PINNTrainer:
    """
    Trainer for Physics-Informed Neural Networks.

    Handles the full training loop with:
    - Automatic differentiation for PDE residuals
    - Loss decomposition (L_pde, L_bc, L_ic)
    - Configurable loss weights
    - Metric logging and validation
    - Model checkpointing

    Parameters
    ----------
    model : nn.Module
        PINN model (must inherit from BasePINN).
    problem : BaseProblem
        PDE problem definition.
    optimizer : torch.optim.Optimizer
        Optimizer for training.
    n_interior : int
        Number of interior collocation points.
    n_boundary : int
        Number of boundary points per edge.
    n_initial : int, optional
        Number of initial condition points (for time-dependent PDEs). Default: 0.
    loss_weights : Dict[str, float], optional
        Weights for loss components: {'pde': w_pde, 'bc': w_bc, 'ic': w_ic}.
        Default: {'pde': 1.0, 'bc': 1.0, 'ic': 1.0}.
    device : str, optional
        Device for training ('cpu' or 'cuda'). Default: 'cpu'.
    use_wandb : bool, optional
        Whether to log to Weights & Biases. Default: False.
    wandb_project : str, optional
        W&B project name. Default: None.
    wandb_config : Dict, optional
        Additional config to log to W&B. Default: None.
    """

    def __init__(
        self,
        model: nn.Module,
        problem,  # BaseProblem type
        optimizer: optim.Optimizer,
        n_interior: int,
        n_boundary: int,
        n_initial: int = 0,
        loss_weights: Optional[Dict[str, float]] = None,
        device: str = "cpu",
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_config: Optional[Dict] = None,
    ):
        """Initialize the PINN trainer."""
        self.model = model.to(device)
        self.problem = problem
        self.optimizer = optimizer
        self.n_interior = n_interior
        self.n_boundary = n_boundary
        self.n_initial = n_initial
        self.device = device

        # Loss weights
        self.loss_weights = loss_weights or {"pde": 1.0, "bc": 1.0, "ic": 1.0}

        # W&B setup
        self.use_wandb = use_wandb
        self.wandb_run = None

        if self.use_wandb:
            try:
                import wandb

                self.wandb_run = wandb.init(
                    project=wandb_project or "pinn-training",
                    config=wandb_config or {},
                )
                # Log model architecture
                wandb.watch(self.model, log="all", log_freq=100)
            except ImportError:
                print("Warning: wandb not installed. Logging disabled.")
                self.use_wandb = False

        # Training state
        self.epoch = 0
        self.history = {
            "loss_total": [],
            "loss_pde": [],
            "loss_bc": [],
            "loss_ic": [],
            "relative_l2_error": [],
        }

        # Early stopping state
        self.best_val_error = float('inf')
        self.patience_counter = 0
        self.best_model_state = None

    def sample_collocation_points(
        self, random_seed: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Sample collocation points for training.

        Parameters
        ----------
        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        x_interior : torch.Tensor
            Interior points of shape (n_interior, spatial_dim).
        x_boundary : torch.Tensor
            Boundary points of shape (n_boundary_total, spatial_dim).
        x_initial : torch.Tensor or None
            Initial condition points (for time-dependent PDEs).
        """
        # Sample interior points
        x_interior = self.problem.sample_interior_points(
            n=self.n_interior, random_seed=random_seed
        )

        # Sample boundary points
        x_boundary = self.problem.sample_boundary_points(
            n_per_edge=self.n_boundary, random_seed=random_seed
        )

        # Sample initial condition points (if needed)
        x_initial = None
        if self.n_initial > 0:
            # For time-dependent problems, sample at t=0
            # This will be implemented when we add Heat equation
            pass

        # Move to device
        x_interior = x_interior.to(self.device)
        x_boundary = x_boundary.to(self.device)

        return x_interior, x_boundary, x_initial

    def compute_loss(
        self,
        x_interior: torch.Tensor,
        x_boundary: torch.Tensor,
        x_initial: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss and its components.

        Parameters
        ----------
        x_interior : torch.Tensor
            Interior collocation points of shape (n_interior, spatial_dim).
        x_boundary : torch.Tensor
            Boundary points of shape (n_boundary, spatial_dim).
        x_initial : torch.Tensor, optional
            Initial condition points (for time-dependent PDEs).

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary with keys: 'total', 'pde', 'bc', 'ic'.
        """
        # Enable gradients for interior points (needed for PDE residual)
        x_interior = x_interior.clone().requires_grad_(True)

        # --- PDE Loss ---
        # Forward pass
        u_interior = self.model(x_interior)

        # Compute derivatives
        du_dx = compute_derivatives(u_interior, x_interior, order=1)
        d2u_dx2 = compute_derivatives(u_interior, x_interior, order=2)

        # Compute PDE residual
        residual = self.problem.pde_residual(u_interior, x_interior, du_dx, d2u_dx2)
        loss_pde = torch.mean(residual ** 2)

        # --- Boundary Condition Loss ---
        u_boundary = self.model(x_boundary)
        bc_exact = self.problem.boundary_condition(x_boundary)
        loss_bc = torch.mean((u_boundary - bc_exact) ** 2)

        # --- Initial Condition Loss (if applicable) ---
        loss_ic = torch.tensor(0.0, device=self.device)
        if x_initial is not None:
            u_initial = self.model(x_initial)
            ic_exact = self.problem.initial_condition(x_initial)
            loss_ic = torch.mean((u_initial - ic_exact) ** 2)

        # --- Total Loss ---
        loss_total = (
            self.loss_weights["pde"] * loss_pde
            + self.loss_weights["bc"] * loss_bc
            + self.loss_weights["ic"] * loss_ic
        )

        return {
            "total": loss_total,
            "pde": loss_pde,
            "bc": loss_bc,
            "ic": loss_ic,
        }

    def train_step(
        self, x_interior: torch.Tensor, x_boundary: torch.Tensor, x_initial: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Perform a single training step.

        Parameters
        ----------
        x_interior : torch.Tensor
            Interior collocation points.
        x_boundary : torch.Tensor
            Boundary points.
        x_initial : torch.Tensor, optional
            Initial condition points.

        Returns
        -------
        Dict[str, float]
            Dictionary with loss values.
        """
        self.model.train()

        # Zero gradients
        self.optimizer.zero_grad()

        # Compute losses
        losses = self.compute_loss(x_interior, x_boundary, x_initial)

        # Backward pass
        losses["total"].backward()

        # Optimizer step
        self.optimizer.step()

        # Convert to Python floats for logging
        return {k: v.item() for k, v in losses.items()}

    def validate(self, n_test_points: int = 5000) -> float:
        """
        Compute validation metric (relative L2 error).

        Parameters
        ----------
        n_test_points : int, optional
            Number of test points for error computation. Default: 5000.

        Returns
        -------
        float
            Relative L2 error as percentage.
        """
        self.model.eval()
        with torch.no_grad():
            error = self.problem.compute_relative_l2_error(
                self.model, n_test_points=n_test_points, random_seed=None
            )
        return error

    def train(
        self,
        n_epochs: int,
        resample_every: int = 1,
        validate_every: int = 100,
        print_every: int = 100,
        save_every: int = 1000,
        save_path: Optional[str] = None,
        early_stopping: bool = False,
        patience: int = 50,
        min_delta: float = 0.0001,
    ) -> Dict[str, list]:
        """
        Train the PINN for a specified number of epochs.

        Parameters
        ----------
        n_epochs : int
            Number of training epochs.
        resample_every : int, optional
            Resample collocation points every N epochs. Default: 1 (every epoch).
        validate_every : int, optional
            Compute validation error every N epochs. Default: 100.
        print_every : int, optional
            Print progress every N epochs. Default: 100.
        save_every : int, optional
            Save checkpoint every N epochs. Default: 1000.
        save_path : str, optional
            Path to save checkpoints. Default: None (no saving).
        early_stopping : bool, optional
            Whether to enable early stopping. Default: False.
        patience : int, optional
            Number of validation checks with no improvement before stopping. Default: 50.
        min_delta : float, optional
            Minimum change in validation error to qualify as improvement. Default: 0.0001.

        Returns
        -------
        Dict[str, list]
            Training history with loss curves.
        """
        print("=" * 80)
        print("Starting PINN Training")
        print("=" * 80)
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Problem: {self.problem.__class__.__name__}")
        print(f"Interior points: {self.n_interior}")
        print(f"Boundary points: {self.n_boundary * 4} ({self.n_boundary} per edge)")
        print(f"Loss weights: pde={self.loss_weights['pde']}, bc={self.loss_weights['bc']}")
        print(f"Optimizer: {self.optimizer.__class__.__name__}")
        print(f"Device: {self.device}")
        print("=" * 80)
        print()

        start_time = time.time()

        # Initial sampling
        x_interior, x_boundary, x_initial = self.sample_collocation_points()

        for epoch in range(1, n_epochs + 1):
            self.epoch = epoch

            # Resample collocation points
            if epoch % resample_every == 0:
                x_interior, x_boundary, x_initial = self.sample_collocation_points()

            # Training step
            losses = self.train_step(x_interior, x_boundary, x_initial)

            # Log losses
            self.history["loss_total"].append(losses["total"])
            self.history["loss_pde"].append(losses["pde"])
            self.history["loss_bc"].append(losses["bc"])
            self.history["loss_ic"].append(losses["ic"])

            # Validation
            if epoch % validate_every == 0:
                rel_error = self.validate(n_test_points=5000)
                self.history["relative_l2_error"].append(rel_error)

                # Early stopping check
                if early_stopping:
                    if rel_error < self.best_val_error - min_delta:
                        # Improvement detected
                        self.best_val_error = rel_error
                        self.patience_counter = 0
                        # Save best model state
                        self.best_model_state = {
                            k: v.cpu().clone() for k, v in self.model.state_dict().items()
                        }
                    else:
                        # No improvement
                        self.patience_counter += 1

                    # Check if we should stop
                    if self.patience_counter >= patience:
                        print(f"\nEarly stopping triggered at epoch {epoch}")
                        print(f"Best validation error: {self.best_val_error:.4f}%")
                        # Restore best model
                        if self.best_model_state is not None:
                            self.model.load_state_dict(self.best_model_state)
                            print("Restored best model weights")
                        break
            else:
                rel_error = None

            # Logging to W&B
            if self.use_wandb:
                log_dict = {
                    "epoch": epoch,
                    "loss_total": losses["total"],
                    "loss_pde": losses["pde"],
                    "loss_bc": losses["bc"],
                }
                if rel_error is not None:
                    log_dict["relative_l2_error"] = rel_error

                import wandb
                wandb.log(log_dict)

            # Print progress
            if epoch % print_every == 0:
                elapsed = time.time() - start_time
                print(
                    f"Epoch {epoch:5d}/{n_epochs} | "
                    f"Loss: {losses['total']:.6f} "
                    f"(PDE: {losses['pde']:.6f}, BC: {losses['bc']:.6f}) | "
                    f"Time: {elapsed:.1f}s",
                    end="",
                )
                if rel_error is not None:
                    print(f" | L2 Error: {rel_error:.4f}%")
                else:
                    print()

            # Save checkpoint
            if save_path is not None and epoch % save_every == 0:
                self.save_checkpoint(save_path, epoch)

        # Final validation
        final_error = self.validate(n_test_points=10000)
        self.history["relative_l2_error"].append(final_error)

        elapsed = time.time() - start_time
        print()
        print("=" * 80)
        print("Training Complete!")
        print("=" * 80)
        print(f"Total time: {elapsed:.2f}s ({elapsed/n_epochs:.4f}s/epoch)")
        print(f"Final loss: {self.history['loss_total'][-1]:.6f}")
        print(f"Final relative L2 error: {final_error:.4f}%")
        print("=" * 80)

        if self.use_wandb:
            import wandb
            wandb.log({"final_relative_l2_error": final_error})
            wandb.finish()

        return self.history

    def save_checkpoint(self, path: str, epoch: int):
        """
        Save model checkpoint.

        Parameters
        ----------
        path : str
            Directory to save checkpoint.
        epoch : int
            Current epoch number.
        """
        import os

        os.makedirs(path, exist_ok=True)
        checkpoint_path = os.path.join(path, f"checkpoint_epoch_{epoch}.pt")

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss_weights": self.loss_weights,
                "history": self.history,
            },
            checkpoint_path,
        )

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.

        Parameters
        ----------
        checkpoint_path : str
            Path to checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.loss_weights = checkpoint["loss_weights"]
        self.history = checkpoint["history"]

        print(f"Loaded checkpoint from epoch {self.epoch}")

    def generate_solution_heatmap(
        self,
        save_path: str,
        n_points: int = 100,
        figsize: Tuple[int, int] = (15, 5),
        dpi: int = 150,
    ):
        """
        Generate and save solution heatmap visualization.

        Creates a figure with three subplots:
        1. PINN predicted solution
        2. Analytical solution (if available)
        3. Absolute error

        Parameters
        ----------
        save_path : str
            Path to save the figure.
        n_points : int, optional
            Number of points per dimension for visualization grid. Default: 100.
        figsize : Tuple[int, int], optional
            Figure size (width, height) in inches. Default: (15, 5).
        dpi : int, optional
            Resolution in dots per inch. Default: 150.
        """
        self.model.eval()

        # Check if problem is 2D
        if self.problem.spatial_dim != 2:
            raise NotImplementedError(
                "Visualization currently only supports 2D problems"
            )

        # Create evaluation grid
        x = torch.linspace(
            self.problem.domain[0][0], self.problem.domain[0][1], n_points
        )
        y = torch.linspace(
            self.problem.domain[1][0], self.problem.domain[1][1], n_points
        )
        X, Y = torch.meshgrid(x, y, indexing="ij")
        grid_points = torch.stack([X.flatten(), Y.flatten()], dim=1).to(self.device)

        # Compute PINN solution
        with torch.no_grad():
            u_pred = self.model(grid_points).cpu().numpy().reshape(n_points, n_points)

        # Compute analytical solution
        u_exact = (
            self.problem.analytical_solution(grid_points)
            .cpu()
            .numpy()
            .reshape(n_points, n_points)
        )

        # Compute error
        error = np.abs(u_pred - u_exact)

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi)

        # Convert grid to numpy for plotting
        X_np = X.numpy()
        Y_np = Y.numpy()

        # Plot 1: PINN Solution
        im1 = axes[0].contourf(X_np, Y_np, u_pred, levels=50, cmap="viridis")
        axes[0].set_title("PINN Solution", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[0].set_aspect("equal")
        plt.colorbar(im1, ax=axes[0], label="u(x, y)")

        # Plot 2: Analytical Solution
        im2 = axes[1].contourf(X_np, Y_np, u_exact, levels=50, cmap="viridis")
        axes[1].set_title("Analytical Solution", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        axes[1].set_aspect("equal")
        plt.colorbar(im2, ax=axes[1], label="u(x, y)")

        # Plot 3: Absolute Error
        im3 = axes[2].contourf(X_np, Y_np, error, levels=50, cmap="hot")
        axes[2].set_title(
            f"Absolute Error\n(Max: {error.max():.2e}, Mean: {error.mean():.2e})",
            fontsize=12,
            fontweight="bold",
        )
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("y")
        axes[2].set_aspect("equal")
        plt.colorbar(im3, ax=axes[2], label="|u_pred - u_exact|")

        plt.tight_layout()

        # Save figure
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()

        print(f"Solution heatmap saved to: {save_path}")

        # Log to W&B if enabled
        if self.use_wandb:
            try:
                import wandb
                wandb.log({"solution_heatmap": wandb.Image(save_path)})
            except ImportError:
                pass


def train_pinn(
    model: nn.Module,
    problem,  # BaseProblem
    config: Dict,
) -> Tuple[nn.Module, Dict]:
    """
    Convenience function to train a PINN with a configuration dictionary.

    Parameters
    ----------
    model : nn.Module
        PINN model to train.
    problem : BaseProblem
        PDE problem definition.
    config : Dict
        Training configuration with keys:
        - 'optimizer': optimizer type ('adam', 'sgd', 'lbfgs')
        - 'lr': learning rate
        - 'n_epochs': number of training epochs
        - 'n_interior': number of interior points
        - 'n_boundary': number of boundary points per edge
        - 'loss_weights': dict with 'pde', 'bc', 'ic' weights
        - 'device': 'cpu' or 'cuda'
        - 'use_wandb': whether to use W&B logging
        - 'wandb_project': W&B project name
        - Additional optional keys: resample_every, validate_every, etc.

    Returns
    -------
    model : nn.Module
        Trained model.
    history : Dict
        Training history with loss curves.

    Examples
    --------
    >>> config = {
    ...     'optimizer': 'adam',
    ...     'lr': 1e-3,
    ...     'n_epochs': 10000,
    ...     'n_interior': 10000,
    ...     'n_boundary': 100,
    ...     'loss_weights': {'pde': 1.0, 'bc': 1.0, 'ic': 1.0},
    ...     'device': 'cpu',
    ... }
    >>> trained_model, history = train_pinn(model, problem, config)
    """
    # Create optimizer
    optimizer_type = config.get("optimizer", "adam").lower()
    lr = config.get("lr", 1e-3)

    if optimizer_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_type == "lbfgs":
        optimizer = optim.LBFGS(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    # Create trainer
    trainer = PINNTrainer(
        model=model,
        problem=problem,
        optimizer=optimizer,
        n_interior=config.get("n_interior", 10000),
        n_boundary=config.get("n_boundary", 100),
        n_initial=config.get("n_initial", 0),
        loss_weights=config.get("loss_weights", {"pde": 1.0, "bc": 1.0, "ic": 1.0}),
        device=config.get("device", "cpu"),
        use_wandb=config.get("use_wandb", False),
        wandb_project=config.get("wandb_project", None),
        wandb_config=config,
    )

    # Train
    history = trainer.train(
        n_epochs=config.get("n_epochs", 10000),
        resample_every=config.get("resample_every", 1),
        validate_every=config.get("validate_every", 100),
        print_every=config.get("print_every", 100),
        save_every=config.get("save_every", 1000),
        save_path=config.get("save_path", None),
        early_stopping=config.get("early_stopping", False),
        patience=config.get("patience", 50),
        min_delta=config.get("min_delta", 0.0001),
    )

    return model, history
