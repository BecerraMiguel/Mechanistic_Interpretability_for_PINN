"""
Probing classifiers for detecting derivative information in PINN activations.

This module provides tools for training linear probes to detect whether
intermediate activations encode derivative information (du/dx, du/dy, Laplacian, etc.).
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class LinearProbe:
    """
    Linear probe for single-target prediction from activations.

    A linear probe is a simple linear regression model (y = Wx + b) trained
    to predict a target quantity (e.g., du/dx) from network activations.
    High R² scores indicate that the target information is linearly accessible
    in the activations.

    Parameters
    ----------
    input_dim : int
        Dimension of input activations (e.g., hidden layer size).
    output_dim : int, default=1
        Dimension of output targets (typically 1 for scalar derivatives).

    Attributes
    ----------
    weights : torch.nn.Linear
        Linear layer mapping activations to targets.
    input_dim : int
        Input dimension.
    output_dim : int
        Output dimension.
    is_fitted : bool
        Whether the probe has been trained.
    training_history : Dict[str, list]
        Training loss history.

    Examples
    --------
    >>> # Create probe for 64-dimensional activations
    >>> probe = LinearProbe(input_dim=64, output_dim=1)
    >>> # Train on activations and derivative targets
    >>> probe.fit(activations, du_dx_targets, epochs=1000)
    >>> # Evaluate performance
    >>> scores = probe.score(test_activations, test_targets)
    >>> print(f"R² = {scores['r_squared']:.4f}")
    """

    def __init__(self, input_dim: int, output_dim: int = 1):
        """
        Initialize LinearProbe.

        Parameters
        ----------
        input_dim : int
            Dimension of input activations.
        output_dim : int, default=1
            Dimension of output targets.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_fitted = False
        self.training_history = {"loss": []}

        # Create linear layer (y = Wx + b)
        self.weights = nn.Linear(input_dim, output_dim)

        # Initialize with small random weights
        nn.init.xavier_normal_(self.weights.weight)
        nn.init.zeros_(self.weights.bias)

    def fit(
        self,
        activations: torch.Tensor,
        targets: torch.Tensor,
        epochs: int = 1000,
        lr: float = 1e-3,
        batch_size: Optional[int] = None,
        verbose: bool = False,
    ) -> Dict[str, list]:
        """
        Train the linear probe using MSE loss and Adam optimizer.

        Parameters
        ----------
        activations : torch.Tensor
            Input activations of shape (n_samples, input_dim).
        targets : torch.Tensor
            Target values of shape (n_samples, output_dim) or (n_samples,).
        epochs : int, default=1000
            Number of training epochs.
        lr : float, default=1e-3
            Learning rate for Adam optimizer.
        batch_size : Optional[int], default=None
            Batch size for training. If None, uses full batch.
        verbose : bool, default=False
            Whether to print training progress.

        Returns
        -------
        Dict[str, list]
            Training history with 'loss' key containing per-epoch losses.

        Raises
        ------
        ValueError
            If input dimensions don't match expected sizes.
        """
        # Validate inputs
        if activations.shape[0] != targets.shape[0]:
            raise ValueError(
                f"Number of samples mismatch: activations {activations.shape[0]} "
                f"vs targets {targets.shape[0]}"
            )

        if activations.shape[1] != self.input_dim:
            raise ValueError(
                f"Input dimension mismatch: expected {self.input_dim}, "
                f"got {activations.shape[1]}"
            )

        # Ensure targets have correct shape
        if targets.ndim == 1:
            targets = targets.unsqueeze(1)

        if targets.shape[1] != self.output_dim:
            raise ValueError(
                f"Output dimension mismatch: expected {self.output_dim}, "
                f"got {targets.shape[1]}"
            )

        # Move to same device as weights
        device = next(self.weights.parameters()).device
        activations = activations.to(device)
        targets = targets.to(device)

        # Setup training
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.weights.parameters(), lr=lr)

        # Determine batch size
        n_samples = activations.shape[0]
        if batch_size is None:
            batch_size = n_samples  # Full batch

        # Training loop
        self.weights.train()
        self.training_history = {"loss": []}

        for epoch in range(epochs):
            # Shuffle data
            indices = torch.randperm(n_samples)
            epoch_loss = 0.0
            n_batches = 0

            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i : i + batch_size]
                batch_activations = activations[batch_indices]
                batch_targets = targets[batch_indices]

                # Forward pass
                optimizer.zero_grad()
                predictions = self.weights(batch_activations)
                loss = criterion(predictions, batch_targets)

                # Backward pass
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            # Record average loss for epoch
            avg_loss = epoch_loss / n_batches
            self.training_history["loss"].append(avg_loss)

            # Print progress
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

        self.is_fitted = True
        return self.training_history

    def predict(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Predict targets from activations.

        Parameters
        ----------
        activations : torch.Tensor
            Input activations of shape (n_samples, input_dim).

        Returns
        -------
        torch.Tensor
            Predicted targets of shape (n_samples, output_dim).

        Raises
        ------
        RuntimeError
            If probe has not been fitted yet.
        ValueError
            If input dimension doesn't match expected size.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Probe must be fitted before making predictions. Call fit() first."
            )

        if activations.shape[1] != self.input_dim:
            raise ValueError(
                f"Input dimension mismatch: expected {self.input_dim}, "
                f"got {activations.shape[1]}"
            )

        # Move to same device as weights
        device = next(self.weights.parameters()).device
        activations = activations.to(device)

        # Make predictions
        self.weights.eval()
        with torch.no_grad():
            predictions = self.weights(activations)

        return predictions

    def score(
        self, activations: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute performance metrics on test data.

        Parameters
        ----------
        activations : torch.Tensor
            Input activations of shape (n_samples, input_dim).
        targets : torch.Tensor
            True target values of shape (n_samples, output_dim) or (n_samples,).

        Returns
        -------
        Dict[str, float]
            Dictionary containing:
            - 'mse': Mean squared error
            - 'r_squared': R² score (coefficient of determination)
            - 'explained_variance': Explained variance score

        Raises
        ------
        RuntimeError
            If probe has not been fitted yet.

        Notes
        -----
        R² score indicates how well the linear probe can predict the target:
        - R² = 1.0: Perfect prediction
        - R² = 0.0: As good as predicting the mean
        - R² < 0.0: Worse than predicting the mean
        """
        if not self.is_fitted:
            raise RuntimeError("Probe must be fitted before scoring. Call fit() first.")

        # Get predictions
        predictions = self.predict(activations)

        # Ensure targets have correct shape
        if targets.ndim == 1:
            targets = targets.unsqueeze(1)

        # Move to CPU and convert to numpy
        predictions_np = predictions.cpu().numpy()
        targets_np = targets.cpu().numpy()

        # Flatten arrays for metric computation
        predictions_flat = predictions_np.flatten()
        targets_flat = targets_np.flatten()

        # Compute MSE
        mse = np.mean((predictions_flat - targets_flat) ** 2)

        # Compute R² (coefficient of determination)
        # R² = 1 - (SS_res / SS_tot)
        ss_res = np.sum((targets_flat - predictions_flat) ** 2)
        ss_tot = np.sum((targets_flat - np.mean(targets_flat)) ** 2)

        if ss_tot == 0:
            # Perfect predictions or constant targets
            r2 = 1.0 if ss_res == 0 else 0.0
        else:
            r2 = 1.0 - (ss_res / ss_tot)

        # Compute explained variance
        # explained_var = 1 - Var(y - y_pred) / Var(y)
        var_residual = np.var(targets_flat - predictions_flat)
        var_targets = np.var(targets_flat)

        if var_targets == 0:
            # Constant targets
            explained_var = 1.0 if var_residual == 0 else 0.0
        else:
            explained_var = 1.0 - (var_residual / var_targets)

        return {
            "mse": float(mse),
            "r_squared": float(r2),
            "explained_variance": float(explained_var),
        }

    def get_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the learned linear weights and bias.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (weight_matrix, bias_vector) where:
            - weight_matrix: Shape (output_dim, input_dim)
            - bias_vector: Shape (output_dim,)

        Raises
        ------
        RuntimeError
            If probe has not been fitted yet.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Probe must be fitted before accessing weights. Call fit() first."
            )

        weight = self.weights.weight.detach().cpu().numpy()
        bias = self.weights.bias.detach().cpu().numpy()

        return weight, bias

    def to(self, device: torch.device):
        """
        Move probe to specified device.

        Parameters
        ----------
        device : torch.device
            Target device (e.g., 'cuda' or 'cpu').

        Returns
        -------
        LinearProbe
            Self (for chaining).
        """
        self.weights = self.weights.to(device)
        return self

    def __repr__(self) -> str:
        """String representation of LinearProbe."""
        fitted_str = "fitted" if self.is_fitted else "not fitted"
        return (
            f"LinearProbe(input_dim={self.input_dim}, "
            f"output_dim={self.output_dim}, {fitted_str})"
        )
