"""
Activation extraction and storage for mechanistic interpretability.

This module provides tools for extracting neural network activations on dense grids
and storing them efficiently in HDF5 format for subsequent analysis.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.figure import Figure


class ActivationStore:
    """
    Manages extraction and storage of neural network activations.

    Activations are extracted on dense grids and stored in HDF5 format for
    efficient access during interpretability analysis. The HDF5 file contains:
    - /coordinates: (N, input_dim) array of input coordinates
    - /layer_0, /layer_1, ...: (N, hidden_dim) arrays of layer activations

    Attributes:
        save_path (Path): Path to HDF5 file for storage
        grid_resolution (int): Resolution of the extraction grid
        domain_bounds (Tuple): Domain boundaries for grid generation
        layer_names (List[str]): Names of stored layers
    """

    def __init__(self, save_path: str):
        """
        Initialize activation store.

        Parameters:
            save_path (str): Path where HDF5 file will be saved
        """
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.grid_resolution = None
        self.domain_bounds = None
        self.layer_names = []

    def extract_on_grid(
        self,
        model: nn.Module,
        grid_resolution: int = 100,
        domain_bounds: Tuple[Tuple[float, float], ...] = ((0.0, 1.0), (0.0, 1.0)),
        batch_size: int = 1000,
        device: str = "cpu",
    ) -> None:
        """
        Extract activations on a dense regular grid.

        Creates a uniform grid covering the specified domain and extracts
        activations from all layers of the model. Results are saved to HDF5.

        Parameters:
            model (nn.Module): Neural network model with get_activations() method
            grid_resolution (int): Number of points per dimension (default: 100)
            domain_bounds (Tuple): Bounds for each dimension ((x_min, x_max), (y_min, y_max), ...)
            batch_size (int): Batch size for processing (to avoid memory issues)
            device (str): Device to use for computation ('cpu' or 'cuda')

        Raises:
            ValueError: If model doesn't have get_activations() method

        Example:
            >>> store = ActivationStore("data/activations/poisson_mlp.h5")
            >>> store.extract_on_grid(model, grid_resolution=100)
            >>> print(f"Extracted {len(store.layer_names)} layers")
        """
        # Validate model has activation extraction capability
        if not hasattr(model, "get_activations"):
            raise ValueError(
                "Model must have get_activations() method for activation extraction. "
                "Ensure model inherits from BasePINN or implements this interface."
            )

        # Store grid parameters
        self.grid_resolution = grid_resolution
        self.domain_bounds = domain_bounds
        input_dim = len(domain_bounds)

        # Generate grid coordinates
        print(f"Generating {grid_resolution}^{input_dim} grid...")
        grid_coords = self._generate_grid(grid_resolution, domain_bounds)
        n_points = len(grid_coords)
        print(f"Total grid points: {n_points}")

        # Move model to device
        model = model.to(device)
        model.eval()

        # Dictionary to accumulate activations
        all_activations = {}

        # Process grid in batches
        print(f"Extracting activations in batches of {batch_size}...")
        n_batches = (n_points + batch_size - 1) // batch_size

        with torch.no_grad():
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_points)

                # Get batch coordinates
                batch_coords = torch.tensor(
                    grid_coords[start_idx:end_idx],
                    dtype=torch.float32,
                    device=device,
                    requires_grad=False,
                )

                # Forward pass to populate activations
                _ = model(batch_coords)

                # Get activations from model
                activations = model.get_activations()

                # Accumulate activations for each layer
                for layer_name, activation_tensor in activations.items():
                    # Convert to numpy and move to CPU
                    activation_np = activation_tensor.cpu().numpy()

                    if layer_name not in all_activations:
                        # Initialize list for this layer
                        all_activations[layer_name] = []

                    all_activations[layer_name].append(activation_np)

                if (batch_idx + 1) % max(1, n_batches // 10) == 0:
                    progress = (batch_idx + 1) / n_batches * 100
                    print(
                        f"  Progress: {progress:.1f}% ({batch_idx + 1}/{n_batches} batches)"
                    )

        # Concatenate all batches
        print("Concatenating batches...")
        for layer_name in all_activations:
            all_activations[layer_name] = np.concatenate(
                all_activations[layer_name], axis=0
            )

        # Save to HDF5
        self._save_to_hdf5(grid_coords, all_activations)

        print(f"✓ Activations saved to {self.save_path}")
        print(f"  Layers extracted: {len(all_activations)}")
        print(f"  Grid points: {n_points}")

    def _generate_grid(
        self, resolution: int, domain_bounds: Tuple[Tuple[float, float], ...]
    ) -> np.ndarray:
        """
        Generate uniform grid coordinates.

        Parameters:
            resolution (int): Number of points per dimension
            domain_bounds (Tuple): Bounds for each dimension

        Returns:
            Grid coordinates of shape (resolution^input_dim, input_dim)
        """
        # Create 1D grids for each dimension
        grids_1d = []
        for bound_min, bound_max in domain_bounds:
            grids_1d.append(np.linspace(bound_min, bound_max, resolution))

        # Create meshgrid
        meshgrids = np.meshgrid(*grids_1d, indexing="ij")

        # Flatten and stack
        grid_coords = np.stack([g.flatten() for g in meshgrids], axis=1)

        return grid_coords

    def _save_to_hdf5(
        self, coordinates: np.ndarray, activations: Dict[str, np.ndarray]
    ) -> None:
        """
        Save coordinates and activations to HDF5 file.

        Parameters:
            coordinates (np.ndarray): Grid coordinates, shape (N, input_dim)
            activations (Dict[str, np.ndarray]): Layer activations
        """
        print(f"Saving to HDF5: {self.save_path}")

        with h5py.File(self.save_path, "w") as f:
            # Save coordinates
            f.create_dataset("coordinates", data=coordinates, compression="gzip")

            # Save metadata
            f.attrs["grid_resolution"] = self.grid_resolution
            f.attrs["n_points"] = len(coordinates)
            f.attrs["input_dim"] = coordinates.shape[1]
            f.attrs["domain_bounds"] = str(self.domain_bounds)

            # Save each layer's activations
            self.layer_names = sorted(activations.keys())
            for layer_name in self.layer_names:
                activation_data = activations[layer_name]
                f.create_dataset(layer_name, data=activation_data, compression="gzip")
                print(f"  Saved {layer_name}: shape {activation_data.shape}")

            # Save layer names list
            f.attrs["layer_names"] = ",".join(self.layer_names)

        print(f"✓ HDF5 file created: {self.save_path.stat().st_size / 1024:.1f} KB")

    def load_layer(self, layer_name: str) -> np.ndarray:
        """
        Load a specific layer's activations from HDF5 file.

        Uses memory mapping for efficient access without loading entire file.

        Parameters:
            layer_name (str): Name of layer to load (e.g., 'layer_0')

        Returns:
            Activations array of shape (N, hidden_dim)

        Raises:
            FileNotFoundError: If HDF5 file doesn't exist
            KeyError: If layer_name not found in file

        Example:
            >>> store = ActivationStore("data/activations/poisson_mlp.h5")
            >>> layer_0_acts = store.load_layer('layer_0')
            >>> print(layer_0_acts.shape)  # (10000, 64)
        """
        if not self.save_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.save_path}")

        with h5py.File(self.save_path, "r") as f:
            if layer_name not in f:
                available_layers = [k for k in f.keys() if k.startswith("layer_")]
                raise KeyError(
                    f"Layer '{layer_name}' not found. Available layers: {available_layers}"
                )

            # Load data (h5py automatically uses memory mapping)
            activations = f[layer_name][:]

        return activations

    def load_coordinates(self) -> np.ndarray:
        """
        Load grid coordinates from HDF5 file.

        Returns:
            Coordinates array of shape (N, input_dim)

        Raises:
            FileNotFoundError: If HDF5 file doesn't exist
        """
        if not self.save_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.save_path}")

        with h5py.File(self.save_path, "r") as f:
            coordinates = f["coordinates"][:]
            # Load metadata
            self.grid_resolution = f.attrs.get("grid_resolution", None)
            self.layer_names = f.attrs.get("layer_names", "").split(",")

        return coordinates

    def get_metadata(self) -> Dict:
        """
        Get metadata about stored activations.

        Returns:
            Dictionary with metadata (grid_resolution, n_points, layer_names, etc.)
        """
        if not self.save_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.save_path}")

        with h5py.File(self.save_path, "r") as f:
            metadata = {
                "grid_resolution": f.attrs.get("grid_resolution", None),
                "n_points": f.attrs.get("n_points", None),
                "input_dim": f.attrs.get("input_dim", None),
                "domain_bounds": f.attrs.get("domain_bounds", None),
                "layer_names": f.attrs.get("layer_names", "").split(","),
                "file_size_kb": self.save_path.stat().st_size / 1024,
            }

        return metadata

    def visualize_neuron(
        self,
        layer_name: str,
        neuron_idx: int,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 6),
        dpi: int = 100,
        cmap: str = "viridis",
        show_colorbar: bool = True,
    ) -> Figure:
        """
        Visualize a specific neuron's activation pattern as 2D heatmap.

        Creates a heatmap showing how a neuron responds across the spatial domain.
        Only works for 2D input domains.

        Parameters:
            layer_name (str): Name of layer (e.g., 'layer_0')
            neuron_idx (int): Index of neuron within layer
            save_path (Optional[str]): Path to save figure (if None, doesn't save)
            figsize (Tuple[int, int]): Figure size in inches
            dpi (int): Resolution for saved figure
            cmap (str): Matplotlib colormap name
            show_colorbar (bool): Whether to show colorbar

        Returns:
            Matplotlib Figure object

        Raises:
            ValueError: If domain is not 2D
            FileNotFoundError: If HDF5 file doesn't exist
            IndexError: If neuron_idx out of bounds

        Example:
            >>> store = ActivationStore("data/activations/poisson_mlp.h5")
            >>> fig = store.visualize_neuron('layer_0', neuron_idx=5)
            >>> fig.savefig('outputs/neuron_5_heatmap.png')
        """
        # Load data
        coordinates = self.load_coordinates()
        activations = self.load_layer(layer_name)

        # Validate input dimension
        if coordinates.shape[1] != 2:
            raise ValueError(
                f"Visualization only supports 2D domains. "
                f"Current domain has {coordinates.shape[1]} dimensions."
            )

        # Validate neuron index
        if neuron_idx < 0 or neuron_idx >= activations.shape[1]:
            raise IndexError(
                f"Neuron index {neuron_idx} out of bounds. "
                f"Layer has {activations.shape[1]} neurons (valid indices: 0-{activations.shape[1]-1})"
            )

        # Extract neuron activations
        neuron_activations = activations[:, neuron_idx]

        # Reshape to 2D grid
        if self.grid_resolution is None:
            # Try to infer grid resolution
            n_points = len(coordinates)
            resolution = int(np.sqrt(n_points))
            if resolution * resolution != n_points:
                raise ValueError(
                    f"Cannot infer grid resolution from {n_points} points. "
                    "Ensure grid is square or set grid_resolution explicitly."
                )
            self.grid_resolution = resolution

        activation_grid = neuron_activations.reshape(
            self.grid_resolution, self.grid_resolution
        )

        # Extract x and y coordinates
        x = coordinates[:, 0].reshape(self.grid_resolution, self.grid_resolution)
        y = coordinates[:, 1].reshape(self.grid_resolution, self.grid_resolution)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Plot heatmap
        im = ax.pcolormesh(x, y, activation_grid, cmap=cmap, shading="auto")

        if show_colorbar:
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Activation", rotation=270, labelpad=20)

        # Labels and title
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"{layer_name}, Neuron {neuron_idx}")
        ax.set_aspect("equal")

        # Statistics text
        stats_text = (
            f"Mean: {neuron_activations.mean():.4f}\n"
            f"Std: {neuron_activations.std():.4f}\n"
            f"Min: {neuron_activations.min():.4f}\n"
            f"Max: {neuron_activations.max():.4f}"
        )
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            fontsize=8,
        )

        plt.tight_layout()

        # Save if path provided
        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"✓ Saved visualization to {save_path}")

        return fig

    def visualize_layer_summary(
        self,
        layer_name: str,
        n_neurons: int = 16,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (16, 12),
        dpi: int = 100,
        cmap: str = "viridis",
    ) -> Figure:
        """
        Visualize multiple neurons from a layer in a grid layout.

        Creates a summary visualization showing the first n_neurons from a layer.

        Parameters:
            layer_name (str): Name of layer (e.g., 'layer_0')
            n_neurons (int): Number of neurons to visualize (default: 16)
            save_path (Optional[str]): Path to save figure
            figsize (Tuple[int, int]): Figure size in inches
            dpi (int): Resolution for saved figure
            cmap (str): Matplotlib colormap name

        Returns:
            Matplotlib Figure object
        """
        # Load data
        coordinates = self.load_coordinates()
        activations = self.load_layer(layer_name)

        # Validate
        if coordinates.shape[1] != 2:
            raise ValueError("Visualization only supports 2D domains")

        n_neurons = min(n_neurons, activations.shape[1])

        # Determine grid layout
        n_cols = int(np.ceil(np.sqrt(n_neurons)))
        n_rows = int(np.ceil(n_neurons / n_cols))

        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi)
        axes = np.atleast_2d(axes)

        # Reshape coordinates
        x = coordinates[:, 0].reshape(self.grid_resolution, self.grid_resolution)
        y = coordinates[:, 1].reshape(self.grid_resolution, self.grid_resolution)

        for i in range(n_neurons):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]

            # Get neuron activations
            neuron_acts = activations[:, i].reshape(
                self.grid_resolution, self.grid_resolution
            )

            # Plot
            im = ax.pcolormesh(x, y, neuron_acts, cmap=cmap, shading="auto")
            ax.set_title(f"Neuron {i}", fontsize=8)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])

        # Hide unused subplots
        for i in range(n_neurons, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].axis("off")

        fig.suptitle(f"{layer_name} - First {n_neurons} Neurons", fontsize=14)
        plt.tight_layout()

        # Save if path provided
        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"✓ Saved layer summary to {save_path}")

        return fig


def extract_activations_from_model(
    model: nn.Module,
    save_path: str,
    grid_resolution: int = 100,
    domain_bounds: Tuple[Tuple[float, float], ...] = ((0.0, 1.0), (0.0, 1.0)),
    batch_size: int = 1000,
    device: str = "cpu",
) -> ActivationStore:
    """
    Convenience function to extract and save activations in one call.

    Parameters:
        model (nn.Module): Neural network model
        save_path (str): Path for HDF5 file
        grid_resolution (int): Grid resolution per dimension
        domain_bounds (Tuple): Domain boundaries
        batch_size (int): Batch size for processing
        device (str): Device for computation

    Returns:
        ActivationStore instance with extracted activations

    Example:
        >>> model = torch.load('poisson_pinn_trained.pt')
        >>> store = extract_activations_from_model(
        ...     model, 'data/activations/poisson.h5', grid_resolution=100
        ... )
    """
    store = ActivationStore(save_path)
    store.extract_on_grid(model, grid_resolution, domain_bounds, batch_size, device)
    return store
