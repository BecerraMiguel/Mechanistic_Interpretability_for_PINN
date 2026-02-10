"""
Train linear probes for first derivatives (du/dx, du/dy) at each layer.

Day 9, Task 1: Layer-wise Derivative Probing
This script trains probes to detect whether first-order derivative information
is linearly accessible in each hidden layer of the trained Poisson PINN.
"""

import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from src.interpretability.activation_store import ActivationStore
from src.interpretability.probing import LinearProbe
from src.problems.poisson import PoissonProblem


def load_activations_and_coordinates(activation_path: str):
    """
    Load activations and coordinates from HDF5 file.

    Parameters
    ----------
    activation_path : str
        Path to HDF5 activation file

    Returns
    -------
    coordinates : torch.Tensor
        Shape (N, 2) - input coordinates
    layer_names : list[str]
        Names of layers in the file
    """
    print(f"\n📂 Loading activations from: {activation_path}")

    with h5py.File(activation_path, "r") as f:
        # Load coordinates
        coordinates = torch.tensor(f["coordinates"][:], dtype=torch.float32)

        # Get layer names (exclude coordinates)
        layer_names = [key for key in f.keys() if key.startswith("layer_")]
        layer_names.sort()  # Ensure order: layer_0, layer_1, ...

        print(f"   ✓ Loaded {len(coordinates)} coordinate points")
        print(f"   ✓ Found {len(layer_names)} layers: {layer_names}")

    return coordinates, layer_names


def compute_ground_truth_derivatives(coordinates: torch.Tensor, problem: PoissonProblem):
    """
    Compute analytical first derivatives (du/dx, du/dy) for probing targets.

    Parameters
    ----------
    coordinates : torch.Tensor
        Shape (N, 2) - input coordinates
    problem : PoissonProblem
        Problem instance with analytical derivative methods

    Returns
    -------
    du_dx : np.ndarray
        Shape (N, 1) - ground-truth ∂u/∂x
    du_dy : np.ndarray
        Shape (N, 1) - ground-truth ∂u/∂y
    """
    print("\n🧮 Computing ground-truth derivatives...")

    # Compute analytical derivatives
    du_dx = problem.analytical_derivative_du_dx(coordinates)  # (N, 1)
    du_dy = problem.analytical_derivative_du_dy(coordinates)  # (N, 1)

    # Convert to numpy for sklearn-like interface
    du_dx_np = du_dx.detach().cpu().numpy()
    du_dy_np = du_dy.detach().cpu().numpy()

    print(f"   ✓ du/dx: shape={du_dx_np.shape}, mean={du_dx_np.mean():.4f}, std={du_dx_np.std():.4f}")
    print(f"   ✓ du/dy: shape={du_dy_np.shape}, mean={du_dy_np.mean():.4f}, std={du_dy_np.std():.4f}")

    return du_dx_np, du_dy_np


def train_probes_for_layer(
    layer_name: str,
    activations: np.ndarray,
    du_dx: np.ndarray,
    du_dy: np.ndarray,
    epochs: int = 1000,
    lr: float = 1e-3,
    device: str = "cpu",
):
    """
    Train two probes (du/dx and du/dy) for a single layer.

    Parameters
    ----------
    layer_name : str
        Name of layer (e.g., "layer_0")
    activations : np.ndarray
        Shape (N, hidden_dim) - layer activations
    du_dx : np.ndarray
        Shape (N, 1) - ground-truth ∂u/∂x
    du_dy : np.ndarray
        Shape (N, 1) - ground-truth ∂u/∂y
    epochs : int
        Number of training epochs
    lr : float
        Learning rate
    device : str
        Device to train on ("cpu" or "cuda")

    Returns
    -------
    results : dict
        Dictionary with probe scores for this layer
    """
    print(f"\n🔬 Training probes for {layer_name}...")
    print(f"   Activations: {activations.shape}")

    input_dim = activations.shape[1]
    output_dim = 1

    # Convert numpy arrays to torch tensors
    activations_tensor = torch.tensor(activations, dtype=torch.float32)
    du_dx_tensor = torch.tensor(du_dx, dtype=torch.float32)
    du_dy_tensor = torch.tensor(du_dy, dtype=torch.float32)

    # Initialize probes
    probe_du_dx = LinearProbe(input_dim=input_dim, output_dim=output_dim)
    probe_du_dy = LinearProbe(input_dim=input_dim, output_dim=output_dim)

    # Move to device
    probe_du_dx.to(device)
    probe_du_dy.to(device)

    # Train probe for du/dx
    print(f"   Training du/dx probe...")
    start_time = time.time()
    probe_du_dx.fit(activations_tensor, du_dx_tensor, epochs=epochs, lr=lr, verbose=False)
    du_dx_time = time.time() - start_time

    # Train probe for du/dy
    print(f"   Training du/dy probe...")
    start_time = time.time()
    probe_du_dy.fit(activations_tensor, du_dy_tensor, epochs=epochs, lr=lr, verbose=False)
    du_dy_time = time.time() - start_time

    # Evaluate probes
    scores_du_dx = probe_du_dx.score(activations_tensor, du_dx_tensor)
    scores_du_dy = probe_du_dy.score(activations_tensor, du_dy_tensor)

    print(f"   ✓ du/dx: R²={scores_du_dx['r_squared']:.4f}, MSE={scores_du_dx['mse']:.6f} (time: {du_dx_time:.2f}s)")
    print(f"   ✓ du/dy: R²={scores_du_dy['r_squared']:.4f}, MSE={scores_du_dy['mse']:.6f} (time: {du_dy_time:.2f}s)")

    # Store results
    results = {
        "layer_name": layer_name,
        "du_dx": {
            "r2": float(scores_du_dx["r_squared"]),
            "mse": float(scores_du_dx["mse"]),
            "explained_variance": float(scores_du_dx["explained_variance"]),
            "training_time_s": du_dx_time,
        },
        "du_dy": {
            "r2": float(scores_du_dy["r_squared"]),
            "mse": float(scores_du_dy["mse"]),
            "explained_variance": float(scores_du_dy["explained_variance"]),
            "training_time_s": du_dy_time,
        },
    }

    return results, probe_du_dx, probe_du_dy


def main():
    """Main script for training first derivative probes."""
    print("=" * 80)
    print("DAY 9, TASK 1: Train Probes for First Derivatives (du/dx, du/dy)")
    print("=" * 80)

    # Configuration
    activation_path = "data/activations/poisson_mlp_100x100.h5"
    output_dir = Path("outputs/day9_task1")
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = 1000
    lr = 1e-3
    device = "cpu"

    print(f"\n⚙️  Configuration:")
    print(f"   Activation file: {activation_path}")
    print(f"   Output directory: {output_dir}")
    print(f"   Training epochs: {epochs}")
    print(f"   Learning rate: {lr}")
    print(f"   Device: {device}")

    # Load activations and coordinates
    coordinates, layer_names = load_activations_and_coordinates(activation_path)

    # Initialize Poisson problem for ground-truth derivatives
    problem = PoissonProblem()

    # Compute ground-truth derivatives
    du_dx, du_dy = compute_ground_truth_derivatives(coordinates, problem)

    # Train probes for each layer
    all_results = []
    all_probes = {}

    print("\n" + "=" * 80)
    print("TRAINING PROBES FOR ALL LAYERS")
    print("=" * 80)

    total_start_time = time.time()

    for layer_name in layer_names:
        # Load activations for this layer
        with h5py.File(activation_path, "r") as f:
            activations = f[layer_name][:]  # (N, hidden_dim)

        # Train probes
        layer_results, probe_du_dx, probe_du_dy = train_probes_for_layer(
            layer_name=layer_name,
            activations=activations,
            du_dx=du_dx,
            du_dy=du_dy,
            epochs=epochs,
            lr=lr,
            device=device,
        )

        all_results.append(layer_results)
        all_probes[layer_name] = {
            "du_dx": probe_du_dx,
            "du_dy": probe_du_dy,
        }

    total_time = time.time() - total_start_time

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY: First Derivative Probing Results")
    print("=" * 80)
    print(f"\n{'Layer':<12} {'du/dx R²':<12} {'du/dy R²':<12} {'du/dx MSE':<14} {'du/dy MSE':<14}")
    print("-" * 80)

    for result in all_results:
        layer = result["layer_name"]
        r2_dx = result["du_dx"]["r2"]
        r2_dy = result["du_dy"]["r2"]
        mse_dx = result["du_dx"]["mse"]
        mse_dy = result["du_dy"]["mse"]
        print(f"{layer:<12} {r2_dx:<12.4f} {r2_dy:<12.4f} {mse_dx:<14.6f} {mse_dy:<14.6f}")

    print("-" * 80)
    print(f"Total training time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    print(f"Average per probe: {total_time / (len(layer_names) * 2):.2f}s")

    # Save results to JSON
    results_file = output_dir / "first_derivative_probe_results.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "task": "First Derivative Probing",
                "configuration": {
                    "activation_file": activation_path,
                    "epochs": epochs,
                    "learning_rate": lr,
                    "device": device,
                    "n_points": len(coordinates),
                    "n_layers": len(layer_names),
                },
                "results": all_results,
                "total_training_time_s": total_time,
            },
            f,
            indent=2,
        )
    print(f"\n💾 Results saved to: {results_file}")

    # Save trained probes
    probes_file = output_dir / "first_derivative_probes.pt"
    probe_state_dicts = {}
    for layer_name, probes in all_probes.items():
        probe_state_dicts[layer_name] = {
            "du_dx": probes["du_dx"].weights.state_dict(),
            "du_dy": probes["du_dy"].weights.state_dict(),
        }
    torch.save(probe_state_dicts, probes_file)
    print(f"💾 Trained probes saved to: {probes_file}")

    # Analysis: Where do derivatives emerge?
    print("\n" + "=" * 80)
    print("ANALYSIS: Where Do First Derivatives Emerge?")
    print("=" * 80)

    threshold_high = 0.85  # High R² indicates explicit encoding
    threshold_moderate = 0.50  # Moderate R² indicates partial encoding

    for result in all_results:
        layer = result["layer_name"]
        r2_dx = result["du_dx"]["r2"]
        r2_dy = result["du_dy"]["r2"]

        status_dx = "🟢 EXPLICIT" if r2_dx > threshold_high else ("🟡 PARTIAL" if r2_dx > threshold_moderate else "🔴 WEAK")
        status_dy = "🟢 EXPLICIT" if r2_dy > threshold_high else ("🟡 PARTIAL" if r2_dy > threshold_moderate else "🔴 WEAK")

        print(f"\n{layer}:")
        print(f"  du/dx: R²={r2_dx:.4f} {status_dx}")
        print(f"  du/dy: R²={r2_dy:.4f} {status_dy}")

    print("\n✅ Task 1 Complete: First derivative probes trained for all layers!")
    print("=" * 80)


if __name__ == "__main__":
    main()
