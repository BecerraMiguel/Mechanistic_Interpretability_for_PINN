"""
Train linear probes for second derivatives (d2u/dx2, d2u/dy2) and Laplacian at each layer.

Day 9, Task 2: Layer-wise Derivative Probing (Second Derivatives)
This script trains probes to detect whether second-order derivative information
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


def compute_ground_truth_second_derivatives(coordinates: torch.Tensor, problem: PoissonProblem):
    """
    Compute analytical second derivatives for probing targets.

    Parameters
    ----------
    coordinates : torch.Tensor
        Shape (N, 2) - input coordinates
    problem : PoissonProblem
        Problem instance with analytical derivative methods

    Returns
    -------
    d2u_dx2 : np.ndarray
        Shape (N, 1) - ground-truth ∂²u/∂x²
    d2u_dy2 : np.ndarray
        Shape (N, 1) - ground-truth ∂²u/∂y²
    laplacian : np.ndarray
        Shape (N, 1) - ground-truth ∇²u = ∂²u/∂x² + ∂²u/∂y²
    """
    print("\n🧮 Computing ground-truth second derivatives...")

    # Compute analytical second derivatives
    d2u_dx2 = problem.analytical_derivative_d2u_dx2(coordinates)  # (N, 1)
    d2u_dy2 = problem.analytical_derivative_d2u_dy2(coordinates)  # (N, 1)
    laplacian = problem.analytical_laplacian(coordinates)  # (N, 1)

    # Convert to numpy for sklearn-like interface
    d2u_dx2_np = d2u_dx2.detach().cpu().numpy()
    d2u_dy2_np = d2u_dy2.detach().cpu().numpy()
    laplacian_np = laplacian.detach().cpu().numpy()

    print(f"   ✓ d2u/dx2: shape={d2u_dx2_np.shape}, mean={d2u_dx2_np.mean():.4f}, std={d2u_dx2_np.std():.4f}")
    print(f"   ✓ d2u/dy2: shape={d2u_dy2_np.shape}, mean={d2u_dy2_np.mean():.4f}, std={d2u_dy2_np.std():.4f}")
    print(f"   ✓ Laplacian: shape={laplacian_np.shape}, mean={laplacian_np.mean():.4f}, std={laplacian_np.std():.4f}")

    # Verify mathematical relationship: ∇²u = ∂²u/∂x² + ∂²u/∂y²
    computed_laplacian = d2u_dx2_np + d2u_dy2_np
    diff = np.abs(laplacian_np - computed_laplacian).max()
    print(f"   ✓ Verification: max|∇²u - (∂²u/∂x² + ∂²u/∂y²)| = {diff:.2e} (should be ~0)")

    return d2u_dx2_np, d2u_dy2_np, laplacian_np


def train_probes_for_layer(
    layer_name: str,
    activations: np.ndarray,
    d2u_dx2: np.ndarray,
    d2u_dy2: np.ndarray,
    laplacian: np.ndarray,
    epochs: int = 1000,
    lr: float = 1e-3,
    device: str = "cpu",
):
    """
    Train three probes (d2u/dx2, d2u/dy2, Laplacian) for a single layer.

    Parameters
    ----------
    layer_name : str
        Name of layer (e.g., "layer_0")
    activations : np.ndarray
        Shape (N, hidden_dim) - layer activations
    d2u_dx2 : np.ndarray
        Shape (N, 1) - ground-truth ∂²u/∂x²
    d2u_dy2 : np.ndarray
        Shape (N, 1) - ground-truth ∂²u/∂y²
    laplacian : np.ndarray
        Shape (N, 1) - ground-truth ∇²u
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
    d2u_dx2_tensor = torch.tensor(d2u_dx2, dtype=torch.float32)
    d2u_dy2_tensor = torch.tensor(d2u_dy2, dtype=torch.float32)
    laplacian_tensor = torch.tensor(laplacian, dtype=torch.float32)

    # Initialize probes
    probe_d2u_dx2 = LinearProbe(input_dim=input_dim, output_dim=output_dim)
    probe_d2u_dy2 = LinearProbe(input_dim=input_dim, output_dim=output_dim)
    probe_laplacian = LinearProbe(input_dim=input_dim, output_dim=output_dim)

    # Move to device
    probe_d2u_dx2.to(device)
    probe_d2u_dy2.to(device)
    probe_laplacian.to(device)

    # Train probe for d2u/dx2
    print(f"   Training d2u/dx2 probe...")
    start_time = time.time()
    probe_d2u_dx2.fit(activations_tensor, d2u_dx2_tensor, epochs=epochs, lr=lr, verbose=False)
    d2u_dx2_time = time.time() - start_time

    # Train probe for d2u/dy2
    print(f"   Training d2u/dy2 probe...")
    start_time = time.time()
    probe_d2u_dy2.fit(activations_tensor, d2u_dy2_tensor, epochs=epochs, lr=lr, verbose=False)
    d2u_dy2_time = time.time() - start_time

    # Train probe for Laplacian
    print(f"   Training Laplacian probe...")
    start_time = time.time()
    probe_laplacian.fit(activations_tensor, laplacian_tensor, epochs=epochs, lr=lr, verbose=False)
    laplacian_time = time.time() - start_time

    # Evaluate probes
    scores_d2u_dx2 = probe_d2u_dx2.score(activations_tensor, d2u_dx2_tensor)
    scores_d2u_dy2 = probe_d2u_dy2.score(activations_tensor, d2u_dy2_tensor)
    scores_laplacian = probe_laplacian.score(activations_tensor, laplacian_tensor)

    print(f"   ✓ d2u/dx2: R²={scores_d2u_dx2['r_squared']:.4f}, MSE={scores_d2u_dx2['mse']:.6f} (time: {d2u_dx2_time:.2f}s)")
    print(f"   ✓ d2u/dy2: R²={scores_d2u_dy2['r_squared']:.4f}, MSE={scores_d2u_dy2['mse']:.6f} (time: {d2u_dy2_time:.2f}s)")
    print(f"   ✓ Laplacian: R²={scores_laplacian['r_squared']:.4f}, MSE={scores_laplacian['mse']:.6f} (time: {laplacian_time:.2f}s)")

    # Store results
    results = {
        "layer_name": layer_name,
        "d2u_dx2": {
            "r2": float(scores_d2u_dx2["r_squared"]),
            "mse": float(scores_d2u_dx2["mse"]),
            "explained_variance": float(scores_d2u_dx2["explained_variance"]),
            "training_time_s": d2u_dx2_time,
        },
        "d2u_dy2": {
            "r2": float(scores_d2u_dy2["r_squared"]),
            "mse": float(scores_d2u_dy2["mse"]),
            "explained_variance": float(scores_d2u_dy2["explained_variance"]),
            "training_time_s": d2u_dy2_time,
        },
        "laplacian": {
            "r2": float(scores_laplacian["r_squared"]),
            "mse": float(scores_laplacian["mse"]),
            "explained_variance": float(scores_laplacian["explained_variance"]),
            "training_time_s": laplacian_time,
        },
    }

    return results, probe_d2u_dx2, probe_d2u_dy2, probe_laplacian


def main():
    """Main script for training second derivative probes."""
    print("=" * 80)
    print("DAY 9, TASK 2: Train Probes for Second Derivatives (d2u/dx2, d2u/dy2, ∇²u)")
    print("=" * 80)

    # Configuration
    activation_path = "data/activations/poisson_mlp_100x100.h5"
    output_dir = Path("outputs/day9_task2")
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

    # Compute ground-truth second derivatives
    d2u_dx2, d2u_dy2, laplacian = compute_ground_truth_second_derivatives(coordinates, problem)

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
        layer_results, probe_d2u_dx2, probe_d2u_dy2, probe_laplacian = train_probes_for_layer(
            layer_name=layer_name,
            activations=activations,
            d2u_dx2=d2u_dx2,
            d2u_dy2=d2u_dy2,
            laplacian=laplacian,
            epochs=epochs,
            lr=lr,
            device=device,
        )

        all_results.append(layer_results)
        all_probes[layer_name] = {
            "d2u_dx2": probe_d2u_dx2,
            "d2u_dy2": probe_d2u_dy2,
            "laplacian": probe_laplacian,
        }

    total_time = time.time() - total_start_time

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY: Second Derivative Probing Results")
    print("=" * 80)
    print(f"\n{'Layer':<12} {'d2u/dx2 R²':<14} {'d2u/dy2 R²':<14} {'Laplacian R²':<14}")
    print("-" * 80)

    for result in all_results:
        layer = result["layer_name"]
        r2_dx2 = result["d2u_dx2"]["r2"]
        r2_dy2 = result["d2u_dy2"]["r2"]
        r2_lap = result["laplacian"]["r2"]
        print(f"{layer:<12} {r2_dx2:<14.4f} {r2_dy2:<14.4f} {r2_lap:<14.4f}")

    print("-" * 80)
    print(f"Total training time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    print(f"Average per probe: {total_time / (len(layer_names) * 3):.2f}s")

    # Save results to JSON
    results_file = output_dir / "second_derivative_probe_results.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "task": "Second Derivative Probing",
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
    probes_file = output_dir / "second_derivative_probes.pt"
    probe_state_dicts = {}
    for layer_name, probes in all_probes.items():
        probe_state_dicts[layer_name] = {
            "d2u_dx2": probes["d2u_dx2"].weights.state_dict(),
            "d2u_dy2": probes["d2u_dy2"].weights.state_dict(),
            "laplacian": probes["laplacian"].weights.state_dict(),
        }
    torch.save(probe_state_dicts, probes_file)
    print(f"💾 Trained probes saved to: {probes_file}")

    # Analysis: Where do second derivatives emerge?
    print("\n" + "=" * 80)
    print("ANALYSIS: Where Do Second Derivatives Emerge?")
    print("=" * 80)

    threshold_high = 0.85  # High R² indicates explicit encoding
    threshold_moderate = 0.50  # Moderate R² indicates partial encoding

    print("\n" + "Expected Pattern (from PDF):")
    print("  - First derivatives (∂u/∂x, ∂u/∂y) emerge in early layers")
    print("  - Second derivatives emerge later (composition of first derivatives)")
    print("  - Laplacian (∇²u = ∂²u/∂x² + ∂²u/∂y²) emerges last")

    for result in all_results:
        layer = result["layer_name"]
        r2_dx2 = result["d2u_dx2"]["r2"]
        r2_dy2 = result["d2u_dy2"]["r2"]
        r2_lap = result["laplacian"]["r2"]

        status_dx2 = "🟢 EXPLICIT" if r2_dx2 > threshold_high else ("🟡 PARTIAL" if r2_dx2 > threshold_moderate else "🔴 WEAK")
        status_dy2 = "🟢 EXPLICIT" if r2_dy2 > threshold_high else ("🟡 PARTIAL" if r2_dy2 > threshold_moderate else "🔴 WEAK")
        status_lap = "🟢 EXPLICIT" if r2_lap > threshold_high else ("🟡 PARTIAL" if r2_lap > threshold_moderate else "🔴 WEAK")

        print(f"\n{layer}:")
        print(f"  d2u/dx2:   R²={r2_dx2:.4f} {status_dx2}")
        print(f"  d2u/dy2:   R²={r2_dy2:.4f} {status_dy2}")
        print(f"  Laplacian: R²={r2_lap:.4f} {status_lap}")

    # Compare with first derivatives (from Task 1)
    print("\n" + "=" * 80)
    print("COMPARISON: First vs Second Derivatives")
    print("=" * 80)

    # Try to load Task 1 results for comparison
    task1_results_file = Path("outputs/day9_task1/first_derivative_probe_results.json")
    if task1_results_file.exists():
        with open(task1_results_file, "r") as f:
            task1_data = json.load(f)

        print(f"\n{'Layer':<12} {'1st deriv':<12} {'2nd deriv':<12} {'Laplacian':<12} {'Pattern':<30}")
        print("-" * 80)

        for i, result in enumerate(all_results):
            layer = result["layer_name"]
            task1_result = task1_data["results"][i]

            # Average R² for first derivatives
            r2_first = (task1_result["du_dx"]["r2"] + task1_result["du_dy"]["r2"]) / 2
            # Average R² for second derivatives
            r2_second = (result["d2u_dx2"]["r2"] + result["d2u_dy2"]["r2"]) / 2
            # Laplacian R²
            r2_lap = result["laplacian"]["r2"]

            # Determine pattern
            if r2_lap > r2_second > r2_first:
                pattern = "Laplacian > 2nd > 1st (unusual)"
            elif r2_second > r2_lap > r2_first:
                pattern = "2nd > Laplacian > 1st (unusual)"
            elif r2_first > r2_second > r2_lap:
                pattern = "1st > 2nd > Laplacian (expected)"
            elif r2_first > r2_lap > r2_second:
                pattern = "1st > Laplacian > 2nd"
            elif r2_lap > r2_first > r2_second:
                pattern = "Laplacian > 1st > 2nd"
            else:
                pattern = "Mixed pattern"

            print(f"{layer:<12} {r2_first:<12.4f} {r2_second:<12.4f} {r2_lap:<12.4f} {pattern:<30}")
    else:
        print("\n⚠️  Task 1 results not found. Run Task 1 first for comparison.")

    print("\n✅ Task 2 Complete: Second derivative probes trained for all layers!")
    print("=" * 80)


if __name__ == "__main__":
    main()
