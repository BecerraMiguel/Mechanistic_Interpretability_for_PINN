#!/usr/bin/env python3
"""
Extend probing to different MLP architectures.

Trains several MLP variants on the Poisson problem and runs the full
derivative probing pipeline on each, comparing R² patterns with the
baseline 4-layer/64-neuron/tanh model from Day 9.

Architecture variants tested:
  1. Shallow:  2 hidden layers, 64 neurons, tanh
  2. Deep:     6 hidden layers, 64 neurons, tanh
  3. Wide:     4 hidden layers, 128 neurons, tanh
  4. Narrow:   4 hidden layers, 32 neurons, tanh
  5. ReLU:     4 hidden layers, 64 neurons, relu

Baseline (from Day 9, not re-trained):
  - Standard:  4 hidden layers, 64 neurons, tanh  (0.99% L2 error)
"""

import json
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.interpretability.activation_store import ActivationStore
from src.interpretability.probing import LinearProbe
from src.models.mlp import MLP
from src.problems.poisson import PoissonProblem
from src.training.trainer import PINNTrainer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRAINING_EPOCHS = 3000
N_INTERIOR = 1000
N_BOUNDARY = 25  # per edge
GRID_RESOLUTION = 50  # 50x50 = 2500 points for probing
PROBE_EPOCHS = 500
PROBE_LR = 1e-3
OUTPUT_DIR = "outputs/day13_architecture_comparison"

ARCHITECTURE_VARIANTS = {
    "shallow_2L": {"hidden_dims": [64, 64], "activation": "tanh"},
    "deep_6L": {"hidden_dims": [64, 64, 64, 64, 64, 64], "activation": "tanh"},
    "wide_128": {"hidden_dims": [128, 128, 128, 128], "activation": "tanh"},
    "narrow_32": {"hidden_dims": [32, 32, 32, 32], "activation": "tanh"},
    "relu_4L": {"hidden_dims": [64, 64, 64, 64], "activation": "relu"},
}

DERIVATIVE_TARGETS = ["du_dx", "du_dy", "d2u_dx2", "d2u_dy2", "laplacian"]


def compute_ground_truth_derivatives(problem, coords_tensor):
    """Compute all 5 derivative targets from analytical solution."""
    targets = {}
    targets["du_dx"] = problem.analytical_derivative_du_dx(coords_tensor)
    targets["du_dy"] = problem.analytical_derivative_du_dy(coords_tensor)
    targets["d2u_dx2"] = problem.analytical_derivative_d2u_dx2(coords_tensor)
    targets["d2u_dy2"] = problem.analytical_derivative_d2u_dy2(coords_tensor)
    targets["laplacian"] = problem.analytical_laplacian(coords_tensor)
    return targets


def train_model(name, config, problem):
    """Train a single MLP variant and return the model + training info."""
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"  hidden_dims={config['hidden_dims']}, activation={config['activation']}")
    print(f"  epochs={TRAINING_EPOCHS}, interior={N_INTERIOR}, boundary={N_BOUNDARY}")
    print(f"{'='*60}")

    model = MLP(
        input_dim=2,
        hidden_dims=config["hidden_dims"],
        output_dim=1,
        activation=config["activation"],
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    trainer = PINNTrainer(
        model=model,
        problem=problem,
        optimizer=optimizer,
        n_interior=N_INTERIOR,
        n_boundary=N_BOUNDARY,
        device="cpu",
    )

    t0 = time.time()
    history = trainer.train(
        n_epochs=TRAINING_EPOCHS,
        validate_every=500,
        print_every=1000,
        resample_every=500,
    )
    train_time = time.time() - t0

    # Get final error
    final_error = trainer.validate()
    final_loss = history["loss_total"][-1]

    print(f"  Final loss: {final_loss:.6f}")
    print(f"  Relative L2 error: {final_error*100:.2f}%")
    print(f"  Training time: {train_time:.1f}s")

    return model, {
        "name": name,
        "hidden_dims": config["hidden_dims"],
        "activation": config["activation"],
        "n_params": n_params,
        "final_loss": float(final_loss),
        "relative_l2_error": float(final_error),
        "training_time_s": train_time,
    }


def extract_and_probe(model, name, problem, ground_truth, output_dir):
    """Extract activations and run probes for all derivatives on all layers."""
    # Extract activations
    h5_path = os.path.join(output_dir, f"activations_{name}.h5")
    store = ActivationStore(h5_path)
    store.extract_on_grid(
        model=model,
        grid_resolution=GRID_RESOLUTION,
        domain_bounds=((0.0, 1.0), (0.0, 1.0)),
        batch_size=1000,
        device="cpu",
    )

    metadata = store.get_metadata()
    layer_names = metadata["layer_names"]
    coordinates = store.load_coordinates()

    # Compute ground-truth targets using same coordinates
    coords_tensor = torch.tensor(coordinates, dtype=torch.float32)
    targets = compute_ground_truth_derivatives(problem, coords_tensor)

    # Run probes for each (layer, derivative) pair
    results = {}
    print(f"\n  Probing {name} ({len(layer_names)} layers)...")

    for layer_name in layer_names:
        activations_np = store.load_layer(layer_name)
        activations = torch.tensor(activations_np, dtype=torch.float32)
        hidden_dim = activations.shape[1]

        results[layer_name] = {}

        for deriv_name in DERIVATIVE_TARGETS:
            target = targets[deriv_name]
            if isinstance(target, torch.Tensor):
                target_t = target.float()
            else:
                target_t = torch.tensor(target, dtype=torch.float32)

            if target_t.dim() == 1:
                target_t = target_t.unsqueeze(1)

            probe = LinearProbe(input_dim=hidden_dim, output_dim=1)
            probe.fit(activations, target_t, epochs=PROBE_EPOCHS, lr=PROBE_LR)
            scores = probe.score(activations, target_t)

            results[layer_name][deriv_name] = {
                "r_squared": float(scores["r_squared"]),
                "mse": float(scores["mse"]),
            }

        # Print summary for this layer
        r2_str = "  ".join(
            f"{d[:6]}={results[layer_name][d]['r_squared']:+.3f}"
            for d in DERIVATIVE_TARGETS
        )
        print(f"    {layer_name}: {r2_str}")

    return results


def load_baseline_results():
    """Load the baseline Day 9 probe results (4L/64/tanh, well-trained)."""
    baseline = {}
    base_dir = os.path.dirname(__file__)
    proj_root = os.path.join(base_dir, "..")

    first_path = os.path.join(proj_root, "outputs/day9_task1/first_derivative_probe_results.json")
    second_path = os.path.join(proj_root, "outputs/day9_task2/second_derivative_probe_results.json")

    with open(first_path) as f:
        first = json.load(f)
    with open(second_path) as f:
        second = json.load(f)

    for entry in first["results"]:
        layer = entry["layer_name"]
        baseline[layer] = {
            "du_dx": {"r_squared": entry["du_dx"]["r2"]},
            "du_dy": {"r_squared": entry["du_dy"]["r2"]},
        }

    for entry in second["results"]:
        layer = entry["layer_name"]
        baseline[layer]["d2u_dx2"] = {"r_squared": entry["d2u_dx2"]["r2"]}
        baseline[layer]["d2u_dy2"] = {"r_squared": entry["d2u_dy2"]["r2"]}
        baseline[layer]["laplacian"] = {"r_squared": entry["laplacian"]["r2"]}

    return baseline


def generate_comparison_table(all_results, output_dir):
    """Generate a text comparison table of peak R² across architectures."""
    lines = []
    lines.append("=" * 90)
    lines.append("ARCHITECTURE COMPARISON: Peak R² for Each Derivative")
    lines.append("=" * 90)
    lines.append("")

    header = f"{'Architecture':<20} {'du/dx':>8} {'du/dy':>8} {'d2u/dx2':>8} {'d2u/dy2':>8} {'Lap':>8} {'Err%':>8}"
    lines.append(header)
    lines.append("-" * 90)

    for arch_name, info in all_results.items():
        probe_results = info["probe_results"]
        peak_r2 = {}
        for deriv in DERIVATIVE_TARGETS:
            best = max(
                probe_results[layer][deriv]["r_squared"]
                for layer in probe_results
            )
            peak_r2[deriv] = best

        err_str = f"{info['training_info']['relative_l2_error']*100:.1f}" if "training_info" in info else "0.99"
        row = (
            f"{arch_name:<20} "
            f"{peak_r2['du_dx']:>8.4f} "
            f"{peak_r2['du_dy']:>8.4f} "
            f"{peak_r2['d2u_dx2']:>8.4f} "
            f"{peak_r2['d2u_dy2']:>8.4f} "
            f"{peak_r2['laplacian']:>8.4f} "
            f"{err_str:>8}"
        )
        lines.append(row)

    lines.append("-" * 90)
    lines.append("")

    # Analysis
    lines.append("KEY OBSERVATIONS:")
    lines.append("")

    # Check if two-stage pattern holds for all architectures
    for arch_name, info in all_results.items():
        probe_results = info["probe_results"]
        # Get peak R² for first vs second derivatives
        first_r2 = []
        second_r2 = []
        for layer in probe_results:
            first_r2.append(probe_results[layer]["du_dx"]["r_squared"])
            first_r2.append(probe_results[layer]["du_dy"]["r_squared"])
            second_r2.append(probe_results[layer]["d2u_dx2"]["r_squared"])
            second_r2.append(probe_results[layer]["d2u_dy2"]["r_squared"])

        avg_first = np.mean(first_r2)
        avg_second = np.mean(second_r2)
        gap = avg_first - avg_second
        pattern = "YES" if gap > 0.1 else "WEAK" if gap > 0 else "NO"
        lines.append(
            f"  {arch_name:<20}: "
            f"avg 1st={avg_first:.3f}, avg 2nd={avg_second:.3f}, "
            f"gap={gap:.3f}  -> Two-stage pattern: {pattern}"
        )

    lines.append("")
    text = "\n".join(lines)
    print(text)

    with open(os.path.join(output_dir, "architecture_comparison.txt"), "w") as f:
        f.write(text)

    return text


def generate_comparison_figure(all_results, output_dir):
    """Generate a multi-panel comparison figure."""
    arch_names = list(all_results.keys())
    n_arch = len(arch_names)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Peak R² bar chart for each architecture
    ax = axes[0]
    x = np.arange(n_arch)
    width = 0.15

    for i, deriv in enumerate(DERIVATIVE_TARGETS):
        peak_vals = []
        for arch in arch_names:
            probe_results = all_results[arch]["probe_results"]
            best = max(
                probe_results[layer][deriv]["r_squared"]
                for layer in probe_results
            )
            peak_vals.append(best)

        label_map = {
            "du_dx": "du/dx",
            "du_dy": "du/dy",
            "d2u_dx2": "d2u/dx2",
            "d2u_dy2": "d2u/dy2",
            "laplacian": "Laplacian",
        }
        ax.bar(x + i * width, peak_vals, width, label=label_map[deriv], alpha=0.85)

    ax.set_xlabel("Architecture")
    ax.set_ylabel("Peak R-squared")
    ax.set_title("Peak Derivative R-squared by Architecture")
    ax.set_xticks(x + width * 2)
    short_names = []
    for name in arch_names:
        if name == "baseline_4L64_tanh":
            short_names.append("Baseline\n4L/64/tanh")
        elif name == "shallow_2L":
            short_names.append("Shallow\n2L/64/tanh")
        elif name == "deep_6L":
            short_names.append("Deep\n6L/64/tanh")
        elif name == "wide_128":
            short_names.append("Wide\n4L/128/tanh")
        elif name == "narrow_32":
            short_names.append("Narrow\n4L/32/tanh")
        elif name == "relu_4L":
            short_names.append("ReLU\n4L/64/relu")
        else:
            short_names.append(name)
    ax.set_xticklabels(short_names, fontsize=8)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(-0.5, 1.1)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

    # Panel 2: Two-stage gap (first - second derivative R²) per architecture
    ax = axes[1]
    gaps = []
    first_avgs = []
    second_avgs = []

    for arch in arch_names:
        probe_results = all_results[arch]["probe_results"]
        layers = list(probe_results.keys())
        # Use final layer R²
        final_layer = layers[-1]
        first_avg = np.mean([
            probe_results[final_layer]["du_dx"]["r_squared"],
            probe_results[final_layer]["du_dy"]["r_squared"],
        ])
        second_avg = np.mean([
            probe_results[final_layer]["d2u_dx2"]["r_squared"],
            probe_results[final_layer]["d2u_dy2"]["r_squared"],
        ])
        first_avgs.append(first_avg)
        second_avgs.append(second_avg)
        gaps.append(first_avg - second_avg)

    bar_width = 0.35
    ax.bar(x - bar_width / 2, first_avgs, bar_width, label="1st deriv (final layer)", color="#2196F3")
    ax.bar(x + bar_width / 2, second_avgs, bar_width, label="2nd deriv (final layer)", color="#FF5722")

    ax.set_xlabel("Architecture")
    ax.set_ylabel("R-squared (final layer)")
    ax.set_title("First vs Second Derivative Encoding (Final Layer)")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=8)
    ax.legend(fontsize=9)
    ax.set_ylim(-0.5, 1.1)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "architecture_comparison.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved comparison figure: {fig_path}")

    # Layer-by-layer R² progression for each architecture (one plot per derivative)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    deriv_groups = [
        ("du_dx", "du/dx (1st order)"),
        ("d2u_dx2", "d2u/dx2 (2nd order)"),
        ("laplacian", "Laplacian"),
    ]

    colors = plt.cm.tab10(np.linspace(0, 1, n_arch))

    for ax, (deriv_key, deriv_label) in zip(axes, deriv_groups):
        for idx, arch in enumerate(arch_names):
            probe_results = all_results[arch]["probe_results"]
            layers = sorted(probe_results.keys())
            r2_vals = [probe_results[l][deriv_key]["r_squared"] for l in layers]
            layer_indices = list(range(len(layers)))
            ax.plot(
                layer_indices, r2_vals,
                marker="o", linewidth=2, markersize=6,
                color=colors[idx],
                label=short_names[idx].replace("\n", " "),
            )

        ax.set_xlabel("Layer Index")
        ax.set_ylabel("R-squared")
        ax.set_title(deriv_label)
        ax.legend(fontsize=7, loc="lower right")
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
        ax.set_ylim(-0.6, 1.05)

    plt.tight_layout()
    fig_path2 = os.path.join(output_dir, "layer_progression_by_architecture.png")
    plt.savefig(fig_path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved layer progression figure: {fig_path2}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    problem = PoissonProblem()
    torch.manual_seed(42)

    # ---- Load baseline results from Day 9 ----
    print("Loading baseline results (4L/64/tanh, well-trained)...")
    baseline_probes = load_baseline_results()
    all_results = {
        "baseline_4L64_tanh": {
            "probe_results": baseline_probes,
            "training_info": {
                "name": "baseline_4L64_tanh",
                "hidden_dims": [64, 64, 64, 64],
                "activation": "tanh",
                "n_params": 12737,
                "relative_l2_error": 0.00995,
                "training_time_s": 0,
            },
        }
    }

    # ---- Train and probe each variant ----
    total_start = time.time()

    for variant_name, config in ARCHITECTURE_VARIANTS.items():
        torch.manual_seed(42)
        np.random.seed(42)

        model, train_info = train_model(variant_name, config, problem)
        probe_results = extract_and_probe(model, variant_name, problem, None, OUTPUT_DIR)

        all_results[variant_name] = {
            "probe_results": probe_results,
            "training_info": train_info,
        }

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"Total time for all variants: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"{'='*60}")

    # ---- Save all results to JSON ----
    json_path = os.path.join(OUTPUT_DIR, "all_probe_results.json")
    # Make JSON-serializable
    json_data = {}
    for arch, info in all_results.items():
        json_data[arch] = {
            "training_info": info["training_info"],
            "probe_results": info["probe_results"],
        }

    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"\nSaved all results: {json_path}")

    # ---- Generate comparison table and figures ----
    generate_comparison_table(all_results, OUTPUT_DIR)
    generate_comparison_figure(all_results, OUTPUT_DIR)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
