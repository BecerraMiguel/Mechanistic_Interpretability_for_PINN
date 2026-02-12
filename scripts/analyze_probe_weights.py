"""
Day 11 Task 1: Extract and visualize probe weights for first-layer probes.

Analyzes the learned probe weights to understand which neurons contribute
most to predicting each derivative. For first-layer probes (layer_0),
each weight corresponds to a neuron that maps (x, y) -> activation.

Key analysis:
- Weight magnitude by neuron: identifies which neurons encode derivative info
- Sign patterns: positive vs negative weights indicate difference operations
- Comparison across derivatives: do du/dx and du/dy use different neurons?
- Weight evolution across layers: how does derivative encoding change?

Analysis Focus (from implementation plan):
- For du/dx probe on first-layer activations: visualize weight magnitude by neuron
- Look for neurons with high positive vs. negative weights (indicating difference)
- For second derivative probes: look for weights proportional to [1, -2, 1] pattern
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_probe_weights(probe_path: str) -> dict:
    """Load probe weights from checkpoint file."""
    data = torch.load(probe_path, weights_only=False, map_location="cpu")
    return data


def load_pinn_weights(model_path: str) -> dict:
    """Load PINN model weights to understand neuron structure."""
    ckpt = torch.load(model_path, weights_only=False, map_location="cpu")
    return ckpt["model_state_dict"]


def plot_probe_weight_bars(
    weights_dict: dict,
    layer_name: str,
    derivative_names: list,
    save_path: str,
    title_prefix: str = "",
):
    """
    Bar chart of probe weights for a specific layer, all derivatives side by side.

    Parameters
    ----------
    weights_dict : dict
        Nested dict: {derivative_name: {'weight': tensor, 'bias': tensor}}
    layer_name : str
        Name of the layer (e.g., 'layer_0')
    derivative_names : list
        List of derivative names to plot
    save_path : str
        Path to save figure
    title_prefix : str
        Optional prefix for figure title
    """
    n_derivs = len(derivative_names)
    fig, axes = plt.subplots(n_derivs, 1, figsize=(16, 4 * n_derivs), squeeze=False)

    for i, deriv_name in enumerate(derivative_names):
        ax = axes[i, 0]
        w = weights_dict[deriv_name]["weight"].numpy().flatten()
        n_neurons = len(w)

        # Color by sign: positive=blue, negative=red
        colors = ["#2196F3" if v >= 0 else "#F44336" for v in w]

        ax.bar(range(n_neurons), w, color=colors, width=0.8, edgecolor="none")
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_xlabel("Neuron Index", fontsize=11)
        ax.set_ylabel("Probe Weight", fontsize=11)

        # Format derivative name for display
        display_name = format_derivative_name(deriv_name)
        ax.set_title(
            f"{title_prefix}{display_name} probe weights ({layer_name})",
            fontsize=13,
            fontweight="bold",
        )

        # Add statistics
        abs_w = np.abs(w)
        stats_text = (
            f"max|w|={abs_w.max():.3f}, "
            f"mean|w|={abs_w.mean():.3f}, "
            f"std={w.std():.3f}, "
            f"bias={weights_dict[deriv_name]['bias'].numpy()[0]:.4f}"
        )
        ax.text(
            0.02,
            0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
        )

        # Mark top-5 neurons by absolute weight
        top_idx = np.argsort(abs_w)[-5:]
        for idx in top_idx:
            ax.annotate(
                f"n{idx}",
                xy=(idx, w[idx]),
                xytext=(idx, w[idx] + np.sign(w[idx]) * abs_w.max() * 0.1),
                fontsize=7,
                ha="center",
                color="black",
                fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_weight_magnitude_heatmap(
    all_probes: dict,
    layer_names: list,
    derivative_names: list,
    save_path: str,
    title: str = "Probe Weight Magnitudes",
):
    """
    Heatmap showing |weight| for each (layer, derivative, neuron) combination.

    Parameters
    ----------
    all_probes : dict
        Nested dict: {layer: {derivative: {'weight': tensor}}}
    layer_names : list
        Layer names
    derivative_names : list
        Derivative names
    save_path : str
        Path to save figure
    title : str
        Figure title
    """
    n_layers = len(layer_names)
    n_derivs = len(derivative_names)

    fig, axes = plt.subplots(
        n_derivs, 1, figsize=(16, 3 * n_derivs), squeeze=False
    )

    for d_idx, deriv_name in enumerate(derivative_names):
        ax = axes[d_idx, 0]
        # Collect weights across layers
        weight_matrix = []
        for layer_name in layer_names:
            w = all_probes[layer_name][deriv_name]["weight"].numpy().flatten()
            weight_matrix.append(np.abs(w))
        weight_matrix = np.array(weight_matrix)  # (n_layers, n_neurons)

        im = ax.imshow(
            weight_matrix,
            aspect="auto",
            cmap="YlOrRd",
            interpolation="nearest",
        )
        ax.set_xlabel("Neuron Index", fontsize=11)
        ax.set_ylabel("Layer", fontsize=11)
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels(layer_names)
        display_name = format_derivative_name(deriv_name)
        ax.set_title(f"|Probe Weights| for {display_name}", fontsize=12, fontweight="bold")
        plt.colorbar(im, ax=ax, label="|weight|")

    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_top_neurons_comparison(
    all_probes: dict,
    layer_name: str,
    derivative_names: list,
    save_path: str,
    top_k: int = 10,
):
    """
    Compare the top-K most important neurons across derivatives for a given layer.

    Parameters
    ----------
    all_probes : dict
        Probe weights dict
    layer_name : str
        Layer to analyze
    derivative_names : list
        Derivative names
    save_path : str
        Path to save
    top_k : int
        Number of top neurons to highlight
    """
    n_derivs = len(derivative_names)
    fig, axes = plt.subplots(1, n_derivs, figsize=(6 * n_derivs, 5))
    if n_derivs == 1:
        axes = [axes]

    for i, deriv_name in enumerate(derivative_names):
        ax = axes[i]
        w = all_probes[layer_name][deriv_name]["weight"].numpy().flatten()
        abs_w = np.abs(w)

        # Sort by magnitude
        sorted_idx = np.argsort(abs_w)[::-1]
        top_neurons = sorted_idx[:top_k]

        # Plot all neurons as scatter
        ax.scatter(
            range(len(w)),
            abs_w,
            s=10,
            alpha=0.3,
            color="gray",
            label="All neurons",
        )
        # Highlight top neurons
        ax.scatter(
            top_neurons,
            abs_w[top_neurons],
            s=60,
            color="#E91E63",
            zorder=5,
            label=f"Top {top_k}",
            edgecolors="black",
            linewidths=0.5,
        )

        for idx in top_neurons[:5]:
            sign = "+" if w[idx] >= 0 else "-"
            ax.annotate(
                f"n{idx}({sign})",
                xy=(idx, abs_w[idx]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold",
            )

        display_name = format_derivative_name(deriv_name)
        ax.set_title(f"{display_name} ({layer_name})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Neuron Index", fontsize=10)
        ax.set_ylabel("|Probe Weight|", fontsize=10)
        ax.legend(fontsize=9)

    plt.suptitle(
        f"Top-{top_k} Neurons for Derivative Prediction ({layer_name})",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_weight_sign_patterns(
    all_probes: dict,
    layer_name: str,
    save_path: str,
):
    """
    Visualize sign patterns of probe weights for du/dx vs du/dy.

    Neurons with positive weights for du/dx but negative for du/dy (or vice versa)
    suggest directional derivative computation.

    Parameters
    ----------
    all_probes : dict
        Probe weights dict
    layer_name : str
        Layer to analyze
    save_path : str
        Path to save
    """
    w_dx = all_probes[layer_name]["du_dx"]["weight"].numpy().flatten()
    w_dy = all_probes[layer_name]["du_dy"]["weight"].numpy().flatten()
    n_neurons = len(w_dx)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Scatter plot of du/dx vs du/dy weights
    ax = axes[0]
    ax.scatter(w_dx, w_dy, s=25, alpha=0.7, c=range(n_neurons), cmap="viridis")
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(x=0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("du/dx probe weight", fontsize=11)
    ax.set_ylabel("du/dy probe weight", fontsize=11)
    ax.set_title("du/dx vs du/dy Probe Weights", fontsize=12, fontweight="bold")

    # Annotate top neurons
    abs_total = np.abs(w_dx) + np.abs(w_dy)
    top5 = np.argsort(abs_total)[-5:]
    for idx in top5:
        ax.annotate(
            f"n{idx}",
            xy=(w_dx[idx], w_dy[idx]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
        )

    # Correlation
    corr = np.corrcoef(w_dx, w_dy)[0, 1]
    ax.text(
        0.02,
        0.95,
        f"Correlation: {corr:.3f}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )

    # Panel 2: Side-by-side sorted weights
    ax = axes[1]
    sort_idx = np.argsort(w_dx)
    ax.bar(
        np.arange(n_neurons) - 0.2,
        w_dx[sort_idx],
        width=0.4,
        label="du/dx",
        color="#2196F3",
        alpha=0.7,
    )
    ax.bar(
        np.arange(n_neurons) + 0.2,
        w_dy[sort_idx],
        width=0.4,
        label="du/dy",
        color="#FF9800",
        alpha=0.7,
    )
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("Neuron (sorted by du/dx weight)", fontsize=10)
    ax.set_ylabel("Probe Weight", fontsize=10)
    ax.set_title("Sorted Probe Weights Comparison", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

    # Panel 3: Weight ratio / specialization
    ax = axes[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        dx_dominance = np.abs(w_dx) / (np.abs(w_dx) + np.abs(w_dy) + 1e-10)
    sorted_dominance = np.sort(dx_dominance)[::-1]
    ax.bar(range(n_neurons), sorted_dominance, color="#9C27B0", alpha=0.7)
    ax.axhline(y=0.5, color="red", linewidth=1, linestyle="--", label="Equal weight")
    ax.set_xlabel("Neuron (sorted by dx-dominance)", fontsize=10)
    ax.set_ylabel("|w_dx| / (|w_dx| + |w_dy|)", fontsize=10)
    ax.set_title("Directional Specialization", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1)

    # Count specialized neurons
    dx_spec = np.sum(dx_dominance > 0.7)
    dy_spec = np.sum(dx_dominance < 0.3)
    mixed = n_neurons - dx_spec - dy_spec
    ax.text(
        0.98,
        0.95,
        f"dx-specialized: {dx_spec}\ndy-specialized: {dy_spec}\nmixed: {mixed}",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )

    plt.suptitle(
        f"Weight Sign Patterns: du/dx vs du/dy ({layer_name})",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_weight_evolution_across_layers(
    first_probes: dict,
    second_probes: dict,
    layer_names: list,
    save_path: str,
):
    """
    Show how probe weight distributions change across layers.

    Parameters
    ----------
    first_probes : dict
        First derivative probe weights
    second_probes : dict
        Second derivative probe weights
    layer_names : list
        Layer names
    save_path : str
        Path to save
    """
    all_derivs = ["du_dx", "du_dy", "d2u_dx2", "d2u_dy2", "laplacian"]
    display_names = [format_derivative_name(d) for d in all_derivs]
    n_layers = len(layer_names)

    fig, axes = plt.subplots(len(all_derivs), n_layers, figsize=(4 * n_layers, 3 * len(all_derivs)))

    for d_idx, deriv_name in enumerate(all_derivs):
        for l_idx, layer_name in enumerate(layer_names):
            ax = axes[d_idx, l_idx]
            if deriv_name in ("du_dx", "du_dy"):
                w = first_probes[layer_name][deriv_name]["weight"].numpy().flatten()
            else:
                w = second_probes[layer_name][deriv_name]["weight"].numpy().flatten()

            # Histogram of weights
            ax.hist(w, bins=20, color="#607D8B", alpha=0.7, edgecolor="white")
            ax.axvline(x=0, color="red", linewidth=0.8, linestyle="--")
            ax.set_title(f"{display_names[d_idx]}\n{layer_name}", fontsize=9)

            if l_idx == 0:
                ax.set_ylabel("Count", fontsize=8)
            if d_idx == len(all_derivs) - 1:
                ax.set_xlabel("Weight", fontsize=8)

            # Stats
            ax.text(
                0.95,
                0.95,
                f"std={w.std():.3f}",
                transform=ax.transAxes,
                fontsize=7,
                ha="right",
                va="top",
            )

    plt.suptitle(
        "Probe Weight Distributions Across Layers and Derivatives",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_neuron_pinn_weight_connection(
    pinn_weights: dict,
    probe_weights_layer0: dict,
    save_path: str,
):
    """
    Relate PINN first-layer weights (input->neuron) to probe weights (neuron->derivative).

    Each neuron in layer_0 computes: h_i = tanh(w_x * x + w_y * y + b)
    The probe then combines: du/dx_pred = sum_i(probe_w_i * h_i)

    Neurons with large |w_x| and large |probe_w| for du/dx suggest
    they're computing x-derivatives. Similarly for w_y and du/dy.

    Parameters
    ----------
    pinn_weights : dict
        PINN model state dict
    probe_weights_layer0 : dict
        Probe weights for layer_0
    save_path : str
        Path to save
    """
    # PINN first layer: maps (x, y) -> 64 neurons
    W_input = pinn_weights["layers.0.weight"].numpy()  # (64, 2)
    b_input = pinn_weights["layers.0.bias"].numpy()  # (64,)
    w_x = W_input[:, 0]  # weight from x input (64,)
    w_y = W_input[:, 1]  # weight from y input (64,)

    # Probe weights for du/dx and du/dy
    pw_dx = probe_weights_layer0["du_dx"]["weight"].numpy().flatten()  # (64,)
    pw_dy = probe_weights_layer0["du_dy"]["weight"].numpy().flatten()  # (64,)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel 1: PINN w_x vs probe du/dx weight
    ax = axes[0, 0]
    sc = ax.scatter(w_x, pw_dx, s=30, alpha=0.7, c=range(64), cmap="viridis")
    ax.set_xlabel("PINN w_x (input x weight)", fontsize=11)
    ax.set_ylabel("Probe du/dx weight", fontsize=11)
    ax.set_title("PINN x-weight vs du/dx probe weight", fontsize=12, fontweight="bold")
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(x=0, color="gray", linewidth=0.5, linestyle="--")
    corr1 = np.corrcoef(w_x, pw_dx)[0, 1]
    ax.text(
        0.02, 0.95, f"Corr: {corr1:.3f}",
        transform=ax.transAxes, fontsize=10, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )
    plt.colorbar(sc, ax=ax, label="Neuron idx")

    # Panel 2: PINN w_y vs probe du/dy weight
    ax = axes[0, 1]
    sc = ax.scatter(w_y, pw_dy, s=30, alpha=0.7, c=range(64), cmap="viridis")
    ax.set_xlabel("PINN w_y (input y weight)", fontsize=11)
    ax.set_ylabel("Probe du/dy weight", fontsize=11)
    ax.set_title("PINN y-weight vs du/dy probe weight", fontsize=12, fontweight="bold")
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(x=0, color="gray", linewidth=0.5, linestyle="--")
    corr2 = np.corrcoef(w_y, pw_dy)[0, 1]
    ax.text(
        0.02, 0.95, f"Corr: {corr2:.3f}",
        transform=ax.transAxes, fontsize=10, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )
    plt.colorbar(sc, ax=ax, label="Neuron idx")

    # Panel 3: PINN w_x vs probe du/dy weight (cross-check)
    ax = axes[1, 0]
    sc = ax.scatter(w_x, pw_dy, s=30, alpha=0.7, c=range(64), cmap="viridis")
    ax.set_xlabel("PINN w_x (input x weight)", fontsize=11)
    ax.set_ylabel("Probe du/dy weight", fontsize=11)
    ax.set_title("Cross-check: PINN x-weight vs du/dy probe", fontsize=12, fontweight="bold")
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(x=0, color="gray", linewidth=0.5, linestyle="--")
    corr3 = np.corrcoef(w_x, pw_dy)[0, 1]
    ax.text(
        0.02, 0.95, f"Corr: {corr3:.3f}",
        transform=ax.transAxes, fontsize=10, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )
    plt.colorbar(sc, ax=ax, label="Neuron idx")

    # Panel 4: PINN w_y vs probe du/dx weight (cross-check)
    ax = axes[1, 1]
    sc = ax.scatter(w_y, pw_dx, s=30, alpha=0.7, c=range(64), cmap="viridis")
    ax.set_xlabel("PINN w_y (input y weight)", fontsize=11)
    ax.set_ylabel("Probe du/dx weight", fontsize=11)
    ax.set_title("Cross-check: PINN y-weight vs du/dx probe", fontsize=12, fontweight="bold")
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(x=0, color="gray", linewidth=0.5, linestyle="--")
    corr4 = np.corrcoef(w_y, pw_dx)[0, 1]
    ax.text(
        0.02, 0.95, f"Corr: {corr4:.3f}",
        transform=ax.transAxes, fontsize=10, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )
    plt.colorbar(sc, ax=ax, label="Neuron idx")

    plt.suptitle(
        "PINN Input Weights vs Probe Weights (layer_0)\n"
        "If corr(w_x, probe_du/dx) is high, neurons sensitive to x also predict du/dx",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")

    return {
        "corr_wx_dudx": float(corr1),
        "corr_wy_dudy": float(corr2),
        "corr_wx_dudy": float(corr3),
        "corr_wy_dudx": float(corr4),
    }


def format_derivative_name(name: str) -> str:
    """Format derivative name for display."""
    mapping = {
        "du_dx": "du/dx",
        "du_dy": "du/dy",
        "d2u_dx2": "d2u/dx2",
        "d2u_dy2": "d2u/dy2",
        "laplacian": "Laplacian",
    }
    return mapping.get(name, name)


def generate_weight_statistics(
    first_probes: dict,
    second_probes: dict,
    layer_names: list,
    results_json: dict,
    save_path: str,
):
    """
    Generate comprehensive text report of probe weight statistics.

    Parameters
    ----------
    first_probes : dict
        First derivative probes
    second_probes : dict
        Second derivative probes
    layer_names : list
        Layer names
    results_json : dict
        Combined results JSON for R2 context
    save_path : str
        Path to save report
    """
    lines = []
    lines.append("=" * 80)
    lines.append("PROBE WEIGHT ANALYSIS REPORT - Day 11 Task 1")
    lines.append("=" * 80)
    lines.append("")

    all_derivs_first = ["du_dx", "du_dy"]
    all_derivs_second = ["d2u_dx2", "d2u_dy2", "laplacian"]

    # Section 1: First derivative probe weights
    lines.append("1. FIRST DERIVATIVE PROBE WEIGHTS")
    lines.append("-" * 50)

    for layer_name in layer_names:
        lines.append(f"\n  {layer_name}:")
        for deriv in all_derivs_first:
            w = first_probes[layer_name][deriv]["weight"].numpy().flatten()
            b = first_probes[layer_name][deriv]["bias"].numpy()[0]
            abs_w = np.abs(w)

            # Find top neurons
            top5_idx = np.argsort(abs_w)[-5:][::-1]
            top5_str = ", ".join(
                [f"n{i}({w[i]:+.4f})" for i in top5_idx]
            )

            lines.append(f"    {format_derivative_name(deriv)}:")
            lines.append(f"      mean|w|={abs_w.mean():.4f}, max|w|={abs_w.max():.4f}, "
                         f"std={w.std():.4f}, bias={b:.4f}")
            lines.append(f"      Top-5: {top5_str}")
            lines.append(f"      Positive: {np.sum(w > 0)}, Negative: {np.sum(w < 0)}, "
                         f"Near-zero (<0.01): {np.sum(abs_w < 0.01)}")

    # Section 2: Second derivative probe weights
    lines.append("\n\n2. SECOND DERIVATIVE PROBE WEIGHTS")
    lines.append("-" * 50)

    for layer_name in layer_names:
        lines.append(f"\n  {layer_name}:")
        for deriv in all_derivs_second:
            w = second_probes[layer_name][deriv]["weight"].numpy().flatten()
            b = second_probes[layer_name][deriv]["bias"].numpy()[0]
            abs_w = np.abs(w)

            top5_idx = np.argsort(abs_w)[-5:][::-1]
            top5_str = ", ".join(
                [f"n{i}({w[i]:+.4f})" for i in top5_idx]
            )

            lines.append(f"    {format_derivative_name(deriv)}:")
            lines.append(f"      mean|w|={abs_w.mean():.4f}, max|w|={abs_w.max():.4f}, "
                         f"std={w.std():.4f}, bias={b:.4f}")
            lines.append(f"      Top-5: {top5_str}")
            lines.append(f"      Positive: {np.sum(w > 0)}, Negative: {np.sum(w < 0)}, "
                         f"Near-zero (<0.01): {np.sum(abs_w < 0.01)}")

    # Section 3: Sparsity analysis
    lines.append("\n\n3. WEIGHT SPARSITY ANALYSIS")
    lines.append("-" * 50)
    lines.append("  How concentrated is derivative information?")
    lines.append("  (Fraction of total |weight| in top-K neurons)")
    lines.append("")

    for layer_name in ["layer_0", "layer_3"]:
        lines.append(f"  {layer_name}:")
        for deriv in all_derivs_first + all_derivs_second:
            if deriv in all_derivs_first:
                w = first_probes[layer_name][deriv]["weight"].numpy().flatten()
            else:
                w = second_probes[layer_name][deriv]["weight"].numpy().flatten()
            abs_w = np.abs(w)
            total = abs_w.sum()
            sorted_abs = np.sort(abs_w)[::-1]

            top5_frac = sorted_abs[:5].sum() / total if total > 0 else 0
            top10_frac = sorted_abs[:10].sum() / total if total > 0 else 0
            top20_frac = sorted_abs[:20].sum() / total if total > 0 else 0

            lines.append(
                f"    {format_derivative_name(deriv):12s}: "
                f"top5={top5_frac:.1%}, top10={top10_frac:.1%}, top20={top20_frac:.1%}"
            )
        lines.append("")

    # Section 4: Neuron overlap between du/dx and du/dy
    lines.append("\n4. NEURON OVERLAP: du/dx vs du/dy")
    lines.append("-" * 50)
    lines.append("  Do the same neurons encode both derivatives?")
    lines.append("")

    for layer_name in layer_names:
        w_dx = first_probes[layer_name]["du_dx"]["weight"].numpy().flatten()
        w_dy = first_probes[layer_name]["du_dy"]["weight"].numpy().flatten()

        top10_dx = set(np.argsort(np.abs(w_dx))[-10:])
        top10_dy = set(np.argsort(np.abs(w_dy))[-10:])
        overlap = top10_dx & top10_dy
        corr = np.corrcoef(w_dx, w_dy)[0, 1]

        lines.append(
            f"  {layer_name}: overlap={len(overlap)}/10, "
            f"shared neurons={sorted(overlap)}, corr={corr:.3f}"
        )

    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    report = "\n".join(lines)
    with open(save_path, "w") as f:
        f.write(report)
    print(f"  Saved: {save_path}")

    return report


def main():
    output_dir = PROJECT_ROOT / "outputs" / "day11_probe_weights"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Day 11 Task 1: Probe Weight Extraction and Visualization")
    print("=" * 60)

    # Load probe weights
    print("\n1. Loading probe weights...")
    first_probes = load_probe_weights(
        PROJECT_ROOT / "outputs" / "day9_task1" / "first_derivative_probes.pt"
    )
    second_probes = load_probe_weights(
        PROJECT_ROOT / "outputs" / "day9_task2" / "second_derivative_probes.pt"
    )

    # Load probe results (R2 values for context)
    with open(PROJECT_ROOT / "outputs" / "day9_task1" / "first_derivative_probe_results.json") as f:
        first_results = json.load(f)
    with open(PROJECT_ROOT / "outputs" / "day9_task2" / "second_derivative_probe_results.json") as f:
        second_results = json.load(f)

    # Load PINN model weights
    print("   Loading PINN model weights...")
    pinn_weights = load_pinn_weights(
        PROJECT_ROOT / "outputs" / "models" / "poisson_pinn_trained.pt"
    )

    layer_names = ["layer_0", "layer_1", "layer_2", "layer_3"]

    # Viz 1: Bar charts of first derivative probe weights per layer
    print("\n2. Generating probe weight bar charts...")
    for layer_name in layer_names:
        plot_probe_weight_bars(
            first_probes[layer_name],
            layer_name,
            ["du_dx", "du_dy"],
            str(output_dir / f"first_deriv_weights_{layer_name}.png"),
            title_prefix="",
        )

    # Viz 2: Bar chart of layer_0 first + second derivative probes
    print("\n3. Generating layer_0 all-derivatives bar chart...")
    # Combine for layer_0
    combined_layer0 = {**first_probes["layer_0"], **second_probes["layer_0"]}
    plot_probe_weight_bars(
        combined_layer0,
        "layer_0",
        ["du_dx", "du_dy", "d2u_dx2", "d2u_dy2", "laplacian"],
        str(output_dir / "all_deriv_weights_layer0.png"),
    )

    # Same for layer_3 (best performing)
    combined_layer3 = {**first_probes["layer_3"], **second_probes["layer_3"]}
    plot_probe_weight_bars(
        combined_layer3,
        "layer_3",
        ["du_dx", "du_dy", "d2u_dx2", "d2u_dy2", "laplacian"],
        str(output_dir / "all_deriv_weights_layer3.png"),
    )

    # Viz 3: Weight magnitude heatmaps
    print("\n4. Generating weight magnitude heatmaps...")
    # Combine all probes
    all_probes = {}
    for layer in layer_names:
        all_probes[layer] = {**first_probes[layer], **second_probes[layer]}

    plot_weight_magnitude_heatmap(
        all_probes,
        layer_names,
        ["du_dx", "du_dy", "d2u_dx2", "d2u_dy2", "laplacian"],
        str(output_dir / "weight_magnitude_heatmap.png"),
    )

    # Viz 4: Top neurons comparison
    print("\n5. Generating top-neuron comparisons...")
    plot_top_neurons_comparison(
        all_probes,
        "layer_0",
        ["du_dx", "du_dy"],
        str(output_dir / "top_neurons_layer0_first_deriv.png"),
    )
    plot_top_neurons_comparison(
        all_probes,
        "layer_3",
        ["du_dx", "du_dy", "d2u_dx2", "d2u_dy2", "laplacian"],
        str(output_dir / "top_neurons_layer3_all.png"),
    )

    # Viz 5: Sign patterns du/dx vs du/dy
    print("\n6. Generating weight sign patterns...")
    for layer_name in layer_names:
        plot_weight_sign_patterns(
            first_probes,
            layer_name,
            str(output_dir / f"sign_patterns_{layer_name}.png"),
        )

    # Viz 6: Weight evolution across layers
    print("\n7. Generating weight evolution across layers...")
    plot_weight_evolution_across_layers(
        first_probes,
        second_probes,
        layer_names,
        str(output_dir / "weight_distribution_evolution.png"),
    )

    # Viz 7: PINN input weights vs probe weights (layer_0)
    print("\n8. Analyzing PINN input weights vs probe weights...")
    correlations = plot_neuron_pinn_weight_connection(
        pinn_weights,
        first_probes["layer_0"],
        str(output_dir / "pinn_vs_probe_weights_layer0.png"),
    )
    print(f"   Correlations:")
    print(f"     corr(w_x, probe_du/dx) = {correlations['corr_wx_dudx']:.4f}")
    print(f"     corr(w_y, probe_du/dy) = {correlations['corr_wy_dudy']:.4f}")
    print(f"     corr(w_x, probe_du/dy) = {correlations['corr_wx_dudy']:.4f}  (cross)")
    print(f"     corr(w_y, probe_du/dx) = {correlations['corr_wy_dudx']:.4f}  (cross)")

    # Generate text report
    print("\n9. Generating weight statistics report...")
    report = generate_weight_statistics(
        first_probes,
        second_probes,
        layer_names,
        {"first": first_results, "second": second_results},
        str(output_dir / "probe_weight_report.txt"),
    )

    # Save correlations as JSON
    with open(output_dir / "pinn_probe_correlations.json", "w") as f:
        json.dump(correlations, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nKey findings from probe weight analysis:")
    print(f"  - PINN w_x <-> du/dx probe correlation: {correlations['corr_wx_dudx']:.4f}")
    print(f"  - PINN w_y <-> du/dy probe correlation: {correlations['corr_wy_dudy']:.4f}")

    # Weight sparsity for layer_0 du/dx
    w = first_probes["layer_0"]["du_dx"]["weight"].numpy().flatten()
    abs_w = np.abs(w)
    total = abs_w.sum()
    sorted_abs = np.sort(abs_w)[::-1]
    top10_frac = sorted_abs[:10].sum() / total
    print(f"  - Layer_0 du/dx: top 10 neurons carry {top10_frac:.1%} of total weight")

    w3 = first_probes["layer_3"]["du_dx"]["weight"].numpy().flatten()
    abs_w3 = np.abs(w3)
    total3 = abs_w3.sum()
    sorted_abs3 = np.sort(abs_w3)[::-1]
    top10_frac3 = sorted_abs3[:10].sum() / total3
    print(f"  - Layer_3 du/dx: top 10 neurons carry {top10_frac3:.1%} of total weight")

    files = list(output_dir.glob("*.png")) + list(output_dir.glob("*.txt")) + list(output_dir.glob("*.json"))
    print(f"\nTotal output files: {len(files)}")
    total_size = sum(f.stat().st_size for f in files)
    print(f"Total size: {total_size / 1024:.1f} KB")

    print("\nTask 1 complete!")


if __name__ == "__main__":
    main()
