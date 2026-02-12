"""
Day 11 Task 2: Analyze whether probe weights show finite-difference-like patterns.

Core idea:
  Layer_0 neurons compute h_i = tanh(w_x_i * x + w_y_i * y + b_i).
  tanh is a soft step function, so each neuron is roughly a shifted sigmoid.

  If two neurons have similar (w_x, w_y) but different biases, they produce
  shifted versions of the same function. Combining them with opposite probe
  weights (+p, -p) creates a difference operation:
    p * tanh(w*x + b1) - p * tanh(w*x + b2) ≈ derivative approximation

  For second derivatives, triplets with [+1, -2, +1]-like probe weights
  and evenly spaced biases approximate central difference stencils.

Analysis steps:
  1. Identify neuron pairs with similar input weights (same orientation)
  2. Check if paired neurons have opposite-sign probe weights
  3. Compute effective "stencil spacing" from bias differences
  4. For second derivatives, look for [1, -2, 1] triplet patterns
  5. Quantify how well the actual patterns match ideal finite differences
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from itertools import combinations

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_all_data():
    """Load PINN weights and probe weights."""
    # PINN weights
    ckpt = torch.load(
        PROJECT_ROOT / "outputs" / "models" / "poisson_pinn_trained.pt",
        weights_only=False,
        map_location="cpu",
    )
    pinn_state = ckpt["model_state_dict"]

    # Probe weights
    first_probes = torch.load(
        PROJECT_ROOT / "outputs" / "day9_task1" / "first_derivative_probes.pt",
        weights_only=False,
        map_location="cpu",
    )
    second_probes = torch.load(
        PROJECT_ROOT / "outputs" / "day9_task2" / "second_derivative_probes.pt",
        weights_only=False,
        map_location="cpu",
    )

    return pinn_state, first_probes, second_probes


def compute_neuron_similarity(W_input: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between neuron input weight vectors.

    Parameters
    ----------
    W_input : np.ndarray
        Input weight matrix of shape (n_neurons, input_dim).

    Returns
    -------
    np.ndarray
        Similarity matrix of shape (n_neurons, n_neurons).
    """
    norms = np.linalg.norm(W_input, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    W_normalized = W_input / norms
    return W_normalized @ W_normalized.T


def find_neuron_pairs(
    W_input: np.ndarray,
    biases: np.ndarray,
    probe_weights: np.ndarray,
    cos_sim_threshold: float = 0.9,
):
    """
    Find neuron pairs that could form finite-difference stencils.

    Criteria for a difference pair:
    1. Similar input weight direction (high cosine similarity)
    2. Different biases (shifted versions of same function)
    3. Opposite-sign probe weights (difference operation)

    Parameters
    ----------
    W_input : np.ndarray
        PINN input weights, shape (n_neurons, input_dim)
    biases : np.ndarray
        PINN biases, shape (n_neurons,)
    probe_weights : np.ndarray
        Probe weights, shape (n_neurons,)
    cos_sim_threshold : float
        Minimum cosine similarity for a pair

    Returns
    -------
    list of dict
        Each dict describes a candidate pair.
    """
    n_neurons = W_input.shape[0]
    cos_sim = compute_neuron_similarity(W_input)
    norms = np.linalg.norm(W_input, axis=1)

    pairs = []
    for i, j in combinations(range(n_neurons), 2):
        sim = cos_sim[i, j]
        if abs(sim) < cos_sim_threshold:
            continue

        # Check opposite probe weight signs
        pw_i, pw_j = probe_weights[i], probe_weights[j]
        if pw_i * pw_j >= 0:
            continue  # Same sign, not a difference

        bias_diff = abs(biases[i] - biases[j])
        if bias_diff < 0.01:
            continue  # Too similar biases

        # Effective grid spacing: h ≈ |Δb| / |w|
        avg_norm = (norms[i] + norms[j]) / 2
        effective_h = bias_diff / avg_norm if avg_norm > 0 else float("inf")

        # "Difference strength": how much the probe weights differ
        diff_strength = abs(pw_i - pw_j)

        # Ratio of probe weights (ideally close to -1 for symmetric diff)
        if abs(pw_j) > 1e-10:
            weight_ratio = pw_i / pw_j
        else:
            weight_ratio = float("inf")

        pairs.append({
            "neuron_i": i,
            "neuron_j": j,
            "cos_sim": float(sim),
            "bias_i": float(biases[i]),
            "bias_j": float(biases[j]),
            "bias_diff": float(bias_diff),
            "effective_h": float(effective_h),
            "probe_w_i": float(pw_i),
            "probe_w_j": float(pw_j),
            "diff_strength": float(diff_strength),
            "weight_ratio": float(weight_ratio),
            "w_norm_i": float(norms[i]),
            "w_norm_j": float(norms[j]),
            "w_direction_i": W_input[i].tolist(),
            "w_direction_j": W_input[j].tolist(),
        })

    # Sort by difference strength (strongest pairs first)
    pairs.sort(key=lambda p: p["diff_strength"], reverse=True)
    return pairs


def find_triplets_for_second_deriv(
    W_input: np.ndarray,
    biases: np.ndarray,
    probe_weights: np.ndarray,
    cos_sim_threshold: float = 0.85,
    top_k: int = 20,
):
    """
    Find neuron triplets that could form [1, -2, 1] second derivative stencils.

    For a central difference second derivative:
      d²u/dx² ≈ [u(x-h) - 2u(x) + u(x+h)] / h²

    We look for three neurons with:
    1. Similar input weight direction
    2. Approximately evenly-spaced biases
    3. Probe weights proportional to [α, -2α, α]

    Parameters
    ----------
    W_input : np.ndarray
        PINN input weights
    biases : np.ndarray
        PINN biases
    probe_weights : np.ndarray
        Probe weights for second derivative
    cos_sim_threshold : float
        Minimum pairwise cosine similarity
    top_k : int
        Number of top triplets to return

    Returns
    -------
    list of dict
        Candidate triplets sorted by pattern quality.
    """
    n_neurons = W_input.shape[0]
    cos_sim = compute_neuron_similarity(W_input)

    # Pre-filter: only consider neurons with significant probe weight
    abs_pw = np.abs(probe_weights)
    significant = np.where(abs_pw > np.percentile(abs_pw, 25))[0]

    triplets = []
    for i, j, k in combinations(significant, 3):
        # Check pairwise similarity
        if (abs(cos_sim[i, j]) < cos_sim_threshold or
            abs(cos_sim[i, k]) < cos_sim_threshold or
            abs(cos_sim[j, k]) < cos_sim_threshold):
            continue

        # Sort by bias to get ordered triplet
        indices = np.array([i, j, k])
        bias_vals = biases[indices]
        sort_order = np.argsort(bias_vals)
        i_s, j_s, k_s = indices[sort_order]
        b1, b2, b3 = bias_vals[sort_order]

        # Check approximately evenly spaced
        gap1 = b2 - b1
        gap2 = b3 - b2
        if gap1 < 0.05 or gap2 < 0.05:
            continue
        evenness = min(gap1, gap2) / max(gap1, gap2) if max(gap1, gap2) > 0 else 0

        # Get probe weights for this triplet
        pw = probe_weights[np.array([i_s, j_s, k_s])]

        # Check [1, -2, 1] pattern: outer weights same sign, middle opposite
        if not (pw[0] * pw[2] > 0 and pw[1] * pw[0] < 0):
            continue

        # How close to ideal [1, -2, 1]?
        # Normalize: ideal ratio is |middle| / |outer_avg| = 2
        outer_avg = (abs(pw[0]) + abs(pw[2])) / 2
        if outer_avg < 1e-10:
            continue
        ratio = abs(pw[1]) / outer_avg

        # Pattern quality: how close ratio is to 2.0
        ratio_quality = 1.0 - abs(ratio - 2.0) / 2.0
        ratio_quality = max(0, ratio_quality)

        # Overall quality: evenness * ratio_quality * strength
        strength = abs(pw[0]) + abs(pw[1]) + abs(pw[2])
        quality = evenness * ratio_quality * strength

        triplets.append({
            "neurons": [int(i_s), int(j_s), int(k_s)],
            "biases": [float(b1), float(b2), float(b3)],
            "probe_weights": pw.tolist(),
            "gap1": float(gap1),
            "gap2": float(gap2),
            "evenness": float(evenness),
            "middle_to_outer_ratio": float(ratio),
            "ideal_ratio_deviation": float(abs(ratio - 2.0)),
            "quality": float(quality),
            "strength": float(strength),
        })

    triplets.sort(key=lambda t: t["quality"], reverse=True)
    return triplets[:top_k]


def analyze_tanh_derivative_connection(
    W_input: np.ndarray,
    biases: np.ndarray,
    probe_weights: np.ndarray,
    derivative_direction: int,
):
    """
    Analyze how tanh neurons relate to derivative computation.

    For h_i = tanh(w_i . x + b_i), the derivative is:
      dh_i/dx_j = w_ij * sech²(w_i . x + b_i)

    So the analytical derivative of the network output through layer_0 is:
      du/dx_j ≈ Σ_i (next_layer_contribution_i) * w_ij * sech²(...)

    The probe approximates: du/dx_j ≈ Σ_i p_i * h_i

    If the probe works well, p_i should correlate with
    how much neuron i contributes to the derivative.

    Parameters
    ----------
    W_input : np.ndarray
        PINN input weights (n_neurons, input_dim)
    biases : np.ndarray
        PINN biases (n_neurons,)
    probe_weights : np.ndarray
        Probe weights (n_neurons,)
    derivative_direction : int
        0 for du/dx, 1 for du/dy

    Returns
    -------
    dict
        Analysis results.
    """
    w_dir = W_input[:, derivative_direction]  # Weight in derivative direction
    w_norm = np.linalg.norm(W_input, axis=1)

    # The product p_i * w_dir_i should be informative
    # If p_i encodes how much neuron i contributes to du/dx_j,
    # and w_dir_i is how sensitive neuron i is to x_j,
    # then their product represents the derivative contribution
    product = probe_weights * w_dir

    # Neurons where product is large and positive contribute to increasing du/dx
    # Neurons where product is large and negative contribute to decreasing du/dx

    return {
        "w_direction": w_dir,
        "w_norms": w_norm,
        "products": product,
        "mean_product": float(np.mean(product)),
        "sum_positive_products": float(np.sum(product[product > 0])),
        "sum_negative_products": float(np.sum(product[product < 0])),
    }


def plot_pair_analysis(
    pairs: list,
    W_input: np.ndarray,
    biases: np.ndarray,
    derivative_name: str,
    layer_name: str,
    save_path: str,
    top_k: int = 8,
):
    """
    Visualize the top neuron pairs forming difference operations.

    Parameters
    ----------
    pairs : list
        List of pair dicts from find_neuron_pairs
    W_input : np.ndarray
        PINN input weights
    biases : np.ndarray
        PINN biases
    derivative_name : str
        Name of derivative
    layer_name : str
        Layer name
    save_path : str
        Path to save
    top_k : int
        Number of top pairs to show
    """
    top_pairs = pairs[:top_k]
    if not top_pairs:
        print(f"  No pairs found for {derivative_name} at {layer_name}")
        return

    n_pairs = len(top_pairs)
    fig, axes = plt.subplots(2, min(4, n_pairs), figsize=(5 * min(4, n_pairs), 8))
    if n_pairs < 4:
        # Pad axes
        axes = np.array(axes).reshape(2, -1)

    x_range = np.linspace(-3, 3, 200)

    for idx in range(min(4, n_pairs)):
        pair = top_pairs[idx]
        ni, nj = pair["neuron_i"], pair["neuron_j"]

        # Top row: individual neuron activation profiles (1D slice)
        ax = axes[0, idx]
        # For du/dx: project along x-axis (y=0.5)
        if derivative_name in ("du_dx", "d2u_dx2"):
            proj_dim = 0
            other_val = 0.5
            xlabel = "x"
        else:
            proj_dim = 1
            other_val = 0.5
            xlabel = "y"

        # Compute activations along 1D slice
        for neuron_idx, color, label_prefix in [(ni, "#2196F3", "n"), (nj, "#F44336", "n")]:
            w = W_input[neuron_idx]
            b = biases[neuron_idx]
            # 1D activation: tanh(w[proj_dim]*t + w[other_dim]*other_val + b)
            if proj_dim == 0:
                z = w[0] * x_range + w[1] * other_val + b
            else:
                z = w[0] * other_val + w[1] * x_range + b
            activation = np.tanh(z)
            pw = pair[f"probe_w_{['i', 'j'][neuron_idx != ni]}"]
            ax.plot(
                x_range, activation, color=color, linewidth=2,
                label=f"{label_prefix}{neuron_idx} (pw={pw:.3f})"
            )

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("tanh activation", fontsize=10)
        ax.set_title(
            f"Pair {idx+1}: n{ni} vs n{nj}\ncosim={pair['cos_sim']:.2f}",
            fontsize=10,
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Bottom row: weighted difference
        ax = axes[1, idx]
        for neuron_idx, pw_key, color in [
            (ni, "probe_w_i", "#2196F3"),
            (nj, "probe_w_j", "#F44336"),
        ]:
            w = W_input[neuron_idx]
            b = biases[neuron_idx]
            if proj_dim == 0:
                z = w[0] * x_range + w[1] * other_val + b
            else:
                z = w[0] * other_val + w[1] * x_range + b
            weighted = pair[pw_key] * np.tanh(z)
            ax.plot(x_range, weighted, color=color, linewidth=1.5, alpha=0.5,
                    label=f"pw*h_{neuron_idx}")

        # Combined (difference)
        w_i, b_i = W_input[ni], biases[ni]
        w_j, b_j = W_input[nj], biases[nj]
        if proj_dim == 0:
            z_i = w_i[0] * x_range + w_i[1] * other_val + b_i
            z_j = w_j[0] * x_range + w_j[1] * other_val + b_j
        else:
            z_i = w_i[0] * other_val + w_i[1] * x_range + b_i
            z_j = w_j[0] * other_val + w_j[1] * x_range + b_j

        combined = pair["probe_w_i"] * np.tanh(z_i) + pair["probe_w_j"] * np.tanh(z_j)
        ax.plot(x_range, combined, color="#4CAF50", linewidth=2.5, label="Combined")
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Weighted activation", fontsize=10)
        ax.set_title(f"h={pair['effective_h']:.3f}, ratio={pair['weight_ratio']:.2f}", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    display = {"du_dx": "du/dx", "du_dy": "du/dy", "d2u_dx2": "d2u/dx2",
               "d2u_dy2": "d2u/dy2", "laplacian": "Laplacian"}
    plt.suptitle(
        f"Finite-Difference Pairs for {display.get(derivative_name, derivative_name)} ({layer_name})\n"
        f"Top pairs: opposite probe weights + similar input directions = difference operation",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_triplet_analysis(
    triplets: list,
    W_input: np.ndarray,
    biases: np.ndarray,
    derivative_name: str,
    layer_name: str,
    save_path: str,
):
    """Visualize top triplets for second derivative [1, -2, 1] patterns."""
    if not triplets:
        print(f"  No triplets found for {derivative_name} at {layer_name}")
        return

    n_show = min(4, len(triplets))
    fig, axes = plt.subplots(2, n_show, figsize=(5 * n_show, 8))
    if n_show == 1:
        axes = axes.reshape(2, 1)

    x_range = np.linspace(-3, 3, 200)

    for idx in range(n_show):
        trip = triplets[idx]
        neurons = trip["neurons"]
        pws = trip["probe_weights"]

        proj_dim = 0 if derivative_name in ("d2u_dx2", "laplacian") else 1
        other_val = 0.5

        # Top row: individual activations
        ax = axes[0, idx]
        colors = ["#2196F3", "#F44336", "#4CAF50"]
        for k, (n_idx, pw) in enumerate(zip(neurons, pws)):
            w = W_input[n_idx]
            b = biases[n_idx]
            if proj_dim == 0:
                z = w[0] * x_range + w[1] * other_val + b
            else:
                z = w[0] * other_val + w[1] * x_range + b
            ax.plot(x_range, np.tanh(z), color=colors[k], linewidth=2,
                    label=f"n{n_idx} (pw={pw:.3f})")

        ax.set_title(
            f"Triplet {idx+1}: [{neurons[0]},{neurons[1]},{neurons[2]}]\n"
            f"ratio={trip['middle_to_outer_ratio']:.2f} (ideal=2.0)",
            fontsize=10,
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("x" if proj_dim == 0 else "y", fontsize=10)
        ax.set_ylabel("tanh activation", fontsize=10)

        # Bottom row: weighted sum showing [1, -2, 1] pattern
        ax = axes[1, idx]
        combined = np.zeros_like(x_range)
        for k, (n_idx, pw) in enumerate(zip(neurons, pws)):
            w = W_input[n_idx]
            b = biases[n_idx]
            if proj_dim == 0:
                z = w[0] * x_range + w[1] * other_val + b
            else:
                z = w[0] * other_val + w[1] * x_range + b
            weighted = pw * np.tanh(z)
            ax.plot(x_range, weighted, color=colors[k], linewidth=1, alpha=0.5)
            combined += weighted

        ax.plot(x_range, combined, color="black", linewidth=2.5, label="Combined [1,-2,1]")
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_title(
            f"evenness={trip['evenness']:.2f}, quality={trip['quality']:.3f}",
            fontsize=10,
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("x" if proj_dim == 0 else "y", fontsize=10)
        ax.set_ylabel("Weighted activation", fontsize=10)

    display = {"d2u_dx2": "d2u/dx2", "d2u_dy2": "d2u/dy2", "laplacian": "Laplacian"}
    plt.suptitle(
        f"[1, -2, 1] Triplet Patterns for {display.get(derivative_name, derivative_name)} ({layer_name})\n"
        f"Looking for central-difference second derivative stencils",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_product_analysis(
    W_input: np.ndarray,
    probe_weights_dx: np.ndarray,
    probe_weights_dy: np.ndarray,
    save_path: str,
):
    """
    Analyze the product p_i * w_{i,direction} which represents
    the effective derivative contribution of each neuron.

    If the network computes derivatives via tanh derivative chain rule:
      du/dx ≈ Σ_i c_i * w_{x,i} * sech²(w_i·x + b_i)
    Then the probe (on tanh activations) approximates this via:
      du/dx ≈ Σ_i p_i * tanh(w_i·x + b_i)

    The product p_i * w_{x,i} should be systematically signed.
    """
    w_x = W_input[:, 0]
    w_y = W_input[:, 1]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Product distribution for du/dx
    ax = axes[0, 0]
    product_dx = probe_weights_dx * w_x
    colors = ["#2196F3" if v >= 0 else "#F44336" for v in product_dx]
    sorted_idx = np.argsort(product_dx)
    ax.bar(range(len(product_dx)), product_dx[sorted_idx], color=[colors[i] for i in sorted_idx])
    ax.set_xlabel("Neuron (sorted)", fontsize=10)
    ax.set_ylabel("p_i * w_{x,i}", fontsize=10)
    ax.set_title("du/dx: probe weight * PINN x-weight product", fontsize=11, fontweight="bold")
    ax.axhline(y=0, color="black", linewidth=0.5)
    n_pos = np.sum(product_dx > 0)
    n_neg = np.sum(product_dx < 0)
    ax.text(0.02, 0.95, f"+:{n_pos}, -:{n_neg}\nsum={product_dx.sum():.3f}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(facecolor="wheat", alpha=0.8))

    # Panel 2: Product distribution for du/dy
    ax = axes[0, 1]
    product_dy = probe_weights_dy * w_y
    colors = ["#2196F3" if v >= 0 else "#F44336" for v in product_dy]
    sorted_idx = np.argsort(product_dy)
    ax.bar(range(len(product_dy)), product_dy[sorted_idx], color=[colors[i] for i in sorted_idx])
    ax.set_xlabel("Neuron (sorted)", fontsize=10)
    ax.set_ylabel("p_i * w_{y,i}", fontsize=10)
    ax.set_title("du/dy: probe weight * PINN y-weight product", fontsize=11, fontweight="bold")
    ax.axhline(y=0, color="black", linewidth=0.5)
    n_pos = np.sum(product_dy > 0)
    n_neg = np.sum(product_dy < 0)
    ax.text(0.02, 0.95, f"+:{n_pos}, -:{n_neg}\nsum={product_dy.sum():.3f}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(facecolor="wheat", alpha=0.8))

    # Panel 3: Histogram of products
    ax = axes[1, 0]
    ax.hist(product_dx, bins=25, alpha=0.6, color="#2196F3", label="du/dx: p*w_x", edgecolor="white")
    ax.hist(product_dy, bins=25, alpha=0.6, color="#FF9800", label="du/dy: p*w_y", edgecolor="white")
    ax.set_xlabel("Product p_i * w_{dir,i}", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Distribution of derivative contribution products", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.axvline(x=0, color="black", linewidth=0.5, linestyle="--")

    # Panel 4: Systematic sign bias
    ax = axes[1, 1]
    # If computation is derivative-like, products should be systematically negative
    # (because d/dx[tanh(w*x+b)] = w*sech²(...) which is always positive for w>0,
    #  but the PROBE maps tanh->derivative, and tanh = 2*sigmoid - 1, so
    #  the product sign tells us about the effective computation direction)
    sign_dx = np.sign(product_dx)
    sign_dy = np.sign(product_dy)

    categories = ["Both +", "Both -", "dx+/dy-", "dx-/dy+"]
    counts = [
        np.sum((sign_dx > 0) & (sign_dy > 0)),
        np.sum((sign_dx < 0) & (sign_dy < 0)),
        np.sum((sign_dx > 0) & (sign_dy < 0)),
        np.sum((sign_dx < 0) & (sign_dy > 0)),
    ]
    bars = ax.bar(categories, counts, color=["#4CAF50", "#F44336", "#9C27B0", "#FF9800"])
    ax.set_ylabel("Number of neurons", fontsize=10)
    ax.set_title("Product sign patterns across derivatives", fontsize=11, fontweight="bold")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha="center", fontsize=10, fontweight="bold")

    plt.suptitle(
        "Derivative Contribution Analysis: p_i * w_{direction,i}\n"
        "Systematic sign patterns indicate derivative-like computation",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")

    return {
        "product_dx_sum": float(product_dx.sum()),
        "product_dy_sum": float(product_dy.sum()),
        "product_dx_mean": float(product_dx.mean()),
        "product_dy_mean": float(product_dy.mean()),
        "frac_negative_dx": float(np.mean(product_dx < 0)),
        "frac_negative_dy": float(np.mean(product_dy < 0)),
    }


def generate_fd_analysis_report(
    pairs_results: dict,
    triplets_results: dict,
    product_stats: dict,
    save_path: str,
):
    """Generate comprehensive text report of finite-difference pattern analysis."""
    lines = []
    lines.append("=" * 80)
    lines.append("FINITE-DIFFERENCE PATTERN ANALYSIS REPORT - Day 11 Task 2")
    lines.append("=" * 80)

    lines.append("")
    lines.append("BACKGROUND")
    lines.append("-" * 40)
    lines.append("A finite difference approximation computes derivatives by combining")
    lines.append("function values at nearby points:")
    lines.append("  First derivative:  du/dx ≈ [u(x+h) - u(x-h)] / (2h)")
    lines.append("  Second derivative: d2u/dx2 ≈ [u(x-h) - 2u(x) + u(x+h)] / h2")
    lines.append("")
    lines.append("In our PINN, layer_0 neurons compute h_i = tanh(w_i . x + b_i).")
    lines.append("If two neurons have similar input weights but different biases,")
    lines.append("they evaluate the same function at different spatial locations.")
    lines.append("Probe weights that combine them with opposite signs create a")
    lines.append("difference operation analogous to finite differences.")
    lines.append("")

    # Section 1: Pair analysis
    lines.append("1. NEURON PAIR ANALYSIS (First Derivatives)")
    lines.append("-" * 50)

    for deriv_name in ["du_dx", "du_dy"]:
        display = {"du_dx": "du/dx", "du_dy": "du/dy"}[deriv_name]
        key = f"layer_0_{deriv_name}"
        pairs = pairs_results.get(key, [])
        lines.append(f"\n  {display} at layer_0:")
        lines.append(f"    Total candidate pairs found: {len(pairs)}")

        if pairs:
            lines.append(f"    Top 5 pairs by difference strength:")
            for i, p in enumerate(pairs[:5]):
                lines.append(
                    f"      #{i+1}: n{p['neuron_i']}-n{p['neuron_j']} | "
                    f"cosim={p['cos_sim']:.3f} | "
                    f"pw=[{p['probe_w_i']:+.3f}, {p['probe_w_j']:+.3f}] | "
                    f"ratio={p['weight_ratio']:.2f} | "
                    f"h_eff={p['effective_h']:.3f}"
                )

            # Statistics
            ratios = [abs(p["weight_ratio"]) for p in pairs[:20]]
            h_vals = [p["effective_h"] for p in pairs[:20]]
            lines.append(f"\n    Statistics (top 20 pairs):")
            lines.append(f"      Mean |weight ratio|: {np.mean(ratios):.3f} (ideal=-1.0 for symmetric)")
            lines.append(f"      Mean effective h: {np.mean(h_vals):.3f}")
            lines.append(f"      Median effective h: {np.median(h_vals):.3f}")

            # How many have ratio close to -1 (symmetric difference)?
            near_symmetric = sum(1 for r in ratios if 0.5 < r < 2.0)
            lines.append(f"      Near-symmetric pairs (0.5 < |ratio| < 2.0): {near_symmetric}/{len(ratios)}")
        else:
            lines.append(f"    No pairs found meeting criteria")

    # Section 2: Triplet analysis
    lines.append("\n\n2. NEURON TRIPLET ANALYSIS (Second Derivatives)")
    lines.append("-" * 50)

    for deriv_name in ["d2u_dx2", "d2u_dy2", "laplacian"]:
        display = {"d2u_dx2": "d2u/dx2", "d2u_dy2": "d2u/dy2", "laplacian": "Laplacian"}[deriv_name]
        key = f"layer_0_{deriv_name}"
        triplets = triplets_results.get(key, [])
        lines.append(f"\n  {display} at layer_0:")
        lines.append(f"    Total candidate triplets found: {len(triplets)}")

        if triplets:
            lines.append(f"    Top 3 triplets by quality:")
            for i, t in enumerate(triplets[:3]):
                pw_str = ", ".join([f"{w:+.3f}" for w in t["probe_weights"]])
                lines.append(
                    f"      #{i+1}: neurons=[{t['neurons'][0]},{t['neurons'][1]},{t['neurons'][2]}] | "
                    f"pw=[{pw_str}] | "
                    f"ratio={t['middle_to_outer_ratio']:.2f} (ideal=2.0) | "
                    f"evenness={t['evenness']:.2f}"
                )
        else:
            lines.append(f"    No [1,-2,1] triplets found meeting criteria")

    # Section 3: Product analysis
    lines.append("\n\n3. DERIVATIVE CONTRIBUTION PRODUCT ANALYSIS")
    lines.append("-" * 50)
    lines.append("  Product p_i * w_{direction,i} reveals the effective derivative")
    lines.append("  contribution of each neuron.")
    lines.append("")
    lines.append(f"  du/dx products (p_i * w_x_i):")
    lines.append(f"    Sum: {product_stats['product_dx_sum']:.4f}")
    lines.append(f"    Mean: {product_stats['product_dx_mean']:.4f}")
    lines.append(f"    Fraction negative: {product_stats['frac_negative_dx']:.1%}")
    lines.append(f"  du/dy products (p_i * w_y_i):")
    lines.append(f"    Sum: {product_stats['product_dy_sum']:.4f}")
    lines.append(f"    Mean: {product_stats['product_dy_mean']:.4f}")
    lines.append(f"    Fraction negative: {product_stats['frac_negative_dy']:.1%}")

    # Section 4: Interpretation
    lines.append("\n\n4. INTERPRETATION")
    lines.append("-" * 50)

    # Determine findings
    any_pairs_dx = len(pairs_results.get("layer_0_du_dx", [])) > 0
    any_pairs_dy = len(pairs_results.get("layer_0_du_dy", [])) > 0
    any_triplets = any(
        len(triplets_results.get(f"layer_0_{d}", [])) > 0
        for d in ["d2u_dx2", "d2u_dy2", "laplacian"]
    )

    if any_pairs_dx or any_pairs_dy:
        lines.append("  FINDING 1: Neuron pairs with difference-like patterns EXIST")
        lines.append("    - Neurons with similar spatial orientation but different biases")
        lines.append("    - Combined with opposite-sign probe weights")
        lines.append("    - This is structurally analogous to finite differences")
        lines.append("    - However, the network uses smooth tanh functions, not point evaluations")
        lines.append("    - This represents a 'soft' or 'continuous' finite difference")
    else:
        lines.append("  FINDING 1: No clear difference pairs found at the strict threshold")
        lines.append("    - The network may use a more distributed computation strategy")

    lines.append("")
    if any_triplets:
        lines.append("  FINDING 2: [1,-2,1] triplet patterns partially present")
        lines.append("    - Some neuron triplets show the characteristic sign pattern")
        lines.append("    - Quality varies; not perfect central difference stencils")
        lines.append("    - Consistent with weak second derivative R2 scores")
    else:
        lines.append("  FINDING 2: No clear [1,-2,1] triplets found")
        lines.append("    - Second derivatives are NOT explicitly encoded as finite differences")
        lines.append("    - Consistent with Day 9 finding: second derivatives computed via autograd")

    lines.append("")
    lines.append("  FINDING 3: Product analysis p_i * w_{dir,i}")
    if product_stats["frac_negative_dx"] > 0.6 or product_stats["frac_negative_dx"] < 0.4:
        lines.append("    - Systematic sign bias in products indicates directional computation")
        lines.append("    - Neurons are not randomly contributing to derivative prediction")
    else:
        lines.append("    - Roughly balanced positive/negative products")
        lines.append("    - Computation is distributed without strong directional bias")

    lines.append("")
    lines.append("  OVERALL CONCLUSION:")
    lines.append("    The network's derivative computation is PARTIALLY analogous to")
    lines.append("    finite differences, but operates in a continuous, distributed manner.")
    lines.append("    Rather than discrete point evaluations, neurons create smooth")
    lines.append("    spatial basis functions that are combined via probe weights.")
    lines.append("    This is a 'continuous finite difference' or 'spectral-like' approach.")

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
    print("Day 11 Task 2: Finite-Difference Pattern Analysis")
    print("=" * 60)

    # Load data
    print("\n1. Loading all data...")
    pinn_state, first_probes, second_probes = load_all_data()

    # PINN first layer weights
    W_input = pinn_state["layers.0.weight"].numpy()  # (64, 2)
    b_input = pinn_state["layers.0.bias"].numpy()  # (64,)

    print(f"   PINN layer_0: {W_input.shape[0]} neurons, input_dim={W_input.shape[1]}")
    print(f"   Bias range: [{b_input.min():.3f}, {b_input.max():.3f}]")

    # Cosine similarity between neurons
    cos_sim = compute_neuron_similarity(W_input)
    upper_tri = cos_sim[np.triu_indices(64, k=1)]
    print(f"   Mean pairwise cosine similarity: {upper_tri.mean():.3f}")
    print(f"   Pairs with |cos_sim| > 0.9: {np.sum(np.abs(upper_tri) > 0.9)}")
    print(f"   Pairs with |cos_sim| > 0.8: {np.sum(np.abs(upper_tri) > 0.8)}")

    # ---- First derivative pair analysis ----
    print("\n2. Finding difference pairs for first derivatives...")
    pairs_results = {}
    for deriv_name in ["du_dx", "du_dy"]:
        pw = first_probes["layer_0"][deriv_name]["weight"].numpy().flatten()
        pairs = find_neuron_pairs(W_input, b_input, pw, cos_sim_threshold=0.8)
        key = f"layer_0_{deriv_name}"
        pairs_results[key] = pairs
        display = {"du_dx": "du/dx", "du_dy": "du/dy"}[deriv_name]
        print(f"   {display}: {len(pairs)} candidate pairs (cos_sim > 0.8, opposite signs)")

        if pairs:
            plot_pair_analysis(
                pairs, W_input, b_input, deriv_name, "layer_0",
                str(output_dir / f"fd_pairs_{deriv_name}_layer0.png"),
            )

    # ---- Second derivative triplet analysis ----
    print("\n3. Finding [1,-2,1] triplets for second derivatives...")
    triplets_results = {}
    for deriv_name in ["d2u_dx2", "d2u_dy2", "laplacian"]:
        pw = second_probes["layer_0"][deriv_name]["weight"].numpy().flatten()
        triplets = find_triplets_for_second_deriv(
            W_input, b_input, pw, cos_sim_threshold=0.8, top_k=10
        )
        key = f"layer_0_{deriv_name}"
        triplets_results[key] = triplets
        display = {"d2u_dx2": "d2u/dx2", "d2u_dy2": "d2u/dy2", "laplacian": "Laplacian"}[deriv_name]
        print(f"   {display}: {len(triplets)} candidate triplets")

        if triplets:
            plot_triplet_analysis(
                triplets, W_input, b_input, deriv_name, "layer_0",
                str(output_dir / f"fd_triplets_{deriv_name}_layer0.png"),
            )

    # ---- Also check layer_3 (best R2) ----
    print("\n4. Checking layer_3 patterns (for comparison)...")
    W_layer3 = pinn_state["layers.3.weight"].numpy()  # (64, 64)
    b_layer3 = pinn_state["layers.3.bias"].numpy()

    for deriv_name in ["du_dx", "du_dy"]:
        pw = first_probes["layer_3"][deriv_name]["weight"].numpy().flatten()
        pairs = find_neuron_pairs(W_layer3, b_layer3, pw, cos_sim_threshold=0.8)
        key = f"layer_3_{deriv_name}"
        pairs_results[key] = pairs
        display = {"du_dx": "du/dx", "du_dy": "du/dy"}[deriv_name]
        print(f"   layer_3 {display}: {len(pairs)} candidate pairs")

    # ---- Product analysis ----
    print("\n5. Analyzing derivative contribution products (layer_0)...")
    pw_dx = first_probes["layer_0"]["du_dx"]["weight"].numpy().flatten()
    pw_dy = first_probes["layer_0"]["du_dy"]["weight"].numpy().flatten()
    product_stats = plot_product_analysis(
        W_input, pw_dx, pw_dy,
        str(output_dir / "derivative_product_analysis.png"),
    )
    print(f"   du/dx: sum(p*w_x)={product_stats['product_dx_sum']:.4f}, "
          f"frac_neg={product_stats['frac_negative_dx']:.1%}")
    print(f"   du/dy: sum(p*w_y)={product_stats['product_dy_sum']:.4f}, "
          f"frac_neg={product_stats['frac_negative_dy']:.1%}")

    # ---- Generate report ----
    print("\n6. Generating analysis report...")
    report = generate_fd_analysis_report(
        pairs_results, triplets_results, product_stats,
        str(output_dir / "finite_difference_analysis_report.txt"),
    )

    # Save structured results
    summary = {
        "pairs_counts": {k: len(v) for k, v in pairs_results.items()},
        "triplets_counts": {k: len(v) for k, v in triplets_results.items()},
        "product_stats": product_stats,
        "top_pairs": {
            k: v[:5] for k, v in pairs_results.items() if v
        },
        "top_triplets": {
            k: v[:3] for k, v in triplets_results.items() if v
        },
    }
    # Remove numpy arrays that can't be serialized
    for key in list(summary["top_pairs"].keys()):
        for p in summary["top_pairs"][key]:
            p.pop("w_direction_i", None)
            p.pop("w_direction_j", None)

    with open(output_dir / "finite_difference_analysis.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {output_dir / 'finite_difference_analysis.json'}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n  Pair analysis (cos_sim > 0.8, opposite probe signs):")
    for k, v in pairs_results.items():
        print(f"    {k}: {len(v)} pairs")
    print(f"\n  Triplet analysis ([1,-2,1] pattern):")
    for k, v in triplets_results.items():
        print(f"    {k}: {len(v)} triplets")
    print(f"\n  Product analysis:")
    print(f"    du/dx sum(p*w_x) = {product_stats['product_dx_sum']:.4f}")
    print(f"    du/dy sum(p*w_y) = {product_stats['product_dy_sum']:.4f}")
    print("\nTask 2 complete!")


if __name__ == "__main__":
    main()
