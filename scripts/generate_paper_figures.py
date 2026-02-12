#!/usr/bin/env python3
"""
Generate publication-quality figures for the mechanistic interpretability paper.

Produces 5 figures with consistent styling:
  Fig 1: PINN solution quality (solution heatmaps + error)
  Fig 2: Derivative probing R² heatmap (core finding)
  Fig 3: Layer-by-layer emergence (first vs second derivatives)
  Fig 4: Probe weight analysis (FD-like patterns + correlations)
  Fig 5: Architecture comparison (generalization of findings)
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.mlp import MLP
from src.problems.poisson import PoissonProblem

# ---------------------------------------------------------------------------
# Consistent publication style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

OUTPUT_DIR = "outputs/paper_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color palette
C_BLUE = "#2171B5"
C_LIGHT_BLUE = "#6BAED6"
C_RED = "#CB181D"
C_ORANGE = "#FD8D3C"
C_PURPLE = "#7B2D8E"
C_GREEN = "#238B45"
C_GRAY = "#636363"


# ===========================================================================
# Figure 1: PINN Solution Quality
# ===========================================================================
def figure1_solution_quality():
    """3-panel: Analytical solution, PINN prediction, pointwise error."""
    print("Generating Figure 1: Solution quality...")

    problem = PoissonProblem()
    model = MLP(input_dim=2, hidden_dims=[64, 64, 64, 64], output_dim=1, activation="tanh")

    checkpoint = torch.load(
        "outputs/models/poisson_pinn_trained.pt", map_location="cpu", weights_only=False
    )
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Generate grid
    n = 100
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    coords = np.stack([X.flatten(), Y.flatten()], axis=1)
    coords_t = torch.tensor(coords, dtype=torch.float32)

    with torch.no_grad():
        u_pred = model(coords_t).numpy().reshape(n, n)
    u_exact = problem.analytical_solution(coords_t).numpy().reshape(n, n)
    error = np.abs(u_pred - u_exact)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))

    # Panel a: Analytical
    im0 = axes[0].pcolormesh(X, Y, u_exact, cmap="RdBu_r", shading="auto")
    axes[0].set_title("(a) Analytical Solution", fontweight="bold")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_aspect("equal")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Panel b: PINN
    im1 = axes[1].pcolormesh(
        X, Y, u_pred, cmap="RdBu_r", shading="auto",
        vmin=im0.get_clim()[0], vmax=im0.get_clim()[1],
    )
    axes[1].set_title("(b) PINN Prediction", fontweight="bold")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_aspect("equal")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel c: Error
    im2 = axes[2].pcolormesh(X, Y, error, cmap="hot_r", shading="auto")
    axes[2].set_title("(c) Absolute Error", fontweight="bold")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    axes[2].set_aspect("equal")
    cb = plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    cb.formatter.set_powerlimits((-2, -2))
    cb.update_ticks()

    plt.tight_layout(w_pad=1.5)
    path = os.path.join(OUTPUT_DIR, "fig1_solution_quality.png")
    fig.savefig(path)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close()
    print(f"  Saved: {path}")


# ===========================================================================
# Figure 2: R² Heatmap (core finding)
# ===========================================================================
def figure2_r2_heatmap():
    """Heatmap of R² values: layers (rows) × derivatives (columns)."""
    print("Generating Figure 2: R-squared heatmap...")

    # Load baseline probe results
    with open("outputs/day9_task1/first_derivative_probe_results.json") as f:
        first = json.load(f)
    with open("outputs/day9_task2/second_derivative_probe_results.json") as f:
        second = json.load(f)

    layer_names = ["Layer 0", "Layer 1", "Layer 2", "Layer 3"]
    deriv_names = [
        r"$\partial u/\partial x$",
        r"$\partial u/\partial y$",
        r"$\partial^2 u/\partial x^2$",
        r"$\partial^2 u/\partial y^2$",
        r"$\nabla^2 u$",
    ]

    # Build R² matrix
    r2_matrix = np.zeros((4, 5))
    for i, entry_f in enumerate(first["results"]):
        entry_s = second["results"][i]
        r2_matrix[i, 0] = entry_f["du_dx"]["r2"]
        r2_matrix[i, 1] = entry_f["du_dy"]["r2"]
        r2_matrix[i, 2] = entry_s["d2u_dx2"]["r2"]
        r2_matrix[i, 3] = entry_s["d2u_dy2"]["r2"]
        r2_matrix[i, 4] = entry_s["laplacian"]["r2"]

    fig, ax = plt.subplots(figsize=(7, 3.5))

    im = ax.imshow(r2_matrix, cmap="RdYlGn", aspect="auto", vmin=-0.5, vmax=1.0)

    # Annotations
    for i in range(4):
        for j in range(5):
            val = r2_matrix[i, j]
            color = "white" if val < -0.1 or val > 0.8 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)

    ax.set_xticks(range(5))
    ax.set_xticklabels(deriv_names, fontsize=10)
    ax.set_yticks(range(4))
    ax.set_yticklabels(layer_names, fontsize=10)
    ax.set_xlabel("Probing Target", fontsize=11)
    ax.set_ylabel("Hidden Layer", fontsize=11)

    # Dividing line between 1st and 2nd order
    ax.axvline(x=1.5, color="white", linewidth=3)

    # Bracket labels
    ax.text(0.5, -0.8, "1st order", ha="center", fontsize=9,
            fontstyle="italic", color=C_BLUE)
    ax.text(3, -0.8, "2nd order", ha="center", fontsize=9,
            fontstyle="italic", color=C_RED)

    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cb.set_label("R-squared", fontsize=10)

    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig2_r2_heatmap.png")
    fig.savefig(path)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close()
    print(f"  Saved: {path}")


# ===========================================================================
# Figure 3: Layer-by-layer emergence
# ===========================================================================
def figure3_emergence():
    """Line plot showing R² progression across layers, with shaded regions."""
    print("Generating Figure 3: Derivative emergence...")

    with open("outputs/day9_task1/first_derivative_probe_results.json") as f:
        first = json.load(f)
    with open("outputs/day9_task2/second_derivative_probe_results.json") as f:
        second = json.load(f)

    layers = [0, 1, 2, 3]

    du_dx = [first["results"][i]["du_dx"]["r2"] for i in range(4)]
    du_dy = [first["results"][i]["du_dy"]["r2"] for i in range(4)]
    d2u_dx2 = [second["results"][i]["d2u_dx2"]["r2"] for i in range(4)]
    d2u_dy2 = [second["results"][i]["d2u_dy2"]["r2"] for i in range(4)]
    lap = [second["results"][i]["laplacian"]["r2"] for i in range(4)]

    fig, ax = plt.subplots(figsize=(6, 4.5))

    # First derivatives
    ax.plot(layers, du_dx, "o-", color=C_BLUE, linewidth=2.5, markersize=8,
            label=r"$\partial u/\partial x$", zorder=5)
    ax.plot(layers, du_dy, "s-", color=C_LIGHT_BLUE, linewidth=2.5, markersize=8,
            label=r"$\partial u/\partial y$", zorder=5)

    # Second derivatives
    ax.plot(layers, d2u_dx2, "^-", color=C_RED, linewidth=2.5, markersize=8,
            label=r"$\partial^2 u/\partial x^2$", zorder=5)
    ax.plot(layers, d2u_dy2, "D-", color=C_ORANGE, linewidth=2.5, markersize=8,
            label=r"$\partial^2 u/\partial y^2$", zorder=5)
    ax.plot(layers, lap, "v-", color=C_PURPLE, linewidth=2.5, markersize=8,
            label=r"$\nabla^2 u$ (Laplacian)", zorder=5)

    # Shaded regions for interpretation
    ax.axhspan(0.85, 1.0, alpha=0.08, color="green", label="_nolegend_")
    ax.axhspan(-0.5, 0.0, alpha=0.05, color="red", label="_nolegend_")

    ax.text(3.05, 0.92, "Explicit\nencoding", fontsize=8, color=C_GREEN, fontstyle="italic")
    ax.text(3.05, -0.25, "Not\nencoded", fontsize=8, color=C_RED, fontstyle="italic")

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Hidden Layer Index")
    ax.set_ylabel(r"Probe $R^2$")
    ax.set_xticks(layers)
    ax.set_xticklabels(["Layer 0", "Layer 1", "Layer 2", "Layer 3"])
    ax.set_ylim(-0.5, 1.02)
    ax.legend(loc="center left", framealpha=0.9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig3_derivative_emergence.png")
    fig.savefig(path)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close()
    print(f"  Saved: {path}")


# ===========================================================================
# Figure 4: Probe weight analysis
# ===========================================================================
def figure4_weight_analysis():
    """2-panel: (a) PINN-probe weight correlation, (b) FD pair structure."""
    print("Generating Figure 4: Probe weight analysis...")

    # Load probe weights and PINN weights
    probes_first = torch.load(
        "outputs/day9_task1/first_derivative_probes.pt",
        map_location="cpu", weights_only=False,
    )
    model = MLP(input_dim=2, hidden_dims=[64, 64, 64, 64], output_dim=1, activation="tanh")
    checkpoint = torch.load(
        "outputs/models/poisson_pinn_trained.pt",
        map_location="cpu", weights_only=False,
    )
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # Get layer 0 PINN weights (first linear layer)
    pinn_w = list(model.parameters())[0].detach().numpy()  # shape (64, 2)
    w_x = pinn_w[:, 0]
    w_y = pinn_w[:, 1]

    # Get layer 0 probe weights
    probe_du_dx_w = probes_first["layer_0"]["du_dx"]["weight"].numpy().flatten()
    probe_du_dy_w = probes_first["layer_0"]["du_dy"]["weight"].numpy().flatten()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    # Panel (a): PINN w_x vs probe du/dx weights
    ax = axes[0]
    ax.scatter(w_x, probe_du_dx_w, alpha=0.6, s=30, color=C_BLUE, edgecolors="none",
               label=r"$w_x$ vs probe $\partial u/\partial x$")
    ax.scatter(w_y, probe_du_dy_w, alpha=0.6, s=30, color=C_RED, edgecolors="none",
               marker="s", label=r"$w_y$ vs probe $\partial u/\partial y$")

    # Trend lines
    for wx_data, pw_data, color in [
        (w_x, probe_du_dx_w, C_BLUE),
        (w_y, probe_du_dy_w, C_RED),
    ]:
        z = np.polyfit(wx_data, pw_data, 1)
        p = np.poly1d(z)
        xs = np.linspace(wx_data.min(), wx_data.max(), 100)
        ax.plot(xs, p(xs), "--", color=color, linewidth=1.5, alpha=0.7)

    corr_x = np.corrcoef(w_x, probe_du_dx_w)[0, 1]
    corr_y = np.corrcoef(w_y, probe_du_dy_w)[0, 1]

    ax.set_xlabel("PINN Input-Layer Weight")
    ax.set_ylabel("Probe Weight (Layer 0)")
    ax.set_title("(a) PINN-Probe Weight Correlation", fontweight="bold")
    ax.legend(fontsize=8)
    ax.text(0.05, 0.05,
            f"corr(w_x, probe_dx) = {corr_x:.2f}\ncorr(w_y, probe_dy) = {corr_y:.2f}",
            transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    # Panel (b): Probe weights for du/dx at layer 0 — show FD-like pairs
    ax = axes[1]
    n_neurons = len(probe_du_dx_w)
    indices = np.arange(n_neurons)

    colors_bar = [C_BLUE if v >= 0 else C_RED for v in probe_du_dx_w]
    ax.bar(indices, probe_du_dx_w, color=colors_bar, alpha=0.7, width=0.8)

    # Highlight top positive and negative neurons
    sorted_idx = np.argsort(probe_du_dx_w)
    top_neg = sorted_idx[:3]
    top_pos = sorted_idx[-3:]

    for idx in top_pos:
        ax.annotate(f"n{idx}", (idx, probe_du_dx_w[idx]),
                    textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=7, color=C_BLUE)
    for idx in top_neg:
        ax.annotate(f"n{idx}", (idx, probe_du_dx_w[idx]),
                    textcoords="offset points", xytext=(0, -12),
                    ha="center", fontsize=7, color=C_RED)

    ax.axhline(y=0, color="gray", linewidth=0.8)
    ax.set_xlabel("Neuron Index")
    ax.set_ylabel("Probe Weight")
    ax.set_title(r"(b) Probe Weights for $\partial u/\partial x$ (Layer 0)", fontweight="bold")

    # Annotation: positive = f(x+h), negative = f(x-h)
    ax.text(0.97, 0.95, "Positive: f(x+h)\nNegative: f(x-h)",
            transform=ax.transAxes, fontsize=8, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))

    plt.tight_layout(w_pad=2)
    path = os.path.join(OUTPUT_DIR, "fig4_weight_analysis.png")
    fig.savefig(path)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close()
    print(f"  Saved: {path}")


# ===========================================================================
# Figure 5: Architecture comparison
# ===========================================================================
def figure5_architecture_comparison():
    """2-panel: (a) peak R² bar chart, (b) first vs second derivative gap."""
    print("Generating Figure 5: Architecture comparison...")

    with open("outputs/day13_architecture_comparison/all_probe_results.json") as f:
        data = json.load(f)

    DERIVS = ["du_dx", "du_dy", "d2u_dx2", "d2u_dy2", "laplacian"]

    # Select subset for clarity (skip narrow, keep 5 most informative)
    arch_order = [
        "baseline_4L64_tanh",
        "shallow_2L",
        "deep_6L",
        "wide_128",
        "relu_4L",
    ]
    short = {
        "baseline_4L64_tanh": "Baseline\n(4L/64/tanh)",
        "shallow_2L": "Shallow\n(2L/64/tanh)",
        "deep_6L": "Deep\n(6L/64/tanh)",
        "wide_128": "Wide\n(4L/128/tanh)",
        "relu_4L": "ReLU\n(4L/64/relu)",
    }

    n = len(arch_order)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel (a): Peak R² for first vs second derivatives
    ax = axes[0]
    x = np.arange(n)
    width = 0.25

    peak_first = []
    peak_second = []
    peak_lap = []

    for arch in arch_order:
        pr = data[arch]["probe_results"]
        layers = list(pr.keys())
        best_first = max(
            max(pr[l]["du_dx"]["r_squared"], pr[l]["du_dy"]["r_squared"])
            for l in layers
        )
        best_second = max(
            max(pr[l]["d2u_dx2"]["r_squared"], pr[l]["d2u_dy2"]["r_squared"])
            for l in layers
        )
        best_lap = max(pr[l]["laplacian"]["r_squared"] for l in layers)
        peak_first.append(best_first)
        peak_second.append(best_second)
        peak_lap.append(best_lap)

    ax.bar(x - width, peak_first, width, label="1st derivatives (best)",
           color=C_BLUE, alpha=0.85)
    ax.bar(x, peak_second, width, label="2nd derivatives (best)",
           color=C_RED, alpha=0.85)
    ax.bar(x + width, peak_lap, width, label="Laplacian (best)",
           color=C_PURPLE, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([short[a] for a in arch_order], fontsize=8)
    ax.set_ylabel("Peak R-squared")
    ax.set_title("(a) Peak Derivative Encoding by Architecture", fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_ylim(-1.1, 1.05)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.7)

    # Panel (b): The "gap" — first minus second R² at final layer
    ax = axes[1]
    gaps = []
    first_final = []
    second_final = []

    for arch in arch_order:
        pr = data[arch]["probe_results"]
        layers = sorted(pr.keys())
        fl = layers[-1]
        f_avg = np.mean([pr[fl]["du_dx"]["r_squared"], pr[fl]["du_dy"]["r_squared"]])
        s_avg = np.mean([pr[fl]["d2u_dx2"]["r_squared"], pr[fl]["d2u_dy2"]["r_squared"]])
        first_final.append(f_avg)
        second_final.append(s_avg)
        gaps.append(f_avg - s_avg)

    bar_colors = [C_GREEN if g > 0.3 else C_ORANGE if g > 0 else C_GRAY for g in gaps]
    bars = ax.bar(x, gaps, 0.5, color=bar_colors, alpha=0.85, edgecolor="white")

    # Label each bar with the gap value
    for i, (bar, gap) in enumerate(zip(bars, gaps)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{gap:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([short[a] for a in arch_order], fontsize=8)
    ax.set_ylabel(r"$R^2_{1st} - R^2_{2nd}$ (final layer)")
    ax.set_title("(b) Two-Stage Gap Across Architectures", fontweight="bold")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.7)
    ax.set_ylim(-0.1, 1.4)

    ax.text(0.02, 0.95, "Larger gap = stronger\ntwo-stage pattern",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))

    plt.tight_layout(w_pad=2)
    path = os.path.join(OUTPUT_DIR, "fig5_architecture_comparison.png")
    fig.savefig(path)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close()
    print(f"  Saved: {path}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    print(f"Generating paper figures in {OUTPUT_DIR}/\n")

    figure1_solution_quality()
    figure2_r2_heatmap()
    figure3_emergence()
    figure4_weight_analysis()
    figure5_architecture_comparison()

    print(f"\nAll 5 figures saved to {OUTPUT_DIR}/")
    print("Each figure available as both PNG (300 DPI) and PDF (vector).")


if __name__ == "__main__":
    main()
