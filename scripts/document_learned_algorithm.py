"""
Day 11 Task 4: Document initial hypothesis about the learned algorithm.

Synthesizes all findings from Days 9-11 (probing results, weight analysis,
finite difference patterns, stencil comparisons) into a coherent hypothesis
about the computational algorithm the PINN has learned.

Produces:
  1. A comprehensive hypothesis document
  2. A publication-quality summary figure
  3. An evidence table linking findings to hypothesis components
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_all_results():
    """Load all results from Days 9-11."""
    results = {}

    # Day 9: Probe R2 results
    with open(PROJECT_ROOT / "outputs" / "day9_task1" / "first_derivative_probe_results.json") as f:
        results["first_r2"] = json.load(f)
    with open(PROJECT_ROOT / "outputs" / "day9_task2" / "second_derivative_probe_results.json") as f:
        results["second_r2"] = json.load(f)

    # Day 11: Probe weight correlations
    with open(PROJECT_ROOT / "outputs" / "day11_probe_weights" / "pinn_probe_correlations.json") as f:
        results["correlations"] = json.load(f)

    # Day 11: FD analysis
    with open(PROJECT_ROOT / "outputs" / "day11_probe_weights" / "finite_difference_analysis.json") as f:
        results["fd_analysis"] = json.load(f)

    # Day 11: Stencil match scores
    with open(PROJECT_ROOT / "outputs" / "day11_probe_weights" / "stencil_match_scores.json") as f:
        results["stencil_scores"] = json.load(f)

    # Probe weights (for summary figure)
    results["first_probes"] = torch.load(
        PROJECT_ROOT / "outputs" / "day9_task1" / "first_derivative_probes.pt",
        weights_only=False, map_location="cpu",
    )
    results["second_probes"] = torch.load(
        PROJECT_ROOT / "outputs" / "day9_task2" / "second_derivative_probes.pt",
        weights_only=False, map_location="cpu",
    )

    # PINN weights
    ckpt = torch.load(
        PROJECT_ROOT / "outputs" / "models" / "poisson_pinn_trained.pt",
        weights_only=False, map_location="cpu",
    )
    results["pinn_state"] = ckpt["model_state_dict"]

    return results


def extract_r2_table(results: dict) -> dict:
    """Extract R2 values into a clean table structure."""
    table = {}
    layers = ["layer_0", "layer_1", "layer_2", "layer_3"]
    derivs = ["du_dx", "du_dy", "d2u_dx2", "d2u_dy2", "laplacian"]

    for layer_result in results["first_r2"]["results"]:
        layer = layer_result["layer_name"]
        table.setdefault(layer, {})
        table[layer]["du_dx"] = layer_result["du_dx"]["r2"]
        table[layer]["du_dy"] = layer_result["du_dy"]["r2"]

    for layer_result in results["second_r2"]["results"]:
        layer = layer_result["layer_name"]
        table.setdefault(layer, {})
        table[layer]["d2u_dx2"] = layer_result["d2u_dx2"]["r2"]
        table[layer]["d2u_dy2"] = layer_result["d2u_dy2"]["r2"]
        table[layer]["laplacian"] = layer_result["laplacian"]["r2"]

    return table


def create_summary_figure(results: dict, save_path: str):
    """
    Create a publication-quality summary figure with 6 panels.

    Panel layout:
      [A] R2 heatmap (layer x derivative)
      [B] PINN-probe weight correlation scatter (layer_0)
      [C] Top pair visualization (du/dx)
      [D] Product sign analysis
      [E] Stencil match scores
      [F] Hypothesis diagram

    Parameters
    ----------
    results : dict
        All loaded results
    save_path : str
        Path to save figure
    """
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(3, 3, hspace=0.4, wspace=0.35)

    # ---- Panel A: R2 Heatmap ----
    ax = fig.add_subplot(gs[0, 0])
    r2_table = extract_r2_table(results)
    layers = ["layer_0", "layer_1", "layer_2", "layer_3"]
    derivs = ["du_dx", "du_dy", "d2u_dx2", "d2u_dy2", "laplacian"]
    display_derivs = ["du/dx", "du/dy", "d2u/dx2", "d2u/dy2", "Lap."]

    matrix = np.array([[r2_table[l][d] for d in derivs] for l in layers])
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(derivs)))
    ax.set_xticklabels(display_derivs, fontsize=9)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"L{i}" for i in range(4)], fontsize=9)
    for i in range(len(layers)):
        for j in range(len(derivs)):
            color = "white" if matrix[i, j] < 0.3 else "black"
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold")
    plt.colorbar(im, ax=ax, label="R2", shrink=0.8)
    ax.set_title("A. Derivative Encoding (R2)", fontsize=11, fontweight="bold")

    # ---- Panel B: PINN-Probe Correlation ----
    ax = fig.add_subplot(gs[0, 1])
    W_input = results["pinn_state"]["layers.0.weight"].numpy()
    pw_dx = results["first_probes"]["layer_0"]["du_dx"]["weight"].numpy().flatten()
    pw_dy = results["first_probes"]["layer_0"]["du_dy"]["weight"].numpy().flatten()

    ax.scatter(W_input[:, 0], pw_dx, s=25, alpha=0.7, color="#2196F3", label="du/dx vs w_x")
    ax.scatter(W_input[:, 1], pw_dy, s=25, alpha=0.7, color="#FF9800", label="du/dy vs w_y")
    corr_dx = results["correlations"]["corr_wx_dudx"]
    corr_dy = results["correlations"]["corr_wy_dudy"]
    ax.set_xlabel("PINN input weight", fontsize=10)
    ax.set_ylabel("Probe weight", fontsize=10)
    ax.set_title("B. Input-Probe Correlation (L0)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.text(0.02, 0.95, f"r(w_x, p_dx)={corr_dx:.2f}\nr(w_y, p_dy)={corr_dy:.2f}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(facecolor="lightyellow", alpha=0.8))

    # ---- Panel C: Example difference pair ----
    ax = fig.add_subplot(gs[0, 2])
    # Show the top pair for du/dx
    top_pairs = results["fd_analysis"].get("top_pairs", {}).get("layer_0_du_dx", [])
    if top_pairs:
        pair = top_pairs[0]
        ni, nj = pair["neuron_i"], pair["neuron_j"]
        w_i = W_input[ni]
        b_i = results["pinn_state"]["layers.0.bias"].numpy()[ni]
        w_j = W_input[nj]
        b_j = results["pinn_state"]["layers.0.bias"].numpy()[nj]

        x_range = np.linspace(-2, 2, 200)
        z_i = w_i[0] * x_range + w_i[1] * 0.5 + b_i
        z_j = w_j[0] * x_range + w_j[1] * 0.5 + b_j

        h_i = np.tanh(z_i)
        h_j = np.tanh(z_j)
        combined = pair["probe_w_i"] * h_i + pair["probe_w_j"] * h_j

        ax.plot(x_range, pair["probe_w_i"] * h_i, color="#2196F3", alpha=0.5,
                label=f"n{ni} (pw={pair['probe_w_i']:.2f})")
        ax.plot(x_range, pair["probe_w_j"] * h_j, color="#F44336", alpha=0.5,
                label=f"n{nj} (pw={pair['probe_w_j']:.2f})")
        ax.plot(x_range, combined, color="#4CAF50", linewidth=2.5, label="Sum (derivative-like)")
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
        ax.legend(fontsize=7)
        ax.set_xlabel("x (y=0.5)", fontsize=10)
        ax.set_ylabel("Weighted activation", fontsize=10)
    ax.set_title("C. Difference Pair (du/dx, L0)", fontsize=11, fontweight="bold")

    # ---- Panel D: Product sign analysis ----
    ax = fig.add_subplot(gs[1, 0])
    product_dx = pw_dx * W_input[:, 0]
    product_dy = pw_dy * W_input[:, 1]

    frac_neg_dx = np.mean(product_dx < 0)
    frac_neg_dy = np.mean(product_dy < 0)

    categories = ["du/dx\n(p*w_x < 0)", "du/dx\n(p*w_x > 0)",
                   "du/dy\n(p*w_y < 0)", "du/dy\n(p*w_y > 0)"]
    values = [frac_neg_dx, 1 - frac_neg_dx, frac_neg_dy, 1 - frac_neg_dy]
    colors = ["#E91E63", "#FFCDD2", "#3F51B5", "#C5CAE9"]
    ax.bar(range(4), values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylabel("Fraction of neurons", fontsize=10)
    ax.set_title("D. Systematic Sign Bias", fontsize=11, fontweight="bold")
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=9, fontweight="bold")

    # ---- Panel E: Stencil match scores ----
    ax = fig.add_subplot(gs[1, 1])
    stencil = results["stencil_scores"]

    derivs_all = ["du_dx", "du_dy", "d2u_dx2", "d2u_dy2", "laplacian"]
    display_all = ["du/dx", "du/dy", "d2u/dx2", "d2u/dy2", "Lap."]
    # Get the main match score for each
    match_scores = []
    for d in derivs_all:
        s = stencil.get(d, {})
        if "mean_cos_forward" in s:
            match_scores.append(s["mean_cos_forward"])
        elif "mean_cos_sim" in s:
            match_scores.append(s["mean_cos_sim"])
        else:
            match_scores.append(0)

    bar_colors = ["#4CAF50" if v > 0.9 else "#FF9800" if v > 0.7 else "#F44336" for v in match_scores]
    ax.barh(range(len(derivs_all)), match_scores, color=bar_colors,
            edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(derivs_all)))
    ax.set_yticklabels(display_all, fontsize=10)
    ax.set_xlabel("Pattern match score", fontsize=10)
    ax.set_xlim(0, 1.1)
    ax.axvline(x=0.9, color="red", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_title("E. FD Stencil Match", fontsize=11, fontweight="bold")
    for i, v in enumerate(match_scores):
        ax.text(v + 0.02, i, f"{v:.3f}", va="center", fontsize=9)

    # ---- Panel F: Hypothesis diagram ----
    ax = fig.add_subplot(gs[1, 2])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw the computation flow
    box_props = dict(boxstyle="round,pad=0.4", facecolor="lightblue", edgecolor="black", linewidth=1.5)
    arrow_props = dict(arrowstyle="->", color="black", linewidth=2)

    ax.text(5, 9.2, "Learned Algorithm", ha="center", fontsize=12, fontweight="bold")

    # Input
    ax.annotate("Input (x, y)", xy=(5, 8.5), fontsize=10, ha="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#E3F2FD", edgecolor="black"))

    # Layer 0
    ax.annotate("", xy=(5, 7.2), xytext=(5, 8.1),
                arrowprops=dict(arrowstyle="->", color="black", linewidth=1.5))
    ax.annotate("Layer 0: Shifted tanh basis\ntanh(w_x*x + w_y*y + b)",
                xy=(5, 6.7), fontsize=8, ha="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#C8E6C9", edgecolor="black"))

    # Probe combination
    ax.annotate("", xy=(5, 5.5), xytext=(5, 6.2),
                arrowprops=dict(arrowstyle="->", color="black", linewidth=1.5))
    ax.annotate("Layers 1-3: Combine via\nopposite-sign weights",
                xy=(5, 5.0), fontsize=8, ha="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF9C4", edgecolor="black"))

    # Two branches
    ax.annotate("", xy=(2.5, 3.5), xytext=(4, 4.5),
                arrowprops=dict(arrowstyle="->", color="#4CAF50", linewidth=2))
    ax.annotate("", xy=(7.5, 3.5), xytext=(6, 4.5),
                arrowprops=dict(arrowstyle="->", color="#F44336", linewidth=2))

    ax.annotate("1st Derivatives\ndu/dx, du/dy\nR2 > 0.9\nEXPLICIT",
                xy=(2.5, 2.8), fontsize=8, ha="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#C8E6C9", edgecolor="#4CAF50", linewidth=2))

    ax.annotate("2nd Derivatives\nd2u/dx2, Lap.\nR2 ~ 0.5\nIMPLICIT (autograd)",
                xy=(7.5, 2.8), fontsize=8, ha="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFCDD2", edgecolor="#F44336", linewidth=2))

    # Bottom: output
    ax.annotate("", xy=(5, 1.2), xytext=(2.5, 2.2),
                arrowprops=dict(arrowstyle="->", color="black", linewidth=1.5))
    ax.annotate("", xy=(5, 1.2), xytext=(7.5, 2.2),
                arrowprops=dict(arrowstyle="->", color="black", linewidth=1.5))
    ax.annotate("PDE Solution u(x,y)\nL2 error < 1%",
                xy=(5, 0.7), fontsize=9, ha="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#E1BEE7", edgecolor="black", linewidth=1.5))

    ax.set_title("F. Hypothesis: Computation Flow", fontsize=11, fontweight="bold")

    # ---- Bottom row: Key numbers summary ----
    ax = fig.add_subplot(gs[2, :])
    ax.axis("off")

    summary_text = (
        "HYPOTHESIS SUMMARY: Multi-Scale Continuous Finite Difference Algorithm\n\n"
        "Evidence:  "
        f"(1) First derivatives explicitly encoded: R2={matrix[3,0]:.2f}, {matrix[3,1]:.2f} at layer 3   "
        f"(2) PINN-probe correlation: r={corr_dx:.2f} (w_x vs p_dx), {corr_dy:.2f} (w_y vs p_dy)   "
        f"(3) Systematic sign bias: {frac_neg_dx:.0%} / {frac_neg_dy:.0%} negative products\n"
        f"           "
        f"(4) FD pair match: {match_scores[0]:.3f} / {match_scores[1]:.3f} cosine similarity   "
        f"(5) [1,-2,1] triplet match: {match_scores[2]:.3f} / {match_scores[3]:.3f} / {match_scores[4]:.3f}   "
        f"(6) Multi-scale: h ranges 7-10x\n\n"
        "Conclusion: The PINN has learned a continuous, multi-scale generalization of finite differences. "
        "Layer 0 neurons create shifted tanh basis functions. Deeper layers combine them with opposite-sign weights "
        "to form difference operations. First derivatives are explicitly encoded (R2>0.9); second derivatives are computed "
        "implicitly via autograd. This is an efficient two-stage strategy: cache frequently-used first derivatives, "
        "compute higher-order derivatives on demand."
    )

    ax.text(0.02, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            va="top", ha="left", family="monospace",
            bbox=dict(facecolor="#F5F5F5", edgecolor="black", linewidth=1, pad=10))

    plt.suptitle(
        "Mechanistic Interpretability of PINN: Learned Derivative Computation Algorithm",
        fontsize=15, fontweight="bold", y=0.98,
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def write_hypothesis_document(results: dict, save_path: str):
    """
    Write the comprehensive hypothesis document.

    This is the primary deliverable of Task 4: a written analysis of
    weight patterns (required: 1-2 paragraphs) plus a preliminary
    hypothesis about the learned algorithm.

    Parameters
    ----------
    results : dict
        All loaded results
    save_path : str
        Path to save
    """
    r2_table = extract_r2_table(results)
    corr = results["correlations"]
    stencil = results["stencil_scores"]
    fd = results["fd_analysis"]

    lines = []
    lines.append("=" * 80)
    lines.append("PRELIMINARY HYPOTHESIS: THE LEARNED ALGORITHM OF A POISSON PINN")
    lines.append("Days 11-12 Probe Weight Analysis — Initial Findings")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Project: Mechanistic Interpretability for Physics-Informed Neural Networks")
    lines.append("Model: MLP PINN (4x64, tanh), trained on 2D Poisson equation")
    lines.append("       u(x,y) = sin(pi*x)*sin(pi*y), relative L2 error = 0.99%")
    lines.append("")

    # ================================================================
    # Section 1: Weight Pattern Analysis (required 1-2 paragraphs)
    # ================================================================
    lines.append("-" * 80)
    lines.append("1. WEIGHT PATTERN ANALYSIS")
    lines.append("-" * 80)
    lines.append("")
    lines.append("Analysis of the linear probe weights reveals a structured relationship")
    lines.append("between the PINN's learned representations and derivative computation.")
    lines.append("For first-derivative probes at layer 0 (the first hidden layer), the")
    lines.append("probe weights for du/dx are strongly anticorrelated with the PINN's")
    lines.append(f"input x-weights (r = {corr['corr_wx_dudx']:.2f}), while du/dy probe")
    lines.append(f"weights anticorrelate with the y-weights (r = {corr['corr_wy_dudy']:.2f}).")
    lines.append("Cross-correlations are negligible (r ~ 0.03), demonstrating clean")
    lines.append("directional specificity: the network uses separate neuron populations")
    lines.append("for each spatial derivative. The product p_i * w_{dir,i} is negative")
    lines.append(f"for {fd['product_stats']['frac_negative_dx']:.0%} (du/dx) and")
    lines.append(f"{fd['product_stats']['frac_negative_dy']:.0%} (du/dy) of neurons,")
    lines.append("revealing a systematic sign pattern consistent with the chain rule:")
    lines.append("d/dx[tanh(w*x+b)] = w * sech^2(w*x+b), where the tanh-to-derivative")
    lines.append("mapping introduces a sign flip that the probe must compensate for.")
    lines.append("")
    lines.append("Neuron pairs with similar input weight directions but different biases")
    lines.append(f"— {fd['pairs_counts']['layer_0_du_dx']} pairs for du/dx and")
    lines.append(f"{fd['pairs_counts']['layer_0_du_dy']} for du/dy — carry opposite-sign")
    lines.append("probe weights, forming continuous analogues of finite difference stencils.")
    lines.append("The weight ratios in these pairs cluster near -1.0 (mean symmetry")
    lines.append(f"scores: {stencil['du_dx'].get('mean_symmetry', 0):.2f} for du/dx,")
    lines.append(f"{stencil['du_dy'].get('mean_symmetry', 0):.2f} for du/dy),")
    lines.append("matching the symmetric central difference pattern [-1, +1]. For second")
    lines.append("derivatives, triplet patterns with the characteristic [+, -, +] sign")
    lines.append(f"structure achieve cosine similarities of {stencil['d2u_dx2'].get('mean_cos_sim', 0):.3f}")
    lines.append(f"(d2u/dx2) and {stencil['laplacian'].get('mean_cos_sim', 0):.3f} (Laplacian) against the")
    lines.append("ideal [1, -2, 1] stencil, though the middle-to-outer weight ratios")
    lines.append("(~1.3-1.8) fall short of the ideal 2.0, explaining why second derivatives")
    lines.append(f"remain only partially encoded (R2 ~ 0.5 vs R2 > 0.9 for first")
    lines.append("derivatives). Weight concentration increases with depth: the top 10")
    lines.append("neurons carry 25% of total weight at layer 0, growing to 32-40% at")
    lines.append("layer 3, indicating progressive specialization of derivative circuits.")
    lines.append("")

    # ================================================================
    # Section 2: The Hypothesis
    # ================================================================
    lines.append("-" * 80)
    lines.append("2. HYPOTHESIS: MULTI-SCALE CONTINUOUS FINITE DIFFERENCE ALGORITHM")
    lines.append("-" * 80)
    lines.append("")
    lines.append("Based on the convergent evidence from probing experiments (Day 9) and")
    lines.append("weight analysis (Day 11), we propose the following hypothesis about")
    lines.append("the computational algorithm learned by the Poisson PINN:")
    lines.append("")
    lines.append("  HYPOTHESIS: The PINN implements a two-stage, multi-scale continuous")
    lines.append("  generalization of finite differences, where:")
    lines.append("")
    lines.append("  Stage 1 — Explicit First-Derivative Encoding:")
    lines.append("    Layer 0 neurons create a bank of shifted tanh basis functions,")
    lines.append("    h_i(x,y) = tanh(w_{x,i}*x + w_{y,i}*y + b_i), that tile the")
    lines.append("    domain at multiple spatial positions and orientations. Subsequent")
    lines.append("    layers combine these with opposite-sign weights to form difference")
    lines.append("    operations, producing an explicit representation of du/dx and du/dy")
    lines.append("    that is linearly decodable (R2 > 0.9) from the final hidden layer.")
    lines.append("    This is structurally analogous to a central difference [-1, +1]/h,")
    lines.append("    but operates on smooth basis functions at multiple grid spacings")
    lines.append("    (h ranges from 0.14 to 4.2), creating a multi-resolution")
    lines.append("    derivative approximation.")
    lines.append("")
    lines.append("  Stage 2 — Implicit Second-Derivative Computation:")
    lines.append("    Second-order derivatives (d2u/dx2, d2u/dy2, Laplacian) are NOT")
    lines.append("    explicitly stored in the activations (R2 ~ 0.3-0.5). Instead,")
    lines.append("    they are computed on-demand by PyTorch's autograd during the")
    lines.append("    backward pass, differentiating through the first-derivative")
    lines.append("    representations. Partial [1,-2,1] triplet patterns exist in the")
    lines.append("    weights (cosine similarity > 0.97) but with imperfect magnitudes,")
    lines.append("    indicating the network encodes partial second-derivative information")
    lines.append("    while relying on autograd for the complete computation.")
    lines.append("")
    lines.append("  This two-stage strategy is computationally efficient: only 2 values")
    lines.append("  (du/dx, du/dy) are cached in activations, while 3+ higher-order")
    lines.append("  derivatives are computed on-demand via the chain rule. The network")
    lines.append("  discovered this efficient allocation automatically through gradient")
    lines.append("  descent, analogous to how a human might memorize multiplication tables")
    lines.append("  (frequently-used values) while computing rare products by hand.")
    lines.append("")

    # ================================================================
    # Section 3: Evidence Summary
    # ================================================================
    lines.append("-" * 80)
    lines.append("3. EVIDENCE SUMMARY")
    lines.append("-" * 80)
    lines.append("")
    lines.append("Evidence supporting each component of the hypothesis:")
    lines.append("")
    lines.append("+----------------------------------+---------------------------------------+")
    lines.append("| Hypothesis Component             | Supporting Evidence                   |")
    lines.append("+----------------------------------+---------------------------------------+")
    lines.append(f"| First derivs explicitly encoded  | R2(du/dx, L3) = {r2_table['layer_3']['du_dx']:.3f}              |")
    lines.append(f"|                                  | R2(du/dy, L3) = {r2_table['layer_3']['du_dy']:.3f}              |")
    lines.append("+----------------------------------+---------------------------------------+")
    lines.append(f"| Second derivs implicitly computed| R2(Lap., L3)  = {r2_table['layer_3']['laplacian']:.3f}              |")
    lines.append(f"|                                  | R2(d2u/dx2)   = {r2_table['layer_3']['d2u_dx2']:.3f}              |")
    lines.append("+----------------------------------+---------------------------------------+")
    lines.append(f"| Directional specificity          | corr(w_x, p_dx) = {corr['corr_wx_dudx']:.3f}             |")
    lines.append(f"|                                  | corr(w_y, p_dy) = {corr['corr_wy_dudy']:.3f}             |")
    lines.append(f"|                                  | cross-corr ~ {corr['corr_wx_dudy']:.3f}                  |")
    lines.append("+----------------------------------+---------------------------------------+")
    lines.append(f"| FD-like difference pairs         | {fd['pairs_counts']['layer_0_du_dx']} pairs (du/dx), {fd['pairs_counts']['layer_0_du_dy']} pairs (du/dy)  |")
    lines.append(f"|                                  | symmetry ~ {stencil['du_dx'].get('mean_symmetry', 0):.2f}-{stencil['du_dy'].get('mean_symmetry', 0):.2f}                    |")
    lines.append("+----------------------------------+---------------------------------------+")
    lines.append(f"| [1,-2,1] triplet patterns        | cos_sim to ideal: {stencil['d2u_dx2'].get('mean_cos_sim', 0):.3f}-{stencil['laplacian'].get('mean_cos_sim', 0):.3f}       |")
    lines.append(f"|                                  | but ratios ~1.3-1.8 (not 2.0)        |")
    lines.append("+----------------------------------+---------------------------------------+")
    lines.append(f"| Systematic sign in p*w products  | {fd['product_stats']['frac_negative_dx']:.0%} negative (du/dx)               |")
    lines.append(f"|                                  | {fd['product_stats']['frac_negative_dy']:.0%} negative (du/dy)               |")
    lines.append("+----------------------------------+---------------------------------------+")
    lines.append(f"| Multi-scale grid spacings        | h range: 0.14 - 4.2 (7-10x)          |")
    lines.append("|                                  | Not single-scale like classical FD    |")
    lines.append("+----------------------------------+---------------------------------------+")
    lines.append(f"| Progressive specialization       | Top-10 weight fraction:               |")
    lines.append(f"|                                  | L0: 25% -> L3: 32-40%                 |")
    lines.append("+----------------------------------+---------------------------------------+")
    lines.append("")

    # ================================================================
    # Section 4: Comparison to Classical Methods
    # ================================================================
    lines.append("-" * 80)
    lines.append("4. COMPARISON TO CLASSICAL NUMERICAL METHODS")
    lines.append("-" * 80)
    lines.append("")
    lines.append("The learned algorithm can be positioned within the landscape of")
    lines.append("classical numerical methods for PDEs:")
    lines.append("")
    lines.append("  Finite Differences (FD):")
    lines.append("    Similarity:  Difference of nearby function evaluations")
    lines.append("    Difference:  PINN uses smooth tanh basis, not point evaluations")
    lines.append("    Difference:  PINN uses multiple grid spacings simultaneously")
    lines.append("    Implication: A 'soft' or 'continuous' version of FD")
    lines.append("")
    lines.append("  Spectral Methods:")
    lines.append("    Similarity:  Global basis functions (tanh ≈ shifted sigmoid)")
    lines.append("    Similarity:  Smooth, differentiable basis functions")
    lines.append("    Difference:  tanh is not an eigenfunction of differential operators")
    lines.append("    Implication: Resembles a learned spectral basis")
    lines.append("")
    lines.append("  Radial Basis Function (RBF) Methods:")
    lines.append("    Similarity:  Shifted basis functions centered at different locations")
    lines.append("    Similarity:  Multi-scale resolution (different widths)")
    lines.append("    Difference:  tanh is not radially symmetric")
    lines.append("    Implication: Closest classical analogue may be RBF-FD")
    lines.append("")
    lines.append("  Multi-Resolution / Wavelet Methods:")
    lines.append("    Similarity:  Multiple grid spacings operating simultaneously")
    lines.append("    Similarity:  Different neurons sensitive to different scales")
    lines.append("    Difference:  No explicit scale decomposition")
    lines.append("    Implication: The network naturally develops multi-scale computation")
    lines.append("")
    lines.append("  BEST DESCRIPTION: The learned algorithm is a 'multi-scale RBF-FD")
    lines.append("  hybrid' — it uses shifted smooth basis functions (like RBF) combined")
    lines.append("  via difference operations (like FD) at multiple resolution scales")
    lines.append("  (like wavelets). This was discovered automatically by gradient descent.")
    lines.append("")

    # ================================================================
    # Section 5: Relation to Research Hypotheses
    # ================================================================
    lines.append("-" * 80)
    lines.append("5. RELATION TO ORIGINAL RESEARCH HYPOTHESES")
    lines.append("-" * 80)
    lines.append("")
    lines.append("  Hypothesis 1 (from CLAUDE.md): 'Early layers develop circuits")
    lines.append("  approximating local derivatives using weighted combinations of")
    lines.append("  nearby input coordinates (finite-difference-like patterns).'")
    lines.append("")
    lines.append("  STATUS: PARTIALLY CONFIRMED")
    lines.append("    - YES: Layers develop derivative computation circuits")
    lines.append("    - YES: Weighted combinations of shifted functions ≈ finite differences")
    lines.append("    - NUANCE: Not 'nearby input coordinates' directly, but nearby")
    lines.append("      *basis function evaluations* (tanh at shifted positions)")
    lines.append("    - NUANCE: 'Early layers' is partially correct — derivative info")
    lines.append("      is present from layer 0 (R2 ~ 0.78) but peaks at layer 3 (R2 ~ 0.91)")
    lines.append("    - SURPRISE: The computation is multi-scale, not single-scale")
    lines.append("    - SURPRISE: Second derivatives are NOT computed via FD but via autograd")
    lines.append("")

    # ================================================================
    # Section 6: Predictions and Next Steps
    # ================================================================
    lines.append("-" * 80)
    lines.append("6. PREDICTIONS AND TESTABLE IMPLICATIONS")
    lines.append("-" * 80)
    lines.append("")
    lines.append("If this hypothesis is correct, we predict:")
    lines.append("")
    lines.append("  P1: Removing layer 3 neurons with high |probe weight| for du/dx")
    lines.append("      should degrade the PDE solution quality, while removing neurons")
    lines.append("      with low |probe weight| should have minimal effect.")
    lines.append("      -> Testable via activation patching (Week 3)")
    lines.append("")
    lines.append("  P2: A wider network (more neurons) should allow more FD pairs and")
    lines.append("      potentially achieve higher R2 for second derivatives, by providing")
    lines.append("      more basis functions at different positions.")
    lines.append("      -> Testable via architecture comparison (Week 3-4)")
    lines.append("")
    lines.append("  P3: A deeper network (more layers) should show second derivative")
    lines.append("      information emerging at intermediate layers, as additional")
    lines.append("      layers can compose first-derivative representations.")
    lines.append("      -> Testable via probing deeper architectures")
    lines.append("")
    lines.append("  P4: Modified Fourier Networks (MFN), which use sinusoidal basis")
    lines.append("      functions instead of tanh, should show different stencil patterns")
    lines.append("      — potentially closer to spectral methods than FD.")
    lines.append("      -> Testable via MFN probe analysis (Week 3-4)")
    lines.append("")
    lines.append("  P5: For the Heat equation (time-dependent), the network should")
    lines.append("      develop separate temporal and spatial derivative circuits,")
    lines.append("      with temporal derivatives (du/dt) potentially encoded differently")
    lines.append("      from spatial derivatives (du/dx).")
    lines.append("      -> Testable via Heat equation probing (Week 3)")
    lines.append("")

    # ================================================================
    # Section 7: Limitations
    # ================================================================
    lines.append("-" * 80)
    lines.append("7. LIMITATIONS AND CAVEATS")
    lines.append("-" * 80)
    lines.append("")
    lines.append("  L1: Analysis is based on LINEAR probes. Non-linear probes might")
    lines.append("      reveal additional encoded information that linear probes miss.")
    lines.append("      Second derivatives might be encoded non-linearly.")
    lines.append("")
    lines.append("  L2: Results are for a single architecture (4x64 MLP, tanh) on a")
    lines.append("      single PDE (2D Poisson). Generalization to other architectures")
    lines.append("      and equations remains to be tested.")
    lines.append("")
    lines.append("  L3: The 'multi-scale FD' interpretation is a functional description.")
    lines.append("      The actual computation involves all 64 neurons simultaneously,")
    lines.append("      not isolated pairs/triplets. The pair analysis identifies")
    lines.append("      structure within a fundamentally distributed computation.")
    lines.append("")
    lines.append("  L4: Cosine similarity to ideal stencils is high (~0.98), but this")
    lines.append("      could partially reflect the mathematical constraints of the")
    lines.append("      [+, -, +] sign pattern rather than deliberate FD computation.")
    lines.append("")
    lines.append("  L5: The PINN was trained with a specific configuration (Adam, 20K")
    lines.append("      epochs, lr=1e-3). Different training hyperparameters might lead")
    lines.append("      to different computational strategies.")
    lines.append("")

    lines.append("=" * 80)
    lines.append("END OF HYPOTHESIS DOCUMENT")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Generated: Day 11 of Mechanistic Interpretability for PINNs project")
    lines.append("Based on: Day 9 probing results + Day 11 weight analysis")

    document = "\n".join(lines)
    with open(save_path, "w") as f:
        f.write(document)
    print(f"  Saved: {save_path}")
    return document


def main():
    output_dir = PROJECT_ROOT / "outputs" / "day11_probe_weights"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Day 11 Task 4: Document Learned Algorithm Hypothesis")
    print("=" * 60)

    # Load all results
    print("\n1. Loading all results from Days 9-11...")
    results = load_all_results()

    # Generate summary figure
    print("\n2. Creating publication-quality summary figure...")
    create_summary_figure(results, str(output_dir / "hypothesis_summary_figure.png"))

    # Write hypothesis document
    print("\n3. Writing hypothesis document...")
    document = write_hypothesis_document(
        results, str(output_dir / "learned_algorithm_hypothesis.txt")
    )

    # Print key excerpts
    print("\n" + "=" * 60)
    print("HYPOTHESIS (condensed)")
    print("=" * 60)
    print()
    print("The PINN implements a TWO-STAGE, MULTI-SCALE CONTINUOUS")
    print("GENERALIZATION OF FINITE DIFFERENCES:")
    print()
    print("  Stage 1: First derivatives (du/dx, du/dy) are EXPLICITLY")
    print("    encoded in hidden layer activations using shifted tanh")
    print("    basis functions combined with opposite-sign weights.")
    print(f"    Evidence: R2 > 0.9, corr(w_x,p_dx) = {results['correlations']['corr_wx_dudx']:.2f}")
    print()
    print("  Stage 2: Second derivatives (d2u/dx2, Laplacian) are")
    print("    IMPLICITLY computed via autograd during backpropagation.")
    print(f"    Evidence: R2 ~ 0.5, partial [1,-2,1] patterns")
    print()
    print("  Key properties:")
    print("    - Multi-scale (h ranges 0.14 to 4.2, 7-10x)")
    print("    - Continuous basis (smooth tanh, not point evaluations)")
    print("    - Directionally specific (separate neuron populations)")
    print("    - Computationally efficient (cache 2 values, compute 3+)")
    print()
    print("  Best classical analogue: Multi-scale RBF-FD hybrid")
    print()

    # Count all output files
    all_files = list(output_dir.glob("*"))
    total_size = sum(f.stat().st_size for f in all_files if f.is_file())
    print(f"Total Day 11 output files: {len([f for f in all_files if f.is_file()])}")
    print(f"Total size: {total_size / 1024:.1f} KB")
    print("\nTask 4 complete! All Day 11 tasks finished.")


if __name__ == "__main__":
    main()
