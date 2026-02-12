#!/usr/bin/env python3
"""Fix error percentages in JSON and regenerate comparison table + figures."""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "outputs/day13_architecture_comparison"
JSON_PATH = os.path.join(OUTPUT_DIR, "all_probe_results.json")

DERIVATIVE_TARGETS = ["du_dx", "du_dy", "d2u_dx2", "d2u_dy2", "laplacian"]

# Correct error values from trainer output (already percentage)
CORRECT_ERRORS = {
    "baseline_4L64_tanh": 0.995,
    "shallow_2L": 7.67,
    "deep_6L": 7.87,
    "wide_128": 5.93,
    "narrow_32": 10.43,
    "relu_4L": 100.21,
}


def main():
    with open(JSON_PATH) as f:
        data = json.load(f)

    # Fix error values to be fractional
    for arch, err_pct in CORRECT_ERRORS.items():
        data[arch]["training_info"]["relative_l2_error"] = err_pct / 100.0

    with open(JSON_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print("Fixed JSON error values.")

    # Regenerate comparison table
    lines = []
    lines.append("=" * 95)
    lines.append("ARCHITECTURE COMPARISON: Peak R-squared for Each Derivative")
    lines.append("=" * 95)
    lines.append("")
    header = f"{'Architecture':<22} {'Params':>7} {'du/dx':>8} {'du/dy':>8} {'d2u/dx2':>8} {'d2u/dy2':>8} {'Lap':>8} {'L2 Err%':>8}"
    lines.append(header)
    lines.append("-" * 95)

    for arch_name, info in data.items():
        probe_results = info["probe_results"]
        ti = info["training_info"]
        peak_r2 = {}
        for deriv in DERIVATIVE_TARGETS:
            best = max(
                probe_results[layer][deriv]["r_squared"]
                for layer in probe_results
            )
            peak_r2[deriv] = best

        err_pct = ti["relative_l2_error"] * 100
        row = (
            f"{arch_name:<22} "
            f"{ti['n_params']:>7,} "
            f"{peak_r2['du_dx']:>8.4f} "
            f"{peak_r2['du_dy']:>8.4f} "
            f"{peak_r2['d2u_dx2']:>8.4f} "
            f"{peak_r2['d2u_dy2']:>8.4f} "
            f"{peak_r2['laplacian']:>8.4f} "
            f"{err_pct:>7.1f}%"
        )
        lines.append(row)

    lines.append("-" * 95)
    lines.append("")
    lines.append("KEY OBSERVATIONS:")
    lines.append("")

    # Two-stage pattern analysis
    for arch_name, info in data.items():
        probe_results = info["probe_results"]
        first_r2 = []
        second_r2 = []
        for layer in probe_results:
            first_r2.extend([
                probe_results[layer]["du_dx"]["r_squared"],
                probe_results[layer]["du_dy"]["r_squared"],
            ])
            second_r2.extend([
                probe_results[layer]["d2u_dx2"]["r_squared"],
                probe_results[layer]["d2u_dy2"]["r_squared"],
            ])

        avg_first = np.mean(first_r2)
        avg_second = np.mean(second_r2)
        gap = avg_first - avg_second
        pattern = "YES" if gap > 0.1 else "WEAK" if gap > 0 else "NO"
        lines.append(
            f"  {arch_name:<22}: "
            f"avg 1st={avg_first:.3f}, avg 2nd={avg_second:.3f}, "
            f"gap={gap:.3f}  -> Two-stage pattern: {pattern}"
        )

    lines.append("")
    lines.append("INTERPRETATION:")
    lines.append("")
    lines.append("  1. The two-stage pattern (1st derivs >> 2nd derivs) holds for ALL tanh architectures,")
    lines.append("     regardless of depth (2-6 layers), width (32-128 neurons), or training quality.")
    lines.append("  2. ReLU completely fails to learn the Poisson problem (100% error), confirming")
    lines.append("     that smooth activations (tanh) are critical for PINN derivative computation.")
    lines.append("  3. The baseline (well-trained, 4L/64/tanh) achieves the best probe R-squared,")
    lines.append("     suggesting model accuracy improves derivative encoding quality.")
    lines.append("  4. Wider networks (128 neurons) achieve similar peak R-squared to baseline")
    lines.append("     despite less training, suggesting width helps derivative encoding.")
    lines.append("  5. Deeper networks (6L) show more gradual emergence but similar peak values,")
    lines.append("     confirming that the derivative computation distributes across available layers.")

    text = "\n".join(lines)
    print(text)

    with open(os.path.join(OUTPUT_DIR, "architecture_comparison.txt"), "w") as f:
        f.write(text)

    # ---- Regenerate figures ----
    arch_names = list(data.keys())
    n_arch = len(arch_names)

    short_names = {
        "baseline_4L64_tanh": "Baseline\n4L/64/tanh",
        "shallow_2L": "Shallow\n2L/64/tanh",
        "deep_6L": "Deep\n6L/64/tanh",
        "wide_128": "Wide\n4L/128/tanh",
        "narrow_32": "Narrow\n4L/32/tanh",
        "relu_4L": "ReLU\n4L/64/relu",
    }
    short_list = [short_names.get(n, n) for n in arch_names]

    # Figure 1: Peak R² bar chart + first vs second comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    x = np.arange(n_arch)
    width = 0.15
    colors_deriv = ["#2196F3", "#03A9F4", "#FF5722", "#FF9800", "#9C27B0"]
    label_map = {
        "du_dx": "du/dx", "du_dy": "du/dy",
        "d2u_dx2": "d2u/dx2", "d2u_dy2": "d2u/dy2",
        "laplacian": "Laplacian",
    }

    for i, deriv in enumerate(DERIVATIVE_TARGETS):
        peak_vals = []
        for arch in arch_names:
            probe_results = data[arch]["probe_results"]
            best = max(
                probe_results[layer][deriv]["r_squared"]
                for layer in probe_results
            )
            peak_vals.append(best)
        ax.bar(x + i * width, peak_vals, width,
               label=label_map[deriv], color=colors_deriv[i], alpha=0.85)

    ax.set_xlabel("Architecture", fontsize=11)
    ax.set_ylabel("Peak R-squared", fontsize=11)
    ax.set_title("Peak Derivative R-squared by Architecture", fontsize=12)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(short_list, fontsize=8)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(-1.5, 1.1)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

    # Panel 2: first vs second at final layer
    ax = axes[1]
    first_avgs = []
    second_avgs = []

    for arch in arch_names:
        probe_results = data[arch]["probe_results"]
        layers = sorted(probe_results.keys())
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

    bar_width = 0.35
    ax.bar(x - bar_width / 2, first_avgs, bar_width,
           label="1st deriv (final layer)", color="#2196F3")
    ax.bar(x + bar_width / 2, second_avgs, bar_width,
           label="2nd deriv (final layer)", color="#FF5722")

    ax.set_xlabel("Architecture", fontsize=11)
    ax.set_ylabel("R-squared (final layer)", fontsize=11)
    ax.set_title("First vs Second Derivative Encoding (Final Layer)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(short_list, fontsize=8)
    ax.legend(fontsize=9)
    ax.set_ylim(-1.5, 1.1)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "architecture_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved architecture_comparison.png")

    # Figure 2: Layer progression for 3 key derivatives
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    deriv_groups = [
        ("du_dx", "du/dx (1st order)"),
        ("d2u_dx2", "d2u/dx2 (2nd order)"),
        ("laplacian", "Laplacian"),
    ]
    colors = plt.cm.tab10(np.linspace(0, 1, n_arch))

    for ax, (deriv_key, deriv_label) in zip(axes, deriv_groups):
        for idx, arch in enumerate(arch_names):
            probe_results = data[arch]["probe_results"]
            layers = sorted(probe_results.keys())
            r2_vals = [probe_results[l][deriv_key]["r_squared"] for l in layers]
            layer_indices = list(range(len(layers)))
            ax.plot(layer_indices, r2_vals,
                    marker="o", linewidth=2, markersize=6,
                    color=colors[idx],
                    label=short_list[idx].replace("\n", " "))

        ax.set_xlabel("Layer Index", fontsize=11)
        ax.set_ylabel("R-squared", fontsize=11)
        ax.set_title(deriv_label, fontsize=12)
        ax.legend(fontsize=7, loc="lower right")
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
        ax.set_ylim(-1.5, 1.05)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "layer_progression_by_architecture.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved layer_progression_by_architecture.png")

    print("\nDone! All outputs regenerated.")


if __name__ == "__main__":
    main()
