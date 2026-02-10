"""
Demo: Extract activations from trained Poisson PINN and visualize.

This script demonstrates the complete activation extraction workflow:
1. Load trained Poisson PINN model
2. Extract activations on 100x100 grid
3. Save to HDF5 file
4. Visualize sample neurons
"""

import torch
from pathlib import Path
from src.interpretability import ActivationStore, extract_activations_from_model
from src.models import MLP


def main():
    print("=" * 70)
    print("Activation Extraction Demo")
    print("=" * 70)

    # Load trained model
    print("\n1. Loading trained Poisson PINN model...")
    model_path = Path("poisson_pinn_trained.pt")

    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        print("Please train a model first using demo_training_pipeline.py")
        return

    checkpoint = torch.load(model_path, map_location='cpu')

    # Reconstruct model from config
    config = checkpoint['config']
    model = MLP(
        input_dim=2,  # Poisson 2D problem
        output_dim=1,  # Single scalar output
        hidden_dims=config['hidden_dims'],
        activation=config['activation']
    )

    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"   ✓ Model loaded:")
    print(f"      Architecture: {config['architecture']}")
    print(f"      Hidden dims: {config['hidden_dims']}")
    print(f"      Activation: {config['activation']}")
    print(f"      Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"      Final training error: {checkpoint['final_error']:.4f}%")

    # Extract activations
    print("\n2. Extracting activations on 100x100 grid...")
    save_path = "data/activations/poisson_mlp_100x100.h5"

    store = extract_activations_from_model(
        model,
        save_path,
        grid_resolution=100,
        domain_bounds=((0.0, 1.0), (0.0, 1.0)),
        batch_size=1000,
        device='cpu'
    )

    print(f"\n   ✓ Extraction complete!")

    # Display metadata
    print("\n3. Activation Store Metadata:")
    metadata = store.get_metadata()
    print(f"   Grid resolution: {metadata['grid_resolution']}x{metadata['grid_resolution']}")
    print(f"   Total points: {metadata['n_points']}")
    print(f"   Input dimension: {metadata['input_dim']}")
    print(f"   Layers stored: {', '.join(metadata['layer_names'])}")
    print(f"   File size: {metadata['file_size_kb']:.1f} KB")

    # Load and inspect a layer
    print("\n4. Inspecting layer activations:")
    for layer_name in metadata['layer_names'][:2]:  # First 2 layers
        activations = store.load_layer(layer_name)
        print(f"   {layer_name}:")
        print(f"      Shape: {activations.shape}")
        print(f"      Mean: {activations.mean():.4f}")
        print(f"      Std: {activations.std():.4f}")
        print(f"      Min: {activations.min():.4f}")
        print(f"      Max: {activations.max():.4f}")

    # Visualize sample neurons
    print("\n5. Generating visualizations...")
    output_dir = Path("outputs/day5_activations")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Single neuron heatmap
    print("   Creating single neuron heatmap...")
    fig = store.visualize_neuron(
        'layer_0',
        neuron_idx=5,
        save_path=str(output_dir / "neuron_layer0_idx5.png"),
        figsize=(8, 6),
        dpi=150
    )

    # Layer summary
    print("   Creating layer summary (16 neurons)...")
    fig_summary = store.visualize_layer_summary(
        'layer_0',
        n_neurons=16,
        save_path=str(output_dir / "layer0_summary.png"),
        figsize=(16, 12),
        dpi=150
    )

    print(f"\n   ✓ Visualizations saved to {output_dir}/")

    # Summary
    print("\n" + "=" * 70)
    print("✓ Demo Complete!")
    print("=" * 70)
    print(f"\nOutputs:")
    print(f"  HDF5 file: {save_path}")
    print(f"  Visualizations: {output_dir}/")
    print(f"\nNext steps:")
    print(f"  - Explore other neurons and layers")
    print(f"  - Use activations for probing experiments")
    print(f"  - Compare activations across different models")


if __name__ == "__main__":
    main()
