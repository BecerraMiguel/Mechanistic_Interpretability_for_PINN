# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project applying mechanistic interpretability techniques to Physics-Informed Neural Networks (PINNs). The goal is to reverse-engineer the computational algorithms that PINNs learn when solving differential equations, using techniques like activation patching, probing classifiers, and circuit analysis.

**Core Research Question:** What computational mechanisms do neural networks develop when learning to solve differential equations, and how do these mechanisms relate to classical numerical methods?

## Project Structure

```
mechinterp-pinns/
├── src/
│   ├── models/          # PINN architectures (MLP, Modified Fourier Network, Attention-Enhanced)
│   ├── problems/        # PDE problem definitions (Poisson, Heat, Burgers, Helmholtz)
│   ├── training/        # Training loops, loss computation, W&B integration
│   ├── interpretability/
│   │   ├── activation_store.py  # HDF5-based activation extraction and storage
│   │   ├── probing.py           # Linear probes for derivative detection
│   │   ├── patching.py          # Activation patching experiments
│   │   └── circuits.py          # Circuit analysis and ACDC
│   └── utils/           # Derivative computation, visualization helpers
├── tests/               # pytest test suite
├── notebooks/           # Jupyter analysis notebooks
├── configs/             # Experiment configuration files
├── data/                # Raw PDE data, processed activations
└── outputs/             # Figures, models, experiment results
```

## Key Technical Concepts

### PINN Formulation

PINNs approximate solutions u(x,t) to PDEs by minimizing:
```
L = w_pde * L_pde + w_bc * L_bc + w_ic * L_ic
```

where:
- `L_pde`: PDE residual at interior collocation points
- `L_bc`: Boundary condition violations
- `L_ic`: Initial condition errors
- Derivatives in N[u] are computed via automatic differentiation

### Three PINN Architectures

1. **Standard MLP**: 4-8 hidden layers, 50-200 neurons/layer, tanh activation
2. **Modified Fourier Network (MFN)**: Fourier feature embedding followed by MLP
3. **Attention-Enhanced PINN**: Self-attention layers interspersed with MLPs

### Mechanistic Interpretability Toolkit

1. **Activation Patching**: Replace activations from one input with another to identify causal components
2. **Probing Classifiers**: Train linear probes on intermediate activations to detect numerical derivatives (du/dx, du/dt, d2u/dx2, Laplacian)
3. **Circuit Analysis**: Decompose network into interpretable subgraphs (derivative circuits, boundary enforcement circuits)
4. **ACDC**: Automated Circuit Discovery using iterative edge pruning

### PDE Test Suite Progression

1. **Poisson Equation** (elliptic, steady-state): Focus on derivative computation, boundary handling
2. **Heat Equation** (parabolic, diffusion): Focus on temporal integration, smoothing
3. **Burgers Equation** (nonlinear, shock formation): Focus on nonlinearity handling, multi-scale
4. **Helmholtz Equation** (wave propagation, oscillatory): Focus on spectral bias, frequency representation

## Development Commands

### Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify PyTorch with CUDA
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/models/test_base.py

# Run with coverage report
pytest --cov=src --cov-report=html

# Run tests matching pattern
pytest -k "probing"
```

### Training PINNs

```bash
# Train Poisson equation (basic)
python src/training/trainer.py --problem poisson --config configs/poisson_basic.yaml

# Train with W&B logging
python src/training/trainer.py --problem heat --wandb --project mechinterp-pinns

# Resume from checkpoint
python src/training/trainer.py --resume outputs/checkpoints/model_epoch_100.pt
```

### Running Interpretability Analysis

```bash
# Extract activations from trained model
python src/interpretability/activation_store.py --model outputs/models/poisson_mlp.pt --grid 100x100

# Train probing classifiers
python src/interpretability/probing.py --activations data/activations/poisson_mlp.h5 --targets derivatives

# Run activation patching experiments
python src/interpretability/patching.py --model outputs/models/heat_mlp.pt --experiment temporal
```

### Code Quality

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Type checking (if using type hints)
mypy src/
```

## Architecture Patterns

### BasePINN Interface

All PINN models inherit from `BasePINN` in `src/models/base.py`:
- `forward(x)`: Network forward pass
- `compute_pde_residual(x)`: Compute PDE loss term
- `train()`: Full training method with loss decomposition

### Activation Storage

Activations are stored in HDF5 format with structure:
```
/coordinates (N, 2)          # Input (x, y) coordinates
/layer_0 (N, hidden_dim)     # First hidden layer activations
/layer_1 (N, hidden_dim)     # Second hidden layer activations
...
```

Access via `ActivationStore.load_layer(layer_name)`.

### Probing Workflow

1. Train PINN on PDE → save model
2. Extract activations on dense grid → save to HDF5
3. Compute ground-truth derivatives analytically
4. Train LinearProbe for each (layer, derivative_type) pair
5. Analyze R-squared scores to identify where derivative information emerges

## Research Hypotheses

**Hypothesis 1: Local Derivative Circuits**
Early layers develop circuits approximating local derivatives using weighted combinations of nearby input coordinates (finite-difference-like patterns).

**Hypothesis 2: Spectral Bias in Initialization**
The well-documented spectral bias (preference for low-frequency solutions) results from smoothness prior induced by random initialization + gradient descent. High-frequency circuits require larger weight magnitudes.

**Hypothesis 3: Boundary Enforcement via Specialized Subnetworks**
PINNs develop distinct computational pathways for interior PDE residual minimization vs. boundary condition enforcement.

## Success Criteria

The project is successful if we achieve at least two of:
- Identify at least one interpretable circuit corresponding to a known numerical operation (e.g., central difference derivative)
- Provide mechanistic explanation for spectral bias with intervention predictions (e.g., specific weight modifications that alleviate bias)
- Train probing classifiers extracting derivatives from intermediate layers with R² > 0.9
- Identify architectural modifications improving PINN performance, motivated by mechanistic findings

## Key Dependencies

- PyTorch ≥2.0.0 (automatic differentiation for PDE residuals)
- NumPy ≥1.24.0 (numerical computations)
- SciPy ≥1.10.0 (analytical solutions for validation)
- matplotlib ≥3.7.0 (visualization)
- h5py ≥3.8.0 (activation storage)
- wandb ≥0.15.0 (experiment tracking)
- pytest ≥7.3.0 (testing)
- pytest-cov ≥4.0.0 (coverage)
- black ≥23.0.0 (formatting)
- isort ≥5.12.0 (import sorting)

## Important Notes

- **Automatic Differentiation**: All derivatives (du/dx, d2u/dx2, etc.) are computed via `torch.autograd.grad()`. Never use finite differences for PDE residual computation.
- **Collocation Points**: Use Latin Hypercube sampling for interior points (better coverage than uniform grid), uniform/grid sampling for boundary points.
- **Activation Extraction**: Always extract on a dense regular grid (e.g., 100×100) even if training used random collocation points. This enables spatial analysis.
- **Probing Target**: Ground-truth derivatives for probing must come from analytical solutions, not from the PINN's own predictions.
- **Test Coverage**: Aim for >80% coverage. Place tests in `tests/` mirroring `src/` structure.
- **Reproducibility**: Set random seeds in training configs. Save all hyperparameters with model checkpoints.

## Common Pitfalls

- **Gradient Flow**: When extracting activations, ensure `requires_grad=True` for input coordinates, otherwise derivatives cannot be computed.
- **Spectral Bias**: Standard MLPs struggle with high-frequency components. If training fails on Helmholtz equation, try Modified Fourier Network or increase hidden dimensions.
- **Memory**: Activation storage for large grids can exceed RAM. Use `ActivationStore.extract_on_grid()` with batching.
- **Boundary Points**: Ensure boundary points lie exactly on domain edges, not approximately. Use `torch.linspace()` or explicit edge values.
