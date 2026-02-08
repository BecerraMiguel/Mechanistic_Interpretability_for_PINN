# Mechanistic Interpretability for Physics-Informed Neural Networks

A research project applying mechanistic interpretability techniques to understand the computational mechanisms learned by Physics-Informed Neural Networks when solving partial differential equations.

## Project Overview

This project aims to reverse-engineer the algorithms implemented by PINNs using techniques such as activation patching, probing classifiers, and circuit analysis. The goal is to identify interpretable computational graphs corresponding to numerical methods and provide mechanistic explanations for known PINN failure modes such as spectral bias.

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Mechanistic_Interpretability_for_PINN
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Verify PyTorch installation:
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## Project Structure

```
Mechanistic_Interpretability_for_PINN/
├── src/
│   ├── models/              # PINN architectures
│   ├── problems/            # PDE problem definitions
│   ├── training/            # Training loops and utilities
│   ├── interpretability/    # Mechanistic interpretability tools
│   └── utils/               # Helper functions
├── tests/                   # Unit tests
├── notebooks/               # Jupyter notebooks for analysis
├── configs/                 # Experiment configurations
├── data/                    # Dataset storage
│   ├── raw/
│   └── processed/
└── outputs/                 # Results and artifacts
    ├── figures/
    ├── models/
    └── activations/
```

## Usage

### Training a PINN

```bash
python src/training/trainer.py --problem poisson --config configs/poisson_basic.yaml
```

### Running Tests

```bash
pytest
pytest --cov=src --cov-report=html
```

## Research Objectives

1. Identify interpretable circuits corresponding to numerical methods
2. Explain PINN failure modes through mechanistic analysis
3. Discover novel computational strategies learned by networks

## License

This project is for research purposes.
