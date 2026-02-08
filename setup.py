from setuptools import setup, find_packages

setup(
    name="mechinterp-pinns",
    version="0.1.0",
    description="Mechanistic Interpretability for Physics-Informed Neural Networks",
    author="Miguel",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "h5py>=3.8.0",
        "wandb>=0.15.0",
        "pytest>=7.3.0",
        "pytest-cov>=4.0.0",
        "black>=23.0.0",
        "isort>=5.12.0",
        "tqdm>=4.65.0",
    ],
    python_requires=">=3.8",
)
