"""
Tests for activation extraction and storage.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import h5py
from pathlib import Path
import tempfile
import shutil
from src.interpretability.activation_store import (
    ActivationStore,
    extract_activations_from_model
)
from src.models.mlp import MLP


class TestActivationStoreInit:
    """Test ActivationStore initialization."""

    def test_init_creates_parent_dirs(self, tmp_path):
        """Test that initialization creates parent directories."""
        save_path = tmp_path / "subdir" / "activations.h5"
        store = ActivationStore(str(save_path))
        assert store.save_path.parent.exists()

    def test_init_sets_path(self, tmp_path):
        """Test that path is correctly set."""
        save_path = tmp_path / "activations.h5"
        store = ActivationStore(str(save_path))
        assert store.save_path == save_path


class TestGridGeneration:
    """Test grid generation methods."""

    def test_generate_grid_2d(self):
        """Test 2D grid generation."""
        store = ActivationStore("dummy.h5")
        grid = store._generate_grid(
            resolution=10,
            domain_bounds=((0.0, 1.0), (0.0, 1.0))
        )

        # Check shape
        assert grid.shape == (100, 2)

        # Check bounds
        assert grid[:, 0].min() == pytest.approx(0.0)
        assert grid[:, 0].max() == pytest.approx(1.0)
        assert grid[:, 1].min() == pytest.approx(0.0)
        assert grid[:, 1].max() == pytest.approx(1.0)

    def test_generate_grid_1d(self):
        """Test 1D grid generation."""
        store = ActivationStore("dummy.h5")
        grid = store._generate_grid(
            resolution=50,
            domain_bounds=((0.0, 2.0),)
        )

        assert grid.shape == (50, 1)
        assert grid[:, 0].min() == pytest.approx(0.0)
        assert grid[:, 0].max() == pytest.approx(2.0)

    def test_generate_grid_3d(self):
        """Test 3D grid generation."""
        store = ActivationStore("dummy.h5")
        grid = store._generate_grid(
            resolution=5,
            domain_bounds=((0.0, 1.0), (-1.0, 1.0), (0.0, 2.0))
        )

        assert grid.shape == (125, 3)  # 5^3 points


class TestActivationExtraction:
    """Test activation extraction on grids."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple MLP model for testing."""
        model = MLP(
            input_dim=2,
            output_dim=1,
            hidden_dims=[16, 16],
            activation='tanh'
        )
        model.eval()
        return model

    def test_extract_on_grid_basic(self, simple_model, tmp_path):
        """Test basic activation extraction."""
        save_path = tmp_path / "test_activations.h5"
        store = ActivationStore(str(save_path))

        # Extract activations on small grid
        store.extract_on_grid(
            simple_model,
            grid_resolution=10,
            domain_bounds=((0.0, 1.0), (0.0, 1.0)),
            batch_size=50
        )

        # Check file was created
        assert save_path.exists()

        # Check metadata
        assert store.grid_resolution == 10
        assert len(store.layer_names) == 2  # 2 hidden layers

    def test_extract_on_grid_saves_correct_structure(self, simple_model, tmp_path):
        """Test that HDF5 has correct structure."""
        save_path = tmp_path / "test_activations.h5"
        store = ActivationStore(str(save_path))

        store.extract_on_grid(
            simple_model,
            grid_resolution=10,
            domain_bounds=((0.0, 1.0), (0.0, 1.0))
        )

        # Open HDF5 and check structure
        with h5py.File(save_path, 'r') as f:
            # Check coordinates
            assert 'coordinates' in f
            assert f['coordinates'].shape == (100, 2)

            # Check layers
            assert 'layer_0' in f
            assert 'layer_1' in f
            assert f['layer_0'].shape == (100, 16)
            assert f['layer_1'].shape == (100, 16)

            # Check metadata
            assert 'grid_resolution' in f.attrs
            assert f.attrs['grid_resolution'] == 10

    def test_extract_on_grid_different_resolutions(self, simple_model, tmp_path):
        """Test extraction with different grid resolutions."""
        for resolution in [5, 10, 20]:
            save_path = tmp_path / f"test_res_{resolution}.h5"
            store = ActivationStore(str(save_path))

            store.extract_on_grid(simple_model, grid_resolution=resolution)

            with h5py.File(save_path, 'r') as f:
                assert f['coordinates'].shape[0] == resolution * resolution

    def test_extract_on_grid_batching(self, simple_model, tmp_path):
        """Test that batching doesn't affect results."""
        save_path1 = tmp_path / "batch_10.h5"
        save_path2 = tmp_path / "batch_50.h5"

        store1 = ActivationStore(str(save_path1))
        store2 = ActivationStore(str(save_path2))

        # Same grid, different batch sizes
        store1.extract_on_grid(simple_model, grid_resolution=10, batch_size=10)
        store2.extract_on_grid(simple_model, grid_resolution=10, batch_size=50)

        # Results should be identical
        with h5py.File(save_path1, 'r') as f1, h5py.File(save_path2, 'r') as f2:
            coords1 = f1['coordinates'][:]
            coords2 = f2['coordinates'][:]
            acts1 = f1['layer_0'][:]
            acts2 = f2['layer_0'][:]

            np.testing.assert_array_almost_equal(coords1, coords2)
            np.testing.assert_array_almost_equal(acts1, acts2, decimal=5)

    def test_extract_on_grid_without_get_activations_raises(self, tmp_path):
        """Test that model without get_activations() raises error."""
        # Create a model without get_activations method
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(2, 1)

            def forward(self, x):
                return self.layer(x)

        model = DummyModel()
        store = ActivationStore(str(tmp_path / "test.h5"))

        with pytest.raises(ValueError, match="must have get_activations"):
            store.extract_on_grid(model, grid_resolution=10)


class TestLoadingData:
    """Test loading data from HDF5."""

    @pytest.fixture
    def populated_store(self, tmp_path):
        """Create a store with extracted activations."""
        model = MLP(
            input_dim=2,
            output_dim=1,
            hidden_dims=[8, 8],
            activation='tanh'
        )
        model.eval()

        save_path = tmp_path / "test_activations.h5"
        store = ActivationStore(str(save_path))
        store.extract_on_grid(model, grid_resolution=10)

        return store

    def test_load_layer(self, populated_store):
        """Test loading a specific layer."""
        activations = populated_store.load_layer('layer_0')

        assert isinstance(activations, np.ndarray)
        assert activations.shape == (100, 8)

    def test_load_layer_nonexistent_raises(self, populated_store):
        """Test loading non-existent layer raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            populated_store.load_layer('layer_999')

    def test_load_layer_from_nonexistent_file_raises(self, tmp_path):
        """Test loading from non-existent file raises FileNotFoundError."""
        store = ActivationStore(str(tmp_path / "nonexistent.h5"))

        with pytest.raises(FileNotFoundError):
            store.load_layer('layer_0')

    def test_load_coordinates(self, populated_store):
        """Test loading coordinates."""
        coords = populated_store.load_coordinates()

        assert isinstance(coords, np.ndarray)
        assert coords.shape == (100, 2)

        # Check bounds
        assert coords[:, 0].min() >= 0.0
        assert coords[:, 0].max() <= 1.0
        assert coords[:, 1].min() >= 0.0
        assert coords[:, 1].max() <= 1.0

    def test_get_metadata(self, populated_store):
        """Test getting metadata."""
        metadata = populated_store.get_metadata()

        assert metadata['grid_resolution'] == 10
        assert metadata['n_points'] == 100
        assert metadata['input_dim'] == 2
        assert 'layer_0' in metadata['layer_names']
        assert 'layer_1' in metadata['layer_names']
        assert metadata['file_size_kb'] > 0


class TestVisualization:
    """Test visualization methods."""

    @pytest.fixture
    def populated_store(self, tmp_path):
        """Create a store with extracted activations."""
        model = MLP(
            input_dim=2,
            output_dim=1,
            hidden_dims=[8],
            activation='tanh'
        )
        model.eval()

        save_path = tmp_path / "test_activations.h5"
        store = ActivationStore(str(save_path))
        store.extract_on_grid(model, grid_resolution=10)

        return store

    def test_visualize_neuron_basic(self, populated_store, tmp_path):
        """Test basic neuron visualization."""
        fig = populated_store.visualize_neuron('layer_0', neuron_idx=0)

        assert fig is not None
        assert len(fig.axes) > 0

    def test_visualize_neuron_save(self, populated_store, tmp_path):
        """Test saving neuron visualization."""
        save_path = tmp_path / "neuron_viz.png"
        fig = populated_store.visualize_neuron(
            'layer_0',
            neuron_idx=0,
            save_path=str(save_path)
        )

        assert save_path.exists()

    def test_visualize_neuron_invalid_index_raises(self, populated_store):
        """Test that invalid neuron index raises IndexError."""
        with pytest.raises(IndexError):
            populated_store.visualize_neuron('layer_0', neuron_idx=999)

    def test_visualize_neuron_nonexistent_layer_raises(self, populated_store):
        """Test that non-existent layer raises KeyError."""
        with pytest.raises(KeyError):
            populated_store.visualize_neuron('layer_999', neuron_idx=0)

    def test_visualize_layer_summary(self, populated_store, tmp_path):
        """Test layer summary visualization."""
        fig = populated_store.visualize_layer_summary('layer_0', n_neurons=4)

        assert fig is not None
        assert len(fig.axes) >= 4  # At least 4 subplots

    def test_visualize_layer_summary_save(self, populated_store, tmp_path):
        """Test saving layer summary."""
        save_path = tmp_path / "layer_summary.png"
        fig = populated_store.visualize_layer_summary(
            'layer_0',
            n_neurons=8,
            save_path=str(save_path)
        )

        assert save_path.exists()


class TestConvenienceFunction:
    """Test convenience function for extraction."""

    def test_extract_activations_from_model(self, tmp_path):
        """Test convenience function."""
        model = MLP(
            input_dim=2,
            output_dim=1,
            hidden_dims=[8],
            activation='tanh'
        )
        model.eval()

        save_path = tmp_path / "test.h5"
        store = extract_activations_from_model(
            model,
            str(save_path),
            grid_resolution=10
        )

        # Check store is returned
        assert isinstance(store, ActivationStore)
        assert store.save_path.exists()

        # Check can load data
        coords = store.load_coordinates()
        assert coords.shape == (100, 2)


class TestIntegration:
    """Integration tests with real trained models."""

    def test_extract_from_trained_model(self, tmp_path):
        """Test extraction from a simple trained model."""
        # Create and train a simple model
        model = MLP(
            input_dim=2,
            output_dim=1,
            hidden_dims=[16, 16],
            activation='tanh'
        )

        # Simple training (just a few steps to get non-random weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for _ in range(10):
            x = torch.randn(10, 2, requires_grad=True)
            y = model(x)
            loss = y.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()

        # Extract activations
        save_path = tmp_path / "trained.h5"
        store = ActivationStore(str(save_path))
        store.extract_on_grid(model, grid_resolution=20)

        # Verify
        metadata = store.get_metadata()
        assert metadata['n_points'] == 400
        assert len(metadata['layer_names']) == 2

        # Check activations are reasonable
        acts = store.load_layer('layer_0')
        assert not np.isnan(acts).any()
        assert not np.isinf(acts).any()

    def test_workflow_extract_load_visualize(self, tmp_path):
        """Test complete workflow: extract -> load -> visualize."""
        # 1. Create model
        model = MLP(
            input_dim=2,
            output_dim=1,
            hidden_dims=[8],
            activation='relu'
        )
        model.eval()

        # 2. Extract
        save_path = tmp_path / "workflow.h5"
        store = ActivationStore(str(save_path))
        store.extract_on_grid(model, grid_resolution=10)

        # 3. Load
        coords = store.load_coordinates()
        acts = store.load_layer('layer_0')

        assert coords.shape[0] == acts.shape[0]

        # 4. Visualize
        fig = store.visualize_neuron('layer_0', neuron_idx=0)
        assert fig is not None

        # 5. Summary
        fig_summary = store.visualize_layer_summary('layer_0', n_neurons=4)
        assert fig_summary is not None


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_small_grid(self, tmp_path):
        """Test with very small grid."""
        model = MLP(input_dim=2, output_dim=1, hidden_dims=[4], activation='tanh')
        model.eval()

        save_path = tmp_path / "small.h5"
        store = ActivationStore(str(save_path))
        store.extract_on_grid(model, grid_resolution=2)

        coords = store.load_coordinates()
        assert coords.shape == (4, 2)

    def test_large_batch_size(self, tmp_path):
        """Test with batch size larger than grid."""
        model = MLP(input_dim=2, output_dim=1, hidden_dims=[4], activation='tanh')
        model.eval()

        save_path = tmp_path / "large_batch.h5"
        store = ActivationStore(str(save_path))
        store.extract_on_grid(model, grid_resolution=5, batch_size=1000)

        coords = store.load_coordinates()
        assert coords.shape == (25, 2)

    def test_different_domain_bounds(self, tmp_path):
        """Test with non-standard domain bounds."""
        model = MLP(input_dim=2, output_dim=1, hidden_dims=[4], activation='tanh')
        model.eval()

        save_path = tmp_path / "custom_domain.h5"
        store = ActivationStore(str(save_path))
        store.extract_on_grid(
            model,
            grid_resolution=10,
            domain_bounds=((-1.0, 1.0), (0.0, 2.0))
        )

        coords = store.load_coordinates()
        assert coords[:, 0].min() >= -1.0
        assert coords[:, 0].max() <= 1.0
        assert coords[:, 1].min() >= 0.0
        assert coords[:, 1].max() <= 2.0
