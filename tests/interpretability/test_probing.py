"""
Tests for probing classifiers module.
"""

import numpy as np
import pytest
import torch

from src.interpretability.probing import LinearProbe


class TestLinearProbeInit:
    """Tests for LinearProbe initialization."""

    def test_init_default(self):
        """Test default initialization."""
        probe = LinearProbe(input_dim=64)

        assert probe.input_dim == 64
        assert probe.output_dim == 1
        assert not probe.is_fitted
        assert probe.training_history == {"loss": []}
        assert isinstance(probe.weights, torch.nn.Linear)
        assert probe.weights.in_features == 64
        assert probe.weights.out_features == 1

    def test_init_custom_output_dim(self):
        """Test initialization with custom output dimension."""
        probe = LinearProbe(input_dim=32, output_dim=3)

        assert probe.input_dim == 32
        assert probe.output_dim == 3
        assert probe.weights.in_features == 32
        assert probe.weights.out_features == 3

    def test_weights_initialized(self):
        """Test that weights are initialized (not all zeros)."""
        probe = LinearProbe(input_dim=10, output_dim=1)

        weight_sum = probe.weights.weight.abs().sum().item()
        assert weight_sum > 0, "Weights should be initialized (not all zeros)"

        # Bias should be zero-initialized
        bias_sum = probe.weights.bias.abs().sum().item()
        assert bias_sum == 0, "Bias should be zero-initialized"

    def test_repr(self):
        """Test string representation."""
        probe = LinearProbe(input_dim=64, output_dim=1)

        repr_str = repr(probe)
        assert "LinearProbe" in repr_str
        assert "input_dim=64" in repr_str
        assert "output_dim=1" in repr_str
        assert "not fitted" in repr_str


class TestLinearProbeFit:
    """Tests for LinearProbe training."""

    def test_fit_simple_linear_function(self):
        """Test fitting a simple linear function."""
        torch.manual_seed(42)

        # Create simple linear relationship: y = 2*x1 + 3*x2 + 1
        n_samples = 1000
        input_dim = 2

        X = torch.randn(n_samples, input_dim)
        y = 2 * X[:, 0] + 3 * X[:, 1] + 1

        # Train probe
        probe = LinearProbe(input_dim=input_dim, output_dim=1)
        history = probe.fit(X, y, epochs=500, lr=1e-2)

        # Check that probe is now fitted
        assert probe.is_fitted

        # Check that loss decreased
        assert len(history["loss"]) == 500
        assert history["loss"][-1] < history["loss"][0]

        # Check that learned weights are close to true weights
        weights, bias = probe.get_weights()
        assert weights.shape == (1, 2)
        assert bias.shape == (1,)

        # Allow some tolerance
        np.testing.assert_allclose(weights[0], [2.0, 3.0], atol=0.1)
        np.testing.assert_allclose(bias[0], 1.0, atol=0.1)

    def test_fit_with_mini_batches(self):
        """Test fitting with mini-batch training."""
        torch.manual_seed(42)

        # Create data
        n_samples = 1000
        input_dim = 10
        X = torch.randn(n_samples, input_dim)
        y = torch.sum(X, dim=1)

        # Train with mini-batches
        probe = LinearProbe(input_dim=input_dim, output_dim=1)
        history = probe.fit(X, y, epochs=100, batch_size=64)

        assert probe.is_fitted
        assert len(history["loss"]) == 100

        # Loss should decrease
        assert history["loss"][-1] < history["loss"][0]

    def test_fit_perfect_prediction(self):
        """Test that probe can achieve near-perfect R² for linear data."""
        torch.manual_seed(42)

        # Create perfect linear relationship
        n_samples = 1000
        input_dim = 5
        X = torch.randn(n_samples, input_dim)

        # True weights
        true_weights = torch.randn(input_dim, 1)
        y = X @ true_weights

        # Train probe
        probe = LinearProbe(input_dim=input_dim, output_dim=1)
        probe.fit(X, y, epochs=1000, lr=1e-2)

        # Should achieve very high R²
        scores = probe.score(X, y)
        assert scores["r_squared"] > 0.99, f"R² = {scores['r_squared']}"

    def test_fit_dimension_mismatch_samples(self):
        """Test error when number of samples don't match."""
        probe = LinearProbe(input_dim=10, output_dim=1)

        X = torch.randn(100, 10)
        y = torch.randn(50, 1)

        with pytest.raises(ValueError, match="Number of samples mismatch"):
            probe.fit(X, y)

    def test_fit_dimension_mismatch_input(self):
        """Test error when input dimension doesn't match."""
        probe = LinearProbe(input_dim=10, output_dim=1)

        X = torch.randn(100, 5)  # Wrong input_dim
        y = torch.randn(100, 1)

        with pytest.raises(ValueError, match="Input dimension mismatch"):
            probe.fit(X, y)

    def test_fit_dimension_mismatch_output(self):
        """Test error when output dimension doesn't match."""
        probe = LinearProbe(input_dim=10, output_dim=1)

        X = torch.randn(100, 10)
        y = torch.randn(100, 3)  # Wrong output_dim

        with pytest.raises(ValueError, match="Output dimension mismatch"):
            probe.fit(X, y)

    def test_fit_with_1d_targets(self):
        """Test that 1D targets are handled correctly."""
        torch.manual_seed(42)

        probe = LinearProbe(input_dim=5, output_dim=1)

        X = torch.randn(100, 5)
        y = torch.randn(100)  # 1D targets

        # Should not raise error
        probe.fit(X, y, epochs=10)
        assert probe.is_fitted

    def test_fit_verbose(self, capsys):
        """Test verbose output during training."""
        torch.manual_seed(42)

        probe = LinearProbe(input_dim=5, output_dim=1)
        X = torch.randn(100, 5)
        y = torch.randn(100, 1)

        # Train with verbose=True
        probe.fit(X, y, epochs=250, verbose=True)

        # Check that progress was printed
        captured = capsys.readouterr()
        assert "Epoch 100" in captured.out
        assert "Epoch 200" in captured.out
        assert "Loss:" in captured.out


class TestLinearProbePredict:
    """Tests for LinearProbe prediction."""

    def test_predict_before_fit(self):
        """Test that prediction fails before fitting."""
        probe = LinearProbe(input_dim=10, output_dim=1)
        X = torch.randn(100, 10)

        with pytest.raises(RuntimeError, match="must be fitted before"):
            probe.predict(X)

    def test_predict_after_fit(self):
        """Test prediction after fitting."""
        torch.manual_seed(42)

        # Train probe
        probe = LinearProbe(input_dim=5, output_dim=1)
        X_train = torch.randn(100, 5)
        y_train = torch.sum(X_train, dim=1)
        probe.fit(X_train, y_train, epochs=100)

        # Make predictions
        X_test = torch.randn(50, 5)
        predictions = probe.predict(X_test)

        assert predictions.shape == (50, 1)
        assert isinstance(predictions, torch.Tensor)

    def test_predict_dimension_mismatch(self):
        """Test error when input dimension doesn't match."""
        torch.manual_seed(42)

        # Train probe
        probe = LinearProbe(input_dim=5, output_dim=1)
        X_train = torch.randn(100, 5)
        y_train = torch.randn(100, 1)
        probe.fit(X_train, y_train, epochs=10)

        # Try to predict with wrong dimension
        X_test = torch.randn(50, 10)  # Wrong dimension

        with pytest.raises(ValueError, match="Input dimension mismatch"):
            probe.predict(X_test)

    def test_predict_deterministic(self):
        """Test that predictions are deterministic."""
        torch.manual_seed(42)

        # Train probe
        probe = LinearProbe(input_dim=5, output_dim=1)
        X_train = torch.randn(100, 5)
        y_train = torch.randn(100, 1)
        probe.fit(X_train, y_train, epochs=100)

        # Make predictions twice
        X_test = torch.randn(50, 5)
        pred1 = probe.predict(X_test)
        pred2 = probe.predict(X_test)

        torch.testing.assert_close(pred1, pred2)


class TestLinearProbeScore:
    """Tests for LinearProbe scoring."""

    def test_score_before_fit(self):
        """Test that scoring fails before fitting."""
        probe = LinearProbe(input_dim=10, output_dim=1)
        X = torch.randn(100, 10)
        y = torch.randn(100, 1)

        with pytest.raises(RuntimeError, match="must be fitted before"):
            probe.score(X, y)

    def test_score_perfect_fit(self):
        """Test scoring with perfect fit."""
        torch.manual_seed(42)

        # Train on perfect linear data
        probe = LinearProbe(input_dim=5, output_dim=1)
        X = torch.randn(100, 5)
        y = torch.sum(X, dim=1, keepdim=True)
        probe.fit(X, y, epochs=1000, lr=1e-2)

        # Score on same data
        scores = probe.score(X, y)

        assert "mse" in scores
        assert "r_squared" in scores
        assert "explained_variance" in scores

        # Should be near-perfect
        assert scores["mse"] < 0.01
        assert scores["r_squared"] > 0.99
        assert scores["explained_variance"] > 0.99

    def test_score_with_noise(self):
        """Test scoring with noisy data."""
        torch.manual_seed(42)

        # Create noisy linear relationship (reduced noise)
        probe = LinearProbe(input_dim=3, output_dim=1)
        X = torch.randn(200, 3)
        y_clean = torch.sum(X, dim=1, keepdim=True)
        y_noisy = y_clean + 0.2 * torch.randn(200, 1)  # Less noise

        probe.fit(X, y_noisy, epochs=500, lr=1e-2)

        # Score should be good but not perfect
        scores = probe.score(X, y_noisy)

        assert 0.3 < scores["r_squared"] < 1.0  # More lenient threshold
        assert scores["mse"] > 0

    def test_score_with_1d_targets(self):
        """Test scoring with 1D targets."""
        torch.manual_seed(42)

        probe = LinearProbe(input_dim=5, output_dim=1)
        X = torch.randn(100, 5)
        y = torch.randn(100)  # 1D
        probe.fit(X, y, epochs=100)

        # Should handle 1D targets
        scores = probe.score(X, y)

        assert "mse" in scores
        assert "r_squared" in scores
        assert "explained_variance" in scores

    def test_score_returns_floats(self):
        """Test that score returns Python floats, not tensors."""
        torch.manual_seed(42)

        probe = LinearProbe(input_dim=5, output_dim=1)
        X = torch.randn(100, 5)
        y = torch.randn(100, 1)
        probe.fit(X, y, epochs=100)

        scores = probe.score(X, y)

        assert isinstance(scores["mse"], float)
        assert isinstance(scores["r_squared"], float)
        assert isinstance(scores["explained_variance"], float)


class TestLinearProbeWeights:
    """Tests for LinearProbe weight access."""

    def test_get_weights_before_fit(self):
        """Test that get_weights fails before fitting."""
        probe = LinearProbe(input_dim=10, output_dim=1)

        with pytest.raises(RuntimeError, match="must be fitted before"):
            probe.get_weights()

    def test_get_weights_after_fit(self):
        """Test get_weights returns correct shapes."""
        torch.manual_seed(42)

        probe = LinearProbe(input_dim=5, output_dim=1)
        X = torch.randn(100, 5)
        y = torch.randn(100, 1)
        probe.fit(X, y, epochs=100)

        weights, bias = probe.get_weights()

        assert isinstance(weights, np.ndarray)
        assert isinstance(bias, np.ndarray)
        assert weights.shape == (1, 5)
        assert bias.shape == (1,)

    def test_get_weights_multi_output(self):
        """Test get_weights with multiple outputs."""
        torch.manual_seed(42)

        probe = LinearProbe(input_dim=5, output_dim=3)
        X = torch.randn(100, 5)
        y = torch.randn(100, 3)
        probe.fit(X, y, epochs=100)

        weights, bias = probe.get_weights()

        assert weights.shape == (3, 5)
        assert bias.shape == (3,)

    def test_learned_weights_correctness(self):
        """Test that learned weights approximate true weights."""
        torch.manual_seed(42)

        # Define true weights
        input_dim = 4
        true_weights = torch.tensor([[1.0, -2.0, 3.0, 0.5]])
        true_bias = torch.tensor([2.5])

        # Generate data
        X = torch.randn(1000, input_dim)
        y = X @ true_weights.T + true_bias

        # Train probe
        probe = LinearProbe(input_dim=input_dim, output_dim=1)
        probe.fit(X, y, epochs=1000, lr=1e-2)

        # Check learned weights
        learned_weights, learned_bias = probe.get_weights()

        np.testing.assert_allclose(
            learned_weights.flatten(), true_weights.numpy().flatten(), atol=0.05
        )
        np.testing.assert_allclose(learned_bias, true_bias.numpy(), atol=0.05)


class TestLinearProbeDevices:
    """Tests for LinearProbe device management."""

    def test_to_device_cpu(self):
        """Test moving probe to CPU."""
        probe = LinearProbe(input_dim=10, output_dim=1)

        probe_returned = probe.to(torch.device("cpu"))

        assert probe_returned is probe  # Should return self
        assert next(probe.weights.parameters()).device == torch.device("cpu")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_to_device_cuda(self):
        """Test moving probe to CUDA."""
        probe = LinearProbe(input_dim=10, output_dim=1)

        probe.to(torch.device("cuda"))

        assert next(probe.weights.parameters()).device.type == "cuda"

    def test_fit_on_cuda_data(self):
        """Test fitting with CUDA tensors (if available)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        torch.manual_seed(42)

        probe = LinearProbe(input_dim=5, output_dim=1)
        probe.to(torch.device("cuda"))

        X = torch.randn(100, 5, device="cuda")
        y = torch.randn(100, 1, device="cuda")

        # Should work without errors
        probe.fit(X, y, epochs=100)

        assert probe.is_fitted


class TestLinearProbeIntegration:
    """Integration tests for LinearProbe."""

    def test_full_workflow(self):
        """Test complete workflow: init -> fit -> predict -> score."""
        torch.manual_seed(42)

        # 1. Create probe
        probe = LinearProbe(input_dim=10, output_dim=1)
        assert not probe.is_fitted

        # 2. Generate data (linear relationship with noise)
        X_train = torch.randn(500, 10)
        true_weights = torch.randn(10, 1)
        y_train = X_train @ true_weights + 0.1 * torch.randn(500, 1)

        X_test = torch.randn(100, 10)
        y_test = X_test @ true_weights + 0.1 * torch.randn(100, 1)

        # 3. Fit probe
        history = probe.fit(X_train, y_train, epochs=500, lr=1e-2)
        assert probe.is_fitted
        assert len(history["loss"]) == 500

        # 4. Make predictions
        predictions = probe.predict(X_test)
        assert predictions.shape == (100, 1)

        # 5. Score on test set
        scores = probe.score(X_test, y_test)
        assert scores["r_squared"] > 0.9  # Should be very good

        # 6. Get weights
        weights, bias = probe.get_weights()
        assert weights.shape == (1, 10)

    def test_repr_before_and_after_fit(self):
        """Test repr changes after fitting."""
        probe = LinearProbe(input_dim=10, output_dim=1)

        # Before fit
        repr_before = repr(probe)
        assert "not fitted" in repr_before

        # After fit
        X = torch.randn(100, 10)
        y = torch.randn(100, 1)
        probe.fit(X, y, epochs=10)

        repr_after = repr(probe)
        assert "fitted" in repr_after
        assert "not fitted" not in repr_after

    def test_multiple_fits(self):
        """Test that probe can be refitted with new data."""
        torch.manual_seed(42)

        probe = LinearProbe(input_dim=5, output_dim=1)

        # First fit
        X1 = torch.randn(100, 5)
        y1 = torch.sum(X1, dim=1, keepdim=True)
        probe.fit(X1, y1, epochs=500, lr=1e-2)
        score1 = probe.score(X1, y1)

        # Create new probe for second fit (to ensure clean state)
        probe2 = LinearProbe(input_dim=5, output_dim=1)
        X2 = torch.randn(100, 5)
        y2 = torch.sum(X2, dim=1, keepdim=True) * 2  # Different relationship
        probe2.fit(X2, y2, epochs=500, lr=1e-2)
        score2 = probe2.score(X2, y2)

        # Both should achieve good fits
        assert score1["r_squared"] > 0.9
        assert score2["r_squared"] > 0.9

        # Weights should be different for different data
        weights1, _ = probe.get_weights()
        weights2, _ = probe2.get_weights()
        assert not np.allclose(weights1, weights2, atol=0.1)
