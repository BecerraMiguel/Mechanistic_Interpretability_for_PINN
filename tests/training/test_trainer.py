"""
Tests for PINN training loop.

Tests verify:
- PINNTrainer initialization
- Collocation point sampling
- Loss computation (PDE, BC, IC)
- Training step execution
- Validation/evaluation
- Full training loop
- Checkpoint saving/loading
- Convenience function train_pinn
"""

import os
import tempfile

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from src.models import MLP
from src.problems import PoissonProblem
from src.training import PINNTrainer, train_pinn


class TestPINNTrainerInitialization:
    """Test PINNTrainer initialization."""

    def test_basic_initialization(self):
        """Test basic trainer initialization."""
        model = MLP(input_dim=2, hidden_dims=[32, 32], output_dim=1)
        problem = PoissonProblem()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        trainer = PINNTrainer(
            model=model,
            problem=problem,
            optimizer=optimizer,
            n_interior=100,
            n_boundary=25,
        )

        assert trainer.model is model
        assert trainer.problem is problem
        assert trainer.optimizer is optimizer
        assert trainer.n_interior == 100
        assert trainer.n_boundary == 25
        assert trainer.n_initial == 0
        assert trainer.device == "cpu"
        assert trainer.epoch == 0

    def test_custom_loss_weights(self):
        """Test initialization with custom loss weights."""
        model = MLP(input_dim=2, hidden_dims=[32], output_dim=1)
        problem = PoissonProblem()
        optimizer = optim.Adam(model.parameters())

        loss_weights = {"pde": 2.0, "bc": 0.5, "ic": 0.0}
        trainer = PINNTrainer(
            model=model,
            problem=problem,
            optimizer=optimizer,
            n_interior=100,
            n_boundary=25,
            loss_weights=loss_weights,
        )

        assert trainer.loss_weights == loss_weights

    def test_default_loss_weights(self):
        """Test default loss weights are all 1.0."""
        model = MLP(input_dim=2, hidden_dims=[32], output_dim=1)
        problem = PoissonProblem()
        optimizer = optim.Adam(model.parameters())

        trainer = PINNTrainer(
            model=model,
            problem=problem,
            optimizer=optimizer,
            n_interior=100,
            n_boundary=25,
        )

        assert trainer.loss_weights == {"pde": 1.0, "bc": 1.0, "ic": 1.0}

    def test_history_initialization(self):
        """Test that history dict is initialized correctly."""
        model = MLP(input_dim=2, hidden_dims=[32], output_dim=1)
        problem = PoissonProblem()
        optimizer = optim.Adam(model.parameters())

        trainer = PINNTrainer(
            model=model,
            problem=problem,
            optimizer=optimizer,
            n_interior=100,
            n_boundary=25,
        )

        assert "loss_total" in trainer.history
        assert "loss_pde" in trainer.history
        assert "loss_bc" in trainer.history
        assert "loss_ic" in trainer.history
        assert "relative_l2_error" in trainer.history

        # All should be empty lists initially
        for key in trainer.history:
            assert trainer.history[key] == []


class TestCollocationPointSampling:
    """Test collocation point sampling."""

    def test_sample_collocation_points_shapes(self):
        """Test that sampled points have correct shapes."""
        model = MLP(input_dim=2, hidden_dims=[32], output_dim=1)
        problem = PoissonProblem()
        optimizer = optim.Adam(model.parameters())

        trainer = PINNTrainer(
            model=model,
            problem=problem,
            optimizer=optimizer,
            n_interior=100,
            n_boundary=25,
        )

        x_interior, x_boundary, x_initial = trainer.sample_collocation_points()

        assert x_interior.shape == (100, 2)
        assert x_boundary.shape == (100, 2)  # 4 edges * 25 points
        assert x_initial is None  # No initial conditions for Poisson

    def test_sample_collocation_points_device(self):
        """Test that sampled points are on correct device."""
        model = MLP(input_dim=2, hidden_dims=[32], output_dim=1)
        problem = PoissonProblem()
        optimizer = optim.Adam(model.parameters())

        trainer = PINNTrainer(
            model=model,
            problem=problem,
            optimizer=optimizer,
            n_interior=100,
            n_boundary=25,
            device="cpu",
        )

        x_interior, x_boundary, _ = trainer.sample_collocation_points()

        assert x_interior.device.type == "cpu"
        assert x_boundary.device.type == "cpu"

    def test_sample_collocation_points_reproducibility(self):
        """Test that sampling with same seed gives same points."""
        model = MLP(input_dim=2, hidden_dims=[32], output_dim=1)
        problem = PoissonProblem()
        optimizer = optim.Adam(model.parameters())

        trainer = PINNTrainer(
            model=model,
            problem=problem,
            optimizer=optimizer,
            n_interior=100,
            n_boundary=25,
        )

        x_int1, x_bound1, _ = trainer.sample_collocation_points(random_seed=42)
        x_int2, x_bound2, _ = trainer.sample_collocation_points(random_seed=42)

        assert torch.allclose(x_int1, x_int2)
        assert torch.allclose(x_bound1, x_bound2)


class TestLossComputation:
    """Test loss computation."""

    def test_compute_loss_returns_correct_keys(self):
        """Test that compute_loss returns all required keys."""
        model = MLP(input_dim=2, hidden_dims=[32, 32], output_dim=1)
        problem = PoissonProblem()
        optimizer = optim.Adam(model.parameters())

        trainer = PINNTrainer(
            model=model,
            problem=problem,
            optimizer=optimizer,
            n_interior=100,
            n_boundary=25,
        )

        x_interior, x_boundary, _ = trainer.sample_collocation_points()
        losses = trainer.compute_loss(x_interior, x_boundary)

        assert "total" in losses
        assert "pde" in losses
        assert "bc" in losses
        assert "ic" in losses

    def test_compute_loss_values_are_tensors(self):
        """Test that loss values are torch tensors."""
        model = MLP(input_dim=2, hidden_dims=[32], output_dim=1)
        problem = PoissonProblem()
        optimizer = optim.Adam(model.parameters())

        trainer = PINNTrainer(
            model=model,
            problem=problem,
            optimizer=optimizer,
            n_interior=50,
            n_boundary=10,
        )

        x_interior, x_boundary, _ = trainer.sample_collocation_points()
        losses = trainer.compute_loss(x_interior, x_boundary)

        for key, value in losses.items():
            assert isinstance(value, torch.Tensor)
            assert value.ndim == 0  # Scalar

    def test_compute_loss_all_positive(self):
        """Test that all loss components are non-negative."""
        model = MLP(input_dim=2, hidden_dims=[32], output_dim=1)
        problem = PoissonProblem()
        optimizer = optim.Adam(model.parameters())

        trainer = PINNTrainer(
            model=model,
            problem=problem,
            optimizer=optimizer,
            n_interior=50,
            n_boundary=10,
        )

        x_interior, x_boundary, _ = trainer.sample_collocation_points()
        losses = trainer.compute_loss(x_interior, x_boundary)

        for key, value in losses.items():
            assert value.item() >= 0

    def test_compute_loss_decomposition(self):
        """Test that total loss equals weighted sum of components."""
        model = MLP(input_dim=2, hidden_dims=[32], output_dim=1)
        problem = PoissonProblem()
        optimizer = optim.Adam(model.parameters())

        loss_weights = {"pde": 2.0, "bc": 0.5, "ic": 0.0}
        trainer = PINNTrainer(
            model=model,
            problem=problem,
            optimizer=optimizer,
            n_interior=50,
            n_boundary=10,
            loss_weights=loss_weights,
        )

        x_interior, x_boundary, _ = trainer.sample_collocation_points()
        losses = trainer.compute_loss(x_interior, x_boundary)

        expected_total = (
            loss_weights["pde"] * losses["pde"]
            + loss_weights["bc"] * losses["bc"]
            + loss_weights["ic"] * losses["ic"]
        )

        assert torch.isclose(losses["total"], expected_total, rtol=1e-5)

    def test_compute_loss_gradients_flow(self):
        """Test that gradients flow through loss computation."""
        model = MLP(input_dim=2, hidden_dims=[32], output_dim=1)
        problem = PoissonProblem()
        optimizer = optim.Adam(model.parameters())

        trainer = PINNTrainer(
            model=model,
            problem=problem,
            optimizer=optimizer,
            n_interior=50,
            n_boundary=10,
        )

        x_interior, x_boundary, _ = trainer.sample_collocation_points()
        losses = trainer.compute_loss(x_interior, x_boundary)

        # Backward should not raise error
        losses["total"].backward()

        # Check that gradients exist
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()


class TestTrainingStep:
    """Test single training step."""

    def test_train_step_returns_dict(self):
        """Test that train_step returns dictionary of losses."""
        model = MLP(input_dim=2, hidden_dims=[32], output_dim=1)
        problem = PoissonProblem()
        optimizer = optim.Adam(model.parameters())

        trainer = PINNTrainer(
            model=model,
            problem=problem,
            optimizer=optimizer,
            n_interior=50,
            n_boundary=10,
        )

        x_interior, x_boundary, _ = trainer.sample_collocation_points()
        losses = trainer.train_step(x_interior, x_boundary)

        assert isinstance(losses, dict)
        assert "total" in losses
        assert "pde" in losses
        assert "bc" in losses

    def test_train_step_updates_parameters(self):
        """Test that train_step updates model parameters."""
        model = MLP(input_dim=2, hidden_dims=[32], output_dim=1)
        problem = PoissonProblem()
        optimizer = optim.Adam(model.parameters(), lr=1e-2)

        trainer = PINNTrainer(
            model=model,
            problem=problem,
            optimizer=optimizer,
            n_interior=50,
            n_boundary=10,
        )

        # Get initial parameters
        initial_params = [p.clone() for p in model.parameters()]

        # Training step
        x_interior, x_boundary, _ = trainer.sample_collocation_points()
        trainer.train_step(x_interior, x_boundary)

        # Check that parameters changed
        for initial, current in zip(initial_params, model.parameters()):
            assert not torch.allclose(initial, current)

    def test_train_step_reduces_loss(self):
        """Test that multiple train steps reduce loss."""
        model = MLP(input_dim=2, hidden_dims=[64, 64], output_dim=1)
        problem = PoissonProblem()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        trainer = PINNTrainer(
            model=model,
            problem=problem,
            optimizer=optimizer,
            n_interior=100,
            n_boundary=25,
        )

        x_interior, x_boundary, _ = trainer.sample_collocation_points(random_seed=42)

        # Initial loss
        initial_loss = trainer.train_step(x_interior, x_boundary)["total"]

        # Train for several steps
        for _ in range(50):
            current_loss = trainer.train_step(x_interior, x_boundary)["total"]

        # Loss should decrease
        assert current_loss < initial_loss


class TestValidation:
    """Test validation/evaluation."""

    def test_validate_returns_float(self):
        """Test that validate returns a float."""
        model = MLP(input_dim=2, hidden_dims=[32], output_dim=1)
        problem = PoissonProblem()
        optimizer = optim.Adam(model.parameters())

        trainer = PINNTrainer(
            model=model,
            problem=problem,
            optimizer=optimizer,
            n_interior=50,
            n_boundary=10,
        )

        error = trainer.validate(n_test_points=100)

        assert isinstance(error, float)
        assert error >= 0

    def test_validate_no_gradients(self):
        """Test that validation doesn't compute gradients."""
        model = MLP(input_dim=2, hidden_dims=[32], output_dim=1)
        problem = PoissonProblem()
        optimizer = optim.Adam(model.parameters())

        trainer = PINNTrainer(
            model=model,
            problem=problem,
            optimizer=optimizer,
            n_interior=50,
            n_boundary=10,
        )

        # Validation should not require gradients
        with torch.no_grad():
            error = trainer.validate(n_test_points=100)

        assert isinstance(error, float)


class TestTrainingLoop:
    """Test full training loop."""

    def test_train_basic(self):
        """Test basic training loop execution."""
        model = MLP(input_dim=2, hidden_dims=[32, 32], output_dim=1)
        problem = PoissonProblem()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        trainer = PINNTrainer(
            model=model,
            problem=problem,
            optimizer=optimizer,
            n_interior=100,
            n_boundary=25,
        )

        history = trainer.train(
            n_epochs=10,
            validate_every=5,
            print_every=5,
        )

        assert len(history["loss_total"]) == 10
        assert len(history["loss_pde"]) == 10
        assert len(history["loss_bc"]) == 10

    def test_train_loss_decreases(self):
        """Test that training decreases loss."""
        model = MLP(input_dim=2, hidden_dims=[64, 64], output_dim=1)
        problem = PoissonProblem()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        trainer = PINNTrainer(
            model=model,
            problem=problem,
            optimizer=optimizer,
            n_interior=200,
            n_boundary=50,
        )

        history = trainer.train(
            n_epochs=100,
            validate_every=50,
            print_every=50,
        )

        # Loss should decrease
        initial_loss = history["loss_total"][0]
        final_loss = history["loss_total"][-1]

        assert final_loss < initial_loss

    def test_train_with_resampling(self):
        """Test training with periodic resampling."""
        model = MLP(input_dim=2, hidden_dims=[32], output_dim=1)
        problem = PoissonProblem()
        optimizer = optim.Adam(model.parameters())

        trainer = PINNTrainer(
            model=model,
            problem=problem,
            optimizer=optimizer,
            n_interior=50,
            n_boundary=10,
        )

        history = trainer.train(
            n_epochs=20,
            resample_every=5,  # Resample every 5 epochs
            validate_every=10,
            print_every=10,
        )

        assert len(history["loss_total"]) == 20

    def test_train_validation_frequency(self):
        """Test that validation happens at correct frequency."""
        model = MLP(input_dim=2, hidden_dims=[32], output_dim=1)
        problem = PoissonProblem()
        optimizer = optim.Adam(model.parameters())

        trainer = PINNTrainer(
            model=model,
            problem=problem,
            optimizer=optimizer,
            n_interior=50,
            n_boundary=10,
        )

        history = trainer.train(
            n_epochs=10,
            validate_every=3,
            print_every=10,
        )

        # Should validate at epochs 3, 6, 9, and final (10)
        # That's 4 validations (including final)
        assert len(history["relative_l2_error"]) == 4


class TestCheckpointing:
    """Test checkpoint saving and loading."""

    def test_save_checkpoint(self):
        """Test that checkpoint saving works."""
        model = MLP(input_dim=2, hidden_dims=[32], output_dim=1)
        problem = PoissonProblem()
        optimizer = optim.Adam(model.parameters())

        trainer = PINNTrainer(
            model=model,
            problem=problem,
            optimizer=optimizer,
            n_interior=50,
            n_boundary=10,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.save_checkpoint(tmpdir, epoch=10)

            checkpoint_path = os.path.join(tmpdir, "checkpoint_epoch_10.pt")
            assert os.path.exists(checkpoint_path)

    def test_load_checkpoint(self):
        """Test that checkpoint loading works."""
        model = MLP(input_dim=2, hidden_dims=[32], output_dim=1)
        problem = PoissonProblem()
        optimizer = optim.Adam(model.parameters())

        trainer = PINNTrainer(
            model=model,
            problem=problem,
            optimizer=optimizer,
            n_interior=50,
            n_boundary=10,
        )

        # Train a bit
        trainer.train(n_epochs=5, print_every=10)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save checkpoint
            checkpoint_path = os.path.join(tmpdir, "checkpoint.pt")
            torch.save(
                {
                    "epoch": trainer.epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss_weights": trainer.loss_weights,
                    "history": trainer.history,
                },
                checkpoint_path,
            )

            # Create new trainer and load
            new_model = MLP(input_dim=2, hidden_dims=[32], output_dim=1)
            new_optimizer = optim.Adam(new_model.parameters())
            new_trainer = PINNTrainer(
                model=new_model,
                problem=problem,
                optimizer=new_optimizer,
                n_interior=50,
                n_boundary=10,
            )

            new_trainer.load_checkpoint(checkpoint_path)

            assert new_trainer.epoch == trainer.epoch


class TestConvenienceFunction:
    """Test train_pinn convenience function."""

    def test_train_pinn_basic(self):
        """Test basic usage of train_pinn function."""
        model = MLP(input_dim=2, hidden_dims=[32, 32], output_dim=1)
        problem = PoissonProblem()

        config = {
            "optimizer": "adam",
            "lr": 1e-3,
            "n_epochs": 10,
            "n_interior": 100,
            "n_boundary": 25,
            "loss_weights": {"pde": 1.0, "bc": 1.0, "ic": 1.0},
            "device": "cpu",
            "print_every": 5,
            "validate_every": 5,
        }

        trained_model, history = train_pinn(model, problem, config)

        assert len(history["loss_total"]) == 10
        assert trained_model is model

    def test_train_pinn_different_optimizers(self):
        """Test train_pinn with different optimizers."""
        problem = PoissonProblem()

        for optimizer_type in ["adam", "sgd"]:
            model = MLP(input_dim=2, hidden_dims=[32], output_dim=1)
            config = {
                "optimizer": optimizer_type,
                "lr": 1e-3,
                "n_epochs": 5,
                "n_interior": 50,
                "n_boundary": 10,
                "print_every": 10,
            }

            trained_model, history = train_pinn(model, problem, config)
            assert len(history["loss_total"]) == 5

    def test_train_pinn_invalid_optimizer_raises_error(self):
        """Test that invalid optimizer raises error."""
        model = MLP(input_dim=2, hidden_dims=[32], output_dim=1)
        problem = PoissonProblem()

        config = {
            "optimizer": "invalid_optimizer",
            "n_epochs": 5,
            "n_interior": 50,
            "n_boundary": 10,
        }

        with pytest.raises(ValueError, match="Unknown optimizer"):
            train_pinn(model, problem, config)


class TestIntegration:
    """Integration tests for full training pipeline."""

    def test_full_training_pipeline(self):
        """Test complete training pipeline from initialization to convergence."""
        # Create model and problem
        model = MLP(
            input_dim=2, hidden_dims=[64, 64, 64], output_dim=1, activation="tanh"
        )
        problem = PoissonProblem()

        # Training config
        config = {
            "optimizer": "adam",
            "lr": 1e-3,
            "n_epochs": 500,
            "n_interior": 1000,
            "n_boundary": 100,
            "loss_weights": {"pde": 1.0, "bc": 1.0, "ic": 1.0},
            "device": "cpu",
            "resample_every": 1,
            "validate_every": 100,
            "print_every": 100,
        }

        # Train
        trained_model, history = train_pinn(model, problem, config)

        # Verify training happened
        assert len(history["loss_total"]) == 500

        # Verify loss decreased
        assert history["loss_total"][-1] < history["loss_total"][0]

        # Verify validation errors were computed
        assert len(history["relative_l2_error"]) > 0

        # Final error should be reasonable (model should learn something)
        final_error = history["relative_l2_error"][-1]
        assert final_error < 100  # At least better than 100% error


class TestEarlyStopping:
    """Test early stopping functionality."""

    def test_early_stopping_triggers(self):
        """Test that early stopping triggers when validation doesn't improve."""
        torch.manual_seed(999)
        model = MLP(input_dim=2, hidden_dims=[8], output_dim=1)
        problem = PoissonProblem()
        optimizer = optim.Adam(
            model.parameters(), lr=1e-6
        )  # Very small lr ensures negligible improvement

        trainer = PINNTrainer(
            model=model,
            problem=problem,
            optimizer=optimizer,
            n_interior=50,
            n_boundary=10,
        )

        history = trainer.train(
            n_epochs=1000,
            validate_every=10,
            print_every=1000,
            early_stopping=True,
            patience=3,
            min_delta=1.0,  # Require 100% improvement — impossible
        )

        # Should stop before 1000 epochs
        assert len(history["loss_total"]) < 1000
        assert trainer.patience_counter >= 3

    def test_early_stopping_restores_best_model(self):
        """Test that early stopping restores the best model weights."""
        model = MLP(input_dim=2, hidden_dims=[32], output_dim=1)
        problem = PoissonProblem()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        trainer = PINNTrainer(
            model=model,
            problem=problem,
            optimizer=optimizer,
            n_interior=100,
            n_boundary=20,
        )

        # Train with early stopping
        trainer.train(
            n_epochs=200,
            validate_every=10,
            print_every=200,
            early_stopping=True,
            patience=3,
            min_delta=0.01,
        )

        # Best model state should be saved
        assert trainer.best_model_state is not None
        assert trainer.best_val_error < float("inf")

    def test_no_early_stopping_by_default(self):
        """Test that early stopping is disabled by default."""
        model = MLP(input_dim=2, hidden_dims=[32], output_dim=1)
        problem = PoissonProblem()
        optimizer = optim.Adam(model.parameters())

        trainer = PINNTrainer(
            model=model,
            problem=problem,
            optimizer=optimizer,
            n_interior=50,
            n_boundary=10,
        )

        history = trainer.train(
            n_epochs=20,
            validate_every=5,
            print_every=20,
        )

        # Should run all epochs
        assert len(history["loss_total"]) == 20

    def test_train_pinn_with_early_stopping(self):
        """Test train_pinn convenience function with early stopping."""
        model = MLP(input_dim=2, hidden_dims=[32, 32], output_dim=1)
        problem = PoissonProblem()

        config = {
            "optimizer": "adam",
            "lr": 1e-3,
            "n_epochs": 500,
            "n_interior": 100,
            "n_boundary": 20,
            "validate_every": 20,
            "print_every": 500,
            "early_stopping": True,
            "patience": 5,
            "min_delta": 0.01,
        }

        trained_model, history = train_pinn(model, problem, config)

        # Training should have run (possibly stopped early)
        assert len(history["loss_total"]) > 0
        assert len(history["loss_total"]) <= 500


class TestVisualization:
    """Test solution visualization functionality."""

    def test_generate_solution_heatmap(self):
        """Test that solution heatmap generation works."""
        model = MLP(input_dim=2, hidden_dims=[32, 32], output_dim=1)
        problem = PoissonProblem()
        optimizer = optim.Adam(model.parameters())

        trainer = PINNTrainer(
            model=model,
            problem=problem,
            optimizer=optimizer,
            n_interior=100,
            n_boundary=20,
        )

        # Train briefly
        trainer.train(n_epochs=50, print_every=50)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_heatmap.png")
            trainer.generate_solution_heatmap(save_path, n_points=50)

            # Check that file was created
            assert os.path.exists(save_path)

            # Check that file is not empty
            assert os.path.getsize(save_path) > 0

    def test_generate_solution_heatmap_custom_params(self):
        """Test solution heatmap with custom parameters."""
        model = MLP(input_dim=2, hidden_dims=[32], output_dim=1)
        problem = PoissonProblem()
        optimizer = optim.Adam(model.parameters())

        trainer = PINNTrainer(
            model=model,
            problem=problem,
            optimizer=optimizer,
            n_interior=100,
            n_boundary=20,
        )

        # Train briefly
        trainer.train(n_epochs=20, print_every=20)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "custom_heatmap.png")
            trainer.generate_solution_heatmap(
                save_path,
                n_points=30,
                figsize=(12, 4),
                dpi=100,
            )

            assert os.path.exists(save_path)
            assert os.path.getsize(save_path) > 0
