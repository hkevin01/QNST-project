"""Tests for neural pattern recognition system."""

import os

# Add path to qnst package
import sys
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from qnst.pattern_recognition.recognizer import (
    NeuralPatternClassifier,
    PatternRecognitionSystem,
    SpikingNeuralNetwork,
)
from qnst.signal_processing.processor import SignalProcessor

# Split long path into multiple lines for readability
qnst_path = os.path.dirname(__file__)
qnst_path = os.path.join(qnst_path, "../../../src")
qnst_path = os.path.abspath(qnst_path)
sys.path.insert(0, qnst_path)


@pytest.fixture(name="mock_signal_processor")
def fixture_mock_signal_processor():
    """Create a mock signal processor."""
    mock = Mock(spec=SignalProcessor)
    rng = np.random.default_rng(42)
    mock.process_chunk.return_value = rng.standard_normal(size=(4, 1000))
    return mock


@pytest.fixture(name="test_rng")
def fixture_rng():
    """Create a seeded random number generator."""
    return np.random.default_rng(42)


@pytest.fixture(name="sample_neural_data")
def fixture_sample_neural_data(test_rng):
    """Create sample neural data for testing."""
    return test_rng.standard_normal(size=(4, 1000))


class TestNeuralPatternClassifier:
    """Test suite for NeuralPatternClassifier."""

    def test_init(self):
        """Test model initialization."""
        model = NeuralPatternClassifier(
            input_size=128, hidden_sizes=[256, 128, 64], num_classes=5, dropout_rate=0.2
        )

        assert model.num_classes == 5
        assert len(model.network) > 0

    def test_forward_pass(self, test_rng):
        """Test forward pass through the network."""
        model = NeuralPatternClassifier(input_size=128, num_classes=10)

        # Create test input
        test_input = torch.FloatTensor(test_rng.standard_normal(size=(32, 128)))

        # Forward pass
        output = model(test_input)

        assert output.shape == (32, 10)
        assert not torch.isnan(output).any()

    def test_parameter_count(self):
        """Test that model has reasonable number of parameters."""
        model = NeuralPatternClassifier(input_size=128, num_classes=10)
        param_count = sum(p.numel() for p in model.parameters())

        assert param_count > 0
        assert param_count < 10000000  # Reasonable upper bound


class TestSpikingNeuralNetwork:
    """Test suite for SpikingNeuralNetwork."""

    def test_init(self):
        """Test SNN initialization."""
        snn = SpikingNeuralNetwork(input_size=64, hidden_size=128, output_size=10)

        assert snn.input_size == 64
        assert snn.hidden_size == 128
        assert snn.output_size == 10
        assert snn.w_input.shape == (64, 128)
        assert snn.w_hidden.shape == (128, 10)

    def test_reset_state(self):
        """Test state reset functionality."""
        snn = SpikingNeuralNetwork(input_size=64, output_size=10)

        # Modify state
        snn.v_hidden[0] = 1.0
        snn.v_output[0] = 1.0

        # Reset
        snn.reset_state()

        assert np.all(snn.v_hidden == 0.0)
        assert np.all(snn.v_output == 0.0)

    def test_step_simulation(self, test_rng):
        """Test single time step simulation."""
        snn = SpikingNeuralNetwork(input_size=64, output_size=10)

        # Create spike input
        input_spikes = test_rng.integers(0, 2, size=64)

        # Simulate step
        output_spikes = snn.step(input_spikes)

        assert output_spikes.shape == (10,)
        assert np.all((output_spikes == 0) | (output_spikes == 1))

    def test_membrane_dynamics(self):
        """Test membrane potential dynamics."""
        snn = SpikingNeuralNetwork(input_size=10, hidden_size=5, output_size=3)

        # Strong input should cause membrane potential to rise
        strong_input = np.ones(10)

        # Store initial state
        initial_v = snn.v_hidden.copy()

        # Simulate step
        snn.step(strong_input)

        # Membrane potential should have changed
        assert not np.allclose(snn.v_hidden, initial_v)


class TestPatternRecognitionSystem:
    """Test suite for PatternRecognitionSystem."""

    def test_init(self, mock_signal_processor):
        """Test system initialization."""
        system = PatternRecognitionSystem(
            signal_processor=mock_signal_processor, feature_dim=64, num_classes=5
        )

        assert system.feature_dim == 64
        assert system.num_classes == 5
        assert system.dl_model.num_classes == 5
        assert system.snn.output_size == 5

    def test_feature_extraction(self, mock_signal_processor, sample_neural_data):
        """Test feature extraction from neural data."""
        system = PatternRecognitionSystem(
            signal_processor=mock_signal_processor, feature_dim=64
        )

        features = system.extract_features(sample_neural_data)

        # Should extract features for each channel
        expected_features_per_channel = 10  # 5 time + 5 frequency features
        expected_total = sample_neural_data.shape[0] * expected_features_per_channel

        assert len(features) == expected_total
        assert not np.isnan(features).any()

    def test_feature_preprocessing(self, mock_signal_processor, test_rng):
        """Test feature preprocessing pipeline."""
        system = PatternRecognitionSystem(
            signal_processor=mock_signal_processor, feature_dim=64
        )

        # Create test features
        raw_features = test_rng.standard_normal(size=(100, 200))

        # Fit and transform
        processed = system.preprocess_features(raw_features, fit_transforms=True)

        assert processed.shape[0] == 100
        assert processed.shape[1] == 64  # Reduced to feature_dim

        # Transform without fitting
        new_features = test_rng.standard_normal(size=(50, 200))
        processed_new = system.preprocess_features(new_features)

        assert processed_new.shape[0] == 50
        assert processed_new.shape[1] == 64

    def test_train_deep_model(self, mock_signal_processor, test_rng):
        """Test deep learning model training."""
        system = PatternRecognitionSystem(
            signal_processor=mock_signal_processor, feature_dim=32, num_classes=3
        )

        # Create training data
        train_data = test_rng.standard_normal(size=(100, 128))
        train_labels = test_rng.integers(0, 3, size=100)

        # Train model
        history = system.train_deep_model(
            train_data, train_labels, epochs=5, batch_size=16
        )

        assert "loss" in history
        assert "accuracy" in history
        assert len(history["loss"]) == 5
        assert len(history["accuracy"]) == 5

    def test_prediction(self, mock_signal_processor, sample_neural_data):
        """Test prediction functionality."""
        system = PatternRecognitionSystem(
            signal_processor=mock_signal_processor, feature_dim=32, num_classes=5
        )

        # Need to fit transforms first
        dummy_features = np.random.default_rng(42).standard_normal(size=(10, 40))
        system.preprocess_features(dummy_features, fit_transforms=True)

        predictions = system.predict(sample_neural_data)

        assert "deep_learning" in predictions
        assert "spiking_network" in predictions
        assert "ensemble" in predictions

        # Check probability distributions
        for pred_type in predictions:
            probs = predictions[pred_type]
            assert len(probs) == 5  # num_classes
            assert np.all(probs >= 0)
            assert np.isclose(np.sum(probs), 1.0, rtol=1e-3)

    def test_model_stats(self, mock_signal_processor):
        """Test model statistics retrieval."""
        system = PatternRecognitionSystem(
            signal_processor=mock_signal_processor, feature_dim=64, num_classes=10
        )

        stats = system.get_model_stats()

        assert "deep_learning_parameters" in stats
        assert "snn_parameters" in stats
        assert "feature_dimension" in stats
        assert "num_classes" in stats
        assert "device" in stats

        assert stats["feature_dimension"] == 64
        assert stats["num_classes"] == 10
        assert stats["deep_learning_parameters"] > 0
        assert stats["snn_parameters"] > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_support(self, mock_signal_processor):
        """Test GPU acceleration if available."""
        system = PatternRecognitionSystem(signal_processor=mock_signal_processor)

        assert system.device.type == "cuda"
        assert next(system.dl_model.parameters()).is_cuda


class TestIntegration:
    """Integration tests for the pattern recognition system."""

    def test_end_to_end_workflow(self, mock_signal_processor, test_rng):
        """Test complete workflow from training to prediction."""
        system = PatternRecognitionSystem(
            signal_processor=mock_signal_processor, feature_dim=32, num_classes=3
        )

        # Generate training data
        train_data = test_rng.standard_normal(size=(50, 128))
        train_labels = test_rng.integers(0, 3, size=50)

        # Train the model
        history = system.train_deep_model(
            train_data, train_labels, epochs=3, batch_size=8
        )

        # Make predictions on new data
        test_data = test_rng.standard_normal(size=(4, 1000))
        predictions = system.predict(test_data)

        # Verify training occurred
        assert len(history["loss"]) == 3

        # Verify predictions are reasonable
        assert "ensemble" in predictions
        ensemble_pred = predictions["ensemble"]
        assert len(ensemble_pred) == 3
        assert np.all(ensemble_pred >= 0)

    def test_feature_consistency(self, mock_signal_processor, sample_neural_data):
        """Test that feature extraction is consistent."""
        system = PatternRecognitionSystem(signal_processor=mock_signal_processor)

        # Extract features multiple times
        features1 = system.extract_features(sample_neural_data)
        features2 = system.extract_features(sample_neural_data)

        # Should be identical (deterministic processing)
        np.testing.assert_array_equal(features1, features2)
        np.testing.assert_array_equal(features1, features2)
