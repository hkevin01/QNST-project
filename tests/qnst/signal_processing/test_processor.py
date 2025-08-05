"""Tests for the signal processing module."""

import os

# Add path to qnst package
import sys
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from qnst.electrode_array.electrode_array import ElectrodeArray
from qnst.signal_processing.processor import SignalProcessor

# Split long path into multiple lines for readability
qnst_path = os.path.dirname(__file__)
qnst_path = os.path.join(qnst_path, "../../../src")
qnst_path = os.path.abspath(qnst_path)
sys.path.insert(0, qnst_path)


@pytest.fixture(name="test_electrode_array")
def fixture_mock_electrode_array():
    """Create a mock electrode array."""
    mock = Mock(spec=ElectrodeArray)
    mock.num_channels = 4
    mock.sampling_rate = 1000
    return mock


@pytest.fixture(name="test_processor")
def fixture_signal_processor(test_electrode_array):
    """Create a signal processor instance."""
    return SignalProcessor(
        electrode_array=test_electrode_array, buffer_size=1000, max_workers=2
    )


@pytest.fixture(name="test_rng")
def fixture_rng():
    """Create a seeded random number generator."""
    return np.random.default_rng(42)  # Fixed seed for reproducibility


def test_signal_processor_init(test_processor, test_electrode_array):
    """Test signal processor initialization."""
    assert test_processor.electrode_array == test_electrode_array
    assert test_processor.buffer_size == 1000
    assert test_processor.max_workers == 2
    assert test_processor.raw_buffer.shape == (4, 1000)
    assert test_processor.filtered_buffer.shape == (4, 1000)
    assert abs(test_processor.notch_freq - 60.0) < 1e-10
    assert test_processor.bandpass == (0.1, 300.0)


def test_process_chunk(test_processor, test_rng):
    """Test processing a data chunk."""
    # Create test data with seeded Generator
    test_data = test_rng.standard_normal(size=(4, 100))

    # Process data
    processed = test_processor.process_chunk(test_data)

    # Check output shape
    assert processed.shape == test_data.shape

    # Check that filtering was applied (output should be different from input)
    assert not np.allclose(processed, test_data)


def test_update_buffer(test_processor, test_rng):
    """Test buffer update with new data."""
    # Create test data with seeded Generator
    test_data = test_rng.standard_normal(size=(4, 100))

    # Initial state
    initial_raw = test_processor.raw_buffer.copy()
    initial_filtered = test_processor.filtered_buffer.copy()

    # Update buffer
    test_processor.update_buffer(test_data)

    # Check that buffers were updated
    assert not np.allclose(test_processor.raw_buffer, initial_raw)
    assert not np.allclose(test_processor.filtered_buffer, initial_filtered)

    # Check that new data was added correctly
    np.testing.assert_array_equal(
        test_processor.raw_buffer[:, -test_data.shape[1] :], test_data
    )


def test_get_buffer_stats(test_processor, test_rng):
    """Test computing buffer statistics."""
    # Fill buffer with known data
    test_data = test_rng.standard_normal(size=(4, 1000))
    test_processor.filtered_buffer = test_data

    # Get stats
    stats = test_processor.get_buffer_stats()

    # Check that all expected stats are present
    assert "rms" in stats
    assert "peak_amp" in stats
    assert "std" in stats
    assert "snr" in stats

    # Verify stats are reasonable
    assert stats["rms"] > 0
    assert stats["peak_amp"] > 0
    assert stats["std"] > 0

    # Verify SNR calculation
    expected_snr = 20 * np.log10(stats["peak_amp"] / stats["std"])
    np.testing.assert_almost_equal(stats["snr"], expected_snr)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_processing(test_processor, test_rng):
    """Test GPU-accelerated processing if available."""
    # Create test data with seeded Generator
    test_data = test_rng.standard_normal(size=(4, 100))

    # Ensure GPU is being used
    assert test_processor.device.type == "cuda"

    # Process data
    processed = test_processor.process_chunk(test_data)

    # Check output
    assert isinstance(processed, np.ndarray)
    assert processed.shape == test_data.shape
    assert processed.shape == test_data.shape
