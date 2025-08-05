"""
Test suite for QNST utility modules.
"""

import os
import pytest
import numpy as np
from pathlib import Path
from src.utils import (
    ConfigLoader,
    PerformanceMonitor,
    DataValidator,
    DataType,
    ValidationResult
)

# Test Configuration
@pytest.fixture
def config_loader():
    """Create a config loader instance."""
    # Create temporary config directory
    config_dir = Path("test_config")
    config_dir.mkdir(exist_ok=True)
    
    # Create test config file
    test_config = config_dir / "test.yaml"
    test_config.write_text("""
test_values:
  value1: 10
  value2: "test"
  value3:
    - item1
    - item2
""")
    
    loader = ConfigLoader(str(config_dir))
    yield loader
    
    # Cleanup
    test_config.unlink()
    config_dir.rmdir()

def test_config_loader(config_loader):
    """Test configuration loading functionality."""
    config = config_loader.load_config("test")
    
    assert "test_values" in config
    assert config["test_values"]["value1"] == 10
    assert config["test_values"]["value2"] == "test"
    assert len(config["test_values"]["value3"]) == 2

def test_config_value_retrieval(config_loader):
    """Test configuration value retrieval."""
    value = config_loader.get_value("test", "test_values", "value1")
    assert value == 10
    
    # Test default value
    value = config_loader.get_value("test", "nonexistent", default="default")
    assert value == "default"

# Test Performance Monitoring
@pytest.fixture
def performance_monitor():
    """Create a performance monitor instance."""
    return PerformanceMonitor(history_size=10)

def test_performance_metrics(performance_monitor):
    """Test performance metrics collection."""
    metrics = performance_monitor.get_current_metrics()
    
    assert metrics.cpu_usage >= 0
    assert metrics.memory_usage >= 0
    assert metrics.network_io[0] >= 0
    assert metrics.network_io[1] >= 0
    assert metrics.disk_io[0] >= 0
    assert metrics.disk_io[1] >= 0

def test_system_health(performance_monitor):
    """Test system health calculation."""
    health = performance_monitor.get_system_health()
    
    assert 0 <= health["cpu_health"] <= 1
    assert 0 <= health["memory_health"] <= 1
    assert 0 <= health["overall_health"] <= 1

# Test Data Validation
@pytest.fixture
def data_validator():
    """Create a data validator instance."""
    return DataValidator()

def test_neural_data_validation(data_validator):
    """Test neural data validation."""
    # Generate test data
    data = np.random.normal(0, 100, 1000)  # μV range
    sampling_rate = 1000  # Hz
    
    result = data_validator.validate_neural_data(data, sampling_rate)
    
    assert isinstance(result, ValidationResult)
    assert "noise_level" in result.metrics
    assert "snr" in result.metrics

def test_quantum_state_validation(data_validator):
    """Test quantum state validation."""
    # Generate normalized quantum state
    state = np.array([0.7071067811865476, 0.7071067811865476])  # |+⟩ state
    
    result = data_validator.validate_quantum_state(state)
    
    assert isinstance(result, ValidationResult)
    assert result.is_valid
    assert "coherence" in result.metrics
    assert "purity" in result.metrics

def test_robotic_data_validation(data_validator):
    """Test robotic sensor data validation."""
    # Generate test sensor data
    sensor_data = {
        "pressure": np.random.uniform(0, 100, 1000),
        "temperature": np.random.uniform(20, 30, 1000)
    }
    timestamps = np.linspace(0, 1, 1000)  # 1 second of data
    
    result = data_validator.validate_robotic_data(sensor_data, timestamps)
    
    assert isinstance(result, ValidationResult)
    assert "mean_latency" in result.metrics
    assert "pressure_accuracy" in result.metrics
    assert "temperature_accuracy" in result.metrics

def test_consciousness_validation(data_validator):
    """Test consciousness state validation."""
    state_data = {
        "awareness": 0.8,
        "coherence": 0.9,
        "stability": 0.85
    }
    
    result = data_validator.validate_consciousness_state(state_data)
    
    assert isinstance(result, ValidationResult)
    assert result.is_valid
    assert "consciousness_quality" in result.metrics
