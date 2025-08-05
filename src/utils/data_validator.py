"""
Data validation utility for the QNST project.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
from dataclasses import dataclass
from enum import Enum
import logging
from .logger import get_logger

logger = get_logger(__name__)

class DataType(Enum):
    NEURAL = "neural"
    QUANTUM = "quantum"
    ROBOTIC = "robotic"
    CONSCIOUSNESS = "consciousness"

@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metrics: Dict[str, float]

class DataValidator:
    """Data validation for QNST project components."""
    
    def __init__(self):
        """Initialize data validator."""
        self.validation_rules: Dict[DataType, Dict] = {
            DataType.NEURAL: {
                'value_range': (-1000, 1000),  # μV
                'sampling_rate': (100, 50000),  # Hz
                'max_noise': 0.1,
                'min_snr': 10.0
            },
            DataType.QUANTUM: {
                'coherence_range': (0, 1),
                'max_entanglement': 1000000,
                'error_threshold': 0.001
            },
            DataType.ROBOTIC: {
                'sensor_range': (-1e6, 1e6),
                'max_latency': 0.1,  # seconds
                'min_accuracy': 0.9
            },
            DataType.CONSCIOUSNESS: {
                'awareness_range': (0, 1),
                'coherence_threshold': 0.7,
                'stability_threshold': 0.8
            }
        }
    
    def validate_neural_data(
        self,
        data: np.ndarray,
        sampling_rate: float,
        channel_info: Optional[Dict] = None
    ) -> ValidationResult:
        """
        Validate neural recording data.
        
        Args:
            data (np.ndarray): Neural data array
            sampling_rate (float): Sampling rate in Hz
            channel_info (Optional[Dict]): Additional channel information
            
        Returns:
            ValidationResult: Validation results
        """
        errors = []
        warnings = []
        metrics = {}
        
        rules = self.validation_rules[DataType.NEURAL]
        
        # Check value range
        value_min, value_max = np.min(data), np.max(data)
        if not (rules['value_range'][0] <= value_max <= rules['value_range'][1]):
            errors.append(f"Data values outside valid range: [{value_min}, {value_max}]")
        
        # Check sampling rate
        if not (rules['sampling_rate'][0] <= sampling_rate <= rules['sampling_rate'][1]):
            errors.append(f"Invalid sampling rate: {sampling_rate} Hz")
        
        # Calculate signal quality metrics
        noise_level = np.std(data)
        if noise_level > rules['max_noise']:
            warnings.append(f"High noise level: {noise_level}")
        
        signal_power = np.mean(np.square(data))
        snr = signal_power / (noise_level ** 2) if noise_level > 0 else float('inf')
        if snr < rules['min_snr']:
            warnings.append(f"Low signal-to-noise ratio: {snr}")
        
        metrics.update({
            'noise_level': noise_level,
            'snr': snr,
            'value_range': value_max - value_min,
            'mean': np.mean(data),
            'std': np.std(data)
        })
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metrics=metrics
        )
    
    def validate_quantum_state(
        self,
        state_vector: np.ndarray,
        entanglement_map: Optional[Dict] = None
    ) -> ValidationResult:
        """
        Validate quantum state data.
        
        Args:
            state_vector (np.ndarray): Quantum state vector
            entanglement_map (Optional[Dict]): Map of entangled qubits
            
        Returns:
            ValidationResult: Validation results
        """
        errors = []
        warnings = []
        metrics = {}
        
        rules = self.validation_rules[DataType.QUANTUM]
        
        # Check normalization
        norm = np.linalg.norm(state_vector)
        if not np.isclose(norm, 1.0, atol=1e-6):
            errors.append(f"State vector not normalized: norm = {norm}")
        
        # Check coherence
        coherence = np.abs(np.vdot(state_vector, state_vector))
        if not (rules['coherence_range'][0] <= coherence <= rules['coherence_range'][1]):
            errors.append(f"Invalid coherence value: {coherence}")
        
        # Check entanglement
        if entanglement_map:
            total_entanglements = sum(len(v) for v in entanglement_map.values())
            if total_entanglements > rules['max_entanglement']:
                warnings.append(
                    f"High entanglement count: {total_entanglements}"
                )
        
        # Calculate quantum metrics
        purity = np.abs(np.vdot(state_vector, state_vector)) ** 2
        metrics.update({
            'norm': norm,
            'coherence': coherence,
            'purity': purity
        })
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metrics=metrics
        )
    
    def validate_robotic_data(
        self,
        sensor_data: Dict[str, np.ndarray],
        timestamps: np.ndarray
    ) -> ValidationResult:
        """
        Validate robotic sensor data.
        
        Args:
            sensor_data (Dict[str, np.ndarray]): Sensor readings
            timestamps (np.ndarray): Measurement timestamps
            
        Returns:
            ValidationResult: Validation results
        """
        errors = []
        warnings = []
        metrics = {}
        
        rules = self.validation_rules[DataType.ROBOTIC]
        
        # Check sensor ranges
        for sensor_name, data in sensor_data.items():
            value_min, value_max = np.min(data), np.max(data)
            if not (rules['sensor_range'][0] <= value_max <= rules['sensor_range'][1]):
                errors.append(
                    f"Sensor {sensor_name} values outside range: [{value_min}, {value_max}]"
                )
        
        # Check timing
        if len(timestamps) > 1:
            latencies = np.diff(timestamps)
            max_latency = np.max(latencies)
            if max_latency > rules['max_latency']:
                warnings.append(f"High sensor latency: {max_latency*1000:.2f} ms")
            
            metrics['mean_latency'] = np.mean(latencies)
            metrics['max_latency'] = max_latency
        
        # Calculate accuracy metrics
        for sensor_name, data in sensor_data.items():
            accuracy = self._estimate_sensor_accuracy(data)
            if accuracy < rules['min_accuracy']:
                warnings.append(f"Low accuracy for sensor {sensor_name}: {accuracy:.2f}")
            metrics[f'{sensor_name}_accuracy'] = accuracy
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metrics=metrics
        )
    
    def validate_consciousness_state(
        self,
        state_data: Dict[str, float]
    ) -> ValidationResult:
        """
        Validate consciousness state data.
        
        Args:
            state_data (Dict[str, float]): Consciousness state metrics
            
        Returns:
            ValidationResult: Validation results
        """
        errors = []
        warnings = []
        metrics = {}
        
        rules = self.validation_rules[DataType.CONSCIOUSNESS]
        
        # Check awareness level
        if 'awareness' in state_data:
            awareness = state_data['awareness']
            if not (rules['awareness_range'][0] <= awareness <= rules['awareness_range'][1]):
                errors.append(f"Invalid awareness level: {awareness}")
            metrics['awareness'] = awareness
        
        # Check coherence
        if 'coherence' in state_data:
            coherence = state_data['coherence']
            if coherence < rules['coherence_threshold']:
                warnings.append(f"Low consciousness coherence: {coherence}")
            metrics['coherence'] = coherence
        
        # Check stability
        if 'stability' in state_data:
            stability = state_data['stability']
            if stability < rules['stability_threshold']:
                warnings.append(f"Low consciousness stability: {stability}")
            metrics['stability'] = stability
        
        # Overall consciousness metrics
        metrics['consciousness_quality'] = np.mean([
            v for k, v in state_data.items()
            if k in ['awareness', 'coherence', 'stability']
        ])
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metrics=metrics
        )
    
    def _estimate_sensor_accuracy(self, data: np.ndarray) -> float:
        """
        Estimate sensor accuracy from data.
        
        Args:
            data (np.ndarray): Sensor data
            
        Returns:
            float: Estimated accuracy (0-1)
        """
        # Simple estimation based on noise and stability
        noise_level = np.std(data)
        stability = 1.0 - min(np.abs(np.diff(data)).mean(), 1.0)
        
        return (1.0 - noise_level) * stability
