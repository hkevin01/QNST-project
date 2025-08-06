"""Neurochemical monitoring system for QNST project."""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, savgol_filter
from sklearn.preprocessing import MinMaxScaler

from ..signal_processing.processor import SignalProcessor


class SensorType(Enum):
    """Types of neurochemical sensors."""
    ELECTROCHEMICAL = "electrochemical"
    OPTICAL = "optical"
    MICRODIALYSIS = "microdialysis"


class NeurotransmitterType(Enum):
    """Types of neurotransmitters to monitor."""
    DOPAMINE = "dopamine"
    SEROTONIN = "serotonin"
    NOREPINEPHRINE = "norepinephrine"
    ACETYLCHOLINE = "acetylcholine"
    GABA = "gaba"
    GLUTAMATE = "glutamate"


@dataclass
class ChemicalMeasurement:
    """Data structure for chemical measurements."""
    timestamp: float
    sensor_id: str
    sensor_type: SensorType
    neurotransmitter: NeurotransmitterType
    concentration: float
    confidence: float
    location: Tuple[float, float, float]


class ElectrochemicalSensor:
    """Electrochemical sensor for neurotransmitter detection."""

    def __init__(
        self,
        sensor_id: str,
        target_neurotransmitter: NeurotransmitterType,
        position: Tuple[float, float, float],
        sensitivity: float = 1.0
    ) -> None:
        """
        Initialize electrochemical sensor.

        Args:
            sensor_id: Unique sensor identifier
            target_neurotransmitter: Target neurotransmitter
            position: 3D position of sensor
            sensitivity: Sensor sensitivity factor
        """
        self.sensor_id = sensor_id
        self.target_neurotransmitter = target_neurotransmitter
        self.position = position
        self.sensitivity = sensitivity

        # Calibration parameters
        self.baseline_current = 0.0
        self.calibration_curve = None
        self.noise_level = 0.01

        # State variables
        self.is_calibrated = False
        self.measurement_history = []

    def calibrate(self, known_concentrations: List[float],
                  measured_currents: List[float]) -> None:
        """
        Calibrate sensor with known standards.

        Args:
            known_concentrations: Known concentration values
            measured_currents: Corresponding measured currents
        """
        if len(known_concentrations) != len(measured_currents):
            msg = "Concentration and current arrays must be same length"
            raise ValueError(msg)

        # Create interpolation function for calibration curve
        self.calibration_curve = interp1d(
            measured_currents,
            known_concentrations,
            kind='linear',
            fill_value='extrapolate'
        )

        self.is_calibrated = True

    def measure_concentration(
        self,
        raw_current: float,
        timestamp: float
    ) -> ChemicalMeasurement:
        """
        Convert raw current to concentration measurement.

        Args:
            raw_current: Raw sensor current
            timestamp: Measurement timestamp

        Returns:
            Chemical measurement data
        """
        if not self.is_calibrated:
            raise RuntimeError("Sensor must be calibrated before use")

        # Apply baseline correction
        corrected_current = raw_current - self.baseline_current

        # Convert to concentration using calibration curve
        concentration = float(self.calibration_curve(corrected_current))

        # Calculate confidence based on signal-to-noise ratio
        snr = abs(corrected_current) / self.noise_level
        confidence = min(1.0, snr / 10.0)  # Normalized confidence

        measurement = ChemicalMeasurement(
            timestamp=timestamp,
            sensor_id=self.sensor_id,
            sensor_type=SensorType.ELECTROCHEMICAL,
            neurotransmitter=self.target_neurotransmitter,
            concentration=max(0.0, concentration),  # No negatives
            confidence=confidence,
            location=self.position
        )

        self.measurement_history.append(measurement)
        return measurement


class OpticalBiosensor:
    """Optical biosensor for fluorescent neurotransmitter detection."""

    def __init__(
        self,
        sensor_id: str,
        target_neurotransmitter: NeurotransmitterType,
        position: Tuple[float, float, float],
        excitation_wavelength: float = 488.0,
        emission_wavelength: float = 520.0
    ) -> None:
        """
        Initialize optical biosensor.

        Args:
            sensor_id: Unique sensor identifier
            target_neurotransmitter: Target neurotransmitter
            position: 3D position of sensor
            excitation_wavelength: Excitation wavelength in nm
            emission_wavelength: Emission wavelength in nm
        """
        self.sensor_id = sensor_id
        self.target_neurotransmitter = target_neurotransmitter
        self.position = position
        self.excitation_wavelength = excitation_wavelength
        self.emission_wavelength = emission_wavelength

        # Optical parameters
        self.quantum_yield = 0.8
        self.extinction_coefficient = 50000  # M^-1 cm^-1
        self.background_fluorescence = 100.0

        # State variables
        self.is_calibrated = False
        self.measurement_history = []

    def measure_fluorescence(
        self,
        fluorescence_intensity: float,
        timestamp: float
    ) -> ChemicalMeasurement:
        """
        Convert fluorescence intensity to concentration.

        Args:
            fluorescence_intensity: Measured fluorescence intensity
            timestamp: Measurement timestamp

        Returns:
            Chemical measurement data
        """
        # Subtract background fluorescence
        bg_fluor = self.background_fluorescence
        corrected_intensity = fluorescence_intensity - bg_fluor
        corrected_intensity = max(0.0, corrected_intensity)

        # Convert to concentration (simplified Beer-Lambert law)
        # F = quantum_yield * extinction_coeff * concentration * path_length
        # Assuming path_length = 1 cm for simplicity
        concentration = corrected_intensity / (
            self.quantum_yield * self.extinction_coefficient
        )

        # Calculate confidence based on signal strength
        signal_ratio = corrected_intensity / self.background_fluorescence
        confidence = min(1.0, signal_ratio / 2.0)

        measurement = ChemicalMeasurement(
            timestamp=timestamp,
            sensor_id=self.sensor_id,
            sensor_type=SensorType.OPTICAL,
            neurotransmitter=self.target_neurotransmitter,
            concentration=concentration,
            confidence=confidence,
            location=self.position
        )

        self.measurement_history.append(measurement)
        return measurement


class MicrodialysisProbe:
    """Microdialysis probe for continuous sampling."""

    def __init__(
        self,
        probe_id: str,
        position: Tuple[float, float, float],
        membrane_cutoff: float = 20.0,  # kDa
        flow_rate: float = 1.0  # μL/min
    ) -> None:
        """
        Initialize microdialysis probe.

        Args:
            probe_id: Unique probe identifier
            position: 3D position of probe
            membrane_cutoff: Molecular weight cutoff in kDa
            flow_rate: Perfusion flow rate in μL/min
        """
        self.probe_id = probe_id
        self.position = position
        self.membrane_cutoff = membrane_cutoff
        self.flow_rate = flow_rate

        # Recovery efficiency for different molecules
        self.recovery_rates = {
            NeurotransmitterType.DOPAMINE: 0.15,
            NeurotransmitterType.SEROTONIN: 0.12,
            NeurotransmitterType.NOREPINEPHRINE: 0.18,
            NeurotransmitterType.ACETYLCHOLINE: 0.10,
            NeurotransmitterType.GABA: 0.20,
            NeurotransmitterType.GLUTAMATE: 0.25
        }

        self.measurement_history = []

    def collect_sample(
        self,
        dialysate_concentrations: Dict[NeurotransmitterType, float],
        timestamp: float
    ) -> List[ChemicalMeasurement]:
        """
        Process collected dialysate sample.

        Args:
            dialysate_concentrations: Measured concentrations in dialysate
            timestamp: Collection timestamp

        Returns:
            List of corrected concentration measurements
        """
        measurements = []

        items = dialysate_concentrations.items()
        for neurotransmitter, dialysate_conc in items:
            # Correct for recovery efficiency
            recovery_rate = self.recovery_rates.get(neurotransmitter, 0.15)
            actual_concentration = dialysate_conc / recovery_rate

            # Calculate confidence based on recovery rate
            confidence = min(1.0, recovery_rate * 5.0)

            measurement = ChemicalMeasurement(
                timestamp=timestamp,
                sensor_id=self.probe_id,
                sensor_type=SensorType.MICRODIALYSIS,
                neurotransmitter=neurotransmitter,
                concentration=actual_concentration,
                confidence=confidence,
                location=self.position
            )

            measurements.append(measurement)

        self.measurement_history.extend(measurements)
        return measurements


class NeurochemicalAnalyzer:
    """Neural network for neurochemical pattern analysis."""

    def __init__(
        self,
        input_features: int = 64,
        hidden_size: int = 128,
        num_neurotransmitters: int = 6
    ) -> None:
        """
        Initialize neurochemical analyzer.

        Args:
            input_features: Number of input features
            hidden_size: Hidden layer size
            num_neurotransmitters: Number of neurotransmitter types
        """
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Build neural network
        self.network = nn.Sequential(
            nn.Linear(input_features, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Linear(hidden_size // 2, num_neurotransmitters),
            nn.Softmax(dim=1)
        ).to(self.device)

        self.scaler = MinMaxScaler()
        self.is_trained = False

    def extract_features(
        self,
        measurements: List[ChemicalMeasurement],
        time_window: float = 1.0
    ) -> np.ndarray:
        """
        Extract features from chemical measurements.

        Args:
            measurements: List of chemical measurements
            time_window: Time window for feature extraction

        Returns:
            Feature vector
        """
        if not measurements:
            return np.zeros(64)  # Default feature size

        # Group measurements by neurotransmitter
        neurotransmitter_data = {}
        for measurement in measurements:
            nt = measurement.neurotransmitter
            if nt not in neurotransmitter_data:
                neurotransmitter_data[nt] = []
            neurotransmitter_data[nt].append(measurement)

        features = []

        # Extract features for each neurotransmitter
        for nt_type in NeurotransmitterType:
            if nt_type in neurotransmitter_data:
                concentrations = [
                    m.concentration for m in neurotransmitter_data[nt_type]
                ]

                # Statistical features
                features.extend([
                    np.mean(concentrations),
                    np.std(concentrations),
                    np.max(concentrations),
                    np.min(concentrations),
                    len(concentrations)  # Count of measurements
                ])

                # Temporal features
                if len(concentrations) > 1:
                    concentrations_array = np.array(concentrations)
                    # Smooth and find peaks
                    if len(concentrations_array) > 5:
                        smoothed = savgol_filter(concentrations_array, 5, 2)
                        peaks, _ = find_peaks(smoothed)
                        features.extend([
                            len(peaks),  # Number of peaks
                            # Rate of change
                            np.mean(np.diff(concentrations_array))
                        ])
                    else:
                        features.extend([0, 0])
                else:
                    features.extend([0, 0])
            else:
                # No data for this neurotransmitter
                features.extend([0, 0, 0, 0, 0, 0, 0])

        # Spatial correlation features
        if len(measurements) > 1:
            positions = np.array([m.location for m in measurements])
            concentrations = np.array([m.concentration for m in measurements])

            # Spatial variance
            spatial_var = np.var(positions, axis=0)
            features.extend(spatial_var.tolist())

            # Concentration gradient
            if len(set(m.location for m in measurements)) > 1:
                grad = np.gradient(concentrations)
                features.append(np.mean(np.abs(grad)))
            else:
                features.append(0)
        else:
            features.extend([0, 0, 0, 0])

        # Pad or trim to fixed size
        feature_array = np.array(features)
        if len(feature_array) < 64:
            feature_array = np.pad(
                feature_array,
                (0, 64 - len(feature_array)),
                'constant'
            )
        elif len(feature_array) > 64:
            feature_array = feature_array[:64]

        return feature_array

    def predict_pattern(
        self,
        measurements: List[ChemicalMeasurement]
    ) -> Dict[str, float]:
        """
        Predict neurochemical pattern from measurements.

        Args:
            measurements: List of chemical measurements

        Returns:
            Pattern prediction probabilities
        """
        features = self.extract_features(measurements)

        if not self.is_trained:
            # Return uniform distribution if not trained
            num_patterns = 6  # Number of basic patterns
            patterns = {}
            for i in range(num_patterns):
                patterns[f"pattern_{i}"] = 1.0/num_patterns
            return patterns

        # Normalize features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        features_tensor = torch.FloatTensor(features_scaled).to(self.device)

        with torch.no_grad():
            predictions = self.network(features_tensor)

        # Convert to dictionary
        pattern_names = [
            "baseline", "excitatory", "inhibitory",
            "reward", "stress", "attention"
        ]

        result = {}
        for i, pattern in enumerate(pattern_names):
            result[pattern] = float(predictions[0, i].cpu())

        return result


class NeurochemicalMonitor:
    """Main neurochemical monitoring system."""

    def __init__(self, signal_processor: SignalProcessor) -> None:
        """
        Initialize neurochemical monitoring system.

        Args:
            signal_processor: Neural signal processor
        """
        self.signal_processor = signal_processor

        # Sensor arrays
        self.electrochemical_sensors = []
        self.optical_sensors = []
        self.microdialysis_probes = []

        # Analysis components
        self.analyzer = NeurochemicalAnalyzer()

        # Data storage
        self.measurement_buffer = []
        self.pattern_history = []

        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def add_electrochemical_sensor(
        self,
        sensor_id: str,
        target_nt: NeurotransmitterType,
        position: Tuple[float, float, float]
    ) -> None:
        """Add electrochemical sensor to system."""
        sensor = ElectrochemicalSensor(sensor_id, target_nt, position)
        self.electrochemical_sensors.append(sensor)

    def add_optical_sensor(
        self,
        sensor_id: str,
        target_nt: NeurotransmitterType,
        position: Tuple[float, float, float]
    ) -> None:
        """Add optical sensor to system."""
        sensor = OpticalBiosensor(sensor_id, target_nt, position)
        self.optical_sensors.append(sensor)

    def add_microdialysis_probe(
        self,
        probe_id: str,
        position: Tuple[float, float, float]
    ) -> None:
        """Add microdialysis probe to system."""
        probe = MicrodialysisProbe(probe_id, position)
        self.microdialysis_probes.append(probe)

    def collect_measurements(
        self,
        sensor_data: Dict[str, Any],
        timestamp: float
    ) -> List[ChemicalMeasurement]:
        """
        Collect measurements from all sensors.

        Args:
            sensor_data: Raw sensor data dictionary
            timestamp: Current timestamp

        Returns:
            List of chemical measurements
        """
        measurements = []

        # Process electrochemical sensors
        for sensor in self.electrochemical_sensors:
            if sensor.sensor_id in sensor_data:
                raw_current = sensor_data[sensor.sensor_id]
                if sensor.is_calibrated:
                    current = raw_current
                    measurement = sensor.measure_concentration(
                        current, timestamp
                    )
                    measurements.append(measurement)

        # Process optical sensors
        for sensor in self.optical_sensors:
            if sensor.sensor_id in sensor_data:
                fluorescence = sensor_data[sensor.sensor_id]
                fluoro = fluorescence
                measurement = sensor.measure_fluorescence(
                    fluoro, timestamp
                )
                measurements.append(measurement)

        # Process microdialysis probes
        for probe in self.microdialysis_probes:
            if probe.probe_id in sensor_data:
                dialysate_data = sensor_data[probe.probe_id]
                data = dialysate_data
                probe_measurements = probe.collect_sample(
                    data, timestamp
                )
                measurements.extend(probe_measurements)

        self.measurement_buffer.extend(measurements)
        return measurements

    def analyze_neurochemical_state(
        self,
        time_window: float = 5.0
    ) -> Dict[str, Any]:
        """
        Analyze current neurochemical state.

        Args:
            time_window: Analysis time window in seconds

        Returns:
            Neurochemical state analysis
        """
        # Get recent measurements
        current_time = max(m.timestamp for m in self.measurement_buffer) \
                      if self.measurement_buffer else 0.0
        recent_measurements = [
            m for m in self.measurement_buffer
            if current_time - m.timestamp <= time_window
        ]

        if not recent_measurements:
            return {"status": "no_data", "patterns": {}}

        # Pattern recognition
        patterns = self.analyzer.predict_pattern(recent_measurements)

        # Calculate neurotransmitter concentrations
        nt_concentrations = {}
        for nt_type in NeurotransmitterType:
            nt_measurements = [
                m for m in recent_measurements
                if m.neurotransmitter == nt_type
            ]
            if nt_measurements:
                concentrations = [m.concentration for m in nt_measurements]
                confidences = [m.confidence for m in nt_measurements]

                # Weighted average
                weights = np.array(confidences)
                weights = weights / np.sum(weights) if np.sum(weights) > 0 else weights
                avg_concentration = np.average(concentrations, weights=weights)

                nt_concentrations[nt_type.value] = {
                    "concentration": float(avg_concentration),
                    "confidence": float(np.mean(confidences)),
                    "num_measurements": len(nt_measurements)
                }

        return {
            "status": "active",
            "patterns": patterns,
            "neurotransmitter_levels": nt_concentrations,
            "num_sensors_active": len(set(m.sensor_id for m in recent_measurements)),
            "measurement_count": len(recent_measurements),
            "time_window": time_window
        }

    def get_system_stats(self) -> Dict[str, Union[int, float]]:
        """Get monitoring system statistics."""
        return {
            "electrochemical_sensors": len(self.electrochemical_sensors),
            "optical_sensors": len(self.optical_sensors),
            "microdialysis_probes": len(self.microdialysis_probes),
            "total_measurements": len(self.measurement_buffer),
            "analyzer_parameters": sum(
                p.numel() for p in self.analyzer.network.parameters()
            ),
            "calibrated_sensors": sum(
                1 for s in self.electrochemical_sensors if s.is_calibrated
            )
        }
        }
