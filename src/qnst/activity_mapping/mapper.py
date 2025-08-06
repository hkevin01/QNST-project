"""Neural activity mapping system for QNST project."""

import logging
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE

from ..pattern_recognition.recognizer import PatternRecognitionSystem
from ..signal_processing.processor import SignalProcessor


class SpatialMapper:
    """3D spatial mapping of neural activity."""

    def __init__(self, electrode_positions: np.ndarray) -> None:
        """
        Initialize spatial mapper.

        Args:
            electrode_positions: Array of electrode 3D positions (N, 3)
        """
        self.electrode_positions = electrode_positions
        self.num_electrodes = len(electrode_positions)
        self.activity_maps = []

    def create_activity_map(
        self, neural_activity: np.ndarray, interpolation_resolution: int = 50
    ) -> np.ndarray:
        """
        Create 3D activity map from electrode data.

        Args:
            neural_activity: Activity values for each electrode
            interpolation_resolution: Resolution of 3D grid

        Returns:
            3D activity map
        """
        # Create 3D grid
        pos = self.electrode_positions
        x_min, x_max = pos[:, 0].min(), pos[:, 0].max()
        y_min, y_max = pos[:, 1].min(), pos[:, 1].max()
        z_min, z_max = pos[:, 2].min(), pos[:, 2].max()

        x_grid = np.linspace(x_min, x_max, interpolation_resolution)
        y_grid = np.linspace(y_min, y_max, interpolation_resolution)
        z_grid = np.linspace(z_min, z_max, interpolation_resolution)

        # Initialize activity map
        activity_map = np.zeros(
            (
                interpolation_resolution,
                interpolation_resolution,
                interpolation_resolution,
            )
        )

        # Simple inverse distance weighting interpolation
        for i, x in enumerate(x_grid):
            for j, y in enumerate(y_grid):
                for k, z in enumerate(z_grid):
                    point = np.array([x, y, z])

                    # Calculate distances to all electrodes
                    distances = np.linalg.norm(self.electrode_positions - point, axis=1)

                    # Avoid division by zero
                    distances = np.maximum(distances, 1e-6)

                    # Inverse distance weighting
                    weights = 1 / (distances**2)
                    weights /= np.sum(weights)

                    # Interpolated activity
                    activity_map[i, j, k] = np.sum(weights * neural_activity)

        return activity_map

    def get_activity_hotspots(
        self, activity_map: np.ndarray, threshold_percentile: float = 90
    ) -> List[Tuple]:
        """
        Identify activity hotspots in the 3D map.

        Args:
            activity_map: 3D activity map
            threshold_percentile: Percentile threshold for hotspots

        Returns:
            List of hotspot coordinates
        """
        threshold = np.percentile(activity_map, threshold_percentile)
        hotspot_indices = np.where(activity_map > threshold)

        hotspots = []
        for i in range(len(hotspot_indices[0])):
            coords = (
                hotspot_indices[0][i],
                hotspot_indices[1][i],
                hotspot_indices[2][i],
            )
            intensity = activity_map[coords]
            hotspots.append((coords, intensity))

        # Sort by intensity
        hotspots.sort(key=lambda x: x[1], reverse=True)

        return hotspots


class ConnectivityMapper:
    """Neural connectivity mapping using correlation analysis."""

    def __init__(self, num_channels: int) -> None:
        """
        Initialize connectivity mapper.

        Args:
            num_channels: Number of neural channels
        """
        self.num_channels = num_channels
        self.connectivity_history = []

    def compute_correlation_matrix(self, neural_data: np.ndarray) -> np.ndarray:
        """
        Compute correlation matrix between channels.

        Args:
            neural_data: Neural data (channels x time)

        Returns:
            Correlation matrix (channels x channels)
        """
        return np.corrcoef(neural_data)

    def compute_coherence_matrix(
        self, neural_data: np.ndarray, sampling_rate: float
    ) -> np.ndarray:
        """
        Compute coherence matrix between channels.

        Args:
            neural_data: Neural data (channels x time)
            sampling_rate: Sampling rate in Hz

        Returns:
            Coherence matrix (channels x channels)
        """
        from scipy.signal import coherence

        coherence_matrix = np.zeros((self.num_channels, self.num_channels))

        for i in range(self.num_channels):
            for j in range(i, self.num_channels):
                if i == j:
                    coherence_matrix[i, j] = 1.0
                else:
                    freq, coh = coherence(
                        neural_data[i, :], neural_data[j, :], fs=sampling_rate
                    )
                    # Average coherence across frequencies
                    coherence_matrix[i, j] = np.mean(coh)
                    coherence_matrix[j, i] = coherence_matrix[i, j]

        return coherence_matrix

    def identify_functional_networks(
        self, connectivity_matrix: np.ndarray, threshold: float = 0.7
    ) -> List[List[int]]:
        """
        Identify functional networks using clustering.

        Args:
            connectivity_matrix: Connectivity matrix
            threshold: Connectivity threshold

        Returns:
            List of functional networks (lists of channel indices)
        """
        # Threshold the connectivity matrix
        binary_connectivity = connectivity_matrix > threshold

        # Simple approach: find connected components
        networks = []
        visited = set()

        for i in range(self.num_channels):
            if i not in visited:
                network = []
                stack = [i]

                while stack:
                    node = stack.pop()
                    if node not in visited:
                        visited.add(node)
                        network.append(node)

                        # Add connected nodes
                        connected = np.where(binary_connectivity[node, :])[0]
                        for neighbor in connected:
                            if neighbor not in visited:
                                stack.append(neighbor)

                if len(network) > 1:  # Only keep networks with >1 node
                    networks.append(network)

        return networks


class ActivityMapper:
    """Main neural activity mapping system."""

    def __init__(
        self,
        signal_processor: SignalProcessor,
        pattern_recognizer: PatternRecognitionSystem,
        electrode_positions: np.ndarray,
    ) -> None:
        """
        Initialize activity mapper.

        Args:
            signal_processor: Signal processing pipeline
            pattern_recognizer: Pattern recognition system
            electrode_positions: 3D positions of electrodes
        """
        self.signal_processor = signal_processor
        self.pattern_recognizer = pattern_recognizer
        self.electrode_positions = electrode_positions

        # Initialize mappers
        self.spatial_mapper = SpatialMapper(electrode_positions)
        self.connectivity_mapper = ConnectivityMapper(len(electrode_positions))

        # Neural network for activity-based mapping
        self.activity_net = self._build_activity_network()

        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _build_activity_network(self) -> nn.Module:
        """Build neural network for activity-based mapping."""

        class ActivityMappingNet(nn.Module):
            def __init__(self, input_size: int, hidden_size: int = 256):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_size),
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_size // 2),
                    nn.Linear(hidden_size // 2, hidden_size // 4),
                    nn.ReLU(),
                )

                self.decoder = nn.Sequential(
                    nn.Linear(hidden_size // 4, hidden_size // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_size // 2, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, input_size),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return encoded, decoded

        return ActivityMappingNet(len(self.electrode_positions))

    def create_comprehensive_map(
        self, neural_data: np.ndarray, time_window: float = 1.0
    ) -> Dict[str, Any]:
        """
        Create comprehensive neural activity map.

        Args:
            neural_data: Neural data (channels x time)
            time_window: Time window for analysis in seconds

        Returns:
            Comprehensive mapping results
        """
        # Process signal
        processed_data = self.signal_processor.process_chunk(neural_data)

        # Extract activity levels
        activity_levels = np.std(processed_data, axis=1)

        # Spatial mapping
        spatial_map = self.spatial_mapper.create_activity_map(activity_levels)
        hotspots = self.spatial_mapper.get_activity_hotspots(spatial_map)

        # Connectivity mapping
        correlation_matrix = self.connectivity_mapper.compute_correlation_matrix(
            processed_data
        )
        coherence_matrix = self.connectivity_mapper.compute_coherence_matrix(
            processed_data, self.signal_processor.electrode_array.sampling_rate
        )

        # Functional networks
        functional_networks = self.connectivity_mapper.identify_functional_networks(
            correlation_matrix
        )

        # Pattern recognition
        pattern_predictions = self.pattern_recognizer.predict(neural_data)

        # Activity-based mapping using autoencoder
        activity_tensor = torch.FloatTensor(activity_levels).unsqueeze(0)
        encoded_activity, reconstructed = self.activity_net(activity_tensor)

        # Dimensionality reduction for visualization
        if len(activity_levels) > 2:
            tsne = TSNE(n_components=2, random_state=42)
            activity_2d = tsne.fit_transform(activity_levels.reshape(1, -1))
        else:
            activity_2d = activity_levels.reshape(1, -1)

        return {
            "spatial_map": spatial_map,
            "hotspots": hotspots,
            "correlation_matrix": correlation_matrix,
            "coherence_matrix": coherence_matrix,
            "functional_networks": functional_networks,
            "pattern_predictions": pattern_predictions,
            "encoded_activity": encoded_activity.detach().numpy(),
            "activity_2d": activity_2d,
            "activity_levels": activity_levels,
            "reconstruction_error": torch.mse_loss(
                activity_tensor, reconstructed
            ).item(),
        }

    def track_activity_over_time(
        self, neural_data_sequence: List[np.ndarray], time_points: List[float]
    ) -> Dict[str, Any]:
        """
        Track neural activity changes over time.

        Args:
            neural_data_sequence: Sequence of neural data arrays
            time_points: Corresponding time points

        Returns:
            Temporal activity analysis
        """
        activity_timeline = []
        connectivity_timeline = []
        pattern_timeline = []

        for data, time_point in zip(neural_data_sequence, time_points):
            # Create map for this time point
            current_map = self.create_comprehensive_map(data)

            activity_timeline.append(
                {
                    "time": time_point,
                    "activity_levels": current_map["activity_levels"],
                    "hotspots": len(current_map["hotspots"]),
                    "reconstruction_error": current_map["reconstruction_error"],
                }
            )

            connectivity_timeline.append(
                {
                    "time": time_point,
                    "avg_correlation": np.mean(current_map["correlation_matrix"]),
                    "avg_coherence": np.mean(current_map["coherence_matrix"]),
                    "num_networks": len(current_map["functional_networks"]),
                }
            )

            pattern_timeline.append(
                {
                    "time": time_point,
                    "pattern_confidence": np.max(
                        current_map["pattern_predictions"]["ensemble"]
                    ),
                }
            )

        return {
            "activity_timeline": activity_timeline,
            "connectivity_timeline": connectivity_timeline,
            "pattern_timeline": pattern_timeline,
        }

    def get_mapping_statistics(self) -> Dict[str, Union[int, float]]:
        """Get mapping system statistics."""
        return {
            "num_electrodes": len(self.electrode_positions),
            "spatial_mapper_resolution": 50,
            "activity_net_parameters": sum(
                p.numel() for p in self.activity_net.parameters()
            ),
            "electrode_coverage_volume": self._calculate_electrode_volume(),
        }

    def _calculate_electrode_volume(self) -> float:
        """Calculate the volume covered by electrodes."""
        if len(self.electrode_positions) < 4:
            return 0.0

        # Simple bounding box volume
        mins = np.min(self.electrode_positions, axis=0)
        maxs = np.max(self.electrode_positions, axis=0)
        volume = np.prod(maxs - mins)

        return float(volume)
        return float(volume)
