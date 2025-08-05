"""
Performance monitoring utility for the QNST project.
"""

import time
import psutil
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
from .logger import get_logger

logger = get_logger(__name__)

@dataclass
class SystemMetrics:
    timestamp: float
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float]
    network_io: Tuple[float, float]  # bytes sent/received per second
    disk_io: Tuple[float, float]     # bytes read/written per second

class PerformanceMonitor:
    """System performance monitoring for QNST."""
    
    def __init__(self, history_size: int = 3600):
        """
        Initialize performance monitor.
        
        Args:
            history_size (int): Number of historical metrics to keep
        """
        self.history_size = history_size
        self.metrics_history: List[SystemMetrics] = []
        self.last_network_io: Optional[Tuple[float, float]] = None
        self.last_disk_io: Optional[Tuple[float, float]] = None
        self.last_time: Optional[float] = None
        
    def get_current_metrics(self) -> SystemMetrics:
        """
        Get current system performance metrics.
        
        Returns:
            SystemMetrics: Current system metrics
        """
        current_time = time.time()
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=None)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # GPU usage (if available)
        gpu_percent = self._get_gpu_usage()
        
        # Network I/O
        network = psutil.net_io_counters()
        network_io = self._calculate_rate(
            (network.bytes_sent, network.bytes_recv),
            self.last_network_io,
            current_time
        )
        self.last_network_io = (network.bytes_sent, network.bytes_recv)
        
        # Disk I/O
        disk = psutil.disk_io_counters()
        disk_io = self._calculate_rate(
            (disk.read_bytes, disk.write_bytes),
            self.last_disk_io,
            current_time
        )
        self.last_disk_io = (disk.read_bytes, disk.write_bytes)
        
        # Update time
        self.last_time = current_time
        
        # Create metrics
        metrics = SystemMetrics(
            timestamp=current_time,
            cpu_usage=cpu_percent,
            memory_usage=memory_percent,
            gpu_usage=gpu_percent,
            network_io=network_io,
            disk_io=disk_io
        )
        
        # Update history
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.history_size:
            self.metrics_history.pop(0)
        
        return metrics
    
    def _get_gpu_usage(self) -> Optional[float]:
        """
        Get GPU usage if available.
        
        Returns:
            Optional[float]: GPU usage percentage or None if not available
        """
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return float(util.gpu)
        except:
            return None
    
    def _calculate_rate(
        self,
        current: Tuple[float, float],
        last: Optional[Tuple[float, float]],
        current_time: float
    ) -> Tuple[float, float]:
        """
        Calculate rate of change for I/O metrics.
        
        Args:
            current (Tuple[float, float]): Current values
            last (Optional[Tuple[float, float]]): Last values
            current_time (float): Current timestamp
            
        Returns:
            Tuple[float, float]: Rate of change in values per second
        """
        if last is None or self.last_time is None:
            return (0.0, 0.0)
        
        time_diff = current_time - self.last_time
        if time_diff == 0:
            return (0.0, 0.0)
        
        return (
            (current[0] - last[0]) / time_diff,
            (current[1] - last[1]) / time_diff
        )
    
    def get_average_metrics(self, window: int = 60) -> SystemMetrics:
        """
        Get average metrics over a time window.
        
        Args:
            window (int): Number of seconds to average over
            
        Returns:
            SystemMetrics: Average metrics
        """
        if not self.metrics_history:
            return self.get_current_metrics()
        
        # Get relevant metrics
        current_time = time.time()
        relevant_metrics = [
            m for m in self.metrics_history
            if current_time - m.timestamp <= window
        ]
        
        if not relevant_metrics:
            return self.get_current_metrics()
        
        # Calculate averages
        avg_cpu = np.mean([m.cpu_usage for m in relevant_metrics])
        avg_memory = np.mean([m.memory_usage for m in relevant_metrics])
        
        gpu_metrics = [m.gpu_usage for m in relevant_metrics if m.gpu_usage is not None]
        avg_gpu = np.mean(gpu_metrics) if gpu_metrics else None
        
        avg_network = (
            np.mean([m.network_io[0] for m in relevant_metrics]),
            np.mean([m.network_io[1] for m in relevant_metrics])
        )
        
        avg_disk = (
            np.mean([m.disk_io[0] for m in relevant_metrics]),
            np.mean([m.disk_io[1] for m in relevant_metrics])
        )
        
        return SystemMetrics(
            timestamp=current_time,
            cpu_usage=avg_cpu,
            memory_usage=avg_memory,
            gpu_usage=avg_gpu,
            network_io=avg_network,
            disk_io=avg_disk
        )
    
    def get_system_health(self) -> Dict[str, float]:
        """
        Get overall system health metrics.
        
        Returns:
            Dict[str, float]: Health metrics (0-1 scale)
        """
        metrics = self.get_average_metrics(window=300)  # 5-minute average
        
        return {
            'cpu_health': 1.0 - (metrics.cpu_usage / 100.0),
            'memory_health': 1.0 - (metrics.memory_usage / 100.0),
            'gpu_health': 1.0 - (metrics.gpu_usage / 100.0) if metrics.gpu_usage else 1.0,
            'network_health': self._calculate_io_health(metrics.network_io),
            'disk_health': self._calculate_io_health(metrics.disk_io),
            'overall_health': self._calculate_overall_health(metrics)
        }
    
    def _calculate_io_health(self, io_rates: Tuple[float, float]) -> float:
        """
        Calculate health metric for I/O rates.
        
        Args:
            io_rates (Tuple[float, float]): I/O rates
            
        Returns:
            float: Health metric (0-1)
        """
        # Assume healthy if I/O is below 80% of maximum observed
        max_rate = max(
            max(m.network_io[0] for m in self.metrics_history),
            max(m.network_io[1] for m in self.metrics_history)
        ) if self.metrics_history else float('inf')
        
        current_max = max(io_rates)
        return 1.0 - min(current_max / (max_rate if max_rate > 0 else float('inf')), 0.8)
    
    def _calculate_overall_health(self, metrics: SystemMetrics) -> float:
        """
        Calculate overall system health.
        
        Args:
            metrics (SystemMetrics): Current system metrics
            
        Returns:
            float: Overall health metric (0-1)
        """
        weights = {
            'cpu': 0.3,
            'memory': 0.3,
            'gpu': 0.2,
            'io': 0.2
        }
        
        cpu_health = 1.0 - (metrics.cpu_usage / 100.0)
        memory_health = 1.0 - (metrics.memory_usage / 100.0)
        gpu_health = 1.0 - (metrics.gpu_usage / 100.0) if metrics.gpu_usage else 1.0
        io_health = (self._calculate_io_health(metrics.network_io) + 
                    self._calculate_io_health(metrics.disk_io)) / 2
        
        return (
            weights['cpu'] * cpu_health +
            weights['memory'] * memory_health +
            weights['gpu'] * gpu_health +
            weights['io'] * io_health
        )
    
    def log_metrics(self) -> None:
        """Log current system metrics."""
        metrics = self.get_current_metrics()
        health = self.get_system_health()
        
        logger.info(f"System Metrics:")
        logger.info(f"CPU Usage: {metrics.cpu_usage:.1f}%")
        logger.info(f"Memory Usage: {metrics.memory_usage:.1f}%")
        if metrics.gpu_usage is not None:
            logger.info(f"GPU Usage: {metrics.gpu_usage:.1f}%")
        logger.info(f"Network I/O: {metrics.network_io[0]:.1f} B/s in, "
                   f"{metrics.network_io[1]:.1f} B/s out")
        logger.info(f"Disk I/O: {metrics.disk_io[0]:.1f} B/s read, "
                   f"{metrics.disk_io[1]:.1f} B/s write")
        logger.info(f"Overall Health: {health['overall_health']:.2f}")
