"""
Configuration loader utility for the QNST project.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

class ConfigLoader:
    """Configuration loader for QNST project."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize config loader.
        
        Args:
            config_dir (str, optional): Path to config directory
        """
        self.config_dir = Path(config_dir) if config_dir else Path("config")
        self._config_cache: Dict[str, Dict[str, Any]] = {}
    
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_name (str): Name of the config file (without .yaml extension)
            
        Returns:
            Dict[str, Any]: Configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid
        """
        # Return cached config if available
        if config_name in self._config_cache:
            return self._config_cache[config_name]
        
        # Load config file
        config_path = self.config_dir / f"{config_name}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Cache the config
        self._config_cache[config_name] = config
        return config
    
    def get_value(self, config_name: str, *keys: str, default: Any = None) -> Any:
        """
        Get a specific value from a config file using dot notation.
        
        Args:
            config_name (str): Name of the config file
            *keys (str): Keys to traverse
            default (Any, optional): Default value if key doesn't exist
            
        Returns:
            Any: Configuration value
        """
        config = self.load_config(config_name)
        
        result = config
        for key in keys:
            if isinstance(result, dict) and key in result:
                result = result[key]
            else:
                return default
        
        return result
    
    def reload_config(self, config_name: str) -> Dict[str, Any]:
        """
        Force reload of a config file.
        
        Args:
            config_name (str): Name of the config file
            
        Returns:
            Dict[str, Any]: Reloaded configuration
        """
        # Remove from cache if present
        self._config_cache.pop(config_name, None)
        
        # Load and return fresh config
        return self.load_config(config_name)
    
    def get_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Load all config files in the config directory.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of all configurations
        """
        configs = {}
        
        for config_file in self.config_dir.glob("*.yaml"):
            config_name = config_file.stem
            configs[config_name] = self.load_config(config_name)
        
        return configs
    
    def validate_config(self, config_name: str, required_keys: Dict[str, type]) -> bool:
        """
        Validate config file against required keys and their types.
        
        Args:
            config_name (str): Name of the config file
            required_keys (Dict[str, type]): Dictionary of required keys and their types
            
        Returns:
            bool: True if config is valid, False otherwise
        """
        try:
            config = self.load_config(config_name)
            
            for key, expected_type in required_keys.items():
                if key not in config:
                    return False
                if not isinstance(config[key], expected_type):
                    return False
            
            return True
            
        except (FileNotFoundError, yaml.YAMLError):
            return False
