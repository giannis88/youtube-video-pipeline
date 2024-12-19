import os
import yaml
import logging
from typing import Dict, Any, Optional

class ConfigManager:
    """
    Manage configuration for the video processing pipeline
    Supports multiple configuration sources with precedence
    """
    
    def __init__(self, 
                 config_file: Optional[str] = None, 
                 env_prefix: str = 'VIDPIPE_'):
        """
        Initialize configuration manager
        
        :param config_file: Path to YAML configuration file
        :param env_prefix: Prefix for environment variable overrides
        """
        self.logger = logging.getLogger(__name__)
        self.config: Dict[str, Any] = {}
        self.env_prefix = env_prefix
        
        # Default configuration
        self.config = {
            'input_directory': os.path.join(os.path.dirname(__file__), '..', 'input'),
            'output_directory': os.path.join(os.path.dirname(__file__), '..', 'output'),
            'logging': {
                'level': 'INFO',
                'file': 'video_pipeline.log'
            },
            'video_processing': {
                'target_resolution': (1280, 720),
                'crf': 23,
                'preset': 'medium'
            },
            'ai': {
                'api_keys': [],
                'rate_limit': {
                    'requests_per_minute': 15,
                    'total_requests_per_day': 1500
                }
            }
        }
        
        # Load configuration from file if provided
        if config_file and os.path.exists(config_file):
            self._load_yaml_config(config_file)
        
        # Override with environment variables
        self._load_env_config()
        
        # Configure logging based on config
        self._configure_logging()

    def _load_yaml_config(self, config_file: str):
        """
        Load configuration from YAML file
        
        :param config_file: Path to YAML configuration file
        """
        try:
            with open(config_file, 'r') as f:
                yaml_config = yaml.safe_load(f)
                self._deep_update(self.config, yaml_config)
            self.logger.info(f"Loaded configuration from {config_file}")
        except Exception as e:
            self.logger.error(f"Error loading configuration file: {e}")

    def _load_env_config(self):
        """
        Override configuration with environment variables
        Uses self.env_prefix to identify configuration overrides
        """
        for key, value in os.environ.items():
            if key.startswith(self.env_prefix):
                # Remove prefix and convert to lowercase
                config_key = key[len(self.env_prefix):].lower()
                
                # Try to convert value to appropriate type
                try:
                    # Convert to appropriate type (bool, int, float, or keep as string)
                    if value.lower() in ['true', 'false']:
                        converted_value = value.lower() == 'true'
                    elif value.isdigit():
                        converted_value = int(value)
                    elif self._is_float(value):
                        converted_value = float(value)
                    else:
                        converted_value = value
                    
                    # Update nested configuration
                    self._set_nested_config(self.config, config_key.split('_'), converted_value)
                except Exception as e:
                    self.logger.warning(f"Could not process environment variable {key}: {e}")

    def _set_nested_config(self, config: Dict, keys: list, value: Any):
        """
        Set a nested configuration value
        
        :param config: Configuration dictionary
        :param keys: List of nested keys
        :param value: Value to set
        """
        for key in keys[:-1]:
            config = config.setdefault(key, {})
        config[keys[-1]] = value

    def _deep_update(self, original: Dict, update: Dict):
        """
        Recursively update nested dictionaries
        
        :param original: Original configuration dictionary
        :param update: Dictionary with updates
        """
        for key, value in update.items():
            if isinstance(value, dict):
                original[key] = self._deep_update(original.get(key, {}), value)
            else:
                original[key] = value
        return original

    def _configure_logging(self):
        """
        Configure logging based on configuration
        """
        log_level = getattr(logging, self.config['logging']['level'].upper(), logging.INFO)
        log_file = self.config['logging']['file']
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename=log_file,
            filemode='a'
        )

    def _is_float(self, value: str) -> bool:
        """
        Check if a string can be converted to a float
        
        :param value: String to check
        :return: True if convertible to float, False otherwise
        """
        try:
            float(value)
            return True
        except ValueError:
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value with optional default
        
        :param key: Configuration key (dot-separated for nested)
        :param default: Default value if key not found
        :return: Configuration value
        """
        try:
            # Navigate through nested dictionary
            value = self.config
            for k in key.split('.'):
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def save(self, config_file: str):
        """
        Save current configuration to a YAML file
        
        :param config_file: Path to save configuration
        """
        try:
            with open(config_file, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            self.logger.info(f"Configuration saved to {config_file}")
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")

# Global configuration instance
config = ConfigManager()
