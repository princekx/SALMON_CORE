import yaml
import os
from typing import Dict, Any

def load_global_config(config_path: str = None) -> Dict[str, Any]:
    """Load the global SALMON configuration file.

    This function searches for `salmon_config.yaml` in default locations
    if no path is provided. It automatically expands environment variables
    in the configuration values.

    Args:
        config_path (str, optional): Absolute path to the configuration file.
            If None, searches in the current directory and project root.
            Defaults to None.

    Returns:
        Dict[str, Any]: The loaded configuration dictionary. Returns an
            empty dictionary if no file is found.
    """
    if not config_path:
        # Default locations
        candidates = [
            "salmon_config.yaml",
            os.path.join(os.path.dirname(__file__), "../../salmon_config.yaml")
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                config_path = candidate
                break
    
    if not config_path or not os.path.exists(config_path):
        # Return empty config if no file found, to avoid crashing if defaults are provided in recipes
        return {}

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return expand_env_vars(config)

def expand_env_vars(data: Any) -> Any:
    """Recursively expand environment variables in configuration values.

    Args:
        data (Any): Dictionary, list, or string to expand.

    Returns:
        Any: The data with environment variables expanded.
    """
    if isinstance(data, dict):
        return {k: expand_env_vars(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [expand_env_vars(i) for i in data]
    elif isinstance(data, str):
        return os.path.expandvars(data)
    else:
        return data
