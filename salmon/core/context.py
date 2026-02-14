from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional
import os

@dataclass
class Context:
    """Runtime context passed between tasks in a SALMON recipe.

    The Context object serves as a shared state container, carrying configuration
    details, runtime data, and state information through the execution pipeline.

    Attributes:
        date (datetime): The target date for the SALMON run (e.g., forecast date).
        recipe_name (str): The name of the recipe being executed.
        config (Dict[str, Any]): Default configuration values from the recipe.
        shared_data (Dict[str, Any]): A dictionary for tasks to share intermediate
            results or state.
    """
    date: datetime
    recipe_name: str = "unnamed"
    config: Dict[str, Any] = field(default_factory=dict)
    shared_data: Dict[str, Any] = field(default_factory=dict)
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Retrieve a configuration value by key.

        Args:
            key (str): The configuration key to look up.
            default (Any, optional): The default value to return if the key is not found.
                Defaults to None.

        Returns:
            Any: The configuration value or the default.
        """
        return self.config.get(key, default)

    def set_data(self, key: str, value: Any):
        """Store internal data in the shared context.

        Args:
            key (str): The key to store the data under.
            value (Any): The data to store.
        """
        self.shared_data[key] = value

    def get_data(self, key: str) -> Any:
        """Retrieve internal data from the shared context.

        Args:
            key (str): The key of the data to retrieve.

        Returns:
            Any: The stored data, or None if the key doesn't exist.
        """
        return self.shared_data.get(key)
