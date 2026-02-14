from abc import ABC, abstractmethod
from typing import Dict, Any
from salmon.core.context import Context

class Task(ABC):
    """Abstract base class for all SALMON tasks.

    A Task represents a single unit of work in a recipe. Subclasses must
    implement the `run` method.

    Attributes:
        context (Context): The shared runtime context.
        config (Dict[str, Any]): Task-specific configuration dictionary.
    """
    def __init__(self, context: Context, config: Dict[str, Any]):
        """Initialize the task with context and configuration.

        Args:
            context (Context): The shared runtime context.
            config (Dict[str, Any]): Task-specific configuration.
        """
        self.context = context
        self.config = config

    @abstractmethod
    def run(self):
        """Execute the task logic.

        This method must be implemented by subclasses.
        """
        pass
