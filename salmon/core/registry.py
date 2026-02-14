import importlib
from typing import Type
from salmon.core.task import Task

class Registry:
    """Registry for dynamically loading SALMON Task classes.

    The Registry provides mechanisms to import and instantiate Task subclasses
    at runtime based on their string-based module and class names.
    """
    @staticmethod
    def load_task_class(module_name: str, class_name: str) -> Type[Task]:
        """Dynamically load a Task class from a module.

        Args:
            module_name (str): Full Python module path (e.g., 'salmon.modules.mjo.tasks').
            class_name (str): Name of the class to load within the module.

        Returns:
            Type[Task]: The loaded Task class.

        Raises:
            ImportError: If the module or class cannot be found.
            TypeError: If the loaded class is not a subclass of `Task`.
        """
        try:
            module = importlib.import_module(module_name)
            task_class = getattr(module, class_name)
            if not issubclass(task_class, Task):
                raise TypeError(f"Class {class_name} is not a subclass of Task")
            return task_class
        except (AttributeError, ImportError) as e:
            raise ImportError(f"Could not load task {class_name} from {module_name}: {e}")
