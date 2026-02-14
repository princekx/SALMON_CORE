import yaml
import logging
import os
from datetime import datetime
from salmon.core.context import Context
from salmon.core.registry import Registry

logger = logging.getLogger(__name__)

class Runner:
    """Orchestrates the execution of a SALMON recipe.

    The Runner loads a YAML recipe, sets up the execution context, and
    dynamically loads and runs a series of tasks.

    Attributes:
        recipe_path (str): Path to the YAML recipe file.
        date (datetime): Target date for the run.
        recipe (dict): The loaded recipe configuration.
        context (Context): The shared runtime context.
    """
    def __init__(self, recipe_path: str, date: datetime, debug: bool = False):
        """Initialize the Runner.

        Args:
            recipe_path (str): Path to the YAML recipe file.
            date (datetime): Target date for the run.
            debug (bool, optional): If True, set logging to DEBUG level.
                Defaults to False.
        """
        self.recipe_path = recipe_path
        self.date = date
        self._setup_logging(debug)
        self.recipe = self._load_recipe()
        
        # Get recipe name from YAML or filename
        recipe_name = self.recipe.get('name')
        if not recipe_name:
            recipe_name = os.path.splitext(os.path.basename(recipe_path))[0]
            
        self.context = Context(
            date=self.date, 
            recipe_name=recipe_name,
            config=self.recipe.get('defaults', {})
        )

    def _setup_logging(self, debug: bool):
        """Configure the logging system.

        Args:
            debug (bool): If True, use DEBUG level; otherwise, use INFO.
        """
        logging.basicConfig(
            level=logging.DEBUG if debug else logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def _load_recipe(self) -> dict:
        """Load and parse the YAML recipe from disk.

        Returns:
            dict: The parsed recipe content.
        """
        with open(self.recipe_path, 'r') as f:
            return yaml.safe_load(f)

    def run(self):
        """Execute all tasks defined in the loaded recipe.

        This method iterates through the task list in the recipe, loads
        the corresponding Task classes via the Registry, and executes
        their `run` methods.

        Raises:
            Exception: If any task execution fails.
        """
        logger.info(f"Starting recipe: {self.recipe.get('name', 'Unnamed')}")
        
        tasks_config = self.recipe.get('tasks', [])
        for task_conf in tasks_config:
            task_name = task_conf.get('name', 'Unknown Task')
            module_name = task_conf.get('module')
            class_name = task_conf.get('class')
            task_config = task_conf.get('config', {})
            
            logger.info(f"Running task: {task_name}")
            try:
                TaskClass = Registry.load_task_class(module_name, class_name)
                task_instance = TaskClass(self.context, task_config)
                task_instance.run()
            except Exception as e:
                logger.error(f"Task '{task_name}' failed: {e}")
                raise

        logger.info("Recipe completed successfully.")
