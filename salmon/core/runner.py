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
    def __init__(self, recipe_path: str, date: datetime, model: str = None, debug: bool = False):
        """Initialize the Runner.

        Args:
            recipe_path (str): Path to the YAML recipe file.
            date (datetime): Target date for the run.
            debug (bool, optional): If True, set logging to DEBUG level.
                Defaults to False.
        """
        self.recipe_path = recipe_path
        self.date = date
        self.model = model.lower() if model else None
        self._setup_logging(debug)
        self.recipe = self._load_recipe()
        
        # Get recipe name from YAML or filename
        recipe_name = self.recipe.get('name')
        if not recipe_name:
            recipe_name = os.path.splitext(os.path.basename(recipe_path))[0]
            
        context_config = dict(self.recipe.get('defaults', {}))
        if self.model:
            context_config['model'] = self.model

        self.context = Context(
            date=self.date,
            recipe_name=recipe_name,
            config=context_config
        )

    def _task_text(self, task_conf: dict) -> str:
        """Return lowercase searchable text built from common task fields."""
        return " ".join(
            str(task_conf.get(key, "")) for key in ("name", "module", "class")
        ).lower()

    def _selected_task_model_match(self, task_conf: dict, selected_model: str) -> bool:
        """Match selected model against optional explicit task model fields."""
        task_model = task_conf.get("model")
        if isinstance(task_model, str):
            return task_model.lower() == selected_model

        task_models = task_conf.get("models")
        if isinstance(task_models, list):
            return any(str(m).lower() == selected_model for m in task_models)

        return False

    def _filter_tasks_for_model(self, tasks_config: list, selected_model: str) -> list:
        """Filter tasks so model-specific recipes run only selected-model tasks."""
        recipe_defaults = self.recipe.get("defaults", {})
        known_models = set()

        defaults_models = recipe_defaults.get("models")
        if isinstance(defaults_models, list):
            known_models.update(str(m).lower() for m in defaults_models)

        default_model = recipe_defaults.get("model")
        if isinstance(default_model, str):
            known_models.add(default_model.lower())

        known_models.add(selected_model)
        other_models = {m for m in known_models if m != selected_model}

        filtered = []
        for task_conf in tasks_config:
            text = self._task_text(task_conf)

            if "model" in task_conf or "models" in task_conf:
                if self._selected_task_model_match(task_conf, selected_model):
                    filtered.append(task_conf)
                continue

            if selected_model in text:
                filtered.append(task_conf)
                continue

            if any(other in text for other in other_models):
                continue

            # Keep generic tasks that are not tied to a specific model.
            filtered.append(task_conf)

        return filtered

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
        if self.model:
            total = len(tasks_config)
            tasks_config = self._filter_tasks_for_model(tasks_config, self.model)
            logger.info(
                "Model filter enabled: %s (tasks selected: %d/%d)",
                self.model,
                len(tasks_config),
                total,
            )
            if not tasks_config:
                raise ValueError(
                    f"No tasks matched model '{self.model}' in recipe '{self.recipe_path}'."
                )

        for task_conf in tasks_config:
            task_name = task_conf.get('name', 'Unknown Task')
            module_name = task_conf.get('module')
            class_name = task_conf.get('class')
            task_config = task_conf.get('config', {})
            
            print(f"Task: {task_name}, Module: {module_name}, Class: {class_name}, Config: {task_config}")
            
            logger.info(f"Running task: {task_name}")
            try:
                TaskClass = Registry.load_task_class(module_name, class_name)
                task_instance = TaskClass(self.context, task_config)
                task_instance.run()
            except Exception as e:
                logger.error(f"Task '{task_name}' failed: {e}")
                raise

        logger.info("Recipe completed successfully.")
