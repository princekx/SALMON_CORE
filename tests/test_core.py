"""
Tests for SALMON core components:
  - Context
  - Task (abstract base)
  - Registry
  - Runner
  - CLI entry point
"""
import os
import sys
import pytest
import textwrap
import tempfile
from datetime import datetime
from unittest.mock import MagicMock, patch, mock_open

# ---------------------------------------------------------------------------
# Context tests
# ---------------------------------------------------------------------------
class TestContext:
    """Tests for salmon.core.context.Context"""

    def _make_context(self, **kwargs):
        from salmon.core.context import Context
        defaults = dict(date=datetime(2024, 1, 15), recipe_name="test_recipe", config={})
        defaults.update(kwargs)
        return Context(**defaults)

    def test_default_fields(self):
        ctx = self._make_context()
        assert ctx.recipe_name == "test_recipe"
        assert ctx.date == datetime(2024, 1, 15)
        assert ctx.config == {}
        assert ctx.shared_data == {}

    def test_get_config_existing_key(self):
        ctx = self._make_context(config={"model": "mogreps", "members": 18})
        assert ctx.get_config("model") == "mogreps"
        assert ctx.get_config("members") == 18

    def test_get_config_missing_key_returns_default(self):
        ctx = self._make_context()
        assert ctx.get_config("nonexistent") is None
        assert ctx.get_config("nonexistent", "fallback") == "fallback"

    def test_set_and_get_data(self):
        ctx = self._make_context()
        ctx.set_data("cubes", {"olr": object()})
        result = ctx.get_data("cubes")
        assert "olr" in result

    def test_get_data_missing_key_returns_none(self):
        ctx = self._make_context()
        assert ctx.get_data("not_set") is None

    def test_shared_data_isolation_between_instances(self):
        ctx1 = self._make_context()
        ctx2 = self._make_context()
        ctx1.set_data("key", "value")
        assert ctx2.get_data("key") is None


# ---------------------------------------------------------------------------
# Task tests
# ---------------------------------------------------------------------------
class TestTask:
    """Tests for salmon.core.task.Task (abstract base)"""

    def _make_concrete_task(self, run_fn=None):
        from salmon.core.task import Task
        from salmon.core.context import Context

        class ConcreteTask(Task):
            def run(self):
                if run_fn:
                    run_fn(self)

        ctx = Context(date=datetime(2024, 1, 15), recipe_name="test")
        return ConcreteTask(ctx, {"param": 42})

    def test_cannot_instantiate_abstract_task(self):
        from salmon.core.task import Task
        with pytest.raises(TypeError):
            Task(MagicMock(), {})

    def test_concrete_task_stores_context_and_config(self):
        task = self._make_concrete_task()
        assert task.config == {"param": 42}
        assert task.context.recipe_name == "test"

    def test_concrete_task_run_is_callable(self):
        called = []
        task = self._make_concrete_task(run_fn=lambda t: called.append(True))
        task.run()
        assert called == [True]


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------
class TestRegistry:
    """Tests for salmon.core.registry.Registry"""

    def test_load_task_class_returns_correct_class(self):
        from salmon.core.registry import Registry
        TaskClass = Registry.load_task_class(
            "salmon.modules.mjo.tasks", "RetrieveAnalysisData"
        )
        from salmon.modules.mjo.tasks import RetrieveAnalysisData
        assert TaskClass is RetrieveAnalysisData

    def test_load_task_class_bad_module_raises_import_error(self):
        from salmon.core.registry import Registry
        with pytest.raises(ImportError):
            Registry.load_task_class("salmon.modules.nonexistent", "SomeClass")

    def test_load_task_class_bad_class_raises_import_error(self):
        from salmon.core.registry import Registry
        with pytest.raises(ImportError):
            Registry.load_task_class("salmon.modules.mjo.tasks", "NoSuchClass")

    def test_load_task_class_not_a_task_raises_type_error(self):
        """Loading something that is not a Task subclass should fail."""
        from salmon.core.registry import Registry

        # Patch in a non-Task class on a real module
        import salmon.modules.mjo.tasks as mjo_tasks
        class NotATask:
            pass
        mjo_tasks._TmpFakeClass = NotATask
        with pytest.raises((ImportError, TypeError)):
            Registry.load_task_class("salmon.modules.mjo.tasks", "_TmpFakeClass")
        del mjo_tasks._TmpFakeClass


# ---------------------------------------------------------------------------
# Runner tests
# ---------------------------------------------------------------------------
class TestRunner:
    """Tests for salmon.core.runner.Runner"""

    def _write_recipe(self, path, content):
        with open(path, "w") as f:
            f.write(textwrap.dedent(content))

    def test_runner_loads_recipe_and_sets_name(self):
        from salmon.core.runner import Runner
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write("name: MyTestRecipe\ndefaults:\n  model: test\ntasks: []\n")
            tmp = f.name
        try:
            runner = Runner(tmp, datetime(2024, 1, 15))
            assert runner.context.recipe_name == "MyTestRecipe"
        finally:
            os.unlink(tmp)

    def test_runner_uses_filename_when_no_name_in_recipe(self):
        from salmon.core.runner import Runner
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False,
                                        prefix="my_recipe_") as f:
            f.write("tasks: []\n")
            tmp = f.name
        try:
            runner = Runner(tmp, datetime(2024, 1, 15))
            # recipe_name should be derived from filename (without extension)
            assert runner.context.recipe_name == os.path.splitext(os.path.basename(tmp))[0]
        finally:
            os.unlink(tmp)

    def test_runner_empty_tasks_succeeds(self):
        from salmon.core.runner import Runner
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write("name: Empty\ntasks: []\n")
            tmp = f.name
        try:
            runner = Runner(tmp, datetime(2024, 1, 15))
            # Should not raise
            runner.run()
        finally:
            os.unlink(tmp)

    def test_runner_passes_date_to_context(self):
        from salmon.core.runner import Runner
        target_date = datetime(2025, 6, 21)
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write("name: DateTest\ntasks: []\n")
            tmp = f.name
        try:
            runner = Runner(tmp, target_date)
            assert runner.context.date == target_date
        finally:
            os.unlink(tmp)

    def test_runner_executes_task_via_registry(self):
        """Runner should dynamically load and call a real task class."""
        from salmon.core.runner import Runner
        from salmon.core.task import Task

        # Create a minimal concrete task in a temp module
        task_was_run = []

        # We monkey-patch Registry.load_task_class for this test
        class FakeTask(Task):
            def run(self):
                task_was_run.append(True)

        recipe_yaml = textwrap.dedent("""\
            name: TaskRunTest
            tasks:
              - name: "Fake Task"
                module: fake.module
                class: FakeTask
                config: {}
        """)
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(recipe_yaml)
            tmp = f.name
        try:
            with patch("salmon.core.registry.Registry.load_task_class", return_value=FakeTask):
                runner = Runner(tmp, datetime(2024, 1, 15))
                runner.run()
            assert task_was_run == [True]
        finally:
            os.unlink(tmp)

    def test_runner_task_failure_raises(self):
        from salmon.core.runner import Runner
        from salmon.core.task import Task

        class FailingTask(Task):
            def run(self):
                raise RuntimeError("Task intentionally failed")

        recipe_yaml = textwrap.dedent("""\
            name: FailTest
            tasks:
              - name: "Failing Task"
                module: fake.module
                class: FailingTask
                config: {}
        """)
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(recipe_yaml)
            tmp = f.name
        try:
            with patch("salmon.core.registry.Registry.load_task_class", return_value=FailingTask):
                runner = Runner(tmp, datetime(2024, 1, 15))
                with pytest.raises(RuntimeError, match="intentionally failed"):
                    runner.run()
        finally:
            os.unlink(tmp)


# ---------------------------------------------------------------------------
# CLI tests (smoke tests via Click test runner)
# ---------------------------------------------------------------------------
class TestCLI:
    """Smoke tests for the salmon CLI entry point."""

    def test_main_help(self):
        from click.testing import CliRunner
        from salmon.cli import main
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "SALMON" in result.output

    def test_run_help(self):
        from click.testing import CliRunner
        from salmon.cli import main
        runner = CliRunner()
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "--date" in result.output

    def test_run_missing_date_fails(self):
        from click.testing import CliRunner
        from salmon.cli import main
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write("name: Test\ntasks: []\n")
            tmp = f.name
        try:
            result = runner.invoke(main, ["run", tmp])
            # Missing required --date option should cause a non-zero exit
            assert result.exit_code != 0
        finally:
            os.unlink(tmp)

    def test_run_with_valid_recipe_and_no_tasks(self):
        """End-to-end: run the CLI with a valid empty recipe — should succeed."""
        from click.testing import CliRunner
        from salmon.cli import main
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write("name: EmptyCLI\ntasks: []\n")
            tmp = f.name
        try:
            result = runner.invoke(main, ["run", tmp, "--date", "2024-01-15"])
            assert result.exit_code == 0, result.output
        finally:
            os.unlink(tmp)
