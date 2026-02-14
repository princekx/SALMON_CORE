import click
import os
import sys
import logging
from datetime import datetime
from salmon.core.runner import Runner
from salmon.utils.config import load_global_config

@click.group(help="SALMON: Southeast Asia Large Scale Monitoring tool")
def main():
    """SALMON: Southeast Asia Large Scale Monitoring tool CLI"""
    pass

@main.command()
@click.argument('recipe_path', type=click.Path(exists=True))
@click.option('--date', type=click.DateTime(formats=["%Y-%m-%d"]), required=True,
              help="Target date (forecast init date) in YYYY-MM-DD format.")
@click.option('--debug', is_flag=True, help="Enable debug logging.")
def run(recipe_path, date, debug):
    """Execute a SALMON analysis recipe for a specific date."""
    try:
        runner = Runner(recipe_path, date, debug=debug)
        runner.run()
    except Exception as e:
        click.echo(f"Run failed: {e}", err=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
