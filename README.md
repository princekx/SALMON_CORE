# SALMON_CORE - Southeast Asia Large Scale Monitoring tool

SALMON_CORE is a suite of tools for sub-seasonal to seasonal (S3) climate analysis in Southeast Asia.

## Key Features
- **Modular Architecture**: Re-engineered into a `salmon` package with clear separation of core logic, utilities, and scientific modules.
- **Recipe-Driven Workflows**: Workflows are defined in YAML recipes, making it easy to create new experiments without changing code.
- **CLI Interface**: A unified command-line interface for running recipes.
- **Automated Path Resolution**: Supports environment variables and automatically generates output directories based on the model and recipe name.

## Installation
```bash
pip install -e .
```

## Usage
### Running a Recipe
```bash
salmon run recipes/mjo_mogreps.yaml --date 2026-02-12
```

## Documentation
- **Development Guide**: [How to add new recipes and tasks](docs/RECIPE_GUIDE.md)
- **API Documentation**: Detailed API documentation can be generated using Sphinx:
```bash
cd docs
make html
```
