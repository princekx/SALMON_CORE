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

## Running with Cylc 8

This project includes a ready-to-run Cylc 8 workflow to automate the execution of SALMON recipes.

### 1. Choose Your Platform

The workflow can be run on two platforms:
- `local`: Runs tasks as background jobs on your local machine.
- `spice`: Submits tasks to the SLURM scheduler on the SPICE platform.

### 2. One-Time Setup for SPICE

If you are running on `spice`, you need to perform this one-time setup to tell Cylc where to find the platform configuration:
```bash
cylc config --set 'global.cylc[symlink dirs] = /home/users/prince.xavier/MJO/SALMON_v2/SALMON_CORE/cylc/site'
```

### 3. Install the Workflow

Navigate to the `cylc` directory and install the workflow. This step registers the workflow with Cylc and only needs to be done once, or whenever `flow.cylc` changes.

```bash
cd cylc
cylc install --symlink
```

### 4. Run the Workflow

To start the workflow, use the `cylc play` command with your chosen platform:
```bash
cylc play SALMON --set="platform=<YOUR_PLATFORM>"
```
Replace `<YOUR_PLATFORM>` with either `local` or `spice`.

This will start the scheduler in the foreground. You can monitor all your running workflows with the command:
```bash
cylc gscan
```

### Local Run Example

Here is a complete example of how to install and run the workflow on your local machine:

1.  **Navigate to the workflow directory:**
    ```bash
    cd $USER/MJO/SALMON_v2/SALMON_CORE/cylc
    ```

2.  **Install the workflow:**
    ```bash
    cylc install --workflow-name=SALMON
    ```

3.  **Run the workflow:**
    ```bash
    cylc play SALMON --set="platform='local'"
    ```

## Documentation
- **Development Guide**: [How to add new recipes and tasks](docs/RECIPE_GUIDE.md)
- **API Documentation**: Detailed API documentation can be generated using Sphinx:
```bash
cd docs
make html
```
