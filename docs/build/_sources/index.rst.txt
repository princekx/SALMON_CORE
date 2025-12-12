.. SALMON documentation master file, created by
   sphinx-quickstart on Tue Mar 25 13:25:19 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SALMON documentation
====================

.. Salmon Forecast Processing Script Documentation
================================================

This documentation provides an overview of the Salmon Forecast Processing system.
It explains how to use the script, its parameters, and its functions.

.. contents:: Table of Contents
   :depth: 2
   :local:

Introduction
------------
This script processes meteorological data for different areas and models using forecast and analysis tools.
It reads input parameters from the command line, loads configuration settings, retrieves data,
processes analysis and forecast information, and generates visual outputs.

Usage
-----
To run the script, use the following command:

.. code-block:: bash

   python script.py -d YYYY-MM-DD -t HH -a AREA -m MODEL

Arguments:

- ``-d, --date``  : Required. Date in `YYYY-MM-DD` format.
- ``-t, --time``  : Optional. Time in `HH` format (default: `00`). Choices: `00`, `06`, `12`, `18`.
- ``-a, --area``  : Required. Area of interest. Choices: `mjo`, `coldsurge`, `eqwaves`, `bsiso`.
- ``-m, --model`` : Required. Model selection. Choices: `mogreps`, `glosea`.

Functions
---------

.. autofunction:: read_inputs_from_command_line

.. autofunction:: print_dict

.. autofunction:: create_directories

.. autofunction:: load_config

Function Details
----------------

``read_inputs_from_command_line``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Parses command-line arguments and validates them.

Returns:
    - A dictionary with parsed date, hour, area, and model.

``print_dict``
^^^^^^^^^^^^^^
Prints the key-value pairs of a dictionary.

``create_directories``
^^^^^^^^^^^^^^^^^^^^^^
Reads a configuration file and creates directories if they do not exist.

``load_config``
^^^^^^^^^^^^^^^
Loads the YAML configuration file and ensures required directories exist.

Processing Workflow
-------------------
1. **Read Input Parameters**: Parses the command-line arguments.
2. **Load Configuration**: Reads `salmon_config.yaml` to extract required paths and settings.
3. **Process Data Based on Area and Model**:
    - **MJO**: Runs analysis and retrieves forecast data.
    - **Cold Surge**: Processes forecast data and generates visual outputs.
    - **Equatorial Waves**: Runs analysis and retrieves forecast data.

Expected Output
---------------
- Processed forecast data for the selected area and model.
- Data visualization using `bokeh` plots.

Error Handling
--------------
- Validates date and time formats.
- Checks existence of required configuration settings and directories.
- Handles missing or incorrect input parameters.

Example Usage
-------------

.. code-block:: bash

   python script.py -d 2025-03-25 -t 06 -a mjo -m mogreps

This command processes MJO data using the `mogreps` model for `March 25, 2025, 06:00 UTC`.



.. toctree::
   :maxdepth: 2
   :caption: Modules

   modules

