from _datetime import datetime
import os
import subprocess
import sys
import pandas as pd
import uuid
import logging
import warnings
import time
import yaml
import argparse
# Set the global warning filter to ignore all warnings
warnings.simplefilter("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the absolute path of the project root (one level up from MJO)
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
print('PROJECT_ROOT', PROJECT_ROOT)

# Path to the config file
CONFIG_FILE = os.path.join(PROJECT_ROOT, "salmon_config.yaml")
print(CONFIG_FILE)

sys.path.insert(0, os.path.abspath(os.getcwd()))


def read_inputs_from_command_line():
    """
    Parse command-line arguments for date, hour, area, and model.

    This function reads and validates the command-line arguments passed to the script.
    It expects a date in 'YYYY-MM-DD' format, an optional hour in 'HH' format (default: '00'),
    a valid area, and a valid model.

    Returns:
        dict: A dictionary containing the parsed date, hour, area, and model.

    Exits:
        If arguments are missing, invalid, or in the wrong format, the function prints
        usage instructions and exits the script.
    """
    parser = argparse.ArgumentParser(description="Process date, hour, area, and model inputs.")

    parser.add_argument('-d', '--date', type=str, required=True, help='Date in YYYY-MM-DD format')
    parser.add_argument('-t', '--time', type=str, required=False, default='00', help='Optional hour in HH format (default: 00)', choices=['00', '06', '12', '18'])
    parser.add_argument('-a', '--area', type=str, required=True, choices=['mjo', 'coldsurge', 'eqwaves', 'indices', 'bsiso'],
                        help='Area of interest')
    parser.add_argument('-m', '--model', type=str, required=True, choices=['analysis', 'mogreps', 'glosea'], help='Model selection')

    args = parser.parse_args()

    # Validate date format
    try:
        date_obj = datetime.strptime(args.date, '%Y-%m-%d')
    except ValueError as e:
        print(f"Error: {e}. Please provide the date in YYYY-MM-DD format.")
        parser.print_help()
        sys.exit(1)


    # Validate hour if provided
    if not args.time.isdigit() or not (0 <= int(args.time) <= 23):
        print(f"Error: Invalid hour '{args.time}'. Please provide an hour from ['00', '06', '12', '18']")
        parser.print_help()
        sys.exit(1)

    hour = int(args.time)  # Convert to integer
    date_obj = date_obj.replace(hour=hour)
    print(date_obj)

    return {
        'date': date_obj,
        'area': args.area.lower(),
        'model': args.model.lower()
    }

def print_dict(config_values):
    if config_values:
        for option, value in config_values.items():
            print(f'{option}: {value}')


def create_directories(config_values):
    """
    Reads a config.ini file, extracts directory paths, and creates the directories if they don't exist.

    Args:
        config_file (str): Path to the configuration file.
    """

    for key, path in config_values.items():
        # Skip paths that contain "moose" or have a "." (likely a file)
        if "moose" in path or "." in os.path.basename(path):
            print(f"Skipping: {path}")
            continue

        # Create directory if it doesn't exist
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created: {path}")
        else:
            print(f"Already exists: {path}")


def load_config(model=None, section=None):
    """
    Loads configuration values from a YAML file for a specified model and section.

    This function reads the configuration file specified by CONFIG_FILE, extracts options for the given model,
    and flattens any nested dictionaries into a single dictionary of configuration values. It also ensures that
    any directories specified in the configuration are created if they do not already exist.

    Args:
        model (str, optional): The key corresponding to the model section in the YAML configuration file.
        section (str, optional): (Currently unused) Intended for specifying a subsection within the model.

    Returns:
        dict: A dictionary containing configuration options and their corresponding values.

    Side Effects:
        - Prints the keys of the loaded configuration values.
        - Creates directories as specified in the configuration if they do not exist.
    """
    # Load the YAML file
    config_values = {}
    with open(CONFIG_FILE, "r") as file:
        config = yaml.safe_load(file)
        # Get options in the 'analysis' section and store in the dictionary
        for option, value in config[model].items():
            if isinstance(value, dict):
                for op, val in value.items():
                    config_values[op] = val
            else:
                config_values[option] = value
    print(config_values.keys())

    # Create the directories in the salmon_config.yaml file unless exist
    create_directories(config_values)

    return config_values

def retrieve_analysis_data(date, model, area):
    """
    Retrieve MJO data for a given date, model, and area.

    Args:
        date (datetime): The date for which to retrieve data.
        model (str): The model to use for data retrieval.
        area (str): The area of interest.

    Returns:
        None
    """
    config_values = load_config(model=model, section=area)
    print_dict(config_values)
    if model == 'analysis':    
        if area == 'mjo':
            print(area, model)
    
            from MJO.analysis import analysis_process as mjo_analysis_process

            reader = mjo_analysis_process.AnalysisProcess(config_values)
            status = reader.check_retrieve_201_prev_days(date, parallel=True)
            print(status)
            if status == 0:
                status = reader.combine_201_days_analysis_data(date)
                print(status)
            else:
                print(f'Error: Not all jobs in AnalysisProcess().check_retrieve_201_prev_days completed.')
                sys.exit()
                

if __name__ == '__main__':
    inputs = read_inputs_from_command_line()

    date = inputs['date']
    model = inputs['model']
    area = inputs['area']


    if area == 'mjo':
        if model == 'analysis':
            config_values = load_config(model=model, section=area)
            moosedir = os.path.join(config_values['analysis_moose_dir'], f'{str(date.year)}.pp')

            fc_times = [0]  # just the analysis data
            hr_list = ['00', '12']

            print(config_values)
            retrieve_analysis_data(date, model, area)

