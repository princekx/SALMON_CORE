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
print(PROJECT_ROOT)
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
    parser.add_argument('-m', '--model', type=str, required=True, choices=['mogreps', 'glosea'], help='Model selection')

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


if __name__ == '__main__':
    inputs = read_inputs_from_command_line()

    date = inputs['date']
    model = inputs['model']
    area = inputs['area']

"""
class RetrieveData:


    def analysis_data_for_all(self, date):
        '''
        Retrieves analysis data for all 201 days of past data.

        This method generates and sorts a list of dates for the past 201 days
        from the forecast date. It then formats these dates and constructs the necessary
        file names and directories to retrieve the analysis data.

        :param date: The reference date from which to retrieve past data.
        :type date: datetime.date
        :return: None
        :rtype: None

        This function performs the following steps:
        1. Generates a list of dates for the past 201 days from the given date.
        2. Sorts the list of dates in ascending order.
        3. Formats the first and last dates to a specific string format.
        4. Creates a list of unique years involved in the date range.
        5. Constructs the file pattern to be used for data retrieval.
        6. Prints the range of dates for which data is being retrieved.
        7. Iterates over the unique years and:
           - Constructs the directory paths for the remote and local storage.
           - Generates a unique query file name.
           - Calls a worker function to perform the data retrieval.

        Example usage:
            >>> from datetime import datetime
            >>> analysis = AnalysisClass(config_values)
            >>> analysis.analysis_data_for_all(datetime.now())

        Note:
            - The method relies on the `self.config_values` dictionary to obtain various configuration values.
            - The method calls `self.retrieve_worker` to perform the actual data retrieval. This worker method
              should be defined in the class to handle the file operations.
        '''
        num_prev_days = 201

        missing_dates = [xdate for xdate in (date - datetime.timedelta(days=i)
                                             for i in range(num_prev_days))]
        missing_dates.sort()

        T1 = missing_dates[0].strftime('%Y/%m/%d 00:00:00')
        T2 = missing_dates[-1].strftime('%Y/%m/%d 18:00:00')

        # If there are is a change of year between the 120 dates, make a list of the two years
        data_years = list({str(xdate.year) for xdate in [missing_dates[0], missing_dates[-1]]})

        fct = '003'
        # Downloading 00, 06, 12, 18 files
        # 00 and 12 are used for MJO
        # all 00, 06, 12, 18 files are used for equatorial waves
        # prods_op_gl-mn_20240526_00_003.pp
        # prods_op_gl-mn_20240526_06_003.pp
        # prods_op_gl-mn_20240526_12_003.pp
        # prods_op_gl-mn_20240526_18_003.pp

        filemoose = f"prods_op_gl-mn_*_*_{fct}.pp"
        replacements = {"StartDate": T1, "EndDate": T2, 'filemoose': filemoose}

        print('##########################################################################################')
        print(f'Retrieving {num_prev_days} days analysis from {T1} - {T2} ')
        print('##########################################################################################')

        for data_year in data_years:

            moosedir = os.path.join(self.config_values['analysis_moose_dir'], f'{data_year}.pp')
            remote_data_dir = self.config_values['analysis_raw_dir']

            # Generate a unique query file
            local_query_file = os.path.join(self.config_values['analysis_dummy_queryfiles_dir'],
                                             f'mjo_eqwaves_analysis_query_{uuid.uuid4()}')

            # Do the work
            self.retrieve_worker(moosedir, self.config_values['analysis_combined_queryfile'],
                               local_query_file, replacements, remote_data_dir)

    def mogreps_data_for_all(self, date):
        '''
        Retrieves MOGREPS data for all lead times, all members, and for 00, 06, 12, and 18 hours.

        This method constructs the necessary directories and file names to retrieve the MOGREPS data
        for a given date. It handles the creation of directories if they do not exist and prepares
        the query files required for the data retrieval process.

        :param date: The reference date for which to retrieve MOGREPS data.
        :type date: datetime.date
        :return: None
        :rtype: None

        This function performs the following steps:
        1. Constructs the directory paths for MOGREPS data based on the provided date.
        2. Ensures the remote data directory exists, creating it if necessary.
        3. Constructs the file pattern to be used for data retrieval, taking into account
           the change in file names that occurred on 25/09/2018.
        4. Prints a status message indicating the start of data retrieval.
        5. Generates a unique local query file name.
        6. Calls a worker function to perform the data retrieval.

        Example usage:
            >>> from datetime import datetime
            >>> mogreps = MogrepsClass(config_values)
            >>> mogreps.mogreps_data_for_all(datetime.now())

        Note:
            - The method relies on the `self.config_values` dictionary to obtain various configuration values.
            - The method calls `self.retrieve_worker` to perform the actual data retrieval. This worker method
              should be defined in the class to handle the file operations.
        '''
        moosedir = os.path.join(self.config_values['mogreps_moose_dir'], f'{date.strftime("%Y%m")}.pp')
        remote_data_dir = self.config_values['mogreps_raw_dir']

        if not os.path.exists(remote_data_dir):
            os.makedirs(remote_data_dir)


        # File names changed on moose on 25/09/2018
        filemoose = f'prods_op_mogreps-g_{date.strftime("%Y%m%d")}_*.pp'
        replacements = {'filemoose': filemoose}

        # Generate a unique query file
        local_query_file = os.path.join(self.config_values['mogreps_dummy_queryfiles_dir'],
                                         f'localquery_{uuid.uuid1()}')

        # Do the work
        self.retrieve_worker(moosedir, self.config_values['mogreps_combined_queryfile'],
                           local_query_file, replacements, remote_data_dir)

    def retrieve_worker(self, moosedir, query_file, local_query_file, replacements, remote_data_dir):
        '''
        Performs the data retrieval operation by generating a local query file with replacements
        and executing the MOOSE client command to fetch data from the specified directory.

        :param moosedir: The directory on MOOSE from which data is to be retrieved.
        :type moosedir: str
        :param query_file: The path to the original query file.
        :type query_file: str
        :param local_query_file: The path to the locally generated query file with replacements.
        :type local_query_file: str
        :param replacements: A dictionary of placeholders and their replacements for the query file.
        :type replacements: dict
        :param remote_data_dir: The directory where the retrieved data will be stored locally.
        :type remote_data_dir: str
        :return: None
        :rtype: None

        This function performs the following steps:
        1. Ensures the remote data directory exists, creating it if necessary.
        2. Reads the original query file and writes a new local query file with the specified replacements.
        3. Constructs the command to execute the MOOSE client for data retrieval.
        4. Executes the command and logs the execution details, including any errors and the time taken for retrieval.

        Example usage:
            >>> replacements = {'StartDate': '2022/01/01 00:00:00', 'EndDate': '2022/01/31 18:00:00', 'filemoose': 'prods_op_gl-mn_*_*_003.pp'}
            >>> moosedir = 'moose:/opfc/atm/global/prods/202201.pp'
            >>> query_file = '/path/to/original_query_file'
            >>> local_query_file = '/path/to/local_query_file'
            >>> remote_data_dir = '/path/to/remote_data_dir'
            >>> instance.retrieve_worker(moosedir, query_file, local_query_file, replacements, remote_data_dir)

        Note:
            - This method uses the `subprocess` module to execute the command. Ensure that the necessary
              permissions and environment settings are in place for successful execution.
            - The method relies on `self.config_values` for configuration settings related to the MOOSE client and directories.
        '''
        if not os.path.exists(remote_data_dir):
            os.makedirs(remote_data_dir)

        with open(query_file) as query_infile, \
                open(local_query_file, 'w') as query_outfile:
            for line in query_infile:
                for src, target in replacements.items():
                    line = line.replace(src, target)
                query_outfile.write(line)

        command = f'/opt/moose-client-wrapper/bin/moo select --fill-gaps {local_query_file} {moosedir} {os.path.join(remote_data_dir)}'
        logger.info('Executing command: %s', command)
        print(command)
        # call command
        self.command_helper(command)

    def command_helper(self, command):
        try:
            start_time = time.time()  # Record the start time
            subprocess.run(command, shell=True, check=True)
            end_time = time.time()  # Record the end time
            elapsed_time = end_time - start_time  # Calculate the elapsed time
            logger.info('Data retrieval successful.')
            logger.info('Time taken for data retrieval: %.2f mins', (elapsed_time / 60.))
        except subprocess.CalledProcessError as e:
            logger.error('Error during data retrieval: %s', e)

    def glosea_data_for_all(self, date, prod='prodf',
                             gs_mass_get_command='~sfcpp/bin/MassGet/gs_mass_get'):
        '''
        Retrieve Glosea data using the gs_mass_get tool.
        It first reads the data catalogue and make sure the data is available
        Then generates a mass query based on sample with modifications to date, suite etc.
        :param date:
        :type date:
        :param prod:
        :type prod:
        :param fcast_in_dir_new:
        :type fcast_in_dir_new:
        :param gs_mass_get_command:
        :type gs_mass_get_command:
        :return:
        :rtype:
        '''
        # read data catalogue
        df = pd.read_csv('~sfcpp/bin/MassGet_py3/archived_data/%s.csv' % prod)
        # select moose info for the date
        df = df.loc[(df.iyr == date.year) & (df.imon == date.month) & (df.iday == date.day)]
        if not df.empty:
            # Add a few helper columns in to the DataFrame
            df['sys'] = df['sys'].replace('op', 'operational')
            df['psnum'] = [ps[2:] for ps in df['osuite'].values]
            df.loc[(df['prod'] == ('prodf')), 'suite'] = 'forecast'
            df.loc[(df['prod'] == ('prodm')), 'suite'] = 'monthly'

            sample_query_file = self.config_values['glosea_combined_queryfile']
            # read query sample files
            # the file to dump the new query based on sample above

            # Generate a unique query file
            local_query_file = os.path.join(self.config_values['glosea_dummy_queryfiles_dir'],
                                             f'localquery_{uuid.uuid1()}')
            fcast_in_dir = self.config_values['glosea_raw_dir']
            # Replace the key words with real filter info in query file
            replacements = {'MODE': df.iloc[0].sys, 'PSNUM': df.iloc[0].psnum, 'SUITE': df.iloc[0].suite,
                            'SYSTEM': df.iloc[0].config, 'START_DATE': date.strftime("%d/%m/%Y"),
                            'FINAL_DATE': date.strftime("%d/%m/%Y"), 'OUTPUT_DIR': fcast_in_dir}
            # Replace keywords
            with open(sample_query_file) as query_infile, open(local_query_file, 'w') as query_outfile:
                for line in query_infile:
                    for src, target in replacements.items():
                        line = line.replace(src, target)
                    query_outfile.write(line)


            flag_file = os.path.join(fcast_in_dir, df.iloc[0].suite, date.strftime("%Y%m%d"),
                                     'glosea_data_retrieval_flag')

            print(flag_file)

            # Write a flag if data is written
            if not os.path.exists(flag_file):
                # linux command to execute
                command = '%s %s' % (gs_mass_get_command, local_query_file)
                # call the linux command
                status = self.command_helper(command)

                if status == 0:
                    command = '/usr/bin/touch %s' % flag_file
                    self.command_helper(command)
                    print('Retrieval flag: %s' %flag_file)
                elif os.path.exists(flag_file):
                    command = '/usr/bin/rm -f %s' %flag_file
                    self.command_helper(command)

                # list files
                list_command = 'ls -lrt  %s' % os.path.join(fcast_in_dir, df.iloc[0].suite, date.strftime("%Y%m%d"))
                self.command_helper(list_command)


if __name__ == '__main__':
    date = datetime.datetime(2024, 6, 3)
    reader = RetrieveData('analysis')
    reader.analysis_data_for_all(date)

    reader = RetrieveData('mogreps')
    reader.mogreps_data_for_all(date)

    reader = RetrieveData('glosea')
    reader.glosea_data_for_all(date, prod='prodf')
    reader.glosea_data_for_all(date, prod='prodm')
"""