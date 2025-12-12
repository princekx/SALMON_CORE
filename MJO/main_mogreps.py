#!/usr/bin/env /data/apps/sss/environments/default-2024_09_02/bin/python
from datetime import datetime
import sys
sys.path.append('/home/users/prince.xavier/MJO/SALMON/MJO')
import time
from analysis import analysis_process
from mogreps import mogreps_process
from glosea import glosea_process
from lib import mjo_utils
from display import bokeh_display



# Function to read date from command line and create a datetime object
def read_date_from_command_line():
    """
    Reads a date string from the command line arguments and converts it to a datetime object.

    Expects the script to be called with a single argument representing the date in 'YYYY-MM-DD' format.
    If the argument is missing or the format is incorrect, prints usage instructions or an error message.

    Returns:
        datetime.datetime: The parsed datetime object if successful.
        None: If the argument is missing or the format is incorrect.
    """
    if len(sys.argv) != 2:
        print("Usage: python script.py <date>")
        print("Date format should be YYYY-MM-DD")
        return

    date_str = sys.argv[1]
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        print(f"Successfully created datetime object: {date_obj}")
    except ValueError as e:
        print(f"Error: {e}. Please provide the date in YYYY-MM-DD format.")

    return date_obj

def do_analysis(date):
    """
    Perform analysis for a given date by retrieving and combining analysis data.

    This function initializes an AnalysisProcess object, checks the retrieval status
    of analysis data for the previous 201 days (optionally in parallel), and then
    combines the analysis data for those days. The status of the combination process
    is printed to the console.

    Args:
        date (str or datetime): The date for which the analysis should be performed.

    Returns:
        None
    """
    #
    # Usage:
    reader = analysis_process.AnalysisProcess('analysis')
    status = reader.check_retrieve_201_prev_days(date, parallel=True)
    #print(status)
    #if status == 0:
    status = reader.combine_201_days_analysis_data(date)
    print(status)
    #else:
    #    print(f'Error: Not all jobs in AnalysisProcess().check_retrieve_201_prev_days completed.')
    #    #sys.exit()

def do_mogreps(date):
    """
    Processes MOGREPS ensemble data for a given date, including retrieval, combination, MJO calculation, and visualization.

    Args:
        date (str): The date for which to process MOGREPS data, typically in 'YYYYMMDD' format.

    Workflow:
        1. Initializes a MOGREPS data processor.
        2. Retrieves MOGREPS ensemble data for the specified date.
        3. Combines 201 days of analysis and forecast data for all ensemble members.
        4. Runs parallel MJO (Madden-Julian Oscillation) calculations on the processed data.
        5. Generates and displays a Bokeh plot of the RMM (Real-time Multivariate MJO) indices.

    Prints:
        - Configuration values, retrieval status, member list, combination status, MJO process status.

    Note:
        - The function assumes the existence of `mogreps_process`, `mjo_utils`, and `bokeh_display` modules.
        - Error handling and waiting mechanisms are present but commented out.
    """
    reader = mogreps_process.MOGProcess('mogreps')
    print(reader.config_values)
    # This retrieves 35 members
    status1 = reader.retrieve_mogreps_data(date)
    print(status1)

    # All ensemble members
    members = [str('%03d' % mem) for mem in range(36)]
    print(members)

    #if status1:
    #    # Combine and prepare the data
    status2 = reader.combine_201_days_analysis_and_forecast_data(date, members)
    print(f'combine_201_days_analysis_and_forecast_data: {status2}')
    #else:
    #    print(f'Error: Not all jobs in retrieve_mogreps_data() completed.')
    #    #sys.exit()
    #time.sleep(60)  # Wait for 60 seconds

    # MJO calculations
    #if status2:
    mjo_proc = mjo_utils.MJOUtils('mogreps')
    status3 = mjo_proc.run_parallel_mjo_process(date, members)
    print(f'run_parallel_mjo_process: {status3}')
    #else:
    #    print('Task Error in : reader.combine_201_days_analysis_and_forecast_data(date, members)')
    #    #sys.exit()

    #time.sleep(60)  # Wait for 60 seconds
    #if status3:
    rmm_display = bokeh_display.MJODisplay('mogreps')
    rmm_display.bokeh_rmm_plot(date, members, title_prefix='MOGREPS')
    #else:
    #    print(f'Error: Not all jobs in mjo_utils.run_parallel_mjo_process() completed.')


def do_glosea(date):
    reader = glosea_process.GLOProcess('glosea')
    reader.retrieve_glosea_data(date)

    # All ensemble members
    members = [str('%03d' % mem) for mem in range(4)]
    print(members)

    mjo_proc = mjo_utils.MJOUtils('glosea')
    status3 = mjo_proc.run_parallel_mjo_process(date, members, parallel=False)
    print(f'run_parallel_mjo_process: {status3}')

    time.sleep(60)  # Wait for 60 seconds

    rmm_display = bokeh_display.MJODisplay('glosea')
    rmm_display.bokeh_rmm_plot(date, members, title_prefix='GLOSEA')

if __name__ == '__main__':
    #today = datetime.date.today()
    #yesterday = today - datetime.timedelta(days=1)
    #yesterday = datetime.datetime(2024, 1, 17)

    yesterday = read_date_from_command_line()

    # MOGREPS
    do_analysis(yesterday)
    #time.sleep(60)  # Wait for 60 seconds
    #do_analysis(yesterday)
    #time.sleep(60)  # Wait for 60 seconds


    # a second run to make sure all parallel jobs are completed
    #do_mogreps(yesterday)
    #time.sleep(60)  # Wait for 60 seconds
    #do_mogreps(yesterday)

    # GLOSEA
    #do_analysis(yesterday)
    #do_glosea(yesterday)
    #time.sleep(60)  # Wait for 60 seconds
    #do_glosea(yesterday)
    '''
    start_date = datetime.date(2024, 1, 31)
    end_date = datetime.date(2024, 2, 4)

    current_date = start_date
    while current_date <= end_date:
        # Glosea
        do_analysis(current_date)
        do_mogreps(current_date)
        do_mogreps(current_date)
        current_date += datetime.timedelta(days=1)
    '''


