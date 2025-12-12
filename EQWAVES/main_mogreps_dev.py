#!/usr/bin/env /data/apps/sss/environments/default-current/bin/python
import datetime
import sys
sys.path.append('/home/users/prince.xavier/MJO/Monitoring_new/EQWAVES')

from analysis import analysis_process
from mogreps import mogreps_process
from display import eqwaves_plot_bokeh_v2_test as eqwaves_plot_bokeh
import warnings
warnings.filterwarnings("ignore")


def collect_hour_argument():
    """
    Collects the hour argument from the command line.

    Returns:
        int: The hour value if valid, otherwise None.
    """
    if len(sys.argv) != 3:
        print("Usage: python main_mogreps.py <hour>")
        return None

    date_str, hour_input = sys.argv[1], sys.argv[2]

    try:
        hour = int(hour_input)
    except ValueError:
        print("Invalid hour value. Please enter an integer.")
        return None

    if hour not in [0, 6, 12, 18]:
        print("Invalid hour value. Please enter 0, 6, 12, or 18.")
        return None

    try:
        date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d')
        print(f"Successfully created datetime object: {date_obj}")
    except ValueError as e:
        print(f"Error: {e}. Please provide the date in YYYY-MM-DD format.")

    return date_obj, hour

def do_mogreps(date):
    """
    Retrieves and processes analysis and forecast data for the given date.

    Parameters:
        date (datetime.datetime): The date for which data is to be processed.

    Returns:
        None

    This function retrieves analysis data and MOGREPS forecast data for the specified date,
    processes the data by combining analysis and forecasts, computes wave data, divergence,
    and vorticity, and finally generates plots for forecast ensemble probability.

    The function follows these steps:
    1. Retrieve analysis data using the AnalysisProcess class.
    2. Process analysis cubes.
    3. Retrieve MOGREPS forecast data using the MOGProcess class.
    4. Process forecast data by combining analysis and forecasts.
    5. Compute wave data, divergence, and vorticity.
    6. Generate plots for forecast ensemble probability using EqWavesDisplay.

    Example:
    >>> import datetime
    >>> do_mogreps(datetime.datetime(2024, 4, 30, 12, 0))
    Data Processed.
    Waves computed.
    """
    # Retrieve analysis data
    reader = analysis_process.AnalysisProcess('analysis')
    status1 = reader.retrieve_analysis_data(date)
    status2 = reader.process_analysis_cubes(date)

    # Retrieve MOGREPS data
    reader = mogreps_process.MOGProcess('mogreps')
    # retrieve data
    futures1 = reader.retrieve_mogreps_forecast_data(date)

    # process data by combining analysis and forecasts
    reader.process_mogreps_forecast_data(date)
    print('Data Processed.')
    # compute and write wave data, divergence and vorticity
    reader.compute_waves_forecast_data(date)
    print('Waves computed.')

    # Generate plots for forecast ensemble probability
    reader = eqwaves_plot_bokeh.EqWavesDisplay('mogreps')
    reader.bokeh_plot_forecast_ensemble_probability_multiwave(date)



if __name__ == '__main__':
    #today = datetime.datetime.today()
    #yesterday = today - datetime.timedelta(days=1)

    #yesterday, hour = collect_hour_argument()
    #print(yesterday)
    #do_mogreps(yesterday)


    d1 = datetime.date(2024, 10, 12)  # start date
    d2 = datetime.date(2024, 10, 27)  # end date

    # Generate a list of dates between d1 and d2 (inclusive)
    date_list = [d1 + datetime.timedelta(days=i) for i in range((d2 - d1).days + 1)]
    print(date_list)
    #yesterday, hour = collect_hour_argument()
    for date in date_list:
        for hour in ['00', '06', '12', '18']:
            print(date, hour)
            yesterday = datetime.datetime(date.year,
                                          date.month,
                                          date.day, int(hour), 0)
            print(yesterday)
            do_mogreps(yesterday)