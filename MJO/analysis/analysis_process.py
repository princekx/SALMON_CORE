import concurrent
import configparser
import datetime
import multiprocessing
import os
import subprocess
import sys
import uuid
import iris
import numpy as np
import logging
import warnings

# Set the global warning filter to ignore all warnings
warnings.simplefilter("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalysisProcess:

    def __init__(self, config_values):
        self.config_values = config_values
        self.num_prev_days = 201
        self.var_names = ['toa_outgoing_longwave_flux', 'precipitation_amount']
        self.parent_dir = '/home/users/prince.xavier/MJO/SALMON/MJO'
        print(self.config_values)

    def regrid2obs(self, my_cubes):

        base_cube = iris.load_cube(os.path.join(self.parent_dir, 'data', 'obsgrid_145x73.nc'))

        base_cube.coord('latitude').units = my_cubes.coord('latitude').units = 'degrees_north'
        base_cube.coord('longitude').units = my_cubes.coord('longitude').units = 'degrees_east'

        base_cube.coord('latitude').coord_system = None
        base_cube.coord('longitude').coord_system = None
        my_cubes.coord('latitude').coord_system = None
        my_cubes.coord('longitude').coord_system = None

        # For lat/lon regriding, make sure coordinates have bounds
        if my_cubes.coord('longitude').bounds is None:
            my_cubes.coord('longitude').guess_bounds()
        if my_cubes.coord('latitude').bounds is None:
            my_cubes.coord('latitude').guess_bounds()
        if base_cube.coord('longitude').bounds is None:
            base_cube.coord('longitude').guess_bounds()
        if base_cube.coord('latitude').bounds is None:
            base_cube.coord('latitude').guess_bounds()

        reg_cube = my_cubes.regrid(base_cube, iris.analysis.Linear())
        return reg_cube

    def load_and_process_cube(self, files, constraint, time_coord_name):
        cube = iris.load_cube(files, constraint)
        if len(cube.coord('forecast_period').points) > 1:
            cube = cube.collapsed(time_coord_name, iris.analysis.MEAN)
        return cube

    def average_and_regrid(self, cube_00, cube_12):
        cube_00.data += cube_12.data
        cube_00 /= 2.
        return self.regrid2obs(cube_00)

    def save_cube(self, cube, file_path):
        iris.save(cube, file_path, netcdf_format='NETCDF4_CLASSIC')
        print(f'Written {file_path}')

    def process_variable(self, varname, files_00, files_12, save_path,
                         pressure_constraint, time_coord_name):

        print(f'Processing {varname}...')
        cube_00 = self.load_and_process_cube(files_00, pressure_constraint, time_coord_name)
        cube_12 = self.load_and_process_cube(files_12, pressure_constraint, time_coord_name)
        averaged_cube = self.average_and_regrid(cube_00, cube_12)
        self.save_cube(averaged_cube, save_path)

    def check_retrieve_201_prev_days(self, start_date, parallel=True):

        missing_dates = [xdate for xdate in (start_date - datetime.timedelta(days=i)
                                           for i in range(self.num_prev_days))]
        print(f'Missing dates: {len(missing_dates)}')

        if parallel:
            # Create a pool of worker processes
            num_processes = multiprocessing.cpu_count() # Number of available CPU cores
            pool = multiprocessing.Pool(processes=num_processes)

            # Use map_async instead of map to get AsyncResult objects
            async_results = pool.map_async(self.retrieve_analysis_data, missing_dates)

            # Close the pool of worker processes
            pool.close()

            # Wait for all tasks to complete and get the results
            results = async_results.get()

            # Check the status of each task
            all_jobs_completed = all(result is None for result in results)
            if all_jobs_completed:
                print("All jobs completed successfully.")
                return 0
            else:
                print("Some jobs failed.")
                return 999
        else:
            print('Processing serial...')
            for date in missing_dates:
                self.retrieve_analysis_data(date)


    def retrieve_analysis_data(self, date):
        print('Retrieving data for date:', date)

        moosedir = os.path.join(self.config_values['analysis_moose_dir'], f'{str(date.year)}.pp')

        fc_times = [0] # just the analysis data
        hr_list = ['00', '12']

        for hr in hr_list:
            remote_data_dir = os.path.join(self.config_values['analysis_raw_dir'], date.strftime("%Y%m%d"))
            if not os.path.exists(remote_data_dir):
                os.makedirs(remote_data_dir)

            for fc in fc_times:
                print(f'Retrieving hr: {fc}')
                self.retrieve_fc_data(date, hr, fc, remote_data_dir, moosedir)

    def retrieve_fc_data(self, date, hr, fc, remote_data_dir, moosedir):
        # analysis is retrieved as the 003 hr data from the forecasts
        # this code can be used to extract forecast data as well.


        #fct = f'{fc:03d}' if fc != 0 else '003'
        fct = '003'

        # File names changed on moose on 25/09/2018
        filemoose = f'prods_op_gl-mn_{date.strftime("%Y%m%d")}_{hr}_{fct}.pp'
        if date >= datetime.datetime(2018, 9, 25):
            filemoose = f'prods_op_gl-mn_{date.strftime("%Y%m%d")}_{hr}_{fct}.pp'

        outfile = f'qg{hr}T{fct}.pp'

        # file on moose
        file_moose = os.path.join(moosedir, filemoose)

        # Generate a unique query file
        local_query_file1 = os.path.join(self.config_values['analysis_dummy_queryfiles_dir'],
                                         f'localquery_{uuid.uuid1()}')
        self.create_query_file(local_query_file1, filemoose, fct)

        if not self.check_retrieval_complete(outfile, remote_data_dir):
            print('HERE')
            self.retrieve_missing_data(local_query_file1, moosedir, outfile, remote_data_dir)
        else:
            print(f'{os.path.join(remote_data_dir, outfile)} exists. Skip...')

    def create_query_file(self, local_query_file1, filemoose, fct):
        query_file = self.config_values['analysis_combined_queryfile']

        replacements = {'fctime': fct, 'filemoose': filemoose}
        with open(query_file) as query_infile, open(local_query_file1, 'w') as query_outfile:
            for line in query_infile:
                for src, target in replacements.items():
                    line = line.replace(src, target)
                query_outfile.write(line)

    def check_retrieval_complete(self, outfile, remote_data_dir):
        outfile_path = os.path.join(remote_data_dir, outfile)
        return os.path.exists(outfile_path) and os.path.getsize(outfile_path) > 0

    def retrieve_missing_data(self, local_query_file1, moosedir, outfile, remote_data_dir):

        outfile_path = os.path.join(remote_data_dir, outfile)
        # delete the file
        print(f'Deleting empty file if exists : {outfile_path}')
        if os.path.exists(outfile_path):
            os.remove(outfile_path)

        command = f'/opt/moose-client-wrapper/bin/moo select --fill-gaps {local_query_file1} {moosedir} {outfile_path}'
        logger.info('Executing command: %s', command)

        try:
            subprocess.run(command, shell=True, check=True)
            logger.info('Data retrieval successful.')
        except subprocess.CalledProcessError as e:
            logger.error('Error during data retrieval: %s', e)# Replace the fctime and filemoose in query file

    def combine_201_days_analysis_data(self, date, parallel=True):

        past_analysis_dates = [date for date in (date - datetime.timedelta(days=i)
                                             for i in range(self.num_prev_days))]
        past_analysis_dates.sort()
        fc = 3
        fct = f'{fc:03d}'
        past_analysis_files_00 = [os.path.join(self.config_values['analysis_raw_dir'],
                                               past_analysis_date.strftime("%Y%m%d"), f'qg00T{fct}.pp')
                                  for past_analysis_date in past_analysis_dates]
        past_analysis_files_12 = [os.path.join(self.config_values['analysis_raw_dir'],
                                               past_analysis_date.strftime("%Y%m%d"), f'qg12T{fct}.pp')
                                  for past_analysis_date in past_analysis_dates]

        assert len(past_analysis_files_00) == self.num_prev_days, 'Assert error! Missing analysis files!'
        assert len(past_analysis_files_12) == self.num_prev_days, 'Assert error! Missing analysis files!'

        # Define variable-specific information
        variable_info = {
            'olr': {'constraint': iris.Constraint(name='toa_outgoing_longwave_flux')},
            'u850': {'constraint': iris.Constraint(name='x_wind', pressure=850)},
            'u200': {'constraint': iris.Constraint(name='x_wind', pressure=200)},
        }
        variable_time_coord = {
            'olr': 'forecast_period',
            'u850': 'time',
            'u200': 'time',
        }
        
        print('Combining 201 days of analysis as parallel tasks.')
        tasks = []
        for varname in ['olr', 'u850', 'u200']:
            file_write_dir = os.path.join(
                self.config_values['analysis_mjo_processed_dir'], varname)

            if not os.path.exists(file_write_dir):
                os.makedirs(file_write_dir)

            mean_analysis_201d_file = os.path.join(file_write_dir,
                f'{varname}_mean_nrt_{date.strftime("%Y%m%d")}.nc')
            if not os.path.exists(mean_analysis_201d_file):
                info = variable_info[varname]
                coord_time_name = variable_time_coord[varname]
                tasks.append((varname, past_analysis_files_00, past_analysis_files_12,
                              mean_analysis_201d_file, info['constraint'], coord_time_name))

        if parallel:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                # Using list comprehension to collect results
                results = [executor.submit(self.process_variable, *task) for task in tasks]

            # Check the status of each task
            all_tasks_completed = all(result.result() is None for result in results)

            if all_tasks_completed:
                print("All tasks completed successfully.")
            else:
                print("Some tasks failed.")

            return all_tasks_completed
        else:
            for task in tasks:
                self.process_variable(*task)