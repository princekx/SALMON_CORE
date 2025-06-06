import configparser
import datetime
import multiprocessing
import concurrent
import os
import subprocess
import sys
import uuid
import iris
from iris.fileformats.pp import load_pairs_from_fields
import numpy as np
import logging
import warnings

# Set the global warning filter to ignore all warnings
warnings.simplefilter("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MOGProcess:
    def __init__(self, config_values_analysis, config_values):
        self.config_values_analysis = config_values_analysis
        self.config_values = config_values
        self.num_prev_days = 201
        self.parent_dir = '/home/users/prince.xavier/MJO/SALMON/MJO'

    def get_all_members(self, hr):
        if hr == 0:
            return [str('%02d' % mem) for mem in range(18)]
        elif hr == 6:
            return [str('%02d' % mem) for mem in range(18, 35)] + ['00']
        elif hr == 12:
            return [str('%02d' % mem) for mem in range(18)]
        elif hr == 18:
            return [str('%02d' % mem) for mem in range(18, 35)] + ['00']

    def retrieve_fc_data_parallel(self, date, hr, fc, digit2_mem):
        print('In retrieve_fc_data_parallel()')
        moosedir = os.path.join(self.config_values['mogreps_moose_dir'], f'{date.strftime("%Y%m")}.pp')
        digit3_mem = '035' if (hr == 18 and digit2_mem == '00') else str('%03d' % int(digit2_mem))

        remote_data_dir = os.path.join(self.config_values['mogreps_raw_dir'],
                                       date.strftime("%Y%m%d"), digit3_mem)
        if not os.path.exists(remote_data_dir):
            os.makedirs(remote_data_dir)

        print(f'Retrieving hr: {fc}')
        #self.retrieve_fc_data(date, hr, fc, digit2_mem, remote_data_dir, moosedir)
        fct = f'{fc:03d}'  # if fc != 0 else '003'

        # File names changed on moose on 25/09/2018
        filemoose = f'prods_op_mogreps-g_{date.strftime("%Y%m%d")}_{hr}_{digit2_mem}_{fct}.pp'
        outfile = f'englaa_pd{fct}.pp'

        local_dummy_query_dir = self.config_values['mogreps_dummy_queryfiles_dir']
        if not os.path.exists(local_dummy_query_dir):
            os.makedirs(local_dummy_query_dir)

        # Generate a unique query file
        local_query_file1 = os.path.join(local_dummy_query_dir,
                                         f'localquery_{uuid.uuid1()}')

        self.create_query_file(local_query_file1, filemoose, fct)
        outfile_path = os.path.join(remote_data_dir, outfile)

        if os.path.exists(outfile_path):
            if os.path.getsize(outfile_path) == 0:
                # delete the file
                print(os.path.getsize(outfile_path))
                print(f'Deleting empty file {outfile_path}')
                os.remove(outfile_path)

        if not os.path.exists(outfile_path):
            print('EXECCCC')
            command = f'/opt/moose-client-wrapper/bin/moo select --fill-gaps {local_query_file1} {moosedir} {os.path.join(remote_data_dir, outfile)}'
            logger.info('Executing command: %s', command)

            try:
                subprocess.run(command, shell=True, check=True)
                logger.info('Data retrieval successful.')
            except subprocess.CalledProcessError as e:
                logger.error('Error during data retrieval: %s', e)
            except Exception as e:
                logger.error('An unexpected error occurred: %s', e)
        else:
            print(f'{os.path.join(remote_data_dir, outfile)} exists. Skip...')



    def check_if_all_data_exist(self, date, hr, fc, digit2_mem):
        digit3_mem = str('%03d' % int(digit2_mem))
        remote_data_dir = os.path.join(self.config_values['mogreps_raw_dir'],
                                       date.strftime("%Y%m%d"), digit3_mem)
        fct = f'{fc:03d}'
        outfile = f'englaa_pd{fct}.pp'
        outfile_path = os.path.join(remote_data_dir, outfile)
        outfile_status = os.path.exists(outfile_path) and os.path.getsize(outfile_path) > 0
        return outfile_status

    def create_query_file(self, local_query_file1, filemoose, fct):
        query_file = self.config_values['mogreps_combined_queryfile']

        replacements = {'fctime': fct, 'filemoose': filemoose}
        with open(query_file) as query_infile, open(local_query_file1, 'w') as query_outfile:
            for line in query_infile:
                for src, target in replacements.items():
                    line = line.replace(src, target)
                query_outfile.write(line)

    def retrieve_mogreps_data(self, date, parallel=True):
        print('Retrieving data for date:', date)

        hr_list = [0, 12]
        fc_times = np.arange(0, 174, 24)

        # Create a list of tuples for all combinations of hr and mem
        tasks = [(date, hr, fc, digit2_mem) for hr in hr_list for fc in fc_times for digit2_mem in
                 self.get_all_members(hr)]

        #for task in tasks:
        #    self.retrieve_fc_data_parallel(*task)
        if parallel:
            # Use ThreadPoolExecutor to run tasks in parallel
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Submit tasks to the executor
                futures = [executor.submit(self.retrieve_fc_data_parallel, *task) for task in tasks]

                # Wait for all tasks to complete
                concurrent.futures.wait(futures)

            # Wait for all tasks to complete
            #print(concurrent.futures.as_completed(futures))

            file_present = all([self.check_if_all_data_exist(*task) for task in tasks ])

            return file_present
        else:
            for task in tasks:
                self.retrieve_fc_data_parallel(*task)



        # Check if all tasks are completed successfully
        #all_tasks_completed = all(future.done() and not future.cancelled() for future in futures)

        #print("All MOGREPS retrieval tasks completed.")
        #return all_tasks_completed


    def load_base_cube(self):
        #parent_dir = os.getcwd()
        base_cube = iris.load_cube(os.path.join(self.parent_dir, 'data', 'obsgrid_145x73.nc'))

        for coord_name in ['latitude', 'longitude']:
            base_cube.coord(coord_name).units = 'degrees_north' \
                if coord_name == 'latitude' else 'degrees_east'
            base_cube.coord(coord_name).coord_system = None

        return base_cube

    def regrid2obs(self, my_cubes):
        base_cube = self.load_base_cube()

        for coord_name in ['latitude', 'longitude']:
            my_cubes.coord(coord_name).units = 'degrees_north' \
                if coord_name == 'latitude' else 'degrees_east'
            my_cubes.coord(coord_name).coord_system = None
            if my_cubes.coord(coord_name).bounds is None:
                my_cubes.coord(coord_name).guess_bounds()

        reg_cube = my_cubes.regrid(base_cube, iris.analysis.Linear())
        return reg_cube

    def read_olr_correctly(self, data_files, varname, lbproc=0):
        if varname == 'olr':
            stash_code = 'm01s02i205'

        data_files.sort()
        cubes = []
        for data_file in data_files:
            filtered_fields = []
            assert os.path.exists(data_file), f'{data_file} does not exist.'

            for field in iris.fileformats.pp.load(data_file):
                if field.stash == stash_code and field.lbproc == lbproc:
                    filtered_fields.append(field)
            cube_field_pairs = load_pairs_from_fields(filtered_fields)
            for cube, field in cube_field_pairs:
                cube.attributes['lbproc'] = field.lbproc

            cubes.append(cube)

        # Equalise attributes
        iris.util.equalise_attributes(cubes)
        # Merge was failing because of some stupid cell_methods mismatch (Iris is evil!)
        cubes = iris.cube.CubeList(cubes).merge_cube()

        return self.regrid2obs(cubes)

    def read_winds_correctly(self, data_files, varname, pressure_level=None):
        data_files.sort()
        # print(data_files)
        cubes = []
        for data_file in data_files:
            # Some files have 3 hourly wind data which need to be averaged
            cube = iris.load_cube(data_file, varname)
            if pressure_level is not None:
                cube = cube.extract(iris.Constraint(pressure=pressure_level))
            if len(cube.shape) == 3:
                cube = cube.collapsed('time', iris.analysis.MEAN)
            if cube.coord('forecast_period').bounds is None:
                bounds = [[cube.coord('forecast_period').points[0] - 1., cube.coord('forecast_period').points[0] + 1.]]
                cube.coord('forecast_period').bounds = bounds
            if cube.coord('time').bounds is None:
                bounds = [[cube.coord('time').points[0] - 1., cube.coord('time').points[0] + 1.]]
                cube.coord('time').bounds = bounds
            cubes.append(cube)

        # Equalise attributes
        iris.util.equalise_attributes(cubes)
        # Merge was failing because of some stupid cell_methods mismatch (Iris is evil!)
        for cube in cubes:
            cube.cell_methods = ()

        cubes = iris.cube.CubeList(cubes).merge_cube()
        return self.regrid2obs(cubes)

    def load_analysis_cubes(self, date):
        olr_mean_analysis_201d_file, u850_mean_analysis_201d_file, u200_mean_analysis_201d_file = (
            os.path.join(self.config_values_analysis['analysis_mjo_processed_dir'], varname,
                         f'{varname}_mean_nrt_{date.strftime("%Y%m%d")}.nc')
            for varname in ['olr', 'u850', 'u200'])

        olr_analysis = iris.load_cube(olr_mean_analysis_201d_file)
        u850_analysis = iris.load_cube(u850_mean_analysis_201d_file)
        u200_analysis = iris.load_cube(u200_mean_analysis_201d_file)

        return olr_analysis, u850_analysis, u200_analysis

    def concat_analysis_fcast(self, analysis_cube, fcast_cube):
        nfcast_days, _, _ =  fcast_cube.shape
        time_unit = analysis_cube.coord('time').units
        ntime_analysis, nlat, nlon = analysis_cube.shape
        cat_cube = iris.cube.CubeList([analysis_cube[0].copy() for i in range(ntime_analysis + nfcast_days)])
        # redefine time coordinates
        for n, cube in enumerate(cat_cube):
            cube.add_aux_coord(iris.coords.AuxCoord(analysis_cube.coord('time').points[0]+n*24,
                                                    long_name='forecast_time', units=time_unit))

        cat_cube = cat_cube.merge_cube()
        cat_cube.data = np.concatenate((analysis_cube.data, fcast_cube.data))
        return cat_cube

    def process_member_combine_data(self, mem, date, fc_times, olr_analysis, u850_analysis, u200_analysis):
        mog_files = [os.path.join(self.config_values['mogreps_raw_dir'],
                                  date.strftime("%Y%m%d"), mem, f'englaa_pd{fct}.pp')
                     for fct in fc_times]
        mog_files.sort()

        print(f'Combining member: {mem}')

        for varname, analysis_cube in [('olr', olr_analysis),
                                       ('u850', u850_analysis),
                                       ('u200', u200_analysis)]:
            concated_dir = os.path.join(
                self.config_values['mogreps_mjo_processed_dir'], varname)

            if not os.path.exists(concated_dir):
                os.makedirs(concated_dir)

            concated_file = os.path.join(concated_dir,
                                         f'{varname}_concat_nrt_{date.strftime("%Y%m%d")}_{mem}.nc')

            if not os.path.exists(concated_file):
                if varname == 'olr':
                    cube = self.read_olr_correctly(mog_files, varname)
                    cat_cube = self.concat_analysis_fcast(olr_analysis, cube)

                elif varname == 'u850':
                    cube = self.read_winds_correctly(mog_files, 'x_wind', pressure_level=850)
                    cat_cube = self.concat_analysis_fcast(u850_analysis, cube)

                elif varname == 'u200':
                    cube = self.read_winds_correctly(mog_files, 'x_wind', pressure_level=200)
                    cat_cube = self.concat_analysis_fcast(u200_analysis, cube)

                iris.save(cat_cube, concated_file, netcdf_format='NETCDF4_CLASSIC')
                print(f'Written {concated_file}')
            else:
                print(f'{concated_file} exits. Skip.')

    def combine_201_days_analysis_and_forecast_data(self, date, members, parallel=True):
        print('Concat 201 days of analysis and forecast data.')
        olr_analysis, u850_analysis, u200_analysis = self.load_analysis_cubes(date)
        fc_times = [str('%03d' % fct) for fct in np.arange(24, 174, 24)]

        if parallel:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = {executor.submit(self.process_member_combine_data, mem,
                                           date, fc_times, olr_analysis,
                                           u850_analysis, u200_analysis): mem for mem in members}

            concurrent.futures.wait(futures)
        else:
            for mem in members:
                self.process_member_combine_data(mem,date, fc_times, olr_analysis,
                u850_analysis, u200_analysis)


        # Check if all tasks are completed successfully
        #all_tasks_completed = all(future.done() and not future.cancelled() for future in futures)

        #if all_tasks_completed:
        #    print("All mogreps.combine_201_days_analysis_and_forecast_data tasks completed successfully.")
        #else:
        #    print("Some tasks failed.")

        #return all_tasks_completed
