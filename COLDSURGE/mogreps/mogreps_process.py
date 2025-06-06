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
    def __init__(self, config_values):
        self.config_values = config_values
        # Navigate to the parent directory
        self.parent_dir = '/home/users/prince.xavier/MJO/SALMON/COLDSURGE'


    def get_all_members(self, hr):
        if hr == 12:
            return [str('%02d' % mem) for mem in range(18)]
        elif hr == 18:
            return [str('%02d' % mem) for mem in range(18, 35)] + ['00']
    def print_progress_bar(self, iteration, total):
        percentage = 100 * iteration / total
        progress_bar = f"Progress: [{percentage:.2f}%]"
        print(progress_bar, end="\r")

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

        # Generate a unique query file
        local_query_file1 = os.path.join(self.config_values['mogreps_dummy_queryfiles_dir'],
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

        hr_list = [12, 18]
        fc_times = np.arange(0, 174, 24)
        print(fc_times)

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

            #file_present = all([self.check_if_all_data_exist(*task) for task in tasks ])
            # Check if all tasks are completed
            all_tasks_completed = all(future.done() for future in futures)

            return all_tasks_completed
        else:
            for task in tasks:
                self.retrieve_fc_data_parallel(*task)


        # Check if all tasks are completed successfully
        #all_tasks_completed = all(future.done() and not future.cancelled() for future in futures)

        #print("All MOGREPS retrieval tasks completed.")
        #return all_tasks_completed


    def load_base_cube(self):
        parent_dir = os.getcwd()
        base_cube = iris.load_cube(os.path.join(parent_dir, 'data', 'obsgrid_145x73.nc'))

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
    def read_precip_correctly(self, data_files, varname, lbproc=0):
        cubes = []
        for data_file in data_files:
            # Some files have 3 hourly wind data which need to be averaged
            cube = iris.load_cube(data_file, varname)
            if len(cube.shape) == 3:
                cube = cube.collapsed('time', iris.analysis.MEAN)
            if cube.coord('forecast_period').bounds is None:
                bounds = [[cube.coord('forecast_period').points[0] - 1., cube.coord('forecast_period').points[0] + 1.]]
                cube.coord('forecast_period').bounds = bounds
            if cube.coord('time').bounds is None:
                bounds = [[cube.coord('time').points[0] - 1., cube.coord('time').points[0] + 1.]]
                cube.coord('time').bounds = bounds

            # Massaging the data for mergine
            for coord in ['forecast_reference_time', 'realization', 'time']:
                cube.remove_coord(coord) if cube.coords(coord) else None
            #cube.coord('forecast_period').points = cube.coord('forecast_period').bounds[:, 1]

            cubes.append(cube)

        # Correcting metadata for merging
        for i, cube in enumerate(cubes):
            # print(f"Cube {i} metadata:\n{cube.metadata}\n")
            cube.cell_methods = cubes[0].cell_methods
            if cube.coords("forecast_period"):
                new_fp = iris.coords.DimCoord(
                    cube.coord("forecast_period").points,
                    standard_name="forecast_period",
                    units=cube.coord("forecast_period").units
                )
                cube.replace_coord(new_fp)

        # Merge
        cubes = iris.cube.CubeList(cubes).merge_cube()
        return self.subset_seasia(cubes)

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
        return self.subset_seasia(cubes)

    def subset_seasia(self, cube):
        return cube.intersection(latitude=(-10, 25), longitude=(85, 145))

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
        return self.subset_seasia(cubes)

    def process_forecast_data(self, date, members):
        fc_times = [str('%03d' % fct) for fct in np.arange(0, 174, 24)]

        for varname in ['precip', 'u850', 'v850']:
            concated_dir = os.path.join(
                self.config_values['mogreps_cs_processed_dir'], varname )

            if not os.path.exists(concated_dir):
                os.makedirs(concated_dir)


            combined_allmember_file = os.path.join(concated_dir,
                                         f'{varname}_ColdSurge_24h_allMember_{date.strftime("%Y%m%d")}.nc')

            time_coord = iris.coords.DimCoord(
                points=[int(t) for t in fc_times],  standard_name='time',
                units=f'days since {date.strftime("%Y-%m-%d")}')

            if not os.path.exists(combined_allmember_file):
                cubes = []
                for mem in members:
                    # progress bar
                    self.print_progress_bar(int(mem), len(members))

                    mog_files = [os.path.join(self.config_values['mogreps_raw_dir'],
                                              date.strftime("%Y%m%d"), mem, f'englaa_pd{fct}.pp')
                                 for fct in fc_times]
                    mog_files.sort()
                    realiz_coord = iris.coords.DimCoord([int(mem)], standard_name='realization',
                                                        var_name='realization')
                    if varname == 'precip':
                        #cube = iris.load_cube(mog_files, )
                        cube = self.read_precip_correctly(mog_files, 'precipitation_amount')
                        cube.data[1:] -= cube.data[:-1]
                    elif varname == 'u850':
                        cube = self.read_winds_correctly(mog_files, 'x_wind', pressure_level=850)
                    elif varname == 'v850':
                        cube = self.read_winds_correctly(mog_files, 'y_wind', pressure_level=850)


                    # Massaging the data for mergine
                    for coord in ['forecast_reference_time', 'realization', 'time']:
                        cube.remove_coord(coord) if cube.coords(coord) else None
                    cube.add_aux_coord(realiz_coord)
                    #cube.coord('forecast_period').points = cube.coord('forecast_period').bounds[:, 1]
                    cubes.append(cube)


                cube = iris.cube.CubeList(cubes).merge_cube()
                print(cube)
                #cube.add_dim_coord(time_coord, 1)
                #print(cube)
                iris.save(cube, combined_allmember_file, netcdf_format='NETCDF4_CLASSIC')
                print(f'Written {combined_allmember_file}')
            else:
                print(f'{combined_allmember_file} exits. Skip.')
