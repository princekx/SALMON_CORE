import configparser
import datetime
import os
import subprocess
import sys
import uuid
import concurrent.futures
import iris
import numpy as np

# Set up logging
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnalysisProcess:
    def __init__(self, config_values):
        self.config_values = config_values
        self.parent_dir = '/home/users/prince.xavier/MJO/SALMON/EQWAVES'
        self.ntimes_total = 360
        self.ntimes_analysis = 332
        self.ntimes_forecast = 28
        self.latmax = 24
        self.ref_grid = self.create_cube().intersection(latitude=(-self.latmax, self.latmax))


    def create_cube(self, latitudes=(-90.5, 90.5), longitudes=(-0.5, 359.5), spacing=1):
        """
        Create empty cube with regular lat-long grid.

        Args:
            latitudes: Range of latitudes in degree
            longitudes: Range of longitudes in degree
            spacing: lat-long grid spacing in degree

        Returns:
            Empty cube with specified grid spacing, covering specifed
            latitude and longitude intervals.
        """
        # TODO most used coord system?
        cs = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)

        # lat coord
        lower_bound = latitudes[0] + spacing / 2.
        upper_bound = latitudes[1] - spacing / 2.
        number_of_points = int((upper_bound - lower_bound) / spacing) + 1
        lat_coord = iris.coords.DimCoord(np.linspace(lower_bound, upper_bound, number_of_points),
                                         standard_name='latitude',
                                         units='degrees',
                                         coord_system=cs)
        lat_coord.guess_bounds()

        # long coord
        lower_bound = longitudes[0] + spacing / 2.
        upper_bound = longitudes[1] - spacing / 2.
        number_of_points = int((upper_bound - lower_bound) / spacing) + 1
        lon_coord = iris.coords.DimCoord(np.linspace(lower_bound, upper_bound, number_of_points),
                                         standard_name='longitude',
                                         units='degrees',
                                         coord_system=cs)
        lon_coord.guess_bounds()

        # data
        data = np.zeros((len(lat_coord.points), len(lon_coord.points)))

        # build cube
        cube = iris.cube.Cube(
            data,
            long_name='zeros',
            units='',
            attributes=None,
            dim_coords_and_dims=[(lat_coord, 0), (lon_coord, 1)]
        )

        return cube
    def create_query_file(self, local_query_file1, filemoose, fct):
        query_file = self.config_values['analysis_combined_queryfile']

        replacements = {'fctime': fct, 'filemoose': filemoose}
        with open(query_file) as query_infile, open(local_query_file1, 'w') as query_outfile:
            for line in query_infile:
                for src, target in replacements.items():
                    line = line.replace(src, target)
                query_outfile.write(line)

    def retrieve_data(self, date):
        str_year = str(date.year)
        str_month = date.strftime('%m')
        str_day = date.strftime('%d')
        str_hr = date.strftime('%H')

        date_label = f"{str_year}{str_month}{str_day}_{str_hr}"
        logger.info(f'Doing date: {date_label}')

        fcx = 0  # analysis data
        fct = str('%03d' % fcx)

        moose_dir = os.path.join(self.config_values['analysis_moose_dir'], f'{str_year}.pp')
        remote_data_dir = os.path.join(self.config_values['analysis_raw_dir'], date.strftime("%Y%m%d"))

        if not os.path.exists(remote_data_dir):
            os.makedirs(remote_data_dir)

        file_moose = f'prods_op_gl-mn_{date.strftime("%Y%m%d")}_{str_hr}_{fct}.pp'
        if date.date() >= datetime.date(2018, 9, 25):
            file_moose = f'prods_op_gl-mn_{date.strftime("%Y%m%d")}_{str_hr}_{fct}.pp'

        outfile = f'qg{str_hr}T{fct}.pp'

        local_query_file1 = os.path.join(self.config_values['analysis_dummy_queryfiles_dir'],
                                         f'eqw_localquery_{uuid.uuid1()}')
        self.create_query_file(local_query_file1, file_moose, fct)

        outfile_path = os.path.join(remote_data_dir, outfile)

        if os.path.exists(outfile_path):
            if os.path.getsize(outfile_path) == 0:
                # delete the file
                print(os.path.getsize(outfile_path))
                print(f'Deleting empty file {outfile_path}')
                os.remove(outfile_path)
                print('HERE')
                sys.exit()

        if not os.path.exists(outfile_path):
            command = f'/opt/moose-client-wrapper/bin/moo select --fill-gaps {local_query_file1} {moose_dir} {outfile_path}'
            logger.info(command)
            subprocess.run(command, shell=True, check=True)
        else:
            logger.info(f'{outfile_path} exists. Skipping retrieval.')

    def remove_um_version(self, cube, field, filename):
        """Callback to remove the 'um_version' attribute."""
        cube.attributes.pop('um_version', None)

    def process_analysis_cubes(self, date):
        analysis_dates = sorted([date - datetime.timedelta(hours=i * 6)
                                 for i in range(self.ntimes_analysis)])

        str_year = str(date.year)
        str_month = date.strftime('%m')
        str_day = date.strftime('%d')
        str_hr = date.strftime('%H')

        date_label = f"{str_year}{str_month}{str_day}_{str_hr}"
        logger.info(f'Doing date: {date_label}')

        analysis_data_files = [os.path.join(self.config_values['analysis_raw_dir'],
                                       an_date.strftime("%Y%m%d"), f"qg{an_date.strftime('%H')}T000.pp")
                               for an_date in analysis_dates]
        print(f'Reading {len(analysis_data_files)} analysis files')




        for var in ['x_wind', 'y_wind', 'geopotential_height', 'precipitation_amount']:
        #for var in ['x_wind', 'y_wind', 'geopotential_height']:
            outfile_name = os.path.join(self.config_values['analysis_eqwaves_processed_dir'],
                                        f'{var}_analysis_{date_label}.nc')
            print(outfile_name)

            if not os.path.exists(outfile_name):
                print(f'Generating {outfile_name}')
                if var == 'precipitation_amount':
                    cube = iris.load_cube(analysis_data_files, var,
                                          callback=self.remove_um_version).regrid(self.ref_grid, iris.analysis.Linear())
                else:
                    cube = iris.load_cube(analysis_data_files, iris.Constraint(pressure=[200, 850]) & var,
                                          callback=self.remove_um_version).regrid(self.ref_grid,
                                                                                  iris.analysis.Linear())
                print(cube)

                iris.save(cube, outfile_name)

            else:
                print(f'{outfile_name} exists. Skip.')

    def retrieve_analysis_data(self, date):
        analysis_dates = sorted([date - datetime.timedelta(hours=i * 6)
                               for i in range(self.ntimes_analysis)])
        #print(analysis_dates)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit tasks to the executor
            futures = [executor.submit(self.retrieve_data, analysis_date) for analysis_date in analysis_dates]

            # Wait for all tasks to complete
            concurrent.futures.wait(futures)

        # Wait for all tasks to complete
        all_tasks_completed = all(future.done() for future in futures)
        return all_tasks_completed