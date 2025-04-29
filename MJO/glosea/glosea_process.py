import configparser
import glob
import os, sys
import numpy as np
import pandas as pd
import datetime
import iris
import logging
import warnings
import uuid

# Set the global warning filter to ignore all warnings
warnings.simplefilter("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GLOProcess:
    def __init__(self, config_values_analysis, config_values):
        self.config_values_analysis = config_values_analysis
        self.config_values = config_values
        self.num_prev_days = 201



        self.nforecasts = 30
        self.parent_dir = '/home/users/prince.xavier/MJO/SALMON/MJO'

    def command_helper(self, command):
        print(command)
        # Execute command
        status = os.system(command)
        print('Command Execution status: %s' % str(status))
        return status

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

    def load_analysis_cubes(self, date, varname='olr'):
        var_mean_analysis_201d_file = os.path.join(self.config_values_analysis['analysis_mjo_processed_dir'], varname,
                         f'{varname}_mean_nrt_{date.strftime("%Y%m%d")}.nc')
        var_analysis = iris.load_cube(var_mean_analysis_201d_file)

        return self.regrid2obs(var_analysis)
    def retrieve_glosea_forecast_data(self, date, prod='prodf',
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
        df = pd.read_csv('~sfcpp/bin/MassGet/archived_data/%s.csv' % prod)
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
                print(command)

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

        else:
            print('Data not found in catelogue. Not retrieving.')

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

    def combine_data(self, date, prod='prodf', mem_name_start='000'):
        '''
        Data is by default retrieved as 10-day chunks which needs to be combined
        to 60 days time series
        :param date:
        :type date:
        :param prod:
        :type prod:
        :param fcast_in_dir_new:
        :type fcast_in_dir_new:
        :param gs_out_dir:
        :type gs_out_dir:
        :return:
        :rtype:
        '''
        # Combine 10-day chunks in to combined files
        suite_dic = {'prodm': 'monthly', 'prodf': 'forecast'}
        date_label = date.strftime("%Y%m%d")

        gs_out_dir = os.path.join(self.config_values['glosea_mjo_processed_dir'], date_label)
        if not os.path.exists(gs_out_dir):
            os.makedirs(gs_out_dir)

        # get all members
        command = '%s/%s_*_%s_*.pp' % (os.path.join(self.config_values['glosea_raw_dir'],
                                                    suite_dic[prod], date_label), prod, date_label)
        print(command)
        files = glob.glob(command)
        members = list(set([file.split('_')[-1].split('.')[0] for file in files]))
        members.sort()

        print('Total number of files: %s' % len(files))
        print('Ensemble members:')
        print(members)

        varnames = ['olr', 'u850', 'u200']


        for varname in varnames:
            concated_dir = os.path.join(
                self.config_values['glosea_mjo_processed_dir'], varname)

            if not os.path.exists(concated_dir):
                os.makedirs(concated_dir)

            # read analysis data
            analysis_cube = self.load_analysis_cubes(date, varname=varname)

            for m, mem in enumerate(members):
                xmem = str('%03d' % (int(mem_name_start)+m))

                concated_file = os.path.join(concated_dir,
                                             f'{varname}_concat_nrt_{date.strftime("%Y%m%d")}_{xmem}.nc')

                print(concated_file)
                if not os.path.exists(concated_file):
                    command = '%s/%s_*_%s_*_%s.pp' % (
                        os.path.join(self.config_values['glosea_raw_dir'],
                                     suite_dic[prod], date_label), prod, date_label, mem)
                    member_files = glob.glob(command)
                    member_files.sort()
                    if varname == 'olr':
                        print('reading OLR...')
                        fcast_cube = iris.load_cube(member_files, 'toa_outgoing_longwave_flux')

                    if varname == 'u850':
                        print('reading U850...')
                        fcast_cube = iris.load_cube(member_files, 'x_wind')
                        fcast_cube = fcast_cube.extract(iris.Constraint(pressure=850))

                    if varname == 'u200':
                        print('reading U200...')
                        fcast_cube = iris.load_cube(member_files, 'x_wind')
                        fcast_cube = fcast_cube.extract(iris.Constraint(pressure=200))

                    # Only take first 60 time points
                    fcast_cube = self.regrid2obs(fcast_cube[:self.nforecasts])

                    cat_cube = self.concat_analysis_fcast(analysis_cube, fcast_cube)
                    iris.save(cat_cube, concated_file, netcdf_format='NETCDF4_CLASSIC')
                    print(f'Written {concated_file}')
                else:
                    print('%s exists. Skipping conversion.' % concated_file)

    def retrieve_glosea_data(self, date):

        self.retrieve_glosea_forecast_data(date, prod='prodf')
        self.retrieve_glosea_forecast_data(date, prod='prodm')

        # mem_name_start indicates the label of the first member
        self.combine_data(date, prod='prodf', mem_name_start='000')
        self.combine_data(date, prod='prodm', mem_name_start='002')

