import configparser
import datetime
import os
import subprocess
import sys
import uuid
import concurrent.futures
import numpy as np
import logging
import iris.coord_systems
import iris.fileformats
import multiprocessing
from multiprocessing import Pool, Manager
import gc  # Garbage collection

from . import calculus

import warnings

# Set the global warning filter to ignore all warnings
warnings.simplefilter("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MOGProcess:

    def __init__(self, config_values_analysis, config_values):
        """
        Initializes the MOGProcess class with configuration values.

        Args:
        model (str): The model section in the configuration file.
        """
        self.config_values_analysis = config_values_analysis
        self.config_values = config_values

        # Navigate to the parent directory
        self.parent_dir = '/home/users/prince.xavier/MJO/SALMON/EQWAVES'

        self.ntimes_total = 360
        self.ntimes_analysis = 332
        self.ntimes_forecast = 28
        self.latmax = 24
        self.fc_times = np.arange(6, 174, 6)  # 6 hourly data
        self.ref_grid = self.create_cube().intersection(latitude=(-self.latmax, self.latmax))

        # For wave computations
        # define some physical parameters
        # ----------------------------------
        self.g = 9.8
        self.beta = 2.3e-11
        self.radea = 6.371e6
        self.spd = 86400.
        self.ww = 2 * np.pi / self.spd

        # Define some parameters spefic to the methodology
        self.kmin = 2  # minimum zonal wavenumber
        self.kmax = 40  # maximum zonal wavenumber
        self.pmin = 2.0  # minimum period (in days)
        self.pmax = 30.0  # maximum period (in days)
        self.y0 = 6.0  # meridional trapping scale (degrees)
        self.wave_names = np.array(['Kelvin', 'WMRG', 'R1', 'R2'])  # List of wave types to output

        self.y0real = 2 * np.pi * self.radea * self.y0 / 360.0  # convert trapping scale to metres
        self.ce = 2 * self.y0real ** 2 * self.beta
        self.g_on_c = self.g / self.ce
        self.c_on_g = self.ce / self.g

        self.R = 6378388.  # Radius of the earth
        self.deg2rad = 0.0174533  # pi/180.

    def print_progress_bar(self, iteration, total):
        percentage = 100 * iteration / total
        progress_bar = f"Progress: [{percentage:.2f}%]"
        print(progress_bar, end="\r")

    def generate_members(self, date):
        str_hr = date.strftime('%H')
        members_tuple = {'00': ['00'] + [f'{fc:02}' for fc in range(1, 18)],
                         '06': ['00'] + [f'{fc:02}' for fc in range(18, 35)],
                         '12': ['00'] + [f'{fc:02}' for fc in range(1, 18)],
                         '18': ['00'] + [f'{fc:02}' for fc in range(18, 35)]}
        members = members_tuple[str_hr]
        mem_labels = [f'{fc:03}' for fc in range(0, 18)]
        return members, mem_labels

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

    def concat_analysis_forecast(self, date, analysis_cube, forecast_cube):

        print(analysis_cube.shape, forecast_cube.shape)
        concatenated_array = np.concatenate((analysis_cube.data, forecast_cube.data), axis=0)

        analysis_datetime_list = sorted([date + datetime.timedelta(hours=(i + 1) * 6)
                                         for i in range(-self.ntimes_analysis, 0)])

        forecast_datetime_list = sorted([date + datetime.timedelta(hours=(i + 1) * 6)
                                         for i in range(0, self.ntimes_forecast)])

        all_dates = analysis_datetime_list + forecast_datetime_list

        # Convert datetime values to Iris-compatible units
        time_units = 'hours since 1970-01-01 00:00:00'
        time_values = iris.util.cf_units.date2num(all_dates, time_units, calendar='gregorian')

        # Create time coordinate
        time_coord = iris.coords.DimCoord(time_values, standard_name='time', units=time_units)

        # build cube
        dim_coords_and_dims = [(time_coord, 0)]
        print(len(dim_coords_and_dims))

        for coord in analysis_cube.dim_coords[1:]:
            dim_coords_and_dims.append((coord, analysis_cube.coord_dims(coord)[0]))

        print(concatenated_array.shape)

        concat_cube = iris.cube.Cube(
            concatenated_array,
            long_name=forecast_cube.long_name,
            units=forecast_cube.units,
            attributes=None,
            dim_coords_and_dims=dim_coords_and_dims
        )

        return concat_cube

    def process_mogreps_forecast_data_member(self, date, mem_label):

        str_hr = date.strftime('%H')

        date_label = date.strftime('%Y%m%d_%H')
        # read forecast data
        remote_data_dir = os.path.join(self.config_values['mogreps_raw_dir'],
                                       date.strftime("%Y%m%d"), str_hr, mem_label)

        outfile_dir = os.path.join(self.config_values['mogreps_eqwaves_processed_dir'], date_label)

        if not os.path.exists(outfile_dir):
            try:
                # Attempt to create the directory
                os.makedirs(outfile_dir)
            except FileExistsError:
                # If the directory already exists, continue execution without attempting to create it again
                pass

        # Generate a realization coordinate
        realiz_coord = iris.coords.DimCoord([int(mem_label)], standard_name='realization',
                                            var_name='realization')

        forecast_files = [os.path.join(remote_data_dir, f'qg{str_hr}T{fct:03d}.pp') for fct in self.fc_times]
        forecast_files = [file for file in forecast_files if os.path.exists(file)]

        for var in ['x_wind', 'y_wind', 'geopotential_height', 'precipitation_amount']:
            outfile_name = os.path.join(outfile_dir, f'{var}_combined_{date_label}Z_{mem_label}.nc')
            print('outfile_name', outfile_name)
            if not os.path.exists(outfile_name):
                # Read forecast data
                # Read the analysis data here
                analysis_combined_file = os.path.join(self.config_values_analysis['analysis_eqwaves_processed_dir'],
                                                      f'{var}_analysis_{date_label}.nc')
                print(f'analysis_combined_file: {analysis_combined_file}')

                analysis_cube = iris.load_cube(analysis_combined_file)
                print(forecast_files)
                forecast_cube = self.read_forecasts(date, forecast_files, var)

                if var == 'precipitation_amount':
                    # 6 hourly values from accumulations.
                    forecast_cube.data[1:] -= forecast_cube.data[:-1]

                    # convert units
                    analysis_cube.data *= 3600.

                # regrid to the reference grid
                forecast_cube = forecast_cube.regrid(self.ref_grid, iris.analysis.Linear())

                # Combining analysis and forecasts
                combined_cube = self.concat_analysis_forecast(date, analysis_cube, forecast_cube)

                # Check if 'realization' coordinate exists
                if combined_cube.coords('realization'):
                    combined_cube.remove_coord('realization')

                # Add 'realization' coordinate as an auxiliary coordinate
                combined_cube.add_aux_coord(realiz_coord)

                # Write data out

                iris.save(combined_cube, outfile_name)
                print(f'Written {outfile_name}')
            else:
                print(f'{outfile_name} exists. Skip')

    def make_mergable(self, cubes, date):
        for i, cube in enumerate(cubes):
            cube.cell_methods = ()
            if cube.coords("forecast_period"):  # If missing
                cube.remove_coord("forecast_period")
            forecast_period_coord = iris.coords.AuxCoord(
                self.fc_times[i], standard_name="forecast_period",
                units=f"hours since {date.strftime('%Y-%m-%d %H:00:00')}")
            cube.add_aux_coord(forecast_period_coord)

            if cube.coords("time"):  # If missing
                cube.remove_coord("time")

            forecast_time_coord = iris.coords.AuxCoord(
                self.fc_times[i], standard_name="time",
                units=f"hours since {date.strftime('%Y-%m-%d %H:00:00')}")
            cube.add_aux_coord(forecast_time_coord)

            if cube.coords("time"):
                print(f"Cube {i} forecast_period: {cube.coord('time').units}")
            else:
                print(f"Cube {i} has no time coordinate")
        return iris.cube.CubeList(cubes).merge_cube()

    def read_forecasts(self, date, forecast_files, var):
        forecast_cubes = []
        for forecast_file in forecast_files:
            forecast_cube = iris.load_cube(forecast_file, var)

            if var == 'precipitation_amount':
                if len(forecast_cube.shape) == 3:
                    forecast_cube = forecast_cube.collapsed('time', iris.analysis.MEAN)
            else:
                if len(forecast_cube.shape) == 4:
                    forecast_cube = forecast_cube.collapsed('time', iris.analysis.MEAN)

            forecast_cubes.append(forecast_cube)

        return self.make_mergable(forecast_cubes, date)

    def create_query_file(self, local_query_file1, filemoose, fct):
        """
        Creates a query file based on template and replacements.

        Args:
            local_query_file1 (str): Local query file path.
            filemoose (str): File name for the query.
            fct (str): Forecast time.

        """
        query_file = self.config_values['mogreps_combined_queryfile']

        replacements = {'fctime': fct, 'filemoose': filemoose}
        with open(query_file) as query_infile, open(local_query_file1, 'w') as query_outfile:
            for line in query_infile:
                for src, target in replacements.items():
                    line = line.replace(src, target)
                query_outfile.write(line)

    def retrieve_data_member(self, date, mem, mem_label, fc):
        """
        Retrieves data for a specific date, member, and forecast time.

        Args:
            date (datetime): Date object.
            mem (str): Ensemble member.
            mem_label (str): Label for the ensemble member.
            fc (int): Forecast time.

        """
        str_year = str(date.year)
        str_month = date.strftime('%m')
        str_day = date.strftime('%d')
        str_hr = date.strftime('%H')

        date_label = f"{str_year}{str_month}{str_day}_{str_hr}"
        logger.info(f'Doing date: {date_label}, Member: {mem_label}, Forecast: {fc}')
        print(f'Retrieving date: {date_label}, Member: {mem_label}, Forecast: {fc}')  # Add this line

        moose_dir = os.path.join(self.config_values['mogreps_moose_dir'], f"{date.strftime('%Y%m')}.pp")
        fcx = 3 if fc == 0 else fc
        fct = f"{fcx:03d}"

        remote_data_dir = os.path.join(self.config_values['mogreps_raw_dir'],
                                       date.strftime("%Y%m%d"), str_hr, mem_label)

        if not os.path.exists(remote_data_dir):
            os.makedirs(remote_data_dir)

        outfile = f'qg{str_hr}T{fct}.pp'
        outfile_path = os.path.join(remote_data_dir, outfile)

        outfile_status = os.path.exists(outfile_path)

        if outfile_status:
            print(outfile_status)
            if os.path.getsize(outfile_path) == 0:
                # delete the file
                print(os.path.getsize(outfile_path))
                print(f'Deleting empty file {outfile_path}')
                os.remove(outfile_path)

        if not outfile_status:
            self.run_retrieval(date, str_hr, mem, fct, moose_dir, outfile_path)
        else:
            logger.info(f'{outfile_path} exists. Skipping retrieval.')

        # except subprocess.CalledProcessError as e:
        #    logger.error(f'{file_moose} not returned. Check file on moose. Error: {e}')
        #    sys.exit()

    def run_retrieval(self, date, str_hr, mem, fct, moose_dir, outfile_path):
        # the retrieval call.
        # now retrieve again
        file_moose = f'prods_op_mogreps-g_{date.strftime("%Y%m%d")}_{str_hr}_{mem}_{fct}.pp'
        local_query_file1 = os.path.join(self.config_values['mogreps_dummy_queryfiles_dir'],
                                         f'eqw_localquery_{uuid.uuid1()}')
        self.create_query_file(local_query_file1, file_moose, fct)
        command = f'/opt/moose-client-wrapper/bin/moo select --fill-gaps {local_query_file1} {moose_dir} {outfile_path}'
        logger.info(command)
        subprocess.run(command, shell=True, check=True)

    def retrieve_mogreps_forecast_data(self, date, parallel=True):
        """
        Retrieves MOGREPS forecast data for a given date.

        Args:
            date (datetime): Date object.
        """

        members, mem_labels = self.generate_members(date)

        # List to hold all submitted futures
        futures = []

        tasks = [(date, member, mem_labels[m], fc_time) for fc_time in self.fc_times
                 for m, member in enumerate(members)]
        if parallel:
            # # Use ThreadPoolExecutor to run tasks in parallel
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Submit tasks to the executor
                futures = [executor.submit(self.retrieve_data_member, *task) for task in tasks]

                # Wait for all tasks to complete
                concurrent.futures.wait(futures)

            # Wait for all tasks to complete
            all_tasks_completed = all(future.done() for future in futures)
            return all_tasks_completed
        else:
            print(f'Working in serial mode')
            for task in tasks:
                self.retrieve_data_member(*task)

    def makes_5d_cube(self, data, wave_names, time_coord, pressure_coord,
                      lat_coord, lon_coord):
        # ===========================================================================
        # # Make a 3D cube of Latitude, wavenumber & frequency dimensions
        # ===========================================================================
        var_cube = iris.cube.Cube(data)

        var_cube.rename('wave_anomalies')
        wave_coord = iris.coords.DimCoord(range(len(wave_names)), long_name='wave_name')
        wave_coord.guess_bounds()
        # ['Kelvin', 'WMRG', 'R1', 'R2']
        var_cube.attributes = {'Kelvin': 0, 'WMRG': 1, 'R1': 2, 'R2': 3}
        wave_coord.attributes = {'Kelvin': 0, 'WMRG': 1, 'R1': 2, 'R2': 3}

        var_cube.add_dim_coord(wave_coord, 0)
        var_cube.add_dim_coord(time_coord, 1)
        var_cube.add_dim_coord(pressure_coord, 2)
        var_cube.add_dim_coord(lat_coord, 3)
        var_cube.add_dim_coord(lon_coord, 4)

        return var_cube

    def uz_to_qr(self, u, z):
        # transform u,z to q, r using q=z*(g/c) + u; r=z*(g/c) - u
        q = z * self.g_on_c + u
        r = z * self.g_on_c - u
        return q, r

    def filt_project(self, qf, rf, vf, lats, y0, waves, pmin, pmax, kmin, kmax, c_on_g):
        '''
        Author: Steve J. Woolnough (November 2019)
        :param qf:
        :param rf:
        :param vf:
        :param lats:
        :param y0:
        :param waves:
        :param pmin:
        :param pmax:
        :param kmin:
        :param kmax:
        :param c_on_g:
        :return:
        '''

        # find size of arrays
        nf = qf.shape[0]
        nz = qf.shape[1]
        nlats = lats.size
        nk = qf.shape[3]

        # Find frequencies and wavenumbers corresponding to pmin,pmax and kmin,kmax in coeff matrices
        f = np.fft.fftfreq(nf, 0.25)
        fmin = np.where((f >= 1. / pmax))[0][0]
        fmax = (np.where((f > 1. / pmin))[0][0]) - 1
        f1p = fmin
        f2p = fmax + 1
        f1n = nf - fmax
        f2n = nf - fmin + 1
        k1p = kmin
        k2p = kmax + 1
        k1n = nk - kmax
        k2n = nk - kmin + 1

        ### note that need to adjust for the fact that array referencing doesn't include last point!!!!!!

        # Define the parobolic cylinder functions
        spi2 = np.sqrt(2 * np.pi)
        # Normalization for the 1st 4 paroblic Cylindrical Functions
        dsq = np.array([spi2, spi2, 2 * spi2, 6 * spi2])
        d = np.zeros([dsq.size, nlats])
        y = lats[:] / y0
        ysq = y ** 2
        d[0, :] = np.exp(-ysq / 4.0)
        d[1, :] = y * d[0, :]
        d[2, :] = (ysq - 1.0) * d[0, :]
        d[3, :] = y * (ysq - 3.0) * d[0, :]
        dlat = np.abs(lats[0] - lats[1])

        qf_Kel = np.zeros([nf, nz, nk], dtype=complex)
        qf_mode = np.zeros([dsq.size, nf, nz, nk], dtype=complex)
        rf_mode = np.zeros([dsq.size, nf, nz, nk], dtype=complex)
        vf_mode = np.zeros([dsq.size, nf, nz, nk], dtype=complex)

        # reorder the spectral coefficents to make the latitudes the last dimension
        qf = np.transpose(qf, [0, 1, 3, 2])
        rf = np.transpose(rf, [0, 1, 3, 2])
        vf = np.transpose(vf, [0, 1, 3, 2])

        for m in np.arange(dsq.size):
            if m == 0:  # For eastward moving Kelvin Waves
                qf_Kel[f1n:f2n, :, k1p:k2p] = np.sum(qf[f1n:f2n, :, k1p:k2p, :] \
                                                     * np.squeeze(d[m, :]) * dlat, axis=-1) / (dsq[m] * y0)
                qf_Kel[f1p:f2p, :, k1n:k2n] = np.sum(qf[f1p:f2p, :, k1n:k2n, :] \
                                                     * np.squeeze(d[m, :]) * dlat, axis=-1) / (dsq[m] * y0)
            # For westward moving waves
            qf_mode[m, f1n:f2n, :, k1n:k2n] = np.sum(qf[f1n:f2n, :, k1n:k2n, :] \
                                                     * np.squeeze(d[m, :]) * dlat, axis=-1) / (dsq[m] * y0)
            qf_mode[m, f1p:f2p, :, k1p:k2p] = np.sum(qf[f1p:f2p, :, k1p:k2p, :] \
                                                     * np.squeeze(d[m, :]) * dlat, axis=-1) / (dsq[m] * y0)
            rf_mode[m, f1n:f2n, :, k1n:k2n] = np.sum(rf[f1n:f2n, :, k1n:k2n, :] \
                                                     * np.squeeze(d[m, :]) * dlat, axis=-1) / (dsq[m] * y0)
            rf_mode[m, f1p:f2p, :, k1p:k2p] = np.sum(rf[f1p:f2p, :, k1p:k2p, :] \
                                                     * np.squeeze(d[m, :]) * dlat, axis=-1) / (dsq[m] * y0)
            vf_mode[m, f1n:f2n, :, k1n:k2n] = np.sum(vf[f1n:f2n, :, k1n:k2n, :] \
                                                     * np.squeeze(d[m, :]) * dlat, axis=-1) / (dsq[m] * y0)
            vf_mode[m, f1p:f2p, :, k1p:k2p] = np.sum(vf[f1p:f2p, :, k1p:k2p, :] \
                                                     * np.squeeze(d[m, :]) * dlat, axis=-1) / (dsq[m] * y0)

        uf_wave = np.zeros([waves.size, nf, nz, nlats, nk], dtype=complex)
        zf_wave = np.zeros([waves.size, nf, nz, nlats, nk], dtype=complex)
        vf_wave = np.zeros([waves.size, nf, nz, nlats, nk], dtype=complex)

        for w in np.arange(waves.size):
            if waves[w] == 'Kelvin':
                for j in np.arange(nlats):
                    uf_wave[w, :, :, j, :] = \
                        0.5 * qf_Kel[:, :, :] * d[0, j]
                    zf_wave[w, :, :, j, :] = \
                        0.5 * qf_Kel[:, :, :] * d[0, j] * c_on_g

            if waves[w] == 'WMRG':
                for j in np.arange(nlats):
                    uf_wave[w, :, :, j, :] = \
                        0.5 * qf_mode[1, :, :, :] * d[1, j]
                    zf_wave[w, :, :, j, :] = \
                        0.5 * qf_mode[1, :, :, :] * d[1, j] * c_on_g
                    vf_wave[w, :, :, j, :] = \
                        vf_mode[0, :, :, :] * d[0, j]

            if waves[w] == 'R1':
                for j in np.arange(nlats):
                    uf_wave[w, :, :, j, :] = \
                        0.5 * (qf_mode[2, :, :, :] * d[2, j] - rf_mode[0, :, :, :] * d[0, j])
                    zf_wave[w, :, :, j, :] = \
                        0.5 * (qf_mode[2, :, :, :] * d[2, j] + rf_mode[0, :, :, :] * d[0, j]) * c_on_g
                    vf_wave[w, :, :, j, :] = \
                        vf_mode[1, :, :, :] * d[1, j]

            if waves[w] == 'R2':
                for j in np.arange(nlats):
                    uf_wave[w, :, :, j, :] = \
                        0.5 * (qf_mode[3, :, :, :] * d[3, j] - rf_mode[1, :, :, :] * d[1, j])
                    zf_wave[w, :, :, j, :] = \
                        0.5 * (qf_mode[3, :, :, :] * d[3, j] + rf_mode[1, :, :, :] * d[1, j]) * c_on_g
                    vf_wave[w, :, :, j, :] = \
                        vf_mode[2, :, :, :] * d[2, j]

        return uf_wave, zf_wave, vf_wave

    def write_wave_data(self, date, wave_cube, mem_label, var_name='u_wave'):

        str_hr = date.strftime('%H')
        date_label = date.strftime('%Y%m%d_%H')
        outfile_dir = os.path.join(self.config_values['mogreps_eqwaves_processed_dir'], date_label)

        # Generate a realization coordinate
        realiz_coord = iris.coords.DimCoord([int(mem_label)], standard_name='realization',
                                            var_name='realization')

        wave_cube.name = var_name
        wave_cube.long_name = var_name
        print(wave_cube.name)

        # Check if 'realization' coordinate exists
        if wave_cube.coords('realization'):
            wave_cube.remove_coord('realization')

        # Add 'realization' coordinate as an auxiliary coordinate
        wave_cube.add_aux_coord(realiz_coord)

        # Realistically you will probably only want to write out say (T-4:T+7) so that
        # you can plot an animation of the last few days and the forecast
        # total of 45 time points
        # write_out_times = 45
        # This has been moved to the plotting step.

        wave_names = wave_cube.coord('wave_name').attributes
        # Writing files out for each wave separately
        for wname, index in wave_names.items():
            print(wname, index)
            var_file_out = os.path.join(outfile_dir, f'{var_name}_{wname}_{date_label}Z_{mem_label}.nc')
            iris.save(wave_cube[index], var_file_out)
            print(var_file_out)

    def derivative(self, cube, axisname):
        dcube = cube.copy()

        coord_names = np.array([c.var_name for c in cube.coords()])
        print(coord_names, axisname)
        if axisname == 'latitude':
            lats = cube.coord('latitude').points
            axis_index = np.where(coord_names == 'latitude')[0][0]
            dlat = np.diff(lats) * self.deg2rad  # convert to radians
            dy = self.R * np.sin(dlat)  # constant at this latitude
            dcube = calculus.differentiate(cube, 'latitude')
            dcube /= iris.util.broadcast_to_shape(dy, dcube.shape, (axis_index,))

        if axisname == 'longitude':
            lats = cube.coord('latitude').points
            lons = cube.coord('longitude').points
            axis_index = np.where(coord_names == 'latitude')[0][0]
            print(axis_index)
            dlon = (lons[1] - lons[0]) * self.deg2rad  # convert to radians
            dx = np.array([self.R * np.cos(self.deg2rad * lat) * dlon for lat in lats])
            dcube = calculus.differentiate(cube, 'longitude')
            dcube /= iris.util.broadcast_to_shape(dx, dcube.shape, (axis_index,))
        return dcube

    def check_if_wave_computed(self, date, mem_label):
        wnames = ['Kelvin', 'WMRG', 'R1', 'R2']
        str_hr = date.strftime('%H')
        date_label = date.strftime('%Y%m%d_%H')
        outfile_dir = os.path.join(self.config_values['mogreps_eqwaves_processed_dir'], date_label)

        var_files = [os.path.join(outfile_dir, f'{var_name}_{wname}_{date_label}Z_{mem_label}.nc')
                     for var_name in ['vort_wave', 'div_wave'] for wname in wnames]

        return all([os.path.exists(var_file) for var_file in var_files])

    def compute_waves_driver_member(self, date, mem_label):
        """
        Computes wave-related variables for a specific member of the MOGREPS forecast dataset.

        Parameters:
        - date (datetime.datetime): The date and time for which the computation is performed.
        - mem_label (str): The label of the member for which the computation is performed.

        This method loads wind and geopotential height data for the specified date and member from NetCDF files,
        computes wave-related variables such as Fourier transforms, wave projection, and inverse Fourier transforms,
        and writes the resulting wave data to NetCDF files. It also computes vorticity and divergence of the wind
        field and writes these data to NetCDF files.

        Author: Steve J. Woolnough (November 2019)
        Adapted: Prince Xavier

        Note: The configuration settings, such as the directory path for processed forecast data, and other parameters
        required for computation, should be properly configured before calling this method.

        Returns:
        None
        """
        print(f'Computing wave for member: {mem_label}')

        if not self.check_if_wave_computed(date, mem_label):

            str_hr = date.strftime('%H')
            date_label = date.strftime('%Y%m%d_%H')
            outfile_dir = os.path.join(self.config_values['mogreps_eqwaves_processed_dir'], date_label)

            u_file = os.path.join(outfile_dir, f'x_wind_combined_{date_label}Z_{mem_label}.nc')
            v_file = os.path.join(outfile_dir, f'y_wind_combined_{date_label}Z_{mem_label}.nc')
            z_file = os.path.join(outfile_dir, f'geopotential_height_combined_{date_label}Z_{mem_label}.nc')

            u = iris.load_cube(u_file)
            v = iris.load_cube(v_file)
            z = iris.load_cube(z_file)

            lons = u.coord('longitude')
            lats = u.coord('latitude')
            press = u.coord('pressure')
            time_coord = u.coord('time')

            # convert u,z to q,r
            q, r = self.uz_to_qr(u.data, z.data)

            # Fourier transform in time and longitude
            qf = np.fft.fft2(q.data, axes=(0, -1))
            rf = np.fft.fft2(r.data, axes=(0, -1))
            vf = np.fft.fft2(v.data, axes=(0, -1))

            del q, r, v  # Free up memory
            gc.collect()

            # Project onto individual wave modes
            uf_wave, zf_wave, vf_wave = self.filt_project(qf, rf, vf, lats.points,
                                                          self.y0, self.wave_names, self.pmin, self.pmax,
                                                          self.kmin, self.kmax, self.c_on_g)
            del qf, rf, vf  # Free up memory
            gc.collect()

            # Inverse Fourier transform in time and longitude
            u_wave = np.real(np.fft.ifft2(uf_wave, axes=(1, -1)))
            z_wave = np.real(np.fft.ifft2(zf_wave, axes=(1, -1)))
            v_wave = np.real(np.fft.ifft2(vf_wave, axes=(1, -1)))

            del uf_wave, zf_wave, vf_wave  # Free memory
            gc.collect()

            # Make iris cubes before writing the data
            u_wave_cube = self.makes_5d_cube(u_wave, self.wave_names, time_coord, press, lats, lons)
            v_wave_cube = self.makes_5d_cube(v_wave, self.wave_names, time_coord, press, lats, lons)
            z_wave_cube = self.makes_5d_cube(z_wave, self.wave_names, time_coord, press, lats, lons)
            print(z_wave_cube)
            del u_wave, v_wave, z_wave  # Free memory
            gc.collect()

            self.write_wave_data(date, u_wave_cube, mem_label, var_name='u_wave')
            self.write_wave_data(date, v_wave_cube, mem_label, var_name='v_wave')
            self.write_wave_data(date, z_wave_cube, mem_label, var_name='z_wave')

            # Compute vorticty and divergence
            # Divergence
            div_wave = self.derivative(u_wave_cube, 'longitude').regrid(u_wave_cube, iris.analysis.Linear())
            div_wave += self.derivative(v_wave_cube, 'latitude').regrid(u_wave_cube, iris.analysis.Linear())
            self.write_wave_data(date, div_wave, mem_label, var_name='div_wave')
            del div_wave  # Free memory
            gc.collect()

            # Vorticity
            vort_wave = self.derivative(v_wave_cube, 'longitude').regrid(u_wave_cube, iris.analysis.Linear())
            vort_wave -= self.derivative(u_wave_cube, 'latitude').regrid(u_wave_cube, iris.analysis.Linear())
            self.write_wave_data(date, vort_wave, mem_label, var_name='vort_wave')

            del vort_wave, u_wave_cube, v_wave_cube, z_wave_cube  # Free memory
            gc.collect()
        else:
            print(f'All wave files are found. skipping computation.')

    def process_mogreps_forecast_data(self, date):
        """
        Retrieves and processes MOGREPS forecast data for a given date.

        Args:
            date (datetime): The date for which the forecast data is retrieved and processed.

        This method retrieves MOGREPS forecast data for the specified date and processes it in parallel for each member.
        It first determines the forecast initialization hour ('00', '06', '12', or '18') from the given date,
        and then selects the corresponding member labels. For each member label, it creates a pool of processes
        to execute the processing function `process_mogreps_forecast_data_member` in parallel.

        After processing the forecast data, it computes wave-related variables for each member using
        the `compute_waves_driver_member` method in parallel.

        Note: Ensure that the necessary methods `process_mogreps_forecast_data_member` and
        `compute_waves_driver_member` are defined and implemented correctly within the class.

        Returns:
            None
        """

        members, mem_labels = self.generate_members(date)

        for i, mem_label in enumerate(mem_labels):
            self.print_progress_bar(i, len(mem_labels))
            self.process_mogreps_forecast_data_member(date, mem_label)

        '''
        # Create a pool of processes for processing forecast data
        with multiprocessing.Pool() as pool:
            # Submit tasks asynchronously using starmap_async
            results = pool.starmap_async(self.process_mogreps_forecast_data_member,
                                         [(date, mem_label) for mem_label in mem_labels])

            # Close the pool to prevent further task submission
            pool.close()

            # Wait for all processes to complete
            results.wait()

        # Check if all tasks are completed successfully
        #all_tasks_completed = all(result.successful() for result in results.get())
        
        return results
        '''

    def compute_waves_forecast_data(self, date):
        """
        Retrieves and processes MOGREPS forecast data for a given date.

        Args:
            date (datetime): The date for which the forecast data is retrieved and processed.

        After processing the forecast data, it computes wave-related variables for each member using
        the `compute_waves_driver_member` method in parallel.


        Returns:
            None
        """
        print('Computing waves...')
        members, mem_labels = self.generate_members(date)

        total_tasks = len(mem_labels)
        completed_tasks = 0

        for mem_label in mem_labels:
            self.compute_waves_driver_member(date, mem_label)
            completed_tasks += 1
            progress = (completed_tasks / total_tasks) * 100
            print(f'\rProgress: {progress:.2f}%', end='', flush=True)

        # print()  # Print newline after progress

        # Create a new pool of processes for computing wave-related variables
        # pool = multiprocessing.Pool()
        # Execute the compute_waves_driver_member function in parallel for each member
        # pool.starmap(self.compute_waves_driver_member, [(date, mem_label) for mem_label in mem_labels])
        # Close the pool to free up resources
        # pool.close()
        # Wait for all processes to finish
        # pool.join()

        # Check if all tasks are completed
        # if pool._state == multiprocessing.pool.CLOSE:
        #    return "All tasks completed"
        # else:
        #    return "Tasks are still running"
