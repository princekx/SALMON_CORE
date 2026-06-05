import os
import sys
import glob
import json
import logging
import datetime
import uuid
import concurrent.futures
import warnings
import configparser
import cf_units
import numpy as np
import pandas as pd

import iris
import iris.coords
import iris.coord_categorisation

from bokeh.plotting import figure, show, save, output_file
from bokeh.models import ColumnDataSource, Patches, Plot, Title, HoverTool
from bokeh.models import Range1d, LinearColorMapper, ColorBar, GeoJSONDataSource
from bokeh.palettes import GnBu9, Magma6, Greys256, Greys9, RdPu9, TolRainbow12
from bokeh.palettes import Iridescent23, TolYlOrBr9, Bokeh8, Blues9
from bokeh.models import CheckboxGroup, CheckboxButtonGroup, CustomJS, Button
from bokeh.models import Legend, LegendItem
from bokeh.layouts import column, row, Spacer
from skimage import measure

from salmon.core.task import Task
from salmon.utils.moose import MooseClient
from salmon.utils.config import load_global_config
from salmon.utils.cube import create_latlon_grid, remove_um_version
from .wave_processor import WaveProcessor

logger = logging.getLogger(__name__)

# Suppress non-critical warnings to keep task logs concise.
warnings.simplefilter('ignore')

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_UTILS_DIR = os.path.normpath(os.path.join(_THIS_DIR, '..', '..', 'utils'))
_DEFAULT_QUERY_DIR = os.path.normpath(os.path.join(_UTILS_DIR, 'query_files'))
_DEFAULT_MAP_JSON = os.path.normpath(os.path.join(_UTILS_DIR, 'map_data', 'custom.geo.json'))

NTIMES_ANALYSIS = 332
FC_TIMES_6H = tuple(np.arange(6, 174, 6))
EQW_VARS = ('x_wind', 'y_wind', 'geopotential_height', 'precipitation_amount')


class RetrieveEqWavesData(Task):
    """Retrieve analysis and forecast files required for Equatorial Waves."""

    def run(self):
        date = self.context.date
        parallel = bool(self.config.get('parallel', True))
        max_workers = int(self.config.get('max_workers', 10))

        self._init_config_values()

        ok_analysis = self._retrieve_analysis(date=date, parallel=parallel, max_workers=max_workers)
        ok_forecast = self._retrieve_forecast(date=date, parallel=parallel, max_workers=max_workers)

        if ok_analysis and ok_forecast:
            logger.info('EqWaves data retrieval complete.')
        else:
            logger.warning('EqWaves data retrieval completed with errors.')

    def _init_config_values(self):
        if hasattr(self, 'config_values'):
            return

        global_cfg = load_global_config()
        analysis_cfg = global_cfg.get('analysis', {})
        mogreps_cfg = global_cfg.get('mogreps', {})

        shared_query = self.config.get('query')
        analysis_query = self.config.get('analysis_query') or analysis_cfg.get('query') or shared_query or 'analysis_combined.query'
        forecast_query = self.config.get('forecast_query') or mogreps_cfg.get('query') or shared_query or 'mogreps_combined.query'

        def _resolve_query_path(value):
            if not value:
                return None
            expanded = os.path.expandvars(os.path.expanduser(str(value)))
            if os.path.isabs(expanded):
                return expanded
            if os.path.exists(expanded):
                return expanded
            return os.path.join(_DEFAULT_QUERY_DIR, expanded)

        self.config_values = {
            'analysis_moose_dir': analysis_cfg.get('moose', 'moose:/opfc/atm/global/prods/'),
            'analysis_raw_dir': analysis_cfg.get('raw', '/tmp/salmon_raw/analysis'),
            'analysis_query_file': _resolve_query_path(analysis_query),
            'forecast_moose_dir': mogreps_cfg.get('moose', 'moose:/opfc/atm/mogreps-g/prods/'),
            'forecast_raw_dir': mogreps_cfg.get('raw', '/tmp/salmon_raw/mogreps'),
            'forecast_query_file': _resolve_query_path(forecast_query),
            'temp_query_dir': mogreps_cfg.get('temp', '/tmp/salmon_temp'),
        }

        os.makedirs(self.config_values['temp_query_dir'], exist_ok=True)
        logger.info(
            'EqWaves query templates: analysis=%s forecast=%s',
            self.config_values['analysis_query_file'],
            self.config_values['forecast_query_file'],
        )

    def _iter_analysis_dates(self, date):
        for i in range(NTIMES_ANALYSIS):
            yield date - datetime.timedelta(hours=i * 6)

    def _iter_forecast_tasks(self, date):
        members, mem_labels = self._generate_members(date)
        for i, member in enumerate(members):
            for fc in FC_TIMES_6H:
                yield (date, member, mem_labels[i], int(fc))

    def _resolve_forecast_moose_dir(self, date):
        os47_date = datetime.datetime(2026, 1, 21)
        if date > os47_date:
            return os.path.join(self.config_values['forecast_moose_dir'], f'{date.year}.pp')
        return os.path.join(self.config_values['forecast_moose_dir'], f'{date:%Y%m}.pp')

    def _create_query_file(self, query_template, replacements, suffix):
        local_query = os.path.join(self.config_values['temp_query_dir'], f'eqw_{suffix}_{uuid.uuid4()}.query')
        MooseClient().create_query_file(query_template, local_query, replacements)
        return local_query

    def _retrieve_analysis_step(self, date):
        moose_dir = os.path.join(self.config_values['analysis_moose_dir'], f'{date.year}.pp')
        hr = date.strftime('%H')
        fct = '000'

        out_dir = os.path.join(self.config_values['analysis_raw_dir'], date.strftime('%Y%m%d'))
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'qg{hr}T{fct}.pp')

        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return True

        filemoose = f'prods_op_gl-mn_{date:%Y%m%d}_{hr}_{fct}.pp'
        local_query = self._create_query_file(
            self.config_values['analysis_query_file'],
            {'fctime': fct, 'filemoose': filemoose},
            'analysis',
        )

        try:
            MooseClient().retrieve(local_query, moose_dir, out_path)
            return True
        except Exception as exc:
            logger.error('Analysis retrieval failed for %s: %s', out_path, exc)
            return False
        finally:
            if os.path.exists(local_query):
                os.remove(local_query)

    def _retrieve_forecast_step(self, date, member, mem_label, fc):
        moose_dir = self._resolve_forecast_moose_dir(date)
        hr = date.strftime('%H')
        fct = f'{fc:03d}'

        out_dir = os.path.join(self.config_values['forecast_raw_dir'], date.strftime('%Y%m%d'), hr, mem_label)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'qg{hr}T{fct}.pp')

        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return True

        filemoose = f'prods_op_mogreps-g_{date:%Y%m%d}_{hr}_{member}_{fct}.pp'
        local_query = self._create_query_file(
            self.config_values['forecast_query_file'],
            {'fctime': fct, 'filemoose': filemoose},
            'forecast',
        )

        try:
            MooseClient().retrieve(local_query, moose_dir, out_path)
            return True
        except Exception as exc:
            logger.error('Forecast retrieval failed for %s: %s', out_path, exc)
            return False
        finally:
            if os.path.exists(local_query):
                os.remove(local_query)

    def _retrieve_analysis(self, date, parallel=True, max_workers=10):
        if not self.config_values['analysis_query_file']:
            raise ValueError('Missing analysis query file in config (analysis.query or task analysis_query).')

        tasks = list(self._iter_analysis_dates(date))
        logger.info('Retrieving %s analysis steps for EqWaves...', len(tasks))

        if not parallel:
            return all(self._retrieve_analysis_step(d) for d in tasks)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._retrieve_analysis_step, d) for d in tasks]
            return all(f.result() for f in concurrent.futures.as_completed(futures))

    def _retrieve_forecast(self, date, parallel=True, max_workers=10):
        if not self.config_values['forecast_query_file']:
            raise ValueError('Missing forecast query file in config (mogreps.query or task forecast_query).')

        tasks = list(self._iter_forecast_tasks(date))
        logger.info('Retrieving MOGREPS forecast for EqWaves (%s jobs)...', len(tasks))

        if not parallel:
            return all(self._retrieve_forecast_step(*t) for t in tasks)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._retrieve_forecast_step, *t) for t in tasks]
            return all(f.result() for f in concurrent.futures.as_completed(futures))

    def _generate_members(self, date):
        str_hr = date.strftime('%H')
        members_tuple = {
            '00': ['00'] + [f'{fc:02}' for fc in range(1, 18)],
            '06': ['00'] + [f'{fc:02}' for fc in range(18, 35)],
            '12': ['00'] + [f'{fc:02}' for fc in range(1, 18)],
            '18': ['00'] + [f'{fc:02}' for fc in range(18, 35)],
        }
        members = members_tuple[str_hr]
        mem_labels = [f'{fc:03d}' for fc in range(0, 18)]
        return members, mem_labels


class ComputeEqWavesIndices(Task):
    """Compute Equatorial Wave indices using FFT projection workflow."""

    def run(self):
        date = self.context.date
        parallel = bool(self.config.get('parallel', True))
        max_workers = int(self.config.get('max_workers', 6))
        self._init_config_values()

        logger.info('Computing EqWaves indices for %s...', date)

        ref_grid = create_latlon_grid().intersection(latitude=(-24, 24))
        self._ensure_analysis_history(date, ref_grid)

        members, mem_labels = self._generate_members(date)

        if not parallel:
            for i, mem_label in enumerate(mem_labels):
                logger.info('Processing member %s (%s/%s)', mem_label, i + 1, len(mem_labels))
                self._run_member_job(date=date, mem=members[i], mem_label=mem_label, ref_grid=ref_grid)
            return

        logger.info('Processing %s members in parallel (max_workers=%s)', len(mem_labels), max_workers)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    self._run_member_job,
                    date=date,
                    mem=members[i],
                    mem_label=mem_labels[i],
                    ref_grid=ref_grid,
                )
                for i in range(len(mem_labels))
            ]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        if not all(results):
            failed = len([r for r in results if not r])
            logger.warning('EqWaves member processing finished with %s failed member jobs.', failed)

    def _run_member_job(self, date, mem, mem_label, ref_grid):
        try:
            processor = WaveProcessor()
            self._process_member(
                date=date,
                mem=mem,
                mem_label=mem_label,
                fc_times=FC_TIMES_6H,
                ref_grid=ref_grid,
                processor=processor,
            )
            return True
        except Exception:
            logger.exception('EqWaves member processing failed for member %s', mem_label)
            return False

    def _init_config_values(self):
        if hasattr(self, 'config_values'):
            return

        model = self.context.get_config('model', 'mogreps')
        global_cfg = load_global_config()
        analysis_cfg = global_cfg.get('analysis', {})
        model_cfg = global_cfg.get(model, {})

        analysis_processed_dir = os.path.join(
            analysis_cfg.get('processed', '/tmp/salmon_processed/analysis'),
            'eqwaves',
        )
        output_dir = os.path.join(
            model_cfg.get('processed', f'/tmp/salmon_processed/{model}'),
            'eqwaves',
            self.context.recipe_name,
            self.context.date.strftime('%Y%m%d_%H'),
        )

        self.config_values = {
            'model': model,
            'analysis_raw_dir': analysis_cfg.get('raw', '/tmp/salmon_raw/analysis'),
            'analysis_processed_dir': analysis_processed_dir,
            'model_raw_dir': model_cfg.get('raw', f'/tmp/salmon_raw/{model}'),
            'output_dir': output_dir,
        }

        os.makedirs(analysis_processed_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

    def _analysis_dates(self, date):
        return sorted([date - datetime.timedelta(hours=i * 6) for i in range(NTIMES_ANALYSIS)])

    def _analysis_file(self, var, date_label):
        return os.path.join(self.config_values['analysis_processed_dir'], f'{var}_analysis_{date_label}.nc')

    def _ensure_analysis_history(self, date, ref_grid):
        date_label = date.strftime('%Y%m%d_%H')
        analysis_dates = self._analysis_dates(date)

        for var in EQW_VARS:
            out_file = self._analysis_file(var, date_label)
            if os.path.exists(out_file):
                continue

            logger.info('Processing analysis history for %s...', var)
            raw_files = [
                os.path.join(self.config_values['analysis_raw_dir'], d.strftime('%Y%m%d'), f"qg{d.strftime('%H')}T000.pp")
                for d in analysis_dates
            ]
            existing = [f for f in raw_files if os.path.exists(f)]
            if len(existing) < 100:
                logger.error('Not enough analysis files found for %s (%s/%s)', var, len(existing), NTIMES_ANALYSIS)
                continue

            if var == 'precipitation_amount':
                cube = iris.load_cube(existing, var, callback=remove_um_version)
            else:
                cube = iris.load_cube(
                    existing,
                    iris.Constraint(pressure=[200, 850]) & var,
                    callback=remove_um_version,
                )

            cube = cube.regrid(ref_grid, iris.analysis.Linear())
            iris.save(cube, out_file)

    def _forecast_files_for_member(self, date, mem_label, fc_times):
        str_hr = date.strftime('%H')
        return [
            os.path.join(
                self.config_values['model_raw_dir'],
                date.strftime('%Y%m%d'),
                str_hr,
                mem_label,
                f'qg{str_hr}T{fct:03d}.pp',
            )
            for fct in fc_times
        ]

    def _process_member(self, date, mem, mem_label, fc_times, ref_grid, processor):
        str_hr = date.strftime('%H')
        date_label = date.strftime('%Y%m%d_%H')

        if all([
            os.path.exists(os.path.join(self.config_values['output_dir'], f'vort_wave_{wn}_{date_label}Z_{mem_label}.nc'))
            for wn in processor.wave_names
        ]):
            return

        cubes_to_process = {}
        for var in EQW_VARS:
            an_file = self._analysis_file(var, date_label)
            if not os.path.exists(an_file):
                continue

            an_cube = iris.load_cube(an_file)
            fc_raw_files = self._forecast_files_for_member(date, mem_label, fc_times)
            existing_fc = [f for f in fc_raw_files if os.path.exists(f)]
            if not existing_fc:
                logger.warning('No forecast files for member %s', mem_label)
                continue

            fc_cube = self._read_forecasts(date, existing_fc, var, fc_times)
            if var == 'precipitation_amount':
                fc_cube.data[1:] -= fc_cube.data[:-1]
                an_cube.data *= 3600.0

            fc_cube = fc_cube.regrid(ref_grid, iris.analysis.Linear())

            combined = self._concat_analysis_forecast(date, an_cube, fc_cube)
            realiz_coord = iris.coords.DimCoord([int(mem_label)], standard_name='realization', var_name='realization')
            if combined.coords('realization'):
                combined.remove_coord('realization')
            combined.add_aux_coord(realiz_coord)
            cubes_to_process[var] = combined

        if len(cubes_to_process) < 3:
            return

        u = cubes_to_process['x_wind']
        v = cubes_to_process['y_wind']
        z = cubes_to_process['geopotential_height']

        q, r = processor.uz_to_qr(u.data, z.data)
        qf = np.fft.fft2(q, axes=(0, -1))
        rf = np.fft.fft2(r, axes=(0, -1))
        vf = np.fft.fft2(v.data, axes=(0, -1))

        lats = u.coord('latitude').points
        ufw, zfw, vfw = processor.filt_project(qf, rf, vf, lats)

        u_wave = np.real(np.fft.ifft2(ufw, axes=(1, -1)))
        z_wave = np.real(np.fft.ifft2(zfw, axes=(1, -1)))
        v_wave = np.real(np.fft.ifft2(vfw, axes=(1, -1)))

        time_c = u.coord('time')
        press_c = u.coord('pressure')
        lat_c = u.coord('latitude')
        lon_c = u.coord('longitude')

        u_wave_cube = processor.makes_5d_cube(u_wave, time_c, press_c, lat_c, lon_c)
        v_wave_cube = processor.makes_5d_cube(v_wave, time_c, press_c, lat_c, lon_c)
        z_wave_cube = processor.makes_5d_cube(z_wave, time_c, press_c, lat_c, lon_c)

        for wn in processor.wave_names:
            idx = u_wave_cube.coord('wave_name').attributes[wn]
            iris.save(u_wave_cube[idx], os.path.join(self.config_values['output_dir'], f'u_wave_{wn}_{date_label}Z_{mem_label}.nc'))
            iris.save(v_wave_cube[idx], os.path.join(self.config_values['output_dir'], f'v_wave_{wn}_{date_label}Z_{mem_label}.nc'))
            iris.save(z_wave_cube[idx], os.path.join(self.config_values['output_dir'], f'z_wave_{wn}_{date_label}Z_{mem_label}.nc'))

        div = processor.derivative(u_wave_cube, 'longitude').regrid(u_wave_cube, iris.analysis.Linear())
        div += processor.derivative(v_wave_cube, 'latitude').regrid(u_wave_cube, iris.analysis.Linear())

        vort = processor.derivative(v_wave_cube, 'longitude').regrid(u_wave_cube, iris.analysis.Linear())
        vort -= processor.derivative(u_wave_cube, 'latitude').regrid(u_wave_cube, iris.analysis.Linear())

        for wn in processor.wave_names:
            idx = u_wave_cube.coord('wave_name').attributes[wn]
            iris.save(div[idx], os.path.join(self.config_values['output_dir'], f'div_wave_{wn}_{date_label}Z_{mem_label}.nc'))
            iris.save(vort[idx], os.path.join(self.config_values['output_dir'], f'vort_wave_{wn}_{date_label}Z_{mem_label}.nc'))

    def _read_forecasts(self, date, files, var, fc_times):
        cubes = []
        for i, fpath in enumerate(files):
            if var == 'precipitation_amount':
                cube = iris.load_cube(fpath, var)
                if len(cube.shape) == 3:
                    cube = cube.collapsed('time', iris.analysis.MEAN)
            else:
                cube = iris.load_cube(fpath, iris.Constraint(pressure=[200, 850]) & var)
                if len(cube.shape) == 4:
                    cube = cube.collapsed('time', iris.analysis.MEAN)

            cube.cell_methods = ()
            for coord in ('forecast_period', 'time'):
                if cube.coords(coord):
                    cube.remove_coord(coord)

            unit_str = f"hours since {date:%Y-%m-%d %H}:00:00"
            cube.add_aux_coord(iris.coords.AuxCoord(fc_times[i], standard_name='forecast_period', units=unit_str))
            cube.add_aux_coord(iris.coords.AuxCoord(fc_times[i], standard_name='time', units=unit_str))
            cubes.append(cube)

        return iris.cube.CubeList(cubes).merge_cube()

    def _concat_analysis_forecast(self, date, an_cube, fc_cube):
        data = np.concatenate((an_cube.data, fc_cube.data), axis=0)

        an_times = [date + datetime.timedelta(hours=(i + 1) * 6) for i in range(-NTIMES_ANALYSIS, 0)]
        fc_times = [date + datetime.timedelta(hours=(i + 1) * 6) for i in range(0, len(FC_TIMES_6H))]
        all_dates = an_times + fc_times

        time_units = cf_units.Unit('hours since 1970-01-01 00:00:00', calendar='gregorian')
        time_values = time_units.date2num(all_dates)
        time_coord = iris.coords.DimCoord(time_values, standard_name='time', units=time_units)

        dims = [(time_coord, 0)]
        for coord in an_cube.dim_coords[1:]:
            dims.append((coord, an_cube.coord_dims(coord)[0]))

        return iris.cube.Cube(data, long_name=fc_cube.long_name, units=fc_cube.units, dim_coords_and_dims=dims)

    def _generate_members(self, date):
        str_hr = date.strftime('%H')
        members_tuple = {
            '00': ['00'] + [f'{fc:02}' for fc in range(1, 18)],
            '06': ['00'] + [f'{fc:02}' for fc in range(18, 35)],
            '12': ['00'] + [f'{fc:02}' for fc in range(1, 18)],
            '18': ['00'] + [f'{fc:02}' for fc in range(18, 35)],
        }
        members = members_tuple[str_hr]
        mem_labels = [f'{fc:03d}' for fc in range(0, 18)]
        return members, mem_labels


class DisplayEqWavesMaps(Task):
    """Create EqWaves ensemble probability map products as Bokeh HTML files."""

    def run(self):
        date = self.context.date
        self._init_config_values()
        self.bokeh_plot_forecast_ensemble_probability_multiwave(date)

    def _init_config_values(self):
        if hasattr(self, 'config_values'):
            return

        model = self.context.get_config('model', 'mogreps')
        global_cfg = load_global_config()
        model_cfg = global_cfg.get(model, {})
    
        date_label = self.context.date.strftime('%Y%m%d_%H')
        processed_base = model_cfg.get('processed', f'/tmp/salmon_processed/{model}')
        processed_eqwaves_dir = os.path.join(processed_base, 'eqwaves', date_label)

        if model_cfg.get('plots'):
            plot_ens_dir = os.path.join(model_cfg.get('plots'), 'eqwaves', 'plot_ens')
        else:
            plot_ens_dir = os.path.join(processed_base, 'eqwaves', 'plot_ens')

        self.config_values = {
            'model': model,
            f'{model}_forecast_processed_dir': processed_eqwaves_dir,
            f'{model}_plot_ens': plot_ens_dir,
            "map_outline_json_file": self.config.get(
                "map_outline_json_file",
                os.path.normpath(_DEFAULT_MAP_JSON),
            ),
        }

        os.makedirs(self.config_values[f'{model}_plot_ens'], exist_ok=True)

        self.ntimes_total = 360
        self.ntimes_analysis = 332
        self.ntimes_forecast = 28
        self.wave_names = np.array(['Kelvin', 'WMRG', 'R1', 'R2'])
        self.pressures = ['850']
        self.thresholds = {
            'precip': 5,
            'Precip': 5,
            'Kelvin_850': -1 * 1e-6,
            'Kelvin_200': -2 * 1e-6,
            'WMRG_850': -1 * 1e-6,
            'WMRG_200': -2 * 1e-6,
            'R1_850': 5 * 1e-6,
            'R1_200': 2 * 1e-6,
            'R2_850': 2.5 * 1e-6,
            'R2_200': 2 * 1e-6,
        }
        self.times2plot = [t for t in range(-96, 174, 6)]
        self.map_outline_json_file = self.config_values['map_outline_json_file']
        self.plot_width = 1100

    def prepare_calendar(self, cube):
        for coord_name, coord_func in [
            ('year', iris.coord_categorisation.add_year),
            ('month_number', iris.coord_categorisation.add_month_number),
            ('day_of_month', iris.coord_categorisation.add_day_of_month),
            ('hour', iris.coord_categorisation.add_hour),
        ]:
            if not cube.coords(coord_name):
                coord_func(cube, 'time', name=coord_name)
        return cube

    def create_dates_dt(self, cube):
        cube = self.prepare_calendar(cube)
        return [
            datetime.datetime(y, m, d, h)
            for y, m, d, h in zip(
                cube.coord('year').points,
                cube.coord('month_number').points,
                cube.coord('day_of_month').points,
                cube.coord('hour').points,
            )
        ]

    def write_dates_json(self, date, json_file):
        new_date = date.strftime('%Y%m%d_%H')

        if not os.path.exists(json_file):
            with open(json_file, 'w', encoding='utf-8') as jfile:
                json.dump([new_date], jfile, indent=2)
            return

        with open(json_file, 'r', encoding='utf-8') as file:
            existing_dates = json.load(file)

        if new_date not in existing_dates:
            existing_dates.append(new_date)

        existing_dates.sort()

        with open(json_file, 'w', encoding='utf-8') as file:
            json.dump(existing_dates, file, indent=2)

        logger.info('Updated %s with date %s', json_file, new_date)

    def bokeh_plot2html(self, shade_var=None, contour_var=None, figure_tite=None,
                        shade_cbar_title=None, contour_cbar_title=None, html_file='test.html'):
        x_range = (0, 180)
        y_range = (-24, 24)

        width = self.plot_width
        aspect = (max(x_range) - min(x_range)) / (max(y_range) - min(y_range))
        height = int(width / (0.6 * aspect))

        plot = figure(
            height=height,
            width=width,
            x_range=x_range,
            y_range=y_range,
            tools='pan,reset,save,wheel_zoom,hover',
            x_axis_label='Longitude',
            y_axis_label='Latitude',
            aspect_scale=4,
            title=figure_tite,
            tooltips=[('Lat', '$y'), ('Lon', '$x'), ('Value', '@image')],
        )

        plot.title.text_font_size = '14pt'

        shade_levels = np.arange(0.1, 1.1, 0.1)
        color_mapper_z = LinearColorMapper(palette=Iridescent23, low=shade_levels.min(), high=shade_levels.max())
        color_bar = ColorBar(
            color_mapper=color_mapper_z,
            major_label_text_font_size='12pt',
            label_standoff=6,
            border_line_color=None,
            orientation='horizontal',
            location=(0, 0),
            width=400,
            title=shade_cbar_title,
            title_text_font_size='12pt',
        )

        plot.image(
            image=[shade_var.data],
            x=0,
            y=-24,
            dw=360,
            dh=48,
            alpha=0.8,
            color_mapper=color_mapper_z,
        )
        plot.add_layout(color_bar, 'below')

        if contour_var is not None:
            lons, lats = np.meshgrid(contour_var.coord('longitude').points, contour_var.coord('latitude').points)
            contour_levels = np.arange(0.4, 1.2, 0.2)
            contour_renderer = plot.contour(
                lons,
                lats,
                contour_var.data,
                contour_levels,
                fill_color=None,
                fill_alpha=0.3,
                line_color=Bokeh8,
                line_alpha=0.5,
                line_width=5,
            )
            colorbar = contour_renderer.construct_color_bar(
                major_label_text_font_size='12pt',
                orientation='horizontal',
                location=(-500, -135),
                width=400,
                title=contour_cbar_title,
                title_text_font_size='12pt',
            )
            plot.add_layout(colorbar, 'right')

        with open(self.map_outline_json_file, 'r', encoding='utf-8') as f:
            countries = GeoJSONDataSource(geojson=f.read())

        plot.patches('xs', 'ys', color=None, line_color='grey', source=countries, alpha=0.75)

        output_file(html_file)
        save(plot)
        logger.info('Plotted %s', html_file)

    def read_compute_ensemble_prob(self, files, wname=None, pressure_level=None, contour_cbar_title=None):
        _ = contour_cbar_title
        files = [f for f in files if os.path.exists(f)]
        if not files:
            return None

        cube = iris.load_cube(files)
        ntimes = len(self.times2plot)
        cube = cube[:, -ntimes:]

        if pressure_level is not None and cube.coords('pressure'):
            cube = cube.extract(iris.Constraint(pressure=float(pressure_level)))

        if wname == 'Precip':
            return cube.collapsed(
                'realization',
                iris.analysis.PROPORTION,
                function=lambda values: values > self.thresholds['Precip'],
            )

        if wname in ['Kelvin', 'WMRG']:
            threshold = self.thresholds[f'{wname}_{pressure_level}']
            return cube.collapsed(
                'realization',
                iris.analysis.PROPORTION,
                function=lambda values: values <= threshold,
            )

        if wname in ['R1', 'R2']:
            threshold = self.thresholds[f'{wname}_{pressure_level}']
            return cube.collapsed(
                'realization',
                iris.analysis.PROPORTION,
                function=lambda values: values >= threshold,
            )

        return None
    
    def get_skimage_contour_paths(self, lons, lats, cube_data, levels=[0.5, 0.75]):
        paths_x, paths_y = [], []
        for level in levels:
            contours = measure.find_contours(cube_data, level)
            for contour in contours:
                paths_x.append(contour[:, 1] + min(lons))
                paths_y.append(contour[:, 0] + min(lats))
        return paths_x, paths_y

    def bokeh_plot_allwaves2html(self, wave_timestep_dic, pressure_level,
                                 figure_tite=None, shade_cbar_title=None,
                                 contour_cbar_title=None, html_file='test.html'):

        x_range = (0, 180)  # could be anything - e.g.(0,1)
        y_range = (-24, 24)
        contour_alpha = 0.5
        width = self.plot_width
        shade_cbar_title = 'Probability of precipitation >=5 mm/day'
        aspect = (max(x_range) - min(x_range)) / (max(y_range) - min(y_range))
        height = int(width / (0.65 * aspect))


        # Prepare the initial image source
        precip_source = ColumnDataSource(data=dict(Precip=[wave_timestep_dic['Precip']]))
        contour_levels = [0.66]  # [0.5, 0.7, 0.9]
        #print(wave_timestep_dic.keys())

        # Prepare contour paths using functions for Kelvin and WMRG
        lats, lons = wave_timestep_dic['latitude'], wave_timestep_dic['longitude']
        contour_kelvin_x, contour_kelvin_y = self.get_skimage_contour_paths(lons, lats,
                                                                            wave_timestep_dic[f'Kelvin_{pressure_level}'],
                                                                       levels=contour_levels)
        contour_wmrg_x, contour_wmrg_y = self.get_skimage_contour_paths(lons, lats, wave_timestep_dic[f'WMRG_{pressure_level}'],
                                                                   levels=contour_levels)
        contour_R1_x, contour_R1_y = self.get_skimage_contour_paths(lons, lats, wave_timestep_dic[f'R1_{pressure_level}'],
                                                               levels=contour_levels)
        contour_R2_x, contour_R2_y = self.get_skimage_contour_paths(lons, lats, wave_timestep_dic[f'R2_{pressure_level}'],
                                                               levels=contour_levels)

        # Create separate ColumnDataSources for each contour field
        kelvin_source = ColumnDataSource(data=dict(xs=contour_kelvin_x, ys=contour_kelvin_y))
        wmrg_source = ColumnDataSource(data=dict(xs=contour_wmrg_x, ys=contour_wmrg_y))
        r1_source = ColumnDataSource(data=dict(xs=contour_R1_x, ys=contour_R1_y))
        r2_source = ColumnDataSource(data=dict(xs=contour_R2_x, ys=contour_R2_y))

        plot = figure(height=height, width=width, x_range=x_range, y_range=y_range,
                      tools=["pan, reset, save, wheel_zoom, hover"],
                      x_axis_label='Longitude', y_axis_label='Latitude', aspect_scale=4,
                      title=figure_tite)

        # Create a color mapper for the image
        color_mapper_z = LinearColorMapper(palette='Iridescent23', low=0.5, high=1)
        color_bar = ColorBar(color_mapper=color_mapper_z, major_label_text_font_size="12pt",
                             label_standoff=6, border_line_color=None, orientation="horizontal",
                             location=(0, 0), width=400, title=shade_cbar_title, title_text_font_size="12pt")
        image_renderer = plot.image('Precip', source=precip_source, x=0, y=-24, dw=360, dh=48, alpha=0.8,
                                    color_mapper=color_mapper_z)
        plot.add_layout(color_bar, 'below')


        with open(self.map_outline_json_file, "r") as f:
            countries = GeoJSONDataSource(geojson=f.read())

        plot.patches("xs", "ys", color=None, line_color="grey", source=countries, alpha=0.75)

        # Add empty MultiLine renderers for contours
        kelvin_renderer = plot.multi_line(xs='xs', ys='ys', source=kelvin_source, line_width=4, color="blue",
                                          alpha=contour_alpha)
        wmrg_renderer = plot.multi_line(xs='xs', ys='ys', source=wmrg_source, line_width=4, color="green",
                                        alpha=contour_alpha)
        r1_renderer = plot.multi_line(xs='xs', ys='ys', source=r1_source, line_width=4, color="red",
                                      alpha=contour_alpha)
        r2_renderer = plot.multi_line(xs='xs', ys='ys', source=r2_source, line_width=4, color="orange",
                                      alpha=contour_alpha)

        # Create Legend items manually
        legend_items = [
            LegendItem(label="Kelvin convergence", renderers=[kelvin_renderer]),
            LegendItem(label="WMRG convergence", renderers=[wmrg_renderer]),
            LegendItem(label="n=1 Rossby cyclonic vorticity", renderers=[r1_renderer]),
            LegendItem(label="n=2 Rossby cyclonic vorticity", renderers=[r2_renderer]),
        ]

        # Create a Legend and set its properties
        legend = Legend(items=legend_items, title="Click to show/hide (p >= 0.5)", label_text_font_size="10pt",
                        title_text_font_size="11pt",
                        location=(0, 0.5), background_fill_alpha=0.75)
        legend.click_policy = "hide"  # Allow toggling visibility on click

        # Add legend to the plot (set it outside the main plot area)
        plot.add_layout(legend)

        # Create CheckboxGroup to select multiple fields
        # checkbox_group = CheckboxGroup(labels=["Kelvin", "WMRG", "R1", "R2"], active=[0, 1, 2, 3], height=100, width=500)
        checkbox_group = CheckboxButtonGroup(labels=["Kelvin", "WMRG", "n=1 Rossby", "n=2 Rossby"], active=[0, 1, 2, 3])

        # Create a "Clear All" button
        clear_button = Button(label="Clear All", button_type="danger")

        # JavaScript callback to toggle visibility of contour lines based on selection
        checkbox_callback = CustomJS(args=dict(kelvin_renderer=kelvin_renderer, wmrg_renderer=wmrg_renderer,
                                               r1_renderer=r1_renderer, r2_renderer=r2_renderer,
                                               checkbox=checkbox_group), code="""
            // Set visibility based on checkbox selection
            kelvin_renderer.visible = checkbox.active.includes(0); // Kelvin is label 0
            wmrg_renderer.visible = checkbox.active.includes(1); // WMRG is label 1
            r1_renderer.visible = checkbox.active.includes(2); // R1 is label 2
            r2_renderer.visible = checkbox.active.includes(3); // R2 is label 3
        """)

        # Attach the callback to checkbox group
        checkbox_group.js_on_change('active', checkbox_callback)

        # JavaScript callback for the "Clear All" button
        clear_button_callback = CustomJS(args=dict(checkbox=checkbox_group), code="""
            // Clear all active checkboxes
            checkbox.active = [];
            checkbox.change.emit();
        """)
        # Link clear button with its callback
        clear_button.js_on_click(clear_button_callback)

        spacer = Spacer(width=50)  # Adjust width for desired space
        layout = column(row(spacer, checkbox_group, clear_button), plot)

        # Set output HTML file
        output_file(html_file)

        # Save the plot layout as a static HTML file
        save(layout)
        logger.info('Plotted %s', html_file)

    def bokeh_plot_forecast_ensemble_probability_multiwave(self, date):
        model = self.config_values['model']
        mem_labels = [f'{fc:03d}' for fc in range(0, 17)]

        date_label = date.strftime('%Y%m%d_%H')
        outfile_dir = self.config_values[f'{model}_forecast_processed_dir']

        html_file_dir = os.path.join(self.config_values[f'{model}_plot_ens'], date_label)
        if not os.path.exists(html_file_dir):
            os.makedirs(html_file_dir)

        precip_files = [
            os.path.join(outfile_dir, f'precipitation_amount_combined_{date_label}Z_{mem}.nc')
            for mem in mem_labels
        ]
        precip_files = [file for file in precip_files if os.path.exists(file)]
        if not precip_files:
            logger.error('No precipitation files found in %s for %s', outfile_dir, date_label)
            return

        pr_cube = self.read_compute_ensemble_prob(precip_files, wname='Precip')
        if pr_cube is None:
            logger.error('Unable to compute precipitation probability for %s', date_label)
            return

        ntimes = len(self.times2plot)
        data = [(i, l, d) for i, l, d in zip(range(ntimes), self.times2plot, self.create_dates_dt(pr_cube))]
        df = pd.DataFrame(data, columns=['Index', 'Lead', 'Date'])

        shade_cbar_title = f"Probability of Precip >= {self.thresholds['Precip']}"
        precip_prob = self.read_compute_ensemble_prob(
            precip_files,
            wname='Precip',
            contour_cbar_title=shade_cbar_title,
        )
        if precip_prob is None:
            logger.error('Unable to compute precipitation probability for %s', date_label)
            return

        for pressure_level in self.pressures:
            wname = 'Kelvin'
            wave_files = [
                os.path.join(outfile_dir, f'div_wave_{wname}_{date_label}Z_{mem}.nc')
                for mem in mem_labels
            ]
            contour_cbar_title = (
                f"Probability of {wname} divergence <= {self.thresholds[f'{wname}_{pressure_level}']:0.1e} s-1"
            )
            kelvin_prob = self.read_compute_ensemble_prob(
                wave_files,
                wname=wname,
                pressure_level=pressure_level,
                contour_cbar_title=contour_cbar_title,
            )

            wname = 'WMRG'
            wave_files = [
                os.path.join(outfile_dir, f'div_wave_{wname}_{date_label}Z_{mem}.nc')
                for mem in mem_labels
            ]
            contour_cbar_title = (
                f"Probability of {wname} divergence <= {self.thresholds[f'{wname}_{pressure_level}']:0.1e} s-1"
            )
            wmrg_prob = self.read_compute_ensemble_prob(
                wave_files,
                wname=wname,
                pressure_level=pressure_level,
                contour_cbar_title=contour_cbar_title,
            )

            wname = 'R1'
            wave_files = [
                os.path.join(outfile_dir, f'vort_wave_{wname}_{date_label}Z_{mem}.nc')
                for mem in mem_labels
            ]
            contour_cbar_title = (
                f"Probability of {wname} vorticity >= {self.thresholds[f'{wname}_{pressure_level}']:0.1e} s-1"
            )
            r1_prob = self.read_compute_ensemble_prob(
                wave_files,
                wname=wname,
                pressure_level=pressure_level,
                contour_cbar_title=contour_cbar_title,
            )

            wname = 'R2'
            wave_files = [
                os.path.join(outfile_dir, f'vort_wave_{wname}_{date_label}Z_{mem}.nc')
                for mem in mem_labels
            ]
            contour_cbar_title = (
                f"Probability of {wname} vorticity >= {self.thresholds[f'{wname}_{pressure_level}']:0.1e} s-1"
            )
            r2_prob = self.read_compute_ensemble_prob(
                wave_files,
                wname=wname,
                pressure_level=pressure_level,
                contour_cbar_title=contour_cbar_title,
            )

            if any(prob is None for prob in [kelvin_prob, wmrg_prob, r1_prob, r2_prob]):
                logger.warning('Skipping pressure %s due to missing wave probability inputs.', pressure_level)
                continue

            for lead in self.times2plot:
                wave_timestep_dic = {}

                t = df.loc[df['Lead'] == lead].Index.values[0]
                datetime_string = df['Date'].loc[df['Lead'] == lead].astype('O').tolist()[0].strftime('%Y/%m/%d %HZ')

                wave_timestep_dic['Precip'] = precip_prob[t].data
                wave_timestep_dic[f'Kelvin_{pressure_level}'] = kelvin_prob[t].data
                wave_timestep_dic[f'WMRG_{pressure_level}'] = wmrg_prob[t].data
                wave_timestep_dic[f'R1_{pressure_level}'] = r1_prob[t].data
                wave_timestep_dic[f'R2_{pressure_level}'] = r2_prob[t].data

                wave_timestep_dic['latitude'] = precip_prob[t].coord('latitude').points
                wave_timestep_dic['longitude'] = precip_prob[t].coord('longitude').points

                if int(lead) < 0:
                    figure_tite = f'Valid on {datetime_string} at T{lead}'
                else:
                    figure_tite = f'Valid on {datetime_string} at T+{lead}'

                html_file = os.path.join(
                    html_file_dir,
                    f'AllWaves_{pressure_level}_EnsProb_{date_label}Z_T{lead}h.html',
                )

                self.bokeh_plot_allwaves2html(
                    wave_timestep_dic,
                    pressure_level,
                    figure_tite=figure_tite,
                    shade_cbar_title=shade_cbar_title,
                    contour_cbar_title=None,
                    html_file=html_file,
                )

        json_file = os.path.join(self.config_values[f'{model}_plot_ens'], f'{model}_eqw_ens_plot_dates.json')
        self.write_dates_json(date, json_file)

    def bokeh_plot_forecast_ensemble_probability(self, date):
        model = self.config_values['model']
        mem_labels = [f'{fc:03d}' for fc in range(0, 18)]

        date_label = date.strftime('%Y%m%d_%H')
        outfile_dir = self.config_values[f'{model}_forecast_processed_dir']

        html_file_dir = os.path.join(self.config_values[f'{model}_plot_ens'], date_label)
        os.makedirs(html_file_dir, exist_ok=True)

        precip_files = [
            os.path.join(outfile_dir, f'precipitation_amount_combined_{date_label}Z_{mem}.nc')
            for mem in mem_labels
        ]
        print(precip_files)
        precip_files = [file for file in precip_files if os.path.exists(file)]
        if not precip_files:
            logger.error('No precipitation files found in %s for %s', outfile_dir, date_label)
            return

        pr_cube = iris.load_cube(precip_files)
        ntimes = len(self.times2plot)
        pr_cube = pr_cube[:, -ntimes:]

        data = [(i, l, d) for i, l, d in zip(range(ntimes), self.times2plot, self.create_dates_dt(pr_cube))]
        df = pd.DataFrame(data, columns=['Index', 'Lead', 'Date'])

        shade_var = pr_cube.collapsed(
            'realization',
            iris.analysis.PROPORTION,
            function=lambda values: values > self.thresholds['precip'],
        )
        shade_cbar_title = f"Probability of precipitation >= {self.thresholds['precip']} mm day-1"

        for wname in self.wave_names:
            for pressure_level in self.pressures:
                if wname in ['Kelvin', 'WMRG']:
                    wave_files = [
                        os.path.join(outfile_dir, f'div_wave_{wname}_{date_label}Z_{mem}.nc')
                        for mem in mem_labels
                    ]
                    wave_files = [file for file in wave_files if os.path.exists(file)]
                    if not wave_files:
                        logger.warning('No divergence files found for %s at %shPa', wname, pressure_level)
                        continue

                    wave_variable = iris.load_cube(wave_files)
                    wave_variable = wave_variable.extract(iris.Constraint(pressure=float(pressure_level)))
                    wave_variable = wave_variable[:, -ntimes:]
                    contour_var = wave_variable.collapsed(
                        'realization',
                        iris.analysis.PROPORTION,
                        function=lambda values: values <= self.thresholds[f'{wname}_{pressure_level}'],
                    )
                    contour_cbar_title = (
                        f"Probability of {wname} divergence <= {self.thresholds[f'{wname}_{pressure_level}']:0.1e} s-1"
                    )
                elif wname in ['R1', 'R2']:
                    wave_files = [
                        os.path.join(outfile_dir, f'vort_wave_{wname}_{date_label}Z_{mem}.nc')
                        for mem in mem_labels
                    ]
                    wave_files = [file for file in wave_files if os.path.exists(file)]
                    if not wave_files:
                        logger.warning('No vorticity files found for %s at %shPa', wname, pressure_level)
                        continue

                    wave_variable = iris.load_cube(wave_files)
                    wave_variable = wave_variable.extract(iris.Constraint(pressure=float(pressure_level)))
                    wave_variable = wave_variable[:, -ntimes:]
                    contour_var = wave_variable.collapsed(
                        'realization',
                        iris.analysis.PROPORTION,
                        function=lambda values: values >= self.thresholds[f'{wname}_{pressure_level}'],
                    )
                    contour_cbar_title = (
                        f"Probability of {wname} vorticity >= {self.thresholds[f'{wname}_{pressure_level}']:0.1e} s-1"
                    )
                else:
                    continue

                for lead in self.times2plot:
                    t = df.loc[df['Lead'] == lead].Index.values[0]
                    valid_dt = df['Date'].loc[df['Lead'] == lead].astype('O').tolist()[0]
                    datetime_string = valid_dt.strftime('%Y/%m/%d %HZ')

                    if int(lead) < 0:
                        figure_tite = (
                            f'{shade_cbar_title};  {contour_cbar_title} '
                            f'\nAnalysis Valid time: {datetime_string} at lead: T{lead}'
                        )
                    else:
                        figure_tite = (
                            f'{shade_cbar_title};  {contour_cbar_title} '
                            f'\nForecast Valid time: {datetime_string} at lead: T+{lead}'
                        )

                    html_file = os.path.join(
                        html_file_dir,
                        f'{wname}_{pressure_level}_EnsProb_{date_label}Z_T{lead}h.html',
                    )

                    self.bokeh_plot2html(
                        shade_var=shade_var[t],
                        contour_var=contour_var[t],
                        figure_tite=figure_tite,
                        shade_cbar_title=shade_cbar_title,
                        contour_cbar_title=contour_cbar_title,
                        html_file=html_file,
                    )

        json_file = os.path.join(self.config_values[f'{model}_plot_ens'], f'{model}_eqw_ens_plot_dates.json')
        self.write_dates_json(date, json_file)
