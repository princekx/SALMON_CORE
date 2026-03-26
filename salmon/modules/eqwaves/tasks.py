import os
import logging
import datetime
import uuid
import concurrent.futures
import warnings
import cf_units
import numpy as np
import iris
import iris.coords

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

        # Backward-compatible precedence:
        # task.analysis_query -> analysis.query -> task.query -> analysis_combined.query
        # task.forecast_query -> mogreps.query -> task.query -> mogreps_combined.query
        shared_query = self.config.get('query')
        analysis_query = self.config.get('analysis_query') or analysis_cfg.get('query') or shared_query or 'analysis_combined.query'
        forecast_query = self.config.get('forecast_query') or mogreps_cfg.get('query') or shared_query or 'mogreps_combined.query'

        def _resolve_query_path(value):
            if not value:
                return None

            expanded = os.path.expandvars(os.path.expanduser(str(value)))
            if os.path.isabs(expanded):
                return expanded

            # If caller passed a project-relative file, keep that first.
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
            'model_raw_dir': model_cfg.get('raw', '/tmp/salmon_raw/mogreps'),
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
