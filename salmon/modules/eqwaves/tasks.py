import os
import logging
import datetime
import uuid
import concurrent.futures
from typing import Dict, Any, List
import numpy as np
import iris
import iris.coords
import iris.util
from salmon.core.task import Task
from salmon.utils.moose import MooseClient
from salmon.utils.config import load_global_config
from salmon.utils.cube import create_latlon_grid, remove_um_version
from .wave_processor import WaveProcessor

logger = logging.getLogger(__name__)

class RetrieveEqWavesData(Task):
    """Task to retrieve data required for Equatorial Wave analysis."""
    def run(self):
        date = self.context.date
        parallel = self.config.get('parallel', True)
        
        global_config = load_global_config()
        analysis_config = global_config.get('analysis', {})
        mogreps_config = global_config.get('mogreps', {})
        
        # 1. Retrieve Analysis Data (332 points every 6 hours)
        self._retrieve_analysis(date, analysis_config, parallel)
        
        # 2. Retrieve MOGREPS Forecast Data (every 6 hours)
        self._retrieve_forecast(date, mogreps_config, parallel)
        
        logger.info("EqWaves data retrieval complete.")

    def _retrieve_analysis(self, date, config, parallel):
        moose_base = config.get('moose', 'moose:/opfc/atm/global/prods/')
        raw_dir = config.get('raw', '/tmp/salmon_raw/analysis')
        query_template = config.get('query')
        temp_dir = config.get('temp', '/tmp/salmon_temp')
        
        ntimes_analysis = 332
        analysis_dates = [date - datetime.timedelta(hours=i * 6) for i in range(ntimes_analysis)]
        
        logger.info(f"Retrieving {ntimes_analysis} analysis steps for EqWaves...")
        
        if parallel:
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(self._retrieve_analysis_step, d, moose_base, raw_dir, query_template, temp_dir) 
                           for d in analysis_dates]
                concurrent.futures.wait(futures)
        else:
            for d in analysis_dates:
                self._retrieve_analysis_step(d, moose_base, raw_dir, query_template, temp_dir)

    def _retrieve_analysis_step(self, date, moose_base, raw_dir, query_template, temp_dir):
        moosedir = os.path.join(moose_base, f"{date.year}.pp")
        hr = date.strftime('%H')
        fct = '000'
        
        date_raw_dir = os.path.join(raw_dir, date.strftime("%Y%m%d"))
        os.makedirs(date_raw_dir, exist_ok=True)
        
        outfile = f"qg{hr}T{fct}.pp"
        outpath = os.path.join(date_raw_dir, outfile)
        
        if os.path.exists(outpath) and os.path.getsize(outpath) > 0:
            return

        filemoose = f"prods_op_gl-mn_{date.strftime('%Y%m%d')}_{hr}_{fct}.pp"
        moose_client = MooseClient()
        local_query = os.path.join(temp_dir, f"eqw_analysis_query_{uuid.uuid4()}.query")
        
        moose_client.create_query_file(query_template, local_query, {'fctime': fct, 'filemoose': filemoose})
        moose_client.retrieve(local_query, moosedir, outpath)
        
        if os.path.exists(local_query):
            os.remove(local_query)

    def _retrieve_forecast(self, date, config, parallel):
        moose_base = config.get('moose', 'moose:/opfc/atm/mogreps-g/prods/')
        raw_dir = config.get('raw', '/tmp/salmon_raw/mogreps')
        query_template = config.get('query')
        temp_dir = config.get('temp', '/tmp/salmon_temp')
        
        str_hr = date.strftime('%H')
        members, mem_labels = self._generate_members(date)
        fc_times = np.arange(6, 174, 6)
        
        logger.info(f"Retrieving MOGREPS forecast for EqWaves ({len(members)} members)...")
        
        tasks = []
        for i, mem in enumerate(members):
            for fc in fc_times:
                tasks.append((date, mem, mem_labels[i], fc, moose_base, raw_dir, query_template, temp_dir))
        
        if parallel:
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(self._retrieve_forecast_step, *t) for t in tasks]
                concurrent.futures.wait(futures)
        else:
            for t in tasks:
                self._retrieve_forecast_step(*t)

    def _retrieve_forecast_step(self, date, mem, mem_label, fc, moose_base, raw_dir, query_template, temp_dir):
        os47_date = datetime.datetime(2026, 1, 21)
        if date > os47_date:
            moosedir = os.path.join(moose_base, f"{date.year}.pp")
        else:
            moosedir = os.path.join(moose_base, f"{date.strftime('%Y%m')}.pp")
            
        str_hr = date.strftime('%H')
        mem_dir = os.path.join(raw_dir, date.strftime("%Y%m%d"), str_hr, mem_label)
        os.makedirs(mem_dir, exist_ok=True)
        
        fct = f"{fc:03d}"
        outfile = f"qg{str_hr}T{fct}.pp"
        outpath = os.path.join(mem_dir, outfile)
        
        if os.path.exists(outpath) and os.path.getsize(outpath) > 0:
            return
            
        filemoose = f"prods_op_mogreps-g_{date.strftime('%Y%m%d')}_{str_hr}_{mem}_{fct}.pp"
        moose_client = MooseClient()
        local_query = os.path.join(temp_dir, f"eqw_forecast_query_{uuid.uuid4()}.query")
        
        moose_client.create_query_file(query_template, local_query, {'fctime': fct, 'filemoose': filemoose})
        moose_client.retrieve(local_query, moosedir, outpath)
        
        if os.path.exists(local_query):
            os.remove(local_query)

    def _generate_members(self, date):
        str_hr = date.strftime('%H')
        members_tuple = {'00': ['00'] + [f'{fc:02}' for fc in range(1, 18)],
                         '06': ['00'] + [f'{fc:02}' for fc in range(18, 35)],
                         '12': ['00'] + [f'{fc:02}' for fc in range(1, 18)],
                         '18': ['00'] + [f'{fc:02}' for fc in range(18, 35)]}
        members = members_tuple[str_hr]
        mem_labels = [f'{fc:03}' for fc in range(0, 18)]
        return members, mem_labels

class ComputeEqWavesIndices(Task):
    """Task to compute Equatorial Wave indices using 2D FFT and projection."""
    def run(self):
        date = self.context.date
        model = self.context.get_config('model', 'mogreps')
        
        global_config = load_global_config()
        analysis_config = global_config.get('analysis', {})
        model_config = global_config.get(model, {})
        
        analysis_raw = analysis_config.get('raw', '/tmp/salmon_raw/analysis')
        model_raw = model_config.get('raw', '/tmp/salmon_raw/mogreps')
        
        proc_base = model_config.get('processed', f'/tmp/salmon_processed/{model}')
        output_dir = os.path.join(proc_base, 'eqwaves', self.context.recipe_name, date.strftime('%Y%m%d_%H'))
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Computing EqWaves indices for {date}...")
        
        ref_grid = create_latlon_grid().intersection(latitude=(-24, 24))
        processor = WaveProcessor()
        
        # 1. Process Analysis Data if not already done
        analysis_processed_dir = os.path.join(analysis_config.get('processed', '/tmp/salmon_processed/analysis'), 'eqwaves')
        os.makedirs(analysis_processed_dir, exist_ok=True)
        
        date_label = date.strftime('%Y%m%d_%H')
        analysis_dates = sorted([date - datetime.timedelta(hours=i * 6) for i in range(332)])
        
        for var in ['x_wind', 'y_wind', 'geopotential_height', 'precipitation_amount']:
            an_file = os.path.join(analysis_processed_dir, f"{var}_analysis_{date_label}.nc")
            if not os.path.exists(an_file):
                logger.info(f"Processing analysis history for {var}...")
                an_raw_files = [os.path.join(analysis_raw, d.strftime("%Y%m%d"), f"qg{d.strftime('%H')}T000.pp") 
                                for d in analysis_dates]
                existing_raw = [f for f in an_raw_files if os.path.exists(f)]
                if len(existing_raw) < 100: # Heuristic
                    logger.error(f"Not enough analysis files found for {var} ({len(existing_raw)}/332)")
                    continue
                
                if var == 'precipitation_amount':
                    cube = iris.load_cube(existing_raw, var, callback=remove_um_version).regrid(ref_grid, iris.analysis.Linear())
                else:
                    cube = iris.load_cube(existing_raw, iris.Constraint(pressure=[200, 850]) & var, 
                                          callback=remove_um_version).regrid(ref_grid, iris.analysis.Linear())
                
                iris.save(cube, an_file)

        # 2. Process Members
        members, mem_labels = self._generate_members(date)
        fc_times = np.arange(6, 174, 6)
        
        for i, mem_label in enumerate(mem_labels):
            logger.info(f"Processing member {mem_label} ({i+1}/{len(mem_labels)})")
            self._process_member(date, members[i], mem_label, fc_times, model_raw, analysis_processed_dir, output_dir, ref_grid, processor)

    def _process_member(self, date, mem, mem_label, fc_times, model_raw, an_proc_dir, out_dir, ref_grid, processor):
        str_hr = date.strftime('%H')
        date_label = date.strftime('%Y%m%d_%H')
        
        # Check if already computed
        if all([os.path.exists(os.path.join(out_dir, f"vort_wave_{wn}_{date_label}Z_{mem_label}.nc")) for wn in processor.wave_names]):
            return

        cubes_to_process = {}
        for var in ['x_wind', 'y_wind', 'geopotential_height', 'precipitation_amount']:
            an_file = os.path.join(an_proc_dir, f"{var}_analysis_{date_label}.nc")
            if not os.path.exists(an_file): continue
            
            an_cube = iris.load_cube(an_file)
            
            fc_raw_files = [os.path.join(model_raw, date.strftime("%Y%m%d"), str_hr, mem_label, f"qg{str_hr}T{fct:03d}.pp") 
                            for fct in fc_times]
            existing_fc = [f for f in fc_raw_files if os.path.exists(f)]
            
            if not existing_fc:
                logger.warning(f"No forecast files for member {mem_label}")
                continue
                
            if var == 'precipitation_amount':
                fc_cube = self._read_forecasts(date, existing_fc, var, fc_times)
                fc_cube.data[1:] -= fc_cube.data[:-1]
                an_cube.data *= 3600. # Convert to mm/6h if needed? Original code does this.
            else:
                fc_cube = self._read_forecasts(date, existing_fc, var, fc_times)
            
            fc_cube = fc_cube.regrid(ref_grid, iris.analysis.Linear())
            
            # Combine
            combined = self._concat_analysis_forecast(date, an_cube, fc_cube)
            realiz_coord = iris.coords.DimCoord([int(mem_label)], standard_name='realization', var_name='realization')
            if combined.coords('realization'): combined.remove_coord('realization')
            combined.add_aux_coord(realiz_coord)
            
            cubes_to_process[var] = combined

        if len(cubes_to_process) < 3: return # Need at least u, v, z

        # Wave computation
        u = cubes_to_process['x_wind']
        v = cubes_to_process['y_wind']
        z = cubes_to_process['geopotential_height']
        
        q, r = processor.uz_to_qr(u.data, z.data)
        qf = np.fft.fft2(q, axes=(0, -1))
        rf = np.fft.fft2(r, axes=(0, -1))
        vf = np.fft.fft2(v.data, axes=(0, -1))
        
        lats = u.coord('latitude').points
        ufw, zfw, vfw = processor.filt_project(qf, rf, vf, lats)
        
        # IFFT
        u_wave = np.real(np.fft.ifft2(ufw, axes=(1, -1)))
        z_wave = np.real(np.fft.ifft2(zfw, axes=(1, -1)))
        v_wave = np.real(np.fft.ifft2(vfw, axes=(1, -1)))
        
        # Create cubes and save
        time_c, press_c, lat_c, lon_c = u.coord('time'), u.coord('pressure'), u.coord('latitude'), u.coord('longitude')
        u_wave_cube = processor.makes_5d_cube(u_wave, time_c, press_c, lat_c, lon_c)
        v_wave_cube = processor.makes_5d_cube(v_wave, time_c, press_c, lat_c, lon_c)
        z_wave_cube = processor.makes_5d_cube(z_wave, time_c, press_c, lat_c, lon_c)
        
        for wn in processor.wave_names:
            idx = u_wave_cube.coord('wave_name').attributes[wn]
            iris.save(u_wave_cube[idx], os.path.join(out_dir, f"u_wave_{wn}_{date_label}Z_{mem_label}.nc"))
            iris.save(v_wave_cube[idx], os.path.join(out_dir, f"v_wave_{wn}_{date_label}Z_{mem_label}.nc"))
            iris.save(z_wave_cube[idx], os.path.join(out_dir, f"z_wave_{wn}_{date_label}Z_{mem_label}.nc"))

        # Vorticity and Divergence
        div = processor.derivative(u_wave_cube, 'longitude').regrid(u_wave_cube, iris.analysis.Linear())
        div += processor.derivative(v_wave_cube, 'latitude').regrid(u_wave_cube, iris.analysis.Linear())
        
        vort = processor.derivative(v_wave_cube, 'longitude').regrid(u_wave_cube, iris.analysis.Linear())
        vort -= processor.derivative(u_wave_cube, 'latitude').regrid(u_wave_cube, iris.analysis.Linear())
        
        for wn in processor.wave_names:
            idx = u_wave_cube.coord('wave_name').attributes[wn]
            iris.save(div[idx], os.path.join(out_dir, f"div_wave_{wn}_{date_label}Z_{mem_label}.nc"))
            iris.save(vort[idx], os.path.join(out_dir, f"vort_wave_{wn}_{date_label}Z_{mem_label}.nc"))

    def _read_forecasts(self, date, files, var, fc_times):
        cubes = []
        for i, f in enumerate(files):
            if var == 'precipitation_amount':
                c = iris.load_cube(f, var)
                if len(c.shape) == 3: c = c.collapsed('time', iris.analysis.MEAN)
            else:
                c = iris.load_cube(f, iris.Constraint(pressure=[200, 850]) & var)
                if len(c.shape) == 4: c = c.collapsed('time', iris.analysis.MEAN)
            
            # Make mergable
            c.cell_methods = ()
            for coord in ["forecast_period", "time"]:
                if c.coords(coord): c.remove_coord(coord)
            
            unit_str = f"hours since {date.strftime('%Y-%m-%d %H:00:00')}"
            c.add_aux_coord(iris.coords.AuxCoord(fc_times[i], standard_name="forecast_period", units=unit_str))
            c.add_aux_coord(iris.coords.AuxCoord(fc_times[i], standard_name="time", units=unit_str))
            cubes.append(c)
        return iris.cube.CubeList(cubes).merge_cube()

    def _concat_analysis_forecast(self, date, an_cube, fc_cube):
        data = np.concatenate((an_cube.data, fc_cube.data), axis=0)
        
        an_times = [date + datetime.timedelta(hours=(i+1)*6) for i in range(-332, 0)]
        fc_times = [date + datetime.timedelta(hours=(i+1)*6) for i in range(0, 28)]
        all_dates = an_times + fc_times
        
        import cf_units
        time_units = cf_units.Unit('hours since 1970-01-01 00:00:00', calendar='gregorian')
        time_values = time_units.date2num(all_dates)
        time_coord = iris.coords.DimCoord(time_values, standard_name='time', units=time_units)
        
        dims = [(time_coord, 0)]
        for coord in an_cube.dim_coords[1:]:
            dims.append((coord, an_cube.coord_dims(coord)[0]))
            
        return iris.cube.Cube(data, long_name=fc_cube.long_name, units=fc_cube.units, dim_coords_and_dims=dims)

    def _generate_members(self, date):
        str_hr = date.strftime('%H')
        members_tuple = {'00': ['00'] + [f'{fc:02}' for fc in range(1, 18)],
                         '06': ['00'] + [f'{fc:02}' for fc in range(18, 35)],
                         '12': ['00'] + [f'{fc:02}' for fc in range(1, 18)],
                         '18': ['00'] + [f'{fc:02}' for fc in range(18, 35)]}
        members = members_tuple[str_hr]
        mem_labels = [f'{fc:03}' for fc in range(0, 18)]
        return members, mem_labels
