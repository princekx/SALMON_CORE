import os
import logging
import datetime
import sys
import uuid
import subprocess
import concurrent.futures
from typing import Dict, Any, List
import iris
import iris.cube
import iris.coords
import iris.analysis
import iris.analysis.maths
import iris.coord_categorisation
import numpy as np
import pandas as pd
from salmon.core.task import Task
from salmon.utils.moose import MooseClient
from salmon.utils.config import load_global_config
from salmon.utils.cube import regrid_to_obs, load_obs_grid, remove_um_version
import warnings
# Set the global warning filter to ignore all warnings
warnings.simplefilter("ignore")
logger = logging.getLogger(__name__)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_UTILS_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "utils"))
_DEFAULT_QUERY_DIR = os.path.normpath(
    os.path.join(_UTILS_DIR, "query_files")
)
HR_LIST = [0, 12]
FC_TIMES = np.arange(0, 174, 24)

class RetrieveAnalysisData(Task):
    """Task to retrieve previous 201 days of GL-MN analysis data from MOOSE."""
    def run(self):
        date = self.context.date
        num_prev_days = self.config.get('num_prev_days', 201)
        parallel = self.config.get('parallel', True)
        
        global_config = load_global_config()
        analysis_config = global_config.get('analysis', {})
        
        moose_base = analysis_config.get('moose', 'moose:/opfc/atm/global/prods/')
        raw_dir = analysis_config.get('raw', '/tmp/salmon_raw/analysis')
        #query_template = analysis_config.get('query')
        query_template = os.path.join(
                _DEFAULT_QUERY_DIR,
                self.config.get("query", analysis_config.get("query"))
            )
        
        temp_dir = analysis_config.get('temp', '/tmp/salmon_temp')

        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir, exist_ok=True)

        logger.info(f"Retrieving {num_prev_days} days of analysis data for {date}...")
        
        dates_to_retrieve = [date - datetime.timedelta(days=i) for i in range(num_prev_days)]
        
        if parallel:
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(self._retrieve_for_date, d, moose_base, raw_dir, query_template, temp_dir) for d in dates_to_retrieve]
                concurrent.futures.wait(futures)
        else:
            for d in dates_to_retrieve:
                self._retrieve_for_date(d, moose_base, raw_dir, query_template, temp_dir)
        
        logger.info("Analysis data retrieval complete.")

    def _retrieve_for_date(self, date, moose_base, raw_dir, query_template, temp_dir):
        moosedir = os.path.join(moose_base, f"{date.year}.pp")
        hr_list = ['00', '12']
        fct = '003'
        
        date_raw_dir = os.path.join(raw_dir, date.strftime("%Y%m%d"))
        if not os.path.exists(date_raw_dir):
            os.makedirs(date_raw_dir, exist_ok=True)

        moose_client = MooseClient()

        for hr in hr_list:
            outfile = f"qg{hr}T{fct}.pp"
            outpath = os.path.join(date_raw_dir, outfile)
            
            if os.path.exists(outpath) and os.path.getsize(outpath) > 0:
                logger.debug(f"File {outpath} already exists. Skipping.")
                continue

            filemoose = f"prods_op_gl-mn_{date.strftime('%Y%m%d')}_{hr}_{fct}.pp"
            local_query = os.path.join(temp_dir, f"query_{uuid.uuid4()}.query")
            
            moose_client.create_query_file(query_template, local_query, {'fctime': fct, 'filemoose': filemoose})
            success = moose_client.retrieve(local_query, moosedir, outpath)
            
            if os.path.exists(local_query):
                os.remove(local_query)
            
            if not success:
                logger.warning(f"Failed to retrieve {filemoose}")

class CombineAnalysisData(Task):
    """Task to combine 201 days of analysis data into daily means."""
    def run(self):
        date = self.context.date
        num_prev_days = self.config.get('num_prev_days', 201)
        parallel = self.config.get('parallel', True)
        
        global_config = load_global_config()
        analysis_config = global_config.get('analysis', {})
        
        raw_dir = analysis_config.get('raw', '/tmp/salmon_raw/analysis')
        processed_base = analysis_config.get('processed', '/tmp/salmon_processed/analysis')
        
        # Original logic uses 201 days ending at 'date'
        past_dates = [date - datetime.timedelta(days=i) for i in range(num_prev_days)]
        past_dates.sort()
        
        vars_to_process = {
            'olr': {'constraint': iris.Constraint(name='toa_outgoing_longwave_flux'), 'time_coord': 'forecast_period'},
            'u850': {'constraint': iris.Constraint(name='x_wind', pressure=850), 'time_coord': 'time'},
            'u200': {'constraint': iris.Constraint(name='x_wind', pressure=200), 'time_coord': 'time'}
        }

        logger.info(f"Combining {num_prev_days} days of analysis data...")
        
        obs_grid = load_obs_grid(_THIS_DIR)
        print(f"Observation grid loaded with shape: {obs_grid.shape}")
        
        if parallel:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [executor.submit(self._process_variable, var, info, past_dates, raw_dir, processed_base, obs_grid) 
                           for var, info in vars_to_process.items()]
                concurrent.futures.wait(futures)
        else:
            for var, info in vars_to_process.items():
                self._process_variable(var, info, past_dates, raw_dir, processed_base, obs_grid)
        
        logger.info("Analysis data combination complete.")

    def _process_variable(self, varname, info, past_dates, raw_dir, processed_base, obs_grid):
        date_str = past_dates[-1].strftime("%Y%m%d")
        var_processed_dir = os.path.join(processed_base, 'mjo', varname)
        print(f"Processing variable {varname} for date {date_str} to {var_processed_dir}")
        
        if not os.path.exists(var_processed_dir):
            os.makedirs(var_processed_dir, exist_ok=True)
            
        outpath = os.path.join(var_processed_dir, f"{varname}_mean_nrt_{date_str}.nc")
        if os.path.exists(outpath):
            logger.info(f"{outpath} already exists. Skipping.")
            return

        fct = "003"
        files_00 = [os.path.join(raw_dir, d.strftime("%Y%m%d"), f"qg00T{fct}.pp") for d in past_dates]
        files_12 = [os.path.join(raw_dir, d.strftime("%Y%m%d"), f"qg12T{fct}.pp") for d in past_dates]

        logger.info(f"Processing {varname}...")
        
        cube_00 = iris.load_cube(files_00, info['constraint'], callback=remove_um_version)
        if len(cube_00.coord('forecast_period').points) > 1:
            cube_00 = cube_00.collapsed(info['time_coord'], iris.analysis.MEAN)
            
        cube_12 = iris.load_cube(files_12, info['constraint'], callback=remove_um_version)
        if len(cube_12.coord('forecast_period').points) > 1:
            cube_12 = cube_12.collapsed(info['time_coord'], iris.analysis.MEAN)
            
        # Average 00 and 12
        cube_00.data = (cube_00.data + cube_12.data) / 2.0
        
        # Regrid
        reg_cube = regrid_to_obs(cube_00, obs_grid)
        
        iris.save(reg_cube, outpath, netcdf_format='NETCDF4_CLASSIC')
        logger.info(f"Saved {outpath}")

class RetrieveMogrepsData(Task):
    """
    Retrieve MOGREPS files needed for MJO processing.

    Notes
    -----
    - Members are split by cycle:
      * 12Z -> 00..17
      * 18Z -> 18..34 plus '00' (mapped to directory member 035)
    - Forecast steps are 24-hourly from 0 to 168 hours.
    """

    def run(self):
        """Task entrypoint."""
        date = self.context.date
        parallel = self.config.get("parallel", True)
        success = self.retrieve_mogreps_data(date=date, parallel=parallel)

        if success:
            logger.info("MOGREPS data retrieval complete.")
        else:
            logger.warning("MOGREPS data retrieval completed with errors.")

    def _init_config_values(self):
        """Load and cache retrieval paths/config used by helper methods."""
        if hasattr(self, "config_values"):
            return
        
        mogreps = load_global_config().get("mogreps", {})
        self.config_values = {
            "mogreps_moose_dir": mogreps.get("moose", "moose:/opfc/atm/mogreps-g/prods/"),
            "mogreps_raw_dir": mogreps.get("raw", "/tmp/salmon_raw/mogreps"),
            "mogreps_combined_queryfile": os.path.join(
                _DEFAULT_QUERY_DIR,
                self.config.get("query", mogreps.get("query"))
            ),
            "mogreps_dummy_queryfiles_dir": mogreps.get("temp", "/tmp/salmon_temp"),
        }
        print(f"Config values for retrieval: {self.config_values}")
        os.makedirs(self.config_values["mogreps_dummy_queryfiles_dir"], exist_ok=True)

    def _get_all_members(self, hr):
        """Return 2-digit member IDs for a given model cycle hour."""
        if hr == 12:
            return [f"{mem:02d}" for mem in range(18)]
        if hr == 18:
            return [f"{mem:02d}" for mem in range(18, 35)] + ["00"]
        return []

    def _iter_tasks(self, date):
        """Yield (date, hr, fc, member) combinations to retrieve."""
        for hr in HR_LIST:
            for fc in FC_TIMES:
                for mem in self._get_all_members(hr):
                    yield (date, hr, fc, mem)

    def _resolve_moose_dir(self, date):
        """
        Resolve source MOOSE directory.

        Historical OS47 fallback logic is intentionally left disabled.
        """
        return os.path.join(self.config_values["mogreps_moose_dir"], f"{date:%Y%m}.pp")

    def _map_member_to_dir(self, hr, digit2_mem):
        """Map 2-digit member ID to 3-digit on-disk directory name."""
        return "035" if (hr == 18 and digit2_mem == "00") else f"{int(digit2_mem):03d}"

    def _retrieve_fc_data(self, date, hr, fc, digit2_mem):
        """Retrieve one forecast file for one member/cycle."""
        self._init_config_values()

        moose_dir = self._resolve_moose_dir(date)
        digit3_mem = self._map_member_to_dir(hr, digit2_mem)
        fct = f"{fc:03d}"

        remote_data_dir = os.path.join(
            self.config_values["mogreps_raw_dir"], date.strftime("%Y%m%d"), digit3_mem
        )
        os.makedirs(remote_data_dir, exist_ok=True)

        filemoose = f"prods_op_mogreps-g_{date:%Y%m%d}_{hr}_{digit2_mem}_{fct}.pp"
        outfile = f"englaa_pd{fct}.pp"
        outfile_path = os.path.join(remote_data_dir, outfile)

        # Remove corrupt empty file before retry
        if os.path.exists(outfile_path) and os.path.getsize(outfile_path) == 0:
            logger.warning("Deleting empty file: %s", outfile_path)
            os.remove(outfile_path)

        if os.path.exists(outfile_path):
            logger.debug("%s exists. Skip.", outfile_path)
            return True

        local_query_file = os.path.join(
            self.config_values["mogreps_dummy_queryfiles_dir"], f"localquery_{uuid.uuid1()}"
        )

        try:
            self.create_query_file(local_query_file, filemoose, fct)

            command = [
                "/opt/moose-client-wrapper/bin/moo",
                "select",
                "--fill-gaps",
                local_query_file,
                moose_dir,
                outfile_path,
            ]
            logger.info("Executing command: %s", " ".join(command))
            subprocess.run(command, check=True)
            return True

        except subprocess.CalledProcessError as exc:
            logger.error("Data retrieval failed for %s: %s", outfile_path, exc)
            return False
        except Exception as exc:
            logger.error("Unexpected retrieval error for %s: %s", outfile_path, exc)
            return False
        finally:
            if os.path.exists(local_query_file):
                os.remove(local_query_file)

    # Backward-compatible method name used elsewhere in code.
    def retrieve_fc_data_parallel(self, date, hr, fc, digit2_mem):
        """Compatibility wrapper around the streamlined retrieval function."""
        return self._retrieve_fc_data(date, hr, fc, digit2_mem)

    # Backward-compatible signature used by old run() style.
    def _retrieve_for_member(self, date, hr, fc, digit2_mem, *_unused):
        """Compatibility wrapper for legacy tuple-based submit arguments."""
        return self._retrieve_fc_data(date, hr, fc, digit2_mem)

    def check_if_all_data_exist(self, date, hr, fc, digit2_mem):
        """Check if one retrieved file exists and is non-empty."""
        self._init_config_values()
        digit3_mem = self._map_member_to_dir(hr, digit2_mem)
        outfile_path = os.path.join(
            self.config_values["mogreps_raw_dir"],
            date.strftime("%Y%m%d"),
            digit3_mem,
            f"englaa_pd{fc:03d}.pp",
        )
        return os.path.exists(outfile_path) and os.path.getsize(outfile_path) > 0

    def create_query_file(self, local_query_file1, filemoose, fct):
        """Create a per-request query file from the configured template."""
        self._init_config_values()
        query_file = self.config_values["mogreps_combined_queryfile"]
        replacements = {"fctime": fct, "filemoose": filemoose}

        with open(query_file) as query_infile, open(local_query_file1, "w") as query_outfile:
            for line in query_infile:
                for src, target in replacements.items():
                    line = line.replace(src, target)
                query_outfile.write(line)

    def retrieve_mogreps_data(self, date, parallel=True):
        """
        Retrieve all required MOGREPS files for one date.

        Returns
        -------
        bool
            True if all retrieval jobs succeeded, else False.
        """
        self._init_config_values()
        logger.info("Retrieving MOGREPS data for Cold Surge on %s", date)

        tasks = list(self._iter_tasks(date))

        if not parallel:
            return all(self._retrieve_fc_data(*task) for task in tasks)

        max_workers = int(self.config.get("max_workers", 10))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._retrieve_fc_data, *task) for task in tasks]
            return all(f.result() for f in concurrent.futures.as_completed(futures))
        
class ComputeMJOIndices(Task):
    """Task to compute MJO (RMM) indices from combined data."""
    def __init__(self, context, config):
        super().__init__(context, config)
        self.num_prev_days = 201
        self.nanalysis2write = 40
        self.nforecasts = 7 # MOGREPS specific
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data')
        self.obs_grid = load_obs_grid(_THIS_DIR)
        
    def run(self):
        date = self.context.date
        model = self.context.get_config('model', 'mogreps')
        parallel = self.config.get('parallel', True)
        global_config = load_global_config()
        analysis_config = global_config.get('analysis', {})
        model_config = global_config.get(model, {})
        
        analysis_processed_dir = analysis_config.get('processed', '/tmp/salmon_processed/analysis')
        model_processed_dir = model_config.get('processed', f'/tmp/salmon_processed/{model}')
        archive_base = model_config.get('archive', f'/tmp/salmon_archive/{model}')
        
        recipe_processed_dir = os.path.join(model_processed_dir, 'mjo')
        archive_dir = os.path.join(archive_base, 'mjo', 'rmms', date.strftime("%Y%m%d"))
        
        if not os.path.exists(recipe_processed_dir):
            os.makedirs(recipe_processed_dir, exist_ok=True)
        if not os.path.exists(archive_dir):
            os.makedirs(archive_dir, exist_ok=True)

        logger.info(f"Computing MJO indices for {date} (Model: {model})...")
        
        members = self._get_all_members(date)
        
        # Load analysis cubes once
        analysis_cubes = self._load_analysis_cubes(date, analysis_processed_dir)
        print(f"Loaded analysis cubes for variables: {list(analysis_cubes.keys())}")

        if parallel:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [executor.submit(self._process_member, mem, date, model, model_config, 
                                           analysis_cubes, recipe_processed_dir, archive_dir) 
                           for mem in members]
                concurrent.futures.wait(futures)
        else:
            for mem in members:
                self._process_member(mem, date, model, model_config, analysis_cubes, recipe_processed_dir, archive_dir)
            
        logger.info(f"MJO indices computed and archived to {archive_dir}")

    def _get_all_members(self, date):
        # Using 00Z and 12Z members for MOGREPS
        return [f"{m:03d}" for m in range(18)] # Simplified for now, can be expanded

    def _load_analysis_cubes(self, date, analysis_processed_dir):
        cubes = {}
        for var in ['olr', 'u850', 'u200']:
            path = os.path.join(analysis_processed_dir, 'mjo', var, f"{var}_mean_nrt_{date.strftime('%Y%m%d')}.nc")
            print(f"Loading analysis cube for {var} from {path}")
            cubes[var] = iris.load_cube(path)
        return cubes

    def _process_member(self, mem, date, model, model_config, analysis_cubes, processed_dir, archive_dir):
        logger.info(f"Processing member {mem}...")
        
        raw_dir = model_config.get('raw', '/tmp/salmon_raw/mogreps')
        fc_times = [f"{fct:03d}" for fct in np.arange(24, 174, 24)]
        
        # 1. Combine Analysis and Forecast per variable
        anom_120_filenames = {}
        for varname in ['olr', 'u850', 'u200']:
            var_dir = os.path.join(processed_dir, 'mjo', varname)
            os.makedirs(var_dir, exist_ok=True)
            
            concated_file = os.path.join(var_dir, f"{varname}_concat_nrt_{date.strftime('%Y%m%d')}_{mem}.nc")
            sm120d_outfile = os.path.join(var_dir, f"{varname}_120dm_40sn_nrt_{date.strftime('%Y%m%d')}_{mem}.nc")
            anom_120_filenames[varname] = sm120d_outfile

            if not os.path.exists(sm120d_outfile):
                if not os.path.exists(concated_file):
                    from salmon.utils.cube import read_olr_correctly, read_winds_correctly
                    mog_files = [os.path.join(raw_dir, date.strftime("%Y%m%d"), mem, f"englaa_pd{fct}.pp") for fct in fc_times]
                    
                    if varname == 'olr':
                        fc_cube = read_olr_correctly(mog_files)
                    elif varname == 'u850':
                        fc_cube = read_winds_correctly(mog_files, 'x_wind', pressure_level=850)
                    elif varname == 'u200':
                        fc_cube = read_winds_correctly(mog_files, 'x_wind', pressure_level=200)
                    
                    # regrid to analysis grid which is the same as obs grid
                    fc_cube = regrid_to_obs(fc_cube, self.obs_grid)
                    # Concatenate analysis and forecast cubes
                    cat_cube = self._concat_analysis_fcast(analysis_cubes[varname], fc_cube)
                    # Save concatenated cube for potential debugging/reuse
                    iris.save(cat_cube, concated_file, netcdf_format='NETCDF4_CLASSIC')
                else:
                    cat_cube = iris.load_cube(concated_file)

                # 2. Scientific Pre-processing
                harfile = os.path.join(self.data_dir, f"{varname}.nr.79to01.m+3har.nc")
                cube = self._remove3har(cat_cube, harfile)
                cube = cube.extract(self._region([0, -40, 360, 40]))
                cube = self._remRunMean(cube)
                
                # Subset to 40 days analysis + forecast
                cube = cube[-(self.nanalysis2write + self.nforecasts):]
                iris.save(cube, sm120d_outfile, netcdf_format="NETCDF3_CLASSIC")

        # 3. Compute RMMs
        rmm_file = os.path.join(archive_dir, f"createdPCs.15sn.{date.strftime('%Y%m%d')}.fcast.{mem}.txt")
        self._compute_rmms(anom_120_filenames, rmm_file)

    def _concat_analysis_fcast(self, analysis_cube, fcast_cube):
        # Adapted from production logic
        nfcast_days = fcast_cube.shape[0]
        time_unit = analysis_cube.coord('time').units
        ntime_analysis = analysis_cube.shape[0]
        
        cat_cube_list = iris.cube.CubeList([analysis_cube[0].copy() for _ in range(ntime_analysis + nfcast_days)])
        for n, cube in enumerate(cat_cube_list):
            cube.add_aux_coord(iris.coords.AuxCoord(analysis_cube.coord('time').points[0] + n*24,
                                                    long_name='forecast_time', units=time_unit))
        
        cat_cube = cat_cube_list.merge_cube()
        cat_cube.data = np.concatenate((analysis_cube.data, fcast_cube.data))
        return cat_cube

    def _remove3har(self, cube, harfile):
        mm = iris.load_cube(harfile, 'mm')
        aa = iris.load_cube(harfile, 'aa')
        bb = iris.load_cube(harfile, 'bb')
        
        if 'julian_day' not in [c.name() for c in cube.coords()]:
            iris.coord_categorisation.add_day_of_year(cube, 'time', name='julian_day')
            
        anom = cube.copy()
        for i, t in enumerate(cube.coord('julian_day').points):
            seasonal_cycle = (mm.data + 
                aa.data[0] * np.cos(2*np.pi*(t-1)/365.0) + bb.data[0] * np.sin(2*np.pi*(t-1)/365.0) +
                aa.data[1] * np.cos(2*np.pi*(t-1)*2/365.0) + bb.data[1] * np.sin(2*np.pi*(t-1)*2/365.0) +
                aa.data[2] * np.cos(2*np.pi*(t-1)*3/365.0) + bb.data[2] * np.sin(2*np.pi*(t-1)*3/365.0))
            anom.data[i] = cube.data[i] - seasonal_cycle
        return anom

    def _region(self, coords):
        return iris.Constraint(longitude=lambda c: coords[0] <= c <= coords[2],
                               latitude=lambda c: coords[1] <= c <= coords[3])

    def _remRunMean(self, cube, NN=120):
        ntime = cube.shape[0]
        runmean = cube.copy()
        for n in range(1, ntime):
            if n < NN:
                runmean.data[n] = np.mean(cube.data[:n+1], axis=0)
            else:
                runmean.data[n] = np.mean(cube.data[n-NN:n+1], axis=0)
        return cube - runmean

    def _compute_rmms(self, anom_120_filenames, rmm_file_name):
        eigenfile = os.path.join(self.data_dir, 'olr+u850+u200.anom-sst1-120dm.79-.15snAv.ASCII')
        
        with open(eigenfile, 'r') as f:
            lines = f.readlines()
        
        # Parse ASCII eigenfile (simplified based on production logic)
        header = list(map(int, lines[1].split()))
        nspace = header[3]
        num = header[4]
        nsp1, nsp2, nsp3 = header[6:9]
        
        eigval = np.array([float(x.strip()) for x in lines[13:13+nspace]])
        eigvec = np.array([float(x.strip()) for x in lines[446:446+nspace*num]])
        eigvec = np.transpose(eigvec.reshape(num, nspace), axes=(1, 0))
        norm = np.array([float(x.strip()) for x in lines[9087:9087+nspace]])
        databar = np.array([float(x.strip()) for x in lines[9520:9520+nspace]])

        # Load first cube for dimensions
        sample_cube = iris.load_cube(anom_120_filenames['olr'])
        sample_cube = sample_cube.extract(self._region([0, -15, 357.5, 15]))
        sample_cube = sample_cube.collapsed(['latitude'], iris.analysis.MEAN)
        ntime, mlon = sample_cube.shape
        
        cdata = np.zeros((ntime, 3 * mlon))
        for n, var in enumerate(['olr', 'u850', 'u200']):
            cube = iris.load_cube(anom_120_filenames[var])
            cube = cube.extract(self._region([0, -15, 357.5, 15]))
            cube = cube.collapsed(['latitude'], iris.analysis.MEAN)
            cdata[:, n*mlon:(n+1)*mlon] = cube.data
            
        rmm1s, rmm2s, phases, amps = [], [], [], []
        phi = np.array([202.5, 247.5, 292.5, 337.5, 22.5, 67.5, 112.5, 157.5])
        
        for t in range(ntime):
            datmat = (cdata[t, :] - databar) / norm
            pc = (np.dot(np.transpose(eigvec[0:nsp1, 0:2]), datmat[0:nsp1]) +
                  np.dot(np.transpose(eigvec[nsp1:nsp2, 0:2]), datmat[nsp1:nsp2]) +
                  np.dot(np.transpose(eigvec[nsp2:nsp3, 0:2]), datmat[nsp2:nsp3]))
            pc = pc / np.sqrt(eigval[0:2])
            
            ag = (360. / (2. * np.pi)) * np.arctan2(pc[1], pc[0])
            if ag < 0: ag += 360.
            if ag >= 360.: ag -= 360.
            
            pa = 0
            for ipa in range(8):
                phiminus, phiplus = phi[ipa] - 22.5, phi[ipa] + 22.5
                if phiminus < 0: phiminus += 360.
                if phiplus >= 360.: phiplus -= 360.
                
                if phiminus < phiplus:
                    if ag >= phiminus and ag < phiplus: pa = ipa + 1
                else:
                    if ag >= phiminus or ag < phiplus: pa = ipa + 1
            
            rmm1s.append(pc[0])
            rmm2s.append(pc[1])
            phases.append(pa)
            amps.append(np.sqrt(pc[0]**2 + pc[1]**2))

        # Save to CSV using pandas
        df = pd.DataFrame({
            'year': sample_cube.coord('year').points if 'year' in [c.name() for c in sample_cube.coords()] else 0,
            'rmm1': rmm1s, 'rmm2': rmm2s, 'phase': phases, 'amp': amps
        })
        df['label'] = ['analysis'] * (ntime - self.nforecasts) + ['forecast'] * self.nforecasts
        df.to_csv(rmm_file_name, index=False)
        logger.info(f"Saved RMMs to {rmm_file_name}")
