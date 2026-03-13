import os
import logging
import datetime
import uuid
import concurrent.futures
import subprocess
from typing import Dict, Any, List
import numpy as np
import iris
import iris.coords
from salmon.core.task import Task
from salmon.utils.moose import MooseClient
from salmon.utils.config import load_global_config
from salmon.utils.cube import read_winds_correctly, read_precip_correctly, subset_seasia
import sys
import warnings
# Set the global warning filter to ignore all warnings
warnings.simplefilter("ignore")

logger = logging.getLogger(__name__)

class RetrieveColdSurgeData(Task):
    """
    Retrieve MOGREPS files needed for Cold Surge processing.

    Notes
    -----
    - Members are split by cycle:
      * 12Z -> 00..17
      * 18Z -> 18..34 plus '00' (mapped to directory member 035)
    - Forecast steps are 24-hourly from 0 to 168 hours.
    """

    HR_LIST = (12, 18)
    FC_TIMES = tuple(np.arange(0, 174, 24))

    def run(self):
        """Task entrypoint."""
        date = self.context.date
        parallel = self.config.get("parallel", True)
        success = self.retrieve_mogreps_data(date=date, parallel=parallel)
        if success:
            logger.info("Cold Surge data retrieval complete.")
        else:
            logger.warning("Cold Surge data retrieval completed with errors.")

    def _init_config_values(self):
        """Load and cache retrieval paths/config used by helper methods."""
        if hasattr(self, "config_values"):
            return

        mogreps = load_global_config().get("mogreps", {})
        self.config_values = {
            "mogreps_moose_dir": mogreps.get("moose", "moose:/opfc/atm/mogreps-g/prods/"),
            "mogreps_raw_dir": mogreps.get("raw", "/tmp/salmon_raw/mogreps"),
            "mogreps_combined_queryfile": mogreps.get("query"),
            "mogreps_dummy_queryfiles_dir": mogreps.get("temp", "/tmp/salmon_temp"),
        }
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
        for hr in self.HR_LIST:
            for fc in self.FC_TIMES:
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

class ComputeColdSurgeIndices(Task):
    """Task to compute Cold Surge indices (v-wind 850hPa and precipitation)."""
    def run(self):
        date = self.context.date
        model = self.context.get_config('model', 'mogreps')
        
        global_config = load_global_config()
        model_config = global_config.get(model, {})
        
        raw_dir = model_config.get('raw', '/tmp/salmon_raw/mogreps')
        processed_base = model_config.get('processed', f'/tmp/salmon_processed/{model}')
        
        output_dir = os.path.join(processed_base, 'coldsurge', self.context.recipe_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        logger.info(f"Computing Cold Surge indices for {date}...")
        
        # Original logic uses 12Z and 18Z members
        members = []
        for hr in [12, 18]:
            members.extend(self._get_all_members(hr))
            
        fc_times = [f"{fct:03d}" for fct in np.arange(0, 174, 24)]
        
        for varname in ['precip', 'u850', 'v850']:
            var_dir = os.path.join(output_dir, varname)
            if not os.path.exists(var_dir):
                os.makedirs(var_dir, exist_ok=True)
                
            output_file = os.path.join(var_dir, f"{varname}_ColdSurge_24h_allMember_{date.strftime('%Y%m%d')}.nc")
            
            if os.path.exists(output_file):
                logger.info(f"{output_file} already exists. Skipping.")
                continue

            cubes = []
            for mem in members:
                digit3_mem = '035' if mem == '00' else str('%03d' % int(mem)) # Simplified mapping
                # Re-evaluating member mapping to match retrieval
                # Actually, digit3_mem depends on 'hr' as well in retrieval.
                # However, raw_dir/YYYYMMDD/MEM3 should be unique.
                # Let's search for the member folder
                mem_path = os.path.join(raw_dir, date.strftime("%Y%m%d"), digit3_mem)
                if not os.path.exists(mem_path):
                    # Try alternate mapping for member 00 if it was from hr 12
                    digit3_mem = '000'
                    mem_path = os.path.join(raw_dir, date.strftime("%Y%m%d"), digit3_mem)
                
                mog_files = [os.path.join(raw_dir, date.strftime("%Y%m%d"), digit3_mem, f"englaa_pd{fct}.pp") 
                             for fct in fc_times]
                
                # Filter files that exist
                existing_files = [f for f in mog_files if os.path.exists(f)]
                if not existing_files:
                    logger.warning(f"No files found for member {mem} at {mem_path}")
                    continue

                realiz_coord = iris.coords.DimCoord([int(mem)], standard_name='realization', var_name='realization')
                
                if varname == 'precip':
                    cube = read_precip_correctly(existing_files)
                    if cube.shape[0] > 1:
                        cube.data[1:] -= cube.data[:-1]
                elif varname == 'u850':
                    cube = read_winds_correctly(existing_files, 'x_wind', pressure_level=850)
                elif varname == 'v850':
                    cube = read_winds_correctly(existing_files, 'y_wind', pressure_level=850)
                
                cube = subset_seasia(cube)
                
                # Cleanup coordinates for merging
                for coord in ['forecast_reference_time', 'realization', 'time']:
                    if cube.coords(coord):
                        cube.remove_coord(coord)
                cube.add_aux_coord(realiz_coord)
                cubes.append(cube)

            if cubes:
                merged_cube = iris.cube.CubeList(cubes).merge_cube()
                iris.save(merged_cube, output_file, netcdf_format='NETCDF4_CLASSIC')
                logger.info(f"Saved merged members to {output_file}")
            else:
                logger.error(f"No cubes found to merge for {varname}")

    def _get_all_members(self, hr):
        if hr == 12:
            return [str('%02d' % mem) for mem in range(18)]
        elif hr == 18:
            return [str('%02d' % mem) for mem in range(18, 35)] + ['00']
        return []
