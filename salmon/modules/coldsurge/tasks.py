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
    """Task to retrieve data required for Cold Surge indices."""
    def run(self):
        date = self.context.date
        parallel = self.config.get('parallel', True)
        
        global_config = load_global_config()
        mogreps_config = global_config.get('mogreps', {})
        
        self._init_config_values()
        
        logger.info(f"Retrieving MOGREPS data for Cold Surge on {date}...")
        
        hr_list = [12, 18]
        fc_times = np.arange(0, 174, 24)
        
        tasks_to_run = []
        for hr in hr_list:
            members = self._get_all_members(hr)
            for fc in fc_times:
                for mem in members:
                    tasks_to_run.append((date, hr, fc, mem, self.config_values["mogreps_moose_dir"], self.config_values["mogreps_raw_dir"], self.config_values["mogreps_combined_queryfile"], self.config_values["mogreps_dummy_queryfiles_dir"]))
        
        if parallel:
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(self._retrieve_for_member, *t) for t in tasks_to_run]
                concurrent.futures.wait(futures)
        else:
            for t in tasks_to_run:
                self._retrieve_for_member(*t)

        logger.info("Cold Surge data retrieval complete.")

    def _init_config_values(self):
        """Initialize legacy-style config_values used by retrieval helpers."""
        if hasattr(self, "config_values"):
            return

        global_config = load_global_config()
        mogreps_config = global_config.get("mogreps", {})
        self.config_values = {
            "mogreps_moose_dir": mogreps_config.get("moose", "moose:/opfc/atm/mogreps-g/prods/"),
            "mogreps_raw_dir": mogreps_config.get("raw", "/tmp/salmon_raw/mogreps"),
            "mogreps_combined_queryfile": mogreps_config.get("query"),
            "mogreps_dummy_queryfiles_dir": mogreps_config.get("temp", "/tmp/salmon_temp"),
        }
        os.makedirs(self.config_values["mogreps_dummy_queryfiles_dir"], exist_ok=True)

    def _get_all_members(self, hr):
        if hr == 12:
            return [str('%02d' % mem) for mem in range(18)]
        elif hr == 18:
            return [str('%02d' % mem) for mem in range(18, 35)] + ['00']
        return []

    def _retrieve_for_member(self, date, hr, fc, digit2_mem, moose_base, raw_dir, query_template, temp_dir):
        """Check if the date is after the OS47 date (21-01-2026)."""
        os47_date = datetime.datetime(2026, 1, 21)
        # This check was added when they had messed up the folders. Looks like they have fixed it now,
        # but keeping the check just in case.
        """
        if date > os47_date:
            # monthly folders are not used after OS47
            print('Date is after OS47 date. Proceeding with retrieval without monthly folders.')
            moose_dir = os.path.join(self.config_values['mogreps_moose_dir'], f'{date.strftime("%Y")}.pp')
        else:
            print('Date is before OS47 date. Proceeding with retrieval.')
            moose_dir = os.path.join(self.config_values['mogreps_moose_dir'], f'{date.strftime("%Y%m")}.pp')
        """
        moose_dir = os.path.join(self.config_values["mogreps_moose_dir"], f'{date.strftime("%Y%m")}.pp')

        digit3_mem = "035" if (hr == 18 and digit2_mem == "00") else str("%03d" % int(digit2_mem))

        remote_data_dir = os.path.join(
            self.config_values["mogreps_raw_dir"], date.strftime("%Y%m%d"), digit3_mem
        )
        if not os.path.exists(remote_data_dir):
            os.makedirs(remote_data_dir, exist_ok=True)

        print(f"Retrieving hr: {fc}")
        fct = f"{fc:03d}"

        # File names changed on moose on 25/09/2018
        filemoose = f"prods_op_mogreps-g_{date.strftime('%Y%m%d')}_{hr}_{digit2_mem}_{fct}.pp"
        outfile = f"englaa_pd{fct}.pp"

        # Generate a unique query file
        local_query_file1 = os.path.join(
            self.config_values["mogreps_dummy_queryfiles_dir"], f"localquery_{uuid.uuid1()}"
        )
        self.create_query_file(local_query_file1, filemoose, fct)

        outfile_path = os.path.join(remote_data_dir, outfile)

        if os.path.exists(outfile_path):
            if os.path.getsize(outfile_path) == 0:
                print(os.path.getsize(outfile_path))
                print(f"Deleting empty file {outfile_path}")
                os.remove(outfile_path)

        if not os.path.exists(outfile_path):
            print("EXECCCC")
            command = (
                f"/opt/moose-client-wrapper/bin/moo select --fill-gaps "
                f"{local_query_file1} {moose_dir} {os.path.join(remote_data_dir, outfile)}"
            )
            logger.info("Executing command: %s", command)

            try:
                subprocess.run(command, shell=True, check=True)
                logger.info("Data retrieval successful.")
            except subprocess.CalledProcessError as e:
                logger.error("Error during data retrieval: %s", e)
            except Exception as e:
                logger.error("An unexpected error occurred: %s", e)
        else:
            print(f"{os.path.join(remote_data_dir, outfile)} exists. Skip...")

        if os.path.exists(local_query_file1):
            os.remove(local_query_file1)

    def check_if_all_data_exist(self, date, hr, fc, digit2_mem):
        self._init_config_values()
        digit3_mem = str("%03d" % int(digit2_mem))
        remote_data_dir = os.path.join(
            self.config_values["mogreps_raw_dir"], date.strftime("%Y%m%d"), digit3_mem
        )
        fct = f"{fc:03d}"
        outfile = f"englaa_pd{fct}.pp"
        outfile_path = os.path.join(remote_data_dir, outfile)
        outfile_status = os.path.exists(outfile_path) and os.path.getsize(outfile_path) > 0
        return outfile_status

    def create_query_file(self, local_query_file1, filemoose, fct):
        self._init_config_values()
        query_file = self.config_values["mogreps_combined_queryfile"]

        replacements = {"fctime": fct, "filemoose": filemoose}
        with open(query_file) as query_infile, open(local_query_file1, "w") as query_outfile:
            for line in query_infile:
                for src, target in replacements.items():
                    line = line.replace(src, target)
                query_outfile.write(line)

    def retrieve_fc_data_parallel(self, date, hr, fc, digit2_mem):
        print("In retrieve_fc_data_parallel()")
        self._init_config_values()

        """Check if the date is after the OS47 date (21-01-2026)."""
        os47_date = datetime.datetime(2026, 1, 21)
        # This check was added when they had messed up the folders. Looks like they have fixed it now,
        # but keeping the check just in case.
        """
        if date > os47_date:
            # monthly folders are not used after OS47
            print('Date is after OS47 date. Proceeding with retrieval without monthly folders.')
            moose_dir = os.path.join(self.config_values['mogreps_moose_dir'], f'{date.strftime("%Y")}.pp')
        else:
            print('Date is before OS47 date. Proceeding with retrieval.')
            moose_dir = os.path.join(self.config_values['mogreps_moose_dir'], f'{date.strftime("%Y%m")}.pp')
        """
        moose_dir = os.path.join(self.config_values["mogreps_moose_dir"], f'{date.strftime("%Y%m")}.pp')

        digit3_mem = "035" if (hr == 18 and digit2_mem == "00") else str("%03d" % int(digit2_mem))

        remote_data_dir = os.path.join(
            self.config_values["mogreps_raw_dir"], date.strftime("%Y%m%d"), digit3_mem
        )
        if not os.path.exists(remote_data_dir):
            os.makedirs(remote_data_dir, exist_ok=True)

        print(f"Retrieving hr: {fc}")
        fct = f"{fc:03d}"

        # File names changed on moose on 25/09/2018
        filemoose = f"prods_op_mogreps-g_{date.strftime('%Y%m%d')}_{hr}_{digit2_mem}_{fct}.pp"
        outfile = f"englaa_pd{fct}.pp"

        # Generate a unique query file
        local_query_file1 = os.path.join(
            self.config_values["mogreps_dummy_queryfiles_dir"], f"localquery_{uuid.uuid1()}"
        )
        self.create_query_file(local_query_file1, filemoose, fct)

        outfile_path = os.path.join(remote_data_dir, outfile)

        if os.path.exists(outfile_path):
            if os.path.getsize(outfile_path) == 0:
                print(os.path.getsize(outfile_path))
                print(f"Deleting empty file {outfile_path}")
                os.remove(outfile_path)

        if not os.path.exists(outfile_path):
            print("EXECCCC")
            command = (
                f"/opt/moose-client-wrapper/bin/moo select --fill-gaps "
                f"{local_query_file1} {moose_dir} {os.path.join(remote_data_dir, outfile)}"
            )
            logger.info("Executing command: %s", command)

            try:
                subprocess.run(command, shell=True, check=True)
                logger.info("Data retrieval successful.")
            except subprocess.CalledProcessError as e:
                logger.error("Error during data retrieval: %s", e)
            except Exception as e:
                logger.error("An unexpected error occurred: %s", e)
        else:
            print(f"{os.path.join(remote_data_dir, outfile)} exists. Skip...")

        if os.path.exists(local_query_file1):
            os.remove(local_query_file1)

    def retrieve_mogreps_data(self, date, parallel=True):
        print("Retrieving data for date:", date)
        self._init_config_values()

        hr_list = [12, 18]
        fc_times = np.arange(0, 174, 24)
        print(fc_times)

        # Create a list of tuples for all combinations of hr and mem
        tasks = [
            (date, hr, fc, digit2_mem)
            for hr in hr_list
            for fc in fc_times
            for digit2_mem in self._get_all_members(hr)
        ]

        if parallel:
            # Use ThreadPoolExecutor to run tasks in parallel
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Submit tasks to the executor
                futures = [executor.submit(self.retrieve_fc_data_parallel, *task) for task in tasks]

                # Wait for all tasks to complete
                concurrent.futures.wait(futures)

            # Check if all tasks are completed
            all_tasks_completed = all(future.done() for future in futures)
            return all_tasks_completed
        else:
            for task in tasks:
                self.retrieve_fc_data_parallel(*task)
            return True

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
