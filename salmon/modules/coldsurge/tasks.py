import os
import logging
import datetime
import uuid
import concurrent.futures
from typing import Dict, Any, List
import numpy as np
import iris
import iris.coords
from salmon.core.task import Task
from salmon.utils.moose import MooseClient
from salmon.utils.config import load_global_config
from salmon.utils.cube import read_winds_correctly, read_precip_correctly, subset_seasia

logger = logging.getLogger(__name__)

class RetrieveColdSurgeData(Task):
    """Task to retrieve data required for Cold Surge indices."""
    def run(self):
        date = self.context.date
        parallel = self.config.get('parallel', True)
        
        global_config = load_global_config()
        mogreps_config = global_config.get('mogreps', {})
        
        moose_base = mogreps_config.get('moose', 'moose:/opfc/atm/mogreps-g/prods/')
        raw_dir = mogreps_config.get('raw', '/tmp/salmon_raw/mogreps')
        query_template = mogreps_config.get('query')
        temp_dir = mogreps_config.get('temp', '/tmp/salmon_temp')

        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir, exist_ok=True)

        logger.info(f"Retrieving MOGREPS data for Cold Surge on {date}...")
        
        hr_list = [12, 18]
        fc_times = np.arange(0, 174, 24)
        
        tasks_to_run = []
        for hr in hr_list:
            members = self._get_all_members(hr)
            for fc in fc_times:
                for mem in members:
                    tasks_to_run.append((date, hr, fc, mem, moose_base, raw_dir, query_template, temp_dir))
        
        if parallel:
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(self._retrieve_for_member, *t) for t in tasks_to_run]
                concurrent.futures.wait(futures)
        else:
            for t in tasks_to_run:
                self._retrieve_for_member(*t)

        logger.info("Cold Surge data retrieval complete.")

    def _get_all_members(self, hr):
        if hr == 12:
            return [str('%02d' % mem) for mem in range(18)]
        elif hr == 18:
            return [str('%02d' % mem) for mem in range(18, 35)] + ['00']
        return []

    def _retrieve_for_member(self, date, hr, fc, digit2_mem, moose_base, raw_dir, query_template, temp_dir):
        # OS47 date transition logic
        os47_date = datetime.datetime(2026, 1, 21)
        if date > os47_date:
            moosedir = os.path.join(moose_base, f"{date.year}.pp")
        else:
            moosedir = os.path.join(moose_base, f"{date.strftime('%Y%m')}.pp")

        digit3_mem = '035' if (hr == 18 and digit2_mem == '00') else str('%03d' % int(digit2_mem))
        mem_dir = os.path.join(raw_dir, date.strftime("%Y%m%d"), digit3_mem)
        
        if not os.path.exists(mem_dir):
            os.makedirs(mem_dir, exist_ok=True)

        fct = f"{fc:03d}"
        filemoose = f"prods_op_mogreps-g_{date.strftime('%Y%m%d')}_{hr}_{digit2_mem}_{fct}.pp"
        outfile = f"englaa_pd{fct}.pp"
        outpath = os.path.join(mem_dir, outfile)

        if os.path.exists(outpath) and os.path.getsize(outpath) > 0:
            return

        moose_client = MooseClient()
        local_query = os.path.join(temp_dir, f"query_{uuid.uuid4()}.query")
        
        moose_client.create_query_file(query_template, local_query, {'fctime': fct, 'filemoose': filemoose})
        success = moose_client.retrieve(local_query, moosedir, outpath)
        
        if os.path.exists(local_query):
            os.remove(local_query)
        
        if not success:
            logger.warning(f"Failed to retrieve MOGREPS {filemoose}")

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
