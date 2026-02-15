import os
import logging
import iris
import numpy as np
from typing import Dict, Any, List
from salmon.core.task import Task
from salmon.utils.config import load_global_config
from .utils import find_conv_lines

logger = logging.getLogger(__name__)

class ComputeConvLines(Task):
    """Task to compute convergence lines from ensemble wind data."""
    def run(self):
        date = self.context.date
        model = self.config.get('model', 'mogreps')
        
        global_config = load_global_config()
        model_config = global_config.get(model, {})
        
        # Paths
        processed_base = model_config.get('processed', '/tmp/salmon_processed/mogreps')
        output_dir = os.path.join(processed_base, 'convlines', self.context.recipe_name)
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, f'convergence_lines_{date.strftime("%Y%m%d")}.nc')
        
        if os.path.exists(output_file):
            logger.info(f"Convergence lines file already exists: {output_file}")
            return

        logger.info(f"Computing convergence lines for {date}...")
        
        # Load u850 and v850
        # These are assumed to be pre-processed by previous tasks in the recipe (e.g., subset/regrid)
        # We look for them in the processed directory of the current recipe
        u_file = os.path.join(processed_base, 'mjo', self.context.recipe_name, f'u850_combined_{date.strftime("%Y%m%d")}.nc')
        v_file = os.path.join(processed_base, 'mjo', self.context.recipe_name, f'v850_combined_{date.strftime("%Y%m%d")}.nc')
        
        if not (os.path.exists(u_file) and os.path.exists(v_file)):
            logger.error(f"Required wind files not found: {u_file}, {v_file}")
            return
            
        u_cube = iris.load_cube(u_file).intersection(longitude=(90, 135))
        v_cube = iris.load_cube(v_file).intersection(longitude=(90, 135))
        
        # Shape: [realization, time, lat, lon]
        nmem = u_cube.shape[0]
        ntime = u_cube.shape[1]
        
        # Output cube template
        conv_cube = u_cube[0].copy() # realization dimension removed by indexing
        conv_cube.rename('convergence_lines')
        conv_cube.units = '1'
        conv_cube.data = np.zeros(conv_cube.shape)
        
        for t in range(ntime):
            logger.info(f"Processing forecast time index {t}/{ntime-1}")
            member_masks = []
            for mem in range(nmem):
                _, _, cleaned = find_conv_lines(u_cube.data[mem, t], v_cube.data[mem, t])
                if cleaned is not None:
                    member_masks.append(cleaned)
            
            if member_masks:
                conv_cube.data[t] = np.mean(np.array(member_masks), axis=0)

        iris.save(conv_cube, output_file)
        logger.info(f"Convergence lines ensemble mean saved to {output_file}")
