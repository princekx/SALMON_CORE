import os
import logging
import iris
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from salmon.core.task import Task
from salmon.utils.config import load_global_config
from .utils import extract_grid_features

logger = logging.getLogger(__name__)

class ComputeFeatureObjects(Task):
    """Task to identify and characterize 2D features (e.g. rain cells)."""
    def run(self):
        date = self.context.date
        model = self.config.get('model', 'mogreps')
        
        global_config = load_global_config()
        model_config = global_config.get(model, {})
        
        # Paths
        processed_base = model_config.get('processed', '/tmp/salmon_processed/mogreps')
        # Features typically characterized from 24h precip
        output_dir = os.path.join(processed_base, 'features', self.context.recipe_name)
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, f'features_{date.strftime("%Y%m%d")}.csv')
        
        logger.info(f"Computing features for {date}...")
        
        # Input: precipitation_amount (24h)
        # In a real recipe, this would be computed by a previous task.
        # We look for it in the coldsurge processed dir as that module often computes 24h precip.
        recipe_name = self.context.recipe_name
        input_file = os.path.join(processed_base, 'coldsurge', recipe_name, f'precip_24h_combined_{date.strftime("%Y%m%d")}.nc')
        
        if not os.path.exists(input_file):
            logger.error(f"Input file not found: {input_file}")
            return
            
        cube = iris.load_cube(input_file)
        
        # Shape: [realization, time, lat, lon]
        nmem = cube.shape[0] if len(cube.shape) == 4 else 1
        ntime = cube.shape[1] if len(cube.shape) == 4 else cube.shape[0]
        
        thresholds = self.config.get('thresholds', [1.0, 5.0, 10.0, 20.0, 50.0])
        
        all_member_frames = []
        
        for mem in range(nmem):
            logger.info(f"Processing member {mem}/{nmem-1}")
            member_frames = []
            for t in range(ntime):
                # Extract 2D slice
                if len(cube.shape) == 4:
                    slice_cube = cube[mem, t]
                else:
                    slice_cube = cube[t]
                    
                df = extract_grid_features(slice_cube, thresholds=thresholds, time_index=t)
                if not df.empty:
                    df['member'] = mem
                    member_frames.append(df)
            
            if member_frames:
                all_member_frames.append(pd.concat(member_frames, ignore_index=True))

        if all_member_frames:
            final_df = pd.concat(all_member_frames, ignore_index=True)
            final_df.to_csv(output_file, index=False)
            logger.info(f"Features saved to {output_file}")
        else:
            logger.warning("No features identified.")

class ExtractPrecipFeatures(ComputeFeatureObjects):
    """Legacy alias for ComputeFeatureObjects."""
    pass
