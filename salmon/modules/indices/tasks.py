import os
import logging
import numpy as np
from typing import Dict, Any
from salmon.core.task import Task
from salmon.utils.config import load_global_config

logger = logging.getLogger(__name__)

class RetrieveIndicesData(Task):
    """Task to retrieve data for generic climate indices (e.g., Rainfall, Monsoon)."""
    def run(self):
        model = self.config.get('model', 'mogreps')
        
        global_config = load_global_config()
        model_config = global_config.get(model, {})
        
        raw_dir = model_config.get('raw', '/tmp/salmon_raw')
        logger.info(f"Retrieving {model} indices data into {raw_dir}...")
        # Mock retrieval logic
        logger.info("Indices data retrieval complete.")

class ComputeIndices(Task):
    """Task to compute various spatial and temporal indices."""
    def run(self):
        date = self.context.date
        model = self.config.get('model', 'mogreps')
        
        global_config = load_global_config()
        mogreps_config = global_config.get('mogreps', {})
        
        raw_dir = mogreps_config.get('raw', '/tmp/salmon_raw')
        # Dynamic path: processed_base / indices / recipe_name
        processed_base = mogreps_config.get('processed', '/tmp/salmon_processed')
        processed_dir = os.path.join(processed_base, 'indices', self.context.recipe_name)
        
        fc_times = [str('%03d' % fct) for fct in np.arange(0, 174, 24)]
        
        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)

        logger.info(f"Computing spatial indices for {date}...")
        # Mock computation
        logger.info(f"Indices saved to {processed_dir}")
