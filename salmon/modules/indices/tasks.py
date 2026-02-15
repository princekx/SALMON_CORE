import os
import logging
import numpy as np
import xarray as xr
import pandas as pd
import datetime
from typing import Dict, Any
from salmon.core.task import Task
from salmon.utils.config import load_global_config

logger = logging.getLogger(__name__)

class RetrieveIndicesData(Task):
    """Task to retrieve data for generic climate indices (e.g., Rainfall, Monsoon)."""
    def run(self):
        # In this specific case, logic is handled by xarray in ComputeMonsoonIndices
        logger.info("Indices data retrieval is handled on-the-fly by xarray for GEFS.")

class ComputeMonsoonIndices(Task):
    """Task to compute various Monsoon indices (SWMI, NEMI, etc) using GEFS data."""
    def run(self):
        date = self.context.date
        
        global_config = load_global_config()
        indices_config = global_config.get('indices', {})
        
        # Output directory
        processed_base = indices_config.get('processed', '/tmp/salmon_processed/indices')
        processed_dir = os.path.join(processed_base, self.context.recipe_name, date.strftime('%Y%m%d'))
        os.makedirs(processed_dir, exist_ok=True)
        
        # GEFS nomads URL
        yyyy = date.strftime("%Y")
        mm = date.strftime("%m")
        dd = date.strftime("%d")
        url = f'http://nomads.ncep.noaa.gov:80/dods/gefs/gefs{yyyy}{mm}{dd}/gefs_pgrb2ap5_all_00z'
        
        logger.info(f"Opening GEFS dataset from {url}...")
        try:
            ds = xr.open_dataset(url, engine='netcdf4')
        except Exception as e:
            logger.error(f"Failed to open GEFS dataset: {e}")
            return

        dsu = ds['ugrdprs']
        dsv = ds['vgrdprs']
        
        # Region subset
        dsu_region = dsu.sel(lon=slice(90, 160), lat=slice(0, 30))
        dsv_region = dsv.sel(lon=slice(90, 160), lat=slice(0, 30))
        
        # Daily means across ensemble members
        u850 = dsu_region.sel(lev=850).mean(dim='ens').resample(time='D').mean()
        u925 = dsu_region.sel(lev=925).mean(dim='ens').resample(time='D').mean()
        v925 = dsv_region.sel(lev=925).mean(dim='ens').resample(time='D').mean()
        
        # 1. SWMI (Southwest Monsoon Index)
        box1 = u850.sel(lon=slice(90, 130), lat=slice(5, 15)).mean(dim=['lon', 'lat'])
        box2 = u850.sel(lon=slice(100.75, 103.25), lat=slice(1.75, 4.25)).mean(dim=['lon', 'lat'])
        swmi1 = box2 - box1
        swmi1.to_dataframe().reset_index().to_csv(os.path.join(processed_dir, f'swmi1_{date.strftime("%Y%m%d")}.csv'), index=False)
        
        # 2. SWMI2
        swmi2 = u850.sel(lon=slice(100, 115), lat=slice(5, 10)).mean(dim=['lon', 'lat'])
        swmi2.to_dataframe().reset_index().to_csv(os.path.join(processed_dir, f'swmi2_{date.strftime("%Y%m%d")}.csv'), index=False)
        
        # 3. NEMI (Northeast Monsoon Index)
        n850 = u850.sel(lon=slice(102.5, 105), lat=slice(3.75, 6.25)).mean(dim=['lon', 'lat'])
        n925 = u925.sel(lon=slice(102.5, 105), lat=slice(3.75, 6.25)).mean(dim=['lon', 'lat'])
        df_nemi = pd.merge(n850.to_dataframe().rename(columns={'ugrdprs':'u850'}), 
                           n925.to_dataframe().rename(columns={'ugrdprs':'u925'}), on='time')
        df_nemi['average'] = df_nemi[['u850', 'u925']].mean(axis=1)
        df_nemi.to_csv(os.path.join(processed_dir, f'nemi_{date.strftime("%Y%m%d")}.csv'), index=False)
        
        # 4. NEMO (Northeast Monsoon Onset)
        nemo = v925.sel(lon=slice(107, 115), lat=slice(5, 15)).mean(dim=['lon', 'lat'])
        nemo.to_dataframe().reset_index().to_csv(os.path.join(processed_dir, f'nemo_{date.strftime("%Y%m%d")}.csv'), index=False)
        
        # 5. MESI (Easterly Surge Index ESI / Meridional Surge Index MSI)
        esi = u925.sel(lon=120, lat=slice(7.5, 15)).mean(dim='lat')
        msi = v925.sel(lon=slice(110, 117.5), lat=15).mean(dim='lon')
        df_esi = esi.to_dataframe().reset_index()
        df_msi = msi.to_dataframe().reset_index()
        df_esi.rename(columns={'ugrdprs': 'ESI'}).to_csv(os.path.join(processed_dir, f'esi_{date.strftime("%Y%m%d")}.csv'), index=False)
        df_msi.rename(columns={'vgrdprs': 'MSI'}).to_csv(os.path.join(processed_dir, f'msi_{date.strftime("%Y%m%d")}.csv'), index=False)

        logger.info(f"Monsoon indices computed and saved to {processed_dir}")

class ComputeIndices(Task):
    """Generic task for other indices."""
    def run(self):
        logger.info("Computing generic indices...")
