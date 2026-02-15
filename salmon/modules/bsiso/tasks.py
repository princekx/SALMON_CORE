import os
import logging
import iris
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from salmon.core.task import Task
from salmon.utils.config import load_global_config

logger = logging.getLogger(__name__)

class ComputeBSISOIndices(Task):
    """Task to compute BSISO indices from forecast data."""
    def run(self):
        date = self.context.date
        model = self.config.get('model', 'mogreps')
        
        global_config = load_global_config()
        model_config = global_config.get(model, {})
        
        # Paths
        forecast_out_dir = model_config.get('processed', '/tmp/salmon_processed/mogreps')
        archive_base = model_config.get('archive', '/tmp/salmon_archive/mogreps')
        archive_dir = os.path.join(archive_base, date.strftime("%Y%m%d"))
        
        os.makedirs(archive_dir, exist_ok=True)
        
        # Module's data directory for EOFs and coefficients
        module_data_dir = os.path.join(os.path.dirname(__file__), 'data')
        
        logger.info(f"Computing BSISO indices for {date} (Model: {model})...")
        
        members = [f"{mem:03d}" for mem in range(self.config.get('members', 18))] # Default members for BSISO
        
        for mem in members:
            output_file = os.path.join(archive_dir, f"BSISO.{date.strftime('%Y%m%d')}.fcast.{mem}.txt")
            if os.path.exists(output_file):
                continue
                
            # Inputs are 120-day anomalies (pre-computed by MJO module or similar)
            anom_files = {}
            for var in ['olr', 'u850']:
                f = os.path.join(forecast_out_dir, var, f"{var}_120dm_40sn_nrt_{date.strftime('%Y%m%d')}_{mem}.nc")
                if os.path.exists(f):
                    anom_files[var] = f
            
            if len(anom_files) < 2:
                continue
                
            self._compute_for_member(date, mem, anom_files, output_file, module_data_dir)

        logger.info(f"BSISO indices computation complete.")

    def _compute_for_member(self, date, mem, anom_files, output_file, data_dir):
        # Load coefficients
        coeff_df = pd.read_csv(os.path.join(data_dir, 'BSISO_coefficients.csv'))
        
        # Load and constrain data
        reg_constraints = iris.Constraint(latitude=lambda cell: -10 <= cell <= 40) & \
                          iris.Constraint(longitude=lambda cell: 40. <= cell <= 160.)
        
        olr_cube = iris.load_cube(anom_files['olr']).extract(reg_constraints)
        u850_cube = iris.load_cube(anom_files['u850']).extract(reg_constraints)
        
        # Normalization
        olr_cube /= coeff_df['olr_sd_x'].values[0]
        u850_cube /= coeff_df['u850_sd_x'].values[0]
        
        ntime = olr_cube.shape[0]
        olr_resh = np.reshape(olr_cube.data, (ntime, -1))
        u850_resh = np.reshape(u850_cube.data, (ntime, -1))
        
        comb_data = np.concatenate((olr_resh, u850_resh), axis=1)
        
        # Read Predefined EOF structures
        eof_file = os.path.join(data_dir, 'BSISO.EOFstruc.data')
        eofs = pd.read_csv(eof_file, delim_whitespace=True)
        comb_eof1 = np.concatenate((eofs.OLR1, eofs.U8501), axis=0)
        comb_eof2 = np.concatenate((eofs.OLR2, eofs.U8502), axis=0)
        
        # Project
        pc1 = np.dot(comb_data, comb_eof1) / coeff_df['pc1_sd'].values[0]
        pc2 = np.dot(comb_data, comb_eof2) / coeff_df['pc2_sd'].values[0]
        
        amp = np.sqrt(pc1**2 + pc2**2)
        phase_angle = np.arctan2(pc2, pc1)
        phase_angle = np.where(phase_angle < 0, phase_angle + 2 * np.pi, phase_angle)
        
        phase = np.where((phase_angle >= 0) & (phase_angle < np.pi / 4), 5,
                 np.where((phase_angle >= np.pi / 4) & (phase_angle < np.pi / 2), 6,
                  np.where((phase_angle >= np.pi / 2) & (phase_angle < 3 * np.pi / 4), 7,
                   np.where((phase_angle >= 3 * np.pi / 4) & (phase_angle < np.pi), 8,
                    np.where((phase_angle >= np.pi) & (phase_angle < 5 * np.pi / 4), 1,
                     np.where((phase_angle >= 5 * np.pi / 4) & (phase_angle < 3 * np.pi / 2), 2,
                      np.where((phase_angle >= 3 * np.pi / 2) & (phase_angle < 7 * np.pi / 4), 3, 4)))))))

        times = olr_cube.coord('time').units.num2date(olr_cube.coord('time').points)
        df = pd.DataFrame({
            'Year': [t.year for t in times],
            'Month': [t.month for t in times],
            'Day': [t.day for t in times],
            'PC1': pc1,
            'PC2': pc2,
            '8_phases': phase,
            'amp': amp,
            'mem': [mem] * ntime
        })
        
        df['label'] = 'analysis' 
        nfc = self.config.get('nforecasts', 7)
        if ntime > nfc:
             df.iloc[-nfc:, df.columns.get_loc('label')] = 'forecast'
             
        df.to_csv(output_file, index=False)
