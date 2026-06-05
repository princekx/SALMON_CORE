import os
import logging
import numpy as np
import pandas as pd
import datetime
import json
import iris
import iris.coords

from bokeh.plotting import figure, save, output_file
from bokeh.models import (
    Band, ColumnDataSource, DatetimeTickFormatter, Span
)
from bokeh.layouts import gridplot

from salmon.core.task import Task
from salmon.utils.config import load_global_config

logger = logging.getLogger(__name__)

class RetrieveIndicesData(Task):
    """Task to retrieve data for generic climate indices (e.g., Rainfall, Monsoon)."""

    def run(self):
        """Log retrieval strategy for indices products."""
        logger.info("Indices data retrieval is handled on-the-fly by xarray for GEFS.")

class DisplayMonsoonIndices(Task):
    """Create monsoon indices ensemble probability visualizations as Bokeh HTML files."""

    def run(self):
        """Execute monsoon indices display task for the current recipe date."""
        date = self.context.date
        self._init_config_values()
        self.bokeh_plot_forecast_ensemble_mean(date)

    def _init_config_values(self):
        """Load plot directories and model configuration from global settings."""
        if hasattr(self, 'config_values'):
            return

        model = self.context.get_config('model', 'mogreps')
        global_cfg = load_global_config()
        model_cfg = global_cfg.get(model, {})

        processed_base = model_cfg.get('processed', f'/tmp/salmon_processed/{model}')
        indices_processed_dir = os.path.join(processed_base, 'indices')
        indices_plot_ens_dir = os.path.join(
            model_cfg.get('plots', processed_base),
            'indices', 'plot_ens'
        )

        self.config_values = {
            'model': model,
            f'{model}_indices_processed_dir': indices_processed_dir,
            f'{model}_indices_plot_ens': indices_plot_ens_dir,
        }

        os.makedirs(indices_processed_dir, exist_ok=True)
        os.makedirs(indices_plot_ens_dir, exist_ok=True)

    @staticmethod
    def _set_bounds_if_missing(cube, coord_name):
        """Add simple symmetric bounds to a scalar-like coordinate if missing."""
        coord = cube.coord(coord_name)
        if coord.bounds is None:
            p0 = coord.points[0]
            coord.bounds = [[p0 - 1.0, p0 + 1.0]]

    @staticmethod
    def _remove_coords_if_present(cube, coord_names):
        """Remove coordinates from cube only when they exist."""
        for coord_name in coord_names:
            if cube.coords(coord_name):
                cube.remove_coord(coord_name)

    def read_winds_correctly(self, data_files, varname, pressure_level=None):
        """Load and process wind data from NetCDF files using Iris.

        This method:
          - Sorts input files for temporal order
          - Loads a specified variable from each file
          - Optionally extracts a specific pressure level
          - Averages over time if 3-hourly resolution detected
          - Ensures forecast_period and time coordinates have bounds
          - Equalises cube attributes to allow merging
          - Merges all processed cubes into a single cube
          - Restricts to Southeast Asia region (0–30°N, 90–160°E)

        Parameters
        ----------
        data_files : list of str
            Paths to NetCDF files containing the wind variable.
        varname : str
            Name of the wind variable (e.g., 'ugrd', 'vgrd').
        pressure_level : float, optional
            Pressure level in hPa to extract (e.g., 850.0).

        Returns
        -------
        iris.cube.Cube
            Merged cube with processed wind data over Southeast Asia.
        """
        data_files = sorted(data_files)
        cubes = []
        for data_file in data_files:
            cube = iris.load_cube(data_file, varname)
            if pressure_level is not None:
                cube = cube.extract(iris.Constraint(pressure=pressure_level))
            if len(cube.shape) == 3:
                cube = cube.collapsed('time', iris.analysis.MEAN)
            self._set_bounds_if_missing(cube, 'forecast_period')
            self._set_bounds_if_missing(cube, 'time')
            cubes.append(cube)

        iris.util.equalise_attributes(cubes)
        for cube in cubes:
            cube.cell_methods = ()

        cubes = iris.cube.CubeList(cubes).merge_cube()
        return cubes.intersection(latitude=(0, 30), longitude=(90, 160))

    def write_dates_json(self, date, json_file):
        """Append cycle date to JSON index file if not already present."""
        new_date = date.strftime('%Y%m%d')

        if not os.path.exists(json_file):
            with open(json_file, 'w', encoding='utf-8') as jfile:
                json.dump([new_date], jfile, indent=2)
            return

        with open(json_file, 'r', encoding='utf-8') as file:
            existing_dates = json.load(file)

        if new_date not in existing_dates:
            existing_dates.append(new_date)

        existing_dates.sort()

        with open(json_file, 'w', encoding='utf-8') as file:
            json.dump(existing_dates, file, indent=2)

        logger.info('Updated %s with date %s', json_file, new_date)

    def compute_indices(self, dates, members, ugrd850_cubes, ugrd925_cubes, vgrd925_cubes):
        """Compute regional monsoon-related indices from wind data cubes.

        Calculates:
          - SWMI1: Difference in regional mean u850 between two boxes
          - SWMI2: Mean u850 over 5–10°N, 100–115°E
          - NEMI: Average of u850 and u925 over 3.75–6.25°N, 102.5–105°E
          - NEMO: Mean v925 over 5–15°N, 107–115°E

        Returns DataFrame in long format with one row per forecast period and ensemble member.

        Parameters
        ----------
        dates : list
            Forecast valid datetimes.
        members : list
            Ensemble member labels
        ugrd850_cubes : iris.cube.Cube
            Eastward wind at 850 hPa
        ugrd925_cubes : iris.cube.Cube
            Eastward wind at 925 hPa
        vgrd925_cubes : iris.cube.Cube
            Northward wind at 925 hPa

        Returns
        -------
        pandas.DataFrame
            Long-format DataFrame with columns: 'forecast_period', 'realization', 'swmi1', 'swmi2', 'nemi', 'nemo'
        """
        box1 = ugrd850_cubes.intersection(latitude=(5, 15), longitude=(90, 130)).collapsed(
            ['latitude', 'longitude'], iris.analysis.MEAN)
        box2 = ugrd850_cubes.intersection(latitude=(1.75, 4.25), longitude=(100.75, 103.25)).collapsed(
            ['latitude', 'longitude'], iris.analysis.MEAN)
        swmi1 = box2 - box1

        swmi2 = ugrd850_cubes.intersection(latitude=(5, 10), longitude=(100, 115)).collapsed(
            ['latitude', 'longitude'], iris.analysis.MEAN)

        n850 = ugrd850_cubes.intersection(latitude=(3.75, 6.25), longitude=(102.5, 105)).collapsed(
            ['latitude', 'longitude'], iris.analysis.MEAN)
        n925 = ugrd925_cubes.intersection(latitude=(3.75, 6.25), longitude=(102.5, 105)).collapsed(
            ['latitude', 'longitude'], iris.analysis.MEAN)
        nemi = 0.5 * (n850 + n925)

        nemo = vgrd925_cubes.intersection(latitude=(5, 15), longitude=(107, 115)).collapsed(
            ['latitude', 'longitude'], iris.analysis.MEAN)

        n_times = swmi1.shape[1]
        valid_dates = dates[:n_times]
        valid_members = members[:swmi1.shape[0]]

        data_records = []
        for i, fp in enumerate(valid_dates):
            for j, r in enumerate(valid_members):
                data_records.append(
                    {
                        'forecast_period': fp,
                        'realization': r,
                        'swmi1': swmi1.data[j, i],
                        'swmi2': swmi2.data[j, i],
                        'nemi': nemi.data[j, i],
                        'nemo': nemo.data[j, i],
                    }
                )

        return pd.DataFrame(data_records)

    def plot_index_ens(self, date, indices_df, thresholds, titles, index_name=None):
        """Render ensemble probability timeseries for a single index.

        Shows ensemble mean, percentiles, and individual member lines.
        Overlays a reference threshold line.

        Parameters
        ----------
        date : datetime.datetime
            Forecast start date
        indices_df : pandas.DataFrame
            DataFrame with 'forecast_period', 'realization', and index columns
        thresholds : dict
            Threshold values for reference lines
        titles : dict
            Human-readable titles indexed by index name
        index_name : str
            Column name in indices_df to plot

        Returns
        -------
        bokeh.plotting.Figure
            Figure with ensemble timeseries
        """
        pivot = indices_df.pivot(index='forecast_period', columns='realization', values=index_name)

        forecast_periods = pivot.index.values
        mean_vals = pivot.mean(axis=1).values
        min_vals = pivot.min(axis=1).values
        max_vals = pivot.max(axis=1).values
        q25 = pivot.quantile(0.25, axis=1).values
        q75 = pivot.quantile(0.75, axis=1).values

        source = ColumnDataSource(data={
            'forecast_period': forecast_periods,
            'mean': mean_vals,
            'lower': min_vals,
            'q25': q25,
            'q75': q75,
            'upper': max_vals,
        })

        p = figure(
            title=f"{titles[index_name]}: Forecast start: {date.strftime('%Y-%m-%d')}",
            x_axis_label='Forecast valid on',
            y_axis_label=f'{index_name.upper()} values',
            width=500,
            height=400,
            x_range=(forecast_periods[0], forecast_periods[-1]),
            y_range=(min(np.min(min_vals), thresholds[index_name]) - 1,
                    max(np.max(max_vals), thresholds[index_name]) + 1)
        )

        band_full = Band(base='forecast_period', lower='lower', upper='upper', source=source,
                         level='underlay', fill_alpha=0.3, fill_color='lightblue')
        p.line(x=[None], y=[None], line_color='lightblue', line_width=8, alpha=0.3, legend_label='Range')
        p.add_layout(band_full)

        band_iqr = Band(base='forecast_period', lower='q25', upper='q75', source=source,
                        level='underlay', fill_alpha=0.8, fill_color='lightblue')
        p.line(x=[None], y=[None], line_color='lightblue', line_width=8, alpha=0.8, legend_label="P25–P75")
        p.add_layout(band_iqr)

        p.line('forecast_period', 'mean', source=source, line_width=3, color='navy', legend_label='Mean')

        for col in pivot.columns:
            p.line(forecast_periods, pivot[col].values, line_color='gray', line_alpha=0.3)

        p.legend.click_policy = "hide"
        p.xaxis.formatter = DatetimeTickFormatter(
            days="%Y-%m-%d", months="%Y-%m-%d", years="%Y-%m-%d"
        )
        p.xaxis.minor_tick_line_color = None

        ref_line = Span(location=thresholds[index_name], dimension='width', line_color='black',
                        line_width=10, line_alpha=0.4, line_dash='dashed')
        p.add_layout(ref_line)

        return p

    def bokeh_plot_forecast_ensemble_mean(self, date, plot_width=500):
        """Generate monsoon indices ensemble plots for all lead times.

        Loads wind forecast data, computes monsoon indices (SWMI1, SWMI2, NEMI, NEMO),
        saves as CSV, and creates gridded HTML visualization.

        Parameters
        ----------
        date : datetime.datetime
            Forecast initialization date
        plot_width : int, optional
            Plot width in pixels (default 500)
        """
        _ = plot_width
        model = self.config_values['model']
        date_label = date.strftime("%Y%m%d")
        members = [str('%03d' % mem) for mem in range(36)]
        fc_times = [str('%03d' % fct) for fct in np.arange(0, 174, 24)]

        ugrd850_cubes, ugrd925_cubes, vgrd925_cubes = [], [], []
        available_members = []

        model_config = load_global_config().get(model, {})
        raw_dir = model_config.get("raw", "/tmp/salmon_raw/mogreps")

        for mem in members:
            mog_files = [
                os.path.join(raw_dir, date.strftime("%Y%m%d"), mem, f'englaa_pd{fct}.pp')
                for fct in fc_times
            ]
            mog_files = [mf for mf in mog_files if os.path.exists(mf)]
            mog_files.sort()

            if not mog_files:
                logger.warning('No forecast files found for member %s', mem)
                continue

            realiz_coord = iris.coords.DimCoord([int(mem)], standard_name='realization',
                                                var_name='realization')

            ugrd850 = self.read_winds_correctly(mog_files, 'x_wind', pressure_level=850)
            ugrd925 = self.read_winds_correctly(mog_files, 'x_wind', pressure_level=925)
            vgrd925 = self.read_winds_correctly(mog_files, 'y_wind', pressure_level=925)

            for cube in [ugrd850, ugrd925, vgrd925]:
                self._remove_coords_if_present(cube, ['forecast_reference_time', 'realization', 'time'])

            for cube in [ugrd850, ugrd925, vgrd925]:
                cube.add_aux_coord(realiz_coord)

            ugrd850_cubes.append(ugrd850)
            ugrd925_cubes.append(ugrd925)
            vgrd925_cubes.append(vgrd925)
            available_members.append(mem)

        if not ugrd850_cubes:
            logger.error('No wind cubes available for indices computation')
            return

        ugrd850_cubes = iris.cube.CubeList(ugrd850_cubes).merge_cube()
        ugrd925_cubes = iris.cube.CubeList(ugrd925_cubes).merge_cube()
        vgrd925_cubes = iris.cube.CubeList(vgrd925_cubes).merge_cube()

        dates = [date + datetime.timedelta(hours=int(fct)) for fct in fc_times]

        indices_df = self.compute_indices(dates, available_members, ugrd850_cubes, ugrd925_cubes, vgrd925_cubes)

        thresholds = {'nemi': -2.5, 'nemo': -2.5, 'swmi1': 0.0, 'swmi2': 0.0}
        titles = {
            'nemi': 'NEMI (Northeast Monsoon Index)',
            'nemo': 'NEMO (Northeast Monsoon Onset)',
            'swmi1': 'SWMI1: Southwest Monsoon Index 1',
            'swmi2': 'SWMI2: Southwest Monsoon Index 2'
        }

        csv_file_dir = os.path.join(self.config_values[f'{model}_indices_processed_dir'], date_label)
        os.makedirs(csv_file_dir, exist_ok=True)
        csv_file = os.path.join(csv_file_dir, f"indices_{date_label}.csv")
        indices_df.to_csv(csv_file, index=False)
        logger.info('Saved %s', csv_file)

        plots = [self.plot_index_ens(date, indices_df, thresholds, titles, index_name=idx)
                 for idx in thresholds.keys()]
        grid = gridplot([plots[i:i + 2] for i in range(0, len(plots), 2)])

        html_file_dir = os.path.join(self.config_values[f'{model}_indices_plot_ens'], date_label)
        os.makedirs(html_file_dir, exist_ok=True)
        html_file = os.path.join(html_file_dir, f'Monsoon_indices_{date_label}.html')
        output_file(html_file)
        save(grid)
        logger.info('Plotted %s', html_file)

        json_file = os.path.join(self.config_values[f'{model}_indices_plot_ens'],
                                 f'{model}_indices_plot_dates.json')
        self.write_dates_json(date, json_file)
