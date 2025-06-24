import os, sys
import glob
import iris
import datetime
import numpy as np
import configparser
import json
import pandas as pd
import warnings
from itertools import chain
from bokeh.plotting import figure, show, save, output_file
from bokeh.palettes import Category10
from bokeh.models import Legend
from bokeh.models import Band, ColumnDataSource
from bokeh.models import DatetimeTickFormatter
from bokeh.models import Span
from bokeh.layouts import gridplot

class ColdSurgeDisplay:
    def __init__(self, model, config_values):
        self.config_values = config_values
        self.model = model

        # Navigate to the parent directory
        self.parent_dir = '/home/users/prince.xavier/MJO/SALMON/INDICES'

    def read_winds_correctly(self, data_files, varname, pressure_level=None):
        """
            Loads and processes wind data from a list of NetCDF files using Iris.

            This method:
              - Sorts the input files to ensure temporal order.
              - Loads a specified variable from each file.
              - Optionally extracts a specific pressure level.
              - Averages over time if the data has 3-hourly resolution.
              - Ensures forecast_period and time coordinates have bounds.
              - Equalises cube attributes to allow merging.
              - Merges all processed cubes into a single cube.
              - Restricts the final cube to a spatial region over Southeast Asia (0–30°N, 90–160°E).

            Parameters
            ----------
            data_files : list of str
                List of paths to NetCDF files to be read. These should contain the specified wind variable.
            varname : str
                The name of the wind variable to load (e.g., 'ugrd', 'vgrd').
            pressure_level : float, optional
                The pressure level (in hPa) to extract (e.g., 850.0). If None, the full vertical dimension is retained.

            Returns
            -------
            iris.cube.Cube
                A merged Iris cube containing the processed wind data, spatially limited to the region
                0–30° latitude and 90–160° longitude.

            Notes
            -----
            - If the cube has 3D shape (e.g., time, lat, lon), it is averaged over time.
            - Bounds for `forecast_period` and `time` are manually added if missing to ensure compatibility.
            - The method forcibly resets `cell_methods` to avoid merge failures due to inconsistencies.
            """
        data_files.sort()
        # print(data_files)
        cubes = []
        for data_file in data_files:
            # Some files have 3 hourly wind data which need to be averaged
            cube = iris.load_cube(data_file, varname)
            if pressure_level is not None:
                cube = cube.extract(iris.Constraint(pressure=pressure_level))
            if len(cube.shape) == 3:
                cube = cube.collapsed('time', iris.analysis.MEAN)
            if cube.coord('forecast_period').bounds is None:
                bounds = [[cube.coord('forecast_period').points[0] - 1., cube.coord('forecast_period').points[0] + 1.]]
                cube.coord('forecast_period').bounds = bounds
            if cube.coord('time').bounds is None:
                bounds = [[cube.coord('time').points[0] - 1., cube.coord('time').points[0] + 1.]]
                cube.coord('time').bounds = bounds
            cubes.append(cube)

        # Equalise attributes
        iris.util.equalise_attributes(cubes)
        # Merge was failing because of some stupid cell_methods mismatch (Iris is evil!)
        for cube in cubes:
            cube.cell_methods = ()

        cubes = iris.cube.CubeList(cubes).merge_cube()
        return cubes.intersection(latitude=(0, 30), longitude=(90, 160))


    def write_dates_json(self, date, json_file):

        # Add a new date to the list
        new_date = date.strftime('%Y%m%d')

        if not os.path.exists(json_file):
            with open(json_file, 'w') as jfile:
                json.dump([new_date], jfile)

        # Load existing dates from dates.json
        with open(json_file, 'r') as file:
            existing_dates = json.load(file)


        # Check if the value exists in the list
        if new_date not in existing_dates:
            # Append the value if it doesn't exist
            existing_dates.append(new_date)

        existing_dates.sort()

        # Save the updated list back to dates.json
        with open(json_file, 'w') as file:
            json.dump(existing_dates, file, indent=2)

        print(f"The {json_file} file has been updated. New date added: {new_date}")

    def compute_indices(self, dates, members, ugrd850_cubes, ugrd925_cubes, vgrd925_cubes):
        """
            Computes regional monsoon-related indices from wind data cubes at specified pressure levels.

            This method calculates the following indices:
              - SWMI1 (Southwest Monsoon Index 1): Difference in regional mean u850 between two boxes.
              - SWMI2 (Southwest Monsoon Index 2): Mean u850 over the region 5–10°N, 100–115°E.
              - NEMI (Northeast Monsoon Index): Average of u850 and u925 over the region 3.75–6.25°N, 102.5–105°E.
              - NEMO (Northeast Monsoon Onset): Mean v925 over the region 5–15°N, 107–115°E.

            The output is returned as a pandas DataFrame in long format with one row per forecast period and ensemble realization.

            Parameters
            ----------
            dates : list
                List of forecast periods (usually from `cube.coord('forecast_period').points`).
            ugrd850_cubes : iris.cube.Cube
                Eastward wind (u-wind) component at 850 hPa pressure level.
            ugrd925_cubes : iris.cube.Cube
                Eastward wind (u-wind) component at 925 hPa pressure level.
            vgrd925_cubes : iris.cube.Cube
                Northward wind (v-wind) component at 925 hPa pressure level.

            Returns
            -------
            indices_df : pandas.DataFrame
                A long-format DataFrame with columns: 'forecast_period', 'realization', 'swmi1', 'swmi2', 'nemi', 'nemo'.

            References
            ----------
            - SWMI1 (Southwest Monsoon Index 1):
                Diong et al. (2015). *The Definitions of the Southwest Monsoon Climatological Onset and Withdrawal over Malaysian Region*.
                [ResearchGate](https://www.researchgate.net/publication/322634656_The_Definitions_of_the_Southwest_Monsoon_Climatological_Onset_and_Withdrawal_over_Malaysian_Region)

            - SWMI2 (Southwest Monsoon Index 2):
                Diong et al. (2019). *An objective definition of summer monsoon onset in the northwestern maritime continent*.
                https://doi.org/10.1002/joc.6075

            - NEMI (Northeast Monsoon Index):
                Moten et al. (2014). *Statistics of Northeast Monsoon Onset Withdrawal and Cold Surges in Malaysia*.
                [ResearchGate PDF](https://www.researchgate.net/profile/Diong-Jeong-Yik/publication/326890761_Satistics_of_Northeast_Monsoon_Onset_Withdrawal_and_Cold_Surges_in_Malaysia/links/5b6a3c9ea6fdcc87df6d8a63/Satistics-of-Northeast-Monsoon-Onset-Withdrawal-and-Cold-Surges-in-Malaysia.pdf)

            - NEMO (Northeast Monsoon Onset):
                Sheeba et al. (2022). *Objective determination of the winter monsoon onset dates and its interannual variability in Malaysia*.
                https://doi.org/10.1002/joc.7895

            - Meridional Surge (MESI):
                Chang et al. (2005). *Synoptic disturbances over the equatorial South China Sea and western Maritime Continent during boreal winter*.
                https://doi.org/10.1175/MWR-2868.1

            - Easterly Surge:
                Hai et al. (2017). *Extreme rainstorms that caused devastating flood over the east coast of Peninsular Malaysia during November and December 2014*.
                https://doi.org/10.1175/WAF-D-16-0160.1
            """

        # Compute SWMI: difference of two regional means
        # Box1: 90–130E, 5–15N
        # Box2: 100.75–103.25E, 1.75–4.25N
        box1 = ugrd850_cubes.intersection(latitude=(5, 15), longitude=(90, 130)).collapsed(['latitude', 'longitude'],
                                                                                           iris.analysis.MEAN)
        box2 = ugrd850_cubes.intersection(latitude=(1.75, 4.25), longitude=(100.75, 103.25)).collapsed(
            ['latitude', 'longitude'], iris.analysis.MEAN)

        swmi1 = box2 - box1

        ################################################
        # SWMI2 (Southwest Monsoon Index 2)
        # swmi2=u850.sel(lon=slice(100,115),lat=slice(5,10)).mean(dim=['lon','lat'])
        swmi2 = ugrd850_cubes.intersection(latitude=(5, 10), longitude=(100, 115)).collapsed(['latitude', 'longitude'],
                                                                                             iris.analysis.MEAN)

        #############################################
        # NEMI (Northeast Monsoon Index)
        # n850=u850.sel(lon=slice(102.5,105),lat=slice(3.75,6.25)).mean(dim=['lon','lat'])
        # n925=u925.sel(lon=slice(102.5,105),lat=slice(3.75,6.25)).mean(dim=['lon','lat'])
        n850 = ugrd850_cubes.intersection(latitude=(3.75, 6.25), longitude=(102.5, 105)).collapsed(
            ['latitude', 'longitude'], iris.analysis.MEAN)
        n925 = ugrd925_cubes.intersection(latitude=(3.75, 6.25), longitude=(102.5, 105)).collapsed(
            ['latitude', 'longitude'], iris.analysis.MEAN)
        nemi = 0.5 * (n850 + n925)

        # NEMO (Northeast Monsoon Onset)
        # nemo=v925.sel(lon=slice(107,115),lat=slice(5,15)).mean(dim=['lon','lat'])
        nemo = vgrd925_cubes.intersection(latitude=(5, 15), longitude=(107, 115)).collapsed(['latitude', 'longitude'],
                                                                                            iris.analysis.MEAN)

        # Flatten data into long format
        data_records = []

        for i, fp in enumerate(dates):
            for j, r in enumerate(members):
                data_records.append({'forecast_period': fp, 'realization': r,
                                     'swmi1': swmi1.data[j, i],
                                     'swmi2': swmi2.data[j, i],
                                     'nemi': nemi.data[j, i],
                                     'nemo': nemo.data[j, i]})

        indices_df = pd.DataFrame(data_records)
        return indices_df

    def plot_index_ens(self, date, indices_df, thresholds, titles, index_name=None):
        # Pivot the dataframe to get realizations as columns
        pivot = indices_df.pivot(index='forecast_period', columns='realization', values=index_name)

        # Compute statistics across realizations
        forecast_periods = pivot.index.values
        mean_vals = pivot.mean(axis=1).values
        min_vals = pivot.min(axis=1).values
        max_vals = pivot.max(axis=1).values

        # Optional: compute percentiles instead of min/max
        q25 = pivot.quantile(0.25, axis=1).values
        q75 = pivot.quantile(0.75, axis=1).values

        # Create a ColumnDataSource
        source = ColumnDataSource(data={
            'forecast_period': forecast_periods,
            'mean': mean_vals,
            'lower': min_vals,
            'q25': q25,
            'q75': q75,
            'upper': max_vals,
        })

        # Create plot
        p = figure(title=f"{titles[index_name]}: Forecast start: {date.strftime('%Y-%m-%d')}",
                   x_axis_label='Forecast valid on',
                   y_axis_label=f'{index_name.upper()} values',
                   width=600, height=400, x_range=(forecast_periods[0], forecast_periods[-1]),
                   y_range=(min(np.min(min_vals), thresholds[index_name]) - 1,
                            max(np.max(max_vals), thresholds[index_name]) + 1), )

        # Add band (spread area)
        band = Band(base='forecast_period', lower='lower', upper='upper', source=source,
                    level='underlay', fill_alpha=0.3, fill_color='lightblue')
        # Add dummy line just for legend
        p.line(x=[None], y=[None], line_color='lightblue', line_width=8, alpha=0.3, legend_label='Range')

        p.add_layout(band)

        # Add band (spread area)
        band = Band(base='forecast_period', lower='q25', upper='q75', source=source,
                    level='underlay', fill_alpha=0.8, fill_color='lightblue')

        p.line(x=[None], y=[None], line_color='lightblue', line_width=8, alpha=0.8, legend_label="P25–P75")
        p.add_layout(band)

        # Add mean line
        p.line('forecast_period', 'mean', source=source, line_width=3, color='navy', legend_label='Mean')

        # Optionally plot individual ensemble members as faint lines
        for col in pivot.columns:
            p.line(forecast_periods, pivot[col].values, line_color='gray', line_alpha=0.3)

        # p.legend.location = "top_left"
        p.legend.click_policy = "hide"

        # Custom date format on x-axis
        p.xaxis.formatter = DatetimeTickFormatter(
            days="%Y-%m-%d",
            months="%Y-%m-%d",
            years="%Y-%m-%d"
        )
        p.xaxis.minor_tick_line_color = None  # Hides minor ticks
        ref_line = Span(location=thresholds[index_name], dimension='width', line_color='black', line_width=10, line_alpha=0.4,
                        line_dash='dashed')
        p.add_layout(ref_line)

        return p
    def bokeh_plot_forecast_ensemble_mean(self, date, plot_width=500):
        date_label = date.strftime("%Y%m%d")
        members = [str('%03d' % mem) for mem in range(36)]

        fc_times = [str('%03d' % fct) for fct in np.arange(0, 174, 24)]
        time_coord = iris.coords.DimCoord(
            points=[int(t) for t in fc_times], standard_name='time',
            units=f'hours since {date.strftime("%Y-%m-%d")}')

        ugrd850_cubes = []
        ugrd925_cubes = []
        vgrd925_cubes = []

        for mem in members:
            print(mem)
            mog_files = [os.path.join(self.config_values['mogreps_raw_dir'],
                                      date.strftime("%Y%m%d"), mem, f'englaa_pd{fct}.pp')
                         for fct in fc_times]
            mog_files = [mog_file for mog_file in mog_files if os.path.exists(mog_file)]
            mog_files.sort()

            realiz_coord = iris.coords.DimCoord([int(mem)], standard_name='realization',
                                                var_name='realization')

            ugrd850 = self.read_winds_correctly(mog_files, 'x_wind', pressure_level=850)
            ugrd925 = self.read_winds_correctly(mog_files, 'x_wind', pressure_level=925)
            vgrd925 = self.read_winds_correctly(mog_files, 'y_wind', pressure_level=925)

            # Massaging the data for mergine
            for coord in ['forecast_reference_time', 'realization', 'time']:
                ugrd850.remove_coord(coord) if ugrd850.coords(coord) else None
            for coord in ['forecast_reference_time', 'realization', 'time']:
                ugrd925.remove_coord(coord) if ugrd925.coords(coord) else None
            for coord in ['forecast_reference_time', 'realization', 'time']:
                vgrd925.remove_coord(coord) if vgrd925.coords(coord) else None

            ugrd850.add_aux_coord(realiz_coord)
            ugrd925.add_aux_coord(realiz_coord)
            vgrd925.add_aux_coord(realiz_coord)

            # cube.coord('forecast_period').points = cube.coord('forecast_period').bounds[:, 1]
            ugrd850_cubes.append(ugrd850)
            ugrd925_cubes.append(ugrd925)
            vgrd925_cubes.append(vgrd925)

        ugrd850_cubes = iris.cube.CubeList(ugrd850_cubes).merge_cube()
        ugrd925_cubes = iris.cube.CubeList(ugrd925_cubes).merge_cube()
        vgrd925_cubes = iris.cube.CubeList(vgrd925_cubes).merge_cube()

        # Dates list in the forecast data
        dates = time_coord.units.num2date(time_coord.points)
        dates = [datetime.datetime(t.year, t.month, t.day) for t in dates]

        # Call the indices calculation
        indices_df = self.compute_indices(dates, members, ugrd850_cubes, ugrd925_cubes, vgrd925_cubes)

        # Thresholds to  define the levels of monsoon activity to plot as a line in the plots
        thresholds = {'nemi': -2.5, 'nemo': -2.5, 'swmi1': 0.0, 'swmi2': 0.0}
        titles = {'nemi': 'NEMI (Northeast East Monsoon Index)', 'nemo': 'NEMO (Northeast Monsoon Onset)',
                  'swmi1': 'SWMI1: Southwest Monsoon Index 1', 'swmi2': 'SWMI2: Southwest Monsoon Index 2'}

        # Saving the indices as csv file
        ################################
        csv_file_dir = os.path.join(self.config_values[f'{self.model}_indices_processed_dir'], date_label)
        if not os.path.exists(csv_file_dir):
            os.makedirs(csv_file_dir)

        csv_file = os.path.join(csv_file_dir, f"indices_{date_label}.csv")
        indices_df.to_csv(csv_file, index=False)
        print(f'Saved {csv_file}')
        ################################

        # Plotting the indices
        ################################
        plots = []
        for index_name in thresholds.keys():
            plots.append(self.plot_index_ens(date, indices_df, thresholds, titles, index_name=index_name))

        grid = gridplot([plots[i:i + 2] for i in range(0, len(plots), 2)])
        # show(plot)

        html_file_dir = os.path.join(self.config_values[f'{self.model}_indices_plot_ens'], date_label)
        if not os.path.exists(html_file_dir):
            os.makedirs(html_file_dir)

        html_file = os.path.join(html_file_dir,
                                              f'Monsoon_indices_{date_label}.html')
        output_file(html_file)
        save(grid)
        print('Plotted %s' % html_file)
        ################################

        # Add the date to a JSON file for the javascript
        ################################
        json_file = os.path.join(self.config_values[f'{self.model}_indices_plot_ens'],
                                 f'{self.model}_indices_plot_dates.json')
        self.write_dates_json(date, json_file)
        ################################


if __name__ == '__main__':
    today = datetime.date.today()
    #yesterday = today - datetime.timedelta(days=1)
    yesterday = datetime.datetime(2021, 9, 30, 12)

    #bokeh_plot_forecast_ensemble_mean(yesterday)
    #bokeh_plot_forecast_probability_precip(yesterday, precip_thresholds=[10])
    #bokeh_plot_forecast_probability_speed(yesterday, speed_thresholds=[5])