import os, sys
import glob
import iris
import datetime
import numpy as np
import configparser
import json
import iris.coord_categorisation
import pandas as pd
from bokeh.plotting import figure, show, save, output_file
from bokeh.models import ColumnDataSource, Patches, Plot, Title
from bokeh.models import HoverTool
from bokeh.models import Range1d, LinearColorMapper, ColorBar
from bokeh.models import GeoJSONDataSource
from bokeh.palettes import GnBu9, Magma6, Greys256, Greys9, GnBu9, RdPu9, TolRainbow12
from bokeh.palettes import Iridescent23, TolYlOrBr9, Bokeh8, Greys9, Blues9
from bokeh.layouts import column, row, Spacer
from bokeh.models import CheckboxGroup, CheckboxButtonGroup, CustomJS, Button
from bokeh.models import Legend, LegendItem
from skimage import measure

class EqWavesDisplay:
    def __init__(self, model, config_values_analysis, config_values):
        """
        Initializes the MOGProcess class with configuration values.

        Args:
        model (str): The model section in the configuration file.
        """
        self.config_values_analysis = config_values_analysis
        self.config_values = config_values
        """
        Initializes the EqWavesDisplay class with configuration values.

        Args:
            model (str): The model section in the configuration file.
        """
        self.model = model
        self.parent_dir = '/home/users/prince.xavier/MJO/SALMON/EQWAVES'
        self.ntimes_total = 360
        self.ntimes_analysis = 332
        self.ntimes_forecast = 28
        self.wave_names = np.array(['Precip', 'Kelvin', 'WMRG', 'R1', 'R2'])
        self.pressures = ['850']
        # Plot thresholds for probabilities
        self.thresholds = {'Precip': 5,
                           'Kelvin_850': -1 * 1e-6, 'Kelvin_200': -2 * 1e-6,
                           'WMRG_850': -1 * 1e-6, 'WMRG_200': -2 * 1e-6,
                           'R1_850': 5 * 1e-6, 'R1_200': 2 * 1e-6,
                           'R2_850': 3 * 1e-6, 'R2_200': 5 * 1e-6}
        self.times2plot = [t for t in range(-96, 174, 6)]
        # Map outline
        self.map_outline_json_file = os.path.join(self.parent_dir,
                                                  'display', 'custom.geo.json')
        self.plot_width = 1300

    def prepare_calendar(self, cube):
        # Setting up the dates on data
        for coord_name, coord_func in [('year', iris.coord_categorisation.add_year),
                                       ('month_number', iris.coord_categorisation.add_month_number),
                                       ('day_of_month', iris.coord_categorisation.add_day_of_month),
                                       ('hour', iris.coord_categorisation.add_hour)]:
            if not cube.coords(coord_name):
                coord_func(cube, 'time', name=coord_name)
        return cube

    def create_dates_dt(self, cube):
        cube = self.prepare_calendar(cube)
        cube_dates_dt = [datetime.datetime(y, m, d, h) for y, m, d, h in zip(cube.coord('year').points,
                                                                             cube.coord('month_number').points,
                                                                             cube.coord('day_of_month').points,
                                                                             cube.coord('hour').points)]
        return cube_dates_dt

    def write_dates_json(self, date, json_file):

        # Add a new date to the list
        new_date = date.strftime('%Y%m%d_%H')

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

    def bokeh_plot2html(self, shade_var=None, contour_var=None, figure_tite=None,
                        shade_cbar_title=None, contour_cbar_title=None, html_file='test.html',
                        shade_levels=np.arange(0.1, 1.1, 0.1),
                        contour_levels=np.arange(0.5, 1.2, 0.2)):

        x_range = (0, 180)  # could be anything - e.g.(0,1)
        y_range = (-24, 24)


        width = self.plot_width
        aspect = (max(x_range) - min(x_range)) / (max(y_range) - min(y_range))
        height = int(width / (0.6 * aspect))
        #width = 1200
        #height = 500
        #print(width, height)
        plot = figure(height=height, width=width, x_range=x_range, y_range=y_range,
                      tools=["pan, reset, save, wheel_zoom, hover"],
                      x_axis_label='Longitude', y_axis_label='Latitude', aspect_scale=4,
                      title=figure_tite, tooltips=[("Lat", "$y"), ("Lon", "$x"), ("Value", "@image")])

        #plot.title.text_font_size = "14pt"
        # Make background and border transparent
        plot.background_fill_color = None  # No background color (transparent)
        plot.border_fill_color = None  # No border color (transparent)
        plot.outline_line_color = None  # No outline

        color_mapper_z = LinearColorMapper(palette='Iridescent23', low=shade_levels.min(), high=shade_levels.max())
        color_bar = ColorBar(color_mapper=color_mapper_z, major_label_text_font_size="12pt",
                             label_standoff=6, border_line_color=None, orientation="horizontal",
                             location=(0, 0), width=400, title=shade_cbar_title, title_text_font_size="12pt")

        if shade_var is not None:
            plot.image(image=[shade_var.data], x=0, y=-24,
                       dw=360, dh=48, alpha=0.8,
                       color_mapper=color_mapper_z)
        plot.add_layout(color_bar, 'below')

        if contour_var is not None:
            lons, lats = np.meshgrid(contour_var.coord('longitude').points, contour_var.coord('latitude').points)

            contour_renderer = plot.contour(lons, lats, contour_var.data, contour_levels, fill_color=None,
                                            fill_alpha=0.3,
                                            line_color=Bokeh8, line_alpha=0.5, line_width=5)
            colorbar = contour_renderer.construct_color_bar(major_label_text_font_size="12pt",
                                                        orientation="horizontal", location=(-500, -135), width=400,
                                                        title=contour_cbar_title, title_text_font_size="12pt")
            plot.add_layout(colorbar, "right")

        with open(self.map_outline_json_file, "r") as f:
            countries = GeoJSONDataSource(geojson=f.read())

        plot.patches("xs", "ys", color=None, line_color="grey", source=countries, alpha=0.75)

        output_file(html_file)
        save(plot)
        print('Plotted %s' % html_file)

    def get_skimage_contour_paths(self, lons, lats, cube_data, levels=[0.5, 0.75]):
        paths_x, paths_y = [], []
        for level in levels:
            contours = measure.find_contours(cube_data, level)
            for contour in contours:
                paths_x.append(contour[:, 1] + min(lons))
                paths_y.append(contour[:, 0] + min(lats))
        return paths_x, paths_y
    def bokeh_plot_allwaves2html(self, wave_timestep_dic, pressure_level, figure_tite=None,
                        shade_cbar_title=None, contour_cbar_title=None, html_file='test.html',
                        shade_levels=np.arange(0.1, 1.1, 0.1),
                        contour_levels=np.arange(0.4, 1.2, 0.2)):

        x_range = (0, 180)  # could be anything - e.g.(0,1)
        y_range = (-24, 24)
        contour_alpha = 0.5
        width = self.plot_width
        shade_cbar_title = 'Probability of precipitation >=5 mm/day'
        aspect = (max(x_range) - min(x_range)) / (max(y_range) - min(y_range))
        height = int(width / (0.65 * aspect))


        # Prepare the initial image source
        precip_source = ColumnDataSource(data=dict(Precip=[wave_timestep_dic['Precip']]))
        contour_levels = [0.66]  # [0.5, 0.7, 0.9]
        #print(wave_timestep_dic.keys())

        # Prepare contour paths using functions for Kelvin and WMRG
        lats, lons = wave_timestep_dic['latitude'], wave_timestep_dic['longitude']
        contour_kelvin_x, contour_kelvin_y = self.get_skimage_contour_paths(lons, lats,
                                                                            wave_timestep_dic[f'Kelvin_{pressure_level}'],
                                                                       levels=contour_levels)
        contour_wmrg_x, contour_wmrg_y = self.get_skimage_contour_paths(lons, lats, wave_timestep_dic[f'WMRG_{pressure_level}'],
                                                                   levels=contour_levels)
        contour_R1_x, contour_R1_y = self.get_skimage_contour_paths(lons, lats, wave_timestep_dic[f'R1_{pressure_level}'],
                                                               levels=contour_levels)
        contour_R2_x, contour_R2_y = self.get_skimage_contour_paths(lons, lats, wave_timestep_dic[f'R2_{pressure_level}'],
                                                               levels=contour_levels)

        # Create separate ColumnDataSources for each contour field
        kelvin_source = ColumnDataSource(data=dict(xs=contour_kelvin_x, ys=contour_kelvin_y))
        wmrg_source = ColumnDataSource(data=dict(xs=contour_wmrg_x, ys=contour_wmrg_y))
        r1_source = ColumnDataSource(data=dict(xs=contour_R1_x, ys=contour_R1_y))
        r2_source = ColumnDataSource(data=dict(xs=contour_R2_x, ys=contour_R2_y))

        plot = figure(height=height, width=width, x_range=x_range, y_range=y_range,
                      tools=["pan, reset, save, wheel_zoom, hover"],
                      x_axis_label='Longitude', y_axis_label='Latitude', aspect_scale=4,
                      title=figure_tite)

        # Create a color mapper for the image
        color_mapper_z = LinearColorMapper(palette='Iridescent23', low=0.5, high=1)
        color_bar = ColorBar(color_mapper=color_mapper_z, major_label_text_font_size="12pt",
                             label_standoff=6, border_line_color=None, orientation="horizontal",
                             location=(0, 0), width=400, title=shade_cbar_title, title_text_font_size="12pt")
        image_renderer = plot.image('Precip', source=precip_source, x=0, y=-24, dw=360, dh=48, alpha=0.8,
                                    color_mapper=color_mapper_z)
        plot.add_layout(color_bar, 'below')


        with open(self.map_outline_json_file, "r") as f:
            countries = GeoJSONDataSource(geojson=f.read())

        plot.patches("xs", "ys", color=None, line_color="grey", source=countries, alpha=0.75)

        # Add empty MultiLine renderers for contours
        kelvin_renderer = plot.multi_line(xs='xs', ys='ys', source=kelvin_source, line_width=4, color="blue",
                                          alpha=contour_alpha)
        wmrg_renderer = plot.multi_line(xs='xs', ys='ys', source=wmrg_source, line_width=4, color="green",
                                        alpha=contour_alpha)
        r1_renderer = plot.multi_line(xs='xs', ys='ys', source=r1_source, line_width=4, color="red",
                                      alpha=contour_alpha)
        r2_renderer = plot.multi_line(xs='xs', ys='ys', source=r2_source, line_width=4, color="orange",
                                      alpha=contour_alpha)

        # Create Legend items manually
        legend_items = [
            LegendItem(label="Kelvin convergence", renderers=[kelvin_renderer]),
            LegendItem(label="WMRG convergence", renderers=[wmrg_renderer]),
            LegendItem(label="n=1 Rossby cyclonic vorticity", renderers=[r1_renderer]),
            LegendItem(label="n=2 Rossby cyclonic vorticity", renderers=[r2_renderer]),
        ]

        # Create a Legend and set its properties
        legend = Legend(items=legend_items, title="Click to show/hide (p >= 0.5)", label_text_font_size="10pt",
                        title_text_font_size="11pt",
                        location=(0, 0.5), background_fill_alpha=0.75)
        legend.click_policy = "hide"  # Allow toggling visibility on click

        # Add legend to the plot (set it outside the main plot area)
        plot.add_layout(legend)

        # Create CheckboxGroup to select multiple fields
        # checkbox_group = CheckboxGroup(labels=["Kelvin", "WMRG", "R1", "R2"], active=[0, 1, 2, 3], height=100, width=500)
        checkbox_group = CheckboxButtonGroup(labels=["Kelvin", "WMRG", "n=1 Rossby", "n=2 Rossby"], active=[0, 1, 2, 3])

        # Create a "Clear All" button
        clear_button = Button(label="Clear All", button_type="danger")

        # JavaScript callback to toggle visibility of contour lines based on selection
        checkbox_callback = CustomJS(args=dict(kelvin_renderer=kelvin_renderer, wmrg_renderer=wmrg_renderer,
                                               r1_renderer=r1_renderer, r2_renderer=r2_renderer,
                                               checkbox=checkbox_group), code="""
            // Set visibility based on checkbox selection
            kelvin_renderer.visible = checkbox.active.includes(0); // Kelvin is label 0
            wmrg_renderer.visible = checkbox.active.includes(1); // WMRG is label 1
            r1_renderer.visible = checkbox.active.includes(2); // R1 is label 2
            r2_renderer.visible = checkbox.active.includes(3); // R2 is label 3
        """)

        # Attach the callback to checkbox group
        checkbox_group.js_on_change('active', checkbox_callback)

        # JavaScript callback for the "Clear All" button
        clear_button_callback = CustomJS(args=dict(checkbox=checkbox_group), code="""
            // Clear all active checkboxes
            checkbox.active = [];
            checkbox.change.emit();
        """)
        # Link clear button with its callback
        clear_button.js_on_click(clear_button_callback)

        spacer = Spacer(width=50)  # Adjust width for desired space
        layout = column(row(spacer, checkbox_group, clear_button), plot)

        # Set output HTML file
        output_file(html_file)

        # Save the plot layout as a static HTML file
        save(layout)

        print(html_file)
    def read_compute_ensemble_prob(self, wave_files, wname=None, pressure_level=None,
                                   contour_cbar_title=None):
        print(f'Computing probability for {wname}, {pressure_level}')
        wave_files = [file for file in wave_files if os.path.exists(file)]

        if len(wave_files) >= 3:
            ntimes = len(self.times2plot)
            wave_variable = iris.load_cube(wave_files)
            wave_variable = wave_variable[:, -ntimes:]
            if pressure_level:
                wave_variable = wave_variable.extract(iris.Constraint(pressure=float(pressure_level)))

                wave_variable = wave_variable.collapsed('realization', iris.analysis.PROPORTION,
                                                      function=lambda values: values >= self.thresholds[f'{wname}_{pressure_level}'])
            else:
                wave_variable = wave_variable.collapsed('realization', iris.analysis.PROPORTION,
                                                        function=lambda values: values >= self.thresholds[wname])
            wave_variable.rename(contour_cbar_title)
            return wave_variable
        else:
            print(f'Fewer than 3 files. Pass')
            pass

    def bokeh_plot_forecast_ensemble_probability_multiwave(self, date):
        mem_labels = [f'{fc:03}' for fc in range(0, 17)]

        str_hr = date.strftime('%H')
        date_label = date.strftime('%Y%m%d_%H')
        outfile_dir = os.path.join(self.config_values['mogreps_eqwaves_processed_dir'], date_label)

        html_file_dir = os.path.join(self.config_values['mogreps_eqwaves_plot_ens'], date_label)
        if not os.path.exists(html_file_dir):
            os.makedirs(html_file_dir)

        precip_files = [os.path.join(outfile_dir, f'precipitation_amount_combined_{date_label}Z_{mem}.nc') for mem in
                        mem_labels]
        print(precip_files)
        precip_files = [file for file in precip_files if os.path.exists(file)]

        # Precip cubes
        pr_cube = self.read_compute_ensemble_prob(precip_files, wname='Precip')
        ntimes = len(self.times2plot)

        # Assuming `times2plot` and `create_dates_dt(pr_cube)` are lists
        data = [(i, l, d) for i, l, d in zip(range(ntimes), self.times2plot, self.create_dates_dt(pr_cube))]

        # Creating DataFrame
        df = pd.DataFrame(data, columns=['Index', 'Lead', 'Date'])

        # Precip Prob
        shade_cbar_title = f"Probability of Precip >= {self.thresholds['Precip']}"
        precip_prob = self.read_compute_ensemble_prob(precip_files, wname='Precip',
                                                      contour_cbar_title=shade_cbar_title)
        for pressure_level in self.pressures:

            # Kelvin Prob
            wname = 'Kelvin'
            wave_files = [os.path.join(outfile_dir, f'div_wave_{wname}_{date_label}Z_{mem}.nc') for mem in
                          mem_labels]
            contour_cbar_title = f"Probability of {wname} divergence <= {self.thresholds[f'{wname}_{pressure_level}']:0.1e} s-1"
            kelvin_prob = self.read_compute_ensemble_prob(wave_files, wname=wname, pressure_level=pressure_level,
                                                          contour_cbar_title=contour_cbar_title)

            # WMRG Prob
            wname = 'WMRG'
            wave_files = [os.path.join(outfile_dir, f'div_wave_{wname}_{date_label}Z_{mem}.nc') for mem in
                          mem_labels]
            contour_cbar_title = f"Probability of {wname} divergence <= {self.thresholds[f'{wname}_{pressure_level}']:0.1e} s-1"
            wmrg_prob = self.read_compute_ensemble_prob(wave_files, wname=wname, pressure_level=pressure_level,
                                                          contour_cbar_title=contour_cbar_title)

            # R1 Prob
            wname = 'R1'
            wave_files = [os.path.join(outfile_dir, f'vort_wave_{wname}_{date_label}Z_{mem}.nc') for mem in
                          mem_labels]
            contour_cbar_title = f"Probability of {wname} vorticity >= {self.thresholds[f'{wname}_{pressure_level}']:0.1e} s-1"
            r1_prob = self.read_compute_ensemble_prob(wave_files, wname=wname, pressure_level=pressure_level,
                                                        contour_cbar_title=contour_cbar_title)
            # R2 Prob
            wname = 'R2'
            wave_files = [os.path.join(outfile_dir, f'vort_wave_{wname}_{date_label}Z_{mem}.nc') for mem in
                          mem_labels]
            contour_cbar_title = f"Probability of {wname} vorticity >= {self.thresholds[f'{wname}_{pressure_level}']:0.1e} s-1"
            r2_prob = self.read_compute_ensemble_prob(wave_files, wname=wname, pressure_level=pressure_level,
                                                      contour_cbar_title=contour_cbar_title)

            for lead in self.times2plot:
                wave_timestep_dic = {}

                t = df.loc[df['Lead'] == lead].Index.values[0]
                datetime_string = df['Date'].loc[df['Lead'] == lead].astype('O').tolist()[0].strftime(
                    '%Y/%m/%d %HZ')

                wave_timestep_dic['Precip'] = precip_prob[t].data
                wave_timestep_dic[f'Kelvin_{pressure_level}'] = kelvin_prob[t].data
                wave_timestep_dic[f'WMRG_{pressure_level}'] = wmrg_prob[t].data
                wave_timestep_dic[f'R1_{pressure_level}'] = r1_prob[t].data
                wave_timestep_dic[f'R2_{pressure_level}'] = r2_prob[t].data

                wave_timestep_dic['latitude'] = precip_prob[t].coord('latitude').points
                wave_timestep_dic['longitude'] = precip_prob[t].coord('longitude').points

                if int(lead) < 0:
                    figure_tite = f"Valid on {datetime_string} at T{lead}"
                else:
                    figure_tite = f"Valid on {datetime_string} at T+{lead}"

                # Plot just the contours
                html_file = os.path.join(html_file_dir,
                                         f'AllWaves_{pressure_level}_EnsProb_{date_label}Z_T{(lead)}h.html')

                self.bokeh_plot_allwaves2html(wave_timestep_dic, pressure_level,
                                     figure_tite=figure_tite,
                                     shade_cbar_title=shade_cbar_title, contour_cbar_title=None,
                                     html_file=html_file)
        json_file = os.path.join(self.config_values['mogreps_eqwaves_plot_ens'], f'{self.model}_eqw_ens_plot_dates.json')
        self.write_dates_json(date, json_file)


    def bokeh_plot_forecast_ensemble_probability(self, date):
        mem_labels = [f'{fc:03}' for fc in range(0, 17)]

        str_hr = date.strftime('%H')
        date_label = date.strftime('%Y%m%d_%H')
        outfile_dir = os.path.join(self.config_values['mogreps_eqwaves_processed_dir'], date_label)

        html_file_dir = os.path.join(self.config_values['mogreps_eqwaves_plot_ens'], date_label)
        if not os.path.exists(html_file_dir):
            os.makedirs(html_file_dir)

        # Realistically you will probably only want to write out say (T-4:T+7) so that
        # you can plot an animation of the last few days and the forecast
        # total of 45 time points
        # write_out_times = 45
        # This has been moved to the plotting step.

        precip_files = [os.path.join(outfile_dir, f'precipitation_amount_combined_{date_label}Z_{mem}.nc') for mem in
                        mem_labels]
        print(precip_files)
        precip_files = [file for file in precip_files if os.path.exists(file)]

        # Precip cubes
        pr_cube = iris.load_cube(precip_files)

        print(pr_cube)

        ntimes = len(self.times2plot)

        pr_cube = pr_cube[:, -ntimes:]
        # Assuming `times2plot` and `create_dates_dt(pr_cube)` are lists
        data = [(i, l, d) for i, l, d in zip(range(ntimes), self.times2plot, self.create_dates_dt(pr_cube))]

        # Creating DataFrame
        df = pd.DataFrame(data, columns=['Index', 'Lead', 'Date'])

        shade_var = pr_cube.collapsed('realization', iris.analysis.PROPORTION,
                                      function=lambda values: values > self.thresholds['precip'])
        shade_cbar_title = f"Probability of precipitation >= {self.thresholds['precip']} mm day-1"

        for lead in self.times2plot:
            t = df.loc[df['Lead'] == lead].Index.values[0]
            datetime_string = df['Date'].loc[df['Lead'] == lead].astype('O').tolist()[0].strftime(
                '%Y/%m/%d %HZ')
            if int(lead) < 0:
                figure_tite = f"Valid on {datetime_string} at T{lead}"
            else:
                figure_tite = f"Valid on {datetime_string} at T+{lead}"

            # Plot just the background shaded precip probability
            html_file = os.path.join(html_file_dir,
                                     f'Precip_NoLevel_EnsProb_{date_label}Z_T{(lead)}h.html')

            self.bokeh_plot2html(shade_var=shade_var[t], contour_var=None,
                                 figure_tite=figure_tite,
                                 shade_cbar_title=shade_cbar_title, contour_cbar_title=shade_cbar_title,
                                 html_file=html_file)


        for wname in self.wave_names:
            print(f'PLOTTING WAVE {wname}!!!!!!!!!!!!!!!!!!!!!!!!!')
            for pressure_level in self.pressures:
                if wname in ['Kelvin', 'WMRG']:
                    wave_files = [os.path.join(outfile_dir, f'div_wave_{wname}_{date_label}Z_{mem}.nc') for mem in
                                  mem_labels]
                    wave_files = [file for file in wave_files if os.path.exists(file)]
                    print(wave_files)
                    wave_variable = iris.load_cube(wave_files)
                    wave_variable = wave_variable.extract(iris.Constraint(pressure=float(pressure_level)))
                    wave_variable = wave_variable[:, -ntimes:]
                    contour_var = wave_variable.collapsed('realization', iris.analysis.PROPORTION,
                                                          function=lambda values: values <= self.thresholds[
                                                              f'{wname}_{pressure_level}'])
                    contour_cbar_title = f"Probability of {wname} divergence <= {self.thresholds[f'{wname}_{pressure_level}']:0.1e} s-1"
                elif wname in ['R1', 'R2']:
                    wave_files = [os.path.join(outfile_dir, f'vort_wave_{wname}_{date_label}Z_{mem}.nc') for mem in
                                  mem_labels]
                    wave_files = [file for file in wave_files if os.path.exists(file)]
                    print(wave_files)

                    wave_variable = iris.load_cube(wave_files)
                    wave_variable = wave_variable.extract(iris.Constraint(pressure=float(pressure_level)))
                    wave_variable = wave_variable[:, -ntimes:]
                    contour_var = wave_variable.collapsed('realization', iris.analysis.PROPORTION,
                                                          function=lambda values: values >= self.thresholds[
                                                              f'{wname}_{pressure_level}'])
                    contour_cbar_title = f"Probability of {wname} vorticity >= {self.thresholds[f'{wname}_{pressure_level}']:0.1e} s-1"

                for lead in self.times2plot:
                    t = df.loc[df['Lead'] == lead].Index.values[0]
                    datetime_string = df['Date'].loc[df['Lead'] == lead].astype('O').tolist()[0].strftime(
                        '%Y/%m/%d %HZ')
                    #if int(lead) < 0:
                    #    figure_tite = f"{shade_cbar_title};  {contour_cbar_title} \nValid on {datetime_string} at T{lead}"
                    #else:
                    #    figure_tite = f"{shade_cbar_title};  {contour_cbar_title} \nValid on {datetime_string} at T+{lead}"

                    if int(lead) < 0:
                        figure_tite = f"Valid on {datetime_string} at T{lead}"
                    else:
                        figure_tite = f"Valid on {datetime_string} at T+{lead}"

                    # Plot just the contours
                    html_file = os.path.join(html_file_dir,
                                             f'{wname}_{pressure_level}_EnsProb_{date_label}Z_T{(lead)}h.html')

                    self.bokeh_plot2html(shade_var=None, contour_var=contour_var[t],
                                         figure_tite=figure_tite,
                                         shade_cbar_title=shade_cbar_title, contour_cbar_title=contour_cbar_title,
                                         html_file=html_file)

        json_file = os.path.join(self.config_values['mogreps_eqwaves_plot_ens'], f'{self.model}_eqw_ens_plot_dates.json')
        self.write_dates_json(date, json_file)