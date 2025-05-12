import os, sys
import glob
import iris
import datetime
import numpy as np
import configparser
import json
from bokeh.plotting import figure, show, save, output_file
from bokeh.models import ColumnDataSource, Patches, Plot, Title
from bokeh.models import HoverTool
from bokeh.models import  Range1d, LinearColorMapper, ColorBar
from bokeh.models import GeoJSONDataSource
from bokeh.palettes import GnBu9, Magma6, Greys256, Greys9, GnBu9, RdPu9, TolRainbow12
from .bokeh_vector import vector

class ColdSurgeDisplay:
    def __init__(self, model, config_values):
        self.config_values = config_values
        self.model = model

        # Navigate to the parent directory
        self.parent_dir = '/home/users/prince.xavier/MJO/SALMON/COLDSURGE'

        # Get options in the 'analysis' section and store in the dictionary
        if self.model == 'glosea':
            self.xSkip = 2
            self.ySkip = 2
        elif self.model == 'mogreps':
            self.xSkip = 5
            self.ySkip = 5

        # Map outline
        self.map_outline_json_file = os.path.join(self.parent_dir,
                                                  'display', 'data', 'custom.geo.json')

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

    def plot_image_map(self, plot, cube, **kwargs):
        palette = kwargs['palette']
        if 'cbar_title' in kwargs and kwargs['cbar_title']:
            cbar_title = kwargs['cbar_title']
        else:
            cbar_title = None
        if 'palette_reverse' in kwargs and kwargs['palette_reverse']:
            palette = palette[::-1]
        lons = cube.coord('longitude').points
        lats = cube.coord('latitude').points

        color_mapper = LinearColorMapper(palette=palette, low=kwargs['low'], high=kwargs['high'])


        plot.image(image=[cube.data], x=min(lons), y=min(lats), dw=max(lons) - min(lons),
                   dh=max(lats) - min(lats), color_mapper=color_mapper, alpha=0.7)


        plot.x_range = Range1d(start=min(lons), end=max(lons))
        plot.y_range = Range1d(start=min(lats), end=max(lats))

        color_bar = ColorBar(color_mapper=color_mapper, label_standoff=12,
                             border_line_color=None, location=(0, 0),
                             orientation='horizontal', title=cbar_title)
        plot.add_layout(color_bar, 'below')
        return plot


    def plot_vectors(self, plot, u, v, **kwargs):
        xSkip = kwargs.get("xSkip", 5)
        ySkip = kwargs.get("ySkip", 5)
        maxSpeed = kwargs.get("maxSpeed", 10.)
        arrowHeadAngle = kwargs.get("arrowHeadAngle", 50.)
        arrowHeadScale = kwargs.get("arrowHeadScale", 0.1)
        arrowType = kwargs.get("arrowType", "barbed")
        palette = kwargs.get('palette', TolRainbow12)
        palette_reverse = kwargs.get('palette_reverse', False)

        vec = vector(u, v, xSkip=xSkip, ySkip=ySkip,
                     maxSpeed=maxSpeed, arrowType=arrowType, arrowHeadScale=arrowHeadScale,
                     palette=palette, palette_reverse=palette_reverse)

        arrow_source = ColumnDataSource(dict(xs=vec.xs, ys=vec.ys, colors=vec.colors))

        plot.patches(xs="xs", ys="ys", fill_color="colors", line_color="colors", alpha=0.5, source=arrow_source)
        return plot

    def extract_and_collapse(self, cube, box):
        extracted_cube = cube.intersection(latitude=(box[2], box[3]), longitude=(box[0], box[1]))
        return extracted_cube.collapsed(('latitude', 'longitude'), iris.analysis.MEAN)


    def cold_surge_probabilities(self, u850_cube, v850_cube, speed_cube):
        # Cold surge identification
        chang_box = [107, 115, 5, 10]  # CP Chang's 2nd domain

        # Hattori box for cross equatorial surges
        hattori_box = [105, 115, -5, 5]

        Chang_threshold = 9.0  # 10 # wind speed m/s
        Hattori_threshold = -2.0  # m/s meridional wind

        u850_ba = self.extract_and_collapse(u850_cube, chang_box)
        v850_ba = self.extract_and_collapse(v850_cube, chang_box)
        speed_ba = self.extract_and_collapse(speed_cube, chang_box)
        v850_hattori = self.extract_and_collapse(v850_cube, hattori_box)

        # extract the forecast periods and members from the data
        # to create a metrics array
        forecast_periods = u850_cube.coord('forecast_period').points
        members = u850_cube.coord('realization').points

        # Check for cross-equatorial surges
        mask1 = (u850_ba.data < 0.) & (v850_ba.data < 0.) & (speed_ba.data >= Chang_threshold)
        mask2 = mask1 & (v850_hattori.data <= Hattori_threshold)
        cs_prob = [round(p, 1) for p in 100. * np.sum(mask1, axis=0) / float(len(mask1))]
        ces_prob = [round(p, 1) for p in 100. * np.sum(mask2, axis=0) / float(len(mask2))]
        print(cs_prob, ces_prob)

        # return the probabilities as fraction
        return cs_prob, ces_prob


    def get_file_name(self, date, varname):
        concated_dir = os.path.join(self.config_values[f'{self.model}_cs_processed_dir'], varname)
        file_name = os.path.join(concated_dir,
                                        f'{varname}_ColdSurge_24h_allMember_{date.strftime("%Y%m%d")}.nc')
        return file_name


    def bokeh_plot_forecast_ensemble_mean(self, date, plot_width=500):

        precip_file_name = self.get_file_name(date, 'precip')
        u850_file_name = self.get_file_name(date, 'u850')
        v850_file_name = self.get_file_name(date, 'v850')

        date_label = date.strftime("%Y%m%d")

        precip_cube = iris.load_cube(precip_file_name)
        precip_cube = precip_cube.intersection(longitude=(90, 135))
        u850_cube = iris.load_cube(u850_file_name)
        u850_cube = u850_cube.intersection(longitude=(90, 135))
        v850_cube = iris.load_cube(v850_file_name)
        v850_cube = v850_cube.intersection(longitude=(90, 135))

        # Compute speed
        speed_cube = (u850_cube ** 2 + v850_cube ** 2) ** 0.5

        # Compute cold surge probabilities
        cs_prob, ces_prob = self.cold_surge_probabilities(u850_cube, v850_cube, speed_cube)
        print(cs_prob, ces_prob)

        # Ensemble mean
        precip_ens_mean = precip_cube.collapsed('realization', iris.analysis.MEAN)
        u850_ens_mean = u850_cube.collapsed('realization', iris.analysis.MEAN)
        v850_ens_mean = v850_cube.collapsed('realization', iris.analysis.MEAN)
        speed_ens_mean = speed_cube.collapsed('realization', iris.analysis.MEAN)

        cube = precip_ens_mean[0]
        lons = cube.coord('longitude').points
        lats = cube.coord('latitude').points

        ntimes = len(precip_cube.coord('forecast_period').points)

        # Plot setup
        width = plot_width
        aspect = (max(lons) - min(lons)) / (max(lats) - min(lats))
        height = int(width / (0.75 * aspect))
        print(width, height)

        for t in np.arange(ntimes):
            valid_date = date + datetime.timedelta(days=int(t))
            valid_date_label = f'{valid_date.strftime("%Y%m%d")}'

            title = f'Ensemble mean P, UV850 [CS:{cs_prob[t]}%, CES:{ces_prob[t]}%]'
            subtitle = f'Forecast start: {date_label}, Lead: T+{(t)}d Valid for 24H up to {valid_date_label}'

            hover = HoverTool(
                tooltips=[
                    ("Latitude", "@latitude"),
                    ("Longitude", "@longitude"),
                    ("Grid Value", "@data")
                ],
                mode='mouse'  # Set the mode to display the hover tooltip
            )

            plot = figure(height=height, width=width, title=None,
                          tools=["pan, reset, save, box_zoom, wheel_zoom, hover"],
                          )



            cmap = GnBu9#.copy()
            cbar_title = 'Precipitation (mm/day)'
            options = {'palette': cmap, 'palette_reverse': True, 'low': 5, 'high': 30, 'cbar_title': cbar_title}
            plot = self.plot_image_map(plot, precip_ens_mean[t], **options)

            cmap = RdPu9#.copy()
            vector_options = {'palette': cmap, 'palette_reverse': True, 'maxSpeed': 5, 'arrowHeadScale': 0.2,
                              'arrowType': 'barbed', 'xSkip': self.xSkip, 'ySkip': self.ySkip }
            plot = self.plot_vectors(plot, u850_ens_mean[t], v850_ens_mean[t], **vector_options)


            with open(self.map_outline_json_file, 'r', encoding="utf-8") as f:
                countries = GeoJSONDataSource(geojson=f.read())

            plot.patches("xs", "ys", color=None, line_color="black", fill_color=None, fill_alpha=0.2,
                         source=countries,
                         alpha=0.5)

            plot.add_layout(Title(text=subtitle, text_font_style="italic"), 'above')
            plot.add_layout(Title(text=title, text_font_size="12pt"), 'above')

            #show(plot)
            html_file_dir = os.path.join(self.config_values[f'{self.model}_cs_plot_ens'], date_label)
            if not os.path.exists(html_file_dir):
                os.makedirs(html_file_dir)

            html_file = os.path.join(html_file_dir, f'Cold_surge_EnsMean_{date_label}_T{(t * 24)}h.html')

            output_file(html_file)
            save(plot)
            print('Plotted %s' % html_file)

        json_file = os.path.join(self.config_values[f'{self.model}_cs_plot_ens'], f'{self.model}_ensmean_plot_dates.json')
        self.write_dates_json(date, json_file)


    def bokeh_plot_forecast_probability_precip(self, date, precip_thresholds=[10, 20, 30],
                                  plot_width=500):
        '''
        Plots the probability maps of precip and winds for given thresholds
        :param forecast_date_time:
        :type forecast_date_time:
        :param precip_threshold:
        :type precip_threshold:
        :param speed_threshold:
        :type speed_threshold:
        :return:
        :rtype:
        '''
        precip_file_name = self.get_file_name(date, 'precip')
        u850_file_name = self.get_file_name(date, 'u850')
        v850_file_name = self.get_file_name(date, 'v850')

        date_label = date.strftime("%Y%m%d")

        precip_cube = iris.load_cube(precip_file_name)
        precip_cube = precip_cube.intersection(longitude=(90, 135))
        u850_cube = iris.load_cube(u850_file_name)
        u850_cube = u850_cube.intersection(longitude=(90, 135))
        v850_cube = iris.load_cube(v850_file_name)
        v850_cube = v850_cube.intersection(longitude=(90, 135))

        # Compute speed
        speed_cube = (u850_cube ** 2 + v850_cube ** 2) ** 0.5

        # Compute cold surge probabilities
        cs_prob, ces_prob = self.cold_surge_probabilities(u850_cube, v850_cube, speed_cube)
        print(cs_prob, ces_prob)

        lons = precip_cube.coord('longitude').points
        lats = precip_cube.coord('latitude').points

        # Plot setup
        width = plot_width
        aspect = (max(lons) - min(lons)) / (max(lats) - min(lats))
        height = int(width / (0.75 * aspect))

        print(width, height)

        for precip_threshold in precip_thresholds:
            # Compute cold surge probabilities
            precip_prob = precip_cube.collapsed('realization', iris.analysis.PROPORTION,
                                                function=lambda values: values > precip_threshold)

            ntimes = len(precip_cube.coord('forecast_period').points)

            for t in np.arange(ntimes):
                valid_date = date + datetime.timedelta(days=int(t))
                valid_date_label = f'{valid_date.strftime("%Y%m%d")}'

                title = f'Ensemble probability of Precipitation [CS:{cs_prob[t]}%, CES:{ces_prob[t]}%]'
                subtitle = f'Forecast start: {date_label}, Lead: T+{(t)}d Valid for 24H up to {valid_date_label}'
                plot = figure(height=height, width=width, title=None,
                              tools=["pan, reset, save, box_zoom, wheel_zoom, hover"],
                              x_axis_label=None, y_axis_label=None)

                cmap = GnBu9
                cbar_title = 'Precipitation probability (p >= %s mm/day)' % precip_threshold
                options = {'palette': cmap, 'palette_reverse': True, 'low': 0.1, 'high': 1, 'cbar_title':cbar_title}
                plot = self.plot_image_map(plot, precip_prob[t], **options)

                with open(self.map_outline_json_file, 'r', encoding="utf-8") as f:
                    countries = GeoJSONDataSource(geojson=f.read())

                plot.patches("xs", "ys", color=None, line_color="black", fill_color=None, fill_alpha=0.2,
                             source=countries,
                             alpha=0.5)

                plot.add_layout(Title(text=subtitle, text_font_style="italic"), 'above')
                plot.add_layout(Title(text=title, text_font_size="12pt"), 'above')

                html_file_dir = os.path.join(self.config_values[f'{self.model}_cs_plot_ens'], date_label)
                if not os.path.exists(html_file_dir):
                    os.makedirs(html_file_dir)

                # show(plot)
                html_file = os.path.join(html_file_dir,
                                                      f'Cold_surge_ProbMaps_{date_label}_T{(t * 24)}h_Pr{precip_threshold}.html')

                output_file(html_file)
                save(plot)
                print('Plotted %s' % html_file)

            json_file = os.path.join(self.config_values[f'{self.model}_cs_plot_ens'], f'{self.model}_ProbMaps_plot_dates.json')
            self.write_dates_json(date, json_file)




if __name__ == '__main__':
    today = datetime.date.today()
    #yesterday = today - datetime.timedelta(days=1)
    yesterday = datetime.datetime(2021, 9, 30, 12)

    #bokeh_plot_forecast_ensemble_mean(yesterday)
    #bokeh_plot_forecast_probability_precip(yesterday, precip_thresholds=[10])
    #bokeh_plot_forecast_probability_speed(yesterday, speed_thresholds=[5])