import os
import datetime
import json
import numpy as np
import iris
from bokeh.plotting import figure, save, output_file
from bokeh.models import (
    ColumnDataSource, 
    LinearColorMapper, 
    ColorBar, 
    BasicTicker, 
    HoverTool, 
    Title,
    GeoJSONDataSource
)
from bokeh.palettes import GnBu9

class ConvLinesDisplay:
    """
    A class for displaying convergence line probability maps using Bokeh.

    This class generates interactive HTML visualizations of convergence line probabilities
    from ensemble forecast data, with map backgrounds and color-coded probability indicators.

    Attributes
    ----------
    config_values : dict
        Configuration dictionary containing directory paths and settings.
    model : str
        Model identifier (e.g., 'mogreps', 'glosea').
    parent_dir : str
        Parent directory path for the CONVLINES project.
    """
    
    def __init__(self, model, config_values):
        """
        Initialize the ConvLinesDisplay object.

        Parameters
        ----------
        model : str
            Model identifier used to access model-specific configuration values.
        config_values : dict
            Dictionary containing configuration values including directory paths.
        """
        self.config_values = config_values
        self.model = model

        # Navigate to the parent directory
        self.parent_dir = '/home/users/prince.xavier/MJO/SALMON/CONVLINES'

    def write_dates_json(self, date, json_file):
        """
        Write or update a JSON file with forecast dates.

        Maintains a sorted list of dates in JSON format, adding new dates if they don't exist.

        Parameters
        ----------
        date : datetime.date
            The forecast date to add to the JSON file.
        json_file : str
            Path to the JSON file containing the list of dates.

        Side Effects
        ------------
        Creates or updates the specified JSON file with the new date.
        Prints a confirmation message upon successful update.
        """

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

    def lonlat_to_webmercator(self, lon, lat):
        """
        Convert longitude/latitude coordinates to Web Mercator projection (meters).

        Web Mercator is the projection used by most online mapping services including
        OpenStreetMap and Bokeh tile providers.

        Parameters
        ----------
        lon : float or array-like
            Longitude in decimal degrees.
        lat : float or array-like
            Latitude in decimal degrees.

        Returns
        -------
        tuple of (float or array-like, float or array-like)
            x, y coordinates in Web Mercator projection (meters).
        """
        
        k = 6378137.0  # Earth radius in meters
        x = lon * (k * np.pi / 180.0)
        y = np.log(np.tan((90 + lat) * np.pi / 360.0)) * k
        return x, y

    def plot_convergence_lines(self, convergence_lines_cube, title="Convergence Lines Map", 
                               min_val=0, max_val=1.0):
        """
        Create a Bokeh figure displaying convergence line probability data.

        Generates an interactive map with convergence line probabilities overlaid on
        a CartoDB Positron basemap. Data is rendered with transparency based on
        probability values.

        Parameters
        ----------
        convergence_lines_cube : iris.cube.Cube
            Iris cube containing convergence line probability data with latitude and
            longitude coordinates.
        title : str, optional
            Plot title. Default is "Convergence Lines Map".
        min_val : float, optional
            Minimum value for color mapping. Default is 0.
        max_val : float, optional
            Maximum value for color mapping. Default is 1.0.

        Returns
        -------
        bokeh.plotting.figure.Figure
            Bokeh figure object ready for output or display.

        Notes
        -----
        - Uses GnBu9 color palette (reversed) for probability visualization.
        - Alpha channel is set based on data values to show transparency.
        - Map uses Web Mercator projection for compatibility with tile providers.
        """
        
        # Prepare the data for plotting
        lon = convergence_lines_cube.coord('longitude').points
        lat = convergence_lines_cube.coord('latitude').points
        data = np.array(convergence_lines_cube.data)

        x_min, y_min = self.lonlat_to_webmercator(lon.min(), lat.min())
        x_max, y_max = self.lonlat_to_webmercator(lon.max(), lat.max())

        print((x_min-x_max)/10000, y_min-y_max)
        # Create a Bokeh figure
        p = figure(x_range=(x_min, x_max), y_range=(y_min, y_max), 
                  width=int(np.abs(x_max-x_min)/7000), 
                  height=int(np.abs(y_max-y_min)/7000),
                  x_axis_type="mercator", y_axis_type="mercator")

        # Convert data to RGBA with alpha based on data values
        # Normalize data to 0-1 range for color mapping
        data_norm = np.clip((data - min_val) / (max_val - min_val), 0, 1)
        
        # Map to colors from GnBu9 palette
        palette = np.array([tuple(int(c[i:i+2], 16) for i in (1, 3, 5)) for c in GnBu9])
        palette = palette[::-1]
        color_indices = (data_norm * (len(palette) - 1)).astype(int)
        
        # Create RGBA image with alpha channel based on data
        rgba = np.zeros((*data.shape, 4), dtype=np.uint8)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if data[i, j] > 0:
                    rgba[i, j, :3] = palette[color_indices[i, j]]
                    rgba[i, j, 3] = int(data[i, j] * 255)  # Alpha based on data value
        
        # Convert to uint32 format for bokeh
        img = np.zeros(data.shape, dtype=np.uint32)
        img.view(dtype=np.uint8).reshape((*data.shape, 4))[:] = rgba

        # Plot the convergence lines
        p.image_rgba(image=[img], x=x_min, y=y_min, dw=x_max - x_min, dh=y_max - y_min)

        # Create a color mapper for the colorbar (convert RGB tuples to hex strings)
        palette_hex = ['#%02x%02x%02x' % tuple(c) for c in palette]
        color_mapper = LinearColorMapper(palette=palette_hex, low=min_val, high=max_val)
        
        # Add colorbar
        color_bar = ColorBar(color_mapper=color_mapper, 
                            ticker=BasicTicker(),
                            label_standoff=12,
                            border_line_color=None,
                            location=(0, 0))
        p.add_layout(color_bar, 'right')

        p.add_tile("CartoDB Positron", retina=True)
        p.title.text = title
        
        return p

    def bokeh_plot_convlines_prob_map(self, date):
        """
        Generate and save convergence line probability maps for all forecast lead times.

        Creates interactive HTML visualizations for each forecast time step, showing
        the probability of convergence lines across the domain. Updates a JSON file
        tracking all processed dates.

        Parameters
        ----------
        date : datetime.date
            The forecast initialization date.

        Side Effects
        ------------
        - Creates HTML files in the configured output directory.
        - Updates the dates JSON file with the processed date.
        - Prints status messages for each generated plot.

        Notes
        -----
        - Expects input NetCDF file with convergence line probabilities.
        - Generates one plot per forecast time step (typically 24-hour intervals).
        - Uses probability thresholds of 0.3 to 0.9 for color mapping.
        
        Returns
        -------
        None
        """

        date_label = date.strftime('%Y%m%d')
        output_dir = os.path.join(self.config_values[f'{self.model}_convlines_processed_dir'], 'convergence_lines')
        os.makedirs(output_dir, exist_ok=True)
        conv_file = os.path.join(output_dir, f'convergence_lines_24h_mean_{date_label}.nc')

        if os.path.exists(conv_file):
            conv_cube = iris.load_cube(conv_file)
            ntimes = len(conv_cube.coord('forecast_period').points)

            for t in np.arange(ntimes):
                valid_date = date + datetime.timedelta(days=int(t))
                valid_date_label = f'{valid_date.strftime("%Y%m%d")}'

                plot = self.plot_convergence_lines(conv_cube[t], min_val=0.3, max_val=0.9, 
                                                   title=f'Probability of convergence Lines on {valid_date_label} at t+{t*24}h')

                html_file_dir = os.path.join(self.config_values[f'{self.model}_convlines_plot_ens'], date_label)

                if not os.path.exists(html_file_dir):
                    os.makedirs(html_file_dir)

                html_file = os.path.join(html_file_dir,
                                        f'ConvLines_ProbMaps_{date_label}_T{t * 24}h.html')

                output_file(html_file)
                save(plot)
                print('Plotted %s' % html_file)

            json_file = os.path.join(self.config_values[f'{self.model}_convlines_plot_ens'], f'{self.model}_convlines_plot_dates.json')
            self.write_dates_json(date, json_file)




if __name__ == '__main__':
    today = datetime.date.today()
    #yesterday = today - datetime.timedelta(days=1)
    yesterday = datetime.datetime(2021, 9, 30, 12)

    #bokeh_plot_forecast_ensemble_mean(yesterday)
    #bokeh_plot_forecast_probability_precip(yesterday, precip_thresholds=[10])
    #bokeh_plot_forecast_probability_speed(yesterday, speed_thresholds=[5])