import os
import logging
import datetime
import uuid
import concurrent.futures
import subprocess
from typing import Dict, Any, List
import numpy as np
import iris
import json
from bokeh.plotting import figure, save, output_file
from bokeh.models import ColumnDataSource, HoverTool, Title, Range1d, LinearColorMapper, ColorBar, GeoJSONDataSource
from bokeh.palettes import GnBu9, RdPu9, TolRainbow12
from salmon.core.task import Task
from salmon.utils.moose import MooseClient
from salmon.utils.config import load_global_config
from salmon.utils.cube import read_winds_correctly, read_precip_correctly
from salmon.utils.bokeh_utils import Vector
import sys
import warnings
# Set the global warning filter to ignore all warnings
warnings.simplefilter("ignore")

logger = logging.getLogger(__name__)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_UTILS_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "utils"))

_DEFAULT_MAP_JSON = os.path.normpath(
    os.path.join(_UTILS_DIR, "map_data", "custom.geo.json")
)
_DEFAULT_QUERY_DIR = os.path.normpath(
    os.path.join(_UTILS_DIR, "query_files")
)
HR_LIST = (12, 18)
FC_TIMES = tuple(np.arange(0, 174, 24))
DISPLAY_LAT_BOUNDS = (-10, 25)
DISPLAY_LON_BOUNDS = (90, 135)

VAR_SPECS = {
    "precip": {"iris_var": "precipitation_amount"},
    "u850": {"iris_var": "x_wind", "pressure_level": 850},
    "v850": {"iris_var": "y_wind", "pressure_level": 850},
}

class RetrieveGPMColdSurgeData(Task):
    """
    Retrieve GPM IMERG data needed for Cold Surge processing.

    Notes
    -----
    - Expects GPM IMERG data in the configured raw directory.
    - Forecast steps are 24-hourly from 0 to 168 hours.
    """

    def run(self):
        """Task entrypoint."""
        date = self.context.date
        self._init_config_values()
        success = self.process_gpm_data(date=date)

        if success:
            logger.info("GPM Cold Surge data retrieval complete.")
        else:
            logger.warning("GPM Cold Surge data retrieval completed with errors.")

    def _init_config_values(self):
        """Load and cache retrieval paths/config used by helper methods."""
        if hasattr(self, "config_values"):
            return
        
        gpm = load_global_config().get("gpm", {})
        self.config_values = {
            "gpm_raw_dir": gpm.get("raw", "/tmp/salmon_raw/gpm"),
            "gpm_processed_dir": gpm.get("processed", "/tmp/salmon_processed/gpm"),
        }
        print(f"Config values for GPM retrieval: {self.config_values}")
        #os.makedirs(self.config_values["gpm_raw_dir"], exist_ok=True)

    def process_gpm_data(self, date):
        """
        Collect and link available GPM IMERG data files for the cold surge forecast period.
        
        Looks for GPM daily files from 5 days before date to FC_TIMES[-1] days (7 days) after date,
        and creates symlinks in the SALMON processing directory structure as member 000.

        Returns
        -------
        bool
            True if GPM file(s) are found and linked successfully, else False.
        """
        raw_root = self.config_values["gpm_raw_dir"]
        processed_base = self.config_values["gpm_processed_dir"]
        
        if not os.path.isdir(raw_root):
            logger.error("GPM source directory not found: %s", raw_root)
            return False
        
        # Generate date range: 5 days before to FC_TIMES[-1] days (168 hours = 7 days) after
        lookback_days = 5
        forecast_days = int(FC_TIMES[-1] / 24)  # 168 hours = 7 days
        
        start_date = date - datetime.timedelta(days=lookback_days)
        end_date = date + datetime.timedelta(days=forecast_days)
        
        logger.info("Searching for GPM IMERG data from %s to %s", start_date.date(), end_date.date())
        
        # Collect available GPM files in the date range
        gpm_files_available = []
        current_date = start_date
        
        while current_date <= end_date:
            gpm_filename = f"gpm_imerg_NRTlate_V07C_{current_date:%Y%m%d}_daily.nc"
            gpm_filepath = os.path.join(raw_root, gpm_filename)
            
            if os.path.exists(gpm_filepath) and os.path.getsize(gpm_filepath) > 0:
                gpm_files_available.append((current_date, gpm_filepath))
                logger.debug("Found GPM file: %s", gpm_filepath)
            
            current_date += datetime.timedelta(days=1)
        
        if not gpm_files_available:
            logger.warning("No GPM IMERG files found in range %s to %s", start_date.date(), end_date.date())
            return False
        
        logger.info("Found %d GPM IMERG files", len(gpm_files_available))
        
        # Create symlinks for available GPM files into SALMON processing structure
        # Each file is linked to a date directory under processed_base/gpm/coldsurge/
        success = True
        for idx, (file_date, gpm_file) in enumerate(gpm_files_available):
            # Determine the target directory for this date
            target_dir = os.path.join(processed_base, "coldsurge", file_date.strftime("%Y%m%d"))
            os.makedirs(target_dir, exist_ok=True)
            
            # Create a symlink named 'gpm_imerg_daily.nc' in the target directory
            symlink_path = os.path.join(target_dir, "gpm_imerg_daily.nc")
            
            try:
                if os.path.exists(symlink_path):
                    os.remove(symlink_path)  # Remove existing symlink/file
                os.symlink(gpm_file, symlink_path)
                logger.info("Linked GPM file for %s to %s", file_date.date(), symlink_path)
            except Exception as e:
                logger.error("Failed to link GPM file for %s: %s", file_date.date(), e)
                success = False
        return success

class DisplayGPMColdSurgeMaps(Task):
    """
    Create Bokeh map products from processed GPM Cold Surge NetCDF files.

    Products
    --------
    - Ensemble-mean precip + 850hPa wind vectors (HTML)
    - Ensemble probability precip maps for configured thresholds (HTML)
    """
    
    def run(self):
        """Task entrypoint."""
        date = self.context.date
        self._init_config_values()
        success = self.bokeh_plot_gpm_daily_mean(date=date)

        if success:
            logger.info("GPM Cold Surge data retrieval complete.")
        else:
            logger.warning("GPM Cold Surge data retrieval completed with errors.")

    def _init_config_values(self):
        """Load and cache retrieval paths/config used by helper methods."""
        if hasattr(self, "config_values"):
            return
        
        gpm = load_global_config().get("gpm", {})
        gpm_plot_root = gpm.get("plots", gpm.get("processed", "/tmp/salmon_processed/gpm"))
        gpm_plot_dir = os.path.join(gpm_plot_root, "coldsurge", "plot_ens")
        self.config_values = {
            "gpm_raw_dir": gpm.get("raw", "/tmp/salmon_raw/gpm"),
            "gpm_processed_dir": gpm.get("processed", "/tmp/salmon_processed/gpm"),
            "gpm_cs_plot_ens": gpm_plot_dir,
            "map_outline_json_file": self.config.get(
                "map_outline_json_file",
                os.path.normpath(_DEFAULT_MAP_JSON),
            ),
        }
        os.makedirs(gpm_plot_dir, exist_ok=True)
        print(f"Config values for GPM retrieval: {self.config_values}")
        #os.makedirs(self.config_values["gpm_raw_dir"], exist_ok=True)

    def write_dates_json(self, date, json_file):
        """Append YYYYMMDD to a JSON date list (unique + sorted)."""
        new_date = date.strftime("%Y%m%d")
        if not os.path.exists(json_file):
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump([new_date], f, indent=2)
            return

        with open(json_file, "r", encoding="utf-8") as f:
            existing_dates = json.load(f)

        if new_date not in existing_dates:
            existing_dates.append(new_date)
            existing_dates.sort()

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(existing_dates, f, indent=2)

    def _add_map_outline(self, plot):
        """Overlay coast/country outline from GeoJSON, if present."""
        outline = self.config_values["map_outline_json_file"]
        if not os.path.exists(outline):
            logger.warning("Map outline not found: %s", outline)
            return plot

        with open(outline, "r", encoding="utf-8") as f:
            countries = GeoJSONDataSource(geojson=f.read())
        plot.patches(
            "xs",
            "ys",
            color=None,
            line_color="black",
            fill_color=None,
            fill_alpha=0.2,
            source=countries,
            alpha=0.5,
        )
        return plot

    def plot_image_map(self, plot, cube, **kwargs):
        """Draw gridded data as image and attach a colorbar."""
        palette = kwargs.get("palette", GnBu9)
        if kwargs.get("palette_reverse", False):
            palette = palette[::-1]

        lons = cube.coord("longitude").points
        lats = cube.coord("latitude").points

        color_mapper = LinearColorMapper(
            palette=palette, low=kwargs.get("low", 0.0), high=kwargs.get("high", 1.0)
        )
        plot.image(
            image=[cube.data],
            x=min(lons),
            y=min(lats),
            dw=max(lons) - min(lons),
            dh=max(lats) - min(lats),
            color_mapper=color_mapper,
            alpha=0.7,
        )
        plot.x_range = Range1d(start=min(lons), end=max(lons))
        plot.y_range = Range1d(start=min(lats), end=max(lats))

        lon_grid, lat_grid = np.meshgrid(lons, lats)
        hover_source = ColumnDataSource(
            data={
                "lon": lon_grid.ravel(),
                "lat": lat_grid.ravel(),
                "value": np.asarray(cube.data).ravel(),
            }
        )
        hover_renderer = plot.circle(
            x="lon",
            y="lat",
            size=8,
            alpha=0.0,
            line_alpha=0.0,
            fill_alpha=0.0,
            source=hover_source,
        )

        hover = HoverTool(
            renderers=[hover_renderer],
            tooltips=[
                ("Lon", "@lon{0.0}"),
                ("Lat", "@lat{0.0}"),
                ("Value", "@value{0.00}"),
            ],
        )
        plot.add_tools(hover)
        plot.toolbar.active_inspect = None

        color_bar = ColorBar(
            color_mapper=color_mapper,
            label_standoff=12,
            border_line_color=None,
            location=(0, 0),
            orientation="vertical",
            title=kwargs.get("cbar_title"),
        )
        plot.add_layout(color_bar, "right")
        return plot

    def bokeh_plot_gpm_daily_mean(self, date, **kwargs):
        # Read the gpm daily mean cube and plot it as an image with colorbar and map outline.
        
        processed_base = self.config_values["gpm_processed_dir"]
        
        # Check for files for the 5 days before and 7 days after the date
        lookback_days = 5
        forecast_days = 7  # 168 hours / 24
        
        start_date = date - datetime.timedelta(days=lookback_days)
        end_date = date + datetime.timedelta(days=forecast_days)
        
        # Build list of available GPM files in the date range
        available_gpm_files = []
        current_date = start_date
        
        while current_date <= end_date:
            target_dir = os.path.join(processed_base, "coldsurge", current_date.strftime("%Y%m%d"))
            gpm_data_file = os.path.join(target_dir, "gpm_imerg_daily.nc")
            
            if os.path.exists(gpm_data_file) and os.path.getsize(gpm_data_file) > 0:
                available_gpm_files.append((current_date, gpm_data_file))
                logger.debug("Found GPM file for %s: %s", current_date.date(), gpm_data_file)
            
            current_date += datetime.timedelta(days=1)
        
        if not available_gpm_files:
            logger.error("No GPM daily mean data files found in range %s to %s", start_date.date(), end_date.date())
            return False
        
        logger.info("Found %d GPM data files in range %s to %s", len(available_gpm_files), start_date.date(), end_date.date())
        
        # Plot each available GPM file if the plot doesn't already exist
        plot_width = kwargs.get("plot_width", 700)
        overall_success = True
        
        for file_date, gpm_data_file in available_gpm_files:
            date_label = file_date.strftime("%Y%m%d")
            html_dir = os.path.join(self.config_values["gpm_cs_plot_ens"], date_label)
            os.makedirs(html_dir, exist_ok=True)
            
            out_html = os.path.join(html_dir, f"Cold_surge_GPM_DailyMean_{date_label}.html")
            
            # Skip if plot already exists
            if os.path.exists(out_html):
                logger.debug("Plot already exists for %s. Skipping.", date_label)
                continue
            
            try:
                # Load the GPM cube
                cube = iris.load_cube(gpm_data_file, "precipitation_flux")
                # Subset to display bounds
                cube = cube.intersection(latitude=DISPLAY_LAT_BOUNDS, longitude=DISPLAY_LON_BOUNDS)
                
                logger.debug("Loaded GPM daily mean cube for %s: shape=%s", date_label, cube.shape)
                
                # Generate daily mean precip maps
                lons = cube.coord("longitude").points
                lats = cube.coord("latitude").points
                height = int(plot_width / (((max(lons) - min(lons)) / (max(lats) - min(lats))) * 1.0))
                
                plot = figure(height=height, width=plot_width, title=None, tools="pan,reset,save,box_zoom,wheel_zoom")
                plot = self.plot_image_map(
                    plot,
                    cube,
                    palette=GnBu9,
                    palette_reverse=True,
                    low=5,
                    high=30,
                    cbar_title="Precipitation (mm/day)",
                )
                plot = self._add_map_outline(plot)
                
                title = "GPM Daily Mean Precipitation"
                subtitle = f"Valid date: {date_label}"
                plot.add_layout(Title(text=subtitle, text_font_style="italic"), "above")
                plot.add_layout(Title(text=title, text_font_size="12pt"), "above")
                
                output_file(out_html)
                save(plot)
                logger.info("Plotted %s", out_html)
                
                # Update the dates JSON file
                self.write_dates_json(
                    file_date,
                    os.path.join(self.config_values["gpm_cs_plot_ens"], "gpm_dailymean_plot_dates.json"),
                )
            except Exception as e:
                logger.error("Error plotting GPM file for %s: %s", date_label, e)
                overall_success = False
                continue
        
        
        
        return overall_success

        
    
