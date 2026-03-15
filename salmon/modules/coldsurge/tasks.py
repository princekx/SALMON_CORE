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
from bokeh.models import ColumnDataSource, Title, Range1d, LinearColorMapper, ColorBar, GeoJSONDataSource
from bokeh.palettes import GnBu9, RdPu9, TolRainbow12
from salmon.core.task import Task
from salmon.utils.moose import MooseClient
from salmon.utils.config import load_global_config
from salmon.utils.cube import read_winds_correctly, read_precip_correctly, subset_seasia
from salmon.utils.bokeh_utils import Vector
import sys
import warnings
# Set the global warning filter to ignore all warnings
warnings.simplefilter("ignore")

logger = logging.getLogger(__name__)

def _describe_task(task_obj):
    """Return normalized task metadata for logging/debug."""
    cls = task_obj.__class__
    return {
        "task": (
            getattr(task_obj.context, "task_name", None)
            or task_obj.config.get("name")
            or cls.__name__
        ),
        "module": cls.__module__,
        "class": cls.__name__,
        "config": dict(task_obj.config),
    }

class RetrieveColdSurgeData(Task):
    """
    Retrieve MOGREPS files needed for Cold Surge processing.

    Notes
    -----
    - Members are split by cycle:
      * 12Z -> 00..17
      * 18Z -> 18..34 plus '00' (mapped to directory member 035)
    - Forecast steps are 24-hourly from 0 to 168 hours.
    """

    HR_LIST = (12, 18)
    FC_TIMES = tuple(np.arange(0, 174, 24))

    def run(self):
        """Task entrypoint."""
        meta = _describe_task(self)
        logger.info(
            "Task: %s, Module: %s, Class: %s, Config: %s",
            meta["task"], meta["module"], meta["class"], meta["config"]
        )

        date = self.context.date
        print(meta["config"].get("query"))  # Debug print to check context config values
        sys.exit(0)

        parallel = self.config.get("parallel", True)
        success = self.retrieve_mogreps_data(date=date, parallel=parallel)

        if success:
            logger.info("Cold Surge data retrieval complete.")
        else:
            logger.warning("Cold Surge data retrieval completed with errors.")

    def _init_config_values(self):
        """Load and cache retrieval paths/config used by helper methods."""
        if hasattr(self, "config_values"):
            return

        
        mogreps = load_global_config().get("mogreps", {})
        self.config_values = {
            "mogreps_moose_dir": mogreps.get("moose", "moose:/opfc/atm/mogreps-g/prods/"),
            "mogreps_raw_dir": mogreps.get("raw", "/tmp/salmon_raw/mogreps"),
            "mogreps_combined_queryfile": mogreps.get("query"),
            "mogreps_dummy_queryfiles_dir": mogreps.get("temp", "/tmp/salmon_temp"),
        }
        print(f"Config values for retrieval: {self.config_values}")
        sys.exit(0)
        os.makedirs(self.config_values["mogreps_dummy_queryfiles_dir"], exist_ok=True)

    def _get_all_members(self, hr):
        """Return 2-digit member IDs for a given model cycle hour."""
        if hr == 12:
            return [f"{mem:02d}" for mem in range(18)]
        if hr == 18:
            return [f"{mem:02d}" for mem in range(18, 35)] + ["00"]
        return []

    def _iter_tasks(self, date):
        """Yield (date, hr, fc, member) combinations to retrieve."""
        for hr in self.HR_LIST:
            for fc in self.FC_TIMES:
                for mem in self._get_all_members(hr):
                    yield (date, hr, fc, mem)

    def _resolve_moose_dir(self, date):
        """
        Resolve source MOOSE directory.

        Historical OS47 fallback logic is intentionally left disabled.
        """
        return os.path.join(self.config_values["mogreps_moose_dir"], f"{date:%Y%m}.pp")

    def _map_member_to_dir(self, hr, digit2_mem):
        """Map 2-digit member ID to 3-digit on-disk directory name."""
        return "035" if (hr == 18 and digit2_mem == "00") else f"{int(digit2_mem):03d}"

    def _retrieve_fc_data(self, date, hr, fc, digit2_mem):
        """Retrieve one forecast file for one member/cycle."""
        self._init_config_values()

        moose_dir = self._resolve_moose_dir(date)
        digit3_mem = self._map_member_to_dir(hr, digit2_mem)
        fct = f"{fc:03d}"

        remote_data_dir = os.path.join(
            self.config_values["mogreps_raw_dir"], date.strftime("%Y%m%d"), digit3_mem
        )
        os.makedirs(remote_data_dir, exist_ok=True)

        filemoose = f"prods_op_mogreps-g_{date:%Y%m%d}_{hr}_{digit2_mem}_{fct}.pp"
        outfile = f"englaa_pd{fct}.pp"
        outfile_path = os.path.join(remote_data_dir, outfile)

        # Remove corrupt empty file before retry
        if os.path.exists(outfile_path) and os.path.getsize(outfile_path) == 0:
            logger.warning("Deleting empty file: %s", outfile_path)
            os.remove(outfile_path)

        if os.path.exists(outfile_path):
            logger.debug("%s exists. Skip.", outfile_path)
            return True

        local_query_file = os.path.join(
            self.config_values["mogreps_dummy_queryfiles_dir"], f"localquery_{uuid.uuid1()}"
        )

        try:
            self.create_query_file(local_query_file, filemoose, fct)

            command = [
                "/opt/moose-client-wrapper/bin/moo",
                "select",
                "--fill-gaps",
                local_query_file,
                moose_dir,
                outfile_path,
            ]
            logger.info("Executing command: %s", " ".join(command))
            subprocess.run(command, check=True)
            return True

        except subprocess.CalledProcessError as exc:
            logger.error("Data retrieval failed for %s: %s", outfile_path, exc)
            return False
        except Exception as exc:
            logger.error("Unexpected retrieval error for %s: %s", outfile_path, exc)
            return False
        finally:
            if os.path.exists(local_query_file):
                os.remove(local_query_file)

    # Backward-compatible method name used elsewhere in code.
    def retrieve_fc_data_parallel(self, date, hr, fc, digit2_mem):
        """Compatibility wrapper around the streamlined retrieval function."""
        return self._retrieve_fc_data(date, hr, fc, digit2_mem)

    # Backward-compatible signature used by old run() style.
    def _retrieve_for_member(self, date, hr, fc, digit2_mem, *_unused):
        """Compatibility wrapper for legacy tuple-based submit arguments."""
        return self._retrieve_fc_data(date, hr, fc, digit2_mem)

    def check_if_all_data_exist(self, date, hr, fc, digit2_mem):
        """Check if one retrieved file exists and is non-empty."""
        self._init_config_values()
        digit3_mem = self._map_member_to_dir(hr, digit2_mem)
        outfile_path = os.path.join(
            self.config_values["mogreps_raw_dir"],
            date.strftime("%Y%m%d"),
            digit3_mem,
            f"englaa_pd{fc:03d}.pp",
        )
        return os.path.exists(outfile_path) and os.path.getsize(outfile_path) > 0

    def create_query_file(self, local_query_file1, filemoose, fct):
        """Create a per-request query file from the configured template."""
        self._init_config_values()
        query_file = self.config_values["mogreps_combined_queryfile"]
        replacements = {"fctime": fct, "filemoose": filemoose}

        with open(query_file) as query_infile, open(local_query_file1, "w") as query_outfile:
            for line in query_infile:
                for src, target in replacements.items():
                    line = line.replace(src, target)
                query_outfile.write(line)

    def retrieve_mogreps_data(self, date, parallel=True):
        """
        Retrieve all required MOGREPS files for one date.

        Returns
        -------
        bool
            True if all retrieval jobs succeeded, else False.
        """
        self._init_config_values()
        logger.info("Retrieving MOGREPS data for Cold Surge on %s", date)

        tasks = list(self._iter_tasks(date))

        if not parallel:
            return all(self._retrieve_fc_data(*task) for task in tasks)

        max_workers = int(self.config.get("max_workers", 10))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._retrieve_fc_data, *task) for task in tasks]
            return all(f.result() for f in concurrent.futures.as_completed(futures))

class ComputeColdSurgeIndices(Task):
    """
    Compute Cold Surge indices from retrieved MOGREPS PP files.

    Outputs
    -------
    NetCDF files (one per variable) containing all available members:
      - precip
      - u850
      - v850

    Notes
    -----
    - Expects input files in: <raw_dir>/<YYYYMMDD>/<member3>/englaa_pd<fc>.pp
    - Forecast steps are 24-hourly from 0 to 168 hours.
    - Member directories are expected as 000..035.
    """

    FC_TIMES = tuple(np.arange(0, 174, 24))
    VAR_SPECS = {
        "precip": {"iris_var": "precipitation_amount"},
        "u850": {"iris_var": "x_wind", "pressure_level": 850},
        "v850": {"iris_var": "y_wind", "pressure_level": 850},
    }

    def run(self):
        """Task entrypoint."""
        date = self.context.date
        self._init_config_values()
        self.process_forecast_data(date=date, members=self._member_directories())

    def _init_config_values(self):
        """Load and cache model paths used by this task."""
        if hasattr(self, "config_values"):
            return

        model = self.context.get_config("model", "mogreps")
        model_config = load_global_config().get(model, {})

        raw_dir = model_config.get("raw", "/tmp/salmon_raw/mogreps")
        processed_base = model_config.get("processed", f"/tmp/salmon_processed/{model}")
        cs_processed_dir = os.path.join(processed_base, "coldsurge")

        self.config_values = {
            "model": model,
            "mogreps_raw_dir": raw_dir,
            "mogreps_cs_processed_dir": cs_processed_dir,
            "obsgrid_file": self.config.get(
                "obsgrid_file", os.path.join(os.getcwd(), "data", "obsgrid_145x73.nc")
            ),
        }
        os.makedirs(cs_processed_dir, exist_ok=True)

    def _member_directories(self):
        """Return expected 3-digit member directories (000..035)."""
        return [f"{m:03d}" for m in range(36)]

    def _forecast_steps(self):
        """Return forecast-step strings ['000', '024', ..., '168']."""
        return [f"{fct:03d}" for fct in self.FC_TIMES]

    def load_base_cube(self):
        """
        Load observational target grid used for optional regridding.
        """
        base_cube = iris.load_cube(self.config_values["obsgrid_file"])
        for coord_name, units in (("latitude", "degrees_north"), ("longitude", "degrees_east")):
            base_cube.coord(coord_name).units = units
            base_cube.coord(coord_name).coord_system = None
        return base_cube

    def regrid2obs(self, cube):
        """
        Regrid cube to the observational grid using linear interpolation.
        """
        base_cube = self.load_base_cube()

        for coord_name, units in (("latitude", "degrees_north"), ("longitude", "degrees_east")):
            coord = cube.coord(coord_name)
            coord.units = units
            coord.coord_system = None
            if coord.bounds is None:
                coord.guess_bounds()

        return cube.regrid(base_cube, iris.analysis.Linear())

    def _normalise_for_merge(self, cube):
        """Remove known problematic scalar coords before merging."""
        for coord in ("forecast_reference_time", "realization", "time"):
            if cube.coords(coord):
                cube.remove_coord(coord)
        return cube

    def _ensure_bounds(self, cube):
        """Ensure forecast_period/time bounds exist for consistent metadata."""
        for coord_name in ("forecast_period", "time"):
            if cube.coords(coord_name):
                coord = cube.coord(coord_name)
                if coord.bounds is None:
                    p = coord.points[0]
                    coord.bounds = [[p - 1.0, p + 1.0]]
        return cube

    
    def process_forecast_data(self, date, members):
        """
        Build and save all-member Cold Surge files for precip, u850, and v850.
        """
        fc_times = self._forecast_steps()
        raw_root = self.config_values["mogreps_raw_dir"]
        out_root = self.config_values["mogreps_cs_processed_dir"]
        regrid_to_obs = bool(self.config.get("regrid_to_obs", False))

        logger.info("Computing Cold Surge indices for %s", date.strftime("%Y-%m-%d"))

        for varname, spec in self.VAR_SPECS.items():
            out_dir = os.path.join(out_root, varname)
            os.makedirs(out_dir, exist_ok=True)

            out_file = os.path.join(
                out_dir, f"{varname}_ColdSurge_24h_allMember_{date.strftime('%Y%m%d')}.nc"
            )
            if os.path.exists(out_file):
                logger.info("%s already exists. Skipping.", out_file)
                continue

            cubes = []
            for mem in members:
                files = [
                    os.path.join(raw_root, date.strftime("%Y%m%d"), mem, f"englaa_pd{fct}.pp")
                    for fct in fc_times
                ]
                existing_files = [f for f in files if os.path.exists(f)]
                if not existing_files:
                    logger.warning("No files found for member %s", mem)
                    continue

                if varname == "precip":
                    cube = read_precip_correctly(existing_files, spec["iris_var"])
                    if cube.shape[0] > 1:
                        cube.data[1:] -= cube.data[:-1]
                else:
                    cube = read_winds_correctly(
                        existing_files,
                        spec["iris_var"],
                        pressure_level=spec.get("pressure_level"),
                    )

                self._normalise_for_merge(cube)
                cube.add_aux_coord(
                    iris.coords.DimCoord(
                        [int(mem)], standard_name="realization", var_name="realization"
                    )
                )
                cubes.append(cube)

            if not cubes:
                logger.error("No cubes found to merge for %s on %s", varname, date.strftime("%Y%m%d"))
                continue

            merged = iris.cube.CubeList(cubes).merge_cube()
            if regrid_to_obs:
                merged = self.regrid2obs(merged)

            iris.save(merged, out_file, netcdf_format="NETCDF4_CLASSIC")
            logger.info("Saved merged members to %s", out_file)

    def _get_all_members(self, hr):
        """
        Backward-compatible helper retained for external callers.
        """
        if hr == 12:
            return [f"{mem:02d}" for mem in range(18)]
        if hr == 18:
            return [f"{mem:02d}" for mem in range(18, 35)] + ["00"]
        return []


class DisplayColdSurgeMaps(Task):
    """
    Create Bokeh map products from processed Cold Surge NetCDF files.

    Products
    --------
    - Ensemble-mean precip + 850hPa wind vectors (HTML)
    - Ensemble probability precip maps for configured thresholds (HTML)
    """

    def run(self):
        """Task entrypoint."""
        date = self.context.date
        self._init_config_values()

        do_ensmean = bool(self.config.get("plot_ensmean", True))
        do_probmaps = bool(self.config.get("plot_probmaps", True))
        precip_thresholds = self.config.get("precip_thresholds", [10, 20, 30])

        if do_ensmean:
            self.bokeh_plot_forecast_ensemble_mean(date)
        if do_probmaps:
            self.bokeh_plot_forecast_probability_precip(date, precip_thresholds=precip_thresholds)

    def _init_config_values(self):
        """Load and cache plotting/config paths."""
        if hasattr(self, "config_values"):
            return

        model = self.context.get_config("model", "mogreps")
        model_cfg = load_global_config().get(model, {})
        #print(model_cfg)
        #print(self.config)
        #print('**************************')
        processed_base = model_cfg.get("processed", f"/tmp/salmon_processed/{model}")
        cs_processed_dir = os.path.join(processed_base, "coldsurge")
        cs_plot_ens_dir = self.config.get(
            "plots",
            os.path.join(processed_base, "coldsurge", "plot_ens"),
        )
        #print(f'cs_processed_dir: {cs_processed_dir}')
        #print(f'processed_base: {processed_base}')
        #print(f'cs_plot_ens_dir: {self.config.get("plots")}')
        #print('**************************')
        
        self.config_values = {
            "model": model,
            f"{model}_cs_processed_dir": cs_processed_dir,
            f"{model}_cs_plot_ens": cs_plot_ens_dir,
            f"{model}_cs_plot_prob": os.path.join(cs_plot_ens_dir, "prob"),
            "map_outline_json_file": self.config.get(
                "map_outline_json_file",
                os.path.normpath(_DEFAULT_MAP_JSON),
            ),
        }
        os.makedirs(cs_plot_ens_dir, exist_ok=True)
        print(self.config_values)
        
        # vector thinning by model
        if model == "glosea":
            self.xSkip, self.ySkip = 2, 2
        else:
            self.xSkip, self.ySkip = 5, 5

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

    def plot_vectors(self, plot, u, v, **kwargs):
        """Draw vector arrows using the local Bokeh vector helper."""
        vec = Vector(
            u,
            v,
            xSkip=kwargs.get("xSkip", self.xSkip),
            ySkip=kwargs.get("ySkip", self.ySkip),
            maxSpeed=kwargs.get("maxSpeed", 10.0),
            arrowType=kwargs.get("arrowType", "barbed"),
            arrowHeadScale=kwargs.get("arrowHeadScale", 0.1),
            palette=kwargs.get("palette", TolRainbow12),
            palette_reverse=kwargs.get("palette_reverse", False),
        )
        source = ColumnDataSource(dict(xs=vec.xs, ys=vec.ys, colors=vec.colors))
        plot.patches(xs="xs", ys="ys", fill_color="colors", line_color="colors", alpha=0.5, source=source)
        return plot

    def extract_and_collapse(self, cube, box):
        """Area-mean over [lon0, lon1, lat0, lat1]."""
        sub = cube.intersection(latitude=(box[2], box[3]), longitude=(box[0], box[1]))
        return sub.collapsed(("latitude", "longitude"), iris.analysis.MEAN)

    def cold_surge_probabilities(self, u850_cube, v850_cube, speed_cube):
        """
        Compute CS and CES ensemble probabilities (%) by lead time.
        """
        chang_box = [107, 115, 5, 10]
        hattori_box = [105, 115, -5, 5]
        chang_threshold = 9.0
        hattori_threshold = -2.0

        u850_ba = self.extract_and_collapse(u850_cube, chang_box)
        v850_ba = self.extract_and_collapse(v850_cube, chang_box)
        speed_ba = self.extract_and_collapse(speed_cube, chang_box)
        v850_hattori = self.extract_and_collapse(v850_cube, hattori_box)

        mask_cs = (u850_ba.data < 0.0) & (v850_ba.data < 0.0) & (speed_ba.data >= chang_threshold)
        mask_ces = mask_cs & (v850_hattori.data <= hattori_threshold)

        cs_prob = [round(p, 1) for p in 100.0 * np.sum(mask_cs, axis=0) / float(len(mask_cs))]
        ces_prob = [round(p, 1) for p in 100.0 * np.sum(mask_ces, axis=0) / float(len(mask_ces))]
        return cs_prob, ces_prob

    def get_file_name(self, date, varname):
        """Return processed NetCDF file path for variable/date."""
        model = self.config_values["model"]
        root = self.config_values[f"{model}_cs_processed_dir"]
        return os.path.join(root, varname, f"{varname}_ColdSurge_24h_allMember_{date:%Y%m%d}.nc")

    def _load_required_cubes(self, date):
        """Load precip/u850/v850 cubes and compute speed cube."""
        precip_cube = subset_seasia(iris.load_cube(self.get_file_name(date, "precip")))
        u850_cube = subset_seasia(iris.load_cube(self.get_file_name(date, "u850")))
        v850_cube = subset_seasia(iris.load_cube(self.get_file_name(date, "v850")))
        speed_cube = (u850_cube ** 2 + v850_cube ** 2) ** 0.5
        return precip_cube, u850_cube, v850_cube, speed_cube

    def _add_map_outline(self, plot):
        """Overlay coast/country outline from GeoJSON, if present."""
        outline = self.config_values["map_outline_json_file"]
        if not os.path.exists(outline):
            logger.warning("Map outline not found: %s", outline)
            return plot

        with open(outline, "r", encoding="utf-8") as f:
            countries = GeoJSONDataSource(geojson=f.read())
        plot.patches("xs", "ys", color=None, line_color="black", fill_color=None, fill_alpha=0.2, source=countries, alpha=0.5)
        return plot

    def bokeh_plot_forecast_ensemble_mean(self, date, plot_width=700):
        """Generate ensemble-mean precip+wind HTML maps for all lead times."""
        precip_cube, u850_cube, v850_cube, speed_cube = self._load_required_cubes(date)
        cs_prob, ces_prob = self.cold_surge_probabilities(u850_cube, v850_cube, speed_cube)

        precip_mean = precip_cube.collapsed("realization", iris.analysis.MEAN)
        u850_mean = u850_cube.collapsed("realization", iris.analysis.MEAN)
        v850_mean = v850_cube.collapsed("realization", iris.analysis.MEAN)

        lons = precip_mean[0].coord("longitude").points
        lats = precip_mean[0].coord("latitude").points
        height = int(plot_width / (((max(lons) - min(lons)) / (max(lats) - min(lats))) * 1.0))
        date_label = date.strftime("%Y%m%d")
        ntimes = len(precip_cube.coord("forecast_period").points)

        model = self.config_values["model"]
        html_dir = os.path.join(self.config_values[f"{model}_cs_plot_ens"], date_label)
        os.makedirs(html_dir, exist_ok=True)

        for t in np.arange(ntimes):
            valid_date = date + datetime.timedelta(days=int(t))
            title = f"Ensemble mean P, UV850 [CS:{cs_prob[t]}%, CES:{ces_prob[t]}%]"
            subtitle = (
                f"Forecast start: {date_label}, Lead: T+{t}d "
                f"Valid for 24H up to {valid_date:%Y%m%d}"
            )

            plot = figure(height=height, width=plot_width, title=None, tools="pan,reset,save,box_zoom,wheel_zoom,hover")
            plot = self.plot_image_map(
                plot,
                precip_mean[t],
                palette=GnBu9,
                palette_reverse=True,
                low=5,
                high=30,
                cbar_title="Precipitation (mm/day)",
            )
            plot = self.plot_vectors(
                plot,
                u850_mean[t],
                v850_mean[t],
                palette=RdPu9,
                palette_reverse=True,
                maxSpeed=5,
                arrowHeadScale=0.2,
                arrowType="barbed",
            )
            plot = self._add_map_outline(plot)
            plot.add_layout(Title(text=subtitle, text_font_style="italic"), "above")
            plot.add_layout(Title(text=title, text_font_size="12pt"), "above")

            out_html = os.path.join(html_dir, f"Cold_surge_EnsMean_{date_label}_T{t * 24}h.html")
            output_file(out_html)
            save(plot)
            logger.info("Plotted %s", out_html)

        self.write_dates_json(
            date,
            os.path.join(self.config_values[f"{model}_cs_plot_ens"], f"{model}_ensmean_plot_dates.json"),
        )

    def bokeh_plot_forecast_probability_precip(self, date, precip_thresholds=None, plot_width=700):
        """Generate precip exceedance-probability HTML maps for all lead times."""
        if precip_thresholds is None:
            precip_thresholds = [10, 20, 30]

        precip_cube, u850_cube, v850_cube, speed_cube = self._load_required_cubes(date)
        cs_prob, ces_prob = self.cold_surge_probabilities(u850_cube, v850_cube, speed_cube)

        lons = precip_cube.coord("longitude").points
        lats = precip_cube.coord("latitude").points
        height = int(plot_width / (((max(lons) - min(lons)) / (max(lats) - min(lats))) * 1.0))
        date_label = date.strftime("%Y%m%d")
        ntimes = len(precip_cube.coord("forecast_period").points)

        model = self.config_values["model"]
        html_dir = os.path.join(self.config_values[f"{model}_cs_plot_ens"], date_label)
        os.makedirs(html_dir, exist_ok=True)

        for threshold in precip_thresholds:
            precip_prob = precip_cube.collapsed(
                "realization",
                iris.analysis.PROPORTION,
                function=lambda values, thr=threshold: values > thr,
            )

            for t in np.arange(ntimes):
                valid_date = date + datetime.timedelta(days=int(t))
                title = f"Ensemble probability of Precipitation [CS:{cs_prob[t]}%, CES:{ces_prob[t]}%]"
                subtitle = (
                    f"Forecast start: {date_label}, Lead: T+{t}d "
                    f"Valid for 24H up to {valid_date:%Y%m%d}"
                )

                plot = figure(height=height, width=plot_width, title=None, tools="pan,reset,save,box_zoom,wheel_zoom,hover")
                plot = self.plot_image_map(
                    plot,
                    precip_prob[t],
                    palette=GnBu9,
                    palette_reverse=True,
                    low=0.1,
                    high=1.0,
                    cbar_title=f"Precipitation probability (p >= {threshold} mm/day)",
                )
                plot = self._add_map_outline(plot)
                plot.add_layout(Title(text=subtitle, text_font_style="italic"), "above")
                plot.add_layout(Title(text=title, text_font_size="12pt"), "above")

                out_html = os.path.join(
                    html_dir, f"Cold_surge_ProbMaps_{date_label}_T{t * 24}h_Pr{threshold}.html"
                )
                output_file(out_html)
                save(plot)
                logger.info("Plotted %s", out_html)

        self.write_dates_json(
            date,
            os.path.join(self.config_values[f"{model}_cs_plot_ens"], f"{model}_ProbMaps_plot_dates.json"),
        )