import os
import logging
import datetime
import sys
import re
import numpy as np
import iris
import json
from bokeh.plotting import figure, save, output_file
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    Title,
    Range1d,
    LinearColorMapper,
    ColorBar,
    GeoJSONDataSource,
)
from bokeh.palettes import GnBu9, RdPu9, TolRainbow12
from salmon.core.task import Task
from salmon.utils.config import load_global_config
from salmon.utils.cube import read_ifs_grib_precip, read_ifs_grib_winds_correctly, read_winds_correctly, read_precip_correctly
from salmon.utils.bokeh_utils import Vector
import warnings

warnings.simplefilter("ignore")

logger = logging.getLogger(__name__)

_IFS_CYCLE_RE = re.compile(r"^(\d{8})\.eg(00|12)\.(grib|grb2)$")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_UTILS_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "utils"))

_DEFAULT_MAP_JSON = os.path.normpath(
    os.path.join(_UTILS_DIR, "map_data", "custom.geo.json")
)

FC_TIMES = tuple(np.arange(0, 174, 24))
DISPLAY_LAT_BOUNDS = (-10, 25)
DISPLAY_LON_BOUNDS = (90, 135)

# IFS ENS: 51 members (000–050); control member is 000.
IFS_N_MEMBERS = 51

VAR_SPECS = {
    "precip": {"iris_var": "UNKNOWN LOCAL PARAM 228.128"},
    "u850": {"iris_var": "x_wind", "pressure": 850},
    "v850": {"iris_var": "y_wind", "pressure": 850},
}


class RetrieveIFSColdSurgeData(Task):
    """
    Stage IFS GRIB cycle files for Cold Surge processing.

    Notes
    -----
        - Looks for raw IFS cycle files in ``<ifs.raw>`` (or ``<ifs.raw>/<YYYYMMDD>``).
        - Expected filename pattern: ``YYYYMMDD.egHH.grib`` (or ``.grb2``).
        - Uses two cycles to build one synthetic member directory ``000``:
            previous-day ``eg12`` and same-day ``eg00``.
    - Creates symlinks into the SALMON processed directory tree so that
      ``ComputeIFSColdSurgeIndices`` can locate them with the standard path
      convention.
        - Downstream compute averages available staged cycles into a single
            realization (member ``000``).
    """

    def run(self):
        """Task entrypoint."""
        date = self.context.date
        self._init_config_values()
        success = self.stage_ifs_data(date=date)
        if success:
            logger.info("IFS Cold Surge data staging complete.")
        else:
            logger.warning("IFS Cold Surge data staging completed with errors.")

    def _init_config_values(self):
        """Load and cache retrieval paths/config used by helper methods."""
        if hasattr(self, "config_values"):
            return

        ifs = load_global_config().get("ifs", {})
        self.config_values = {
            "ifs_raw_dir": ifs.get("raw", "/tmp/salmon_raw/ifs"),
            "ifs_processed_dir": ifs.get("processed", "/tmp/salmon_processed/ifs"),
        }
        logger.debug("IFS config values: %s", self.config_values)

    def _discover_cycle_files(self, raw_root):
        """Index available cycle files by (YYYYMMDD, HH) from raw directories."""
        cycle_index = {}
        search_dirs = [raw_root]

        # Also scan immediate date subdirectories if present.
        try:
            for child in os.listdir(raw_root):
                child_path = os.path.join(raw_root, child)
                if os.path.isdir(child_path) and child.isdigit() and len(child) == 8:
                    search_dirs.append(child_path)
        except OSError as exc:
            logger.debug("Unable to list raw root %s: %s", raw_root, exc)

        for search_dir in search_dirs:
            try:
                for filename in os.listdir(search_dir):
                    match = _IFS_CYCLE_RE.match(filename)
                    if not match:
                        continue
                    if filename.endswith(".keep.grib"):
                        continue

                    date_str, cycle_hour, ext = match.groups()
                    key = (date_str, cycle_hour)
                    full_path = os.path.join(search_dir, filename)
                    if not (os.path.exists(full_path) and os.path.getsize(full_path) > 0):
                        continue

                    # Prefer .grib over .grb2 when both exist for same key.
                    prev = cycle_index.get(key)
                    if prev is None:
                        cycle_index[key] = full_path
                    else:
                        prev_ext = os.path.splitext(prev)[1]
                        if prev_ext == ".grb2" and ext == "grib":
                            cycle_index[key] = full_path
            except OSError as exc:
                logger.debug("Unable to scan %s: %s", search_dir, exc)

        return cycle_index

    def stage_ifs_data(self, date):
        """
        Create symlinks from the raw IFS source tree into the SALMON processed
        directory tree so that ``ComputeIFSColdSurgeIndices`` can load them.

        Stages two cycle files as a synthetic single member (000):
        - previous day 12 UTC: (date - 1 day).eg12.grib
        - current day  00 UTC: date.eg00.grib

        Returns
        -------
        bool
            True if at least one file was linked, else False.
        """
        raw_root = self.config_values["ifs_raw_dir"]
        processed_base = self.config_values["ifs_processed_dir"]

        if not os.path.isdir(raw_root):
            logger.error("IFS raw source directory not found: %s", raw_root)
            return False

        cycle_index = self._discover_cycle_files(raw_root)
        if not cycle_index:
            logger.error("No IFS cycle files found in %s", raw_root)
            return False

        linked = 0
        errors = 0

        cycle_specs = [
            (date, "12", "ifs_cycle12"),
            (date, "00", "ifs_cycle00"),
        ]
        member_dir = "000"
        dest_dir = os.path.join(
            processed_base, "coldsurge", date.strftime("%Y%m%d"), member_dir
        )
        os.makedirs(dest_dir, exist_ok=True)

        chosen = []
        for cycle_date, cycle_hour, dest_stem in cycle_specs:
            cycle_date_str = cycle_date.strftime("%Y%m%d")
            src = cycle_index.get((cycle_date_str, cycle_hour))
            if src is None:
                logger.warning(
                    "IFS raw cycle file not found for %s eg%s",
                    cycle_date_str,
                    cycle_hour,
                )
                continue

            chosen.append(os.path.basename(src))

            ext = os.path.splitext(src)[1]
            dest = os.path.join(dest_dir, f"{dest_stem}{ext}")
            try:
                if os.path.islink(dest) or os.path.exists(dest):
                    os.remove(dest)
                os.symlink(src, dest)
                logger.debug("Linked %s -> %s", src, dest)
                linked += 1
            except Exception as exc:
                logger.error("Failed to link %s: %s", src, exc)
                errors += 1

        if chosen:
            logger.info(
                "IFS selected cycle files for %s member 000: %s",
                date.strftime("%Y%m%d"),
                ", ".join(chosen),
            )

        logger.info(
            "IFS staging complete for %s: %d files linked, %d errors",
            date.strftime("%Y%m%d"),
            linked,
            errors,
        )
        return linked > 0


class ComputeIFSColdSurgeIndices(Task):
    """
    Compute Cold Surge indices from staged IFS files.

    Outputs
    -------
    NetCDF files (one per variable) containing all available members:
      - precip
      - u850
      - v850

    Notes
    -----
    - Expects input files under:
        ``<ifs.processed>/coldsurge/<YYYYMMDD>/000/ifs_cycle{12,00}.<ext>``
    - Averages available cycle files into one synthetic realization ``000``.
    """

    def run(self):
        """Task entrypoint."""
        date = self.context.date
        self._init_config_values()
        self.process_forecast_data(date=date)

    def _init_config_values(self):
        """Load and cache model paths used by this task."""
        if hasattr(self, "config_values"):
            return

        ifs = load_global_config().get("ifs", {})
        processed_base = ifs.get("processed", "/tmp/salmon_processed/ifs")
        cs_processed_dir = os.path.join(processed_base, "coldsurge")

        self.config_values = {
            "ifs_processed_dir": processed_base,
            "cs_processed_dir": cs_processed_dir,
            "obsgrid_file": self.config.get(
                "obsgrid_file",
                os.path.join(os.getcwd(), "data", "obsgrid_145x73.nc"),
            ),
        }
        os.makedirs(cs_processed_dir, exist_ok=True)

    def _find_staged_cycle_file(self, base_dir, member, cycle_stem):
        """Locate a staged cycle file (ifs_cycle00 / ifs_cycle12) by member."""
        for ext in (".grib", ".grb2", ".nc"):
            candidate = os.path.join(base_dir, member, f"{cycle_stem}{ext}")
            if os.path.exists(candidate) and os.path.getsize(candidate) > 0:
                return candidate
        return None

    def _select_cycle_files(self, staged_root, member):
        """Return available staged cycle files for one member."""
        cycle_files = {
            "ifs_cycle12": self._find_staged_cycle_file(staged_root, member, "ifs_cycle12"),
            "ifs_cycle00": self._find_staged_cycle_file(staged_root, member, "ifs_cycle00"),
        }
        return {k: v for k, v in cycle_files.items() if v is not None}

    def _as_cube(self, obj, varname, source_file):
        """Normalise iris.load outputs (Cube/CubeList) into a single Cube."""
        if isinstance(obj, iris.cube.Cube):
            return obj
        if isinstance(obj, iris.cube.CubeList):
            if len(obj) == 0:
                logger.warning("Empty CubeList for %s from %s", varname, source_file)
                return None
            if len(obj) == 1:
                return obj[0]
            try:
                return obj.merge_cube()
            except Exception as exc:
                logger.warning(
                    "Could not merge CubeList for %s from %s (%s). Using first cube.",
                    varname,
                    source_file,
                    exc,
                )
                return obj[0]
        logger.warning(
            "Unexpected object type for %s from %s: %s",
            varname,
            source_file,
            type(obj),
        )
        return None

    def _build_cube_from_cycles(self, cycle_files, varname, spec, member):
        """Read one variable from staged cycles and return a single averaged cube."""
        cycle_00_cube = None
        cycle_12_cube = None

        if varname == "precip":
            if cycle_files.get("ifs_cycle00") is not None:
                loaded = read_ifs_grib_precip(
                    cycle_files["ifs_cycle00"], spec["iris_var"], deaccumulate=True
                )
                cycle_00_cube = self._as_cube(loaded, varname, cycle_files["ifs_cycle00"])
            if cycle_files.get("ifs_cycle12") is not None:
                loaded = read_ifs_grib_precip(
                    cycle_files["ifs_cycle12"], spec["iris_var"], deaccumulate=True
                )
                cycle_12_cube = self._as_cube(loaded, varname, cycle_files["ifs_cycle12"])

        if varname in ("u850", "v850"):
            if cycle_files.get("ifs_cycle00") is not None:
                loaded = read_ifs_grib_winds_correctly(
                    cycle_files["ifs_cycle00"], spec["iris_var"], pressure_level=spec.get("pressure")
                )
                cycle_00_cube = self._as_cube(loaded, varname, cycle_files["ifs_cycle00"])
            if cycle_files.get("ifs_cycle12") is not None:
                loaded = read_ifs_grib_winds_correctly(
                    cycle_files["ifs_cycle12"], spec["iris_var"], pressure_level=spec.get("pressure")
                )
                cycle_12_cube = self._as_cube(loaded, varname, cycle_files["ifs_cycle12"])

        if cycle_00_cube is not None and cycle_12_cube is not None:
            if cycle_00_cube.shape != cycle_12_cube.shape:
                logger.warning(
                    "Cycle shape mismatch for member %s var %s (00=%s, 12=%s). Using eg00 only.",
                    member,
                    varname,
                    cycle_00_cube.shape,
                    cycle_12_cube.shape,
                )
                return cycle_00_cube
            cycle_00_cube.data = (cycle_00_cube.data + cycle_12_cube.data) / 2.0
            return cycle_00_cube

        if cycle_00_cube is not None:
            return cycle_00_cube
        if cycle_12_cube is not None:
            return cycle_12_cube
        return None

    def load_base_cube(self):   
        """Load observational target grid used for optional regridding."""
        base_cube = iris.load_cube(self.config_values["obsgrid_file"])
        for coord_name, units in (
            ("latitude", "degrees_north"),
            ("longitude", "degrees_east"),
        ):
            base_cube.coord(coord_name).units = units
            base_cube.coord(coord_name).coord_system = None
        return base_cube

    def regrid2obs(self, cube):
        """Regrid cube to the observational grid using linear interpolation."""
        base_cube = self.load_base_cube()
        for coord_name, units in (
            ("latitude", "degrees_north"),
            ("longitude", "degrees_east"),
        ):
            coord = cube.coord(coord_name)
            coord.units = units
            coord.coord_system = None
            if coord.bounds is None:
                coord.guess_bounds()
        return cube.regrid(base_cube, iris.analysis.Linear())

    def process_forecast_data(self, date):
        """
        Build and save all-member Cold Surge NetCDF files for precip, u850, v850.
        """
        date_str = date.strftime("%Y%m%d")
        staged_root = os.path.join(self.config_values["cs_processed_dir"], date_str)
        logger.info("Staged IFS root: %s", staged_root)

        out_root = self.config_values["cs_processed_dir"]
        regrid_to_obs = bool(self.config.get("regrid_to_obs", False))

        logger.info("Computing IFS Cold Surge indices for %s", date_str)

        available_members = ["000"] if os.path.isdir(
            os.path.join(staged_root, "000")
        ) else []
        if not available_members:
            logger.error("No staged IFS synthetic member found in %s", staged_root)
            return
        
        for varname, spec in VAR_SPECS.items():

            out_dir = os.path.join(out_root, varname)
            os.makedirs(out_dir, exist_ok=True)

            out_file = os.path.join(
                out_dir,
                f"{varname}_ColdSurge_24h_allMember_{date_str}.nc",
            )
            if os.path.exists(out_file):
                logger.info("%s already exists. Skipping.", out_file)
                continue

            cubes = []
            for mem in available_members:
                cycle_files = self._select_cycle_files(staged_root, mem)
                if not cycle_files:
                    logger.warning(
                        "No cycle files found for member %s var %s under %s",
                        mem,
                        varname,
                        staged_root,
                    )
                    continue

                logger.info(
                    "Processing member %s var %s with cycles: %s",
                    mem,
                    varname,
                    sorted(cycle_files.keys()),
                )
                merged = self._build_cube_from_cycles(cycle_files, varname, spec, mem)
                if merged is None:
                    logger.warning(
                        "No merged cube produced for member %s var %s",
                        mem,
                        varname,
                    )
                    continue

                if regrid_to_obs:
                    merged = self.regrid2obs(merged)

                merged.add_aux_coord(
                    iris.coords.AuxCoord(
                        int(mem),
                        standard_name="realization",
                        var_name="realization",
                    )
                )
                cubes.append(merged)

            if not cubes:
                logger.error("No cubes to save for %s on %s", varname, date_str)
                continue

            save_cube = cubes[0] if len(cubes) == 1 else iris.cube.CubeList(cubes).merge_cube()
            iris.save(save_cube, out_file, netcdf_format="NETCDF4_CLASSIC")
            logger.info("Saved IFS merged members to %s", out_file)


class DisplayIFSColdSurgeMaps(Task):
    """
    Create Bokeh map products from processed IFS Cold Surge NetCDF files.

    Products
    --------
    - Ensemble-mean precip + 850 hPa wind vectors (HTML)
    - Ensemble probability precip maps for configured thresholds (HTML)
    - JSON exports for JS dashboards (prob maps + ensemble mean)
    """

    def run(self):
        """Task entrypoint."""
        date = self.context.date
        self._init_config_values()

        do_ensmean = bool(self.config.get("plot_ensmean", True))
        do_probmaps = bool(self.config.get("plot_probmaps", True))
        do_prob_json = bool(self.config.get("export_prob_json", True))
        do_ensmean_json = bool(self.config.get("export_ensmean_json", True))
        precip_thresholds = self.config.get("precip_thresholds", [10, 20, 30])

        if do_ensmean:
            self.bokeh_plot_forecast_ensemble_mean(date)
        if do_probmaps:
            self.bokeh_plot_forecast_probability_precip(date, precip_thresholds=precip_thresholds)
        if do_prob_json:
            self.export_forecast_probability_precip_json(date, precip_thresholds=precip_thresholds)
        if do_ensmean_json:
            self.export_forecast_ensemble_mean_json(date)

    def _init_config_values(self):
        """Load and cache plotting/config paths."""
        if hasattr(self, "config_values"):
            return

        ifs = load_global_config().get("ifs", {})
        processed_base = ifs.get("processed", "/tmp/salmon_processed/ifs")
        plot_root = ifs.get("plots", processed_base)
        cs_processed_dir = os.path.join(processed_base, "coldsurge")
        cs_plot_ens_dir = os.path.join(plot_root, "coldsurge", "plot_ens")

        self.config_values = {
            "ifs_cs_processed_dir": cs_processed_dir,
            "ifs_cs_plot_ens": cs_plot_ens_dir,
            "ifs_cs_plot_prob": os.path.join(cs_plot_ens_dir, "prob"),
            "ifs_cs_json_ensmean": os.path.join(cs_plot_ens_dir, "json_ensmean"),
            "map_outline_json_file": self.config.get(
                "map_outline_json_file",
                os.path.normpath(_DEFAULT_MAP_JSON),
            ),
        }
        os.makedirs(cs_plot_ens_dir, exist_ok=True)
        self.xSkip, self.ySkip = 5, 5

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def get_file_name(self, date, varname):
        """Return processed NetCDF path for variable/date."""
        root = self.config_values["ifs_cs_processed_dir"]
        return os.path.join(
            root, varname, f"{varname}_ColdSurge_24h_allMember_{date:%Y%m%d}.nc"
        )

    def _load_required_cubes(self, date):
        """Load precip/u850/v850 cubes and compute wind speed."""
        precip_cube = iris.load_cube(self.get_file_name(date, "precip"))
        u850_cube = iris.load_cube(self.get_file_name(date, "u850"))
        v850_cube = iris.load_cube(self.get_file_name(date, "v850"))
        precip_cube = precip_cube.intersection(
            latitude=DISPLAY_LAT_BOUNDS, longitude=DISPLAY_LON_BOUNDS
        )
        u850_cube = u850_cube.intersection(
            latitude=DISPLAY_LAT_BOUNDS, longitude=DISPLAY_LON_BOUNDS
        )
        v850_cube = v850_cube.intersection(
            latitude=DISPLAY_LAT_BOUNDS, longitude=DISPLAY_LON_BOUNDS
        )
        speed_cube = (u850_cube ** 2 + v850_cube ** 2) ** 0.5
        return precip_cube, u850_cube, v850_cube, speed_cube

    def extract_and_collapse(self, cube, box):
        """Area-mean over [lon0, lon1, lat0, lat1]."""
        sub = cube.intersection(latitude=(box[2], box[3]), longitude=(box[0], box[1]))
        return sub.collapsed(("latitude", "longitude"), iris.analysis.MEAN)

    def cold_surge_probabilities(self, u850_cube, v850_cube, speed_cube):
        """Compute CS and CES ensemble probabilities (%) by lead time."""
        chang_box = [107, 115, 5, 10]
        hattori_box = [105, 115, -5, 5]
        chang_threshold = 9.0
        hattori_threshold = -2.0

        u850_ba = self.extract_and_collapse(u850_cube, chang_box)
        v850_ba = self.extract_and_collapse(v850_cube, chang_box)
        speed_ba = self.extract_and_collapse(speed_cube, chang_box)
        v850_hattori = self.extract_and_collapse(v850_cube, hattori_box)

        mask_cs = (
            (u850_ba.data < 0.0)
            & (v850_ba.data < 0.0)
            & (speed_ba.data >= chang_threshold)
        )
        mask_ces = mask_cs & (v850_hattori.data <= hattori_threshold)

        cs_prob = [
            round(p, 1)
            for p in 100.0 * np.sum(mask_cs, axis=0) / float(len(mask_cs))
        ]
        ces_prob = [
            round(p, 1)
            for p in 100.0 * np.sum(mask_ces, axis=0) / float(len(mask_ces))
        ]
        return cs_prob, ces_prob

    def write_dates_json(self, date, json_file):
        """Append YYYYMMDD to a JSON date list (unique + sorted)."""
        new_date = date.strftime("%Y%m%d")
        if not os.path.exists(json_file):
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump([new_date], f, indent=2)
            return
        with open(json_file, "r", encoding="utf-8") as f:
            existing = json.load(f)
        if new_date not in existing:
            existing.append(new_date)
            existing.sort()
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)

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
            palette=palette,
            low=kwargs.get("low", 0.0),
            high=kwargs.get("high", 1.0),
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
            x="lon", y="lat", size=8,
            alpha=0.0, line_alpha=0.0, fill_alpha=0.0,
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
        plot.patches(
            xs="xs", ys="ys",
            fill_color="colors", line_color="colors",
            alpha=0.5, source=source,
        )
        return plot

    # ------------------------------------------------------------------
    # Plot methods
    # ------------------------------------------------------------------

    def bokeh_plot_forecast_ensemble_mean(self, date, plot_width=700):
        """Generate ensemble-mean precip + wind HTML maps for all lead times."""
        precip_cube, u850_cube, v850_cube, speed_cube = self._load_required_cubes(date)
        cs_prob, ces_prob = self.cold_surge_probabilities(u850_cube, v850_cube, speed_cube)

        precip_mean = precip_cube.collapsed("realization", iris.analysis.MEAN)
        u850_mean = u850_cube.collapsed("realization", iris.analysis.MEAN)
        v850_mean = v850_cube.collapsed("realization", iris.analysis.MEAN)

        lons = precip_mean[0].coord("longitude").points
        lats = precip_mean[0].coord("latitude").points
        height = int(
            plot_width / (((max(lons) - min(lons)) / (max(lats) - min(lats))) * 1.0)
        )
        date_label = date.strftime("%Y%m%d")
        ntimes = len(precip_cube.coord("forecast_period").points)

        html_dir = os.path.join(self.config_values["ifs_cs_plot_ens"], date_label)
        os.makedirs(html_dir, exist_ok=True)

        for t in np.arange(ntimes):
            valid_date = date + datetime.timedelta(days=int(t))
            title = f"IFS Ensemble mean P, UV850 [CS:{cs_prob[t]}%, CES:{ces_prob[t]}%]"
            subtitle = (
                f"Forecast start: {date_label}, Lead: T+{t}d "
                f"Valid for 24H up to {valid_date:%Y%m%d}"
            )
            plot = figure(
                height=height, width=plot_width,
                title=None,
                tools="pan,reset,save,box_zoom,wheel_zoom",
            )
            plot = self.plot_image_map(
                plot, precip_mean[t],
                palette=GnBu9, palette_reverse=True,
                low=5, high=30, cbar_title="Precipitation (mm/day)",
            )
            plot = self.plot_vectors(
                plot, u850_mean[t], v850_mean[t],
                palette=RdPu9, palette_reverse=True,
                maxSpeed=5, arrowHeadScale=0.2, arrowType="barbed",
            )
            plot = self._add_map_outline(plot)
            plot.add_layout(Title(text=subtitle, text_font_style="italic"), "above")
            plot.add_layout(Title(text=title, text_font_size="12pt"), "above")

            out_html = os.path.join(
                html_dir, f"Cold_surge_IFS_EnsMean_{date_label}_T{t * 24}h.html"
            )
            output_file(out_html)
            save(plot)
            logger.info("Plotted %s", out_html)

        self.write_dates_json(
            date,
            os.path.join(self.config_values["ifs_cs_plot_ens"], "ifs_ensmean_plot_dates.json"),
        )

    def bokeh_plot_forecast_probability_precip(
        self, date, precip_thresholds=None, plot_width=700
    ):
        """Generate precip exceedance-probability HTML maps for all lead times."""
        if precip_thresholds is None:
            precip_thresholds = [10, 20, 30]

        precip_cube, u850_cube, v850_cube, speed_cube = self._load_required_cubes(date)
        cs_prob, ces_prob = self.cold_surge_probabilities(u850_cube, v850_cube, speed_cube)

        lons = precip_cube.coord("longitude").points
        lats = precip_cube.coord("latitude").points
        height = int(
            plot_width / (((max(lons) - min(lons)) / (max(lats) - min(lats))) * 1.0)
        )
        date_label = date.strftime("%Y%m%d")
        ntimes = len(precip_cube.coord("forecast_period").points)

        html_dir = os.path.join(self.config_values["ifs_cs_plot_ens"], date_label)
        os.makedirs(html_dir, exist_ok=True)

        for threshold in precip_thresholds:
            precip_prob = precip_cube.collapsed(
                "realization",
                iris.analysis.PROPORTION,
                function=lambda values, thr=threshold: values > thr,
            )
            for t in np.arange(ntimes):
                valid_date = date + datetime.timedelta(days=int(t))
                title = (
                    f"IFS Ensemble probability of Precipitation "
                    f"[CS:{cs_prob[t]}%, CES:{ces_prob[t]}%]"
                )
                subtitle = (
                    f"Forecast start: {date_label}, Lead: T+{t}d "
                    f"Valid for 24H up to {valid_date:%Y%m%d}"
                )
                plot = figure(
                    height=height, width=plot_width,
                    title=None,
                    tools="pan,reset,save,box_zoom,wheel_zoom",
                )
                plot = self.plot_image_map(
                    plot, precip_prob[t],
                    palette=GnBu9, palette_reverse=True,
                    low=0.1, high=1.0,
                    cbar_title=f"Precipitation probability (p >= {threshold} mm/day)",
                )
                plot = self._add_map_outline(plot)
                plot.add_layout(Title(text=subtitle, text_font_style="italic"), "above")
                plot.add_layout(Title(text=title, text_font_size="12pt"), "above")

                out_html = os.path.join(
                    html_dir,
                    f"Cold_surge_IFS_ProbMaps_{date_label}_T{t * 24}h_Pr{threshold}.html",
                )
                output_file(out_html)
                save(plot)
                logger.info("Plotted %s", out_html)

        self.write_dates_json(
            date,
            os.path.join(
                self.config_values["ifs_cs_plot_ens"], "ifs_ProbMaps_plot_dates.json"
            ),
        )

    def export_forecast_probability_precip_json(self, date, precip_thresholds=None):
        """
        Export precip exceedance probability grids to JSON for JS dashboards.

        Output structure
        ----------------
        {
          "model": "ifs",
          "forecast_start": "YYYYMMDD",
          "longitude": [...],
          "latitude": [...],
          "products": [
            {
              "threshold_mm_day": 10,
              "lead_hours": 24,
              "valid_to": "YYYYMMDD",
              "cs_prob_percent": 12.3,
              "ces_prob_percent": 4.5,
              "grid_shape": [nlat, nlon],
              "probability_flat": [...]
            },
            ...
          ]
        }
        """
        if precip_thresholds is None:
            precip_thresholds = [10, 20, 30]

        precip_cube, u850_cube, v850_cube, speed_cube = self._load_required_cubes(date)
        cs_prob, ces_prob = self.cold_surge_probabilities(u850_cube, v850_cube, speed_cube)

        lons = precip_cube.coord("longitude").points
        lats = precip_cube.coord("latitude").points
        ntimes = len(precip_cube.coord("forecast_period").points)

        out_dir = os.path.join(
            self.config_values["ifs_cs_plot_prob"], date.strftime("%Y%m%d")
        )
        os.makedirs(out_dir, exist_ok=True)

        payload = {
            "model": "ifs",
            "forecast_start": date.strftime("%Y%m%d"),
            "longitude": [float(x) for x in lons],
            "latitude": [float(y) for y in lats],
            "products": [],
        }

        for threshold in precip_thresholds:
            precip_prob = precip_cube.collapsed(
                "realization",
                iris.analysis.PROPORTION,
                function=lambda values, thr=threshold: values > thr,
            )
            for t in range(ntimes):
                grid = np.asarray(precip_prob[t].data, dtype=float)
                payload["products"].append(
                    {
                        "threshold_mm_day": float(threshold),
                        "lead_hours": int(t * 24),
                        "valid_to": (
                            date + datetime.timedelta(days=int(t))
                        ).strftime("%Y%m%d"),
                        "cs_prob_percent": float(cs_prob[t]),
                        "ces_prob_percent": float(ces_prob[t]),
                        "grid_shape": [int(grid.shape[0]), int(grid.shape[1])],
                        "probability_flat": [float(v) for v in grid.ravel(order="C")],
                    }
                )

        out_json = os.path.join(
            out_dir, f"Cold_surge_IFS_ProbMaps_{date.strftime('%Y%m%d')}.json"
        )
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("Exported IFS probability overlay JSON: %s", out_json)

    def export_forecast_ensemble_mean_json(self, date):
        """
        Export ensemble-mean precip and 850 hPa wind grids to JSON for JS dashboards.

        Output structure
        ----------------
        {
          "model": "ifs",
          "forecast_start": "YYYYMMDD",
          "variable": "precipitation_amount",
          "longitude": [...],
          "latitude": [...],
          "products": [
            {
              "lead_hours": 0,
              "valid_to": "YYYYMMDD",
              "cs_prob_percent": 12.3,
              "ces_prob_percent": 4.5,
              "grid_shape": [nlat, nlon],
              "precip_mean_flat": [...],
              "u850_mean_flat": [...],
              "v850_mean_flat": [...]
            },
            ...
          ]
        }
        """
        precip_cube, u850_cube, v850_cube, speed_cube = self._load_required_cubes(date)
        cs_prob, ces_prob = self.cold_surge_probabilities(u850_cube, v850_cube, speed_cube)

        precip_mean = precip_cube.collapsed("realization", iris.analysis.MEAN)
        u850_mean = u850_cube.collapsed("realization", iris.analysis.MEAN)
        v850_mean = v850_cube.collapsed("realization", iris.analysis.MEAN)

        lons = precip_mean[0].coord("longitude").points
        lats = precip_mean[0].coord("latitude").points
        ntimes = len(precip_mean.coord("forecast_period").points)

        out_dir = os.path.join(
            self.config_values["ifs_cs_json_ensmean"], date.strftime("%Y%m%d")
        )
        os.makedirs(out_dir, exist_ok=True)

        payload = {
            "model": "ifs",
            "forecast_start": date.strftime("%Y%m%d"),
            "variable": "precipitation_amount",
            "longitude": [float(x) for x in lons],
            "latitude": [float(y) for y in lats],
            "products": [],
        }

        for t in range(ntimes):
            precip_grid = np.asarray(precip_mean[t].data, dtype=float)
            u850_grid = np.asarray(u850_mean[t].data, dtype=float)
            v850_grid = np.asarray(v850_mean[t].data, dtype=float)
            payload["products"].append(
                {
                    "lead_hours": int(t * 24),
                    "valid_to": (
                        date + datetime.timedelta(days=int(t))
                    ).strftime("%Y%m%d"),
                    "cs_prob_percent": float(cs_prob[t]),
                    "ces_prob_percent": float(ces_prob[t]),
                    "grid_shape": [
                        int(precip_grid.shape[0]),
                        int(precip_grid.shape[1]),
                    ],
                    "precip_mean_flat": [
                        round(float(v), 3) for v in precip_grid.ravel(order="C")
                    ],
                    "u850_mean_flat": [
                        round(float(v), 3) for v in u850_grid.ravel(order="C")
                    ],
                    "v850_mean_flat": [
                        round(float(v), 3) for v in v850_grid.ravel(order="C")
                    ],
                }
            )

        out_json = os.path.join(
            out_dir, f"Cold_surge_IFS_EnsMean_{date.strftime('%Y%m%d')}.json"
        )
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("Exported IFS ensemble-mean JSON: %s", out_json)
