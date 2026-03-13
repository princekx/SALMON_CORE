import os
import logging
import datetime
import uuid
import concurrent.futures
import subprocess
from typing import Dict, Any, List
import numpy as np
import iris
import iris.coords
from salmon.core.task import Task
from salmon.utils.moose import MooseClient
from salmon.utils.config import load_global_config
from salmon.utils.cube import read_winds_correctly, read_precip_correctly, subset_seasia
import sys
import warnings
# Set the global warning filter to ignore all warnings
warnings.simplefilter("ignore")

logger = logging.getLogger(__name__)

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
        date = self.context.date
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
        cs_processed_dir = os.path.join(processed_base, "coldsurge", self.context.recipe_name)

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

    def subset_seasia(self, cube):
        """Subset cube to Southeast Asia domain."""
        return cube.intersection(latitude=(-10, 25), longitude=(85, 145))

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

    def read_precip_correctly(self, data_files, varname="precipitation_amount", lbproc=0):
        """
        Read and merge precipitation cubes across forecast steps.

        Parameters
        ----------
        data_files : list[str]
            Input PP files.
        varname : str
            Variable name to load from files.
        lbproc : int
            Kept for compatibility; unused in this implementation.
        """
        cubes = []
        for data_file in sorted(data_files):
            cube = iris.load_cube(data_file, varname)
            if cube.ndim == 3:
                cube = cube.collapsed("time", iris.analysis.MEAN)
            cube = self._ensure_bounds(cube)
            cube = self._normalise_for_merge(cube)
            cubes.append(cube)

        if not cubes:
            raise ValueError("No precipitation cubes loaded.")

        # Align metadata to avoid merge issues
        ref_cell_methods = cubes[0].cell_methods
        for cube in cubes:
            cube.cell_methods = ref_cell_methods
            if cube.coords("forecast_period"):
                fp = cube.coord("forecast_period")
                cube.replace_coord(
                    iris.coords.DimCoord(
                        fp.points, standard_name="forecast_period", units=fp.units
                    )
                )

        merged = iris.cube.CubeList(cubes).merge_cube()
        return self.subset_seasia(merged)

    def read_olr_correctly(self, data_files, varname="olr", lbproc=0):
        """
        Read and merge OLR cubes.

        Notes
        -----
        This keeps compatibility with STASH-based PP filtering.
        """
        if varname != "olr":
            raise ValueError("read_olr_correctly currently supports varname='olr' only.")

        from iris.fileformats.rules import load_pairs_from_fields

        stash_code = "m01s02i205"
        cubes = []

        for data_file in sorted(data_files):
            if not os.path.exists(data_file):
                raise FileNotFoundError(f"{data_file} does not exist.")

            filtered_fields = []
            for field in iris.fileformats.pp.load(data_file):
                if field.stash == stash_code and field.lbproc == lbproc:
                    filtered_fields.append(field)

            cube_field_pairs = load_pairs_from_fields(filtered_fields)
            for cube, field in cube_field_pairs:
                cube.attributes["lbproc"] = field.lbproc
                cubes.append(cube)

        if not cubes:
            raise ValueError("No OLR cubes loaded.")

        iris.util.equalise_attributes(cubes)
        merged = iris.cube.CubeList(cubes).merge_cube()
        return self.subset_seasia(merged)

    def read_winds_correctly(self, data_files, varname, pressure_level=None):
        """
        Read and merge wind cubes across forecast steps.
        """
        cubes = []
        for data_file in sorted(data_files):
            cube = iris.load_cube(data_file, varname)
            if pressure_level is not None:
                cube = cube.extract(iris.Constraint(pressure=pressure_level))
            if cube.ndim == 3:
                cube = cube.collapsed("time", iris.analysis.MEAN)
            cube = self._ensure_bounds(cube)
            cubes.append(cube)

        if not cubes:
            raise ValueError(f"No wind cubes loaded for {varname}.")

        iris.util.equalise_attributes(cubes)
        for cube in cubes:
            cube.cell_methods = ()

        merged = iris.cube.CubeList(cubes).merge_cube()
        return self.subset_seasia(merged)

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
                    cube = self.read_precip_correctly(existing_files, spec["iris_var"])
                    if cube.shape[0] > 1:
                        cube.data[1:] -= cube.data[:-1]
                else:
                    cube = self.read_winds_correctly(
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
