"""
Tests for SALMON utility modules:
  - salmon.utils.config   (load_global_config, expand_env_vars)
  - salmon.utils.cube     (subset_seasia, create_latlon_grid, remove_um_version)
  - salmon.utils.calculus (differentiate, cube_delta)
"""
import os
import textwrap
import tempfile
import pytest
import numpy as np
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Config utils
# ---------------------------------------------------------------------------
class TestConfig:
    """Tests for salmon.utils.config"""

    def test_expand_env_vars_string(self):
        from salmon.utils.config import expand_env_vars
        os.environ["_TEST_SALMON_VAR"] = "/some/path"
        result = expand_env_vars("$_TEST_SALMON_VAR/data")
        assert result == "/some/path/data"

    def test_expand_env_vars_dict(self):
        from salmon.utils.config import expand_env_vars
        os.environ["_SALMON_ROOT"] = "/root"
        data = {"path": "$_SALMON_ROOT/subdir", "count": 5}
        result = expand_env_vars(data)
        assert result["path"] == "/root/subdir"
        assert result["count"] == 5  # non-string unchanged

    def test_expand_env_vars_list(self):
        from salmon.utils.config import expand_env_vars
        os.environ["_SALMON_X"] = "expanded"
        result = expand_env_vars(["$_SALMON_X", "plain"])
        assert result == ["expanded", "plain"]

    def test_expand_env_vars_nested(self):
        from salmon.utils.config import expand_env_vars
        os.environ["_SALMON_DEEP"] = "deep_value"
        data = {"outer": {"inner": "$_SALMON_DEEP"}}
        result = expand_env_vars(data)
        assert result["outer"]["inner"] == "deep_value"

    def test_load_global_config_from_explicit_path(self):
        from salmon.utils.config import load_global_config
        config_content = textwrap.dedent("""\
            analysis:
              raw: /tmp/raw
              processed: /tmp/processed
            global:
              forecast_out_dir: /tmp/out
        """)
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(config_content)
            tmp = f.name
        try:
            cfg = load_global_config(config_path=tmp)
            assert cfg["analysis"]["raw"] == "/tmp/raw"
            assert cfg["global"]["forecast_out_dir"] == "/tmp/out"
        finally:
            os.unlink(tmp)

    def test_load_global_config_returns_empty_when_no_file(self):
        from salmon.utils.config import load_global_config
        cfg = load_global_config(config_path="/nonexistent/path/salmon_config.yaml")
        assert cfg == {}

    def test_load_global_config_expands_env_vars(self):
        from salmon.utils.config import load_global_config
        os.environ["_SALMON_TEST_DIR"] = "/expanded_dir"
        config_content = "analysis:\n  raw: $_SALMON_TEST_DIR/raw\n"
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(config_content)
            tmp = f.name
        try:
            cfg = load_global_config(config_path=tmp)
            assert cfg["analysis"]["raw"] == "/expanded_dir/raw"
        finally:
            os.unlink(tmp)


# ---------------------------------------------------------------------------
# Cube utils (no file I/O — use in-memory iris cubes)
# ---------------------------------------------------------------------------
class TestCubeUtils:
    """Tests for salmon.utils.cube (non-I/O functions)"""

    def _make_simple_cube(self, lat_range=(-90, 90), lon_range=(0, 357.5)):
        """Helper: build a tiny 2D cube with lat/lon coords.

        Longitude spans 0–357.5° (not 360) with circular=True so that
        iris's cube.intersection() works correctly on global grids.
        """
        import iris
        import iris.coords

        lats = np.linspace(lat_range[0], lat_range[1], 7)
        lons = np.linspace(lon_range[0], lon_range[1], 13)
        data = np.ones((len(lats), len(lons)))

        lat_coord = iris.coords.DimCoord(lats, standard_name="latitude", units="degrees")
        lon_coord = iris.coords.DimCoord(
            lons, standard_name="longitude", units="degrees", circular=True
        )
        lat_coord.guess_bounds()
        lon_coord.guess_bounds()

        cube = iris.cube.Cube(
            data,
            dim_coords_and_dims=[(lat_coord, 0), (lon_coord, 1)],
        )
        return cube

    def test_remove_um_version_removes_attribute(self):
        from salmon.utils.cube import remove_um_version
        import iris
        # Pass attributes at construction time to avoid iris 3.14's lazy
        # fileformats import triggered by cube.attributes.__setitem__
        cube = iris.cube.Cube(
            np.zeros((3, 3)),
            attributes={"um_version": "11.1", "other": "keep"},
        )
        remove_um_version(cube, None, "dummy_file")
        assert "um_version" not in cube.attributes
        assert cube.attributes.get("other") == "keep"

    def test_remove_um_version_no_attribute_is_safe(self):
        from salmon.utils.cube import remove_um_version
        import iris
        cube = iris.cube.Cube(np.zeros((3, 3)))
        # Should not raise
        remove_um_version(cube, None, "dummy_file")

    def test_subset_seasia_restricts_domain(self):
        from salmon.utils.cube import subset_seasia, create_latlon_grid
        # Use create_latlon_grid (1° spacing) so grid points align with the SE
        # Asia boundaries (-10 to 25 lat, 85 to 145 lon) exactly.
        global_grid = create_latlon_grid()   # 1° global grid
        sea_cube = subset_seasia(global_grid)
        # All lat points must lie within the SE Asia window
        assert sea_cube.coord("latitude").points.min() >= -10
        assert sea_cube.coord("latitude").points.max() <= 25
        assert sea_cube.coord("longitude").points.min() >= 85
        assert sea_cube.coord("longitude").points.max() <= 145
        # Subsetting must have actually reduced the domain
        assert sea_cube.shape[0] < global_grid.shape[0]
        assert sea_cube.shape[1] < global_grid.shape[1]


    def test_create_latlon_grid_default(self):
        from salmon.utils.cube import create_latlon_grid
        grid = create_latlon_grid()
        assert "latitude" in [c.standard_name for c in grid.coords()]
        assert "longitude" in [c.standard_name for c in grid.coords()]
        # Should be 2D (lat, lon)
        assert grid.ndim == 2

    def test_create_latlon_grid_custom_spacing(self):
        from salmon.utils.cube import create_latlon_grid
        grid = create_latlon_grid(
            latitudes=(-10, 10), longitudes=(0, 30), spacing=2.5
        )
        lats = grid.coord("latitude").points
        assert lats[0] >= -10 and lats[-1] <= 10

    def test_create_latlon_grid_data_all_zeros(self):
        from salmon.utils.cube import create_latlon_grid
        grid = create_latlon_grid()
        assert np.all(grid.data == 0)


# ---------------------------------------------------------------------------
# Calculus utils
# ---------------------------------------------------------------------------
class TestCalculus:
    """Tests for salmon.utils.calculus"""

    def _make_1d_cube(self, values, coord_name="longitude", units="degrees"):
        """Build a simple 1-D cube for differentiation tests."""
        import iris
        import iris.coords
        arr = np.array(values, dtype=float)
        coord_points = np.linspace(0, 10 * (len(arr) - 1), len(arr))
        coord = iris.coords.DimCoord(coord_points, standard_name=coord_name, units=units)
        coord.guess_bounds()
        cube = iris.cube.Cube(arr, dim_coords_and_dims=[(coord, 0)])
        return cube

    def test_differentiate_linear_function(self):
        """d/dx of a linear function y = 2x should give constant ~2."""
        from salmon.utils.calculus import differentiate
        # 5-point linear function: y = 2x where x = 0, 10, 20, 30, 40
        values = [0.0, 20.0, 40.0, 60.0, 80.0]
        cube = self._make_1d_cube(values)
        dcube = differentiate(cube, "longitude")
        # All derivative values should be ≈ 2.0
        np.testing.assert_allclose(dcube.data, 2.0, rtol=1e-5)

    def test_cube_delta_output_shape(self):
        """cube_delta should return an array one shorter than input along the diff axis."""
        from salmon.utils.calculus import cube_delta
        cube = self._make_1d_cube([1.0, 4.0, 9.0, 16.0])
        result = cube_delta(cube, "longitude")
        assert result.shape[0] == len(cube.data) - 1

    def test_cube_delta_values(self):
        """Differences of [1, 4, 9, 16] should be [3, 5, 7]."""
        from salmon.utils.calculus import cube_delta
        cube = self._make_1d_cube([1.0, 4.0, 9.0, 16.0])
        result = cube_delta(cube, "longitude")
        np.testing.assert_allclose(result.data, [3.0, 5.0, 7.0])

    def test_differentiate_renames_cube(self):
        from salmon.utils.calculus import differentiate
        cube = self._make_1d_cube([0.0, 1.0, 2.0])
        cube.rename("air_temperature")
        dcube = differentiate(cube, "longitude")
        assert "derivative_of_air_temperature_wrt_longitude" in dcube.name()
