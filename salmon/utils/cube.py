import iris
import os
from typing import List, Optional

def load_obs_grid(base_dir: str) -> iris.cube.Cube:
    """Load the standard observation grid for regridding.

    Args:
        base_dir (str): Base directory for SALMON data.

    Returns:
        iris.cube.Cube: The target observation grid cube.
    """
    obs_file = os.path.join(base_dir, 'data', 'obsgrid_145x73.nc')
    return iris.load_cube(obs_file)

def regrid_to_obs(cube: iris.cube.Cube, target_grid: iris.cube.Cube) -> iris.cube.Cube:
    """Regrid an input cube to the standard observation grid.

    Args:
        cube (iris.cube.Cube): The source cube to regrid.
        target_grid (iris.cube.Cube): The target grid cube.

    Returns:
        iris.cube.Cube: The regridded cube.
    """
    return cube.regrid(target_grid, iris.analysis.Linear())

def remove_um_version(cube, field, filename):
    """Callback to remove the 'um_version' attribute from cubes.

    This is useful during loading to avoid merge issues between different
    UM version outputs.
    """
    cube.attributes.pop('um_version', None)

def read_winds_correctly(files: List[str], var_name: str, pressure_level: Optional[int] = None) -> iris.cube.Cube:
    """Load and process wind data, handling multiple members and pressure levels.
    
    Averages 3-hourly data to daily means if necessary and handles forecast 
    period/time bounds.
    """
    files.sort()
    cubes = []
    for data_file in files:
        cube = iris.load_cube(data_file, var_name)
        if pressure_level is not None:
            cube = cube.extract(iris.Constraint(pressure=pressure_level))
        
        # Average 3-hourly data to daily mean if 3D (time, lat, lon)
        if len(cube.shape) == 3:
            cube = cube.collapsed('time', iris.analysis.MEAN)
            
        # Ensure bounds for forecast_period and time
        for coord_name in ['forecast_period', 'time']:
            if cube.coord(coord_name).bounds is None:
                p = cube.coord(coord_name).points[0]
                cube.coord(coord_name).bounds = [[p - 1.0, p + 1.0]]
        
        cubes.append(cube)

    iris.util.equalise_attributes(cubes)
    # Clear cell methods to avoid merge conflicts
    for cube in cubes:
        cube.cell_methods = ()

    return iris.cube.CubeList(cubes).merge_cube()

def read_olr_correctly(files: List[str], lbproc: int = 0) -> iris.cube.Cube:
    """Load OLR data from PP files using STASH code m01s02i205.
    
    Handles iris file loading quirks and merges cubes.
    """
    from iris.fileformats.pp import load_pairs_from_fields
    stash_code = 'm01s02i205'
    files.sort()
    cubes = []
    for data_file in files:
        filtered_fields = []
        for field in iris.fileformats.pp.load(data_file):
            if field.stash == stash_code and field.lbproc == lbproc:
                filtered_fields.append(field)
        
        cube_field_pairs = load_pairs_from_fields(filtered_fields)
        for cube, field in cube_field_pairs:
            cube.attributes['lbproc'] = field.lbproc
            cubes.append(cube)

    iris.util.equalise_attributes(cubes)
    return iris.cube.CubeList(cubes).merge_cube()

def read_precip_correctly(files: List[str], var_name: str = 'precipitation_amount') -> iris.cube.Cube:
    """Load and process precipitation data, handling 3-hourly to daily mean conversion."""
    files.sort()
    cubes = []
    for data_file in files:
        cube = iris.load_cube(data_file, var_name)
        if len(cube.shape) == 3:
            cube = cube.collapsed('time', iris.analysis.MEAN)
        
        for coord_name in ['forecast_period', 'time']:
            if cube.coord(coord_name).bounds is None:
                p = cube.coord(coord_name).points[0]
                cube.coord(coord_name).bounds = [[p - 1.0, p + 1.0]]
        
        # Remove coordinates that cause merge conflicts
        for coord in ['forecast_reference_time', 'realization', 'time']:
            if cube.coords(coord):
                cube.remove_coord(coord)
                
        cubes.append(cube)

    for i, cube in enumerate(cubes):
        cube.cell_methods = cubes[0].cell_methods
        if cube.coords("forecast_period"):
            new_fp = iris.coords.DimCoord(
                cube.coord("forecast_period").points,
                standard_name="forecast_period",
                units=cube.coord("forecast_period").units
            )
            cube.replace_coord(new_fp)

    return iris.cube.CubeList(cubes).merge_cube()

def subset_seasia(cube: iris.cube.Cube) -> iris.cube.Cube:
    """Subset a cube to the Southeast Asia region (-10 to 25 lat, 85 to 145 lon)."""
    return cube.intersection(latitude=(-10, 25), longitude=(85, 145))

def create_latlon_grid(latitudes: tuple = (-90.5, 90.5), 
                       longitudes: tuple = (-0.5, 359.5), 
                       spacing: float = 1.0) -> iris.cube.Cube:
    """Create a template cube with a regular latitude-longitude grid.
    
    Used as a target for regridding in EqWaves and other modules.
    """
    import numpy as np
    cs = iris.coord_systems.GeogCS(6371229.0) # Earth radius from UM
    
    # Latitude coordinate
    lo_lat = latitudes[0] + spacing / 2.0
    hi_lat = latitudes[1] - spacing / 2.0
    n_lat = int(round((hi_lat - lo_lat) / spacing)) + 1
    lat_coord = iris.coords.DimCoord(np.linspace(lo_lat, hi_lat, n_lat),
                                    standard_name='latitude', units='degrees',
                                    coord_system=cs)
    lat_coord.guess_bounds()
    
    # Longitude coordinate
    lo_lon = longitudes[0] + spacing / 2.0
    hi_lon = longitudes[1] - spacing / 2.0
    n_lon = int(round((hi_lon - lo_lon) / spacing)) + 1
    lon_coord = iris.coords.DimCoord(np.linspace(lo_lon, hi_lon, n_lon),
                                    standard_name='longitude', units='degrees',
                                    coord_system=cs)
    lon_coord.guess_bounds()
    
    data = np.zeros((n_lat, n_lon))
    cube = iris.cube.Cube(data, dim_coords_and_dims=[(lat_coord, 0), (lon_coord, 1)])
    return cube
