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

def read_winds_correctly(files: List[str], var_name: str, pressure: Optional[int] = None) -> iris.cube.Cube:
    """Load and process wind data, handling multiple members and pressure levels.

    Args:
        files (List[str]): List of NetCDF or PP files to load.
        var_name (str): Variable name in the files.
        pressure (int, optional): Pressure level to extract. Defaults to None.

    Returns:
        iris.cube.Cube: The processed wind cube.
    """
    cubes = iris.load(files, var_name)
    if pressure:
        cubes = cubes.extract(iris.Constraint(pressure=pressure))
    
    iris.util.equalise_attributes(cubes)
    return cubes.merge_cube()

def read_precip_correctly(files: List[str]) -> iris.cube.Cube:
    """Load and process precipitation data.

    Args:
        files (List[str]): List of files to load.

    Returns:
        iris.cube.Cube: The processed precipitation cube.
    """
    cubes = iris.load(files, 'precipitation_amount')
    iris.util.equalise_attributes(cubes)
    return cubes.merge_cube()
