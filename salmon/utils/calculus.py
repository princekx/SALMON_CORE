import iris
import numpy as np
import iris.analysis.maths
import iris.coords
from iris.util import delta

def _construct_delta_coord(coord):
    if coord.ndim != 1:
        raise ValueError("Coordinate must be 1D")
    
    circular = getattr(coord, "circular", False)
    if coord.shape == (1,) and not circular:
        raise ValueError("Cannot take interval differences of a single valued coordinate.")

    if circular:
        circular_kwd = coord.units.modulus or True
    else:
        circular_kwd = False

    if coord.bounds is not None:
        bounds = delta(coord.bounds, 0, circular=circular_kwd)
    else:
        bounds = None

    points = delta(coord.points, 0, circular=circular_kwd)
    new_coord = iris.coords.AuxCoord.from_coord(coord).copy(points, bounds)
    new_coord.rename("change_in_%s" % new_coord.name())

    return new_coord

def _construct_midpoint_coord(coord, circular=None):
    if circular is None:
        circular = getattr(coord, "circular", False)

    if coord.ndim != 1:
        raise ValueError("Coordinate must be 1D")

    mid_point_coord = _construct_delta_coord(coord)
    circular_slice = slice(0, -1 if not circular else None)

    if coord.bounds is not None:
        axis_delta = mid_point_coord.bounds
        mid_point_bounds = axis_delta * 0.5 + coord.bounds[circular_slice, :]
    else:
        mid_point_bounds = None

    axis_delta = mid_point_coord.points
    mid_point_points = axis_delta * 0.5 + coord.points[circular_slice]

    try:
        mid_point_coord = coord.from_coord(coord).copy(mid_point_points, mid_point_bounds)
    except ValueError:
        mid_point_coord = iris.coords.AuxCoord.from_coord(coord).copy(mid_point_points, mid_point_bounds)

    return mid_point_coord

def cube_delta(cube, coord):
    if isinstance(coord, str):
        coord = cube.coord(coord)

    delta_dims = cube.coord_dims(coord.name())
    if not delta_dims:
        raise ValueError(f"Coord {coord.name()} is not a dimension of the cube")
    
    delta_dim = delta_dims[0]
    circular = getattr(coord, "circular", False)
    
    delta_cube_data = delta(cube.data, delta_dim, circular=circular)

    if circular:
        delta_cube = cube.copy(data=delta_cube_data)
    else:
        subset_slice = [slice(None, None)] * cube.ndim
        subset_slice[delta_dim] = slice(None, -1)
        delta_cube = cube[tuple(subset_slice)]
        delta_cube.data = delta_cube_data

    for cube_coord in cube.coords(dimensions=delta_dim):
        delta_cube.replace_coord(_construct_midpoint_coord(cube_coord, circular=circular))

    delta_cube.rename("change_in_{}_wrt_{}".format(delta_cube.name(), coord.name()))
    return delta_cube

def differentiate(cube, coord_to_differentiate):
    """Calculate the differential of a cube with respect to a coordinate."""
    delta_cube = cube_delta(cube, coord_to_differentiate)

    if isinstance(coord_to_differentiate, str):
        coord = cube.coord(coord_to_differentiate)
    else:
        coord = coord_to_differentiate

    delta_coord = _construct_delta_coord(coord)
    delta_dim = cube.coord_dims(coord.name())[0]

    delta_cube = iris.analysis.maths.divide(delta_cube, delta_coord, delta_dim)
    delta_cube.rename("derivative_of_{}_wrt_{}".format(cube.name(), coord.name()))
    return delta_cube
