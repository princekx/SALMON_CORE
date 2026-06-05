"""Numerical calculus helpers for Iris cubes.

These helpers provide finite-difference deltas and first derivatives while
preserving coordinate metadata where possible.
"""

import iris
import iris.analysis.maths
import iris.coords
from iris.util import delta


def _construct_delta_coord(coord):
    """Build a coordinate of first differences for a 1D input coordinate.

    Parameters
    ----------
    coord : iris.coords.Coord
        Source coordinate.

    Returns
    -------
    iris.coords.AuxCoord
        Difference coordinate named ``change_in_<coord_name>``.
    """
    if coord.ndim != 1:
        raise ValueError("Coordinate must be 1D")

    circular = getattr(coord, "circular", False)
    if coord.shape == (1,) and not circular:
        raise ValueError("Cannot take interval differences of a single valued coordinate.")

    circular_kwd = coord.units.modulus or True if circular else False
    bounds = delta(coord.bounds, 0, circular=circular_kwd) if coord.bounds is not None else None
    points = delta(coord.points, 0, circular=circular_kwd)

    new_coord = iris.coords.AuxCoord.from_coord(coord).copy(points, bounds)
    new_coord.rename(f"change_in_{new_coord.name()}")
    return new_coord


def _construct_midpoint_coord(coord, circular=None):
    """Create midpoint coordinate values aligned with differenced data.

    Parameters
    ----------
    coord : iris.coords.Coord
        Source coordinate.
    circular : bool, optional
        Override circular behavior. Defaults to ``coord.circular`` when present.

    Returns
    -------
    iris.coords.Coord
        Midpoint coordinate, preserving original coordinate type when possible.
    """
    if coord.ndim != 1:
        raise ValueError("Coordinate must be 1D")

    if circular is None:
        circular = getattr(coord, "circular", False)

    delta_coord = _construct_delta_coord(coord)
    coord_slice = slice(0, None if circular else -1)

    mid_bounds = None
    if coord.bounds is not None:
        mid_bounds = delta_coord.bounds * 0.5 + coord.bounds[coord_slice, :]

    mid_points = delta_coord.points * 0.5 + coord.points[coord_slice]

    try:
        return type(coord).from_coord(coord).copy(mid_points, mid_bounds)
    except ValueError:
        return iris.coords.AuxCoord.from_coord(coord).copy(mid_points, mid_bounds)


def cube_delta(cube, coord):
    """Compute first differences of a cube along a coordinate.

    Parameters
    ----------
    cube : iris.cube.Cube
        Input cube.
    coord : str or iris.coords.Coord
        Coordinate name or coordinate object for differencing.

    Returns
    -------
    iris.cube.Cube
        Differenced cube with midpoint coordinates on the differenced axis.
    """
    if isinstance(coord, str):
        coord = cube.coord(coord)

    coord_dims = cube.coord_dims(coord.name())
    if not coord_dims:
        raise ValueError(f"Coord {coord.name()} is not a dimension of the cube")

    axis = coord_dims[0]
    circular = getattr(coord, "circular", False)
    delta_data = delta(cube.data, axis, circular=circular)

    if circular:
        result = cube.copy(data=delta_data)
    else:
        indexer = [slice(None)] * cube.ndim
        indexer[axis] = slice(None, -1)
        result = cube[tuple(indexer)]
        result.data = delta_data

    for axis_coord in cube.coords(dimensions=axis):
        result.replace_coord(_construct_midpoint_coord(axis_coord, circular=circular))

    result.rename(f"change_in_{result.name()}_wrt_{coord.name()}")
    return result


def differentiate(cube, coord_to_differentiate):
    """Compute first derivative of a cube with respect to a coordinate.

    Parameters
    ----------
    cube : iris.cube.Cube
        Input cube.
    coord_to_differentiate : str or iris.coords.Coord
        Coordinate name or coordinate object defining derivative axis.

    Returns
    -------
    iris.cube.Cube
        Derivative cube named ``derivative_of_<cube>_wrt_<coord>``.
    """
    coord = cube.coord(coord_to_differentiate) if isinstance(coord_to_differentiate, str) else coord_to_differentiate

    delta_cube = cube_delta(cube, coord)
    delta_coord = _construct_delta_coord(coord)
    axis = cube.coord_dims(coord.name())[0]

    derivative = iris.analysis.maths.divide(delta_cube, delta_coord, axis)
    derivative.rename(f"derivative_of_{cube.name()}_wrt_{coord.name()}")
    return derivative
