import math
import numpy as np
from bokeh.palettes import TolRainbow12


class Vector:
    """
    Compute polygon coordinates and colours for wind vector arrows.

    Supports two arrow styles:

    - ``"kite"``   – simple diamond/kite shape.
    - ``"barbed"`` – barbed arrow with a notched tail.

    Parameters
    ----------
    u : iris.cube.Cube
        Zonal (east–west) wind component on a lat/lon grid.
    v : iris.cube.Cube
        Meridional (north–south) wind component on a lat/lon grid.
    xSkip : int, optional
        Longitude thinning stride. Default ``2``.
    ySkip : int, optional
        Latitude thinning stride. Default ``2``.
    maxSpeed : float, optional
        Speed used to normalise arrow lengths. Default ``20.0``.
    arrowHeadAngle : float, optional
        Half-angle of the arrow head in degrees. Default ``25.0``.
    arrowHeadScale : float, optional
        Arrow head size relative to arrow length. Default ``1.0``.
    arrowType : str, optional
        ``"kite"`` or ``"barbed"``. Default ``"barbed"``.
    palette : list[str], optional
        Bokeh colour palette. Default :data:`TolRainbow12`.
    palette_reverse : bool, optional
        Reverse the palette if ``True``. Default ``False``.

    Attributes
    ----------
    xs : list[list[float]]
        Polygon x-coordinates (one sub-list per arrow).
    ys : list[list[float]]
        Polygon y-coordinates (one sub-list per arrow).
    colors : numpy.ndarray
        Colour string for each arrow, drawn from *palette*.

    Examples
    --------
    >>> vec = Vector(u_cube, v_cube, xSkip=5, ySkip=5, maxSpeed=10)
    >>> source = ColumnDataSource(dict(xs=vec.xs, ys=vec.ys, colors=vec.colors))
    >>> plot.patches("xs", "ys", fill_color="colors", line_color="colors", source=source)
    """

    def __init__(self, u, v, **kwargs):
        xSkip = kwargs.get("xSkip", 2)
        ySkip = kwargs.get("ySkip", 2)
        maxSpeed = kwargs.get("maxSpeed", 20.0)
        arrowHeadAngle = kwargs.get("arrowHeadAngle", 25.0)
        arrowHeadScale = kwargs.get("arrowHeadScale", 1.0)
        arrowType = kwargs.get("arrowType", "barbed").lower()
        palette = kwargs.get("palette", TolRainbow12)
        palette_reverse = kwargs.get("palette_reverse", False)

        if palette_reverse:
            palette = palette[::-1]

        # ── thin the fields ──────────────────────────────────────────────────
        u = u[::ySkip, ::xSkip]
        v = v[::ySkip, ::xSkip]

        lons = u.coord("longitude").points
        lats = u.coord("latitude").points
        U = u.data
        V = v.data

        # ── grid + derived quantities ─────────────────────────────────────────
        X, Y = np.meshgrid(lons, lats)
        speed = np.hypot(U, V)
        theta = np.arctan2(V, U)         # standard mathematical angle

        x0 = X.flatten()
        y0 = Y.flatten()
        angle = theta.flatten()
        length = speed.flatten() / maxSpeed
        x1 = x0 + length * np.cos(angle)
        y1 = y0 + length * np.sin(angle)

        # ── per-arrow colour ──────────────────────────────────────────────────
        cm = np.asarray(palette)
        indices = np.interp(length, (length.min(), length.max()), (0, len(cm) - 1)).astype(int)
        self.colors = cm[indices]

        # ── arrow head geometry ───────────────────────────────────────────────
        head_rad = math.radians(arrowHeadAngle)

        xR = x1 - arrowHeadScale * length * np.cos(angle + head_rad)
        yR = y1 - arrowHeadScale * length * np.sin(angle + head_rad)
        xL = x1 - arrowHeadScale * length * np.cos(angle - head_rad)
        yL = y1 - arrowHeadScale * length * np.sin(angle - head_rad)

        if arrowType == "kite":
            self.xs, self.ys = self._kite_polygons(x0, y0, x1, y1, xR, yR, xL, yL)
        else:
            # barbed is the default
            half_rad = head_rad * 0.5
            half_scale = arrowHeadScale * 0.5

            xR1 = x1 - half_scale * length * np.cos(angle + half_rad)
            yR1 = y1 - half_scale * length * np.sin(angle + half_rad)
            xL1 = x1 - half_scale * length * np.cos(angle - half_rad)
            yL1 = y1 - half_scale * length * np.sin(angle - half_rad)

            self.xs, self.ys = self._barbed_polygons(x0, y0, x1, y1, xR, yR, xL, yL, xR1, yR1, xL1, yL1)

    # ── polygon builders ──────────────────────────────────────────────────────

    @staticmethod
    def _kite_polygons(x0, y0, x1, y1, xR, yR, xL, yL):
        """Return closed kite-shaped polygon lists."""
        xs = [[x1[i], xR[i], x0[i], xL[i], x1[i]] for i in range(len(x0))]
        ys = [[y1[i], yR[i], y0[i], yL[i], y1[i]] for i in range(len(y0))]
        return xs, ys

    @staticmethod
    def _barbed_polygons(x0, y0, x1, y1, xR, yR, xL, yL, xR1, yR1, xL1, yL1):
        """Return closed barbed-arrow polygon lists."""
        xs = [[x1[i], xR[i], xR1[i], x0[i], xL1[i], xL[i], x1[i]] for i in range(len(x0))]
        ys = [[y1[i], yR[i], yR1[i], y0[i], yL1[i], yL[i], y1[i]] for i in range(len(y0))]
        return xs, ys


# ── lowercase alias kept for backward compatibility ───────────────────────────
vector = Vector