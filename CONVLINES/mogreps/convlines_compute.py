import configparser
import datetime
import multiprocessing
import concurrent
import os
import subprocess
import sys
import uuid
import iris
from iris.fileformats.pp import load_pairs_from_fields
import numpy as np
import logging
import warnings
from tqdm import tqdm
# Set the global warning filter to ignore all warnings
warnings.simplefilter("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def findConvLines(u, v):
    import numpy as np
    import math
    import functools
    from skimage import measure
    from skimage.morphology import thin, disk, closing, remove_small_objects
    import scipy.ndimage as ndimage

    """
    Modified
    --------
    ChatGPT 21 May 2025

    Identify convergence lines in a 2D wind field using Weller et al. (2017) method.
    See detailed docstring in the original version for methodology.

    This function computes convergence lines from gridded 2D wind fields (u, v components)
    using a multi-step geometric and statistical approach:

    Steps:
    1. Compute a convergence field from the input wind components.
    2. Apply a neighborhood smoothing filter to the convergence field.
    3. Threshold the convergence field to retain only positive values (converging flow).
    4. Label and isolate connected regions ("objects") of convergence.
    5. For each object:
       a. Analyze each pixel using an 11×11 surrounding window.
       b. Compute the point of inertia and derive the principal axes via eigenvectors.
       c. Evaluate the profile along the minor eigenvector axis.
       d. Fit a quadratic (binomial) curve to the three points along this axis.
       e. Determine if the peak of the binomial lies near the center (±0.5).
       f. If so, classify the point as part of a convergence line.
    6. Compile all such points into a convergence line map.
    7. Clean up the result by removing very small objects (less than 2 pixels in size).

    Parameters
    ----------
    u : 2D array (float)
        Zonal wind component (m/s). Must be gridded and match the shape of `v`.
    v : 2D array (float)
        Meridional wind component (m/s). Must be gridded and match the shape of `u`.

    Returns
    -------
    ConvOrig : 2D array
        The original convergence field calculated from `u` and `v`.
    OrigConvLines : 2D binary array
        A mask (1s and 0s) identifying raw convergence line pixels.
    CleanConvLines : 2D binary array
        The cleaned and connected convergence lines after removing small objects.

    Notes
    -----
    - Input data must be 2D (i.e., instantaneous horizontal wind field).
    - Recommended input: 10m wind data, though any vertical level can be used.
    - Function assumes input wind data are on the same horizontal grid.
    - Compatible with Python ≥3.6.

    References
    ----------
    Weller, H., et al. (2017). "Identification of convergence lines in wind fields."
    [Specific citation details needed]

    Examples
    --------
    >>> ConvOrig, OrigConvLines, CleanConvLines = findConvLines(u, v)

    Author
    ------
    Caroline Bain (caroline.bain@metoffice.gov.uk), 15 Jan 2019

    Modified
    ---------
    Prince Xavier (prince.xavier@metoffice.gov.uk), 29 May 2025
    """

    # === Parameters ===
    conv_min = 0.5
    search_box = 11  # MUST be odd
    deltas = 1
    s_floor = search_box // 2
    s_ceil = -(-search_box // 2)  # Ceiling division

    if u.ndim != 2 or v.ndim != 2:
        print("Input arrays must be 2D")
        return

    # === Step 1: Convergence computation ===
    conv_orig = -1 * (np.gradient(u, axis=1) + np.gradient(v, axis=0))

    # === Step 2: Smoothing ===
    padded = np.pad(conv_orig, ((1, 1), (1, 1)), mode='constant')
    conv_sm = (
        4 * conv_orig +
        padded[2:, 1:-1] + padded[:-2, 1:-1] +
        padded[1:-1, 2:] + padded[1:-1, :-2]
    ) / 8

    # === Step 3: Thresholding ===
    binary_conv = (conv_sm > conv_min).astype(float)
    binary_conv[:s_floor, :] = 0
    binary_conv[:, :s_floor] = 0
    binary_conv[-s_floor:, :] = 0
    binary_conv[:, -s_floor:] = 0
    convergence_lines = np.zeros_like(binary_conv)

    # === Step 4: Object labeling ===
    labels = measure.label(binary_conv)
    num_labels = labels.max()

    x = np.arange(search_box)
    dist = np.abs(x - s_floor)
    Xdist, Ydist = np.meshgrid(dist, dist)

    for label in range(1, num_labels + 1):
        mask = labels == label
        this_blob = conv_sm * mask
        rows, cols = np.nonzero(mask)

        for r, c in zip(rows, cols):
            mm = this_blob[r - s_floor:r + s_ceil, c - s_floor:c + s_ceil]
            if mm.shape != (search_box, search_box):
                continue  # skip borders

            Rx = np.sum(Xdist * mm)
            Ry = np.sum(Ydist * mm)
            a = np.sum(mm * (Xdist - Rx) ** 2)
            b = np.sum(mm * (Ydist - Ry) ** 2)
            c_term = np.sum(mm * (Xdist - Rx) * (Ydist - Ry))

            omega = 0.5 * math.sqrt((a - b) ** 2 + 4 * c_term ** 2)
            eigval1 = 0.5 * (a + b) + omega
            eigval2 = 0.5 * (a + b) - omega

            if abs(1 - abs(eigval2 / eigval1)) < 0.2:
                continue  # too circular

            f_mid = this_blob[r, c]
            dx = dy = 0
            f_pos = f_neg = 0

            if omega == abs(0.5 * (a - b)):
                # aligned with axis
                if eigval1 <= eigval2:
                    f_pos = conv_sm[r + 1, c]
                    f_neg = conv_sm[r - 1, c]
                else:
                    f_pos = conv_sm[r, c + 1]
                    f_neg = conv_sm[r, c - 1]
            else:
                alpha = 0.5 * math.atan2(2 * c_term, a - b)
                dx, dy = math.sin(alpha) * deltas, math.cos(alpha) * deltas

                # Bilinear interpolation here (you could wrap this in a helper function)
                def bilinear_interp(rr, cc):
                    i, j = int(rr), int(cc)
                    di, dj = rr - i, cc - j
                    return (
                        conv_sm[i, j] * (1 - di) * (1 - dj) +
                        conv_sm[i + 1, j] * di * (1 - dj) +
                        conv_sm[i, j + 1] * (1 - di) * dj +
                        conv_sm[i + 1, j + 1] * di * dj
                    )

                f_pos = bilinear_interp(r + dy, c + dx)
                f_neg = bilinear_interp(r - dy, c - dx)

            Ggrad = (f_pos - 2 * f_mid + f_neg) / (2 * deltas ** 2)
            Gtran = (f_pos - f_neg) / (2 * deltas)
            smax = -Gtran / (2 * Ggrad)

            if abs(smax) < 0.5 * math.sqrt(2):
                convergence_lines[r, c] = 1

    # === Final image cleanup (Steps 7+) ===
    lines_thin = thin(convergence_lines)
    joined = closing(lines_thin, disk(2)) + convergence_lines
    joined[joined > 1] = 1
    cleaned = remove_small_objects(thin(joined).astype(bool), 3).astype(int)

    return conv_orig, convergence_lines, cleaned


class MOGProcess:
    def __init__(self, config_values):
        self.config_values = config_values
        # Navigate to the parent directory
        self.parent_dir = '/home/users/prince.xavier/MJO/SALMON/CONVLINES'


    def get_all_members(self, hr):
        if hr == 12:
            return [str('%02d' % mem) for mem in range(18)]
        elif hr == 18:
            return [str('%02d' % mem) for mem in range(18, 35)] + ['00']
        
    def get_file_name(self, date, varname):
        concated_dir = os.path.join(self.config_values[f'mogreps_cs_processed_dir'], varname)
        file_name = os.path.join(concated_dir,
                                        f'{varname}_ColdSurge_24h_allMember_{date.strftime("%Y%m%d")}.nc')
        return file_name

    def lonlat_to_webmercator(self, lon, lat):
        """Convert longitude/latitude to Web Mercator coordinates (meters)."""
        
        k = 6378137.0  # Earth radius in meters
        x = lon * (k * np.pi / 180.0)
        y = np.log(np.tan((90 + lat) * np.pi / 360.0)) * k
        return x, y

    def compute_conv_lines(self, date, plot_width=500):
        """
        Compute ensemble mean convergence lines from MOGREPS wind data.
        This method processes U850 and V850 wind components to identify convergence lines
        for each ensemble member and forecast time, then computes the ensemble mean.
        Parameters
        ----------
        date : datetime.datetime
            The forecast initialization date for which to compute convergence lines.
        plot_width : int, optional
            Width parameter for plotting (default: 500). Currently not used in the implementation.
        Returns
        -------
        None
            The method saves the computed convergence lines to a NetCDF file but does not return a value.
        Notes
        -----
        - Processes 36 ensemble members (000-035)
        - Computes convergence lines for forecast times from 0 to 174 hours at 24-hour intervals
        - Restricts analysis to longitude range 90-135°E
        - Output is saved as NetCDF file with naming convention:
          'convergence_lines_24h_mean_YYYYMMDD.nc'
        - If output file already exists, computation is skipped
        - Uses the findConvLines function to detect convergence lines from wind components
        - Ensemble mean is computed across all members for each forecast time
        Side Effects
        ------------
        - Creates output directory if it doesn't exist
        - Writes NetCDF file to disk
        - Logs information messages about processing status
        """

        date_label = date.strftime("%Y%m%d")
        members = [str('%03d' % mem) for mem in range(36)]


        output_dir = os.path.join(self.config_values[f'mogreps_convlines_processed_dir'], 'convergence_lines')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'convergence_lines_24h_mean_{date_label}.nc')

        if not os.path.exists(output_file):
            logger.info(f'Computing convergence lines for {date_label}')
            u850_file_name = self.get_file_name(date, 'u850')
            v850_file_name = self.get_file_name(date, 'v850')

            
            fc_times = [str('%03d' % fct) for fct in np.arange(0, 174, 24)]
            time_coord = iris.coords.DimCoord(points=[int(t) for t in fc_times], standard_name='time', 
                                            units=f'hours since {date.strftime("%Y-%m-%d")}')
            
            u850_cube = iris.load_cube(u850_file_name)
            u850_cube = u850_cube.intersection(longitude=(90, 135))

            v850_cube = iris.load_cube(v850_file_name)
            v850_cube = v850_cube.intersection(longitude=(90, 135))
            
            ntime = len(u850_cube.coords('forecast_period')[0].points)
            
            convergence_lines_cube = u850_cube[0].copy()


            for t in range(ntime):
                convergence_lines_list = []
                cleaned_list = []
                for mem in tqdm(range(len(members)), desc=f'Lead time {t} Processing members'):
                    conv_orig, convergence_lines, cleaned = findConvLines(u850_cube.data[mem, t], v850_cube.data[mem, t])
                    convergence_lines_list.append(convergence_lines)
                    convergence_lines_cube.data[t] = np.mean(np.array(convergence_lines_list), axis=0)
            print(convergence_lines_cube)

            iris.save(convergence_lines_cube, output_file)
            logger.info(f'Saved convergence lines to {output_file}')
        else:
            logger.info(f'Convergence lines file already exists: {output_file}')

if __name__ == "__main__":
    # Load configuration
    config = configparser.ConfigParser()
    config.read('/home/users/prince.xavier/MJO/SALMON/CONVLINES/config/mogreps_config.ini')
    section = 'convlines'
    model = 'mogreps'

    config_values = {}
    for section in config.sections():
        for key, value in config.items(section):
            config_values[f'{section}_{key}'] = value

    mog_process = MOGProcess(config_values)

    # Example date for processing
    example_date = datetime.datetime(2025, 12, 1)
    mog_process.compute_conv_lines(example_date)
