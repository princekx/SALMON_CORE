import numpy as np
import scipy.ndimage as ndimage
from skimage import measure
import pandas as pd
import datetime

def generate_mask(data, threshold, method='geq'):
    if method == 'geq':
        return data >= threshold
    elif method == 'gt':
        return data > threshold
    elif method == 'leq':
        return data <= threshold
    elif method == 'lt':
        return data < threshold
    return data >= threshold

def extract_grid_features(cube, thresholds=None, time_index=0, threshold_method='geq'):
    """
    Extract properties of objects in a 2D field based on thresholds.
    """
    if thresholds is None:
        thresholds = [1.0, 5.0, 10.0, 20.0, 50.0]

    lons = cube.coord('longitude').points
    lats = cube.coord('latitude').points
    
    # Cube dates and times
    if cube.coords('time'):
        c_date = cube.coord('time').units.num2date(cube.coord('time').points)[0]
        cube_date = datetime.datetime(c_date.year, c_date.month, c_date.day)
    else:
        cube_date = None

    if cube.coords('forecast_period'):
        forecast_p = cube.coord('forecast_period').points[0]
    else:
        forecast_p = np.nan

    all_features = []

    for threshold in thresholds:
        cube_data = cube.data.copy()
        mask = generate_mask(cube_data, threshold, threshold_method)
        
        # Label connected regions
        labeled_array, num_features = ndimage.label(mask)
        
        for feature_num in range(1, num_features + 1):
            loc_mask = labeled_array == feature_num
            
            # Region properties using skimage
            regions = measure.regionprops(labeled_array == feature_num, cube_data)
            if not regions: continue
            region = regions[0]
            
            # Extract basic properties
            area = region.area
            perimeter = region.perimeter
            ecc = region.eccentricity
            orient = region.orientation
            
            # Centroid in lon/lat
            y_cen, x_cen = region.centroid
            centroid_lon = lons[int(round(x_cen))] if 0 <= round(x_cen) < len(lons) else np.nan
            centroid_lat = lats[int(round(y_cen))] if 0 <= round(y_cen) < len(lats) else np.nan
            
            # Polygon vertices (simplified representation as list of coords)
            # Original code stored all pixels, but polygons might be better? 
            # We'll stick to the original "Polygon" as list of coords for consistency if needed.
            # But here we just extract coords of pixels in the object.
            rows, cols = np.where(loc_mask)
            poly_coords = list(zip(lons[cols], lats[rows]))
            
            feature_data = {
                'TimeInds': time_index,
                'Date': cube_date,
                'Forecast_period': forecast_p,
                'Threshold': threshold,
                'ObjectLabel': f"{time_index}_{threshold}_{feature_num}",
                'Area': area,
                'Perimeter': perimeter,
                'GridPoints': len(rows),
                'Eccentricity': ecc,
                'Orientation': orient,
                'Mean': region.mean_intensity,
                'Max': region.max_intensity,
                'Min': region.min_intensity,
                'Centroid': (centroid_lon, centroid_lat),
                'Polygon': poly_coords
            }
            all_features.append(feature_data)

    if not all_features:
        return pd.DataFrame()
        
    return pd.DataFrame(all_features)
