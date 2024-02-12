# Import modules:
import rasterio
import numpy as np
from affine import Affine
from termcolor import colored
from rasterio.mask import mask
from shapely.geometry import box
from components.timing import timer
from components.configuration import config


# ------------------------------------------------------------------------------------------------
# Cropping function:
# ------------------------------------------------------------------------------------------------


# Define a function to crop the data:
def crop_data(data_path, parameter):

    """
    Crop the raster to the extent of the research area.

    Parameters:
    ----------
    data_path : str
        Full path to the data file.
    parameter : str
        The parameter to crop.

    Returns:
    -------
    out_image : numpy array
        The cropped data.
    out_meta : dict
        The metadata of the cropped data.
    """

    # Define the bounds:
    bounds = config.bounds
    
    # Create a bounding box from the given bounds:
    bbox = box(bounds['min_lon'], 
               bounds['min_lat'], 
               bounds['max_lon'], 
               bounds['max_lat'])
    # Crop the data:
    with timer('Cropping the data...'):
        # Read the raster file:
        with rasterio.open(data_path) as src:
            # Crop raster using bounding box:
            out_image, _ = mask(src, [bbox], crop=True)
            # Squeeze the data to 2D if it's 3D:
            if out_image.ndim == 3:
                out_image = np.squeeze(out_image, axis=0)
            # Convert integer to float:
            out_image = out_image.astype(np.float32)
            # Replace no data with np.nan:
            if parameter == 'SOIL-DEPTH':
                out_image[out_image == -32768] = np.nan
            else:
                out_image[out_image == -9999] = np.nan
            # Divide by 10000 to convert to correct values:
            out_image /= 10000
            # Manual construction of the transform matrix:
            min_lon, max_lon = bounds['min_lon'], bounds['max_lon']
            min_lat, max_lat = bounds['min_lat'], bounds['max_lat']
            res_lon = (max_lon - min_lon) / out_image.shape[1]
            res_lat = (min_lat - max_lat) / out_image.shape[0]
            transform = Affine.translation(min_lon, max_lat) * Affine.scale(res_lon, res_lat)
            # Update metadata:
            out_meta = src.meta.copy()
            out_meta.update({'height': out_image.shape[0], 
                             'width': out_image.shape[1], 
                             'transform': transform, 
                             'dtype': out_image.dtype, 
                             'nodata': np.nan, 
                             'crs': src.crs})
        print(colored(' ✔ Done!', 'green'))

    # Return the cropped raster and the new metadata:
    return out_image, out_meta