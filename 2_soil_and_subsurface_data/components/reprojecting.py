# Import modules:
import numpy as np
from affine import Affine
from termcolor import colored
from components.timing import timer
from components.configuration import config
from rasterio.warp import reproject, calculate_default_transform, Resampling


# ------------------------------------------------------------------------------------------------
# Reprojection and resampling function:
# ------------------------------------------------------------------------------------------------


# Define a function to reporject and resample the data:
def reproject_data(data, metadata):

    """
    Reproject and resample data to the projected CRS and spatial resolution
    specified in the configuration file.

    Parameters:
    ----------
    data : numpy array
        The data to be reprojection and resampled.
    metadata : dict
        Dictionary containing the metadata of the data.

    Returns:
    -------
    resampled_data : numpy array
        The reprojection and resampled data.
    resampled_metadata : dict
        Dictionary containing the metadata of the reprojection and resampled data.
    """

    # Get the bounds settings:
    bounds = config.bounds
    # Get the CRS settings:
    crs = config.reproject['crs_global']
    # Get the output resolution settings:
    resolution = config.resolution['output']

    # Reproject and resample the data:
    with timer('Reprojecting and resampling data...'):
        # Calculate the bounds
        bounds = (metadata['transform'].c, 
                  metadata['transform'].f, 
                  metadata['transform'].c + metadata['transform'].a * data.shape[1], 
                  metadata['transform'].f + metadata['transform'].e * data.shape[0])
        # Calculate the new dimensions and affine transform for the target CRS and resolution:
        transform, width, height = calculate_default_transform(
            metadata['crs'], crs, 
            data.shape[1], data.shape[0], *bounds, 
            resolution=(resolution, resolution))
        # Create an empty array for the reprojected and resampled data:
        resampled_data = np.empty((height, width), dtype=np.float32)
        # Reproject the data:
        reproject(source=data,
                  destination=resampled_data,
                  src_transform=metadata['transform'],
                  src_crs=metadata['crs'],
                  dst_transform=transform,
                  dst_crs=crs,
                  resampling=Resampling.bilinear)
        # Calculate the new affine transform:
        new_transform = Affine(resolution, 
                               transform[1], 
                               transform[2], 
                               transform[3], 
                               -resolution, 
                               transform[5])
        # Create a new metadata dictionary:
        resampled_metadata = metadata.copy()
        resampled_metadata.update({'height': resampled_data.shape[0], 
                                   'width': resampled_data.shape[1], 
                                   'dtype': resampled_data.dtype, 
                                   'transform': new_transform, 
                                   'nodata': -9999, 
                                   'crs': crs})
        print(colored(' ✔ Done!', 'green'))

    # Return the reprojection and resampled data and its metadata:
    return resampled_data, resampled_metadata