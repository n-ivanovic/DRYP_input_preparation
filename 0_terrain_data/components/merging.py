# Import modules:
import rasterio
import numpy as np
from affine import Affine
from termcolor import colored
from rasterio.merge import merge
from components.timing import timer
from components.plotting import plot_data
from components.configuration import config


# ------------------------------------------------------------------------------------------------
# Merging functions:
# ------------------------------------------------------------------------------------------------


# Define a generic function to merge raster tiles:
def generic_merge(tiles, labels, temp_path):
    
    """
    Merge raster tiles into a single raster file.

    Parameters:
    ----------
    tiles : list
        A list of all raster tiles specific to the region of interest.
    labels : dict
        Dictionary containing the labels for the merged data.
    temp_path : str
        The path to the temporary directory.
    
    Returns:
    -------
    mosaic : numpy array
        The merged data.
    metadata : dict
        Dictionary containing the metadata of the merged data.
    """
    
    # Get the bounds settings:
    bounds = config.bounds
    # Get the intermediate step settings:
    intermediate_step = config.intermediate_step['plot']

    # List to store the tiles:
    tiles_to_mosaic = []
    # Read the tiles:
    for t in tiles:
        tile = rasterio.open(t)
        tiles_to_mosaic.append(tile)
    # Merge the tiles:
    with timer(f"Merging the {labels['data_name']} tiles..."):
        mosaic, _ = merge(tiles_to_mosaic)
        # Remove the extra dimension:
        mosaic = mosaic[0]
        # Replace the NoData values with 0:
        mosaic = np.where(mosaic == -9999, 0, mosaic)
        # Manual construction of the Affine transform matrix:
        min_lon, max_lon = bounds['min_lon']+1, bounds['max_lon']+1     # +1 to get back to the original coordinates
        min_lat, max_lat = bounds['min_lat'], bounds['max_lat']
        res_lon = (max_lon - min_lon) / mosaic.shape[1]
        res_lat = (min_lat - max_lat) / mosaic.shape[0]
        transform = Affine.translation(min_lon, max_lat) * Affine.scale(res_lon, res_lat)
        # Update the metadata:
        metadata = tiles_to_mosaic[0].meta.copy()
        metadata.update({'crs': tiles_to_mosaic[0].crs, 
                         'height': mosaic.shape[0], 
                         'width': mosaic.shape[1], 
                         'transform': transform, 
                         'dtype': mosaic.dtype, 
                         'nodata': -9999})
        print(colored(' ✔ Done!', 'green'))
    # Plot the merged data:
    if intermediate_step:
        plot_data(mosaic, metadata, temp_path, 
                  data_name=labels['data_name'], title=labels['title'], 
                  cbar_label=labels['cbar_label'], cmap=labels['cmap'], 
                  wgs=labels['wgs'], log_scale=labels['log_scale'])
    # Close the tiles:
    for tile in tiles_to_mosaic:
        tile.close()
    print(colored('==========================================================================================', 'blue'))
    
    # Return the merged data and its metadata:
    return mosaic, metadata


# Define a function to merge the DEM and upstream tiles:
def merge_data(dem_tiles, ups_tiles, temp_path):

    """
    Merge the DEM and upstream area tiles into a single raster file, by 
    using the generic merge function.

    Parameters:
    ----------
    dem_tiles : list
        A list of all DEM tiles specific to the region of interest.
    ups_tiles : list
        A list of all upstream tiles specific to the region of interest.
    temp_path : str
        The path to the temporary directory.
    
    Returns:
    -------
    merged_dem : numpy array
        The merged DEM data.
    merged_ups : numpy array
        The merged upstream.
    metadata_dem : dict
        Dictionary containing the metadata of the merged DEM.
    """

    # Define the data labels:
    labels_dem = {
        'data_name': 'merged_dem',
        'title': 'Merged DEM',
        'cbar_label': 'Elevation [m]',
        'cmap': 'terrain',
        'wgs': True, 
        'log_scale': False}
    labels_ups = {
        'data_name': 'merged_ups',
        'title': 'Merged upstream area',
        'cbar_label': 'Upstream cells',
        'cmap': 'cubehelix',
        'wgs': True, 
        'log_scale': True}
    # Merge the DEM and upstream tiles:
    merged_dem, metadata_dem = generic_merge(dem_tiles, labels_dem, temp_path)
    merged_ups, _ = generic_merge(ups_tiles, labels_ups, temp_path)
    
    # Return the merged DEM and upstream data and their metadata:
    return merged_dem, merged_ups, metadata_dem