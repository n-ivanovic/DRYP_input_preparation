# Import modules:
import numpy as np
from affine import Affine
from termcolor import colored
from components.timing import timer
from components.saving import save_data
from components.plotting import plot_data
from components.configuration import config
from rasterio.warp import reproject, calculate_default_transform, Resampling


# ------------------------------------------------------------------------------------------------
# Reprojection and resampling functions:
# ------------------------------------------------------------------------------------------------


# Define a generic function to reproject and resample data:
def generic_reproject(data, metadata, labels, temp_path, output_path):

    """
    Reproject and resample data to the projected CRS and spatial resolution
    specified in the configuration file.

    Parameters:
    ----------
    data : numpy array
        The data to be reprojection and resampled.
    metadata : dict
        Dictionary containing the metadata of the data.
    labels : dict
        A dictionary containing labels for plotting and saving.
    temp_path : str
        The path to the temporary directory.
    output_path : str
        The path to the output directory.

    Returns:
    -------
    resampled_data : numpy array
        The reprojection and resampled data.
    resampled_metadata : dict
        Dictionary containing the metadata of the reprojection and resampled data.
    """

    # Get the CRS settings:
    crs = config.reproject['crs']
    # Get the resolution settings:
    resolution = config.resolution['output']
    # Get the intermediate step settings:
    intermediate_step = config.intermediate_step['plot']

    # Reproject and resample the data:
    with timer('Reprojecting and resampling data...'):
        # Calculate the bounds
        bounds = (metadata['transform'].c, 
                  metadata['transform'].f, 
                  metadata['transform'].c + metadata['transform'].a * data.shape[1], 
                  metadata['transform'].f + metadata['transform'].e * data.shape[0])
        # Determine the resampling method:
        resample_method = Resampling.max if labels['method'] == 'max' else Resampling.bilinear
        # Calculate the new dimensions and affine transform for the target CRS and resolution:
        transform, width, height = calculate_default_transform(metadata['crs'], crs, 
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
                  resampling=resample_method)
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
    if labels['save']:
        save_data(resampled_data, resampled_metadata, output_path, 
                  data_name=labels['data_name'])
    # Plot the resampled data:
    if intermediate_step:
        plot_data(resampled_data, resampled_metadata, temp_path, 
                  data_name=labels['data_name'], title=labels['title'], 
                  cbar_label=labels['cbar_label'], cmap=labels['cmap'], 
                  log_scale=labels['log_scale'], inverse=labels['inverse'], binary=labels['binary'])
    print(colored('==========================================================================================', 'blue'))

    # Return the reprojection and resampled data and its metadata:
    return resampled_data, resampled_metadata


# Define a function to resample the DEM and upstream data:
def resample_data(merged_dem, merged_upstream, metadata, 
                  temp_path, output_path):

    """
    Resample the merged DEM and UPSTREAM data to the resolution 
    specified in the configuration file.

    Parameters:
    ----------
    merged_dem : numpy array
        The merged DEM data.
    merged_upstream : numpy array
        The merged UPSTREAM data.
    metadata : dict
        Dictionary containing the metadata of the merged data.
    temp_path : str
        Full path to the temporary directory.
    output_path : str
        Full path to the output directory.

    Returns:
    -------
    resampled_merged_dem : numpy array
        The resampled merged DEM data.
    resampled_merged_upstream : numpy array
        The resampled merged UPSTREAM data.
    resampled_merged_dem_metadata : dict
        Dictionary containing the metadata of the resampled data.
    """

    # Define the data labels:
    labels_dem = {
        'method': 'mean',
        'data_name': 'resampled_merged_dem',
        'title': 'Resampled merged DEM',
        'cbar_label': 'Elevation [m]',
        'cmap': 'terrain',
        'log_scale': False,
        'inverse': False,
        'binary': False, 
        'save': True}
    labels_ups = {
        'method': 'max', 
        'data_name': 'resampled_merged_upstream',
        'title': 'Resampled merged UPSTREAM',
        'cbar_label': 'Upstream cells',
        'cmap': 'cubehelix',
        'log_scale': False,
        'inverse': True,
        'binary': False, 
        'save': False}
    # Resample the merged DEM and UPSTREAM data:
    resampled_merged_dem, resampled_merged_dem_metadata = generic_reproject(
        merged_dem, metadata, labels_dem, temp_path, output_path)
    resampled_merged_upstream, _ = generic_reproject(
        merged_upstream, metadata, labels_ups, temp_path, output_path)

    # Return resampled data and metadata:
    return resampled_merged_dem, resampled_merged_upstream, resampled_merged_dem_metadata