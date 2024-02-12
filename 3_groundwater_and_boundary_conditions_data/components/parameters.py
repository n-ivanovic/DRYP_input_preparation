import os
import numpy as np
import pandas as pd
import xarray as xr
from affine import Affine
from termcolor import colored
from scipy.spatial import KDTree
from components.timing import timer
from components.saving import save_data
from components.plotting import plot_data
from components.cropping import crop_data
from components.configuration import config
from components.reprojecting import resample_data, reproject_data


# ------------------------------------------------------------------------------------------------
# GW parameters functions:
# ------------------------------------------------------------------------------------------------


# Define a function to create a WTE geotiff from a .csv file:
def create_wte_tif(input_path, temp_path):

    """
    This function creates a WTE array from a .csv file, and saves it as a .tif file
    in the temporary directory.

    Parameters:
    ----------
    input_path : str
        The path to the input directory.
    temp_path : str
        The path to the temporary directory.

    Returns:
    -------
    None
    """

    # Get the intermediate step for plotting:
    intermediate_step = config.intermediate_step['plot']

    # Generate the global WTE dataset from a .csv file:
    with timer('Generating the global water table elevation dataset...'):
        # Read the .csv file:
        wte_csv = pd.read_csv(os.path.join(input_path, 'water_table_depth.csv'))
        # Create a 2D array from the .csv file:
        wte_array = wte_csv.pivot(index='Y', columns='X', values='WTD(m)').values[::-1].astype('float32')
        # Manually calculate the extent:
        min_x, max_x = wte_csv['X'].min(), wte_csv['X'].max()
        min_y, max_y = wte_csv['Y'].min(), wte_csv['Y'].max()
        res_lon = (max_x - min_x) / wte_array.shape[1]
        res_lat = (min_y - max_y) / wte_array.shape[0]
        # Calculate the transform matrix:
        transform = Affine.translation(min_x, max_y) * Affine.scale(res_lon, res_lat)
        # Create the metadata:
        wte_metadata = {'height': wte_array.shape[0], 
                        'width': wte_array.shape[1], 
                        'dtype': wte_array.dtype, 
                        'transform': transform, 
                        'crs': 'EPSG:4326', 
                        'nodata': np.nan}
        print(colored(' ✔ Done!', 'green'))
    # Resample the data:
    resampled_wte_array, resampled_wte_metadata = resample_data(wte_array, wte_metadata)
    # Save the data:
    save_data(resampled_wte_array, resampled_wte_metadata, temp_path, 
              format= 'GTiff', data_name='wte_global')
    # Plot the data:
    if intermediate_step:
        plot_data(resampled_wte_array, resampled_wte_metadata, temp_path, 
                  data_name='wte_global', title='Water Table Elevation', 
                  cbar_label=r'Water table [m]', cmap='plasma', 
                  wgs=True)
    print(colored('==========================================================================================', 'blue'))


# Define a function to adjust the data range:
def parameter_range(data):

    """
    This function adjusts the data range, and caps the data within the range. The 
    range is determined by the 5th and 95th percentiles of the data, which are 
    calculated and used as the lowerand upper bounds, respectively. The main purpose 
    of this function is to remove outliers, and prevent potential mathematical errors 
    due to division by zero.

    Parameters:
    ----------
    data : numpy array
        The data array.

    Returns:
    -------
    data : numpy array
        The capped data array.
    """
    
    # Calculate the 5th and 95th percentiles of the data:
    lower_bound = np.nanpercentile(data, 5)
    upper_bound = np.nanpercentile(data, 95)
    # Cap the data within the bounds (i.e. remove outliers):
    data[data < lower_bound] = lower_bound
    data[data > upper_bound] = upper_bound
    
    # Return the capped data:
    return data


# Define a function to fill in all NaNs:
def fill_all_nans(data, k=3):

    """
    This function fills in all NaNs in a data array.

    Parameters:
    ----------
    data : numpy array
        The data array.
    k : int
        The number of nearest neighbors to use for filling in the NaNs.

    Returns:
    -------
    filled_array : numpy array
        The filled array.
    """

    # Create a copy of the original array:
    filled_array = data.copy()
    # Find coordinates where data exists (not NaN):
    existing_data_coords = np.column_stack(np.where(~np.isnan(filled_array)))
    # Find coordinates where data is missing (NaN):
    missing_data_coords = np.column_stack(np.where(np.isnan(filled_array)))
    # Build KDTree using existing data coordinates:
    tree = KDTree(existing_data_coords)
    # Query KDTree to find indices of k nearest existing data points for each missing data point:
    _, ind = tree.query(missing_data_coords, k=k)
    # Extract k nearest existing values:
    k_nearest_values = filled_array[tuple(existing_data_coords[ind].reshape(-1, 2).T)].reshape(len(missing_data_coords), k)
    # Fill missing values using the average of k nearest existing values:
    nearest_values = np.nanmean(k_nearest_values, axis=1)
    filled_array[missing_data_coords[:, 0], missing_data_coords[:, 1]] = nearest_values

    # Return the filled array:
    return filled_array


# Define a function to process the data:
def generic_process(labels, temp_path, output_path):

    """
    This function processes the data.

    Parameters:
    ----------
    labels : dict
        A dictionary containing labels of the data.
    temp_path : str
        The path to the temporary directory.
    output_path : str
        The path to the output directory.

    Returns:
    -------
    res_crop_data : xarray DataArray
        The cropped and resampled data.
    res_crop_metadata : dict
        The metadata of the resampled cropped data.
    """

    # Get the intermediate step for plotting:
    intermediate_step = config.intermediate_step['plot']

    # Crop the dataset:
    crop_dataset, crop_metadata = crop_data(labels)
    # Reproject and resample the dataset:
    res_crop_data, res_crop_metadata = reproject_data(crop_dataset, crop_metadata)
    # Correct the data range to remove outliers, and division by zero:
    with timer('Correcting the data range...'):
        res_crop_data = parameter_range(res_crop_data)
        print(colored(' ✔ Done!', 'green'))
    # Fill in the NaNs:
    with timer('Filling in the NaNs...'):
        res_crop_data = fill_all_nans(res_crop_data)
        print(colored(' ✔ Done!', 'green'))
    # Correct the units of the following parameters:
    if labels['data_name'] == 'ksat_aq':
        res_crop_data = res_crop_data * 60 * 60     # for [m/s] to [m/h]
    # Save the resampled data:
    if labels['save']:
        save_data(res_crop_data, res_crop_metadata, output_path, 
                  data_name=labels['data_name'])
    # Plot the resampled data:
    if intermediate_step:
        plot_data(res_crop_data, res_crop_metadata, temp_path,
                  data_name=labels['data_name'], title=labels['title'], 
                  cbar_label=labels['cbar_label'], cmap=labels['cmap'])
    print(colored('==========================================================================================', 'blue'))

    # Return the resampled data and metadata:
    return res_crop_data, res_crop_metadata


# Define a function to process the groundwater data:
def process_data(input_path_ksat_aq, input_path_wte, temp_path, output_path):

    """
    This function processes the groundwater data.

    Parameters:
    ----------
    input_path_ksat_aq : str
        The path to the input directory for the aquifer saturated hydraulic conductivity data.
    input_path_wte : str
        The path to the input directory for the water table elevation data.
    temp_path : str
        The path to the temporary directory.
    output_path : str
        The path to the output directory.

    Returns:
    -------
    res_crop_ksat_aq : numpy array
        The cropped and resampled aquifer saturated hydraulic conductivity data.
    res_crop_wte : numpy array
        The cropped and resampled water table elevation data.
    """

    # Create WTE geotiff from a .csv file:
    create_wte_tif(input_path_wte, temp_path)
    # Define the data labels:
    labels_ksat_aq = {
        'data_path': os.path.join(input_path_ksat_aq, 'GLHYMPS_V2_logK_Ferr_raster_0p025.tif'),
        'data_name': 'ksat_aq',
        'title': 'Aquifer Saturated Hydraulic Conductivity',
        'cbar_label': r'Hydraulic Conductivity [m h$^{-1}$]',
        'cmap': 'plasma',
        'save': True}
    labels_wte = {
        'data_path': os.path.join(temp_path, 'wte_global.tif'),
        'data_name': 'wte',
        'title': 'Water Table Elevation',
        'cbar_label': 'Water table [m]',
        'cmap': 'plasma',
        'save': True}
    # Process the aquifer saturated hydraulic conductivity and water table elevation data:
    res_crop_ksat_aq, _ = generic_process(labels_ksat_aq, temp_path, output_path)
    res_crop_wte, _ = generic_process(labels_wte, temp_path, output_path) 

    # Return resampled and cropped data and metadata:
    return res_crop_ksat_aq, res_crop_wte


# Define a function to calculate the specific yield (Sy):
def calculate_sy(input_path, temp_path, output_path):

    """
    This function roughly estimates the specific yield (Sy), based on the soil porosity (WCsat), 
    and the available water content (WCavail).
    
    Parameters:
    ----------
    input_path : str
        The path to the input directory.
    temp_path : str
        The path to the temporary directory.
    output_path : str
        The path to the output directory.
        
    Returns:
    -------
    sy : numpy array
        Roughly estimated specific yield (Sy).
    """

    # Get the resolution settings:
    resolution = config.resolution['output']
    # Get the intermediate step settings:
    intermediate_step = config.intermediate_step['plot']

    # Calculate the specific yield (Sy):
    with timer('Calculating the specific yield (Sy)...'):
        # Load the WCsat and WCavail NetCDF files:
        wc_sat_ds = xr.open_dataset(os.path.join(input_path, 'resampled_wc_sat.nc'))
        wc_avail_ds = xr.open_dataset(os.path.join(input_path, 'resampled_wc_avail.nc'))
        # Convert xarray DataArrays to numpy arrays:
        wc_sat_np = wc_sat_ds['resampled_wc_sat'].values
        wc_avail_np = wc_avail_ds['resampled_wc_avail'].values
        # Perform calculation:
        sy = np.minimum(wc_sat_np, wc_avail_np)
        # Extract coordinates and crs:
        y_max = wc_sat_ds.y.max().values
        x_min = wc_sat_ds.x.min().values
        crs = wc_sat_ds.crs
        # Generate transform matrix:
        transform = [resolution, 0, float(x_min), 0, -resolution, float(y_max)]
        # Create metadata:
        metadata = {'transform': transform, 
                    'height': sy.shape[0], 
                    'width': sy.shape[1], 
                    'dtype': sy.dtype, 
                    'nodata': np.nan,
                    'crs': crs}
        print(colored(' ✔ Done!', 'green'))
    # Save the data:
    save_data(sy, metadata, output_path,
              data_name='sy')
    # Plot the data:
    if intermediate_step:
        plot_data(sy, metadata, temp_path,
                  data_name='sy', title='Specific Yield', 
                  cbar_label=r'Specific Yield [m$^{3}$ m$^{-3}$]', cmap='plasma')
    print(colored('==========================================================================================', 'blue'))
    
    # Return the specific yield (Sy):
    return sy