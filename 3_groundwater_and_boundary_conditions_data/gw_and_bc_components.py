##################################################################################################
###################################### Author details ############################################
##################################################################################################

__date__        = "September 2023"
__author__      = "Nikola Ivanović"
__email__       = "nikola.ivanovic@kaust.edu.sa"
__institution__ = "King Abdullah University of Science and Technology (KAUST), SA"

##################################################################################################
######################################### Headline ###############################################
##################################################################################################

"""
Aim:
---
This script prepares groundwater and boundary conditions data for the DRYP 1.0 model. The script
downloads, processes, and saves the data outputs.

Input:
-----
1. Aquifer saturated hydraulic conductivity (downloaded from GLHYMPS website).
2. Water table depth (G³M 1.0 steady-state model output).

Operations:
----------
1. Download the data.
2. Process the downloaded data, and save the outputs.

Outputs:
-------
1. Aquifer Saturated Hydraulic Conductivity (Ksat_aq)	->		Ksat_aq
2. Specific Yield (Sy)									->		Sy
3. Initial Conditions Water Table Elevation (WTE)		->		wte
4. Flux Boundary Conditions (FBC)						->		None
5. Head Boundary Conditions (HBC)						->		hbc
"""

##################################################################################################
###################################  Main body of the script  ####################################
##################################################################################################

# Import modules:
import os
import sys
import importlib
import time
import atexit
import threading
from itertools import cycle
try:
    from termcolor import colored
except ImportError:
    print('Consider installing `termcolor` for colored outputs.')
import tarfile
import requests
import subprocess
from tqdm import tqdm
import rasterio
import numpy as np
import pandas as pd
import xarray as xr
from affine import Affine
from rasterio.mask import mask
from shapely.geometry import box
from scipy.ndimage import zoom
from rasterio.warp import reproject, calculate_default_transform, Resampling
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

##################################################################################################
# Define the functions:
##################################################################################################

"""
Contents:
--------
1. Configuration function
2. Timing functions
3. Data download function
4. Cropping function
5. Reprojection and resampling functions
6. GW parameters functions
7. Plotting function
8. Saving functions
"""

# ------------------------------------------------------------------------------------------------
# Configuration functions:
# ------------------------------------------------------------------------------------------------

# Define the configuration singleton class:
class ConfigSingleton:

    """
    This class is used to create a singleton object that holds the configuration 
    settings. The configuration settings are loaded from the configuration module
    specified in the argument, or from the configuration module specified in the
    command line.

    Attributes:
    ----------
    _instance : class
        The single instance of this class.
    config : module
        The configuration module.

    Methods:
    -------
    __new__(cls, config_name="config_AP")
        This function is used to control the creation of a single instance of this
        class, and to load the configuration settings.
    """
    
    # Define the class attribute to hold the single instance of this class:
    _instance = None
    # Define the function to load the configuration settings:
    def __new__(cls, config_name="config_AP"):

        """
        This function is used to control the creation of a single instance of this
        class. If an instance already exists, it is returned. If not, a new instance
        is created and returned. This function is also used to load the configuration
        settings. The configuration settings are loaded from the configuration module
        specified in the argument, or from the configuration module specified in the
        command line.
        
        Parameters:
        ----------
        cls : class
            The class to instantiate.
        config_name : str
            The name of the configuration module to import.
        
        Returns:
        -------
        cls._instance : class
            The single instance of this class.
        """

        # Check if an instance already exists:
        if cls._instance is None:
            # Create and assign a new instance:
            cls._instance = super(ConfigSingleton, cls).__new__(cls)
            try:
                # Check if the script is being run in a Jupyter Notebook environment:
                if 'ipykernel' in sys.modules:
                    # If so, import the configuration module specified in the argument:
                    cls._instance.config = importlib.import_module(config_name)
                else:
                    # If not, check if the user specified the configuration file in the command line:
                    config_name_from_arg = sys.argv[1] if len(sys.argv) > 1 else config_name
                    # Import the configuration module specified in the command line:
                    cls._instance.config = importlib.import_module(config_name_from_arg)
            except ImportError:
                    print(f"Error: The configuration module {config_name} could not be imported.")
                    sys.exit(1)
                     
        # Return the single instance:
        return cls._instance

# Create a singleton object that holds the configuration settings:
config = ConfigSingleton().config

# ------------------------------------------------------------------------------------------------
# Timing functions:
# ------------------------------------------------------------------------------------------------

# Class to time the execution of a block of code:
class timer:

    """
    Class to time the execution of a block of code.

    Attributes:
    ----------
    message : str
        The message to be displayed before the timer.
    start : float
        The time at which the timer started.
    _stop_event : threading.Event
        The event to stop the timer.
    thread : threading.Thread
        The thread to run the timer.
    spinner : itertools.cycle
        The spinner to be displayed while the timer is running.
    
    Methods:
    -------
    _show_time()
        Displays the elapsed time.
    __enter__()
        Starts the timer.
    __exit__()
        Stops the timer.
    """

    # Initialize the class:
    def __init__(self, message):
        # Set the message:
        self.message = message
        # Initialize the timer:
        self.start = None
        # Initialize the stop event:
        self._stop_event = threading.Event()
        # Initialize the thread:
        self.thread = threading.Thread(target=self._show_time)
        # Initialize the spinner:
        self.spinner = cycle(['♩', '♪', '♫', '♬'])

    # Function to show the time:
    def _show_time(self):
        # Loop until the stop event is set:
        while not self._stop_event.is_set():
            # Get the elapsed time:
            elapsed_time = time.time() - self.start
            # Convert the elapsed time to hours, minutes, and seconds:
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            # Get the spinner character:
            spinner_char = next(self.spinner)
            # Format the time string:
            time_str = f"{str(int(hours)).zfill(2)}:{str(int(minutes)).zfill(2)}:{str(int(seconds)).zfill(2)}"
            # Print the message, spinner, and time:
            print(f"\r{self.message} {colored(spinner_char, 'blue')} {colored(time_str, 'light_cyan')}", end="")
            # Wait for 0.2 seconds:
            time.sleep(0.2)

    # Function to start the timer:
    def __enter__(self):
        # Start the timer:
        self.start = time.time()
        # Start the thread:
        self.thread.start()

    # Function to stop the timer:
    def __exit__(self, exc_type, exc_value, exc_traceback):
        # Set the stop event:
        self._stop_event.set()
        # Join the thread:
        self.thread.join()


# Define the script timer class:
class script_timer:

    """
    This class is used to display the runtime of the script at the end.
    
    Attributes:
    ----------
    start_time : float
        The time at which the script started.
    
    Methods:
    -------
    display_runtime()
        Displays the runtime of the script.
    """

    # Define the constructor:
    def __init__(self):
        # Get the start time:
        self.start_time = time.time()
        # Register the display_runtime() method to be called at the end of the script:
        atexit.register(self.display_runtime)

    # Define the display_runtime() method:
    def display_runtime(self):
        # Get the runtime:
        runtime = time.time() - self.start_time
        # Convert the runtime to hours, minutes, and seconds:
        hours, remainder = divmod(runtime, 3600)
        minutes, seconds = divmod(remainder, 60)
        # Format the time string:
        time_str = f"{str(int(hours)).zfill(2)}:{str(int(minutes)).zfill(2)}:{str(int(seconds)).zfill(2)}"
        # Print the runtime:
        print(colored(f"\nTOTAL RUNTIME ▶   {time_str}", 'cyan', attrs=['bold']))
        print('==========================================================================================')

# ------------------------------------------------------------------------------------------------
# Data download functions:
# ------------------------------------------------------------------------------------------------

# Define a function to download and extract the file:
def download_data(output_path):

    """
    This function downloads and unzips G³M 1.0 steady-state model output.

    Parameters:
    ----------
    output_path : str
        The path to the output directory.
    
    Returns:
    -------
    None
    """

    # Get the URL settings:
    url = config.url['wte']

    # Remove query parameters from the URL to clean the filename
    clean_url = url.split('?')[0]
    # Set the filename
    filename = os.path.join(output_path, clean_url.rsplit('/', 1)[-1])
    # Download the file if it doesn't exist
    if not os.path.exists(filename):
        try:
            r = requests.get(url, stream=True)
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            # Open the file to write the chunks and display the tqdm progress bar
            with open(filename, 'wb') as f, tqdm(
                desc=f"Downloading {filename}...",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024) as bar:
                    for chunk in r.iter_content(chunk_size=8192*1000):
                        size = f.write(chunk)
                        bar.update(size)
        # If the download fails, print the error:
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {url}. Error: {e}")
    # Extract the tar.gz file and flatten the folder structure
    if filename.endswith(".tar.gz"):
        with tarfile.open(filename, "r:gz") as tar:
            for member in tar.getmembers():
                if member.isreg():  # Skip if the TarInfo is not files
                    member.name = os.path.basename(member.name)  # Remove folder names
                    tar.extract(member, path=output_path)
        # Remove the tar.gz file
        os.remove(filename)

# ------------------------------------------------------------------------------------------------
# Cropping functions:
# ------------------------------------------------------------------------------------------------

# Define a function to crop the data:
def crop_data(labels):

    """
    Crop the raster to the extent of the research area.

    Parameters:
    ----------
    labels : dict
        Dictionary containing the labels of the data.

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
    with timer(f"Cropping the {labels['data_name']}..."):
        # Read the raster file:
        with rasterio.open(labels['data_path']) as src:
            # Crop raster using bounding box:
            out_image, _ = mask(src, [bbox], crop=True)
            # Squeeze the data to 2D if it's 3D:
            if len(out_image.shape) == 3:
                out_image = np.squeeze(out_image, axis=0)
            # Convert integer to float:
            out_image = out_image.astype(np.float32)
            # Replace no data with np.nan:
            out_image[out_image == 0] = np.nan
            # Correct the values:
            if labels['data_name'] == 'ksat_aq':
                out_image = np.power(10, (out_image / 100)) * 1e+7
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
                             'dtype': out_image.dtype, 
                             'transform': transform, 
                             'nodata': np.nan, 
                             'crs': src.crs})
        print(colored(' ✔ Done!', 'green'))

    # Return the cropped raster and the new metadata:
    return out_image, out_meta

# ------------------------------------------------------------------------------------------------
# Reprojection and resampling functions:
# ------------------------------------------------------------------------------------------------

# Define a function to resample the data:
def resample_data(data, metadata):

    """
    Resample the data to the resolution specified in the configuration file.

    Parameters:
    ----------
    data : numpy array
        The data to be resampled.
    metadata : dict
        Dictionary containing the metadata of the data.

    Returns:
    -------
    resampled_data : numpy array
        The resampled data.
    resampled_metadata : dict
        Dictionary containing the metadata of the resampled data.
    """

    # Get the output resolution settings:
    resolution = config.resolution['wte']

    # Resample the data:
    with timer('Resampling the data...'):
        # Get the bounds of the data:
        bounds = {
            'min_lon': metadata['transform'].c,
            'max_lon': metadata['transform'].c + metadata['transform'].a * data.shape[1],
            'min_lat': metadata['transform'].f + metadata['transform'].e * data.shape[0],
            'max_lat': metadata['transform'].f}
        # Calculate the output shape:
        output_shape_y = int((bounds['max_lat'] - bounds['min_lat']) / resolution)
        output_shape_x = int((bounds['max_lon'] - bounds['min_lon']) / resolution)
        # Calculate the resampling factor:
        factor_y = output_shape_y / data.shape[0]
        factor_x = output_shape_x / data.shape[1]
        # Resample the data (order=1 for bilinear interpolation):
        resampled_data = zoom(data, (factor_y, factor_x), order=1)
        print(colored(' ✔ Done!', 'green'))
        # Updating the metadata:
        original_transform = metadata['transform']
        new_transform = Affine(resolution, 
                               original_transform.b, 
                               original_transform.c, 
                               original_transform.d, 
                               -resolution, 
                               original_transform.f)
        resampled_metadata = metadata.copy()
        resampled_metadata.update({'height': resampled_data.shape[0], 
                                   'width': resampled_data.shape[1], 
                                   'transform': new_transform})
        print(colored(' ✔ Done!', 'green'))

    # Return the resampled data and metadata:
    return resampled_data, resampled_metadata

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
                               0, 
                               transform[3], 
                               resolution, 
                               0)
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
                  cbar_label=r'Water table [m]', cmap='plasma')
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

# ------------------------------------------------------------------------------------------------
# Plotting function:
# ------------------------------------------------------------------------------------------------

# Define a function to plot the output data:
def plot_data(data, metadata, temp_path, 
              data_name=None, title=None, cbar_label=None, cmap=None):
    
    """
    Plots the output data with a colorbar, prints the information, 
    and saves the plot in the temporary directory.

    Parameters:
    ----------
    data : numpy array
        The data to be plotted.
    metadata : dict
        Dictionary containing the metadata of the output data.    
    temp_path : str
        The path to the temporary directory.
    data_name : str
        The name of the data to be plotted.
    title : str
        The title for the plot.
    cbar_label : str
        The label for the colorbar.
    cmap : str
        The colormap to be used for the plot.
    
    Returns:
    -------
    None
    """

    # Define the extent using metadata:
    extent = (metadata['transform'][2], 
              metadata['transform'][2] + metadata['transform'][0] * metadata['width'], 
              metadata['transform'][5] + metadata['transform'][4] * metadata['height'], 
              metadata['transform'][5])
    # Create the plot:
    with timer(f"Plotting the {data_name}..."):
        fig, ax = plt.subplots(figsize=(10, 10), tight_layout=True)
        # Plot the data:
        im = ax.imshow(data, cmap=cmap, extent=extent)
        # Set the title, labels, tick parameters, and grid:
        ax.set_title(title, fontsize=20, fontweight='bold', pad=20)
        ax.set_xlabel('Longitude [m]', fontsize=16, labelpad=10)
        ax.set_ylabel('Latitude [m]', fontsize=16, labelpad=10)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True, color='lightgray', linestyle='--', linewidth=0.5)
        # Set the colorbar:
        colorbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        colorbar.ax.tick_params(labelsize=14)
        colorbar.ax.set_ylabel(cbar_label, fontsize=16, labelpad=10)
        print(colored(' ✔ Done!', 'green'))
    # Store and print the data information:
    print(colored('Data information', 'light_grey', attrs=['underline']))
    info_lines = [
        'Data size: ' + colored(f"{data.nbytes / (1024*1024*1024):.2f} GB", 'yellow'), 
        'Data dtype: ' + colored(f"{data.dtype}", 'yellow'), 
        'Data CRS: ' + colored(f"{metadata['crs']}", 'yellow'), 
        'Data shape: ' + colored(f"{data.shape}", 'yellow'), 
        'Data resolution: ' + colored(f"{metadata['transform'][0]} [m]", 'yellow')]
    print('\n'.join(info_lines))
    # Save the figure:
    with timer(f"Saving the {data_name} plot..."):
        fig.savefig(os.path.join(temp_path, f"{data_name}_plot.png"), 
                    bbox_inches='tight', format='png', dpi=300)
        print(colored(' ✔ Done!', 'green'))
    # Close the figure:
    plt.close(fig)

# ------------------------------------------------------------------------------------------------
# Saving functions:
# ------------------------------------------------------------------------------------------------

# Define a function to save the data as a GeoTIFF file:
def save_as_geotiff(data, metadata, path):
    
    """
    Save the data as a GeoTIFF file.

    Parameters:
    ----------
    data : numpy array
        The data to be saved.
    metadata : dict
        Dictionary containing the metadata of the data.
    path : str
        The path to the output file.

    Returns:
    -------
    None
    """

    with rasterio.open(path, 'w',
                       driver='GTiff',
                       height=metadata['height'],
                       width=metadata['width'],
                       count=1,
                       dtype=data.dtype,
                       crs=metadata['crs'],
                       nodata=metadata['nodata'],
                       transform=metadata['transform']) as dst:
        dst.write(data, 1)


# Define a function to save the data as a format of choice:
def save_data(data, metadata, output_path, 
              format=config.saving['format'],
              data_name=None):
    
    """
    This function saves the data as a format of choice. The supported formats are:
    - GeoTIFF ('GTiff')
    - NetCDF ('NetCDF')
    - ASCII Grid ('AAIGrid')

    Parameters:
    ----------
    data : numpy array
        The data to be saved.
    metadata : dict
        Dictionary containing the metadata of the data.
    output_path : str
        The path to the output directory.
    format : str
        Format of the output file (optional).
    data_name : str
        Name of the data to be saved (optional).

    Returns:
    -------
    None
    """

    # Save the data as a format of choice:
    with timer(f"Saving {data_name} as a {format} file..."):
        # Save as GeoTIFF:
        if format == 'GTiff':
            save_as_geotiff(data, metadata, 
                            os.path.join(output_path, f"{data_name}.tif"))
        # Save as NetCDF4:
        elif format == 'NetCDF':
            # Create x and y coordinates using metadata
            x = np.arange(metadata['width']) * metadata['transform'][0] + metadata['transform'][2]
            y = np.arange(metadata['height']) * metadata['transform'][4] + metadata['transform'][5]
            # Create a dataset:
            ds = xr.Dataset({data_name: (['y', 'x'], data)},
                             coords={'x': x, 'y': y},
                             attrs={'crs': str(metadata['crs']), 
                                    'cellsize': metadata['transform'][0]})
            # Save the dataset:
            ds.to_netcdf(os.path.join(output_path, f"{data_name}.nc"))
        # Save as ASCII Grid:
        elif format == 'AAIGrid':
            # Save as a temporary GeoTIFF:
            temp_path = os.path.join(output_path, f"{data_name}.tif")
            save_as_geotiff(data, metadata, temp_path)
            # Convert the GeoTIFF to AAIGrid using GDAL Translate:
            input_file = temp_path
            output_file = os.path.join(output_path, f"{data_name}.asc")
            command = f"gdal_translate -of AAIGrid {input_file} {output_file}"
            os.system(command)
            # Remove the temporary GeoTIFF file:
            os.remove(input_file)
        else:
            # Raise an error if the format is not supported:
            raise ValueError("Unsupported format. Choose from 'GTiff', 'NetCDF', 'AAIGrid'.")
        print(colored(' ✔ Done!', 'green'))

##################################################################################################
# Script timer:
##################################################################################################

# Display the runtime of the script at the end:
runtime = script_timer()

##################################################################################################
# Set up directories:
##################################################################################################

# Get the directories:
input_dir_ksat_aq = config.dir['input_ksat_aq']
input_dir_wte = config.dir['input_wte']
input_dir_sy = config.dir['input_sy']
output_dir = config.dir['output']
temp_dir = config.dir['temp']

##################################################################################################
# Download the data:
##################################################################################################

# Download the GLHYMPS data manually.
# Download the G³M 1.0 data:
# download_data(input_dir_wte)

##################################################################################################
# Process the data and save the outputs:
##################################################################################################

# Calculate the parameters:
res_crop_ksat_aq, res_crop_wte = process_data(input_dir_ksat_aq, input_dir_wte,  
                                              temp_dir, output_dir)
# Calculate the specific yield (Sy):
# sy = calculate_sy(input_dir_sy, temp_dir, output_dir)

##################################################################################################
# Remove temporary files:
##################################################################################################

# If the user wants to delete the temporary directory:
if config.intermediate_step['delete']:
    os.rmdir(temp_dir)

##################################################################################################
######################################  End of the script  #######################################
##################################################################################################