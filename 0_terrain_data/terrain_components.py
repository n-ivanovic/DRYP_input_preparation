##################################################################################################
###################################### Author details ############################################
##################################################################################################

__date__        = "August 2023"
__author__      = "Nikola Ivanović"
__email__       = "nikola.ivanovic@kaust.edu.sa"
__department__  = "Earth Science and Engineering"
__institution__ = "King Abdullah University of Science and Technology, KSA"

##################################################################################################
######################################### Headline ###############################################
##################################################################################################

"""
Aim:
---
This script prepares terrain data for the DRYP 1.0 model. The script downloads the data, defines
the research area extent, merges the data, resamples the data, and calculates terrain parameters.

Input:
-----
1. DEM tiles (downloaded from MERIT Hydro website);
2. UPSTREAM tiles (downloaded from MERIT Hydro website).

Operations:
----------
1. Download the data;
2. Define the research area extent;
3. Merge the data;
4. Resample the data;
5. Calculate terrain parameters.

Outputs:
-------
1. Topography (DEM)			                    -> 	           	  res_merged_dem
2. Cell factor area     		                ->	    	      cell_area
3. Flow direction				                ->	    	      flow_dir
4. Boundary conditions (CHB)                    ->	              chb
5. Basin mask     	                            ->	              terrain_mask
6. River lenghts				                ->	              river_lengths
7. River widths				                    ->	              None
8. River bottom elevation                       ->	              None
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
from tqdm import tqdm
from bs4 import BeautifulSoup
from requests.auth import HTTPBasicAuth
from concurrent.futures import ThreadPoolExecutor
import rasterio
import numpy as np
from affine import Affine
from rasterio.merge import merge
from rasterio.warp import reproject, calculate_default_transform, Resampling
from landlab import RasterModelGrid
from landlab.components import FlowAccumulator
from shapely.geometry import Polygon
from rasterio.features import geometry_mask
from matplotlib import colors
from rasterio.plot import show
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import xarray as xr

##################################################################################################
# Define the functions:
##################################################################################################

"""
Contents:
--------
1. Configuration function
2. Timing functions
3. Data download functions
4. Data extent functions
5. Merging functions
6. Reprojection and resampling functions
7. Terrain parameters functions
8. Plotting functions
9. Saving functions
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
        print(colored('==========================================================================================', 'cyan'))
        
# ------------------------------------------------------------------------------------------------
# Data download functions:
# ------------------------------------------------------------------------------------------------

# Define a function to extract individual files from the tar file:
def extract_tar(tar_path, output_path):
    
    """
    Extracts individual files from the tar file to the specified output path.

    Parameters:
    ----------
    tar_path : str
        The path to the tar file.
    output_path : str
        The path to the output directory.

    Returns:
    -------
    None
    """

    # Extract individual files from the tar file:
    with tarfile.open(tar_path, "r") as tar:
        for member in tar.getmembers():
            # Check if the member is a regular file:
            if member.isreg():
                # Extract the member to the specified output path:
                member.name = os.path.basename(member.name)
                tar.extract(member, output_path)
        # Remove the tar file:
        os.remove(tar_path)


# Define a function to download and extract the file:
def download_extract(url, auth, link, output_path):

    """
    Downloads and extracts the file.

    Parameters:
    ----------
    url : str
        URL of the data.
    auth : tuple
        Username and password for the authorization.
    link : str
        URL of the file.
    output_path : str
        The path to the output directory.

    Returns:
    -------
    None
    """

    # Set the filename:
    filename = os.path.join(output_path, link.rsplit('/', 1)[-1])
    # Download the file if it doesn't exist:
    if not os.path.exists(filename):
        try:
            # Send GET request to the source URL with authorization:
            r = requests.get(url + link, stream=True, auth=HTTPBasicAuth(*auth))
            r.raise_for_status()
            # Write the file to the specified output path:
            with open(filename, 'wb') as f:
                # Write the file in chunks:
                for chunk in r.iter_content(chunk_size=8192*1000):
                    f.write(chunk)
        # If the download fails, print the error:
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {link}. Error: {e}")
    # Apply the extract_tar function:
    if filename.endswith(".tar"):
        extract_tar(filename, output_path)


# Define a function to download data:
def download_data(output_path, data_type):

    """
    Downloads data from the official sources.

    Parameters:
    ----------
    output_path : str
        The path to the output directory.
    data_type : str
        Type of the data.

    Returns:
    -------
    None
    """

    # Get the URL and authorization settings:
    url = config.url[data_type]
    auth = config.auth[data_type]
    
    # Send GET request to the source URL with authorization:
    response = requests.get(url, auth=HTTPBasicAuth(*auth))
    soup = BeautifulSoup(response.text, 'html.parser')
    # DEM data:
    if data_type == 'DEM':
        # Get the links to the tiff files:
        links = [link['href'] for link in soup.find_all('a', href=True) 
                 if link['href'].startswith('./distribute/v1.0.2/dem') 
                 and link['href'].endswith('.tar')]
        # Print a message:
        print('The number links found:', len(links))
    # Upstream data:
    elif data_type == 'UPSTREAM':
        # Get the links to the tar files ending with 'upa' and '.tar':
        links = [link['href'] for link in soup.find_all('a', href=True) 
                 if link['href'].startswith('./distribute/v1.0/upa') 
                 and link['href'].endswith('.tar')]
    # Download and extract the files:
    with ThreadPoolExecutor(max_workers=10) as executor:
        list(tqdm(executor.map(download_extract, [url]*len(links), [auth]*len(links), links, [output_path]*len(links)), 
                  total=len(links), desc="Downloading and extracting files..."))
    # Print a message:
    print('Download and extraction complete!')
    print(colored('==========================================================================================', 'blue'))

# ------------------------------------------------------------------------------------------------
# Data extent functions:
# ------------------------------------------------------------------------------------------------

# Define a function to check if raster file is within specified bounds:
def data_bounds(file_path):

    """
    Check if raster file is within specified bounds.
    
    Parameters:
    ----------
    file_path : str
        Path to data raster tile.
    
    Returns:
    -------
    bool
        True if raster file is within specified bounds, False otherwise.
    """
    
    # Get the research area bounds settings:
    min_lon = config.bounds['min_lon']
    max_lon = config.bounds['max_lon']
    min_lat = config.bounds['min_lat']
    max_lat = config.bounds['max_lat']

    # Open the file:
    with rasterio.open(file_path) as src:
        # Get the coordinates of the top-left corner:
        lon, lat = src.transform * (0, 0)
        
        # Check if the coordinates are within the specified bounds:
        return min_lon <= lon <= max_lon and min_lat <= lat <= max_lat


# Define a function to get the data tiles:
def get_data(input_path_dem, input_path_ups, temp_path):
    
    """
    Get all the data tiles within the specified bounds.

    Parameters:
    ----------
    input_path_dem : str
        The path to the directory containing the DEM tiles.
    input_path_ups : str
        The path to the directory containing the upstream tiles.
    temp_path : str
        The path to the temporary directory.

    Returns:
    -------
    dem_tiles : list
        A list containing the paths of the DEM .tif files.
    ups_tiles : list
        A list containing the paths of the UPSTREAM .tif files.
    """
    
    # Get the intermediate step settings:
    intermediate_step = config.intermediate_step['plot']

    # Initialize lists to store the data tiles:
    dem_tiles = []
    ups_tiles = []
    # Loop over each data type and corresponding input path
    for data_type, input_path in [('DEM', input_path_dem), ('UPSTREAM', input_path_ups)]:
        # Get all the files in the directory:
        all_files = os.listdir(input_path)        
        # Create a progress bar:
        with tqdm(total=len(all_files), desc=f'Creating the {data_type} data list...') as pbar:
            # Loop through all the files:
            for tile in all_files:
                # Update the progress bar:
                pbar.update()
                # If the file meets your criteria, add it to the appropriate list:
                if tile.endswith('.tif') and data_bounds(os.path.join(input_path, tile)):
                    if data_type == 'DEM':
                        dem_tiles.append(os.path.join(input_path, tile))
                    else:
                        ups_tiles.append(os.path.join(input_path, tile))
        # Get the data tiles info:
        if intermediate_step:
            plot_data_tiles(dem_tiles if data_type == 'DEM' else ups_tiles, temp_path, data_type=data_type)
    print(colored('==========================================================================================', 'blue'))
    
    # Return the data tiles lists:
    return dem_tiles, ups_tiles

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
    crs = config.reproject['crs_global']
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

# ------------------------------------------------------------------------------------------------
# Terrain parameters functions:
# ------------------------------------------------------------------------------------------------

# Define a function to create a cell factor area:
def cell_factor_area(res_merged_dem, res_metadata, temp_path, output_path):

    """
    Create a cell factor area array based on ouput data resolution.

    Parameters:
    ----------
    res_merged_dem : numpy array
        The resampled merged DEM data.
    res_metadata : dict
        Dictionary containing the metadata of the resampled data.
    temp_path : str
        The path to the temporary directory.
    output_path : str
        The path to the output directory.

    Returns:
    -------
    cell_area : numpy array
        The cell factor area array.
    """
    
    # Get the resolution settings:
    resolution = config.resolution['output']

    with timer('Creating a cell factor area array...'):
        # Calculate the cell area:
        cell_area_value = resolution**2
        # Fill the array with the cell area value:
        cell_factor_area = np.full_like(res_merged_dem, cell_area_value, dtype=np.float32)
        print(colored(' ✔ Done!', 'green'))
    # Save the cell factor area array:
    save_data(cell_factor_area, res_metadata, output_path, 
              data_name='cell_factor_area')
    print(colored('==========================================================================================', 'blue'))

    # Return the cell factor area array:
    return cell_factor_area


# Define a function to invert the resampled merged upstream data:
def invert_upstream(res_merged_upstream, res_metadata, temp_path):

    """
    Invert the resampled merged upstream data to create the pseudo elevation map. 
    The pseudo elevation map is used to compute more accurate flow direction and
    flow accumulation maps for the research area.

    Parameters:
    ----------
    res_merged_upstream : numpy array
        The resampled merged upstream data.
    metadata : dict
        Dictionary containing the metadata of the resampled data.
    temp_path : str
        The path to the temporary directory.
    
    Returns:
    -------
    pseudo_elevation : numpy array
        Inverted merged upstream data, or so-called pseudo elevation.
    """

    # Get the intermediate step settings:
    intermediate_step = config.intermediate_step['plot']

    # Invert the resampled merged upstream data:
    with timer('Inverting the upstream data...'):
        p_elev = np.max(res_merged_upstream) - res_merged_upstream
        # Normalize the data to the range [0, 1]:
        p_elev = (p_elev - np.min(p_elev)) / (np.max(p_elev) - np.min(p_elev))
        print(colored(' ✔ Done!', 'green'))
    # Plot the resampled inverted upstream data:
    if intermediate_step:
        plot_data(p_elev, res_metadata, temp_path, 
                  data_name='pseudo_elevation', title='Pseudo elevation', 
                  cbar_label='Upstream cells', cmap='cubehelix', inverse=True)
    print(colored('==========================================================================================', 'blue'))

    # Return the pseudo elevation:
    return p_elev


# Define a function to compute the flow direction and accumulation:
def flow_accumulation(pseudo_elevation, res_metadata, temp_path, output_path):

    """
    Compute a flow direction map from the pseudo elevation map, using the D8 method. The 
    flow direction map is used to compute the flow accumulation map, which is afterwards
    used to extract the river network.

    Parameters:
    ----------
    pseudo_elevation : numpy array
        Inverted merged upstream data, or so-called pseudo elevation.
    res_metadata : dict
        Dictionary containing the metadata of the resampled merged DEM.
    temp_path : str
        The path to the temporary directory.
    output_path : str
        The path to the output directory.
        
    Returns:
    -------
    flow_direction : numpy array
        Flow Direction map.
    flow_accumulation : numpy array
        Flow Accumulation map.
    """

    # Get the intermediate step settings:
    intermediate_step = config.intermediate_step['plot']

    # Compute the flow direction and accumulation:
    with timer('Computing the flow direction and accumulation...'):
        # Initialize a RasterModelGrid instance from Landlab:
        nrows, ncols = pseudo_elevation.shape
        grid = RasterModelGrid((nrows, ncols))
        # Cast the pseudo_elevation to float64:
        pseudo_elevation = pseudo_elevation.astype(np.float64)
        # Add the elevation data to the grid:
        grid.add_field('topographic__elevation', pseudo_elevation, at='node', clobber=True)
        # Initialize and run the FlowAccumulator component:
        fa = FlowAccumulator(grid, flow_director='D8')
        fa.run_one_step()
        # Extract the flow direction and accumulation data:
        flow_direction = grid.at_node['flow__receiver_node']
        flow_accumulation = grid.at_node['drainage_area']
        # Reshape the flow direction and accumulation data:
        flow_direction_2D = np.flip(flow_direction.reshape((nrows, ncols)), 0).astype(np.int32)
        flow_accumulation_2D = flow_accumulation.reshape((nrows, ncols)).astype(np.float32)
        print(colored(' ✔ Done!', 'green'))    
    # Save the flow direction:
    save_data(flow_direction_2D, res_metadata, output_path, 
              data_name='flow_direction')
    # Use landlab 
    # Plot the flow accumulation:
    if intermediate_step:
        plot_data(flow_accumulation_2D, res_metadata, temp_path, 
                  data_name='flow_accumulation', title='Flow accumulation', 
                  cbar_label='Upstream cells', cmap='cubehelix', log_scale=True)
    print(colored('==========================================================================================', 'blue'))

    # Return the flow direction and accumulation maps:
    return flow_direction_2D, flow_accumulation_2D


# Calculate the inferred flow directions (D8) from the flow accumulation:
def infer_fd_d8_from_accumulation(flow_accumulation_array, res_metadata, temp_path):

    """
    Infer the flow directions from the flow accumulation map, using the D8 method.
    CURRENTLY NOT USED/NEEDED.

    Parameters:
    ----------
    flow_accumulation_array : numpy array
        Flow accumulation map.
    res_metadata : dict
        Dictionary containing the metadata of the resampled merged DEM.
    temp_path : str
        The path to the temporary directory.
    
    Returns:
    -------
    fd_d8 : numpy array
        Inferred flow directions.
    """
    # Get the intermediate step settings:
    intermediate_step = config.intermediate_step['plot']

    # Get the number of rows and columns:
    nrows, ncols = flow_accumulation_array.shape
    # Initialize an array to store the inferred flow directions:
    fd_d8 = np.zeros_like(flow_accumulation_array, dtype=float)
    # Loop through each cell in the flow accumulation array:
    for r in range(1, nrows - 1):
        for c in range(1, ncols - 1):
            # Extract the 3x3 neighborhood of the current cell:
            neighborhood = flow_accumulation_array[r-1:r+2, c-1:c+2]
            # Calculate the differences between the center cell and its neighbors:
            diffs = neighborhood - neighborhood[1, 1]
            # Set the flow direction to the direction of the steepest increase in accumulation:
            steepest_increase_direction = np.argmax(diffs)
            fd_d8[r, c] = steepest_increase_direction
    # Plot the inferred flow directions:
    if intermediate_step:
        plot_data(fd_d8, res_metadata, temp_path, 
                  data_name='inferred_flow_directions', title='Inferred flow directions', 
                  cbar_label='Direction', cmap='cubehelix')
    
    # Return the inferred flow directions:
    return fd_d8


# Define a function to extract the river network:
def river_network(flow_accumulation, cell_area, res_metadata, temp_path):

    """
    Extract the river network from the flow accumulation map, using the threshold 
    areal coverage specified in the configuration file. The river network is then 
    used to compute the constant head boundary conditions, and the river lengths.

    Parameters:
    ----------
    flow_accumulation : numpy array
        Flow accumulation map.
    cell_area : numpy array
        The cell factor area array.
    res_metadata : dict
        Dictionary containing the metadata of the resampled data.
    temp_path : str
        The path to the temporary directory.

    Returns:
    -------
    river_network : numpy array
        Extracted river network based on the threshold areal coverage.
    """

    # Get the threshold area [m^2] for the river network extraction:
    threshold = config.threshold['area']
    # Get the intermediate step settings:
    intermediate_step = config.intermediate_step['plot']

    # Convert the upstream cells to area:
    with timer('Converting flow accumulation to area...'):
        flow_accumulation_area =  flow_accumulation * cell_area
        print(colored(' ✔ Done!', 'green'))
    # Extract the river network based on the threshold area [m^2]:
    with timer('Extracting the river network...'):
        river_network = np.where(flow_accumulation_area > threshold, 1, 0).astype(np.int32)
        print(colored(' ✔ Done!', 'green'))
    # Plot the river network:
    if intermediate_step:
        plot_data(river_network, res_metadata, temp_path, 
                  data_name='river_network', title='River network', 
                  cbar_label=f"Rivers for {threshold} [m$^2$] areal coverage", 
                  cmap='Blues', binary=True)
    print(colored('==========================================================================================', 'blue'))

    # Return the river network:
    return river_network


# Define a function to compute constant head boundary conditions:
def boundary_conditions(river_network, res_merged_dem, res_metadata, temp_path, output_path):

    """
    Add terrain elevation values to the extracted river network to create 
    the constant head boundary conditions for the research area.

    Parameters:
    ----------
    river_network : numpy array
        Extracted river network.
    res_merged_dem : numpy array
        Resampled merged DEM.
    res_metadata : dict
        Dictionary containing the metadata of the resampled data.
    temp_path : str
        The path to the temporary directory.
    output_path : str
        The path to the output directory.
        
    Returns:
    -------
    chb : numpy array
        Constant head boundary for the research area.
    """
    
    # Get the intermediate step settings:
    intermediate_step = config.intermediate_step['plot']

    # Add terrain values to the river network to create constant head boundary conditions:
    with timer('Adding terrain values to the river network...'):
        chb = (river_network * res_merged_dem).astype(np.float32)
        print(colored(' ✔ Done!', 'green'))
    # Plot the constant head boundary conditions:
    if intermediate_step:
        plot_data(chb, res_metadata, temp_path, 
                  data_name='chb', title='Constant head boundary', 
                  cbar_label='Head [m]', cmap='cubehelix')
    # Replace the 0 values with -9999:
    chb = np.where(chb == 0, -9999, chb)    
    # Save the constant head boundary conditions:
    save_data(chb, res_metadata, output_path, 
              data_name='chb')
    print(colored('==========================================================================================', 'blue'))

    # Return the constant head boundary conditions:
    return chb


# Define a function to create research area terrain mask:
def terrain_mask(merged_dem, metadata, temp_path, output_path):

    """
    Create a terrain mask to differentiate land mass (elevation >= 0) from the sea (elevation <= 0), and
    to limit the research area to the specified bounds in the configuration file.

    Parameters:
    ----------
    merged_dem : numpy array
        The merged DEM data.
    metadata : dict
        Dictionary containing the metadata of the data.
    temp_path : str
        The path to the temporary directory.
    output_path : str
        The path to the output directory.

    Returns:
    -------
    terrain_mask : numpy array
        Terrain mask with 1s for land, and 0s for sea.
    """

    # Get the terrain mask polygon bounds:
    polygon = config.polygon['bounds']

    # Create a research area mask:
    with timer('Creating a terrain mask...'):
        # Define the data labels:
        labels = {
            'method': 'mean',
            'data_name': 'terrain_mask',
            'title': 'Terrain mask',
            'cbar_label': 'Terrain shape',
            'cmap': 'binary',
            'log_scale': False,
            'inverse': False,
            'binary': True,
            'save': True}
        # Divide the land mass from the sea (elevation >= 0):
        dem_mask = np.where(merged_dem > 0, 1, 0).astype(np.int32)
        # Create a polygon from the bounds:
        bounds = Polygon(polygon)
        # Create a mask from the polygon:
        mask = geometry_mask([bounds], out_shape=dem_mask.shape, transform=metadata['transform'], invert=True)
        # Apply the mask:
        dem_mask = np.where(mask == 1, dem_mask, 0)
        # Resample the terrain mask:
        terrain_mask, _ = generic_reproject(dem_mask, metadata, labels, temp_path, output_path)
        # Convert the reprojected resampled terrain mask values to 1s and 0s:
        terrain_mask = np.where(terrain_mask > 0, 1, 0).astype(np.int32)
        print(colored(' ✔ Done!', 'green'))
    print(colored('==========================================================================================', 'blue'))

    # Return the terrain mask:
    return terrain_mask


# Define a function to compute river lengths:
def river_lengths(river_network, res_metadata, temp_path, output_path):
    
    """
    Compute the river lengths for the river network, by using the resolution 
    of the output data specified in the configuration file.

    Parameters:
    ----------
    river_network : numpy array
        River network array with 1s for rivers, and 0s for non-rivers.
    res_metadata : dict
        Dictionary containing the metadata of the resampled data.
    temp_path : str
        The path to the temporary directory.
    output_path : str
        The path to the output directory.
    
    Returns:
    -------
    river_lengths : numpy array
        River network lengths in meters.
    """

    # Get the resolution settings:
    resolution = config.resolution['output']
    # Get the intermediate step settings:
    intermediate_step = config.intermediate_step['plot']

    # Calculate the river network lengths:
    with timer('Computing the river network lengths...'):
        river_lengths = (river_network * resolution).astype(np.float32)
        print(colored(' ✔ Done!', 'green'))
    # Save the river network length:
    with timer('Saving the river network lengths...'):
        save_data(river_lengths, res_metadata, output_path, 
                  data_name='river_lengths')
        print(colored(' ✔ Done!', 'green'))
    # Plot the river network length:
    if intermediate_step:
        plot_data(river_lengths, res_metadata, temp_path, 
                  data_name='river_lengths', title='River network lengths', 
                  cbar_label='Length [m]', cmap='Blues', binary=True)
    print(colored('==========================================================================================', 'blue'))

    # Return the river network lengths:
    return river_lengths

# ------------------------------------------------------------------------------------------------
# Plotting functions:
# ------------------------------------------------------------------------------------------------

# Define a function to plot the data tiles information:
def plot_data_tiles(data_tiles, temp_path, data_type):

    """
    Check the data extent, and provide information about the entire data list.
    
    Parameters:
    ----------
    data_tiles : list
        List of data tiles covering the specified research area.
    temp_path : str
        Full path to the temporary directory.
    data_type : str
        Type of the data (e.g. 'DEM', 'UPSTREAM').

    Returns:
    -------
    None
    """
    
    # Define a helper function to format latitude:
    def format_lat(lat):
        return f"{lat}{'°N' if lat >= 0 else '°S'}"
    # Define a helper function to format longitude:
    def format_lon(lon):
        return f"{lon}{'°E' if lon >= 0 else '°W'}"
    # Initialize variables to store information:
    with timer(f"Plotting the {data_type} extent..."):
        total_size = 0
        res_counter = Counter()
        min_lon, max_lon, min_lat, max_lat = np.inf, -np.inf, np.inf, -np.inf
        unique_shapes = set()
        crs = None
        # Initialize plot, and set title, axes labels, and grid:
        fig, ax = plt.subplots(figsize=(20, 10), tight_layout=True)
        ax.set_title(f"{data_type} tiles information", fontsize=20, fontweight='bold', pad=20)
        ax.set_xlabel('Longitude [°]', fontsize=16)
        ax.set_ylabel('Latitude [°]', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True, color='lightgray', linestyle='--', linewidth=0.5)
        # Loop through all the data tiles:
        for tile in data_tiles:
            # Open the data tile:
            with rasterio.open(tile) as src:
                # Show the data data with adjusted color range:
                show(src, ax=ax, cmap='terrain')
                # Update total size:
                total_size += os.path.getsize(tile)
                # Get the bounds of the data tile and update min/max longitudes and latitudes:
                bounds = src.bounds
                min_lon = round(min(min_lon, bounds.left), 2)
                max_lon = round(max(max_lon, bounds.right), 2)
                min_lat = round(min(min_lat, bounds.bottom), 2)
                max_lat = round(max(max_lat, bounds.top), 2)
                # Increment the resolution count:
                res_counter[src.res] += 1
                # Add shape to the set of unique shapes:
                unique_shapes.add(src.shape)
                # Get coordinate reference system (assuming all tiles have the same crs):
                crs = src.crs
                # Plot the bounds of the data tile as a rectangle:
                ax.add_patch(patches.Rectangle((bounds.left, bounds.bottom), 
                                                bounds.right - bounds.left, 
                                                bounds.top - bounds.bottom, 
                                                fill=False, edgecolor='black', 
                                                linewidth=2))
        # Adjust axes:
        ax.set_xlim([min_lon, max_lon])
        ax.set_ylim([min_lat, max_lat])
        # Find the resolution:
        resolution, _ = res_counter.most_common(1)[0]
        print(colored(' ✔ Done!', 'green'))
    # Store and print the data tiles information:
    print(colored(f"{data_type} information", 'light_grey', attrs=['underline']))
    info_lines = [
        f"{data_type} tiles count: " + colored(f"{len(data_tiles)+1}", 'yellow'),
        f"{data_type} tiles size: " + colored(f"{total_size / (1024 * 1024 * 1024):.2f} GB", 'yellow'),
        f"{data_type} tiles dtype: " + colored(f"{src.dtypes[0]}", 'yellow'),
        f"{data_type} tiles CRS: " + colored(f"{crs}", 'yellow'),
        f"{data_type} tiles shape: " + colored(f"{unique_shapes}", 'yellow'),
        f"{data_type} tiles total extent: " + colored(f"LON({format_lon(min_lon)}, {format_lon(max_lon)}); LAT({format_lat(min_lat)}, {format_lat(max_lat)})", 'yellow'), 
        f"{data_type} tiles resolution: " + colored(f"{resolution}", 'yellow')]
    print('\n'.join(info_lines))
    # Save the figure:
    with timer(f"Saving the {data_type}_tiles plot..."):
        fig.savefig(os.path.join(temp_path, f"{data_type}_tiles_plot.png"), 
                    bbox_inches='tight', format='png', dpi=300)
        print(colored(' ✔ Done!', 'green'))
    # Close the figure:
    plt.close(fig)


# Define a function to plot the output data:
def plot_data(data, metadata, temp_path,
              data_name=None, title=None, cbar_label=None, cmap=None, 
              wgs=False, log_scale=False, inverse=False, binary=False):
    
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
    wgs : bool
        Whether to use labels for WGS84 coordinate reference system. Default is False.
    log_scale : bool
        Whether to use a logarithmic scale for the colorbar. Default is False.
    inverse : bool
        Whether to use a specific plot for inverted data. Default is False.
    binary : bool
        Whether to use a specific plot for binary data. Default is False.

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
        if log_scale:
            im = ax.imshow(data, cmap=cmap, extent=extent, zorder=2,
                           norm=colors.LogNorm(1, data.max()), interpolation='bilinear')
        elif inverse:
            im = ax.imshow(data, cmap=cmap, extent=extent, zorder=2,
                           norm=colors.Normalize(vmin=np.percentile(data, 1), 
                                                 vmax=np.percentile(data, 99)), 
                                                 interpolation='bilinear')
        else:
            im = ax.imshow(data, cmap=cmap, extent=extent, zorder=1)
        # Set the title, labels, tick parameters, and grid:
        ax.set_title(title, fontsize=20, fontweight='bold', pad=20)
        if wgs:
            ax.set_xlabel('Longitude [°]', fontsize=16, labelpad=10)
            ax.set_ylabel('Latitude [°]', fontsize=16, labelpad=10)
        else:
            ax.set_xlabel('Longitude [m]', fontsize=16, labelpad=10)
            ax.set_ylabel('Latitude [m]', fontsize=16, labelpad=10)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True, color='lightgray', linestyle='--', linewidth=0.5)
        # Set the colorbar:
        if binary:
            colorbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03, 
                                    boundaries=[0, 1], values=[0.5])
            colorbar.ax.tick_params(labelsize=14)
            colorbar.ax.set_ylabel(cbar_label, fontsize=16, labelpad=10)
        else:
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
# Set the directories:
##################################################################################################

# Get the directories:
input_dir_ups = config.dir['input_upstream']
input_dir_dem = config.dir['input_dem']
output_dir = config.dir['output']
temp_dir = config.dir['temp']

##################################################################################################
# Download the data:
##################################################################################################

# # Download the DEM tiles (if not already downloaded):
# download_data(input_dir_dem, 'DEM')
# # Download the upstream data (if not already downloaded):
# download_data(input_dir_ups, 'UPSTREAM')

##################################################################################################
# Define the research area extent:
##################################################################################################

# Get DEM and UPSTREAM tiles:
dem_tiles, ups_tiles = get_data(input_dir_dem, input_dir_ups, temp_dir)

##################################################################################################
# Merge the data:
##################################################################################################

# Merge DEM and UPSTREAM tiles:
merged_dem, merged_ups, metadata = merge_data(dem_tiles, ups_tiles, temp_dir)

##################################################################################################
# Resample the data:
##################################################################################################

# Resample merged DEM and UPSTREAM data:
res_merged_dem, res_merged_ups, res_metadata = resample_data(merged_dem, merged_ups, metadata, 
                                                             temp_dir, output_dir)

##################################################################################################
# Calculate terrain parameters:
##################################################################################################

# Create cell factor area:
cell_area = cell_factor_area(res_merged_dem, res_metadata, temp_dir, output_dir)
# Invert merged upstream data:
inv_res_ups = invert_upstream(res_merged_ups, res_metadata, temp_dir)
# Compute flow direction and flow accumulation:
flow_dir, flow_acc = flow_accumulation(inv_res_ups, res_metadata, temp_dir, output_dir)
# # Call the function to infer flow directions from the flow : (NOT IN USE)
# inferred_fd_d8 = infer_fd_d8_from_accumulation(flow_acc, resaccumulation_metadata, temp_dir)
# Extract river network based on the speficied areal coverage threshold:
riv_net = river_network(flow_acc, cell_area, res_metadata, temp_dir)
# Compute constant head boundary conditions (CHB):
chb = boundary_conditions(riv_net, res_merged_dem, res_metadata, temp_dir, output_dir)
# Create research area terrain mask:
terr_mask = terrain_mask(merged_dem, metadata, temp_dir, output_dir)
# Compute river lengths:
riv_len = river_lengths(riv_net, res_metadata, temp_dir, output_dir)
 
##################################################################################################
# Remove temporary files:
##################################################################################################

# If the user wants to delete the temporary directory:
if config.intermediate_step['delete']:
    os.system(f"rm -r {temp_dir}")

##################################################################################################
######################################  End of the script  #######################################
##################################################################################################