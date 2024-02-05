##################################################################################################
###################################### Author details ############################################
##################################################################################################

__date__        = "September 2023"
__author__      = "Nikola Ivanović"
__email__       = "nikola.ivanovic@kaust.edu.sa"
__department__  = "Earth Science and Engineering"
__institution__ = "King Abdullah University of Science and Technology, KSA"

##################################################################################################
####################################### Headline #################################################
##################################################################################################

"""
Aim:
---
This script prepares soil and subsurface data for the DRYP 1.0 model. The script downloads,
processes, and saves the data outputs.
Input:
-----
1. Soil data (downloaded from ISRIC website).

Operations:
----------
1. Download the data.
2. Process the downloaded data and save the outputs.

Outputs:
-------
1. Soil porosity (porosity)		                ->	    	      WCsat
2. Theta residual			                    -> 	           	  WCres
3. Available water content (AWC)  	            ->	              WCavail
4. Wilting point (wp)		                   	->	    	      CRIT-WILT
5. Soil depth (D)			                    -> 	   	          SOIL-DEPTH
6. Soil particle distribution parameter (b)     ->	    	      N
7. Soil suction head (psi)	                	->                Alfa & N
8. Saturated hydraulic conductivity (Ksat)      -> 	   	          Ksat
9. Sigma Ksat				                    ->	              std_Ksat
10. Initial soil water content                  ->	              None
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
import subprocess
from tqdm import tqdm
import rasterio
import numpy as np
from affine import Affine
from termcolor import colored
from rasterio.mask import mask
from shapely.geometry import box
from rasterio.warp import reproject, calculate_default_transform, Resampling
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
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
4. Cropping function
5. Reprojection and resampling function
6. Soil parameters functions
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
        print(colored('==========================================================================================', 'cyan'))

# ------------------------------------------------------------------------------------------------
# Data download functions:
# ------------------------------------------------------------------------------------------------

# Define a function to download the file:
def download_file(url, output_path):

    """
    This function downloads a file from a given url.
    
    Parameters:
    ----------
    url : str
        The url of the file to be downloaded.
    output_path : str
        The path to the output directory.
            
    Returns:
    -------
    download_success : bool
        The boolean value indicating whether the download was successful.
    """

    # Downloading the zip file:
    download_result = subprocess.run(f"wget -O {output_path} {url}", shell=True)
    # Check if the download was successful:
    if download_result.returncode != 0:
        print(f"Download failed! Error: {download_result.stderr}")
        return False
    return True


# Define a function to unzip the file:
def unzip_file(zip_path, output_path, folder_name):

    """
    This function unzips a file.

    Parameters:
    ----------
    zip_path : str
        The path to the zip file.
    output_path : str
        The path to the output directory.
    folder_name : str
        The name of the folder to be created.

    Returns:
    -------
    unzip_success : bool
        The boolean value indicating whether the unzip was successful.
    """

    # Adjust the output_path to include the folder_name:
    final_output_path = os.path.join(output_path, folder_name)
    # Create the folder if it doesn't exist:
    if not os.path.exists(final_output_path):
        os.makedirs(final_output_path)
    # Unzipping:
    unzip_result = subprocess.run(f"unzip -o {zip_path} -d {final_output_path} >/dev/null 2>&1", shell=True)
    # Check if the unzip was successful:
    if unzip_result.returncode != 0:
        print(f"Unzip failed! Error: {unzip_result.stderr}")
        return False
    return True


# Defining a function to download soil data:
def download_data(output_path, download_type='both'):
    
    """
    This function downloads the soil data.

    Parameters:
    ----------
    output_path : str
        The path to the output directory.
    download_type : str
        'soil', 'depth', or 'both'. Determines the type of data to download.

    Returns:
    -------
    None.
    """


    # Check if the output directory exists, if not, create it:
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    # Downloading and unzipping soil data:
    if download_type in ['soil', 'both']:
        # Define the url for soil data:
        urls = config.url['layers', 'top_subsoil']
        # Loop through the urls:
        for key, url in tqdm(urls.items()):
            # Define the zip folder:
            zip_folder = f"{key}.zip"
            # Define the zip folder path:
            zip_folder_path = os.path.join(output_path, zip_folder)
            # Check if the zip folder exists:
            if not os.path.exists(zip_folder_path):
                # Downloading the file:
                download_success = download_file(url, zip_folder_path)
                # Check if the download was successful:
                if not download_success:
                    print(f"Skipping unzipping for {zip_folder} due to failed download.")
                    continue
            # Unzipping the file:
            unzip_success = unzip_file(zip_folder_path, output_path, key)
            # Check if the unzip was successful:
            if unzip_success:
                print(f"Unzipping {zip_folder} successful!")
                # Remove the zip folder:
                os.remove(zip_folder_path)

    # Downloading and unzipping soil depth data:
    if download_type in ['depth', 'both']:
        # Define the url for soil depth:
        url = config.url['soil_depth']
        # Downloading soil depth data:
        download_soil_depth = subprocess.run(f"wget -O {output_path}/SOIL-DEPTH_M_250m.tif {url}", shell=True)
        # Check if the download was successful:
        if download_soil_depth.returncode != 0:
            print(f"Download failed! Error: {download_soil_depth.stderr}")

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

# ------------------------------------------------------------------------------------------------
# Soil parameters functions:
# ------------------------------------------------------------------------------------------------

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
        The data array to be adjusted.

    Returns:
    -------
    data : numpy array
        The adjusted data array.
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
        The data array to be filled.
    k : int
        The number of nearest neighbors to use for filling in the NaNs.

    Returns:
    -------
    filled_array : numpy array
        The filled data array.
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
def process_data(parameter, input_path, temp_path, output_path, 
                 data_name, title, cbar_label, cmap, save=True):

    """
    This function processes the data by cropping, resampling, and saving the data.

    Parameters:
    ----------
    parameter : str
        The parameter to process.
    input_path : str
        The path to the input directory.
    temp_path : str
        The path to the temporary directory.
    output_path : str
        The path to the output directory.
    data_name : str
        Name of the data to be processed.
    title : str
        The title of the data.
    cbar_label : str
        The colorbar label of the data.
    cmap : str
        The colormap of the data.
    save : bool
        Whether to save the data or not. Default is True.

    Returns:
    -------
    res_crop_data : numpy array
        The cropped and resampled data.
    res_crop_metadata : dict
        The metadata of the resampled cropped data.
    """

    # Get the intermediate step for plotting:
    intermediate_step = config.intermediate_step['plot']
    
    # Construct the file path:
    if parameter == 'SOIL-DEPTH':
        data_path = os.path.join(input_path, f"{parameter}_M_250m.tif")
    else:
        data_path = os.path.join(input_path, 'top_subsoil', parameter, 
                                 f"{parameter}_M_250m_TOPSOIL.tif")
    # If the parameter is 'ALFA', or 'N', don't save the data output:
    if parameter == 'ALFA' or parameter == 'N':
        save = False
    # Crop the dataset:
    crop_dataset, crop_metadata = crop_data(data_path, parameter)
    # Reproject and resample the data:
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
    if parameter == 'ALFA':
        res_crop_data = res_crop_data / 10                  # [cm^-1] to [mm^-1]
    elif parameter == 'Ksat':
        res_crop_data = res_crop_data * 10 / 24             # [cm d^-1] to [mm h^-1]
    elif parameter == 'SOIL-DEPTH':
        res_crop_data = res_crop_data * 1000                # [m] to [mm]
    # Save the resampled data:
    if save:
        save_data(res_crop_data, res_crop_metadata, output_path, 
                  data_name=f"resampled_{data_name}")
    # Plot the resampled data:
    if intermediate_step:
        plot_data(res_crop_data, res_crop_metadata, temp_path,
                  data_name=f"resampled_{data_name}", title=f"Resampled {title}",
                  cbar_label=cbar_label, cmap=cmap)
    # Print a message:
    print(f"{colored(parameter, 'blue', attrs=['bold', 'underline'])}", colored('is processed!', 'green')) 
    print(colored('==========================================================================================', 'blue'))

    # Return the dictionary parameters:
    return {'parameter': parameter, 'data': res_crop_data, 'metadata': res_crop_metadata}


# Define a function to calculate the soil particle distribution parameter (b):
def calculate_b(n, metadata, temp_path, output_path):
        
    """
    This function calculates the soil particle distribution parameter (b).

    Parameters:
    ----------
    n : numpy array
        The soil pore size distribution parameter (n).
    metadata : dict
        The dictionary of metadata.
    temp_path : str
        The path to the temporary directory.
    output_path : str
        The path to the output directory.

    Returns:
    -------
    b : numpy array
        The soil particle distribution parameter (b).
    """

    # Calculate parameter b:
    with timer('Calculating b parameter...'):
        # Calculate:
        b = 1 / (n - 1)
        print(colored(' ✔ Done!', 'green'))
    # Save the data:
    save_data(b, metadata, output_path, 
                data_name='parameter_b')                  
    # Plot the data:
    plot_data(b, metadata, temp_path,
                data_name='parameter_b', title='Parameter (b)',
                cbar_label='Soil Particle Distribution []', cmap='viridis')
    print(colored('==========================================================================================', 'blue'))
    
    # Return soil particle distribution parameter (b):
    return b


# Define a function to calculate the soil suction head parameter (psi):
def calculate_psi(alfa, n, theta_s, theta, metadata, temp_path, output_path):
            
    """
    This function calculates the soil suction head parameter (psi).

    Parameters:
    ----------
    alfa : numpy array
        The soil pore size distribution parameter (alfa).
    n : numpy array
        The soil pore size distribution parameter (n).
    theta_s : numpy array
        The saturated water content (WCsat).
    theta : numpy array
        The current water content (WCavail).
    metadata : dict
        The dictionary of metadata.
    temp_path : str
        The path to the temporary directory.
    output_path : str
        The path to the output directory.

    Returns:
    -------
    psi : numpy array
        The soil suction head parameter (psi).
    """

    # Calculate parameter psi:
    with timer('Calculating psi parameter...'):
        # Calculate:
        psi = (1 / alfa) * (((theta_s / theta) ** ((n - 1) / n)) - 1) ** (1 / n)
        print(colored(' ✔ Done!', 'green'))
    # Save the data:
    save_data(psi, metadata, output_path,
              data_name='parameter_psi')
    # Plot the data:
    plot_data(psi, metadata, temp_path,
              data_name='parameter_psi', title='Parameter (psi)', 
              cbar_label='Soil Suction Head [mm]', cmap='viridis')
    print(colored('==========================================================================================', 'blue'))
    
    # Return soil suction head parameter (psi):
    return psi


# Define a function to calculate sigma Ksat:
def calculate_sigma_ksat(ksat, metadata, temp_path, output_path):
    
    """
    This function creates an ones array as the sigma Ksat.

    Parameters:
    ----------
    ksat : numpy array
        The saturated hydraulic conductivity (Ksat).
    metadata : dict
        The dictionary of metadata.
    temp_path : str
        The path to the temporary directory.
    output_path : str
        The path to the output directory.

    Returns:
    -------
    sigma_ksat : numpy array
        The standard deviation of Ksat.
    """

    # Calculate sigma Ksat:
    with timer('Calculating sigma Ksat...'):
        # Calculate:
        sigma_ksat_value = np.nanstd(ksat)
        # Create an array of the same shape as Ksat, and fill it with the std. dev. of Ksat:
        sigma_ksat = (np.ones(ksat.shape) * sigma_ksat_value).astype(np.float32)
        print(colored(' ✔ Done!', 'green'))
    # Save the data:
    save_data(sigma_ksat, metadata, output_path, 
                data_name='sigma_ksat')
    print(colored('==========================================================================================', 'blue'))

    # Return the standard deviation of Ksat:
    return sigma_ksat


# Define a function to summarize the results:
def summarize_results(results):

    """
    This function prints out the NaN count, maximum value, mean value, minimum value, and 
    Inf count for each parameter in the results.

    Parameters:
    ----------
    results : dict
        The dictionary of soil data processing results.
    
    Returns:
    -------
    None.
    """

    # Print the header of the table:
    print(f"{'Parameter':<15}{'NaN':<10}{'Max':<10}{'Mean':<10}{'Min':<10}{'Inf':<10}")
    # Loop through the parameters in results:
    for key in results.keys():
        if 'data' in results[key]:
            # Check if the data is a numeric numpy array:
            if isinstance(results[key]['data'], np.ndarray) and np.issubdtype(results[key]['data'].dtype, np.number):
                # Compute the NaN count, maximum value, mean value, minimum value, and Inf count:
                nan_count = f"{np.count_nonzero(np.isnan(results[key]['data']))}"
                max_value = f"{np.nanmax(results[key]['data']):.3f}"
                mean_value = f"{np.nanmean(results[key]['data']):.3f}"
                min_value = f"{np.nanmin(results[key]['data']):.3f}"
                inf_count = f"{np.count_nonzero(np.isinf(results[key]['data']))}"
                # Print the results in a row following column positions:
                print(f"{key:<15}{nan_count:<10}{max_value:<10}{mean_value:<10}{min_value:<10}{inf_count:<10}")
                # Save the computed values back to results
                results[key].update({
                    'nan_count': nan_count, 
                    'max_value': max_value,
                    'mean_value': mean_value,
                    'min_value': min_value,
                    'inf_count': inf_count})
            else:
                print(f"{key}:", colored(' data is not a numeric NumPy array!', 'red'))


# Define a function to create a dictionary of soil data processing results:
def results(input_path, temp_path, output_path):

    """
    This function creates a dictionary of soil data processing results.

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
    results : dict
        The dictionary of soil data processing results.
    """

    # Define the data parameters:
    parameters = ['ALFA', 'CRIT-WILT', 'Ksat', 'N', 'WCavail', 'WCres', 'WCsat', 'SOIL-DEPTH']
    # Define the data names:
    data_names = ['alfa', 'crit_wilt', 'ksat', 'n', 'wc_avail', 'wc_res', 'wc_sat', 'soil_depth']
    # Define the data titles:
    titles = ['Alfa Parameter',   
              'Wilting Point', 
              'Saturated Hydraulic Conductivity', 
              'N Parameter', 
              'Available Water Content', 
              'Residual Water Content', 
              'Saturater Water Content',
              'Soil Depth']
    # Define the data colorbar labels:
    cbar_labels = [r'Alfa Parameter of the van Genuchten model [mm$^{-1}$]', 
                   r'Wilting point [m$^3$ m$^{-3}$]', 
                   r'Saturated hydraulic conductivity [mm h$^{-1}$]', 
                   r'N Parameter of the van Genuchten model []', 
                   r'Available water content [m$^3$ m$^{-3}$]', 
                   r'Residual Water Content [m$^3$ m$^{-3}$]', 
                   r'Saturater Water Content [m$^3$ m$^{-3}$]',
                   r'Soil Depth [mm]']
    # Define the data colormaps:
    cmaps = ['viridis', 'viridis', 'viridis', 'viridis', 'viridis', 'viridis', 'viridis', 'viridis']
    # Create a dictionary to store the results:
    results = {}
    # Loop through the parameters:
    for parameter, data_name, title, cbar_label, cmap in zip(parameters, data_names, titles, cbar_labels, cmaps):
        # Process the data:
        result = process_data(parameter, input_path, temp_path, output_path, 
                              data_name, title, cbar_label, cmap)
        # Save the parameter and its corresponding data in the results dictionary:
        results[result['parameter']] = {'data': result['data'], 
                                        'metadata': result['metadata']}
    # Calculate parameter b:
    b = calculate_b(results['N']['data'], results['N']['metadata'], temp_path, output_path)
    # Calculate parameter psi:
    psi = calculate_psi(results['ALFA']['data'], results['N']['data'], results['WCsat']['data'], 
                        results['WCavail']['data'], results['N']['metadata'], temp_path, output_path)
    # Calculate the standard deviation of Ksat:
    sigma_Ksat = calculate_sigma_ksat(results['Ksat']['data'], results['Ksat']['metadata'], temp_path, output_path)
    # Save the b parameter and its corresponding data in the results dictionary:
    results['b'] = {'data': b, 'metadata': results['N']['metadata']}
    # Save the psi parameter and its corresponding data in the results dictionary:
    results['psi'] = {'data': psi, 'metadata': results['N']['metadata']}
    # Save the sigma_Ksat parameter and its corresponding data in the results dictionary:
    results['sigma_Ksat'] = {'data': sigma_Ksat, 'metadata': results['Ksat']['metadata']}
    # Summarize the results:
    summarize_results(results)

    # Return the results dictionary:
    return results

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
input_dir = config.dir['input']
output_dir = config.dir['output']
temp_dir = config.dir['temp']

##################################################################################################
# Download the data:
##################################################################################################

# Download and unzip the data (if not already downloaded):
# download_data(input_dir)

##################################################################################################
# Process the data and save the outputs:
##################################################################################################

# Process the data:
output = results(input_dir, temp_dir, output_dir)

##################################################################################################
# Remove temporary files:
##################################################################################################

# If the user wants to delete the temporary directory:
if config.intermediate_step['delete']:
    os.rmdir(temp_dir)

##################################################################################################
######################################  End of the script  #######################################
##################################################################################################