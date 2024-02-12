# Import modules:
import os
import rasterio
from tqdm import tqdm
from termcolor import colored
from components.configuration import config
from components.plotting import plot_data_tiles


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