# Import modules:
import numpy as np
from termcolor import colored
from components.timing import timer
from landlab import RasterModelGrid
from shapely.geometry import Polygon
from components.saving import save_data
from components.plotting import plot_data
from components.configuration import config
from rasterio.features import geometry_mask
from landlab.components import FlowAccumulator
from components.reprojecting import generic_reproject


# ------------------------------------------------------------------------------------------------
# Terrain parameters functions:
# ------------------------------------------------------------------------------------------------


# Define a function to create a cell factor area:
def cell_factor_area(res_merged_dem, res_metadata, output_path):

    """
    Create a cell factor area array based on ouput data resolution.

    Parameters:
    ----------
    res_merged_dem : numpy array
        The resampled merged DEM data.
    res_metadata : dict
        Dictionary containing the metadata of the resampled data.
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
        cell_factor_area = np.full_like(res_merged_dem, cell_area_value).astype(np.float32)
        print(colored(' ✔ Done!', 'green'))
    # Save the cell factor area array:
    save_data(cell_factor_area, res_metadata, output_path, 
              data_name='cell_factor_area')
    print(colored('==========================================================================================', 'blue'))

    # Return the cell factor area array:
    return cell_factor_area


# Define a function to invert the resampled merged upstream data:
def invert_upstream(res_merged_upstream, res_metadata, temp_path, output_path):

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
    output_path : str
        The path to the output directory.
    
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
    # Save the pseudo elevation:
    save_data(p_elev, res_metadata, output_path,
              data_name='pseudo_elevation')
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
    # Plot the flow accumulation:
    if intermediate_step:
        plot_data(flow_accumulation_2D, res_metadata, temp_path, 
                  data_name='flow_accumulation', title='Flow accumulation', 
                  cbar_label='Upstream cells', cmap='cubehelix', log_scale=True)
    print(colored('==========================================================================================', 'blue'))

    # Return the flow direction and accumulation maps:
    return flow_direction_2D, flow_accumulation_2D


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